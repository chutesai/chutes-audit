import io
import os
import re
import csv
import ast
import uuid
import shutil
import random
import aiohttp
import yaml
import tqdm
import orjson as json
import asyncio
import tempfile
import hashlib
import backoff
import traceback
from pathlib import Path
import numpy as np
import pybase64 as base64
import sounddevice as sd
import soundfile as sf
from typing import Optional
from fiber import Keypair
from fiber.chain import weights
from fiber.chain import fetch_nodes
from fiber.networking.models import NodeWithFernet as Node
from fiber.chain.chain_utils import query_substrate
from functools import lru_cache
from datetime import datetime, timedelta
from langdetect import detect as detect_language
from term_image.image import from_file as image_from_file
from loguru import logger
from typing import AsyncGenerator
from pydantic import BaseModel
from substrateinterface import SubstrateInterface
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Double,
    Integer,
    Boolean,
    BigInteger,
    func,
    select,
    ForeignKey,
    text,
    Index,
)
from munch import munchify
from datasets import load_dataset
from contextlib import asynccontextmanager, contextmanager

# Database configuration.
engine = create_async_engine(
    os.getenv("POSTGRESQL", "postgresql+asyncpg://user:password@127.0.0.1:5432/chutes_audit"),
    echo=False,
    pool_pre_ping=True,
    pool_reset_on_return="rollback",
)
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
Base = declarative_base()

# Query and score weighting values to use for calculating incentive/setting weights.
VERSION_KEY = 69420
FEATURE_WEIGHTS = {
    "compute_units": 0.45,  # Total amount of compute time (compute muliplier * total time).
    "invocation_count": 0.25,  # Total number of invocations.
    "unique_chute_count": 0.20,  # Number of unique chutes over the scoring period.
    "bounty_count": 0.1,  # Number of bounties received (not bounty values, just counts).
}
MINER_METRICS_QUERY = """
WITH computation_rates AS (
    SELECT
        chute_id,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / (metrics->>'steps')::float) as median_step_time,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / ((metrics->>'it')::float + (metrics->>'ot')::float)) as median_token_time
    FROM invocations
    WHERE ((metrics->>'steps' IS NOT NULL and (metrics->>'steps')::float > 0) OR (metrics->>'it' IS NOT NULL AND metrics->>'ot' IS NOT NULL AND (metrics->>'ot')::float > 0 AND (metrics->>'it')::float > 0))
      AND started_at >= NOW() - INTERVAL '2 days'
    GROUP BY chute_id
)
SELECT
    i.miner_hotkey,
    COUNT(*) as invocation_count,
    COUNT(CASE WHEN i.bounty > 0 THEN 1 END) AS bounty_count,
    sum(
        i.bounty +
        i.compute_multiplier *
        CASE
            WHEN i.metrics->>'steps' IS NOT NULL
                AND r.median_step_time IS NOT NULL
            THEN (i.metrics->>'steps')::float * r.median_step_time
            WHEN i.metrics->>'it' IS NOT NULL
                AND i.metrics->>'ot' IS NOT NULL
                AND r.median_token_time IS NOT NULL
            THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * r.median_token_time
            ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
        END
    ) AS compute_units
FROM invocations i
LEFT JOIN computation_rates r ON i.chute_id = r.chute_id
WHERE i.started_at > (now() AT TIME ZONE 'UTC') - INTERVAL '7 days'
    AND (i.error_message IS NULL or i.error_message = '')
    AND i.miner_uid >= 0
    AND i.completed_at IS NOT NULL
    AND NOT EXISTS (
        SELECT 1
        FROM reports
        WHERE invocation_id = i.parent_invocation_id
        AND confirmed_at IS NOT NULL
    )
GROUP BY i.miner_hotkey
ORDER BY compute_units DESC;
"""
UNIQUE_CHUTE_AVERAGE_QUERY = """
WITH time_series AS (
  SELECT
    generate_series(
      date_trunc('hour', now() - INTERVAL '7 days'),
      date_trunc('hour', now()),
      INTERVAL '1 hour'
    ) AS time_point
),
-- Get all instances that had at least one successful invocation (ever) while the instance was alive.
instances_with_success AS (
  SELECT DISTINCT
    instance_id
  FROM invocations ii
  WHERE
    error_message IS NULL
    AND completed_at IS NOT NULL
    AND miner_uid >= 0
    AND NOT EXISTS (
        SELECT 1
        FROM reports
        WHERE invocation_id = ii.parent_invocation_id
        AND confirmed_at IS NOT NULL
    )
),
-- For each time point, find active instances that have had successful invocations
active_instances_per_timepoint AS (
  SELECT
    ts.time_point,
    ia.instance_id,
    ia.chute_id,
    ia.miner_hotkey
  FROM time_series ts
  JOIN (
    SELECT
      instance_id,
      chute_id,
      miner_hotkey,
      MIN(verified_at) AS first_verified_at,
      CASE
        WHEN COUNT(CASE WHEN deleted_at IS NOT NULL THEN 1 END) > 0 THEN
          MIN(deleted_at)
        ELSE NULL
      END AS earliest_deleted_at
    FROM instance_audits
    GROUP BY instance_id, chute_id, miner_hotkey
  ) ia ON
    ia.first_verified_at <= ts.time_point AND
    (ia.earliest_deleted_at IS NULL OR ia.earliest_deleted_at >= ts.time_point)
  JOIN instances_with_success iws ON
    ia.instance_id = iws.instance_id
),
-- Count distinct chute_ids per miner per time point
active_chutes_per_timepoint AS (
  SELECT
    time_point,
    miner_hotkey,
    COUNT(DISTINCT chute_id) AS active_chutes
  FROM active_instances_per_timepoint
  GROUP BY time_point, miner_hotkey
)
-- Calculate average active chutes per miner across all time points
SELECT
  miner_hotkey,
  AVG(active_chutes)::integer AS avg_active_chutes
FROM active_chutes_per_timepoint
GROUP BY miner_hotkey
ORDER BY avg_active_chutes DESC;
"""
MISSING_INVOCATIONS_QUERY = """
SELECT s.*
  FROM synthetics s
  LEFT JOIN invocations i ON s.invocation_id = i.invocation_id
 WHERE (i.invocation_id IS NULL OR s.miner_hotkey != i.miner_hotkey)
   AND created_at < (SELECT MAX(start_time) FROM audit_entries WHERE hotkey = '{hotkey}')
"""
MINER_SUMMARY_METRICS_QUERY = """
SELECT
   COALESCE(i.miner_hotkey, m.hotkey) as hotkey,
   i.invocation_count,
   m.metrics_count
FROM
   (SELECT miner_hotkey, COUNT(*) AS invocation_count
    FROM invocations
    WHERE error_message is null
    AND miner_hotkey is not null
    AND completed_at is not null
    AND miner_uid >= 0
    GROUP BY miner_hotkey) i
FULL OUTER JOIN
   (SELECT hotkey, SUM(total_count) as metrics_count
    FROM miner_metrics
    GROUP BY hotkey) m
ON i.miner_hotkey = m.hotkey
ORDER BY COALESCE(i.invocation_count, 0) DESC
"""
MINER_COVERAGE_QUERY = "SELECT SUM(EXTRACT(EPOCH FROM end_time - start_time)::integer) AS coverage_seconds FROM audit_entries WHERE hotkey = '{hotkey}' AND start_time >= (now() AT TIME ZONE 'UTC') - interval '169 hours'"
EXPECTED_COVERAGE = 7 * 24 * 60 * 60 - (60 * 60)


class IntegrityViolation(RuntimeError): ...


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class Invocation(Base):
    __tablename__ = "invocations"
    parent_invocation_id = Column(String)
    invocation_id = Column(String, primary_key=True)
    chute_id = Column(String)
    chute_user_id = Column(String)
    function_name = Column(String)
    user_id = Column(String)
    image_id = Column(String)
    image_user_id = Column(String)
    instance_id = Column(String)
    miner_uid = Column(Integer)
    miner_hotkey = Column(String)
    error_message = Column(String)
    compute_multiplier = Column(Double)
    bounty = Column(Integer)
    metrics = Column(JSONB, nullable=True)
    started_at = Column(DateTime(timezone=False))
    completed_at = Column(DateTime(timezone=False), nullable=True)


class Report(Base):
    __tablename__ = "reports"
    invocation_id = Column(String, nullable=False, primary_key=True)
    user_id = Column(String, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    confirmed_at = Column(DateTime(timezone=True))
    confirmed_by = Column(String)
    reason = Column(String, nullable=False)

    __table_args__ = (Index("idx_report_inv_cnfrm", "invocation_id", "confirmed_at"),)


class InstanceAudit(Base):
    __tablename__ = "instance_audits"
    audit_id = Column(String, primary_key=True)
    entry_id = Column(String, ForeignKey("audit_entries.entry_id", ondelete="CASCADE"))
    instance_id = Column(String)
    source = Column(String)
    deployment_id = Column(String)
    validator = Column(String)
    chute_id = Column(String)
    version = Column(String)
    deletion_reason = Column(String)
    miner_uid = Column(Integer)
    miner_hotkey = Column(String)
    region = Column(String)
    created_at = Column(DateTime(timezone=False))
    verified_at = Column(DateTime(timezone=False))
    deleted_at = Column(DateTime(timezone=False))


class MinerMetric(Base):
    __tablename__ = "miner_metrics"
    entry_id = Column(
        String, ForeignKey("audit_entries.entry_id", ondelete="CASCADE"), primary_key=True
    )
    deployment_id = Column(String, primary_key=True)
    function = Column(String, primary_key=True)
    hotkey = Column(String)
    chute_id = Column(String)
    total_seconds = Column(Double)
    total_count = Column(Integer)


class AuditEntry(Base):
    __tablename__ = "audit_entries"
    entry_id = Column(String, primary_key=True)
    hotkey = Column(String)
    block = Column(BigInteger)
    path = Column(String)
    created_at = Column(DateTime(timezone=False))
    start_time = Column(DateTime(timezone=False))
    end_time = Column(DateTime(timezone=False))


class Synthetic(Base):
    __tablename__ = "synthetics"
    parent_invocation_id = Column(String, primary_key=True)
    invocation_id = Column(String)
    instance_id = Column(String)
    chute_id = Column(String)
    miner_uid = Column(String)
    miner_hotkey = Column(String)
    created_at = Column(DateTime(timezone=False))
    has_error = Column(Boolean, default=False)


class Target(BaseModel):
    instance_id: str
    invocation_id: str
    child_id: str
    uid: str
    hotkey: str
    error: str = None


class Auditor:
    def __init__(self, config_path: str = None):
        """
        Load config.
        """
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config", "config.yml"
            )
        logger.debug(f"Loading {config_path=}")
        with open(config_path, "r") as infile:
            self.config = munchify(yaml.safe_load(infile))
        if self.config.synthetics.enabled:
            text_config = self.config.synthetics.text
            if text_config.enabled:
                logger.debug(f"Loading text prompt dataset: {text_config.dataset.name}")
                self.text_prompts = load_dataset(
                    text_config.dataset.name, **dict(text_config.dataset.options)
                )
            image_config = self.config.synthetics.image
            if image_config.enabled:
                logger.debug(f"Loading image prompt dataset: {image_config.dataset.name}")
                self.image_prompts = load_dataset(
                    image_config.dataset.name, **dict(image_config.dataset.options)
                )
        self.validators = {v.hotkey: v for v in self.config.validators}
        self._slock = asyncio.Lock()
        self._asession = None
        self._running = True
        self.chutes = {}
        self._substrate = SubstrateInterface(url=self.config.subtensor, ss58_format=42)

        # Keypair -- only set if you are a registered validator.
        self.ss58_address = None
        self.keypair = None
        if self.config.set_weights.enabled:
            self.ss58_address = self.config.set_weights.ss58_address
            self.keypair = Keypair.create_from_seed(self.config.set_weights.secret_seed)

    @contextmanager
    def substrate(self):
        """
        Yield the substrate interface, reconnecting on error.
        """
        try:
            yield self._substrate
        except Exception:
            self._substrate = SubstrateInterface(url=self.config.subtensor, ss58_format=42)
            raise

    @asynccontextmanager
    async def aiosession(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.
        """
        async with self._slock:
            if self._asession is None or self._asession.closed:
                self._asession = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=100, ttl_dns_cache=120, force_close=False),
                    read_bufsize=64 * 1024 * 1024,
                    raise_for_status=False,
                    trust_env=True,
                )
            yield self._asession

    def get_random_image_payload(self, model: str):
        """
        Get a random request payload for diffusion chutes.
        """
        prompt = self.image_prompts[random.randint(0, len(self.image_prompts))][
            self.config.synthetics.image.dataset.field_name
        ]
        prompt = prompt.lstrip('"').rstrip('"').replace('\\"', '"')
        return {
            "prompt": prompt,
            "seed": random.randint(0, 1000000000),
            "num_inference_steps": random.randint(5, 30),
        }

    def get_random_text_payload(self, model: str, endpoint: str = "chat"):
        """
        Get a random prompt for vllm chutes.
        """
        messages = self.text_prompts[random.randint(0, len(self.text_prompts))][
            self.config.synthetics.text.dataset.field_name
        ]
        messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": random.random() + 0.1,
            "seed": random.randint(0, 1000000000),
            "max_tokens": random.randint(10, 200),
            "stream": True,
            "logprobs": True,
        }
        if endpoint != "chat":
            payload["prompt"] = payload.pop("messages")[0]["content"]
        return payload

    async def load_chutes(self):
        """
        Load chutes from the API.
        """
        logger.debug("Loading chutes from API...")
        async with self.aiosession() as session:
            async with session.get(
                "https://api.chutes.ai/chutes/?include_public=true&limit=1000"
            ) as resp:
                data = await resp.json()
                chutes = {}
                for item in data["items"]:
                    item["cords"] = data.get("cord_refs", {}).get(item["cord_ref_id"], [])
                    chutes[item["chute_id"]] = munchify(item)
                self.chutes = chutes

    def _get_vllm_chute(self):
        """
        Randomly select a hot vllm chute.
        """
        vllm_chutes = [
            chute
            for chute in self.chutes.values()
            if chute.standard_template == "vllm"
            and any([instance.active and instance.verified for instance in chute.instances])
        ]
        if not vllm_chutes:
            logger.warning("No vllm chutes hot - this is very bad and should not really happen...")
            return None
        return random.choice(vllm_chutes)

    def _get_diffusion_chute(self):
        """
        Randomly select a hot diffusion chute.
        """
        diffusion_chutes = [
            chute
            for chute in self.chutes.values()
            if chute.standard_template == "diffusion"
            and any([instance.active and instance.verified for instance in chute.instances])
        ]
        if not diffusion_chutes:
            logger.warning(
                "No diffusion chutes hot - this is very bad and should not really happen..."
            )
            return None
        return random.choice(diffusion_chutes)

    def _get_tts_chute(self):
        """
        Randomly select a hot TTS chute.
        """
        tts_chutes = [
            chute
            for chute in self.chutes.values()
            if any([cord.path == "/speak" and not cord.stream for cord in chute.cords])
            and chute.user.username == "chutes"
        ]
        if not tts_chutes:
            logger.warning("No TTS chutes hot.")
            return None
        return random.choice(tts_chutes)

    def _get_tei_chute(self, endpoint: str = "/embed"):
        """
        Get text-embeding-inference chutes.
        """
        tei_chutes = [
            chute
            for chute in self.chutes.values()
            if chute.standard_template == "tei"
            and any([cord.path == endpoint for cord in chute.cords])
            and chute.user.username == "chutes"
        ]
        if not tei_chutes:
            logger.warning("No TEI chutes hot.")
            return None
        return random.choice(tei_chutes)

    def _render(self, chute, data):
        if chute.standard_template == "diffusion" and self.config.synthetics.image.render:
            try:
                if (
                    chute.standard_template == "diffusion"
                    and isinstance(data["result"], dict)
                    and data["result"].get("bytes")
                ):
                    with tempfile.NamedTemporaryFile(mode="wb") as outfile:
                        outfile.write(base64.b64decode(data["result"]["bytes"].encode()))
                        outfile.flush()
                        image = image_from_file(outfile.name)
                        image.draw()
            except Exception as exc:
                logger.warning(f"Could not render image: {exc}")
        elif chute.standard_template == "vllm" and self.config.synthetics.text.render:
            try:
                chunk_data = json.loads(data["result"][6:])
                if chunk_data["choices"][0].get("delta"):
                    print(chunk_data["choices"][0]["delta"]["content"], end="", flush=True)
                else:
                    print(chunk_data["choices"][0]["text"], end="", flush=True)
            except Exception:
                ...
        elif chute.standard_template == "tts" and self.config.synthetics.tts.render:
            try:
                if isinstance(data["result"], dict) and data["result"].get("bytes"):
                    chunk_data = base64.b64decode(data["result"]["bytes"].encode())
                    audio_io = io.BytesIO(chunk_data)
                    audio_chunk, sr = sf.read(audio_io)
                    if len(audio_chunk.shape) > 1:
                        audio_chunk = np.mean(audio_chunk, axis=1)
                    audio_chunk = audio_chunk.astype(np.float32)
                    logger.info("Playing audio, turn up your volume...")
                    sd.play(audio_chunk, 24000)
                    sd.wait()
            except Exception as exc:
                logger.warning(f"Error playing audio: {exc}")
        elif chute.standard_template == "tei" and self.config.synthetics.embed.render:
            try:
                if data["result"].get("json"):
                    logger.info(
                        f"Generated a matrix with shape: {np.array(data['result']['json']).shape}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to render embeddings: {exc}")

    async def _perform_request(self, chute, payload, url) -> list[Synthetic]:
        """
        Perform invocation request.
        """
        try:
            synthetics = []
            async with self.aiosession() as session:
                logger.info(f"Invoking {chute.name=} at {url}")
                async with session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.config.synthetics.api_key}",
                        "X-Chutes-Trace": "true",
                    },
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"Error sending synthetic to {chute.chute_id} [{chute.name}]: {resp.status=} {await resp.text()}"
                        )
                        return []
                    parent_id = resp.headers["X-Chutes-InvocationID"]
                    async for chunk_bytes in resp.content:
                        if not chunk_bytes or not chunk_bytes.startswith(b"data: "):
                            continue
                        data = json.loads(chunk_bytes[6:])
                        target = self._extract_target(data)
                        if (target := self._extract_target(data)) is not None:
                            self._debug_target(data)
                            synthetics.append(
                                Synthetic(
                                    instance_id=target.instance_id,
                                    parent_invocation_id=parent_id,
                                    invocation_id=target.child_id,
                                    chute_id=chute.chute_id,
                                    miner_uid=target.uid,
                                    miner_hotkey=target.hotkey,
                                    created_at=func.timezone("UTC", func.now()),
                                    has_error=False,
                                )
                            )
                        elif (target := self._extract_target_error(data)) is not None:
                            logger.warning(target.error)
                            # Can't really not be the case that we're not talking about the existing attempt.
                            assert target.instance_id == synthetics[-1].instance_id
                            assert target.invocation_id == synthetics[-1].invocation_id
                            synthetics[-1].has_error = True
                        elif data.get("error"):
                            logger.error(data["error"])
                        elif data.get("result"):
                            self._render(chute, data)
            return synthetics
        except Exception as exc:
            logger.warning(f"Error performing synthetic request: {exc}")
        return []

    @staticmethod
    def _debug_target(chunk) -> None:
        """
        Show debug logging for a chute invocation target.
        """
        message = "".join(
            [
                chunk["trace"]["timestamp"],
                " ["
                + " ".join(
                    [
                        f"{key}={value}"
                        for key, value in chunk["trace"].items()
                        if key not in ("timestamp", "message")
                    ]
                ),
                f"]: {chunk['trace']['message']}",
            ]
        )
        logger.info(message)

    @staticmethod
    def _extract_target(chunk) -> Target:
        """
        Extract miner info from trace messages.
        """
        if not chunk.get("trace"):
            return None
        message = chunk["trace"].get("message")
        re_match = re.search(r"query target=([^ ]+) uid=([0-9+]+) hotkey=([^ ]+)", message)
        if re_match:
            return Target(
                invocation_id=chunk["trace"].get("invocation_id"),
                child_id=chunk["trace"].get("child_id"),
                instance_id=re_match.group(1),
                uid=re_match.group(2),
                hotkey=re_match.group(3),
            )
        return None

    @staticmethod
    def _extract_target_error(chunk) -> Target:
        """
        Extract target errors from trace messages.
        """
        if not chunk.get("trace"):
            return None
        message = chunk["trace"].get("message")
        re_match = re.search(
            r"error encountered while querying target=([^ ]+) uid=([0-9]+) hotkey=([^ ]+) coldkey=[^ ]+: (.*)",
            message,
        )
        if re_match:
            return Target(
                invocation_id=chunk["trace"].get("invocation_id"),
                child_id=chunk["trace"].get("child_id"),
                instance_id=re_match.group(1),
                uid=re_match.group(2),
                hotkey=re_match.group(3),
                error=re_match.group(4),
            )

    async def _perform_chat(self) -> list[Synthetic]:
        """
        Perform a single chat request, with trace SSEs to see raw events.
        """
        if (chute := self._get_vllm_chute()) is None:
            return None
        payload = self.get_random_text_payload(model=chute.name, endpoint="chat")
        synthetics = await self._perform_request(
            chute, payload, "https://llm.chutes.ai/v1/chat/completions"
        )
        print("", flush=True)
        logger.info(f"Chat invocation generated {len(synthetics)} invocation objects.")
        return synthetics

    async def _perform_completion(self) -> list[Synthetic]:
        """
        Perform a single LLM completion request, with trace SSEs to see raw events.
        """
        if (chute := self._get_vllm_chute()) is None:
            return []
        payload = self.get_random_text_payload(model=chute.name, endpoint="completion")
        synthetics = await self._perform_request(
            chute, payload, "https://llm.chutes.ai/v1/completions"
        )
        print("", flush=True)
        logger.info(f"Chat invocation generated {len(synthetics)} invocation objects.")
        return synthetics

    async def _perform_image(self) -> list[Synthetic]:
        """
        Perform a single image generation request.
        """
        if (chute := self._get_diffusion_chute()) is None:
            return []
        payload = self.get_random_image_payload(model=chute.name)
        synthetics = await self._perform_request(
            chute, payload, f"https://{chute.slug}.chutes.ai/generate"
        )
        logger.info(f"Image generation request generated {len(synthetics)} invocation objects.")
        return synthetics

    async def _perform_tts(self) -> list[Synthetic]:
        """
        Perform a single text-to-speech request.
        """
        if (chute := self._get_tts_chute()) is None:
            return []
        while text := self.get_random_image_payload(model=chute.name)["prompt"][:1000]:
            try:
                language = detect_language(text)
                if language == "en":
                    break
            except Exception:
                ...
        chute.standard_template = "tts"
        payload = {"text": text}
        if chute.name == "Kokoro-82M":
            payload["voice"] = random.choice(
                [
                    "af",
                    "af_bella",
                    "af_sarah",
                    "am_adam",
                    "am_michael",
                    "bf_emma",
                    "bf_isabella",
                    "bm_george",
                    "bm_lewis",
                    "af_nicole",
                    "af_sky",
                ]
            )

        synthetics = await self._perform_request(
            chute, payload, f"https://{chute.slug}.chutes.ai/speak"
        )
        logger.info(f"TTS generation request generated {len(synthetics)} invocation objects.")
        return synthetics

    async def _perform_embedding(self) -> list[Synthetic]:
        """
        Perform a single text embedding request.
        """
        if (chute := self._get_tei_chute("/embed")) is None:
            return []
        text = self.get_random_image_payload(model=chute.name)["prompt"][:500]
        payload = {"inputs": [text]}
        synthetics = await self._perform_request(
            chute, payload, f"https://{chute.slug}.chutes.ai/embed"
        )
        logger.info(f"Text embedding request generated {len(synthetics)} invocation objects.")
        return synthetics

    async def perform_synthetic(self):
        """
        Send a single, random synthetic request.
        """
        await self.load_chutes()

        # Randomly select a task to perform.
        task_type = random.choice(
            [
                "chat",
                "completion",
                "image",
                "tts",
                "embedding",
            ]
        )
        logger.info(f"Attempting to perform synthetic task: {task_type=}")
        synthetics = await getattr(self, f"_perform_{task_type}")()
        if not synthetics:
            return
        async with get_session() as session:
            for synthetic in synthetics:
                session.add(synthetic)
            await session.commit()
        logger.success(f"Tracked {len(synthetics)} new synthetic records from {task_type} request")

    @lru_cache(maxsize=1024)
    def get_block_hash(self, block):
        """
        Get a block (number) hash.
        """
        with self.substrate() as substrate:
            return substrate.get_block_hash(block)

    def get_block_commit(self, block, who):
        """
        Given a block number, fetch all set_commitment events.
        """
        logger.info(f"Attempting to process {block=}")
        with self.substrate() as substrate:
            block_hash = self.get_block_hash(block)
            commitment = substrate.query(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[64, who],
                block_hash=block_hash,
            )
            if commitment:
                for c in commitment.value.get("info", {}).get("fields"):
                    if "Sha256" in c:
                        return c["Sha256"][2:]
        logger.warning(f"Failed to get commit sha256 for {block=} from {who}")
        return None

    def check_audit_report_integrity(self, record, path, content):
        """
        Check a single audit report's sha256 compared to the set_commitment call's checksum.
        """
        calculated = hashlib.sha256(content).hexdigest()
        committed = None
        try:
            committed = self.get_block_commit(record.block, record.hotkey)
        except Exception as exc:
            if "unknown Block: State already discarded" in str(exc):
                logger.warning(
                    f"State already discarded for block {record.block}, unable to verify!"
                )
                return True
        if not committed:
            logger.warning(
                f"Could not find commitment for hotkey {record.hotkey} on netuid 64 in block {record.block}"
            )
            if record.hotkey in self.validators:
                return False
            return True
        if committed != calculated:
            if record.hotkey in self.validators:
                logger.error(
                    f"Validator committed checksum does not match calculated checksum: {calculated} vs {committed} -> {record}"
                )
                return False
            else:
                logger.warning(
                    f"Miner committed checksum does not match calculated checksum: {calculated} vs {committed} -> {record}"
                )
                # Miners could try to be malicous here so we'll treat this as a warning (perhaps we can set lower weights too?)
        logger.success(
            f"Verified commitment from {record.hotkey} in block {record.block} matches sha256: {calculated}"
        )
        return True

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=10,
        max_tries=7,
    )
    async def _download_csv(self, session, vali_url, remote_path, path, expected_digest, db_record):
        # Download any reports.
        logger.info(f"Downloading and verifying {remote_path}")
        csv_path = Path(os.path.join("reports", db_record.entry_id, path)).resolve()
        try:
            csv_path.relative_to(os.path.dirname(os.path.abspath(__file__)))
        except ValueError:
            raise ValueError(f"Path {csv_path} attempts to escape base directory!")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        async with session.get(f"{vali_url}/{remote_path}") as csv_resp:
            csv_content = await csv_resp.read()
            calculated = hashlib.sha256(csv_content).hexdigest()
            if calculated != expected_digest:
                raise IntegrityViolation(
                    f"CSV export {remote_path} of validator: {vali_url} does not match!"
                )
            with open(csv_path, "wb") as outfile:
                outfile.write(csv_content)
            logger.success(f"Successfully downloaded CSV report data from {remote_path}")
        return csv_path

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=10,
        max_tries=7,
    )
    async def download_and_check_one(self, db_record):
        """
        Download and verify a single audit report (and the associated CSV exports if from validator).
        """
        # Download all exports from the validator.
        path = Path(os.path.join("reports", db_record.entry_id, db_record.path)).resolve()
        try:
            path.relative_to(os.path.dirname(os.path.abspath(__file__)))
        except ValueError:
            raise ValueError(f"Path {db_record.path} attempts to escape base directory!")
        path.parent.mkdir(parents=True, exist_ok=True)
        audit_content = None
        inv_csv_path = None
        reports_csv_path = None
        data = None
        async with self.aiosession() as session:
            async with session.get(
                "https://api.chutes.ai/audit/download", params={"path": db_record.path}
            ) as resp:
                with open(path, "wb") as outfile:
                    audit_content = await resp.read()
                    outfile.write(audit_content)
                    data = json.loads(audit_content)

                if db_record.hotkey in self.validators:
                    vali_url = self.validators[db_record.hotkey]["url"]

                    # Invocations CSV exports.
                    inv = data.get("csv_exports", {}).get("invocations")
                    if inv:
                        remote_path = inv["path"].replace("invocations/", "/invocations/exports/")
                        inv_csv_path = await self._download_csv(
                            session, vali_url, remote_path, inv["path"], inv["sha256"], db_record
                        )

                    # Reports CSV exports.
                    reports = data.get("csv_exports", {}).get("reports")
                    if reports:
                        remote_path = reports["path"].replace(
                            "invocations/", "/invocations/exports/"
                        )
                        reports_csv_path = await self._download_csv(
                            session,
                            vali_url,
                            remote_path,
                            reports["path"],
                            reports["sha256"],
                            db_record,
                        )

        # Now we can compare the sha256 of the report to the commitment on chain.
        logger.success(
            f"Successfully download audit data between {db_record.start_time} and {db_record.end_time} "
            f"for hotkey {db_record.hotkey} committed in block {db_record.block}, now verifying..."
        )
        if not self.check_audit_report_integrity(db_record, path, audit_content):
            raise IntegrityViolation(
                f"Commitment on chain does not match downloaded report! {db_record.record_id}"
            )
        return data, inv_csv_path, reports_csv_path

    async def load_invocations(self, session, csv_path):
        """
        Populate our local database with invocations from the CSV exports.
        """
        logger.info(f"Inserting invocation records from {csv_path}")
        total = 0
        with open(csv_path, "r") as infile:
            reader = csv.DictReader(infile)
            batch = []
            for row in reader:
                row_data = dict(row)
                row_data.update(
                    {
                        "miner_uid": int(row["miner_uid"]),
                        "compute_multiplier": float(row["compute_multiplier"]),
                        "bounty": int(row["bounty"]),
                        "started_at": datetime.fromisoformat(row["started_at"].rstrip("Z")).replace(
                            tzinfo=None
                        ),
                    }
                )
                if row["completed_at"]:
                    row_data.update(
                        {
                            "completed_at": datetime.fromisoformat(
                                row["completed_at"].rstrip("Z")
                            ).replace(tzinfo=None)
                        }
                    )
                else:
                    row_data["completed_at"] = None
                for key in row_data:
                    if isinstance(row_data[key], str) and not row_data[key].strip():
                        row_data[key] = None
                if row.get("metrics"):
                    try:
                        row_data["metrics"] = ast.literal_eval(row["metrics"])
                    except ValueError as exc:
                        logger.warning(f"Error parsing metrics: {exc}: {row['metrics']}")
                else:
                    row_data["metrics"] = None
                batch.append(row_data)
                total += 1
                if len(batch) == 100:
                    bulk_insert = pg_insert(Invocation).values(batch).on_conflict_do_nothing()
                    await session.execute(bulk_insert)
                    batch = []
            if batch:
                bulk_insert = pg_insert(Invocation).values(batch).on_conflict_do_nothing()
                await session.execute(bulk_insert)
            await session.commit()
        if total:
            logger.success(f"Successfully loaded {total} invocations from {csv_path}")

    async def load_reports(self, session, csv_path):
        """
        Populate our local database with invocaton reports from CSV.
        """
        logger.info(f"Inserting invocation report records from {csv_path}")
        total = 0
        with open(csv_path, "r") as infile:
            reader = csv.DictReader(infile)
            batch = []
            for row in reader:
                row_data = dict(row)
                row_data.update(
                    {
                        "timestamp": datetime.fromisoformat(row["timestamp"].rstrip("Z")).replace(
                            tzinfo=None
                        ),
                    }
                )
                if row["confirmed_at"]:
                    row_data.update(
                        {
                            "confirmed_at": datetime.fromisoformat(
                                row["confirmed_at"].rstrip("Z")
                            ).replace(tzinfo=None)
                        }
                    )
                for key in row_data:
                    if isinstance(row_data[key], str) and not row_data[key].strip():
                        row_data[key] = None
                batch.append(row_data)
                total += 1
                if len(batch) == 100:
                    bulk_insert = pg_insert(Report).values(batch).on_conflict_do_nothing()
                    await session.execute(bulk_insert)
                    batch = []
            if batch:
                bulk_insert = pg_insert(Report).values(batch).on_conflict_do_nothing()
                await session.execute(bulk_insert)
            await session.commit()
        if total:
            logger.success(f"Successfully loaded {total} invocations from {csv_path}")

    async def load_audit_entries(self, record, audit_data):
        """
        Load the deployment audit history from the report.
        """
        key = "instance_audit" if record.hotkey in self.validators else "deployment_audit"
        total = 0
        for item in audit_data.get(key, []):
            try:
                item["audit_id"] = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_OID,
                        ":".join(
                            [
                                json.dumps(item).decode(),
                                record.entry_id,
                                record.hotkey,
                            ]
                        ),
                    )
                )
                item["entry_id"] = record.entry_id
                audit = InstanceAudit(**item)
                audit.source = "validator" if record.hotkey in self.validators else "miner"
                audit.created_at = datetime.fromisoformat(audit.created_at.rstrip("Z")).replace(
                    tzinfo=None
                )
                if audit.verified_at:
                    audit.verified_at = datetime.fromisoformat(
                        audit.verified_at.rstrip("Z")
                    ).replace(tzinfo=None)
                if audit.deleted_at:
                    audit.deleted_at = datetime.fromisoformat(audit.deleted_at.rstrip("Z")).replace(
                        tzinfo=None
                    )
                if audit.miner_uid is not None:
                    audit.miner_uid = int(audit.miner_uid)
                async with get_session() as session:
                    session.add(audit)
                    await session.commit()
                total += 1
            except Exception as exc:
                logger.error(
                    f"Error populating instance audit data from {record.hotkey} {record.entry_id=}: {exc}"
                )
                logger.error(item)
        if total:
            logger.success(
                f"Populated {total} instance audit records for {record.hotkey} in {record.entry_id=}"
            )
        else:
            logger.info(
                f"No new instance audit records to process for {record.hotkey} in {record.entry_id=}"
            )

    async def load_miner_metrics(self, record, audit_data):
        """
        Load the miner-reported metrics for a given report.
        """
        total = 0
        for item in audit_data.get("prometheus_metrics", []):
            try:
                async with get_session() as session:
                    item["entry_id"] = record.entry_id
                    item["hotkey"] = record.hotkey
                    session.add(MinerMetric(**item))
                    await session.commit()
                    total += 1
            except Exception as exc:
                logger.error(
                    f"Error populating miner-reported metrics from {record.hotkey} {record.entry_id=}: {exc}"
                )
        if total:
            logger.success(
                f"Populated {total} self-reported chute metric records for {record.hotkey} in {record.entry_id=}"
            )
        else:
            logger.info(
                f"No self-reported chute metric records for {record.hotkey} in {record.entry_id=}"
            )

    async def get_weights_to_set(
        self,
        hotkeys_to_node_ids: Optional[dict[str, int]] = None,
    ) -> tuple[list[int], list[float]] | None:
        """
        Get weights to set from the invocation data.
        """

        if not hotkeys_to_node_ids:
            with self.substrate() as substrate:
                all_nodes = fetch_nodes.get_nodes_for_netuid(substrate, 64)
                hotkeys_to_node_ids = {node.hotkey: node.node_id for node in all_nodes}

        query = text(MINER_METRICS_QUERY)
        raw_compute_values = {}
        async with get_session() as session:
            compute_result = await session.execute(query)
            for hotkey, invocation_count, bounty_count, compute_units in compute_result:
                if hotkey is None:
                    continue
                raw_compute_values[hotkey] = {
                    "invocation_count": invocation_count,
                    "bounty_count": bounty_count,
                    "compute_units": compute_units,
                    "unique_chute_count": 0,
                }
            unique_query = text(UNIQUE_CHUTE_AVERAGE_QUERY)
            unique_result = await session.execute(unique_query)
            for miner_hotkey, average_active_chutes in unique_result:
                if miner_hotkey is None:
                    continue
                if miner_hotkey not in raw_compute_values:
                    raw_compute_values[miner_hotkey] = {
                        "invocation_count": 0,
                        "bounty_count": 0,
                        "compute_units": 0,
                        "unique_chute_count": 0,
                    }
                raw_compute_values[miner_hotkey]["unique_chute_count"] = average_active_chutes

        # Normalize the values based on totals so they are all in the range [0.0, 1.0]
        totals = {
            key: sum(row[key] for row in raw_compute_values.values()) or 1.0
            for key in FEATURE_WEIGHTS
        }
        normalized_values = {
            hotkey: {key: row[key] / totals[key] for key in FEATURE_WEIGHTS}
            for hotkey, row in raw_compute_values.items()
        }

        # Adjust the values by the feature weights, e.g. compute_time gets more weight than bounty count.
        final_scores = {
            hotkey: sum(norm_value * FEATURE_WEIGHTS[key] for key, norm_value in metrics.items())
            for hotkey, metrics in normalized_values.items()
        }

        # Final weights per node.
        node_ids = []
        node_weights = []
        for hotkey, compute_score in final_scores.items():
            if hotkey not in hotkeys_to_node_ids:
                logger.debug(f"Miner {hotkey} not found on metagraph. Ignoring.")
                continue

            node_weights.append(compute_score)
            node_ids.append(hotkeys_to_node_ids[hotkey])
            logger.info(f"Normalized score for {hotkey}: {compute_score}")

        return node_ids, node_weights

    async def get_and_set_weights(self):
        """
        When enabled, and you have a validator registered on the subnet, calculate and set weights from audit data.
        """
        if not self.config.set_weights.enabled or not self.ss58_address:
            logger.warning("Refusing to attempt setting weights, not enabled in config!")
            return False

        with self.substrate() as substrate:
            substrate, uid = query_substrate(
                substrate, "SubtensorModule", "Uids", [64, self.ss58_address], return_value=True
            )
            if not uid:
                logger.warning(
                    "Validator node id not found on the metagraph, are you sure "
                    f"hotkey {self.ss58_address} is registered on subnet 64?"
                )
                return False

            # Load the nodes from the metagraph.
            all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, 64)
            hotkeys_to_node_ids = {node.hotkey: node.node_id for node in all_nodes}

            # Query the audit data for weights to set.
            result = await self.get_weights_to_set(hotkeys_to_node_ids)
            if result is None:
                logger.warning("No weights to set. Skipping weight setting.")
                return
            node_ids, node_weights = result
            if len(node_ids) == 0:
                logger.warning("No nodes to set weights for. Skipping weight setting.")
                return False
            self.compare_weights_to_actual(result)

            # Set weights!
            logger.info("Weights calculated, about to set...")
            all_node_ids = [node.node_id for node in all_nodes]
            all_node_weights = [0.0 for _ in all_nodes]
            for node_id, node_weight in zip(node_ids, node_weights):
                all_node_weights[node_id] = node_weight

            logger.info(f"Node ids: {all_node_ids}")
            logger.info(f"Node weights: {all_node_weights}")
            logger.info(
                f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}"
            )
            try:
                success = weights.set_node_weights(
                    substrate=substrate,
                    keypair=self.keypair,
                    node_ids=all_node_ids,
                    node_weights=all_node_weights,
                    netuid=64,
                    version_key=VERSION_KEY,
                    validator_node_id=int(uid),
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                    max_attempts=3,
                )
            except Exception as e:
                logger.error(f"Failed to set weights: {e}")
                return False
            if success:
                logger.info("Weights set successfully.")
                return True
            else:
                logger.error("Failed to set weights :(")
                return False

    async def download_and_check_audit_reports(self) -> int:
        """
        Fetch and verify all audit reports, returning the number of new reports from validators.
        """
        async with self.aiosession() as session:
            async with session.get("https://api.chutes.ai/audit/") as resp:
                audit_records = await resp.json()
                by_id = {record["entry_id"]: record for record in audit_records}

        # Clean up the old data, plus get a list of existing items to skip.
        delete_directories = []
        async with get_session() as session:
            query = select(AuditEntry).where(
                AuditEntry.created_at
                <= func.timezone("UTC", func.now()) - timedelta(days=7, hours=1)
            )
            result = (await session.execute(query)).unique().scalars().all()
            for entry in result:
                await session.delete(entry)
                delete_directories.append(f"reports/{entry.entry_id}")
            await session.execute(
                text("DELETE FROM synthetics WHERE created_at <= NOW() - interval '169 hours'")
            )
            existing_ids = (
                (await session.execute(select(AuditEntry.entry_id))).unique().scalars().all()
            )
        for directory in delete_directories:
            logger.info(f"Purging old data: {directory}")
            shutil.rmtree(directory, ignore_errors=True)

        # Now load all new audit records.
        total = 0
        validator_total = 0
        by_id = {k: v for k, v in by_id.items() if k not in existing_ids}
        for _id, record in tqdm.tqdm(by_id.items()):
            total += 1
            if record["hotkey"] in self.validators:
                validator_total += 1
            db_record = AuditEntry(
                entry_id=record["entry_id"],
                hotkey=record["hotkey"],
                block=record["block"],
                path=record["path"],
                created_at=datetime.fromisoformat(record["created_at"].rstrip("Z")).replace(
                    tzinfo=None
                ),
                start_time=datetime.fromisoformat(record["start_time"].rstrip("Z")).replace(
                    tzinfo=None
                ),
                end_time=datetime.fromisoformat(record["end_time"].rstrip("Z")).replace(
                    tzinfo=None
                ),
            )
            logger.info(
                f"Need to verify new audit entry: {db_record.entry_id} "
                f"for data between {db_record.start_time} and {db_record.end_time}"
            )

            # Download the report data locally and verify the integrity against commitment calls.
            audit_data, inv_csv_path, reports_csv_path = await self.download_and_check_one(
                db_record
            )

            # Persist the record to DB.
            async with get_session() as session:
                session.add(db_record)
                await session.commit()
                logger.success(
                    f"Successfully verified and persisted record {db_record.entry_id} from {db_record.hotkey}"
                )
                # Load CSV invocation data if it's from a validator.
                if inv_csv_path:
                    await self.load_invocations(session, inv_csv_path)

                # Load reports CSV.
                if reports_csv_path:
                    await self.load_reports(session, reports_csv_path)

                # Persist the actual audit entry data.
                await self.load_audit_entries(db_record, audit_data)

                # Load miner-reported metrics.
                if db_record.hotkey not in self.validators:
                    try:
                        await self.load_miner_metrics(db_record, audit_data)
                    except Exception as exc:
                        logger.warning(
                            f"Error loading miner metrics in {db_record.entry_id=} for {db_record.hotkey}: {exc}"
                        )

        if not total:
            logger.info("No new audit data to process.")
        else:
            logger.success(f"Successfully processed {total} new audit reports.")
        return validator_total

    async def send_and_verify_synthetics(self):
        """
        Continuously send small quantities of synthetic requests and ensure they appear in audit data.
        """
        while self._running and self.config.synthetics.enabled:
            # Check if any of the synthetics did not appear in the audit invocation data,
            # or if the miner hotkey doesn't match, etc.
            async with get_session() as session:
                # XXX the assumption here is that the `validators` configured in the system is a list containing exactly one, the chutes vali.
                results = (
                    (
                        await session.execute(
                            text(MISSING_INVOCATIONS_QUERY.format(hotkey=list(self.validators)[0]))
                        )
                    )
                    .mappings()
                    .all()
                )
                for row in results:
                    message = "SYNTHETIC IS MISSING!\n\t" + "\n\t".join(
                        [f"{key}: {value}" for key, value in dict(row).items()]
                    )
                    logger.warning(message)

            # Send another.
            await self.perform_synthetic()
            await asyncio.sleep(60)

    def compare_weights_to_actual(self, weights_tuple):
        """
        Compare the weights we calculated to those in the metagraph.
        """
        uids, weights = weights_tuple
        weight_map = dict(zip(uids, weights))
        with self.substrate() as substrate:
            nodes = fetch_nodes.get_nodes_for_netuid(substrate, 64)
        incentive_sum = sum([node.incentive for node in nodes])
        for node in nodes:
            expected = weight_map.get(node.node_id, 0.0)
            actual = node.incentive / incentive_sum
            if expected and actual:
                delta = abs(expected - actual)
                message = f"Calculated incentive locally for {node.hotkey} [{node.node_id:3d}]: {expected:.5f} vs actual {actual:.5f}, delta {delta:.5f}"
                if delta <= 0.03:
                    logger.success(message)
                else:
                    logger.warning(message)

    async def compare_miner_metrics(self):
        """
        Check the miner reported metric data against the validator reported data.
        """
        async with get_session() as session:
            # Check how much coverage the miner has.
            hotkeys = (
                (await session.execute(text("SELECT DISTINCT(miner_hotkey) FROM invocations")))
                .scalars()
                .all()
            )
            logger.info("Checking audit data coverage for each miner with any invocations...")
            for hotkey in hotkeys:
                coverages = (
                    (await session.execute(text(MINER_COVERAGE_QUERY.format(hotkey=hotkey))))
                    .scalars()
                    .all()
                )
                for coverage_seconds in coverages:
                    if coverage_seconds != EXPECTED_COVERAGE:
                        logger.warning(
                            f"Miner {hotkey} is missing some audit report data: expecting {EXPECTED_COVERAGE} seconds but have {coverage_seconds}"
                        )
                    else:
                        logger.success(
                            f"Miner {hotkey} has full audit report coverage [{EXPECTED_COVERAGE} seconds]"
                        )

            # Compare prometheus metrics first.
            logger.info(
                "Discrepancies here are somewhat expected, because miner metrics are from ephemeral prometheus metric queries, and not particularly accurate."
            )
            metrics = (await session.execute(text(MINER_SUMMARY_METRICS_QUERY))).all()
            for row in metrics:
                hotkey, audit_count, reported_count = row
                if not reported_count:
                    logger.warning(f"Miner {hotkey} has no reported metrics...")
                    continue
                ratio = min([audit_count, reported_count]) / (
                    max([audit_count, reported_count]) or 1.0
                )
                message = f"Miner {hotkey} reported {reported_count} vs audit {audit_count}: agreement ratio {ratio:.4f}"
                if ratio < 0.9:
                    logger.warning(message)
                else:
                    logger.success(message)

    async def _verify_integrity(self):
        """
        Continuously check for new audit data, verify the numbers line up, and set weights.
        """
        first_run = True
        while self._running:
            if not await self.download_and_check_audit_reports():
                # No new data, let's see how long we should wait before trying again.
                async with get_session() as session:
                    most_recent = (
                        await session.execute(
                            select(AuditEntry).order_by(AuditEntry.start_time.desc()).limit(1)
                        )
                    ).scalar_one_or_none()
                if not most_recent:
                    # This is basically impossible?
                    logger.warning("Should not be here, why???")

                if first_run and most_recent:
                    logger.info(
                        "No new audit data, but here is the most recent weight data output from the "
                        f"report spanning {most_recent.start_time} through {most_recent.end_time}"
                    )
                    self.compare_weights_to_actual(await self.get_weights_to_set())
                    await self.compare_miner_metrics()
            else:
                if self.config.set_weights.enabled:
                    await self.get_and_set_weights()
                else:
                    # If we aren't setting weights, we can at least examine them.
                    self.compare_weights_to_actual(await self.get_weights_to_set())

                # Compare the validator stats to miner self-reported stats.
                await self.compare_miner_metrics()
            await asyncio.sleep(60)
            first_run = False

    async def verify_integrity_and_set_weights(self):
        """
        Wrapper around _verify_integrity with retries.
        """
        while True:
            try:
                await self._verify_integrity()
            except Exception as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                logger.error(
                    f"Unhandled exception performing integrity checks: {exc}\n{traceback.format_exc()}"
                )
                await asyncio.sleep(30)

    async def run(self):
        """
        Main loop, to do all the things.
        """
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        tasks = []
        try:
            tasks.append(asyncio.create_task(self.verify_integrity_and_set_weights()))
            tasks.append(asyncio.create_task(self.send_and_verify_synthetics()))
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            self._running = False
            await asyncio.gather(*tasks)


async def main():
    auditor = Auditor()
    await auditor.run()


if __name__ == "__main__":
    asyncio.run(main())
