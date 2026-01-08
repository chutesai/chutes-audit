import io
import os
import re
import csv
import sys
import uuid
import glob
import yaml
import tqdm
import shutil
import random
import aiohttp
import asyncio
import hashlib
import backoff
import tempfile
import traceback
import numpy as np
import json as jjson
import orjson as json
import soundfile as sf
import sounddevice as sd
import pybase64 as base64

from pathlib import Path
from loguru import logger
from munch import munchify
from typing import Optional
from pydantic import BaseModel
from typing import AsyncGenerator
from datasets import load_dataset
from bittensor_wallet import Keypair
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from langdetect import detect as detect_language
from sqlalchemy.dialects.postgresql import insert
from term_image.image import from_file as image_from_file
from sqlalchemy.orm import sessionmaker, declarative_base
from async_substrate_interface import AsyncSubstrateInterface
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Double,
    Integer,
    Float,
    Boolean,
    BigInteger,
    func,
    select,
    ForeignKey,
    text,
    Index,
    or_,
)

# Database configuration.
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWD", "password"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chutes_audit")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_STR = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/chutes_audit"
engine = create_async_engine(
    DB_STR,
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

# Unified instance audit view.
INSTANCE_AUDIT_VIEW = """
CREATE OR REPLACE VIEW instance_audit AS
SELECT
  i.instance_id,
  latest.audit_id,
  latest.entry_id,
  latest.source,
  latest.deployment_id,
  latest.validator,
  latest.chute_id,
  latest.version,
  latest.miner_uid,
  latest.miner_hotkey,
  latest.region,
  latest.created_at,
  latest.verified_at,
  latest.activated_at,
  latest.compute_multiplier,
  latest.bounty,
  latest.valid_termination,
  latest.deleted_at,
  latest.deletion_reason,
  latest.stop_billing_at,
  latest.billed_to
FROM (SELECT DISTINCT instance_id FROM instance_audits) AS i
JOIN LATERAL (
  SELECT *
  FROM instance_audits ia
  WHERE ia.instance_id = i.instance_id and source = 'validator'
  ORDER BY ia.deleted_at DESC NULLS LAST, ia.verified_at DESC NULLS LAST, ia.created_at DESC
  LIMIT 1
) AS latest ON TRUE;
"""

# Query and score weighting values to use for calculating incentive/setting weights.
VERSION_KEY = 69420

SCORING_INTERVAL = "7 days"

# Instances lifetime/compute units queries - this is the entire basis for scoring!
# Uses instance_compute_history for accurate time-weighted multipliers.
# The history table includes the startup period (created_at to activated_at) at 0.3x rate,
# so billing_start uses created_at to capture this.
# All bonuses (bounty, urgency, TEE, private) are baked into compute_multiplier.
INSTANCES_QUERY = """
WITH billed_instances AS (
    SELECT
        ia.miner_hotkey,
        ia.instance_id,
        ia.chute_id,
        ia.created_at,
        ia.activated_at,
        ia.deleted_at,
        ia.stop_billing_at,
        ia.compute_multiplier,
        ia.bounty,
        -- Start from created_at to include startup period (history has 0.3x rate for this period)
        GREATEST(ia.created_at, now() - interval '{interval}') as billing_start,
        LEAST(
            COALESCE(ia.stop_billing_at, now()),
            COALESCE(ia.deleted_at, now()),
            now()
        ) as billing_end
    FROM instance_audit ia
    WHERE ia.activated_at IS NOT NULL
      AND (
          (
            ia.billed_to IS NULL
            AND ia.deleted_at IS NOT NULL
            AND ia.deleted_at - ia.activated_at >= INTERVAL '1 hour'
          )
          OR ia.valid_termination IS TRUE
          OR ia.deletion_reason in (
              'job has been terminated due to insufficient user balance',
              'user-defined/private chute instance has not been used since shutdown_after_seconds',
              'user has zero/negative balance (private chute)'
          )
          OR ia.deletion_reason LIKE '%has an old version%'
          OR ia.deleted_at IS NULL
      )
      AND (ia.deleted_at IS NULL OR ia.deleted_at >= now() - interval '{interval}')
),

-- Calculate time-weighted compute units using history table.
-- For each instance, sum (overlap_seconds * multiplier) across all history intervals.
instance_weighted_compute AS (
    SELECT
        bi.instance_id,
        bi.miner_hotkey,
        bi.billing_start,
        bi.billing_end,
        bi.bounty,
        bi.compute_multiplier as fallback_multiplier,
        COALESCE(
            SUM(
                EXTRACT(EPOCH FROM (
                    LEAST(COALESCE(ich.ended_at, now()), bi.billing_end)
                    - GREATEST(ich.started_at, bi.billing_start)
                )) * ich.compute_multiplier
            ),
            -- Fallback to instance_audit.compute_multiplier if no history exists
            EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start)) * COALESCE(bi.compute_multiplier, 1.0)
        ) AS weighted_compute_units
    FROM billed_instances bi
    LEFT JOIN instance_compute_history ich
           ON ich.instance_id = bi.instance_id
          AND ich.started_at < bi.billing_end
          AND (ich.ended_at IS NULL OR ich.ended_at > bi.billing_start)
    WHERE bi.billing_end > bi.billing_start
    GROUP BY bi.instance_id, bi.miner_hotkey, bi.billing_start, bi.billing_end, bi.bounty, bi.compute_multiplier
),

-- Aggregate compute units by miner
miner_compute_units AS (
    SELECT
        iwc.miner_hotkey,
        COUNT(*) AS total_instances,
        COUNT(CASE WHEN iwc.bounty IS TRUE THEN 1 END) AS bounty_score,
        SUM(EXTRACT(EPOCH FROM (iwc.billing_end - iwc.billing_start))) AS compute_seconds,
        SUM(iwc.weighted_compute_units) AS compute_units
    FROM instance_weighted_compute iwc
    GROUP BY iwc.miner_hotkey
)

SELECT
    miner_hotkey,
    total_instances,
    bounty_score,
    COALESCE(compute_seconds, 0) AS compute_seconds,
    COALESCE(compute_units, 0) AS compute_units
FROM miner_compute_units
ORDER BY compute_units DESC
"""


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
    activated_at = Column(DateTime(timezone=False), nullable=True)
    stop_billing_at = Column(DateTime(timezone=False), nullable=True)
    billed_to = Column(String, nullable=True)
    compute_multiplier = Column(Double, nullable=True)
    valid_termination = Column(Boolean, default=False)
    bounty = Column(Boolean, default=False)


class Job(Base):
    __tablename__ = "jobs"
    job_id = Column(String, primary_key=True)
    chute_id = Column(String, nullable=False)
    version = Column(String, nullable=False)
    chutes_version = Column(String, nullable=True)
    method = Column(String, nullable=False)
    miner_uid = Column(Integer, nullable=True)
    miner_hotkey = Column(String, nullable=True)
    miner_coldkey = Column(String, nullable=True)
    instance_id = Column(String, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    status = Column(String, nullable=False, default="pending")
    compute_multiplier = Column(Double, nullable=False)
    miner_terminated = Column(Boolean, nullable=True, default=False)


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
    processed = Column(Boolean, default=False)


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


class MetagraphNode(Base):
    __tablename__ = "metagraph_nodes"
    hotkey = Column(String, primary_key=True)
    checksum = Column(String, nullable=False)
    coldkey = Column(String, nullable=False)
    node_id = Column(Integer)
    incentive = Column(Float)
    netuid = Column(Integer)
    stake = Column(Float)
    tao_stake = Column(Float)
    alpha_stake = Column(Float)
    trust = Column(Float)
    vtrust = Column(Float)
    last_updated = Column(Integer)
    ip = Column(String)
    ip_type = Column(Integer)
    port = Column(Integer)
    protocol = Column(Integer)
    real_host = Column(String)
    real_port = Column(Integer)
    synced_at = Column(DateTime, server_default=func.now())


class GPUCount(Base):
    __tablename__ = "gpu_counts"
    chute_id = Column(String, primary_key=True)
    gpu_count = Column(Integer)


class InstanceComputeHistory(Base):
    """
    Tracks compute_multiplier changes over time for each instance.
    Used for time-weighted scoring calculations.
    """

    __tablename__ = "instance_compute_history"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    instance_id = Column(String, nullable=False, index=True)
    compute_multiplier = Column(Double, nullable=False)
    started_at = Column(DateTime(timezone=False), nullable=False)
    ended_at = Column(DateTime(timezone=False), nullable=True)

    __table_args__ = (Index("idx_ich_instance_started", "instance_id", "started_at", unique=True),)


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
        self._running = True
        self.chutes = {}
        self.subtensor_url = self.config.subtensor
        self.netuid = getattr(self.config, "netuid", 64)

        # Keypair -- only set if you are a registered validator.
        self.ss58_address = None
        self.keypair = None
        if self.config.set_weights.enabled:
            self.ss58_address = self.config.set_weights.ss58_address
            self.keypair = Keypair.create_from_seed(self.config.set_weights.secret_seed)

    @staticmethod
    @asynccontextmanager
    async def aiosession():
        """Create a fresh aiohttp session for each request."""
        async with aiohttp.ClientSession() as session:
            yield session

    async def sync_and_save_metagraph(self):
        """
        Sync metagraph to DB.
        """
        async with AsyncSubstrateInterface(url=self.subtensor_url) as substrate:
            nodes = await self._get_nodes_for_netuid(substrate, self.netuid)
        if not nodes:
            raise Exception("Failed to load metagraph nodes!")
        async with get_session() as session:
            hotkeys = ", ".join([f"'{node['hotkey']}'" for node in nodes])
            await session.execute(
                text(
                    f"DELETE FROM metagraph_nodes WHERE netuid = {self.netuid} AND hotkey NOT IN ({hotkeys}) AND node_id >= 0"
                )
            )
            for node in nodes:
                node_dict = dict(node)
                node_dict.pop("last_updated", None)
                node_dict["checksum"] = hashlib.sha256(
                    jjson.dumps(node_dict, sort_keys=True).encode()
                ).hexdigest()
                statement = pg_insert(MetagraphNode).values(node_dict)
                statement = statement.on_conflict_do_update(
                    index_elements=["hotkey"],
                    set_={key: getattr(statement.excluded, key) for key in node_dict.keys()},
                    where=MetagraphNode.checksum != node_dict["checksum"],
                )
                await session.execute(statement)
            logger.info(f"Successfully synced metagraph nodes for netuid {self.netuid}.")
            await session.commit()

    async def sync_gpu_counts(self):
        """
        Sync GPU count info.
        """
        async with self.aiosession() as session:
            async with session.get("https://api.chutes.ai/chutes/gpu_count_history") as resp:
                gpu_counts = await resp.json()

        async with get_session() as session:
            await session.execute(text("DELETE FROM gpu_counts"))
            for info in gpu_counts:
                await session.execute(
                    text(
                        "INSERT INTO gpu_counts (chute_id, gpu_count) VALUES (:chute_id, :gpu_count) ON CONFLICT (chute_id) DO NOTHING"
                    ),
                    {"chute_id": info["chute_id"], "gpu_count": info["gpu_count"]},
                )
            await session.commit()
        logger.success("Synced chute GPU counts.")

    async def reconcile_instance_audit_deletions(self, conn):
        """
        Reconcile instance_audits from the authoritative CSV feed.

        The reconciliation CSV represents the complete set of known instances from the
        validator's perspective. This method:
        1. Deletes any instance_audits records where the instance_id is not in the CSV
           (stale/old records that no longer exist)
        2. Updates deleted_at timestamps for remaining records based on the CSV
        """
        url = "https://api.chutes.ai/instances/reconciliation_csv"
        with open("reconciliation_data.csv", "w") as outfile:
            async with self.aiosession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"Unable to fetch reconciliation csv ({resp.status}), skipping reconciliation"
                        )
                        return
                    csv_content = await resp.text()
                    outfile.write(csv_content)

        # Safety check: ensure we have actual data (not just a header or empty response)
        lines = csv_content.strip().split("\n")
        if len(lines) < 2:
            logger.error(
                "Reconciliation CSV is empty or contains only headers, aborting to prevent data loss"
            )
            Path("reconciliation_data.csv").unlink(missing_ok=True)
            return

        sql_script = """
BEGIN;
CREATE TABLE IF NOT EXISTS tmp_instance_deletions (
    instance_id text PRIMARY KEY,
    deleted_at timestamp without time zone
);
TRUNCATE tmp_instance_deletions;
\\copy tmp_instance_deletions (instance_id, deleted_at) FROM 'reconciliation_data.csv' WITH (FORMAT csv, HEADER true);

-- Delete stale records: any instance_audits where instance_id is not in the reconciliation CSV
DELETE FROM instance_audits
WHERE instance_id NOT IN (SELECT instance_id FROM tmp_instance_deletions);

-- Update deleted_at for remaining records based on the CSV
UPDATE instance_audits ia
SET deleted_at = tmp.deleted_at
FROM tmp_instance_deletions tmp
WHERE ia.instance_id = tmp.instance_id
  AND ia.deleted_at IS NULL
  AND tmp.deleted_at IS NOT NULL;
COMMIT;
"""

        env = os.environ.copy()
        env.update(
            {
                "PGUSER": POSTGRES_USER,
                "PGPASSWORD": POSTGRES_PASSWORD,
                "PGHOST": POSTGRES_HOST,
                "PGPORT": str(POSTGRES_PORT),
                "PGDATABASE": POSTGRES_DB,
            }
        )
        cmd = [
            "psql",
            "--no-psqlrc",
            "-v",
            "ON_ERROR_STOP=1",
            "--dbname",
            POSTGRES_DB,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await process.communicate(sql_script.encode("utf-8"))
        except FileNotFoundError:
            logger.error("psql command not found; ensure PostgreSQL client tools are installed.")
            Path("reconciliation_data.csv").unlink(missing_ok=True)
            return

        Path("reconciliation_data.csv").unlink(missing_ok=True)

        if process.returncode != 0:
            logger.error(
                f"psql reconcile failed (exit {process.returncode}): {stderr.decode().strip()}"
            )
            return

        stdout_text = stdout.decode().strip()
        deleted_rows = 0
        updated_rows = 0
        for line in stdout_text.splitlines():
            delete_match = re.search(r"DELETE (\d+)", line)
            update_match = re.search(r"UPDATE (\d+)", line)
            if delete_match:
                deleted_rows = int(delete_match.group(1))
            if update_match:
                updated_rows = int(update_match.group(1))
        logger.info(
            f"Reconciled instance_audits: deleted {deleted_rows} stale records, "
            f"updated deleted_at for {updated_rows} records"
        )

    async def reconcile_instance_compute_history(self):
        """
        Full reconciliation of instance_compute_history by truncating and rebuilding
        from the API's source of truth. This ensures consistency with the validator's
        compute history data.
        """
        url = "https://api.chutes.ai/instances/compute_history_csv"
        async with self.aiosession() as http_session:
            async with http_session.get(url) as resp:
                resp.raise_for_status()
                csv_content = await resp.text()

        lines = csv_content.strip().split("\n")
        if len(lines) < 2:
            logger.warning("No compute history records from API, skipping reconciliation")
            return

        reader = csv.DictReader(lines)
        records = []
        for row in reader:
            started_at = datetime.fromisoformat(row["started_at"].rstrip("Z")).replace(tzinfo=None)
            ended_at = None
            if row["ended_at"] and row["ended_at"].strip():
                ended_at = datetime.fromisoformat(row["ended_at"].rstrip("Z")).replace(tzinfo=None)
            records.append(
                {
                    "instance_id": row["instance_id"],
                    "compute_multiplier": float(row["compute_multiplier"]),
                    "started_at": started_at,
                    "ended_at": ended_at,
                }
            )

        if not records:
            logger.warning("No compute history records from API, skipping reconciliation")
            return

        # Truncate and rebuild in a single transaction
        async with get_session() as session:
            await session.execute(text("TRUNCATE TABLE instance_compute_history"))
            for batch_start in range(0, len(records), 500):
                batch = records[batch_start : batch_start + 500]
                for record in batch:
                    await session.execute(
                        text("""
                            INSERT INTO instance_compute_history
                                (instance_id, compute_multiplier, started_at, ended_at)
                            VALUES (:instance_id, :compute_multiplier, :started_at, :ended_at)
                        """),
                        record,
                    )
            await session.commit()
        logger.success(
            f"Reconciled instance_compute_history: truncated and rebuilt with {len(records)} records"
        )

    async def reconcile_instance_audit(self):
        """
        Standalone entrypoint for reconciling instance_audit deletions and compute history.
        """
        async with engine.begin() as conn:
            await self.reconcile_instance_audit_deletions(conn)
        await self.reconcile_instance_compute_history()

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
                "https://api.chutes.ai/chutes/?include_public=true&limit=1000&exclude=affine"
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

    async def get_block_hash(self, substrate: AsyncSubstrateInterface, block: int) -> str:
        """
        Get a block (number) hash.
        """
        return await substrate.get_block_hash(block)

    async def get_block_commit(
        self, substrate: AsyncSubstrateInterface, block: int, who: str
    ) -> str | None:
        """
        Given a block number, fetch all set_commitment events.
        """
        logger.info(f"Attempting to process {block=}")
        block_hash = await self.get_block_hash(substrate, block)
        commitment = await substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[self.netuid, who],
            block_hash=block_hash,
        )
        if commitment:
            logger.info(f"COMMITMENT: {commitment=}")
            value = commitment.value if hasattr(commitment, "value") else commitment
            if value:
                for field in value.get("info", {}).get("fields", []):
                    # The substrate tuple encoding can wrap dicts/bytes in 1-item tuples.
                    if isinstance(field, (list, tuple)) and len(field) == 1:
                        field = field[0]
                    if isinstance(field, dict) and "Sha256" in field:
                        raw = field["Sha256"]
                        if isinstance(raw, (list, tuple)) and len(raw) == 1:
                            raw = raw[0]
                        if isinstance(raw, (list, tuple)):
                            try:
                                return bytes(raw).hex()
                            except (TypeError, ValueError):
                                pass
                        if isinstance(raw, (bytes, bytearray)):
                            return bytes(raw).hex()
        logger.warning(f"Failed to get commit sha256 for {block=} from {who}")
        return None

    async def check_audit_report_integrity(
        self, substrate: AsyncSubstrateInterface, record, path, content
    ):
        """
        Check a single audit report's sha256 compared to the set_commitment call's checksum.
        """
        calculated = hashlib.sha256(content).hexdigest()
        committed = None
        try:
            committed = await self.get_block_commit(substrate, record.block, record.hotkey)
        except Exception as exc:
            if "unknown Block: State already discarded" in str(exc):
                logger.warning(
                    f"State already discarded for block {record.block}, unable to verify!"
                )
                return True
        if not committed:
            logger.warning(
                f"Could not find commitment for hotkey {record.hotkey} on netuid {self.netuid} in block {record.block}"
            )
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
        csv_path = Path(os.path.join("/reports", db_record.entry_id, path)).resolve()
        try:
            csv_path.relative_to("/reports")
        except ValueError:
            logger.warning(f"Path {csv_path} attempts to escape base directory!")
            raise ValueError(f"Path {csv_path} attempts to escape base directory!")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        async with session.get(f"{vali_url}/{remote_path}") as csv_resp:
            csv_content = await csv_resp.read()
            calculated = hashlib.sha256(csv_content).hexdigest()
            if calculated != expected_digest:
                logger.warning(f"CSV export {remote_path} of validator: {vali_url} does not match!")
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
        path = Path(os.path.join("/reports", db_record.entry_id, db_record.path)).resolve()
        try:
            path.relative_to("/reports")
        except ValueError:
            raise ValueError(f"Path {db_record.path} attempts to escape base directory!")
        path.parent.mkdir(parents=True, exist_ok=True)
        audit_content = None
        inv_csv_path = None
        reports_csv_path = None
        jobs_csv_path = None
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

                    # Jobs CSV exports.
                    jobs = data.get("csv_exports", {}).get("jobs")
                    if jobs:
                        remote_path = jobs["path"].replace("invocations/", "/invocations/exports/")
                        jobs_csv_path = await self._download_csv(
                            session,
                            vali_url,
                            remote_path,
                            jobs["path"],
                            jobs["sha256"],
                            db_record,
                        )

        # Now we can compare the sha256 of the report to the commitment on chain.
        logger.success(
            f"Successfully download audit data between {db_record.start_time} and {db_record.end_time} "
            f"for hotkey {db_record.hotkey} committed in block {db_record.block}, now verifying..."
        )
        if db_record.hotkey in self.validators:
            async with AsyncSubstrateInterface(url=self.subtensor_url) as substrate:
                if not await self.check_audit_report_integrity(
                    substrate, db_record, path, audit_content
                ):
                    raise IntegrityViolation(
                        f"Commitment on chain does not match downloaded report! {db_record=}"
                    )
        return data, inv_csv_path, reports_csv_path, jobs_csv_path

    async def check_synthetics_in_csv(self, csv_path: str, db_record) -> None:
        """
        Check if our local synthetics appear in the validator's invocation CSV export.
        This validates that synthetic requests were properly tracked without loading
        all invocation data into the database.
        """
        # Build a set of invocation_ids from the CSV for fast lookup
        csv_invocation_ids = set()
        csv_invocations_by_id = {}
        with open(csv_path, "r", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                inv_id = row.get("invocation_id")
                if inv_id:
                    csv_invocation_ids.add(inv_id)
                    csv_invocations_by_id[inv_id] = row

        # Get synthetics that should have been tracked by this validator's report
        async with get_session() as session:
            result = await session.execute(
                text(
                    """
                    SELECT parent_invocation_id, invocation_id, instance_id,
                           chute_id, miner_uid, miner_hotkey, created_at
                    FROM synthetics
                    WHERE created_at < :end_time
                    """
                ),
                {"end_time": db_record.end_time},
            )
            synthetics = result.mappings().all()

        # Check each synthetic against the CSV
        missing_count = 0
        mismatched_count = 0
        for synthetic in synthetics:
            inv_id = synthetic["invocation_id"]
            if inv_id not in csv_invocation_ids:
                missing_count += 1
                logger.warning(
                    f"SYNTHETIC MISSING from CSV: invocation_id={inv_id} "
                    f"instance_id={synthetic['instance_id']} "
                    f"miner_hotkey={synthetic['miner_hotkey']}"
                )
            else:
                # Check miner_hotkey matches
                csv_row = csv_invocations_by_id.get(inv_id)
                if csv_row and csv_row.get("miner_hotkey") != synthetic["miner_hotkey"]:
                    mismatched_count += 1
                    logger.warning(
                        f"SYNTHETIC MINER MISMATCH: invocation_id={inv_id} "
                        f"expected={synthetic['miner_hotkey']} "
                        f"got={csv_row.get('miner_hotkey')}"
                    )

        if missing_count or mismatched_count:
            logger.warning(
                f"Synthetic validation: {missing_count} missing, {mismatched_count} mismatched "
                f"out of {len(synthetics)} synthetics checked"
            )
        else:
            logger.info(f"All {len(synthetics)} synthetics validated against CSV")

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
            logger.success(f"Successfully loaded {total} reports from {csv_path}")

    async def load_jobs(self, session, csv_path):
        """
        Populate our local database with jobs from CSV.
        """
        logger.info(f"Inserting jobs records from {csv_path}")
        total = 0
        with open(csv_path, "r") as infile:
            reader = csv.DictReader(infile)
            batch = []
            for row in reader:
                row_data = dict(row)
                for key in ("created_at", "updated_at", "started_at", "finished_at"):
                    if row_data.get(key) and row_data[key].strip():
                        row_data[key] = datetime.fromisoformat(row_data[key].rstrip("Z")).replace(
                            tzinfo=None
                        )
                for key in row_data:
                    if isinstance(row_data[key], str) and not row_data[key].strip():
                        row_data[key] = None

                # Update types.
                row_data["miner_terminated"] = (
                    row_data["miner_terminated"].strip().lower() == "true"
                )
                row_data["compute_multiplier"] = float(row_data["compute_multiplier"])
                row_data["miner_uid"] = int(row_data["miner_uid"])

                batch.append(row_data)
                total += 1
                if len(batch) == 100:
                    bulk_insert = pg_insert(Job).values(batch).on_conflict_do_nothing()
                    await session.execute(bulk_insert)
                    batch = []
            if batch:
                bulk_insert = pg_insert(Job).values(batch).on_conflict_do_nothing()
                await session.execute(bulk_insert)
            await session.commit()
        if total:
            logger.success(f"Successfully loaded {total} jobs from {csv_path}")

    async def load_audit_entries(self, record, audit_data):
        """
        Load the deployment audit history from the report.
        """
        key = "instance_audit" if record.hotkey in self.validators else "deployment_audit"
        total = 0
        valid_columns = {c.name for c in InstanceAudit.__table__.columns}
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
                audit_data_dict = {k: v for k, v in item.items() if k in valid_columns}
                audit_data_dict["source"] = (
                    "validator" if record.hotkey in self.validators else "miner"
                )
                for field in (
                    "created_at",
                    "verified_at",
                    "deleted_at",
                    "activated_at",
                    "stop_billing_at",
                ):
                    value = audit_data_dict.get(field)
                    if value:
                        audit_data_dict[field] = datetime.fromisoformat(value.rstrip("Z")).replace(
                            tzinfo=None
                        )
                if not audit_data_dict.get("billed_to"):
                    audit_data_dict["billed_to"] = None
                if audit_data_dict.get("miner_uid") is not None:
                    audit_data_dict["miner_uid"] = int(audit_data_dict["miner_uid"])
                if "valid_termination" not in audit_data_dict:
                    audit_data_dict["valid_termination"] = False
                audit_data_dict["valid_termination"] = bool(audit_data_dict["valid_termination"])
                async with get_session() as session:
                    stmt = insert(InstanceAudit).values(**audit_data_dict)
                    stmt = stmt.on_conflict_do_nothing(index_elements=["audit_id"])
                    result = await session.execute(stmt)
                    await session.commit()
                    if result.rowcount > 0:
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
        valid_columns = {c.name for c in MinerMetric.__table__.columns}
        for item in audit_data.get("prometheus_metrics", []):
            try:
                async with get_session() as session:
                    item["entry_id"] = record.entry_id
                    item["hotkey"] = record.hotkey
                    session.add(
                        MinerMetric(**{k: v for k, v in item.items() if k in valid_columns})
                    )
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

    async def load_compute_history(self, audit_data):
        """
        Load compute_history records from validator audit data.
        Uses upsert on (instance_id, started_at) to handle duplicates.
        """
        records = audit_data.get("compute_history", [])
        if not records:
            return

        total = 0
        for item in records:
            try:
                started_at = item["started_at"]
                if isinstance(started_at, str):
                    started_at = datetime.fromisoformat(started_at.rstrip("Z")).replace(tzinfo=None)
                ended_at = item.get("ended_at")
                if ended_at and isinstance(ended_at, str):
                    ended_at = datetime.fromisoformat(ended_at.rstrip("Z")).replace(tzinfo=None)

                async with get_session() as session:
                    await session.execute(
                        text("""
                            INSERT INTO instance_compute_history
                                (instance_id, compute_multiplier, started_at, ended_at)
                            VALUES (:instance_id, :compute_multiplier, :started_at, :ended_at)
                            ON CONFLICT (instance_id, started_at)
                            DO UPDATE SET
                                compute_multiplier = EXCLUDED.compute_multiplier,
                                ended_at = EXCLUDED.ended_at
                        """),
                        {
                            "instance_id": item["instance_id"],
                            "compute_multiplier": float(item["compute_multiplier"]),
                            "started_at": started_at,
                            "ended_at": ended_at,
                        },
                    )
                    await session.commit()
                    total += 1
            except Exception as exc:
                logger.error(f"Error loading compute_history record: {exc} - {item}")
        if total:
            logger.success(f"Loaded {total} compute_history records")

    @backoff.on_exception(backoff.constant, Exception, jitter=None, interval=10, max_tries=10)
    async def _get_blacklisted_hotkeys(self) -> set[str]:
        """Fetch blacklisted hotkeys from the API metagraph endpoint."""
        async with self.aiosession() as session:
            async with session.get("https://api.chutes.ai/miner/metagraph") as resp:
                resp.raise_for_status()
                nodes = await resp.json()
                assert isinstance(nodes, list) and len(nodes) > 0, "Invalid metagraph response"
                blacklisted = {n["hotkey"] for n in nodes if n.get("blacklist_reason")}
                if blacklisted:
                    logger.info(f"Found {len(blacklisted)} blacklisted miners to exclude")
                return blacklisted

    async def get_weights_to_set(
        self,
        hotkeys_to_node_ids: Optional[dict[str, int]] = None,
    ) -> tuple[list[int], list[float]] | None:
        """
        Compute miner scores based purely on compute_units (instance lifetime * compute_multiplier).
        All bonuses (bounty age, urgency, TEE, private) are baked into compute_multiplier at activation.
        """
        if not hotkeys_to_node_ids:
            async with AsyncSubstrateInterface(url=self.subtensor_url) as substrate:
                all_nodes = await self._get_nodes_for_netuid(substrate, self.netuid)
                hotkeys_to_node_ids = {node["hotkey"]: node["node_id"] for node in all_nodes}

        # Fetch blacklisted hotkeys from the API
        blacklisted_hotkeys = await self._get_blacklisted_hotkeys()

        instances_query = text(INSTANCES_QUERY.format(interval=SCORING_INTERVAL))

        # Load active miners from metagraph (and map coldkey pairings to de-dupe multi-hotkey miners).
        raw_values = {}
        logger.info("Loading metagraph for netuid=64...")
        async with get_session() as session:
            metagraph_nodes = await session.execute(
                text(
                    "SELECT coldkey, hotkey FROM metagraph_nodes WHERE netuid = 64 AND node_id >= 0"
                )
            )
            hot_cold_map = {hotkey: coldkey for coldkey, hotkey in metagraph_nodes}
            coldkey_counts = {
                coldkey: sum([1 for _, ck in hot_cold_map.items() if ck == coldkey])
                for coldkey in hot_cold_map.values()
            }

        # Base score - instances active during the scoring period.
        logger.info("Fetching scores based on active instances during scoring interval...")
        async with get_session() as session:
            instances_result = await session.execute(instances_query)
            for (
                hotkey,
                total_instances,
                bounty_score,
                instance_seconds,
                instance_compute_units,
            ) in instances_result:
                if not hotkey or hotkey not in hot_cold_map or hotkey in blacklisted_hotkeys:
                    continue
                raw_values[hotkey] = {
                    "total_instances": float(total_instances or 0.0),
                    "bounty_score": float(bounty_score or 0.0),
                    "instance_seconds": float(instance_seconds or 0.0),
                    "instance_compute_units": float(instance_compute_units or 0.0),
                }

        # Build scores from instance compute units.
        scores = {hk: data["instance_compute_units"] for hk, data in raw_values.items()}

        # Purge multi-hotkey miners - keep only the highest scoring hotkey per coldkey
        hotkeys_to_remove = set()
        for coldkey in set(hot_cold_map.values()):
            if coldkey_counts.get(coldkey, 0) > 1:
                coldkey_hotkeys = [
                    hk for hk, ck in hot_cold_map.items() if ck == coldkey and hk in scores
                ]
                if len(coldkey_hotkeys) > 1:
                    coldkey_hotkeys.sort(key=lambda hk: scores.get(hk, 0.0), reverse=True)
                    hotkeys_to_remove.update(coldkey_hotkeys[1:])

        for hotkey in hotkeys_to_remove:
            scores.pop(hotkey, None)
            raw_values.pop(hotkey, None)
            logger.warning(f"Purging hotkey from multi-uid miner: {hotkey=}")

        # Normalize to distribution.
        score_sum = sum(max(0.0, v) for v in scores.values())
        if score_sum > 0:
            final_scores = {hk: max(0.0, v) / score_sum for hk, v in scores.items()}
        else:
            n = max(len(scores), 1)
            final_scores = {hk: 1.0 / n for hk in scores.keys()}

        sorted_hotkeys = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)
        logger.info(
            f"{'#':<3} {'Hotkey':<48} {'Score':<10} {'Instances':<10} {'Seconds':<12} {'Compute':<12}"
        )
        logger.info("-" * 100)
        for rank, hotkey in enumerate(sorted_hotkeys, 1):
            data = raw_values.get(hotkey, {})
            logger.info(
                f"{rank:<3} "
                f"{hotkey:<48} "
                f"{final_scores[hotkey]:<10.6f} "
                f"{int(data.get('total_instances', 0)):<10} "
                f"{int(data.get('instance_seconds', 0)):<12} "
                f"{int(data.get('instance_compute_units', 0)):<12}"
            )

        # Final weights per node.
        node_ids = []
        node_weights = []
        for hotkey, score in final_scores.items():
            if hotkey not in hotkeys_to_node_ids:
                logger.debug(f"Miner {hotkey} not found on metagraph. Ignoring.")
                continue
            node_weights.append(score)
            node_ids.append(hotkeys_to_node_ids[hotkey])

        return node_ids, node_weights

    @staticmethod
    def _normalize_and_quantize_weights(
        node_ids: list[int], node_weights: list[float]
    ) -> tuple[list[int], list[int]]:
        """
        Normalize weights to sum to 1, then quantize to U16 values.
        """
        U16_MAX = 65535
        if not node_weights:
            return [], []

        total = sum(node_weights)
        if total <= 0:
            return node_ids, [0] * len(node_weights)

        # Normalize and quantize to U16
        normalized = [w / total for w in node_weights]
        quantized = [min(int(w * U16_MAX), U16_MAX) for w in normalized]

        # Filter out zero weights
        filtered_ids = []
        filtered_weights = []
        for nid, w in zip(node_ids, quantized):
            if w > 0:
                filtered_ids.append(nid)
                filtered_weights.append(w)

        return filtered_ids, filtered_weights

    async def _get_nodes_for_netuid(
        self, substrate: AsyncSubstrateInterface, netuid: int
    ) -> list[dict]:
        """
        Fetch all nodes for a given netuid using the SubnetInfoRuntimeApi.
        """
        from scalecodec.utils.ss58 import ss58_encode

        def _ss58_encode(address, ss58_format: int = 42) -> str:
            if isinstance(address, str):
                return address
            if isinstance(address, (list, tuple)) and len(address) > 0:
                if not isinstance(address[0], int):
                    address = address[0]
            return ss58_encode(bytes(address).hex(), ss58_format)

        response = await substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method="get_metagraph",
            params=[netuid],
        )
        metagraph = response if isinstance(response, dict) else response.value

        nodes = []
        for uid in range(len(metagraph["hotkeys"])):
            axon = metagraph["axons"][uid]
            nodes.append(
                {
                    "hotkey": _ss58_encode(metagraph["hotkeys"][uid]),
                    "coldkey": _ss58_encode(metagraph["coldkeys"][uid]),
                    "node_id": uid,
                    "netuid": metagraph["netuid"],
                    "incentive": metagraph["incentives"][uid],
                    "alpha_stake": metagraph["alpha_stake"][uid] * 10**-9,
                    "tao_stake": metagraph["tao_stake"][uid] * 10**-9,
                    "stake": metagraph["total_stake"][uid] * 10**-9,
                    "trust": 0,
                    "vtrust": metagraph["consensus"][uid],
                    "last_updated": float(metagraph["last_update"][uid]),
                    "ip": str(axon["ip"]),
                    "ip_type": axon["ip_type"],
                    "port": axon["port"],
                    "protocol": axon["protocol"],
                }
            )
        return nodes

    async def get_and_set_weights(self):
        """
        When enabled, and you have a validator registered on the subnet, calculate and set weights from audit data.
        """
        if not self.config.set_weights.enabled or not self.ss58_address:
            logger.warning("Refusing to attempt setting weights, not enabled in config!")
            return False

        async with AsyncSubstrateInterface(url=self.subtensor_url) as substrate:
            # Get validator UID
            uid_result = await substrate.query(
                module="SubtensorModule",
                storage_function="Uids",
                params=[self.netuid, self.ss58_address],
            )
            uid = uid_result.value if hasattr(uid_result, "value") else uid_result
            if uid is None:
                logger.warning(
                    "Validator node id not found on the metagraph, are you sure "
                    f"hotkey {self.ss58_address} is registered on subnet {self.netuid}?"
                )
                return False

            # Load the nodes from the metagraph.
            all_nodes = await self._get_nodes_for_netuid(substrate, self.netuid)
            hotkeys_to_node_ids = {node["hotkey"]: node["node_id"] for node in all_nodes}

            # Query the audit data for weights to set.
            result = await self.get_weights_to_set(hotkeys_to_node_ids)
            if result is None:
                logger.warning("No weights to set. Skipping weight setting.")
                return False
            node_ids, node_weights = result
            if len(node_ids) == 0:
                logger.warning("No nodes to set weights for. Skipping weight setting.")
                return False
            await self.compare_weights_to_actual(result)

            # Set weights!
            logger.info("Weights calculated, about to set...")
            all_node_ids = [node["node_id"] for node in all_nodes]
            all_node_weights = [0.0 for _ in all_nodes]
            for node_id, node_weight in zip(node_ids, node_weights):
                all_node_weights[node_id] = node_weight

            logger.info(f"Node ids: {all_node_ids}")
            logger.info(f"Node weights: {all_node_weights}")
            logger.info(
                f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}"
            )

            # Normalize and quantize weights
            quantized_ids, quantized_weights = self._normalize_and_quantize_weights(
                all_node_ids, all_node_weights
            )

            try:
                call = await substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="set_weights",
                    call_params={
                        "dests": quantized_ids,
                        "weights": quantized_weights,
                        "netuid": self.netuid,
                        "version_key": VERSION_KEY,
                    },
                )
                extrinsic = await substrate.create_signed_extrinsic(
                    call=call,
                    keypair=self.keypair,
                    era={"period": 5},
                )
                receipt = await substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=True,
                    wait_for_finalization=False,
                )
                success = await receipt.is_success
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
        logger.info("Checking for new audit report data...")
        async with self.aiosession() as session:
            async with session.get("https://api.chutes.ai/audit/") as resp:
                audit_records = await resp.json()
                by_id = {record["entry_id"]: record for record in audit_records}
        logger.info("Fetched audit data list.")

        # Clean up the old data, plus get a list of existing items to skip.
        delete_directories = []
        async with get_session() as session:
            query = select(AuditEntry).where(
                or_(
                    AuditEntry.created_at < func.timezone("UTC", func.now()) - timedelta(days=7),
                    AuditEntry.processed.is_(False),
                )
            )
            result = (await session.execute(query)).unique().scalars().all()
            for entry in result:
                logger.info(f"Purging old audit entry: {entry.entry_id}")
                await session.delete(entry)
                delete_directories.append(f"/reports/{entry.entry_id}")
            logger.info("Purging old synthetics...")
            await session.execute(
                text("DELETE FROM synthetics WHERE created_at <= NOW() - interval '169 hours'")
            )
            logger.info(f"Purging old compute history data...")
            await session.execute(
                text(
                    "DELETE FROM instance_compute_history WHERE ended_at IS NOT NULL AND ended_at <= NOW() - interval '169 hours'"
                )
            )
            logger.info("Comitting...")
            await session.commit()
            logger.success("Session committed!")

            logger.info("Collecting existing entry IDs...")
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
            elif os.getenv("SKIP_MINER_AUDITS", "true").lower() == "true":
                logger.info(f"Skipping miner audit data: {record['hotkey']}")
                continue

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
                processed=record["hotkey"] not in self.validators,
            )
            logger.info(
                f"Need to verify new audit entry: {db_record.entry_id} "
                f"for data between {db_record.start_time} and {db_record.end_time}"
            )

            # Download the report data locally and verify the integrity against commitment calls.
            (
                audit_data,
                inv_csv_path,
                reports_csv_path,
                jobs_csv_path,
            ) = await self.download_and_check_one(db_record)

            # Persist the record to DB.
            async with get_session() as session:
                session.add(db_record)
                await session.commit()
                logger.success(
                    f"Successfully verified and persisted record {db_record.entry_id} from {db_record.hotkey}"
                )
                # Check synthetics against invocation CSV if it's from a validator.
                if inv_csv_path:
                    await self.check_synthetics_in_csv(inv_csv_path, db_record)
                    # Delete synthetics that have been validated
                    await session.execute(
                        text("DELETE FROM synthetics WHERE created_at < :end_time"),
                        {"end_time": db_record.end_time},
                    )
                    await session.execute(
                        text(
                            "UPDATE audit_entries SET processed = true WHERE entry_id = :entry_id"
                        ),
                        {"entry_id": db_record.entry_id},
                    )

                # Load reports CSV.
                if reports_csv_path:
                    await self.load_reports(session, reports_csv_path)

                # Load jobs CSV.
                if jobs_csv_path:
                    await self.load_jobs(session, jobs_csv_path)

                # Persist the actual audit entry data.
                await self.load_audit_entries(db_record, audit_data)

                # Load compute_history from validator reports.
                if db_record.hotkey in self.validators:
                    await self.load_compute_history(audit_data)

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

        if validator_total > 0:
            logger.info(
                f"Reconciling instance data after processing {validator_total} new validator records..."
            )
            async with engine.begin() as conn:
                await self.reconcile_instance_audit_deletions(conn)
            await self.reconcile_instance_compute_history()

        return validator_total

    async def send_and_verify_synthetics(self):
        """
        Continuously send synthetic requests. Verification is done via check_synthetics_in_csv
        when processing validator audit CSV exports.
        """
        while self._running and self.config.synthetics.enabled:
            await self.perform_synthetic()
            await asyncio.sleep(60)

    async def compare_weights_to_actual(self, weights_tuple):
        """
        Compare the weights we calculated to those in the metagraph.
        """
        uids, weights = weights_tuple
        weight_map = dict(zip(uids, weights))
        async with AsyncSubstrateInterface(url=self.subtensor_url) as substrate:
            nodes = await self._get_nodes_for_netuid(substrate, self.netuid)
        incentive_sum = sum([node["incentive"] for node in nodes])
        if incentive_sum == 0:
            logger.warning("No incentive data in metagraph, skipping comparison.")
            return
        for node in nodes:
            expected = weight_map.get(node["node_id"], 0.0)
            actual = node["incentive"] / incentive_sum
            if expected and actual:
                delta = abs(expected - actual)
                message = f"Calculated incentive locally for {node['hotkey']} [{node['node_id']:3d}]: {expected:.5f} vs actual {actual:.5f}, delta {delta:.5f}"
                if delta <= 0.03:
                    logger.success(message)
                else:
                    logger.warning(message)

    async def compare_miner_metrics(self):
        """
        Compare miner-reported instance data against validator-reported instance data.
        For each (instance_id, miner_hotkey) pair, we get the "latest" record from each source
        and compare key fields that affect scoring.

        Keying on both instance_id and miner_hotkey prevents a malicious miner from
        claiming instances that belong to other miners.
        """
        async with get_session() as session:
            # Get latest record per (instance_id, miner_hotkey) from each source, then compare
            comparison_query = text("""
                WITH miner_latest AS (
                    SELECT DISTINCT ON (instance_id, miner_hotkey)
                        instance_id,
                        miner_hotkey,
                        created_at,
                        activated_at,
                        deleted_at,
                        compute_multiplier
                    FROM instance_audits
                    WHERE source = 'miner'
                    ORDER BY instance_id, miner_hotkey, deleted_at DESC NULLS LAST, created_at DESC
                ),
                validator_latest AS (
                    SELECT DISTINCT ON (instance_id, miner_hotkey)
                        instance_id,
                        miner_hotkey,
                        created_at,
                        activated_at,
                        deleted_at,
                        compute_multiplier
                    FROM instance_audits
                    WHERE source = 'validator'
                    ORDER BY instance_id, miner_hotkey, deleted_at DESC NULLS LAST, verified_at DESC NULLS LAST, created_at DESC
                )
                SELECT
                    COALESCE(m.instance_id, v.instance_id) as instance_id,
                    COALESCE(m.miner_hotkey, v.miner_hotkey) as miner_hotkey,
                    m.created_at as m_created,
                    v.created_at as v_created,
                    m.activated_at as m_activated,
                    v.activated_at as v_activated,
                    m.deleted_at as m_deleted,
                    v.deleted_at as v_deleted,
                    m.compute_multiplier as m_multiplier,
                    v.compute_multiplier as v_multiplier,
                    CASE WHEN m.instance_id IS NULL THEN 'miner_missing'
                         WHEN v.instance_id IS NULL THEN 'validator_missing'
                         ELSE 'both' END as presence
                FROM miner_latest m
                FULL OUTER JOIN validator_latest v
                    ON m.instance_id = v.instance_id AND m.miner_hotkey = v.miner_hotkey
                WHERE COALESCE(m.created_at, v.created_at) >= NOW() - INTERVAL '7 days'
                ORDER BY COALESCE(m.miner_hotkey, v.miner_hotkey), COALESCE(m.created_at, v.created_at) DESC
            """)

            results = (await session.execute(comparison_query)).mappings().all()
            if not results:
                logger.info("No instance audit data to compare")
                return

            # Aggregate discrepancies by miner
            miner_stats = {}
            for row in results:
                hotkey = row["miner_hotkey"]
                if hotkey not in miner_stats:
                    miner_stats[hotkey] = {
                        "total": 0,
                        "miner_only": 0,
                        "validator_only": 0,
                        "both": 0,
                        "timestamp_mismatches": 0,
                        "multiplier_mismatches": 0,
                        "miner_seconds": 0.0,
                        "validator_seconds": 0.0,
                    }

                stats = miner_stats[hotkey]
                stats["total"] += 1

                now = datetime.utcnow()
                if row["presence"] == "miner_missing":
                    stats["validator_only"] += 1
                    if row["v_activated"]:
                        v_end = row["v_deleted"] or now
                        stats["validator_seconds"] += (v_end - row["v_activated"]).total_seconds()
                elif row["presence"] == "validator_missing":
                    stats["miner_only"] += 1
                    if row["m_activated"]:
                        m_end = row["m_deleted"] or now
                        stats["miner_seconds"] += (m_end - row["m_activated"]).total_seconds()
                else:
                    stats["both"] += 1

                    # Calculate seconds for both
                    if row["m_activated"]:
                        m_end = row["m_deleted"] or now
                        stats["miner_seconds"] += (m_end - row["m_activated"]).total_seconds()
                    if row["v_activated"]:
                        v_end = row["v_deleted"] or now
                        stats["validator_seconds"] += (v_end - row["v_activated"]).total_seconds()

                    # Check for timestamp discrepancies (allowing 5 min tolerance)
                    tolerance = timedelta(minutes=5)
                    if row["m_created"] and row["v_created"]:
                        if abs(row["m_created"] - row["v_created"]) > tolerance:
                            stats["timestamp_mismatches"] += 1
                    if row["m_activated"] and row["v_activated"]:
                        if abs(row["m_activated"] - row["v_activated"]) > tolerance:
                            stats["timestamp_mismatches"] += 1
                    # deleted_at can legitimately differ if miner hasn't seen deletion yet
                    if row["m_deleted"] and row["v_deleted"]:
                        if abs(row["m_deleted"] - row["v_deleted"]) > tolerance:
                            stats["timestamp_mismatches"] += 1

                    # Check compute_multiplier match
                    m_mult = row["m_multiplier"] or 1.0
                    v_mult = row["v_multiplier"] or 1.0
                    if abs(m_mult - v_mult) > 0.01:
                        stats["multiplier_mismatches"] += 1

            # Log summary per miner
            logger.info("Instance audit comparison (miner vs validator) - last 7 days:")
            logger.info(
                f"{'Hotkey':<20} {'Total':>6} {'Both':>6} {'M-Only':>7} {'V-Only':>7} "
                f"{'TsMis':>6} {'MulMis':>7} {'M-Secs':>12} {'V-Secs':>12} {'Ratio':>6}"
            )
            logger.info("-" * 110)

            for hotkey in sorted(
                miner_stats.keys(),
                key=lambda h: miner_stats[h]["validator_seconds"],
                reverse=True,
            ):
                s = miner_stats[hotkey]
                if s["total"] == 0:
                    continue

                # Compute agreement ratio based on seconds
                max_secs = max(s["miner_seconds"], s["validator_seconds"], 1.0)
                min_secs = min(s["miner_seconds"], s["validator_seconds"])
                ratio = min_secs / max_secs

                line = (
                    f"{hotkey[:20]:<20} {s['total']:>6} {s['both']:>6} {s['miner_only']:>7} "
                    f"{s['validator_only']:>7} {s['timestamp_mismatches']:>6} "
                    f"{s['multiplier_mismatches']:>7} {int(s['miner_seconds']):>12} "
                    f"{int(s['validator_seconds']):>12} {ratio:>6.2f}"
                )

                # Flag miners with significant discrepancies
                if s["validator_only"] > s["total"] * 0.1 or ratio < 0.8:
                    logger.warning(line)
                elif s["miner_only"] > s["total"] * 0.1:
                    logger.warning(line + " (miner reporting instances validator doesn't see)")
                else:
                    logger.info(line)

    async def _verify_integrity(self):
        """
        Continuously check for new audit data, verify the numbers line up, and set weights.
        """
        first_run = True
        while self._running:
            try:
                await self.sync_and_save_metagraph()
                await self.sync_gpu_counts()
            except Exception as exc:
                logger.error(f"Unhandled exception updating metagraph/gpu counts: {exc}")
                await asyncio.sleep(30)
                continue

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
                    await self.compare_weights_to_actual(await self.get_weights_to_set())
                    await self.compare_miner_metrics()
            else:
                if self.config.set_weights.enabled:
                    await self.get_and_set_weights()
                else:
                    # If we aren't setting weights, we can at least examine them.
                    await self.compare_weights_to_actual(await self.get_weights_to_set())

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

    async def recover_instance_audits(self):
        """
        Recovery function to re-populate missing instance_audits data from existing JSON files from
        upstream schema change causing InstanceAudit(**_) constructor errors.
        """
        async with get_session() as session:
            query = """
                SELECT ae.*
                FROM audit_entries ae
                WHERE ae.hotkey = ANY(:validator_hotkeys)
                AND NOT EXISTS (
                    SELECT 1 FROM instance_audits ia
                    WHERE ia.entry_id = ae.entry_id
                )
                ORDER BY ae.created_at DESC
            """
            result = await session.execute(
                text(query), {"validator_hotkeys": list(self.validators)}
            )
            entries_to_recover = result.fetchall()
            if not entries_to_recover:
                logger.info("No entries need recovery")
                return
            logger.info(f"Found {len(entries_to_recover)} entries to recover")
            for row in entries_to_recover:
                db_record = AuditEntry(
                    entry_id=row.entry_id,
                    hotkey=row.hotkey,
                    block=row.block,
                    path=row.path,
                    created_at=row.created_at,
                    start_time=row.start_time,
                    end_time=row.end_time,
                )
                json_path = Path(os.path.join("/reports", db_record.entry_id, db_record.path))
                if not json_path.exists():
                    logger.info(f"{db_record.path} not found, attempting to download...")
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                "https://api.chutes.ai/audit/download",
                                params={"path": db_record.path},
                            ) as resp:
                                with open(json_path, "wb") as outfile:
                                    audit_content = await resp.read()
                                    outfile.write(audit_content)
                    except Exception as exc:
                        logger.warning(
                            f"Failed attempt to re-download {db_record.path}: {str(exc)}"
                        )
                        continue

                logger.info(
                    f"Recovering instance_audits for entry_id={db_record.entry_id} from {json_path}"
                )
                try:
                    with open(json_path, "r") as f:
                        audit_data = json.loads(f.read())
                    await self.load_audit_entries(db_record, audit_data)
                    logger.success(
                        f"Successfully recovered instance_audits for {db_record.entry_id}"
                    )
                except Exception as exc:
                    logger.error(f"Failed to recover {db_record.entry_id}: {exc}")
                    continue
        logger.success("Recovery complete")

    async def run(self):
        """
        Main loop, to do all the things.
        """
        async with engine.begin() as conn:
            # Drop the invocations table and all partitions - no longer needed for scoring
            await conn.execute(text("DROP TABLE IF EXISTS invocations CASCADE"))
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_reports_parent_confirmed ON reports (invocation_id) INCLUDE (confirmed_at);"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_metagraph_nodes_netuid_hotkey ON metagraph_nodes(netuid, hotkey);"
                )
            )

            await conn.execute(
                text("""
                ALTER TABLE instance_audits ADD COLUMN IF NOT EXISTS valid_termination boolean DEFAULT false
            """)
            )
            await conn.execute(
                text(
                    "ALTER TABLE instance_audits ADD COLUMN IF NOT EXISTS bounty boolean DEFAULT false"
                )
            )
            await conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_ia_id ON instance_audits(instance_id);")
            )
            await conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_ia_da ON instance_audits(deleted_at);")
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_ia_idd ON instance_audits(instance_id, deleted_at);"
                )
            )
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_ich_instance_started ON instance_compute_history(instance_id, started_at);"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_cmh_d ON instance_compute_history (ended_at)"
                )
            )
            await conn.execute(text(INSTANCE_AUDIT_VIEW))

        # Sync compute history from API on startup to ensure consistency
        await self.reconcile_instance_compute_history()

        tasks = []
        try:
            tasks.append(asyncio.create_task(self.verify_integrity_and_set_weights()))
            tasks.append(asyncio.create_task(self.send_and_verify_synthetics()))
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            self._running = False
            await asyncio.gather(*tasks)


def fix_reports_path():
    if not glob.glob("/reports/**/*.json", recursive=True) and glob.glob(
        "/audit/reports/**/*.json", recursive=True
    ):
        source_dir = "/audit/reports"
        dest_dir = "/reports"
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            dest_item = os.path.join(dest_dir, item)
            shutil.move(source_item, dest_item)
            logger.info(f"Moved: {source_item} -> {dest_item}")
        os.rmdir(source_dir)
        logger.success(f"Removed empty directory: {source_dir}")


async def main():
    auditor = Auditor()
    fix_reports_path()
    if "--recover" in sys.argv:
        await auditor.recover_instance_audits()
    elif "--reconcile" in sys.argv:
        await auditor.reconcile_instance_audit()
    else:
        await auditor.run()


if __name__ == "__main__":
    asyncio.run(main())
