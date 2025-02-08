import io
import os
import re
import random
import aiohttp
import yaml
import orjson as json
import asyncio
import tempfile
import numpy as np
import pybase64 as base64
import sounddevice as sd
import soundfile as sf
from langdetect import detect as detect_language
from term_image.image import from_file as image_from_file
from loguru import logger
from typing import AsyncGenerator
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, DateTime, Double, Integer, Boolean, BigInteger, func
from munch import munchify
from datasets import load_dataset
from contextlib import asynccontextmanager

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
    function_name = Column(String)
    user_id = Column(String)
    image_id = Column(String)
    instance_id = Column(String)
    miner_uid = Column(String)
    miner_hotkey = Column(String)
    error_message = Column(String)
    compute_multiplier = Column(Double)
    bounty = Column(Integer)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))


class InstanceAudit(Base):
    __tablename__ = "instance_audit"
    instance_id = Column(String, primary_key=True)
    chute_id = Column(String)
    version = Column(String)
    deletion_reason = Column(String)
    miner_uid = Column(String)
    miner_hotkey = Column(String)
    region = Column(String)  # not actually used right now, perhaps soon
    created_at = Column(DateTime(timezone=True))
    verified_at = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))


class AuditEntry(Base):
    __tablename__ = "audit_entries"
    entry_id = Column(String, primary_key=True)
    hotkey = Column(String)
    block = Column(BigInteger)
    path = Column(String)
    created_at = Column(DateTime(timezone=True))
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))


class Synthetic(Base):
    __tablename__ = "synthetics"
    parent_invocation_id = Column(String, primary_key=True)
    invocation_id = Column(String)
    instance_id = Column(String)
    chute_id = Column(String)
    miner_uid = Column(String)
    miner_hotkey = Column(String)
    created_at = Column(DateTime(timezone=True))
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
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
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
        self._slock = asyncio.Lock()
        self._asession = None
        self._running = True
        self.chutes = {}

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
                if (
                    chute.standard_template == "tts"
                    and isinstance(data["result"], dict)
                    and data["result"].get("bytes")
                ):
                    chunk_data = base64.b64decode(data["result"]["bytes"].encode())
                    audio_io = io.BytesIO(chunk_data)
                    audio_chunk, sr = sf.read(audio_io)
                    if len(audio_chunk.shape) > 1:
                        audio_chunk = np.mean(audio_chunk, axis=1)
                    audio_chunk = audio_chunk.astype(np.float32)
                    logger.info(f"Playing audio, turn up your volume...")
                    sd.play(audio_chunk, 24000)
                    sd.wait()
            except Exception as exc:
                logger.warning(f"Error playing audio: {exc}")

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
                        return
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
                                    invocation_id=target.invocation_id,
                                    chute_id=chute.chute_id,
                                    miner_uid=target.uid,
                                    miner_hotkey=target.hotkey,
                                    created_at=func.now(),
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
        return None

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

    async def perform_synthetic(self):
        """
        Send a single, random synthetic request.
        """
        await self.load_chutes()

        # Randomly select a task to perform.
        task_type = random.choice(
            [
                # "chat",
                # "completion",
                # "image",
                "tts",
                # "embedding",
                # "moderation",
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

    async def fetch_audit_reports(self):
        """
        Pull all audit reports.
        """
        async with self.aiosession() as session:
            async with session.get("https://api.chutes.ai/audit/") as resp:
                data = await resp.json()
                chutes = {}
                for item in data["items"]:
                    item["cords"] = data.get("cord_refs", {}).get(item["cord_ref_id"], [])
                    chutes[item["chute_id"]] = munchify(item)
                self.chutes = chutes

    async def perform_synthetics(self):
        """
        Continuously send small quantities of synthetic requests.
        """
        while self._running:
            await self.perform_synthetic()
            await asyncio.sleep(60)

    async def run(self):
        """
        Main loop, to do all the things.
        """
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        try:
            await self.perform_synthetic()
            return
        finally:
            if self._asession:
                await self._asession.close()

        tasks = []
        try:
            # tasks.append(asyncio.create_task(self.perform_synthetics()))
            # tasks.append(asyncio.create_task(self.check_integrity()))
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
