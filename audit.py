import io
import os
import re
import csv
import ast
import sys
import time
import uuid
import glob
import shutil
import random
import aiohttp
import yaml
import tqdm
import psutil
import orjson as json
import asyncio
import tempfile
import hashlib
import backoff
import traceback
import subprocess
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime, timedelta
from langdetect import detect as detect_language
from term_image.image import from_file as image_from_file
from loguru import logger
from typing import AsyncGenerator
from pydantic import BaseModel
from substrateinterface import SubstrateInterface
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
from munch import munchify
from datasets import load_dataset
from contextlib import asynccontextmanager, contextmanager

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

# Invocation metrics.
NORMALIZED_COMPUTE_QUERY = """
WITH excluded_reports AS (
    SELECT invocation_id
    FROM reports
    WHERE confirmed_at IS NOT NULL
) SELECT
    i.miner_hotkey AS hotkey,
    COUNT(CASE WHEN i.completed_at IS NOT NULL AND (i.error_message IS NULL OR i.error_message = '') THEN 1 END) as successful_count,
    COUNT(CASE WHEN i.error_message IS NOT NULL AND i.error_message NOT IN ('', 'RATE_LIMIT', 'BAD_REQUEST') THEN 1 END) AS error_count,
    sum(
        i.compute_multiplier *
        CASE
            -- For token-based computations (nc = normalized compute, handles prompt & completion tokens).
            WHEN normalized_compute IS NOT NULL AND normalized_compute > 0 THEN normalized_compute

            -- For step-based computations
            WHEN i.metrics->>'steps' IS NOT NULL
                AND (i.metrics->>'steps')::float > 0
                AND i.metrics->>'masps' IS NOT NULL
            THEN (i.metrics->>'steps')::float * (i.metrics->>'masps')::float

            -- Legacy token-based computations when 'nc' is not available.
            WHEN i.metrics->>'it' IS NOT NULL
                AND i.metrics->>'ot' IS NOT NULL
                AND (i.metrics->>'it')::float > 0
                AND (i.metrics->>'ot')::float > 0
                AND i.metrics->>'maspt' IS NOT NULL
            THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * (i.metrics->>'maspt')::float

            -- Fallback to actual elapsed time
            ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
        END
    ) AS compute_units
FROM invocations i
WHERE NOT EXISTS (SELECT 1 FROM reports r WHERE r.confirmed_at IS NOT NULL AND r.invocation_id = i.parent_invocation_id)
GROUP BY hotkey;
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
    GROUP BY miner_hotkey) i
FULL OUTER JOIN
   (SELECT hotkey, SUM(total_count) as metrics_count
    FROM miner_metrics
    GROUP BY hotkey) m
ON i.miner_hotkey = m.hotkey;
"""
MINER_COVERAGE_QUERY = "SELECT SUM(EXTRACT(EPOCH FROM end_time - start_time)::integer) AS coverage_seconds FROM audit_entries WHERE hotkey = '{hotkey}' AND start_time >= (now() AT TIME ZONE 'UTC') - interval '169 hours'"
EXPECTED_COVERAGE = 7 * 24 * 60 * 60 - (60 * 60)

SCORING_INTERVAL = "7 days"

# Bonuses applied to base score, where base score is simply compute units * instance lifetime where termination reason is valid.
BONUS = {
    "demand": 0.25,  # Miner generally meets the platform demands, i.e. chutes with high utilization are deployed more frequently.
    "bounty": 0.2,  # Claimed bounties, i.e. when there was platform demand for a chute, they launched it.
    "breadth": 0.15,  # Non-selectivity of the miner, i.e. deploying all chutes with equal weight.
    "success_rate": 0.15,  # Success rate in serving requests.
}
DEMAND_COMPUTE_WEIGHT = 0.75
DEMAND_COUNT_WEIGHT = 0.25

# GPU inventory (and unique chute GPU).
INVENTORY_HISTORY_QUERY = """
WITH time_series AS (
  SELECT generate_series(
    date_trunc('hour', now() - INTERVAL '{interval}'),
    date_trunc('hour', now()),
    INTERVAL '1 hour'
  ) AS time_point
),
-- Get the latest gpu_count per chute (most recent entry only)
latest_chute_config AS (
  SELECT DISTINCT ON (chute_id)
    chute_id,
    gpu_count
  FROM gpu_counts
),
-- ALL active instances with GPU counts
active_instances_with_gpu AS (
  SELECT
    ts.time_point,
    ia.instance_id,
    ia.chute_id,
    ia.miner_hotkey,
    COALESCE(lcc.gpu_count, 1) AS gpu_count
  FROM time_series ts
  JOIN instance_audit ia
    ON ia.activated_at <= ts.time_point
   AND (ia.deleted_at IS NULL OR ia.deleted_at >= ts.time_point)
   AND ia.activated_at IS NOT NULL
   AND (
        ia.billed_to IS NOT NULL
        OR (COALESCE(ia.deleted_at, ts.time_point) - ia.activated_at >= interval '1 hour')
   )
  LEFT JOIN latest_chute_config lcc
    ON ia.chute_id = lcc.chute_id
),
-- Calculate metrics per timepoint
metrics_per_timepoint AS (
  SELECT
    time_point,
    miner_hotkey,
    -- For breadth: unique chutes with GPU weighting
    (SELECT SUM(gpu_count) FROM (
      SELECT DISTINCT ON (chute_id) chute_id, gpu_count
      FROM active_instances_with_gpu aig2
      WHERE aig2.time_point = aig.time_point
        AND aig2.miner_hotkey = aig.miner_hotkey
    ) unique_chutes) AS gpu_weighted_unique_chutes,
    -- For stability: total GPUs across all instances
    SUM(gpu_count) AS total_active_gpus
  FROM active_instances_with_gpu aig
  GROUP BY time_point, miner_hotkey
)
-- Return the history for both metrics
SELECT
  time_point::text,
  miner_hotkey,
  COALESCE(gpu_weighted_unique_chutes, 0) AS unique_chute_gpus,
  COALESCE(total_active_gpus, 0) AS total_active_gpus
FROM metrics_per_timepoint
ORDER BY miner_hotkey, time_point
"""
INVENTORY_QUERY = (
    """
SELECT
  miner_hotkey,
  AVG(unique_chute_gpus)::integer AS avg_unique_chute_gpus,
  AVG(total_active_gpus)::integer AS avg_total_active_gpus
FROM ("""
    + INVENTORY_HISTORY_QUERY
    + """) AS history_data
GROUP BY miner_hotkey
ORDER BY avg_unique_chute_gpus DESC
"""
)

# Instances lifetime/compute units queries - this is the entire basis for scoring!
INSTANCES_QUERY = """
WITH billed_instances AS (
    SELECT
        ia.miner_hotkey,
        ia.instance_id,
        ia.activated_at,
        ia.deleted_at,
        ia.stop_billing_at,
        ia.compute_multiplier,
        ia.bounty,
        GREATEST(ia.activated_at, now() - interval '{interval}') as billing_start,
        LEAST(
            COALESCE(ia.stop_billing_at, now()),
            COALESCE(ia.deleted_at, now()),
            now()
        ) as billing_end
    FROM instance_audit ia
    WHERE ia.activated_at IS NOT NULL
      AND (
          -- public instances, must be alive for >= 1 hour to be counted.
          (
            ia.billed_to IS NULL
            AND ia.deleted_at IS NOT NULL
            AND ia.deleted_at - ia.activated_at >= INTERVAL '1 hour'
          )
          -- terminated for a valid reason, i.e. validator scaled it down because low usage
          OR ia.valid_termination IS TRUE
          -- legacy termination reasons
          OR ia.deletion_reason in (
              'job has been terminated due to insufficient user balance',
              'user-defined/private chute instance has not been used since shutdown_after_seconds',
              'user has zero/negative balance (private chute)'
          )
          OR ia.deletion_reason LIKE '%has an old version%'
          -- instances that are still active, tenatively assume with meet the other criteria
          OR ia.deleted_at IS NULL
      )
      AND (ia.deleted_at IS NULL OR ia.deleted_at >= now() - interval '{interval}')
),

-- Aggregate compute units by miner
miner_compute_units AS (
    SELECT
        miner_hotkey,
        COUNT(*) AS total_instances,
        COUNT(CASE WHEN bounty IS TRUE THEN 1 END) AS bounties,
        SUM(EXTRACT(EPOCH FROM (billing_end - billing_start))) AS compute_seconds,
        SUM(EXTRACT(EPOCH FROM (billing_end - billing_start)) * compute_multiplier) AS compute_units
    FROM billed_instances
    WHERE billing_end > billing_start
    GROUP BY miner_hotkey
)
SELECT
    miner_hotkey,
    total_instances,
    bounties,
    COALESCE(compute_seconds, 0) as compute_seconds,
    COALESCE(compute_units, 0) as compute_units
FROM miner_compute_units order by compute_units desc
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


@lru_cache(maxsize=1)
def get_optimal_worker_count(ram_per_worker_gb=4):
    """
    Calculate how many workers to use for invocation processing, which is
    basically just free RAM based (ensure each worker has 4GB).
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    max_workers_by_ram = int(available_gb / ram_per_worker_gb)
    cpu_count = psutil.cpu_count(logical=True)
    optimal_workers = max(1, min(max_workers_by_ram, cpu_count))
    logger.info(f"System RAM: {mem.total / (1024**3):.1f}GB total, {available_gb:.1f}GB available")
    logger.info(f"CPU cores: {cpu_count}")
    logger.info(f"Optimal workers: {optimal_workers} (based on {ram_per_worker_gb}GB per worker)")
    return optimal_workers


def escape_for_copy(val):
    """
    Helper function to escape values for COPY format.
    """
    if val is None:
        return "\\N"
    if isinstance(val, (dict, list)):
        import orjson

        val = orjson.dumps(val).decode()
    else:
        val = str(val)
    return val.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def transform_invocation(row):
    """
    Transform a single invocation row.
    """
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
                "completed_at": datetime.fromisoformat(row["completed_at"].rstrip("Z")).replace(
                    tzinfo=None
                )
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
        except ValueError:
            row_data["metrics"] = None
    else:
        row_data["metrics"] = None
    return row_data


def process_chunk(chunk):
    """
    Process a chunk of invocations.
    """
    return [transform_invocation(row) for row in chunk if int(row["miner_uid"]) >= 0]


INVOCATION_COLS = [
    "entry_id",
    "parent_invocation_id",
    "invocation_id",
    "chute_id",
    "chute_user_id",
    "function_name",
    "user_id",
    "image_id",
    "image_user_id",
    "instance_id",
    "miner_uid",
    "miner_hotkey",
    "error_message",
    "compute_multiplier",
    "bounty",
    "metrics",
    "started_at",
    "completed_at",
]


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

    async def sync_and_save_metagraph(self):
        """
        Sync metagraph to DB.
        """
        with self.substrate() as substrate:
            nodes = fetch_nodes.get_nodes_for_netuid(substrate, 64)
        if not nodes:
            raise Exception("Failed to load metagraph nodes!")
        async with get_session() as session:
            hotkeys = ", ".join([f"'{node.hotkey}'" for node in nodes])
            await session.execute(
                text(
                    f"DELETE FROM metagraph_nodes WHERE netuid = 64 AND hotkey NOT IN ({hotkeys}) AND node_id >= 0"
                )
            )
            for node in nodes:
                node_dict = node.dict()
                node_dict.pop("last_updated", None)
                node_dict["checksum"] = hashlib.sha256(json.dumps(node_dict)).hexdigest()
                statement = pg_insert(MetagraphNode).values(node_dict)
                statement = statement.on_conflict_do_update(
                    index_elements=["hotkey"],
                    set_={
                        key: getattr(statement.excluded, key) for key, value in node_dict.items()
                    },
                    where=MetagraphNode.checksum != node_dict["checksum"],
                )
                await session.execute(statement)
            logger.info("Successfully synced metagraph nodes for netuid 64.")
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

    async def reconcile_instance_audit_deletions(self, conn):
        """
        Backfill missing deleted_at values in instance_audits from the authoritative CSV feed.
        """
        url = "https://api.chutes.ai/instances/reconciliation_csv"
        with open("reconciliation_data.csv", "w") as outfile:
            async with self.aiosession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"Unable to fetch reconciliation csv ({resp.status}), skipping deleted_at backfill"
                        )
                        return
                    outfile.write(await resp.text())

        sql_script = f"""
BEGIN;
CREATE TABLE IF NOT EXISTS tmp_instance_deletions (
    instance_id text PRIMARY KEY,
    deleted_at timestamp without time zone
);
TRUNCATE tmp_instance_deletions;
\\copy tmp_instance_deletions (instance_id, deleted_at) FROM 'reconciliation_data.csv' WITH (FORMAT csv, HEADER true);
UPDATE instance_audits ia
SET deleted_at = tmp.deleted_at
FROM tmp_instance_deletions tmp
WHERE ia.instance_id = tmp.instance_id
  AND ia.deleted_at IS NULL
  AND tmp.deleted_at IS NOT NULL
  AND NOT EXISTS (
    SELECT 1
    FROM instance_audits ia_existing
    WHERE ia_existing.instance_id = ia.instance_id
      AND ia_existing.deleted_at IS NOT NULL
  );
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
            Path(tmp_csv_path).unlink(missing_ok=True)
            return

        Path('reconciliation_data.csv').unlink(missing_ok=True)

        if process.returncode != 0:
            logger.error(
                f"psql reconcile failed (exit {process.returncode}): {stderr.decode().strip()}"
            )
            return

        stdout_text = stdout.decode().strip()
        updated_rows = 0
        for line in reversed(stdout_text.splitlines()):
            match = re.search(r"UPDATE (\\d+)", line)
            if match:
                updated_rows = int(match.group(1))
                break
        logger.info(
            f"Reconciled deleted_at for {updated_rows} instance_audits records via psql bulk import"
        )

    async def reconcile_instance_audit(self):
        """
        Standalone entrypoint for reconciling instance_audit deletions.
        """
        async with engine.begin() as conn:
            await self.reconcile_instance_audit_deletions(conn)

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
        if db_record.hotkey in self.validators and not self.check_audit_report_integrity(
            db_record, path, audit_content
        ):
            raise IntegrityViolation(
                f"Commitment on chain does not match downloaded report! {db_record.record_id}"
            )
        return data, inv_csv_path, reports_csv_path, jobs_csv_path

    async def load_invocations(self, csv_path: str, entry_id: str) -> int:
        """
        Load (after transform) invocations from CSV export into Postgres.
        """
        logger.info(f"Loading invocation records to transform and load into DB from {csv_path}")
        with open(csv_path, "r", newline="") as infile:
            reader = csv.DictReader(infile)
            rows = []
            for row in reader:
                row["entry_id"] = entry_id
                rows.append(row)
        if not rows:
            logger.info("No rows to process")
            return 0

        logger.info(f"Read {len(rows)} rows from CSV, starting transformation...")
        base_chunk_size = max(1, len(rows) // 100)
        transform_chunks = [
            rows[i : i + base_chunk_size] for i in range(0, len(rows), base_chunk_size)
        ]
        transformed_rows = []
        with ProcessPoolExecutor(max_workers=get_optimal_worker_count()) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in transform_chunks]
            for future in as_completed(futures):
                try:
                    transformed_rows.extend(future.result())
                except Exception as e:
                    logger.error(f"Chunk transformation failed: {e}")
        if not transformed_rows:
            logger.warning("No rows transformed")
            return 0

        logger.info(f"Transformation complete, {len(transformed_rows)} rows ready to load...")
        timestamp = int(time.time() * 1000)
        temp_csv_path = f"/tmp/invocations_load_{timestamp}.csv"
        try:
            with open(temp_csv_path, "w", newline="") as csvfile:
                for row in transformed_rows:
                    fields = [escape_for_copy(row.get(col)) for col in INVOCATION_COLS]
                    csvfile.write("\t".join(fields) + "\n")
            logger.info(f"Wrote {len(transformed_rows)} rows to temporary CSV: {temp_csv_path}")
            part = f"inv_{entry_id.replace('-', '')}"
            async with get_session() as session:
                await session.execute(text(f"DROP TABLE IF EXISTS {part}"))
                await session.execute(
                    text(
                        f"CREATE TABLE {part} PARTITION OF invocations FOR VALUES IN ('{entry_id}')"
                    )
                )
                await session.commit()
            cmd = [
                "psql",
                "-h",
                POSTGRES_HOST,
                "-p",
                str(POSTGRES_PORT),
                "-U",
                POSTGRES_USER,
                "-d",
                POSTGRES_DB,
                "-c",
                f"\\copy {part} ({','.join(INVOCATION_COLS)}) FROM '{temp_csv_path}' WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')",
            ]
            logger.info("Running psql COPY command...")
            env = os.environ.copy()
            env["PGPASSWORD"] = POSTGRES_PASSWORD
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                logger.error(f"psql COPY failed: {result.stderr}")
                raise Exception(f"psql COPY failed: {result.stderr}")
            logger.info("Successfully copied data to invocations table!")
            return len(transformed_rows)
        except Exception as e:
            logger.error(f"Failed to load invocations: {e}")
            raise
        finally:
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                logger.debug(f"Cleaned up temporary CSV: {temp_csv_path}")

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

        compute_query = text(NORMALIZED_COMPUTE_QUERY.format(interval=SCORING_INTERVAL))
        inventory_query = text(INVENTORY_QUERY.format(interval=SCORING_INTERVAL))
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
        logger.info(
            "Fetching base score values based on active instances during scoring interval..."
        )
        async with get_session() as session:
            instances_result = await session.execute(instances_query)
            for (
                hotkey,
                total_instances,
                bounties,
                instance_seconds,
                instance_compute_units,
            ) in instances_result:
                if not hotkey or hotkey not in hot_cold_map:
                    continue
                raw_values[hotkey] = {
                    "total_instances": total_instances,
                    "bounties": bounties,
                    "instance_seconds": instance_seconds,
                    "instance_compute_units": instance_compute_units,
                    "success_rate": 0.0,
                    "invocation_compute_units": 0.0,
                    "invocation_count": 0.0,
                    "unique_chute_gpus": 0.0,
                }

        # Get the invocation metrics to calculate boosts for "demand" and "success_ratio"
        logger.info("Fetching invocation metrics to calculate demand and success ratio boosts...")
        async with get_session() as session:
            compute_result = await session.execute(compute_query)
            for hotkey, successful_count, error_count, compute_units in compute_result:
                if hotkey not in raw_values:
                    continue
                raw_values[hotkey]["success_rate"] = successful_count / (
                    (successful_count + error_count) or 1.0
                )
                raw_values[hotkey]["invocation_compute_units"] = compute_units
                raw_values[hotkey]["invocation_count"] = successful_count

        # Get the unique chute ("breadth" bonus) data.
        logger.info("Fetching unique chute GPU score to calculate breadth bonus...")
        async with get_session() as session:
            unique_result = await session.execute(inventory_query)
            for hotkey, unique_chute_gpus, total_active_gpus in unique_result:
                if hotkey not in raw_values:
                    continue
                raw_values[hotkey]["unique_chute_gpus"] = unique_chute_gpus

        # First, we'll calculate the scores as [0,1] range based on compute units.
        logger.info("Normalizing scores and adding boosts...")
        base_scores = {}
        for hotkey, data in raw_values.items():
            base_scores[hotkey] = data["instance_compute_units"]

        # Purge multi-hotkey miners - keep only the highest scoring hotkey per coldkey
        hotkeys_to_remove = set()
        for coldkey in set(hot_cold_map.values()):
            if coldkey_counts[coldkey] > 1:
                coldkey_hotkeys = [
                    hk for hk, ck in hot_cold_map.items() if ck == coldkey and hk in base_scores
                ]
                if len(coldkey_hotkeys) > 1:
                    coldkey_hotkeys.sort(key=lambda hk: base_scores[hk], reverse=True)
                    hotkeys_to_remove.update(coldkey_hotkeys[1:])

        # Remove the lower-scoring hotkeys
        for hotkey in hotkeys_to_remove:
            base_scores.pop(hotkey, None)
            raw_values.pop(hotkey, None)
            logger.warning(f"Purging hotkey from multi-uid miner: {hotkey=}")

        # Helper function to normalize and apply exponential
        def normalize_and_exp(values_dict, key, exp=1.4):
            values = [data.get(key, 0) for data in values_dict.values()]
            max_val = max(values) if values else 1.0
            min_val = min(values) if values else 0.0
            range_val = max_val - min_val if max_val != min_val else 1.0
            normalized = {}
            for hotkey, data in values_dict.items():
                norm_val = (data.get(key, 0) - min_val) / range_val if range_val > 0 else 0
                normalized[hotkey] = norm_val**exp
            exp_max = max(normalized.values()) if normalized else 1.0
            if exp_max > 0:
                for hotkey in normalized:
                    normalized[hotkey] /= exp_max
            return normalized

        # Breadth bonus (unique_chute_gpus, non-selectiveness in deploying chutes).
        breadth_scores = normalize_and_exp(raw_values, "unique_chute_gpus", 2.0)

        # Demand bonus (miner deploys chutes that get a lot of real-world invocation usage).
        invoc_compute_scores = normalize_and_exp(raw_values, "invocation_compute_units", 2.0)
        invoc_count_scores = normalize_and_exp(raw_values, "invocation_count", 2.0)
        demand_scores = {}
        for hotkey in raw_values:
            demand_scores[hotkey] = DEMAND_COMPUTE_WEIGHT * invoc_compute_scores.get(
                hotkey, 0
            ) + DEMAND_COUNT_WEIGHT * invoc_count_scores.get(hotkey, 0)

        # Bounties (miner was first to activate an instance of a chute that had a bounty).
        bounty_scores = normalize_and_exp(raw_values, "bounties", 2.0)

        # Success rate (miner generally has a higher success rate in invocations).
        success_scores = normalize_and_exp(raw_values, "success_rate", 2.0)

        # Normalize the base scores to sum to 1.0
        total_base = sum(base_scores.values()) if base_scores else 1.0
        if total_base > 0:
            for hotkey in base_scores:
                base_scores[hotkey] /= total_base

        # Apply bonuses.
        final_scores = {}
        for hotkey in base_scores:
            score = base_scores[hotkey]
            # Add each bonus.
            score += base_scores[hotkey] * BONUS["breadth"] * breadth_scores.get(hotkey, 0)
            score += base_scores[hotkey] * BONUS["demand"] * demand_scores.get(hotkey, 0)
            score += base_scores[hotkey] * BONUS["bounty"] * bounty_scores.get(hotkey, 0)
            score += base_scores[hotkey] * BONUS["success_rate"] * success_scores.get(hotkey, 0)
            final_scores[hotkey] = score

        # Normalize to ensure sum equals 1.0
        total_final = sum(final_scores.values()) if final_scores else 1.0
        if total_final > 0:
            for hotkey in final_scores:
                final_scores[hotkey] /= total_final

        # Logging.
        sorted_hotkeys = sorted(final_scores.keys(), key=lambda k: final_scores[k], reverse=True)
        logger.info(
            f"{'#':<3} "
            f"{'Hotkey':<48} "
            f"{'Score':<10} "
            f"{'Base':<10} "
            f"{'Breadth':<10} "
            f"{'Demand':<10} "
            f"{'Bounty':<10} "
            f"{'Success':<10}"
        )
        logger.info("-" * 120)
        for rank, hotkey in enumerate(sorted_hotkeys, 1):
            logger.info(
                f"{rank:<3} "
                f"{hotkey:<48} "
                f"{final_scores[hotkey]:<10.6f} "
                f"{base_scores.get(hotkey, 0):<10.6f} "
                f"{BONUS['breadth'] * breadth_scores.get(hotkey, 0):<10.6f} "
                f"{BONUS['demand'] * demand_scores.get(hotkey, 0):<10.6f} "
                f"{BONUS['bounty'] * bounty_scores.get(hotkey, 0):<10.6f} "
                f"{BONUS['success_rate'] * success_scores.get(hotkey, 0):<10.6f}"
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
                or_(
                    AuditEntry.created_at < func.timezone("UTC", func.now()) - timedelta(days=7),
                    AuditEntry.processed.is_(False),
                )
            )
            result = (await session.execute(query)).unique().scalars().all()
            for entry in result:
                await session.delete(entry)
                delete_directories.append(f"/reports/{entry.entry_id}")
                await session.execute(
                    text(f"DROP TABLE IF EXISTS inv_{entry.entry_id.replace('-', '')}")
                )
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
                # Load CSV invocation data if it's from a validator.
                if inv_csv_path:
                    if await self.load_invocations(inv_csv_path, db_record.entry_id):
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
            if os.getenv("CALCULATE_MINER_AGREEMENT", "false").lower() == "true":
                logger.info("Calculating miner agreement ratio...")
                metrics = (await session.execute(text(MINER_SUMMARY_METRICS_QUERY))).all()
                logger.info(
                    "Discrepancies here are somewhat expected, because miner metrics are from ephemeral prometheus metric queries, and not particularly accurate."
                )
                for row in metrics:
                    hotkey, audit_count, reported_count = row
                    if not audit_count or not reported_count:
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
            await conn.execute(
                text("""
                DO $$
                DECLARE
                    is_partitioned BOOLEAN;
                BEGIN
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_class c
                        WHERE c.relname = 'invocations'
                        AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = current_schema())
                        AND c.relkind = 'r'
                    ) INTO is_partitioned;
                    IF is_partitioned THEN
                        DROP TABLE invocations;
                        RAISE NOTICE 'Dropped non-partitioned table invocations';
                    ELSE
                        RAISE NOTICE 'Table invocations does not exist or is partitioned, skipping drop';
                    END IF;
                END $$;
                """)
            )
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS invocations (
                    entry_id text not null,
                    parent_invocation_id character varying,
                    invocation_id character varying NOT NULL,
                    chute_id character varying,
                    chute_user_id character varying,
                    function_name character varying,
                    user_id character varying,
                    image_id character varying,
                    image_user_id character varying,
                    instance_id character varying,
                    miner_uid integer,
                    miner_hotkey character varying,
                    error_message character varying,
                    compute_multiplier double precision,
                    bounty integer,
                    metrics jsonb,
                    started_at timestamp without time zone,
                    completed_at timestamp without time zone,
                    is_private boolean GENERATED ALWAYS AS ((metrics->>'p')::bool) STORED,
                    normalized_compute double precision GENERATED ALWAYS AS ((metrics->>'nc')::float) STORED
                ) PARTITION BY LIST(entry_id);
                """)
            )
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_reports_parent_confirmed ON reports (invocation_id) INCLUDE (confirmed_at);"
                )
            )
            await conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_invocations_id ON invocations(invocation_id);")
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_metagraph_nodes_netuid_hotkey ON metagraph_nodes(netuid, hotkey);"
                )
            )

            await conn.execute(
                text("""
                ALTER TABLE instance_audits ADD COLUMN IF NOT EXISTS valid_termination boolean NOT NULL DEFAULT false
            """)
            )
            await conn.execute(
                text("""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'instance_audits'
                  AND column_name = 'bounty'
              ) THEN
                ALTER TABLE instance_audits
                ADD COLUMN bounty boolean DEFAULT false;

                UPDATE instance_audits ia
                SET bounty = true
                WHERE instance_id IN (
                  SELECT DISTINCT(instance_id)
                  FROM invocations
                  WHERE bounty > 0
                );
              END IF;
            END
            $$;
            """)
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
            await conn.execute(text(INSTANCE_AUDIT_VIEW))

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
