"""
build_index.py
===============
One-time script that builds the ChromaDB vector index from data.json.

Designed to work on Gemini's free tier (100 requests/minute on embeddings).

Strategy to stay under rate limits:
  1. BATCH — group multiple texts into a single API call. The SDK supports
     passing a list to embed_content, which returns a list of embeddings.
     One batch call = one rate-limit "request".
  2. THROTTLE — wait between batches so we never exceed N requests/min.
  3. RETRY — exponential backoff on 429 errors with adaptive delay
     (server tells us how long to wait via Retry-After).
  4. RESUME — if a previous run partially succeeded, skip already-indexed
     items on the next run.

Tuning knobs (edit constants below if needed):
  BATCH_SIZE       — items per API call (start: 50)
  REQUESTS_PER_MIN — target throughput (start: 80, well under 100 cap)

Run:
    python build_index.py

Expected runtime: ~5-8 minutes for 377 items on free tier.
"""

import logging
import os
import re
import sys
import time
from typing import Dict, List, Tuple

# Ensure the project root (parent of scripts/) is on sys.path so that
# `shl_agent` is importable regardless of how/where this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from google.genai import types

from shl_agent.data_loader.catalog_loader import CATALOG
from shl_agent.vector_store import (
    _embedding_client,
    _get_chroma_client,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


# ============================================================================
# Tuning constants
# ============================================================================

from scripts.constants import *


# ============================================================================
# Logging setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_index")


# ============================================================================
# Embedding text builder
# ============================================================================

def build_embedding_text(item: Dict) -> str:
    """
    Build a clean text representation of a catalog item for embedding.
    Includes the highest-signal fields: name, description, keys, job_levels.
    """
    parts = [item.get("name", "").strip()]

    description = (item.get("description") or "").strip()
    if description:
        parts.append(description)

    keys = item.get("keys") or []
    if keys:
        parts.append("Categories: " + ", ".join(keys))

    job_levels = item.get("job_levels") or []
    if job_levels:
        parts.append("Job levels: " + ", ".join(job_levels))

    return "\n\n".join(p for p in parts if p)


# ============================================================================
# 429 retry helper
# ============================================================================

_RETRY_AFTER_PATTERN = re.compile(r"retry in ([\d.]+)\s*([msMS]?)")


def _parse_retry_delay(error_message: str) -> float:
    """
    Try to extract the server-suggested retry delay from a 429 error.
    Returns seconds; defaults to 5s if we can't parse it.
    """
    match = _RETRY_AFTER_PATTERN.search(error_message)
    if not match:
        return 5.0
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "m":
        return value * 60
    return value


def _embed_one_with_retry(text: str) -> List[float]:
    """
    Embed a single text, retrying on 429 errors with adaptive backoff.
    Returns the embedding vector. Raises if all retries fail.
    """
    delay = 1.0
    for attempt in range(1, MAX_RETRIES_PER_BATCH + 1):
        try:
            result = _embedding_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="retrieval_document"),
            )
            return result.embeddings[0].values
        except Exception as exc:
            error_str = str(exc)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if not is_rate_limit:
                raise

            server_delay = _parse_retry_delay(error_str)
            wait_time = max(server_delay, delay)

            logger.warning(
                "Rate limited (attempt %d/%d). Waiting %.1fs before retry.",
                attempt, MAX_RETRIES_PER_BATCH, wait_time,
            )
            time.sleep(wait_time)
            delay = min(delay * 2, 30)

    raise RuntimeError(
        f"Item failed after {MAX_RETRIES_PER_BATCH} retries due to rate limiting"
    )


def _embed_batch_with_retry(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts one at a time (gemini-embedding-2 does not support
    true batch embedding — passing a list returns a single combined embedding).
    Returns the list of embedding vectors in the same order as the input.
    """
    return [_embed_one_with_retry(t) for t in texts]


# ============================================================================
# Main build routine
# ============================================================================

def build():
    logger.info("Starting catalog index build")
    logger.info("Catalog loaded: %d items", len(CATALOG))
    logger.info(
        "Configured: batch_size=%d, target_rpm=%d (delay between batches: %.2fs)",
        BATCH_SIZE, REQUESTS_PER_MIN, MIN_DELAY_BETWEEN_BATCHES,
    )

    client = _get_chroma_client()

    try:
        existing = client.get_collection(name=COLLECTION_NAME)
        already_indexed_ids = set(existing.get()["ids"])
        if already_indexed_ids:
            logger.info(
                "Resuming: %d items already indexed in existing collection. "
                "Will only embed the missing ones.",
                len(already_indexed_ids),
            )
        collection = existing
    except Exception:
        logger.info("Creating fresh collection")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        already_indexed_ids = set()

    to_embed: List[Tuple[Dict, str]] = []
    for item in CATALOG:
        entity_id = item.get("entity_id")
        if not entity_id:
            continue
        if entity_id in already_indexed_ids:
            continue
        url = item.get("link")
        if not url:
            logger.warning("Item %s has no URL — skipping", entity_id)
            continue
        text = build_embedding_text(item)
        if not text:
            logger.warning("Item %s has empty embedding text — skipping", entity_id)
            continue
        to_embed.append((item, text))

    if not to_embed:
        logger.info("Nothing to do — all items already indexed")
        logger.info("Collection size: %d", collection.count())
        return

    logger.info("Need to embed %d items (in batches of %d)", len(to_embed), BATCH_SIZE)

    started = time.monotonic()
    total_batches = (len(to_embed) + BATCH_SIZE - 1) // BATCH_SIZE
    failures = 0
    last_call_time = 0.0

    for batch_idx in range(total_batches):
        batch = to_embed[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        texts = [t for _, t in batch]
        items = [i for i, _ in batch]

        elapsed_since_last = time.monotonic() - last_call_time
        if last_call_time > 0 and elapsed_since_last < MIN_DELAY_BETWEEN_BATCHES:
            time.sleep(MIN_DELAY_BETWEEN_BATCHES - elapsed_since_last)
        last_call_time = time.monotonic()

        try:
            vectors = _embed_batch_with_retry(texts)
        except Exception as exc:
            logger.error("Batch %d/%d failed: %s", batch_idx + 1, total_batches, exc)
            failures += len(batch)
            continue

        ids = [item.get("entity_id") for item in items]
        documents = texts
        metadatas = [
            {
                "url": item.get("link", ""),
                "name": item.get("name", ""),
                "keys": ",".join(item.get("keys") or []),
                "job_levels": ",".join(item.get("job_levels") or []),
                "duration": item.get("duration") or "",
                "remote": item.get("remote") or "",
                "languages": ",".join(item.get("languages") or []),
            }
            for item in items
        ]
        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )

        completed_items = min((batch_idx + 1) * BATCH_SIZE, len(to_embed))
        elapsed = time.monotonic() - started
        rate = completed_items / elapsed if elapsed > 0 else 0
        remaining = (len(to_embed) - completed_items) / rate if rate > 0 else 0
        logger.info(
            "Batch %d/%d done — %d/%d items (%.1f items/sec, ~%.0fs left)",
            batch_idx + 1, total_batches,
            completed_items, len(to_embed),
            rate, remaining,
        )

    elapsed = time.monotonic() - started
    logger.info("=" * 60)
    logger.info("Build complete")
    logger.info("  Total catalog items: %d", len(CATALOG))
    logger.info("  Newly indexed:       %d", len(to_embed) - failures)
    logger.info("  Skipped (existing):  %d", len(already_indexed_ids))
    logger.info("  Failed:              %d", failures)
    logger.info("  Elapsed:             %.1fs", elapsed)
    logger.info("  Final collection:    %d items", collection.count())
    logger.info("=" * 60)

    if failures > 0:
        logger.warning(
            "%d items failed — re-run `python build_index.py` to retry them. "
            "The script resumes from where it left off.",
            failures,
        )


if __name__ == "__main__":
    if _embedding_client is None:
        logger.error(
            "GEMINI_API_KEY is not set. Add it to .env and re-run."
        )
        sys.exit(1)
    build()