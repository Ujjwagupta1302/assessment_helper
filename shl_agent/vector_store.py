"""
vector_store.py
================
ChromaDB wrapper for semantic catalog search.

This module exposes two operations:

  embed_and_search(query, top_k) -> List[str]
      Embed a query string with Gemini and return the top-K most
      similar catalog URLs.

  get_collection()
      Load or create the persistent ChromaDB collection. Used by both
      the runtime search and the one-time build script.

The collection is stored on disk at ./chroma_db/. In production it is
built fresh at deploy time by build_index.py, then read-only during
request handling.

Embeddings use Gemini's text-embedding-004 model:
  - retrieval_document type for catalog items (one-time, at index)
  - retrieval_query type for user queries (per-turn, at request)
The two types are tuned for asymmetric retrieval and slightly improve
relevance over using the same type for both.
"""

import logging
import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from shl_agent.constants import *


# Load .env BEFORE reading environment variables. load_dotenv is idempotent
# and respects pre-existing environment variables (production env vars from
# Render/Railway take precedence over .env). This makes the module robust
# to import-order issues between agent.py and vector_store.py.
load_dotenv()


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================



GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    _embedding_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    _embedding_client = None
    logger.warning("GEMINI_API_KEY not set — embeddings will fail at request time")


# ============================================================================
# ChromaDB collection access
# ============================================================================

_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    """Lazily build a singleton ChromaDB client pointing at the persistent dir."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection(create_if_missing: bool = False):
    """
    Return the catalog collection.

    create_if_missing=True is used only by the build script. At request
    time we want to fail loudly if the collection wasn't built yet.
    """
    client = _get_chroma_client()
    if create_if_missing:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=COLLECTION_NAME)


# ============================================================================
# Embedding
# ============================================================================

def embed_text(text: str, task_type: str = "retrieval_query") -> List[float]:
    """
    Embed a single text string with Gemini's text-embedding-004.

    task_type should be:
      - "retrieval_query"     for user-side queries (default)
      - "retrieval_document"  for catalog items at index time
    """
    if _embedding_client is None:
        raise RuntimeError("Cannot embed — GEMINI_API_KEY is not configured")

    result = _embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values


# ============================================================================
# Search
# ============================================================================

def embed_and_search(query: str, top_k: int = 30) -> List[str]:
    """
    Embed a query and return the top-K most similar catalog URLs.

    Returns an empty list if anything goes wrong — the caller should
    fall back to using the full catalog rather than crash.
    """
    if not query.strip():
        return []

    try:
        query_embedding = embed_text(query, task_type="retrieval_query")
    except Exception as exc:
        logger.exception("Failed to embed query: %s", exc)
        return []

    try:
        collection = get_collection()
    except Exception as exc:
        logger.exception("Failed to load Chroma collection: %s", exc)
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas"],
        )
    except Exception as exc:
        logger.exception("ChromaDB query failed: %s", exc)
        return []

    metadatas = results.get("metadatas") or [[]]
    if not metadatas or not metadatas[0]:
        return []

    urls = [m.get("url", "") for m in metadatas[0] if m and m.get("url")]
    return urls


# ============================================================================
# Diagnostics
# ============================================================================

def collection_info() -> dict:
    """Return basic diagnostic info about the collection state."""
    try:
        coll = get_collection()
        return {
            "path": CHROMA_DB_PATH,
            "name": COLLECTION_NAME,
            "count": coll.count(),
            "embedding_model": EMBEDDING_MODEL,
            "ready": True,
        }
    except Exception as exc:
        return {
            "path": CHROMA_DB_PATH,
            "name": COLLECTION_NAME,
            "ready": False,
            "error": str(exc),
        }