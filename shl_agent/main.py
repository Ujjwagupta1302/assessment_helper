"""
main.py
========
FastAPI app for the SHL assessment recommendation agent.

Three endpoints:
  GET  /health      — liveness probe, returns {"status": "ok"}
  POST /chat        — main conversational endpoint
  GET  /diagnostic  — debugging snapshot of agent + validator config

Run locally:
  uvicorn main:app --reload --port 8000

Required environment variables:
  GEMINI_API_KEY    — Google Gemini API key

Optional environment variables:
  GEMINI_MODEL              (default: gemini-2.5-flash)
  GEMINI_TIMEOUT_SECONDS    (default: 25)
  LOG_LEVEL                 (default: INFO)
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shl_agent.model_schemas.models import ChatRequest, ChatResponse
from shl_agent.validate_schema.validation import validate_request, sanitize_response, validation_info
from shl_agent.orchestration.agent import run_agent, diagnostic_info as agent_diagnostic_info
from shl_agent.data_loader.catalog_loader import get_catalog_summary


# ============================================================================
# Logging setup
# ============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan — startup checks
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup diagnostics and log readiness state."""
    catalog = get_catalog_summary()
    logger.info(
        "Catalog loaded: %d items, %d URLs, %d unique keys",
        catalog["total_items"],
        catalog["unique_urls"],
        len(catalog["unique_keys"]),
    )

    agent = agent_diagnostic_info()
    if not agent["api_key_set"]:
        logger.warning(
            "Starting WITHOUT a Gemini API key. All /chat requests will "
            "return a fallback response until GEMINI_API_KEY is set."
        )
    else:
        vector_info = agent.get("vector_store", {})
        if vector_info.get("ready"):
            logger.info(
                "Agent ready: model=%s, vector_index=%d items, top_k=%d",
                agent["model"],
                vector_info.get("count", 0),
                agent.get("top_k_candidates", 0),
            )
        else:
            logger.warning(
                "Agent ready BUT vector index not built — falling back to full "
                "catalog per request. Run `python build_index.py` to enable "
                "vector search. (model=%s)",
                agent["model"],
            )

    yield

    logger.info("Server shutting down")


# ============================================================================
# App
# ============================================================================

app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description=(
        "Conversational agent that recommends SHL pre-employment "
        "assessments to hiring managers."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health() -> dict:
    """Liveness probe. Always returns 200 if the server is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.

    Pipeline:
      1. validate_request   — reject malformed requests (400)
      2. run_agent          — one Gemini call returning a ChatResponse
      3. sanitize_response  — strip hallucinated URLs, normalize names
      4. return             — valid ChatResponse JSON

    The pipeline is designed so that step 3 ALWAYS returns a valid
    response — there's no case in which a request that passes step 1
    fails the rest of the pipeline.
    """
    started = time.monotonic()

    validate_request(request)

    raw_response = run_agent(request)

    final = sanitize_response(raw_response, request.messages)

    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "/chat handled in %.0fms — turn=%d, recs=%s, end=%s",
        elapsed_ms,
        sum(1 for m in request.messages if m.role == "user"),
        len(final.recommendations) if final.recommendations else "null",
        final.end_of_conversation,
    )

    return final


@app.get("/diagnostic")
async def diagnostic() -> dict:
    """
    Debug snapshot of agent + validator + catalog configuration.
    Not part of the SHL spec — useful during development and testing.
    """
    return {
        "agent": agent_diagnostic_info(),
        "validator": validation_info(),
        "catalog": get_catalog_summary(),
        "log_level": LOG_LEVEL,
    }


# ============================================================================
# Global exception handler
# ============================================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """
    Catch-all for unexpected exceptions. Logs the full traceback and
    returns a clean JSON error so the SHL evaluator never gets HTML.
    """
    logger.exception("Unhandled exception in request: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": exc.__class__.__name__,
        },
    )