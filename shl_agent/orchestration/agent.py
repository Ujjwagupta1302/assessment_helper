"""
agent.py
=========
The single Gemini call that drives every conversation turn.

Uses the modern `google-genai` SDK (the older `google-generativeai`
package is deprecated as of 2025).

Architecture: one round-trip per /chat request.
  1. Render the master prompt with the catalog injected (cached at startup)
  2. Format the conversation history as the user message
  3. Call Gemini with response_mime_type="application/json"
  4. Parse the JSON into a ChatResponse
  5. Return — URL validation happens in the next module
"""

import json
import os
import logging
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from shl_agent.model_schemas.models import ChatRequest, ChatResponse
from shl_agent.prompts.master_prompt import build_master_prompt, format_conversation_for_prompt
from shl_agent.data_loader.catalog_loader import get_catalog_for_prompt, get_filtered_catalog_for_prompt
from shl_agent.vector_store import embed_and_search, collection_info, embed_and_search_multi
from shl_agent.query_decomposition.query_decomposer import decompose_query
from shl_agent.orchestration.constants import *


# ============================================================================
# Load .env file (if present) before reading any environment variables
# ============================================================================
# load_dotenv() reads the .env file in the current working directory and
# adds any variables found there to os.environ — but ONLY for keys that
# aren't already set. Real environment variables (set by Docker, Render,
# Railway, etc.) take precedence, which is the correct behaviour.
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
'''
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS"))
'''


# ============================================================================
# One-time setup at module load
# ============================================================================

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY not set in environment. "
        "The agent will fail when handling requests."
    )
    _client = None
else:
    _client = genai.Client(api_key=GEMINI_API_KEY)

#TOP_K_CANDIDATES = int(os.environ.get("VECTOR_SEARCH_TOP_K"))

_VECTOR_STORE_READY = collection_info().get("ready", False)
if _VECTOR_STORE_READY:
    print(">>> AGENT MODULE: vector store check passed", flush=True)
    logger.info(
        "Vector store ready: %d items indexed, will pre-filter to top %d per request",
        collection_info().get("count", 0),
        TOP_K_CANDIDATES,
    )
else:
    logger.warning(
        "Vector store NOT ready — falling back to full catalog per request. "
        "Run `python build_index.py` to enable vector search."
    )

    

# ============================================================================
# Core function
# ============================================================================

def run_agent(request: ChatRequest) -> ChatResponse:
    """
    Run one turn of the conversation through Gemini.
 
    Pipeline per turn:
      1. Format the conversation history as a query string
      2. Use the vector store to find the top-K most relevant catalog items
         (or fall back to the full catalog if the store isn't ready)
      3. Build the master prompt with only those filtered items
      4. Call Gemini with the focused prompt
      5. Return the parsed response (or graceful fallback)
 
    Returns a ChatResponse always. Never raises.
    """
 
    if _client is None:
        return _fallback_response(
            "Service is not configured correctly. "
            "(GEMINI_API_KEY is missing.) Please try again later."
        )
 
    conversation_text = format_conversation_for_prompt(request.messages)
 
    catalog_for_prompt = _build_filtered_catalog(conversation_text)
 
    rendered_prompt = build_master_prompt(catalog_for_prompt)
 
    user_message = (
        "Conversation so far:\n\n"
        f"{conversation_text}\n\n"
        "Produce the JSON response for the most recent USER message, "
        "following all rules in the system prompt."
    )
 
    response = _call_gemini(user_message, rendered_prompt)
 
    if response is None:
        return _fallback_response(
            "I'm having trouble reaching the assessment catalog right now. "
            "Could you try again in a moment?"
        )
 
    return response
 
 
def _build_filtered_catalog(conversation_text: str) -> str:
    """
    Build the catalog subset to send to Gemini.
 
    Pipeline:
      1. Decompose the conversation into 1-4 sub-queries (LLM call)
      2. Run vector search for each sub-query
      3. Round-robin merge the results
      4. Render only those URLs as the catalog JSON
 
    Falls back to the full catalog if the vector store is unavailable
    or if no URLs come back. Falls back to a single-query search if
    the decomposer produces only one query (preserving the simple path).
    """
    if not _VECTOR_STORE_READY:
        return get_catalog_for_prompt()
 
    sub_queries = decompose_query(conversation_text)
 
    if len(sub_queries) <= 1:
        urls = embed_and_search(sub_queries[0], top_k=TOP_K_CANDIDATES)
    else:
        urls = embed_and_search_multi(sub_queries, total_top_k=TOP_K_CANDIDATES)

        #print(f">>>>>>>>>>>>>>>>>>>>>> {len(urls)} <<<<<<<<<<<<<<")
 
    if not urls:
        logger.warning("Vector search returned no results — using full catalog")
        return get_catalog_for_prompt()
 
    filtered_json = get_filtered_catalog_for_prompt(urls)
    '''
    try:
        with open("filtered_catalog_dump.json", "w", encoding="utf-8") as _f:
            json.dump(json.loads(filtered_json), _f, indent=2)
    except OSError as _e:
        logger.warning("Could not write filtered_catalog_dump.json: %s", _e)

    '''

    logger.info(
        "Filtered catalog: %d candidates from %d sub-queries "
        "(prompt size: %d chars / ~%d tokens)",
        len(urls), len(sub_queries),
        len(filtered_json), len(filtered_json) // 4,
    )
    return filtered_json
 
 
# ============================================================================
# Internals
# ============================================================================
 
def _call_gemini(
    user_message: str,
    rendered_prompt: str,
    attempt: int = 1,
) -> Optional[ChatResponse]:
    """
    Make the Gemini API call and parse the result.
 
    Returns None on unrecoverable failure (caller uses fallback).
    Retries once on JSON parse errors.
    """
 
    config = types.GenerateContentConfig(
        system_instruction=rendered_prompt,
        response_mime_type="application/json",
        temperature=0.3,
        max_output_tokens=4096,
    )
 
    try:
        result = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=config,
        )
    except Exception as exc:
        logger.exception(
            "Gemini API call failed (attempt %d): %s", attempt, exc
        )
        if attempt < 2:
            return _call_gemini(user_message, rendered_prompt, attempt + 1)
        return None
 
    raw_text = getattr(result, "text", None)
    if not raw_text:
        logger.error("Gemini returned empty response (attempt %d)", attempt)
        if attempt < 2:
            return _call_gemini(user_message, rendered_prompt, attempt + 1)
        return None
 
    return _parse_response(raw_text, user_message, rendered_prompt, attempt)
 
 
def _parse_response(
    raw_text: str,
    user_message: str,
    rendered_prompt: str,
    attempt: int,
) -> Optional[ChatResponse]:
    """
    Parse Gemini's raw text into a ChatResponse.
 
    Strips common artifacts (markdown fences) before parsing. Retries
    the whole call once if JSON is malformed.
    """
 
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
 
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "Gemini output was not valid JSON (attempt %d): %s\nRaw: %s",
            attempt, exc, raw_text[:500],
        )
        if attempt < 2:
            return _call_gemini(user_message, rendered_prompt, attempt + 1)
        return None
 
    try:
        return ChatResponse(**data)
    except ValidationError as exc:
        logger.error(
            "Gemini output failed schema validation (attempt %d): %s\nData: %s",
            attempt, exc, str(data)[:500],
        )
        if attempt < 2:
            return _call_gemini(user_message, rendered_prompt, attempt + 1)
        return None
 
 
def _fallback_response(reply: str) -> ChatResponse:
    """Graceful fallback when Gemini fails persistently."""
    return ChatResponse(
        reply=reply,
        recommendations=None,
        end_of_conversation=False,
    )
 
 
# ============================================================================
# Diagnostic helper
# ============================================================================
 
def diagnostic_info() -> dict:
    """Return diagnostic info about the agent's configuration."""
    return {
        "model": GEMINI_MODEL,
        "api_key_set": bool(GEMINI_API_KEY),
        "client_ready": _client is not None,
        "timeout_seconds": GEMINI_TIMEOUT_SECONDS,
        "vector_store": collection_info(),
        "top_k_candidates": TOP_K_CANDIDATES,
    }