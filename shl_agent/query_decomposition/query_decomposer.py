"""
query_decomposer.py
====================
Decomposes a conversation into independent semantic sub-queries for
multi-query vector retrieval.

The motivation: a single embedding of "Java developer who speaks Spanish"
blends two distinct concepts (Java + Spanish). Items strong on one axis
get crowded out by items related to the other. By running independent
vector searches for each axis and merging the results, we guarantee
coverage of every dimension the user mentioned.

This module exposes one function:

  decompose_query(conversation_text) -> List[str]
      Returns a list of 1-4 sub-queries. Always returns at least one
      element (the original conversation text) on failure, so the
      caller can rely on getting something usable.

The decomposition uses a focused Gemini call with low max_output_tokens
to keep latency under ~500ms.
"""

import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from shl_agent.query_decomposition.constants import *


load_dotenv()


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


if GEMINI_API_KEY:
    _client = genai.Client(api_key=GEMINI_API_KEY)
else:
    _client = None


# ============================================================================
# The decomposition prompt
# ============================================================================
# Kept very small and focused. The LLM's only job is to identify distinct
# semantic axes in the user's request and output them as separate query
# strings suitable for vector search against an assessment catalog.

DECOMPOSE_PROMPT = """\
You decompose a hiring manager's conversation into 1-4 independent
SEARCH QUERIES that will be sent to a vector database of SHL assessments.

Each query you produce will be embedded separately and used to retrieve
its own list of candidate assessments. Then the results will be merged.

Your job:
  - Identify the distinct DIMENSIONS the user has mentioned
  - Produce ONE focused query per dimension
  - Make each query specific enough that vector search returns the
    right kind of assessment

DIMENSIONS often look like:
  - A specific technology (Java, Python, SQL, .NET, Excel, Salesforce)
  - A specific language (English, Spanish, Portuguese, German)
  - A role/seniority pattern (entry-level developer, executive leader)
  - A behavioral or personality trait (leadership, communication, integrity)
  - A test type (cognitive aptitude, situational judgment, simulation)

RULES:
  1. Output ONLY a JSON array of strings. No prose, no markdown.
  2. Minimum 1 query, maximum 4 queries.
  3. If the request has only ONE dimension, output a single-element array.
  4. Each query should be a short, search-friendly sentence (5-15 words).
  5. Do NOT include conversational filler ("the user wants...").
  6. Do NOT invent dimensions the user did not mention.
  7. MULTI-TURN AWARENESS: If the conversation has prior turns, the
     dimensions ACCUMULATE across turns. The most recent user message
     adds to or modifies the previous context — it does NOT replace it.
     When the user says "also add X", "include something for Y", "drop Z",
     or similar refinement language, output sub-queries covering BOTH
     the original axes (from earlier turns) AND the new request.
     The vector retrieval needs the full picture, not just the latest ask.

EXAMPLES:

Conversation:
  USER: "Hiring a Java developer who speaks Spanish."

Output:
["Java programming knowledge assessment for software developers", "Spanish language proficiency test for workplace communication"]

Conversation:
  USER: "We need an assessment for mid-level Java backend engineers focused on Spring and SQL."

Output:
["Java backend developer technical assessment", "Spring framework knowledge test", "SQL database querying skills assessment"]

Conversation:
  USER: "Looking for personality assessment for executive leadership selection."

Output:
["Executive leadership personality assessment for selection"]

Conversation:
  USER: "We need a customer service simulation in English, and a personality test for entry-level reps."

Output:
["English customer service phone simulation assessment", "Personality assessment for entry-level customer service representatives"]

Conversation:
  USER: "Screening 500 graduates for finance analyst roles. We need numerical reasoning and a finance knowledge test, plus situational judgment."

Output:
["Numerical reasoning cognitive aptitude assessment for graduates", "Financial accounting and finance knowledge test for analysts", "Situational judgment test for graduate work-context decisions"]

MULTI-TURN EXAMPLES — REFINE / ADD / DROP

Conversation:
  USER: "Hiring graduate financial analysts. Need numerical reasoning and finance knowledge."
  ASSISTANT: "Here are 4 items: Numerical Reasoning, Financial Accounting, Basic Statistics, OPQ32r."
  USER: "Also add a situational judgement element for graduates."

Output:
["Numerical reasoning cognitive aptitude assessment for graduates", "Financial accounting and finance knowledge test for analysts", "Personality assessment for graduate analysts", "Situational judgment test for graduate work-context decisions"]

Conversation:
  USER: "Hiring a Java developer who speaks Spanish."
  ASSISTANT: "Here is a spread covering Core Java, Java 8, Spanish, and OPQ."
  USER: "Add something for SQL and drop the personality test."

Output:
["Java programming knowledge assessment for software developers", "Spanish language proficiency test for workplace communication", "SQL database querying knowledge test"]

Conversation:
  USER: "Need an assessment for mid-level Java backend engineers."
  ASSISTANT: "Here are 5 Java technical items."
  USER: "Also include something for Spring and a cognitive aptitude check."

Output:
["Java backend developer technical assessment", "Spring framework knowledge test", "Cognitive aptitude reasoning test for mid-level engineers"]

Note in each multi-turn example above: the sub-queries cover the
ORIGINAL dimensions from the first turn PLUS the new request. The
catalog retrieval needs the full picture so the agent can build an
updated shortlist that keeps the items already in play.

Now decompose the following conversation. Output ONLY the JSON array.
"""


# ============================================================================
# Decompose function
# ============================================================================

def decompose_query(conversation_text: str) -> List[str]:
    """
    Decompose a conversation into 1-4 sub-queries for multi-query retrieval.

    Always returns at least one usable query. On any failure (API down,
    bad JSON, empty result), falls back to returning [conversation_text]
    so the caller can still do a single-query search.
    """
    if not conversation_text.strip():
        return [conversation_text]

    if _client is None:
        logger.warning("Decomposer: GEMINI_API_KEY not set, skipping decomposition")
        return [conversation_text]

    sub_queries = _call_decomposer(conversation_text)

    if not sub_queries:
        logger.warning("Decomposer: empty result, falling back to single query")
        return [conversation_text]

    sub_queries = sub_queries[:MAX_SUB_QUERIES]

    logger.info(
        "Query decomposed into %d sub-queries: %s",
        len(sub_queries),
        sub_queries,
    )
    return sub_queries


def _model_supports_thinking(model_name: str) -> bool:
    """
    Return True if the model has an internal-thinking mode whose budget
    we want to set to zero.

    Only Gemini 2.5 Pro / Flash (the full versions) have thinking mode.
    Flash-Lite, 1.5 Flash, and 1.5 Flash-8b do not — passing thinking_config
    to them either fails or is silently ignored, so we skip it.
    """
    if not model_name:
        return False
    lowered = model_name.lower()
    if "lite" in lowered or "8b" in lowered:
        return False
    if lowered.startswith("gemini-1.5"):
        return False
    return True


def _call_decomposer(conversation_text: str) -> Optional[List[str]]:
    """
    Make the decomposition LLM call. Returns the sub-query list on success,
    None on any failure.
    """
    user_message = (
        f"Conversation:\n{conversation_text}\n\n"
        "Output the JSON array of sub-queries now."
    )

    config_kwargs = {
        "system_instruction": DECOMPOSE_PROMPT,
        "response_mime_type": "application/json",
        "temperature": 0.2,
        "max_output_tokens": 2048,
    }

    if _model_supports_thinking(DECOMPOSE_MODEL):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    config = types.GenerateContentConfig(**config_kwargs)

    try:
        result = _client.models.generate_content(
            model=DECOMPOSE_MODEL,
            contents=user_message,
            config=config,
        )
    except Exception as exc:
        logger.exception("Decomposer API call failed: %s", exc)
        return None

    raw_text = getattr(result, "text", None)
    if not raw_text:
        logger.error("Decomposer returned empty response")
        return None

    return _parse_decomposer_output(raw_text)


def _parse_decomposer_output(raw_text: str) -> Optional[List[str]]:
    """Parse the JSON array, with light cleanup for stray markdown fences."""
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
        logger.warning(
            "Decomposer output not valid JSON, attempting tolerant parse: %s",
            exc,
        )
        data = _tolerant_array_parse(cleaned)
        if not data:
            logger.error("Decomposer tolerant parse also failed. Raw: %s", raw_text[:300])
            return None

    if not isinstance(data, list):
        logger.error("Decomposer output is not a JSON array: %s", type(data).__name__)
        return None

    sub_queries = [str(q).strip() for q in data if str(q).strip()]
    return sub_queries or None


def _tolerant_array_parse(text: str) -> Optional[List[str]]:
    """
    Recover whatever complete strings can be extracted from a truncated
    JSON array. Useful when the response was cut off mid-string.

    Strategy: find every fully-quoted string in the text. This catches
    completed elements even if the array is missing its closing ].
    """
    import re
    matches = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
    items = [m for m in matches if m.strip()]
    if items:
        logger.info("Tolerant parse recovered %d strings from truncated output", len(items))
    return items if items else None