"""
validation.py
==============
Inbound and outbound validation for the /chat endpoint.

Two distinct jobs:

INBOUND — validate_request(request)
  Rejects malformed requests before we burn a Gemini call.
  Raises HTTPException with a 400 status on failure.

OUTBOUND — sanitize_response(response, history)
  Cleans up Gemini's output before returning to the SHL evaluator.
  Strips hallucinated URLs, caps recommendations at 10, and falls
  back gracefully if everything gets stripped.

Inbound failures REJECT (we never reached Gemini, so a 400 is fine).
Outbound failures REPAIR (we'd rather return something useful than fail).
"""

import logging
from typing import List

from fastapi import HTTPException

from shl_agent.model_schemas.models import (
    ChatRequest,
    ChatResponse,
    Recommendation,
    Message,
    KEY_TO_TEST_TYPE,
)
from shl_agent.data_loader.catalog_loader import is_valid_url, get_item_by_url 

from shl_agent.validate_schema.constants import * 


logger = logging.getLogger(__name__)





VALID_TEST_TYPE_LETTERS = set(KEY_TO_TEST_TYPE.values())


# ============================================================================
# Inbound validation
# ============================================================================

def validate_request(request: ChatRequest) -> None:
    """
    Validate an incoming /chat request. Raises HTTPException(400) on failure.

    Checks:
      - Conversation history is non-empty
      - Last message is from the user
      - Turn count is within the 8-turn budget
      - No empty user content
    """
    history = request.messages

    if not history:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty",
        )

    if history[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="The last message in messages must be from the user",
        )

    if not history[-1].content.strip():
        raise HTTPException(
            status_code=400,
            detail="The last user message has empty content",
        )

    user_turns = sum(1 for m in history if m.role == "user")
    if user_turns > MAX_TURNS:
        raise HTTPException(
            status_code=400,
            detail=f"Conversation exceeded the {MAX_TURNS}-turn budget "
                   f"(received {user_turns} user turns)",
        )


# ============================================================================
# Outbound validation / sanitization
# ============================================================================

def sanitize_response(
    response: ChatResponse,
    history: List[Message],
) -> ChatResponse:
    """
    Clean up Gemini's response before returning it.

    Steps:
      1. If recommendations is a list, strip out items with invalid URLs.
      2. Cap the list at MAX_RECOMMENDATIONS.
      3. Normalize each recommendation's name and test_type from the
         catalog (truth-from-data, not from Gemini).
      4. If every recommendation got stripped:
         - First time it happens this turn → convert to clarify response
         - Persistent → convert to refusal
      5. If recommendations is null (clarify/compare/refuse), pass through.

    Never raises. Always returns a valid ChatResponse.
    """

    if response.recommendations is None:
        return response

    cleaned: List[Recommendation] = []
    dropped = 0

    for rec in response.recommendations:
        if not is_valid_url(rec.url):
            dropped += 1
            logger.warning(
                "Dropping hallucinated URL from response: %s (name='%s')",
                rec.url, rec.name,
            )
            continue
        cleaned.append(_normalize_from_catalog(rec))

    if dropped > 0:
        logger.info(
            "Sanitized response: kept %d of %d recommendations",
            len(cleaned), len(cleaned) + dropped,
        )

    if len(cleaned) > MAX_RECOMMENDATIONS:
        logger.info(
            "Truncating %d recommendations down to %d",
            len(cleaned), MAX_RECOMMENDATIONS,
        )
        cleaned = cleaned[:MAX_RECOMMENDATIONS]

    if not cleaned:
        return _recover_from_empty(response, history)

    return ChatResponse(
        reply=response.reply,
        recommendations=cleaned,
        end_of_conversation=response.end_of_conversation,
    )


def _normalize_from_catalog(rec: Recommendation) -> Recommendation:
    """
    Replace the Recommendation's name and test_type with values pulled
    directly from the catalog item that owns this URL.

    Gemini occasionally renames items slightly ("Java 8 Test" instead
    of "Java 8 (New)") or picks the wrong test_type letter. Since the
    URL is the canonical identifier, we use it to look up the real
    name and keys, then output the normalized form.
    """
    item = get_item_by_url(rec.url)
    if item is None:
        return rec

    canonical_name = item.get("name") or rec.name
    keys = item.get("keys", [])
    canonical_test_type = ",".join(
        KEY_TO_TEST_TYPE[k] for k in keys if k in KEY_TO_TEST_TYPE
    )

    if not canonical_test_type:
        canonical_test_type = _filter_test_type_letters(rec.test_type)

    return Recommendation(
        name=canonical_name,
        url=rec.url,
        test_type=canonical_test_type,
    )


def _filter_test_type_letters(value: str) -> str:
    """
    Keep only valid test_type letters from a string.
    Used as a fallback when the catalog item has no keys (rare).
    """
    if not value:
        return ""
    letters = [c.strip() for c in value.split(",")]
    valid = [c for c in letters if c in VALID_TEST_TYPE_LETTERS]
    return ",".join(valid)


def _recover_from_empty(
    response: ChatResponse,
    history: List[Message],
) -> ChatResponse:
    """
    Recovery path when every recommendation was stripped (all URLs fake).

    First failure in a conversation → convert to a clarify response asking
    for more context.
    Persistent failure (we already asked once) → convert to refusal.

    We detect "already asked" by scanning past assistant messages for
    our previous fallback phrasing.
    """
    already_asked_for_more_context = any(
        m.role == "assistant"
        and "could you share a bit more about the role" in m.content.lower()
        for m in history
    )

    if not already_asked_for_more_context:
        return ChatResponse(
            reply=(
                "I couldn't pin down a clean match for that. "
                "Could you share a bit more about the role — the level of seniority, "
                "the primary skill or technology, and whether this is for selection "
                "or development?"
            ),
            recommendations=None,
            end_of_conversation=False,
        )

    return ChatResponse(
        reply=(
            "I'm not finding suitable assessments in the SHL catalog for what "
            "you've described. It may fall outside our coverage — happy to "
            "help if you'd like to explore a different role."
        ),
        recommendations=None,
        end_of_conversation=False,
    )


# ============================================================================
# Diagnostic helper
# ============================================================================

def validation_info() -> dict:
    """Return validation configuration for diagnostics."""
    return {
        "max_turns": MAX_TURNS,
        "max_recommendations": MAX_RECOMMENDATIONS,
        "min_recommendations": MIN_RECOMMENDATIONS,
        "valid_test_type_letters": sorted(VALID_TEST_TYPE_LETTERS),
    }
