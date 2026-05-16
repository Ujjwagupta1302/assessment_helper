"""
models.py
==========
Pydantic models and constants for the SHL agent.

Defines majorly one schema layers:

1. SHL contract — ChatRequest and ChatResponse, the exact format the
   /chat endpoint accepts and returns. Non-negotiable.

Also defines the canonical mapping from catalog 'keys' values to the
single-letter test_type codes SHL expects in recommendations, and the
intent classification that drives the agent's behaviour each turn.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Test-type code mapping
# ============================================================================
# SHL's response schema uses single-letter codes for test_type. When an item
# has multiple keys, the codes are comma-separated in priority order
# (e.g., "P,C" for Personality & Behavior + Competencies, "K,S" for
# Knowledge & Skills + Simulations). Confirmed from the public traces.
KEY_TO_TEST_TYPE = {
    "Ability & Aptitude":              "A",
    "Assessment Exercises":            "E",
    "Biodata & Situational Judgment":  "B",
    "Competencies":                    "C",
    "Development & 360":               "D",
    "Knowledge & Skills":              "K",
    "Personality & Behavior":          "P",
    "Simulations":                     "S",
}

TEST_TYPE_TO_KEY = {v: k for k, v in KEY_TO_TEST_TYPE.items()} # assigning the shortword to each key


def keys_to_test_type(keys: List[str]) -> str:
    """
    Convert a list of catalog `keys` values to a comma-separated test_type string.

    The traces render multi-key items as comma-separated letters in the
    test_type column ("K,S", "P,C"). We preserve the order of keys as they
    appear in the catalog data.
    """
    codes = [KEY_TO_TEST_TYPE[k] for k in keys if k in KEY_TO_TEST_TYPE]
    return ",".join(codes)


# ============================================================================
# Intent enum — drives agent behaviour
# ============================================================================
# Derived from analysing all 10 public traces. Each turn the agent does
# exactly one of these:
#
#   clarify   - user query is too vague; ask one focused question
#   recommend - new shortlist of 1-10 items
#   refine    - update existing shortlist (add, remove, swap items)
#   compare   - explain difference between specific items, no list change
#   confirm   - user signalled satisfaction; re-display final list, end
#   refuse    - off-topic / out-of-scope; refuse politely, no list

Intent = Literal[
    "clarify",
    "recommend",
    "refine",
    "compare",
    "confirm",
    "refuse",
]


# ============================================================================
# SHL contract — request and response schemas
# ============================================================================

class Message(BaseModel):
    """A single message in the conversation history."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """
    Input to POST /chat — the full conversation history each turn.

    """
    messages: List[Message] = Field(
        ..., description="Full conversation so far, ending with a user message."
    )


class Recommendation(BaseModel):
    """A single assessment recommendation in the response."""
    name: str
    url: str
    test_type: str = Field(
        ...,
        description="Single-letter code or comma-separated for multi-key (e.g., 'K', 'P,C')."
    )


class ChatResponse(BaseModel):
    """
    Output from POST /chat — the SHL-required response schema.

    Per the public traces, `recommendations` is `null` when the agent is
    clarifying, comparing, or refusing — NOT an empty list. It is a list
    only when the agent is producing/refining/confirming a shortlist.
    """
    reply: str
    recommendations: Optional[List[Recommendation]] = None
    end_of_conversation: bool = False


# ============================================================================
# Internal profile schema (Gemini Call 1 output)
# ============================================================================

'''
In my initial architecture I planned a 2 gmeini call architecture which is scrapped for now. 
So the profile class and the Selection Output class is of no use for now. 
'''
class Profile(BaseModel):
    """
    Structured profile extracted by Gemini Call 1.

    Contains parsed user requirements plus Gemini's judgement about intent
    (what behaviour to execute this turn) and what to ask if clarifying.
    """

    # Parsed requirements
    hard_keywords: List[str] = Field(
        default_factory=list,
        description="Specific technologies, roles, tools, or domains."
    )
    soft_keywords: List[str] = Field(
        default_factory=list,
        description="Behavioural traits, abstract qualities, competencies."
    )
    seniority: Optional[str] = Field(
        None,
        description="One of catalog's job_levels values, or null."
    )
    purpose: Optional[Literal["selection", "development", "screening"]] = Field(
        None,
        description="Why the assessment is being run."
    )
    test_focus: List[str] = Field(
        default_factory=list,
        description="One or more catalog 'keys' values."
    )
    language: Optional[str] = None
    max_duration: Optional[int] = Field(
        None,
        description="Maximum acceptable test duration in minutes."
    )
    remote_required: Optional[bool] = None

    search_query: str = Field(
        "",
        description="A natural-language query summarising the user's need."
    )

    # Intent classification (drives everything)
    intent: Intent = Field(
        ...,
        description="What behaviour the agent should execute this turn."
    )

    # Used when intent == 'clarify'
    critical_missing_field: Optional[str] = Field(
        None,
        description="If clarifying, which single field most needs clarification."
    )
    suggested_question: Optional[str] = Field(
        None,
        description="If clarifying, the natural-language question to ask."
    )

    # Used when intent == 'compare'
    items_to_compare: List[str] = Field(
        default_factory=list,
        description="Names of items the user wants compared."
    )

    # Bookkeeping
    topics_already_asked: List[str] = Field(
        default_factory=list,
        description="Field names that have been raised in past turns."
    )
    current_shortlist_urls: List[str] = Field(
        default_factory=list,
        description="URLs from the most recent agent shortlist in the history."
    )


# ============================================================================
# Selection output schema (Gemini Call 2 output)
# ============================================================================

class SelectionOutput(BaseModel):
    """Raw output from Gemini Call 2."""

    reply: str = Field(..., description="Conversational message to the user.")
    selected_urls: List[str] = Field(
        default_factory=list,
        description="URLs of chosen items. Empty for compare/refuse/clarify."
    )
    include_recommendations: bool = Field(
        True,
        description="If false, final response has recommendations=null."
    )
    end_of_conversation: bool = False


# ============================================================================
# Convenience constructors
# ============================================================================

'''
These convinience constructors are also not at all needed for the current architecture. 
Now we have many other internts rather than just clarify, refusal and compare. 

'''

def make_clarify_response(question: str) -> ChatResponse:
    """ChatResponse for clarifying — recommendations is null."""
    return ChatResponse(
        reply=question,
        recommendations=None,
        end_of_conversation=False,
    )


def make_compare_response(explanation: str) -> ChatResponse:
    """ChatResponse for comparison/general-question — recommendations is null."""
    return ChatResponse(
        reply=explanation,
        recommendations=None,
        end_of_conversation=False,
    )


def make_refusal_response(reason: str = None) -> ChatResponse:
    """ChatResponse for off-topic/out-of-scope — recommendations is null."""
    default = (
        "That's outside what I can help with — I can recommend SHL assessments "
        "for hiring or development scenarios. Could you tell me about the role?"
    )
    return ChatResponse(
        reply=reason or default,
        recommendations=None,
        end_of_conversation=False,
    )
