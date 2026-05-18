"""
app.py
=======
Streamlit chat interface for the SHL agent.

This app mimics the SHL evaluator's behaviour exactly:
  - Maintains conversation history client-side (in session state)
  - POSTs to /chat over HTTP with {"messages": [...]} body
  - Parses the JSON response and validates it against the SHL schema
  - Renders the reply, recommendations, and end_of_conversation flag
  - Shows raw request/response JSON side-by-side for debugging

Run with:
  streamlit run app.py

Set the FastAPI server URL in the sidebar. By default it points to
http://localhost:8000 — start the FastAPI server first with:
  uvicorn main:app --reload --port 8000
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from constants import *


# ============================================================================
# Page setup
# ============================================================================

URL = LOCAL_HOST

st.set_page_config(
    page_title="SHL Agent — Test Client",
    page_icon="🧪",
    layout="wide",
)


# ============================================================================
# Session state initialisation
# ============================================================================

def _init_state() -> None:
    """Initialise session state on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False
    if "server_url" not in st.session_state:
        st.session_state.server_url = URL


_init_state()


# ============================================================================
# Schema validation — SHL contract
# ============================================================================

EXPECTED_TOP_LEVEL_FIELDS = {"reply", "recommendations", "end_of_conversation"}
EXPECTED_REC_FIELDS = {"name", "url", "test_type"}
VALID_TEST_TYPE_LETTERS = set("ABCDEKPS")


def validate_response_schema(payload: Dict[str, Any]) -> List[str]:
    """
    Return a list of human-readable validation errors.
    Empty list means the response matches SHL's spec.
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        return [f"Response is not a JSON object (got {type(payload).__name__})"]

    extra_fields = set(payload.keys()) - EXPECTED_TOP_LEVEL_FIELDS
    if extra_fields:
        errors.append(f"Unexpected extra fields: {sorted(extra_fields)}")

    missing_fields = EXPECTED_TOP_LEVEL_FIELDS - set(payload.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {sorted(missing_fields)}")

    if "reply" in payload and not isinstance(payload["reply"], str):
        errors.append(f"'reply' must be a string, got {type(payload['reply']).__name__}")

    if "end_of_conversation" in payload and not isinstance(
        payload["end_of_conversation"], bool
    ):
        errors.append(
            f"'end_of_conversation' must be a boolean, got "
            f"{type(payload['end_of_conversation']).__name__}"
        )

    if "recommendations" in payload:
        recs = payload["recommendations"]
        if recs is None:
            pass
        elif not isinstance(recs, list):
            errors.append(
                f"'recommendations' must be a list or null, got {type(recs).__name__}"
            )
        else:
            if len(recs) > 10:
                errors.append(f"'recommendations' has {len(recs)} items, max is 10")
            for i, rec in enumerate(recs):
                if not isinstance(rec, dict):
                    errors.append(f"recommendations[{i}] is not an object")
                    continue
                missing = EXPECTED_REC_FIELDS - set(rec.keys())
                if missing:
                    errors.append(
                        f"recommendations[{i}] missing fields: {sorted(missing)}"
                    )
                extra = set(rec.keys()) - EXPECTED_REC_FIELDS
                if extra:
                    errors.append(
                        f"recommendations[{i}] has unexpected fields: {sorted(extra)}"
                    )
                if "test_type" in rec:
                    tt = rec["test_type"]
                    if not isinstance(tt, str):
                        errors.append(
                            f"recommendations[{i}].test_type must be a string"
                        )
                    else:
                        letters = [c.strip() for c in tt.split(",")]
                        invalid = [c for c in letters if c not in VALID_TEST_TYPE_LETTERS]
                        if invalid:
                            errors.append(
                                f"recommendations[{i}].test_type has invalid "
                                f"letters: {invalid} (valid: A,B,C,D,E,K,P,S)"
                            )

    return errors


# ============================================================================
# HTTP client — mimics SHL evaluator
# ============================================================================

def call_chat_endpoint(
    server_url: str,
    messages: List[Dict[str, str]],
    timeout: int = 30,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """
    POST to /chat with the messages list. Returns:
      (parsed_response, error_message, debug_info)

    parsed_response is the JSON dict on success, None on failure.
    error_message is a human-readable error string on failure, None on success.
    debug_info always contains the request body and raw response (best-effort).
    """
    url = server_url.rstrip("/") + "/chat"
    request_body = {"messages": messages}

    debug: Dict[str, Any] = {
        "request_url": url,
        "request_body": request_body,
        "status_code": None,
        "elapsed_ms": None,
        "raw_response_text": None,
    }

    started = time.monotonic()
    try:
        response = requests.post(url, json=request_body, timeout=timeout)
        debug["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        debug["status_code"] = response.status_code
        debug["raw_response_text"] = response.text

        if response.status_code != 200:
            return (
                None,
                f"HTTP {response.status_code}: {response.text[:300]}",
                debug,
            )

        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            return (
                None,
                f"Response was not valid JSON: {exc}",
                debug,
            )

        return parsed, None, debug

    except requests.exceptions.Timeout:
        debug["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return None, f"Request timed out after {timeout}s", debug
    except requests.exceptions.ConnectionError as exc:
        return None, f"Could not connect to {url}: {exc}", debug
    except Exception as exc:
        return None, f"Unexpected error: {exc}", debug


# ============================================================================
# Sidebar — settings and controls
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")

    st.session_state.server_url = st.text_input(
        "FastAPI server URL",
        value=st.session_state.server_url,
        help="The deployed /chat endpoint will be called over HTTP, mimicking the SHL evaluator.",
    )

    if st.button("🏥 Check /health"):
        try:
            r = requests.get(
                st.session_state.server_url.rstrip("/") + "/health",
                timeout=5,
            )
            if r.status_code == 200:
                st.success(f"✓ Healthy: {r.json()}")
            else:
                st.error(f"HTTP {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Could not reach server: {e}")

    st.divider()
    st.subheader("📊 Conversation Stats")

    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    assistant_turns = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    st.metric("User turns", f"{user_turns}/8")
    st.metric("Assistant turns", assistant_turns)
    st.metric("Conversation ended", "Yes" if st.session_state.conversation_ended else "No")

    st.divider()

    if st.button("🔄 Reset conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.turns = []
        st.session_state.conversation_ended = False
        st.rerun()

    st.divider()
    st.caption(
        "This app mimics the SHL evaluator: it sends `{\"messages\": [...]}` "
        "over HTTP and validates the response against SHL's required schema."
    )


# ============================================================================
# Main layout — chat on left, debug on right
# ============================================================================

st.title("🧪 SHL Agent — Test Client")
st.caption("A Streamlit interface that mimics the SHL evaluator. Send messages, watch the agent respond.")

col_chat, col_debug = st.columns([5, 4])


# ----------------------------------------------------------------------------
# Left column — chat UI
# ----------------------------------------------------------------------------

with col_chat:
    st.subheader("💬 Conversation")

    # Render messages interleaved with their corresponding turn data.
    # Each assistant message corresponds to the turn that produced it,
    # so recommendations and end-of-conversation banners appear right
    # after the assistant's reply rather than dumped at the bottom.
    assistant_msg_idx = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if assistant_msg_idx < len(st.session_state.turns):
                turn = st.session_state.turns[assistant_msg_idx]
                response = turn.get("response") or {}
                recs = response.get("recommendations")
                if recs:
                    with st.chat_message("assistant"):
                        st.caption(f"📋 Recommendations ({len(recs)})")
                        st.dataframe(
                            [
                                {
                                    "Name": r["name"],
                                    "Type": r["test_type"],
                                    "URL": r["url"],
                                }
                                for r in recs
                            ],
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "URL": st.column_config.LinkColumn(),
                            },
                        )
                if response.get("end_of_conversation"):
                    st.info("🏁 Conversation ended (agent set `end_of_conversation: true`)")
            assistant_msg_idx += 1

    if st.session_state.conversation_ended:
        st.warning("Conversation has ended. Click 'Reset conversation' in the sidebar to start over.")
    elif user_turns >= 8:
        st.warning("Reached 8-turn limit. Click 'Reset conversation' to start over.")
    else:
        user_message = st.chat_input("Type a message as the user...")
        if user_message:
            st.session_state.messages.append(
                {"role": "user", "content": user_message}
            )

            with st.spinner("Calling /chat..."):
                response, error, debug = call_chat_endpoint(
                    st.session_state.server_url,
                    st.session_state.messages,
                )

            if error:
                st.session_state.turns.append({
                    "error": error,
                    "debug": debug,
                })
                st.session_state.messages.pop()
                st.error(f"❌ {error}")
            else:
                schema_errors = validate_response_schema(response)

                turn_record = {
                    "request_messages": list(st.session_state.messages),
                    "response": response,
                    "schema_errors": schema_errors,
                    "debug": debug,
                }
                st.session_state.turns.append(turn_record)

                reply = response.get("reply", "")
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )

                if response.get("end_of_conversation"):
                    st.session_state.conversation_ended = True

            st.rerun()


# ----------------------------------------------------------------------------
# Right column — debug panel
# ----------------------------------------------------------------------------

with col_debug:
    st.subheader("🔍 Debug")

    if not st.session_state.turns:
        st.info("Send a message to see request/response details here.")
    else:
        latest = st.session_state.turns[-1]

        if "error" in latest:
            st.error(f"Last call failed: {latest['error']}")
            with st.expander("Debug info", expanded=True):
                st.json(latest["debug"])
        else:
            debug = latest["debug"]
            schema_errors = latest["schema_errors"]

            cols = st.columns(2)
            cols[0].metric("HTTP Status", debug["status_code"])
            cols[1].metric("Latency", f"{debug['elapsed_ms']}ms")

            if schema_errors:
                st.error("⚠️ Schema validation failed:")
                for err in schema_errors:
                    st.markdown(f"- `{err}`")
            else:
                st.success("✓ Response matches SHL schema")

            tab_resp, tab_req, tab_raw = st.tabs(["Parsed Response", "Request Body", "Raw Response"])

            with tab_resp:
                st.json(latest["response"])

            with tab_req:
                st.json({"messages": latest["request_messages"]})

            with tab_raw:
                st.code(debug["raw_response_text"] or "", language="json")

        if len(st.session_state.turns) > 1:
            with st.expander(f"Previous turns ({len(st.session_state.turns) - 1})"):
                for i, t in enumerate(reversed(st.session_state.turns[:-1])):
                    idx = len(st.session_state.turns) - 1 - i
                    st.markdown(f"**Turn {idx}**")
                    if "error" in t:
                        st.error(t["error"])
                    else:
                        errs = t.get("schema_errors", [])
                        if errs:
                            st.warning(f"{len(errs)} schema errors")
                        else:
                            st.success("Schema OK")
                        st.json(t["response"], expanded=False)
                    st.divider()