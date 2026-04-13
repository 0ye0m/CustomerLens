from __future__ import annotations

import os
import time
from typing import Optional

import streamlit as st
from groq import AuthenticationError, Groq, RateLimitError

DEFAULT_MODEL = "llama-3.1-8b-instant"
SMART_MODEL = "mixtral-8x7b-32768"
AVAILABLE_MODELS = [DEFAULT_MODEL, SMART_MODEL]


def _get_api_key() -> str | None:
    """Resolve the Groq API key from Streamlit secrets or environment."""
    try:
        key = st.secrets["groq"]["api_key"]
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GROQ_API_KEY")
    return key


def _create_client() -> Optional[Groq]:
    """Create a Groq client or show an error if no key is available."""
    api_key = _get_api_key()
    if not api_key:
        st.error("\U0001F511 Invalid API Key. Check your secrets.toml")
        return None
    return Groq(api_key=api_key)


def estimate_tokens(*texts: str) -> int:
    """Roughly estimate token usage from character count."""
    char_count = sum(len(text) for text in texts if text)
    return int(char_count / 4)


def _track_tokens(prompt: str, system: str, response: str) -> None:
    """Track approximate token usage in the current session."""
    tokens = estimate_tokens(prompt, system, response)
    st.session_state.groq_tokens_used = st.session_state.get("groq_tokens_used", 0) + tokens


def get_selected_model() -> str:
    """Return the active Groq model from session state."""
    return st.session_state.get("groq_model", DEFAULT_MODEL)


def _fallback_model(current: str) -> str | None:
    """Return a fallback model when the current one is unavailable."""
    for model in AVAILABLE_MODELS:
        if model != current:
            return model
    return None


def _render_error(message: str, exc: Exception) -> None:
    """Show a friendly error with expandable details."""
    st.error(message)
    with st.expander("Show error details"):
        st.code(repr(exc))


def ask_groq(
    prompt: str,
    system: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Send a prompt to Groq and return response text."""
    client = _create_client()
    if client is None:
        return ""

    model = get_selected_model()
    system_prompt = f"{system.strip()} Be concise. Max 400 words."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    def _run_request(model_name: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()
        _track_tokens(prompt, system_prompt, content)
        return content

    try:
        content = _run_request(model)
        time.sleep(1)
        return content
    except RateLimitError as exc:
        st.warning("\U000023F3 Rate limit reached. Wait 10 seconds.")
        time.sleep(10)
        try:
            content = _run_request(model)
            time.sleep(1)
            return content
        except Exception as retry_exc:
            _render_error("Groq request failed after retry.", retry_exc)
    except AuthenticationError as exc:
        _render_error("\U0001F511 Invalid API Key. Check your secrets.toml", exc)
    except Exception as exc:
        if "model_decommissioned" in str(exc):
            fallback = _fallback_model(model)
            if fallback:
                st.warning(f"Model {model} retired. Switching to {fallback}.")
                st.session_state.groq_model = fallback
                try:
                    content = _run_request(fallback)
                    time.sleep(1)
                    return content
                except Exception as fallback_exc:
                    _render_error("Groq request failed after fallback.", fallback_exc)
            else:
                _render_error("Groq request failed. No fallback model available.", exc)
        else:
            _render_error("Groq request failed.", exc)

    return ""


def ping_groq() -> bool:
    """Send a tiny request to validate connectivity."""
    response = ask_groq("Reply with pong.", "Respond with a single word.", temperature=0.2)
    return bool(response)


def render_groq_sidebar() -> None:
    """Render the Groq status widget in the sidebar."""
    st.sidebar.markdown("### \U0001F916 AI Engine")

    def _label(model: str) -> str:
        if model == DEFAULT_MODEL:
            return "llama-3.1-8b-instant (Fast, Free)"
        return "mixtral-8x7b-32768 (Smarter, Free)"

    if "groq_model" not in st.session_state:
        st.session_state.groq_model = DEFAULT_MODEL

    st.sidebar.radio(
        "Model",
        options=AVAILABLE_MODELS,
        key="groq_model",
        format_func=_label,
    )

    if "groq_status_checked" not in st.session_state:
        st.session_state.groq_status = ping_groq()
        st.session_state.groq_status_checked = True

    status = st.session_state.get("groq_status", False)
    status_text = "\U0001F7E2 Connected" if status else "\U0001F534 Check API Key"
    st.sidebar.caption(status_text)

    st.sidebar.caption("Free tier: 30 requests/min")

    tokens_used = st.session_state.get("groq_tokens_used", 0)
    st.sidebar.caption(f"~{tokens_used:,} tokens used this session")
