import os
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional

import config


NOT_FOUND_MESSAGE = "I couldn't find information about this in the available Coforge India policy documents."
GEMINI_FAILURE_MESSAGE = (
    "I'm having trouble reaching the AI service right now. Please try again in a moment."
)
TRANSIENT_STATUS_CODES = {408, 429, 500, 503}
logger = logging.getLogger(__name__)


SOURCE_SECTION_RE = re.compile(
    r"(?ims)^\s*(?:#+\s*)?(?:sources?|references?|citations?)\s*:?\s*$.*\Z"
)


def remove_source_section(answer: str) -> str:
    """Remove source/citation sections from chat text; UI renders sources separately."""
    return SOURCE_SECTION_RE.sub("", answer).strip()


def _retry_count() -> int:
    return max(0, min(int(getattr(config, "GEMINI_RETRY_COUNT", 2)), 3))


def _backoff_seconds(attempt: int) -> float:
    base = max(0.1, float(getattr(config, "GEMINI_BACKOFF_SECONDS", 1.0)))
    cap = max(base, float(getattr(config, "GEMINI_BACKOFF_MAX_SECONDS", 8.0)))
    delay = min(cap, base * (2 ** max(attempt - 1, 0)))
    return delay + random.uniform(0, delay * 0.25)


def _error_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"\b(408|429|500|503)\b", str(exc))
    return int(match.group(1)) if match else None


def _is_transient_error(exc: Exception) -> bool:
    return _error_status_code(exc) in TRANSIENT_STATUS_CODES


def generate_answer(question: str, context_chunks: List[Dict[str, Any]], conversation_state: Optional[Dict[str, Any]] = None) -> str:
    api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "The Gemini API key is not configured. Set GEMINI_API_KEY in your environment or .env before running the app."

    context_text = "\n\n---\n\n".join(
        f"Policy: {chunk.get('policy_name', 'Unknown')}\nSection: {chunk.get('section', 'General')}\nPage: {chunk.get('page', 'N/A')}\nText: {chunk.get('text', '')}"
        for chunk in context_chunks
    )

    prompt = f"""
You are an HR Policy Assistant.

Answer only from the supplied policy context.
Do not use outside knowledge.
Do not invent or infer unsupported policy rules.
Give a direct, straightforward answer to the user's question first.
Keep the response concise and avoid restating unnecessary background.
If the question contains multiple parts, answer every part that can be supported by the supplied policy content.
Preserve important conditions, exceptions, eligibility requirements, procedures, and limitations.
If the supplied policy context does not contain enough information to answer a part of the question, explicitly say that the information was not found in the available policy documents.
Do not fabricate an answer.
Do not include citations, source names, section names, page numbers, or a Sources/References section in the answer text.
The app displays sources separately below the chat.

User question: {question}

Relevant policy context:
{context_text}

Respond in a natural HR-policy style. Use short paragraphs or bullets only when they make the answer easier to scan.
"""
    return remove_source_section(_call_gemini_with_fallback(api_key, prompt))


def _generate_gemini_once(api_key: str, prompt: str, model_name: str) -> str:
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        if hasattr(client, "interactions"):
            response = client.interactions.create(
                model=model_name,
                input=prompt,
            )
            return (
                getattr(response, "output_text", None)
                or getattr(response, "outputText", None)
                or getattr(response, "text", None)
                or str(response)
            )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return getattr(response, "text", str(response))

    except ImportError:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return getattr(response, "text", str(response))


def _call_model_with_retries(api_key: str, prompt: str, model_name: str, role: str) -> str:
    max_attempts = _retry_count() + 1
    for attempt in range(1, max_attempts + 1):
        try:
            return _generate_gemini_once(api_key, prompt, model_name)
        except Exception as exc:
            status_code = _error_status_code(exc)
            transient = _is_transient_error(exc)
            should_retry = transient and attempt < max_attempts
            logger.warning(
                "Gemini %s model request failed; model=%s attempt=%s/%s status=%s transient=%s retry=%s",
                role,
                model_name,
                attempt,
                max_attempts,
                status_code or "unknown",
                transient,
                should_retry,
            )
            if not should_retry:
                raise
            time.sleep(_backoff_seconds(attempt))
    raise RuntimeError("Gemini retry loop ended unexpectedly")


def _call_gemini_with_fallback(api_key: str, prompt: str) -> str:
    primary_model = config.GEMINI_MODEL
    fallback_model = getattr(config, "GEMINI_FALLBACK_MODEL", primary_model)

    try:
        return _call_model_with_retries(api_key, prompt, primary_model, "primary")
    except Exception:
        if fallback_model == primary_model:
            logger.exception("Gemini primary model failed and no distinct fallback model is configured")
            return GEMINI_FAILURE_MESSAGE

        logger.warning(
            "Switching Gemini request to fallback model; primary_model=%s fallback_model=%s",
            primary_model,
            fallback_model,
        )

    try:
        return _call_model_with_retries(api_key, prompt, fallback_model, "fallback")
    except Exception:
        logger.exception("Gemini fallback model failed")
        return GEMINI_FAILURE_MESSAGE
