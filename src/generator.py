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
You are the Coforge HR Companion, a warm, helpful, and trustworthy HR policy assistant.

Your job is to answer employee questions using only the supplied policy context.
Do not use outside knowledge or assumptions.
Do not invent policy rules, eligibility, procedures, or exceptions.
If the context does not support an answer, say clearly that the information was not found in the available policy documents.

Gold-standard response behavior:
- Start with a clear answer to the user's question.
- Keep the tone warm, professional, and conversational.
- Use simple, natural language that feels human and reassuring.
- Be direct, but not robotic.
- Keep the answer concise but complete.
- Preserve important policy conditions, eligibility requirements, procedures, exceptions, and limitations.
- If multiple parts are asked, answer each supported part clearly.
- If something is not covered, say so plainly without guessing.

Formatting guidance:
- Prefer a short paragraph first, then bullets only when they make the answer easier to scan.
- No citations, no page numbers, no source names, no "Sources/References" section in the answer body.
- The app shows sources separately below the chat.

User question: {question}

Relevant policy context:
{context_text}

Write the final answer in a friendly HR-support voice, with calm confidence and practical clarity.
"""
    return remove_source_section(_call_gemini_with_fallback(api_key, prompt))


def extract_text_from_gemini_response(response: Any) -> str:
    """Extract only actual text from Gemini responses while ignoring metadata like thought_signature."""
    if response is None:
        return ""

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            extracted = "".join(
                part.text for part in parts if getattr(part, "text", None)
            )
            if extracted.strip():
                return extracted

    content = getattr(response, "content", None)
    if content is not None:
        parts = getattr(content, "parts", None) or []
        extracted = "".join(part.text for part in parts if getattr(part, "text", None))
        if extracted.strip():
            return extracted

    output_text = getattr(response, "output_text", None) or getattr(response, "outputText", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    return str(response)


def _generate_gemini_once(api_key: str, prompt: str, model_name: str) -> str:
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        if hasattr(client, "interactions"):
            response = client.interactions.create(
                model=model_name,
                input=prompt,
            )
            return extract_text_from_gemini_response(response)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return extract_text_from_gemini_response(response)

    except ImportError:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return extract_text_from_gemini_response(response)


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
