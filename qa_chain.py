"""
Turns retrieved policy chunks + a user question into a grounded answer.

The prompt is written so the model only ever answers from the given
context - this is what satisfies "response should be generated only
from the relevant policy content".
"""

import time
import config

SYSTEM_PROMPT = (
    "You are an HR assistant for Coforge employees in India. "
    "Answer the employee's question using ONLY the policy excerpts "
    "provided below. Do not use outside knowledge, and do not guess. "
    "If the excerpts don't contain the answer, say clearly that the "
    "policy documents don't cover this and suggest the employee check "
    "with HR. Keep answers concise and cite which policy the answer "
    "comes from."
)


def build_prompt(query: str, chunks) -> str:
    if not chunks:
        context = "(No relevant policy content was found.)"
    else:
        context = "\n\n".join(
            f"[Source: {c.metadata.get('policy_name', 'Unknown Policy')}]\n{c.page_content}"
            for c in chunks
        )

    return (
        f"Policy excerpts:\n{context}\n\n"
        f"Employee question: {query}\n\n"
        f"Answer based only on the excerpts above."
    )


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config={"system_instruction": SYSTEM_PROMPT, "max_output_tokens": 600},
            )
            return response.text
        except errors.ServerError as e:
            if e.status_code == 503 and attempt < max_retries - 1:
                # Exponential backoff: 2s, 4s, 8s
                delay = base_delay * (2 ** attempt)
                print(f"⏳ Model busy (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except Exception as e:
            if ("overloaded" in str(e).lower() or "unavailable" in str(e).lower()) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⏳ API busy (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
            )
            return response.choices[0].message.content
        except Exception as e:
            if ("overloaded" in str(e).lower() or "unavailable" in str(e).lower()) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⏳ API busy (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def generate_answer(query: str, chunks) -> str:
    prompt = build_prompt(query, chunks)

    if config.LLM_PROVIDER == "gemini":
        return _call_gemini(prompt)
    elif config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")
