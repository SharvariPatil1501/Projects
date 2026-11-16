"""LLM wrapper: Gemini with retry/backoff, model fallback chain, optional Groq fallback."""
import json
import os
import re
import time

from config import (FALLBACK_MODELS, GEMINI_THINKING_BUDGET, GROQ_MODEL,
                    LLM_ATTEMPTS_PER_MODEL, LLM_TIMEOUT_S, PRIMARY_MODEL)


class LLMUnavailable(Exception):
    """Raised when every model/provider in the chain failed."""


_gemini_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        from google.genai import types
        _gemini_client = genai.Client(  # reads GEMINI_API_KEY
            http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000))
    return _gemini_client


def _call_gemini(model: str, prompt: str, json_mode: bool) -> str:
    from google.genai import types
    kwargs = {"temperature": 0}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if GEMINI_THINKING_BUDGET is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(GEMINI_THINKING_BUDGET))
    resp = _gemini().models.generate_content(
        model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs)
    )
    return resp.text or ""


def _call_groq(model: str, prompt: str, json_mode: bool) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=LLM_TIMEOUT_S)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}], **kwargs
    )
    return resp.choices[0].message.content or ""


def _chain():
    chain = [("gemini", m) for m in [PRIMARY_MODEL, *FALLBACK_MODELS]]
    if os.getenv("GROQ_API_KEY"):
        chain.append(("groq", GROQ_MODEL))
    return chain


def generate(prompt: str, json_mode: bool = False) -> str:
    errors = []
    for provider, model in _chain():
        call = _call_gemini if provider == "gemini" else _call_groq
        for attempt in range(LLM_ATTEMPTS_PER_MODEL):
            try:
                return call(model, prompt, json_mode)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                errors.append(f"{provider}:{model}: {msg[:120]}")
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # wrong/retired model name: don't retry, go to next model
                time.sleep(2 ** attempt)  # 1s, 2s ...
    raise LLMUnavailable("All models failed. Last errors: " + " | ".join(errors[-3:]))


def parse_json(text: str):
    """Parse JSON even if the model wrapped it in code fences or added text."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
