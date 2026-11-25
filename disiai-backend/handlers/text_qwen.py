"""Text handler: Qwen 2.5 through an Ollama OpenAI-compatible endpoint."""

import json
import re
from typing import Any

import httpx

from config import settings


def _endpoint() -> str:
    return f"{settings.ollama_base_url.rstrip('/')}/chat/completions"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.ollama_api_key}",
        "Content-Type": "application/json",
    }


def is_available(timeout: float = 2.0) -> bool:
    """Cheap readiness probe used by /health and the extractor fallback."""
    url = f"{settings.ollama_base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, headers=_headers(), timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    payload = {
        "model": model or settings.ollama_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        response = httpx.post(
            _endpoint(),
            headers=_headers(),
            json=payload,
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Qwen request failed: {exc}") from exc


def generate_json(prompt: str, *, model: str | None = None) -> dict[str, Any]:
    """Ask the model for a single JSON object; strip optional markdown fences."""
    raw = chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "Kamu adalah ekstraktor konteks. Balas HANYA dengan satu objek JSON "
                    "yang valid, tanpa penjelasan dan tanpa markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=model or settings.extractor_model,
        temperature=0.0,
        max_tokens=512,
    )
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model did not return a JSON object")
    return parsed
