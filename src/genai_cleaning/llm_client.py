from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Optional, Type, TypeVar

import httpx
from dotenv import load_dotenv

T = TypeVar("T")


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from model output.

    Ollama models occasionally wrap JSON in code fences or prepend/append commentary.
    This function strips common wrappers and extracts the first top-level JSON object.
    """

    s = text.strip()

    # Strip fenced blocks: ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```\s*$", "", s).strip()

    # Fast path: already valid JSON
    if s.startswith("{") and s.endswith("}"):
        return s

    # Heuristic: take substring from first '{' to last '}'
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()

    return s


class LLMClient:
    """Small wrapper around a local Ollama server.

    - Uses Ollama's chat endpoint (http://localhost:11434/api/chat).
    - For "structured outputs", it instructs the model to emit JSON matching a
      Pydantic JSON Schema and then validates via Pydantic.
    - Implements simple exponential backoff for transient failures.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 120.0,
    ):
        load_dotenv()
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "qwen2.5:7b"
        self.timeout_s = timeout_s

    def parse(
        self,
        model: str,
        system: str,
        user: str,
        schema_model: Type[T],
        max_retries: int = 6,
        min_backoff: float = 0.5,
        max_backoff: float = 8.0,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> T:
        """Generate a JSON response and validate it against a Pydantic model.

        Parameters
        ----------
        model:
            Kept for backward compatibility with the OpenAI-based codebase.
            If provided, it overrides the instance default Ollama model.
        system, user:
            Prompts.
        schema_model:
            A Pydantic BaseModel subclass.
        """

        # Allow callers to pass --model ... as before, but default to qwen2.5:7b.
        ollama_model = (model or self.model).strip()

        # Build a strong JSON-only instruction with an explicit schema.
        schema = schema_model.model_json_schema()
        schema_json = json.dumps(schema, ensure_ascii=False)

        strict_user = (
            "You MUST respond with valid JSON only. "
            "Do not include markdown, code fences, or any explanation. "
            "The JSON MUST conform to this JSON Schema:\n"
            f"{schema_json}\n\n"
            "Task:\n"
            f"{user}\n"
        )

        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": strict_user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    r = client.post(f"{self.base_url}/api/chat", json=payload)
                    r.raise_for_status()
                    data = r.json()

                text = (data.get("message", {}) or {}).get("content", "")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"Empty response from Ollama. Raw payload keys: {list(data.keys())}")

                json_str = _extract_json_object(text)
                obj = json.loads(json_str)

                # Pydantic v2: validate and return typed instance
                return schema_model.model_validate(obj)

            except Exception as e:
                last_err = e
                sleep = min(max_backoff, min_backoff * (2 ** attempt))
                sleep *= (0.5 + random.random())
                time.sleep(sleep)

        raise RuntimeError(
            f"Ollama request failed after {max_retries} retries (model={ollama_model}, base_url={self.base_url}): {last_err}"
        ) from last_err
