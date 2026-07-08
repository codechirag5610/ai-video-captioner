"""Model-agnostic Fireworks client.

Fireworks exposes an OpenAI-compatible API, so we use the openai SDK pointed at
the Fireworks base_url. Everything downstream (Stage A/B, judge) calls through
this thin wrapper, which centralizes retries, backoff, and JSON coercion.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import ApiConfig, ModelSpec

log = logging.getLogger("captioner.client")

# Errors worth retrying (rate limits, transient 5xx, timeouts, conn resets).
try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    _RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
except Exception:  # pragma: no cover - defensive against SDK version drift
    _RETRYABLE = (Exception,)

# json_mode fallback should fire ONLY when json_mode itself is the problem
# (model rejects response_format -> 400) or the output can't be parsed. Retryable
# errors are already handled by tenacity in chat(); don't double-retry them.
try:
    from openai import BadRequestError
    _JSON_FALLBACK = (BadRequestError, ValueError)  # JSONDecodeError subclasses ValueError
except Exception:  # pragma: no cover
    _JSON_FALLBACK = (ValueError,)


class FireworksClient:
    def __init__(self, api: ApiConfig):
        self.api = api
        self._chat = OpenAI(
            api_key=api.api_key,
            base_url=api.base_url,
            timeout=api.timeout_s,
            max_retries=0,  # we handle retries via tenacity for uniform backoff
        )
        self._audio = OpenAI(
            api_key=api.api_key,
            base_url=api.audio_base_url,
            timeout=api.timeout_s,
            max_retries=0,
        )

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One chat completion. Returns the assistant text."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.api.max_retries),
            wait=wait_exponential(multiplier=1.5, min=2, max=60),
            retry=retry_if_exception_type(_RETRYABLE),
            before_sleep=lambda rs: log.warning(
                "retry %s/%s for %s: %s",
                rs.attempt_number, self.api.max_retries, spec.model, rs.outcome.exception(),
            ),
        )
        def _call() -> str:
            kwargs: dict[str, Any] = dict(
                model=spec.model,
                messages=messages,
                temperature=spec.temperature if temperature is None else temperature,
                max_tokens=spec.max_tokens if max_tokens is None else max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if spec.extra_body:
                # provider-specific body params, e.g. {"reasoning_effort": "none"}
                kwargs["extra_body"] = spec.extra_body
            resp = self._chat.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return _call()

    def chat_json(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat that must return JSON. Tries native json_mode, falls back to
        extracting the first JSON object from the text (some models/endpoints
        don't honor response_format)."""
        try:
            text = self.chat(spec, messages, json_mode=True, **kwargs)
            return _parse_json(text)
        except _JSON_FALLBACK as e:  # json_mode unsupported OR parse failure -> retry plain
            log.debug("json_mode path failed (%s); retrying without response_format", e)
            text = self.chat(spec, messages, json_mode=False, **kwargs)
            return _parse_json(text)

    # ----------------------------------------------------------------- audio
    def transcribe(self, audio_path: str, model: str) -> dict[str, Any]:
        """Whisper transcription via Fireworks audio endpoint.
        Returns {text, language} (language may be None if not reported)."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.api.max_retries),
            wait=wait_exponential(multiplier=1.5, min=2, max=60),
            retry=retry_if_exception_type(_RETRYABLE),
        )
        def _call() -> dict[str, Any]:
            with open(audio_path, "rb") as f:
                resp = self._audio.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="verbose_json",
                )
            text = getattr(resp, "text", "") or ""
            language = getattr(resp, "language", None)
            return {"text": text.strip(), "language": language}

        return _call()


def _parse_json(text: str) -> dict[str, Any]:
    """Robustly pull a JSON object out of a model response."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the first {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from model output: {text[:300]!r}")
