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


# Fallback events (e.g. Gemma route failed over to Fireworks), recorded so the
# harness can write an auditable run report next to results.json.
RUN_EVENTS: list[dict[str, Any]] = []


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
        self._route_clients: dict[tuple[str, str], OpenAI] = {}

    def _client_for(self, route, timeout_s: float) -> OpenAI:
        """One cached OpenAI client per (endpoint, key). Empty route fields mean
        the default Fireworks client settings."""
        base = route.base_url or self.api.base_url
        key = route.api_key or self.api.api_key
        cache_key = (base, key)
        if cache_key not in self._route_clients:
            self._route_clients[cache_key] = OpenAI(
                api_key=key, base_url=base, timeout=timeout_s, max_retries=0
            )
        return self._route_clients[cache_key]

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
        """One chat completion. Walks the spec's route chain (primary provider,
        then fallbacks) so a failing provider costs seconds, never the clip.
        Returns the assistant text."""
        timeout_s = float(spec.timeout_s or self.api.timeout_s)
        routes = getattr(spec, "routes", None) or [None]
        last_exc: Exception | None = None
        for i, route in enumerate(routes):
            try:
                text, finish = self._chat_once(
                    spec, route, messages, timeout_s,
                    json_mode=json_mode, temperature=temperature, max_tokens=max_tokens,
                )
                self._last_finish_reason = finish
                if i > 0 and route is not None:
                    RUN_EVENTS.append({
                        "event": "route_fallback", "stage_model": spec.model,
                        "used": route.model, "provider": route.provider or "fireworks",
                    })
                return text
            except Exception as e:  # noqa: BLE001 - any route failure falls through
                last_exc = e
                log.warning("route %s/%s failed for %s: %s", i + 1, len(routes),
                            getattr(route, "model", spec.model), e)
        raise last_exc if last_exc else RuntimeError("no routes configured")

    def _chat_once(self, spec, route, messages, timeout_s, *, json_mode, temperature, max_tokens):
        model = route.model if route is not None else spec.model
        client = self._client_for(route, timeout_s) if route is not None else self._chat

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.api.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=5),  # clamped: a bad call costs seconds
            retry=retry_if_exception_type(_RETRYABLE),
            before_sleep=lambda rs: log.warning(
                "retry %s/%s for %s: %s",
                rs.attempt_number, self.api.max_retries, model, rs.outcome.exception(),
            ),
        )
        def _call() -> tuple[str, str]:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=spec.temperature if temperature is None else temperature,
                max_tokens=spec.max_tokens if max_tokens is None else max_tokens,
                timeout=timeout_s,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if spec.extra_body:
                # provider-specific body params, e.g. {"reasoning_effort": "none"}
                kwargs["extra_body"] = spec.extra_body
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            return choice.message.content or "", (choice.finish_reason or "")

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
        except _JSON_FALLBACK as e:
            # Only worth re-asking when the model actually finished ('stop');
            # a max_tokens truncation would just truncate again.
            if getattr(self, "_last_finish_reason", "") == "length":
                raise
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
