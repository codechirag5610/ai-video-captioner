"""Load model-agnostic config from YAML + secrets from env.

Everything the pipeline needs to talk to Fireworks lives in config/models.yaml.
Launch day = edit YAML, not code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "models.yaml"


@dataclass
class Route:
    """One (model, endpoint, key) a stage can call. Stages hold a primary route
    plus fallbacks; a route whose key env var is empty is skipped entirely."""
    model: str
    base_url: str = ""      # empty => the default Fireworks api.base_url
    api_key: str = ""       # empty => the Fireworks key
    provider: str = ""      # informational, for logging (gemini/openrouter/"")


@dataclass
class ModelSpec:
    model: str
    supports_vision: bool = False
    max_images: int = 16
    image_max_edge: int = 768
    image_format: str = "jpeg"
    image_quality: int = 85
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_s: int = 0                 # 0 => the api-level default
    enabled: bool = True               # judge.enabled gates best-of-N selection
    provider: str = ""                 # name from providers: (empty = Fireworks)
    fallbacks: list = field(default_factory=list)  # resolved to list[Route] in load()
    routes: list = field(default_factory=list)     # [primary Route, *fallback Routes]
    # Best-of-N knobs (only meaningful for the style model).
    n_candidates: int = 4              # candidates generated per style
    temperature_formal: float = 0.3    # low temp: formal wants precision
    temperature_humor: float = 0.9     # high temp: humor/sarcasm want variance
    # Passed verbatim into the request body. Use for provider-specific params like
    # {"reasoning_effort": "none"} to disable a reasoning model's chain-of-thought
    # (which otherwise burns the token budget and truncates the JSON answer).
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


@dataclass
class ApiConfig:
    base_url: str = "https://api.fireworks.ai/inference/v1"
    audio_base_url: str = "https://api.fireworks.ai/inference/v1"
    timeout_s: int = 120
    max_retries: int = 5
    api_key: str = ""


@dataclass
class AsrConfig:
    backend: str = "local"   # local | fireworks | none
    model: str = "whisper-v3"
    local_model_size: str = "base"


@dataclass
class CritiqueConfig:
    enabled: bool = True
    min_score: float = 7.0
    max_retries: int = 1


@dataclass
class ComedyConfig:
    """Stage 3: comedy-material extraction. Reuses the style model unless a
    separate `model` is given in config."""
    enabled: bool = True
    model: str = ""          # empty => use the style model
    temperature: float = 0.7


@dataclass
class Config:
    api: ApiConfig
    understand: ModelSpec
    style: ModelSpec
    judge: ModelSpec
    asr: AsrConfig
    critique: CritiqueConfig
    comedy: ComedyConfig
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else _DEFAULT_CONFIG
        if not path.exists():
            raise RuntimeError(f"config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise RuntimeError(f"config file is empty or malformed: {path}")
        for key in ("understand", "style", "judge"):
            if key not in raw:
                raise RuntimeError(f"config missing required section '{key}': {path}")

        api_raw = raw.get("api", {})
        api = ApiConfig(
            base_url=os.getenv("FIREWORKS_BASE_URL", api_raw.get("base_url", ApiConfig.base_url)),
            audio_base_url=api_raw.get("audio_base_url", api_raw.get("base_url", ApiConfig.audio_base_url)),
            timeout_s=api_raw.get("timeout_s", 25),
            max_retries=api_raw.get("max_retries", 2),
            api_key=os.getenv("FIREWORKS_API_KEY", ""),
        )
        if not api.api_key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export FIREWORKS_API_KEY."
            )

        providers = raw.get("providers", {}) or {}

        def resolve_routes(spec: ModelSpec) -> None:
            """Build spec.routes = [primary, *fallbacks], skipping providers
            whose API key env var is empty so a missing key costs nothing."""
            def route_for(model: str, provider_name: str) -> Route | None:
                if not provider_name:  # Fireworks default
                    return Route(model=model, base_url="", api_key="", provider="")
                p = providers.get(provider_name)
                if not p:
                    return None
                key = os.getenv(p.get("api_key_env", ""), "")
                if not key:
                    return None
                return Route(model=model, base_url=p.get("base_url", ""),
                             api_key=key, provider=provider_name)

            routes = []
            primary = route_for(spec.model, spec.provider)
            if primary:
                routes.append(primary)
            for fb in spec.fallbacks or []:
                r = route_for(fb.get("model", ""), fb.get("provider", "") or "")
                if r and r.model:
                    routes.append(r)
            if not routes:  # everything skipped -> plain Fireworks with the spec model
                routes.append(Route(model=spec.model))
            spec.routes = routes

        understand = ModelSpec.from_dict(raw["understand"])
        style = ModelSpec.from_dict(raw["style"])
        judge = ModelSpec.from_dict(raw["judge"])
        for spec in (understand, style, judge):
            resolve_routes(spec)

        return cls(
            api=api,
            understand=understand,
            style=style,
            judge=judge,
            asr=AsrConfig(**{k: v for k, v in raw.get("asr", {}).items() if k in AsrConfig.__dataclass_fields__}),
            critique=CritiqueConfig(**{k: v for k, v in raw.get("critique", {}).items() if k in CritiqueConfig.__dataclass_fields__}),
            comedy=ComedyConfig(**{k: v for k, v in raw.get("comedy", {}).items() if k in ComedyConfig.__dataclass_fields__}),
            raw=raw,
        )
