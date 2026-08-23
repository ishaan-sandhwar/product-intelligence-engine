"""Provider-agnostic LLM client with structured-JSON and vision support.

One interface, four backends. The first provider with a usable API key wins, and
a failure falls through to the next one, so a rate limit on one vendor does not
stop a batch run. Every response records which model produced it, because that
model id ends up in the provenance of any value the call generated.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import (
    API_KEY_ENV,
    DB_PATH,
    LLM_CACHE_ENABLED,
    LLM_MAX_BACKOFF_SECONDS,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_RPM_LIMITS,
    LLM_TEMPERATURE,
    MODEL_IDS,
    PROVIDER_PRIORITY,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Both shapes a Gemini/OpenAI 429 uses to say how long to wait.
_RETRY_HINT = re.compile(r"(?:retry in|retryDelay['\"]?:\s*['\"]?)\s*([\d.]+)\s*s", re.I)


class LLMError(RuntimeError):
    """Raised when every configured provider failed."""


@dataclass
class LLMResponse:
    """A single completion plus the bookkeeping the pipeline needs."""

    text: str
    provider: str
    model_id: str
    cached: bool = False
    latency_ms: int = 0
    raw: Any = None

    def as_json(self, default: Any = None) -> Any:
        """Best-effort JSON parse: fenced block, bare object, or first {...}."""
        candidates: list[str] = []
        fenced = _JSON_BLOCK.search(self.text)
        if fenced:
            candidates.append(fenced.group(1))
        candidates.append(self.text)

        stripped = self.text.strip()
        first = min(
            (stripped.find(c) for c in "[{" if stripped.find(c) != -1), default=-1
        )
        if first > 0:
            last = max(stripped.rfind("}"), stripped.rfind("]"))
            if last > first:
                candidates.append(stripped[first : last + 1])

        for candidate in candidates:
            try:
                return json.loads(candidate.strip())
            except (json.JSONDecodeError, ValueError):
                continue
        logger.warning("Could not parse JSON from %s response", self.provider)
        return default


# ---------------------------------------------------------------- rate limit
class _RateLimiter:
    """Sliding-window request pacer, one per provider.

    A 429 is not free: the rejected request still counts against the per-minute
    quota, so a burst that overruns the limit poisons the next window too. This
    blocks the caller until a slot is genuinely available instead.
    """

    _WINDOW_SECONDS = 60.0

    def __init__(self, requests_per_minute: int) -> None:
        self._rpm = max(int(requests_per_minute), 0)
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request may be issued. No-op when rpm <= 0."""
        if self._rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self._WINDOW_SECONDS
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()
                if len(self._calls) < self._rpm:
                    self._calls.append(now)
                    return
                wait = self._calls[0] + self._WINDOW_SECONDS - now
            logger.debug("rate limit: sleeping %.1fs for a free slot", wait)
            time.sleep(max(wait, 0.05))

    def penalise(self, seconds: float) -> None:
        """Record a server-side rejection so the window reflects the real quota."""
        if self._rpm <= 0:
            return
        with self._lock:
            self._calls.append(time.monotonic() + max(seconds - self._WINDOW_SECONDS, 0.0))


_limiters: dict[str, _RateLimiter] = {
    provider: _RateLimiter(rpm) for provider, rpm in LLM_RPM_LIMITS.items()
}


def _limiter_for(provider: str) -> _RateLimiter:
    return _limiters.setdefault(provider, _RateLimiter(0))


def _retry_after(message: str) -> float | None:
    """Seconds the server asked us to wait, if it said so."""
    match = _RETRY_HINT.search(message)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------- cache
class _ResponseCache:
    """SQLite-backed prompt cache. Makes reruns during a demo instant and free."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS llm_cache (
                       key TEXT PRIMARY KEY,
                       provider TEXT, model_id TEXT, response TEXT, created_at REAL
                   )"""
            )

    @staticmethod
    def key_for(provider: str, model_id: str, payload: str) -> str:
        return hashlib.sha256(f"{provider}|{model_id}|{payload}".encode()).hexdigest()

    def get(self, key: str) -> tuple[str, str, str] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT provider, model_id, response FROM llm_cache WHERE key=?", (key,)
            ).fetchone()
        return row if row else None

    def put(self, key: str, provider: str, model_id: str, response: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?)",
                (key, provider, model_id, response, time.time()),
            )

    def clear(self) -> int:
        with self._lock, self._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            conn.execute("DELETE FROM llm_cache")
        return n


_cache = _ResponseCache(DB_PATH)


# ----------------------------------------------------------------- backends
def _encode_image(path: str | Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    p = Path(path)
    media_type = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return base64.standard_b64encode(p.read_bytes()).decode(), media_type


def _call_anthropic(
    model_id: str, system: str, prompt: str, images: list[str], max_tokens: int, temperature: float
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ[API_KEY_ENV["anthropic"]])
    content: list[dict[str, Any]] = []
    for img in images:
        data, media_type = _encode_image(img)
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
        )
    content.append({"type": "text", "text": prompt})

    message = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "You are a precise data extraction engine.",
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _call_gemini(
    model_id: str, system: str, prompt: str, images: list[str], max_tokens: int, temperature: float
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ[API_KEY_ENV["gemini"]])
    parts: list[Any] = []
    for img in images:
        data, media_type = _encode_image(img)
        parts.append(
            types.Part.from_bytes(data=base64.b64decode(data), mime_type=media_type)
        )
    parts.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text or ""


def _call_openai(
    model_id: str, system: str, prompt: str, images: list[str], max_tokens: int, temperature: float
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[API_KEY_ENV["openai"]])
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        data, media_type = _encode_image(img)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        )
    completion = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system or "You are a precise data extraction engine."},
            {"role": "user", "content": content},
        ],
    )
    return completion.choices[0].message.content or ""


def _call_groq(
    model_id: str, system: str, prompt: str, images: list[str], max_tokens: int, temperature: float
) -> str:
    from groq import Groq

    if images:
        raise LLMError("groq backend has no vision model configured")
    client = Groq(api_key=os.environ[API_KEY_ENV["groq"]])
    completion = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system or "You are a precise data extraction engine."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content or ""


_BACKENDS = {
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "openai": _call_openai,
    "groq": _call_groq,
}


# ------------------------------------------------------------------- client
@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)


class LLMClient:
    """Unified entry point used by every stage of the pipeline."""

    def __init__(self, preferred: str | None = None) -> None:
        self.stats = LLMStats()
        self._lock = threading.Lock()
        order = list(PROVIDER_PRIORITY)
        if preferred and preferred in order:
            order.remove(preferred)
            order.insert(0, preferred)
        self.order = [p for p in order if self.has_key(p)]

    # ------------------------------------------------------------ discovery
    @staticmethod
    def has_key(provider: str) -> bool:
        key = os.getenv(API_KEY_ENV.get(provider, ""), "").strip()
        return bool(key) and not key.lower().startswith("your")

    @classmethod
    def available_providers(cls) -> list[str]:
        return [p for p in PROVIDER_PRIORITY if cls.has_key(p)]

    @property
    def is_configured(self) -> bool:
        return bool(self.order)

    def active_provider(self, vision: bool = False) -> str | None:
        for provider in self.order:
            if not vision or MODEL_IDS[provider].get("vision"):
                return provider
        return None

    # ---------------------------------------------------------------- call
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list[str] | None = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Run a completion against the first working provider."""
        images = images or []
        need_vision = bool(images)
        if not self.order:
            raise LLMError(
                "No LLM provider configured. Set one of: "
                + ", ".join(API_KEY_ENV.values())
                + " in your .env file."
            )

        errors: list[str] = []
        for provider in self.order:
            model_id = MODEL_IDS[provider]["vision" if need_vision else "text"]
            if not model_id:
                errors.append(f"{provider}: no {'vision' if need_vision else 'text'} model")
                continue

            payload = json.dumps(
                {"s": system, "p": prompt, "i": sorted(images), "t": temperature, "m": max_tokens},
                sort_keys=True,
            )
            cache_key = _ResponseCache.key_for(provider, model_id, payload)

            if LLM_CACHE_ENABLED and use_cache:
                hit = _cache.get(cache_key)
                if hit:
                    with self._lock:
                        self.stats.cache_hits += 1
                    return LLMResponse(
                        text=hit[2], provider=hit[0], model_id=hit[1], cached=True
                    )

            limiter = _limiter_for(provider)
            for attempt in range(1, LLM_MAX_RETRIES + 1):
                limiter.acquire()
                started = time.perf_counter()
                try:
                    text = _BACKENDS[provider](
                        model_id, system, prompt, images, max_tokens, temperature
                    )
                    latency = int((time.perf_counter() - started) * 1000)
                    with self._lock:
                        self.stats.calls += 1
                        self.stats.by_provider[provider] = (
                            self.stats.by_provider.get(provider, 0) + 1
                        )
                    if LLM_CACHE_ENABLED and use_cache and text:
                        _cache.put(cache_key, provider, model_id, text)
                    return LLMResponse(
                        text=text, provider=provider, model_id=model_id, latency_ms=latency
                    )
                except Exception as exc:  # noqa: BLE001 - fall through to next provider
                    message = f"{type(exc).__name__}: {exc}"
                    transient = any(
                        tok in message.lower()
                        for tok in ("rate", "429", "timeout", "overload", "503", "529")
                    )
                    if transient and attempt < LLM_MAX_RETRIES:
                        # A quota error tells us when the window reopens; waiting
                        # that long beats an exponential guess that lands early
                        # and burns another slot on a second rejection.
                        hinted = _retry_after(message)
                        if hinted is not None:
                            limiter.penalise(hinted)
                        delay = min(
                            hinted if hinted is not None else 2 ** attempt,
                            LLM_MAX_BACKOFF_SECONDS,
                        )
                        logger.info(
                            "%s transient failure (attempt %d/%d), waiting %.1fs",
                            provider, attempt, LLM_MAX_RETRIES, delay,
                        )
                        time.sleep(delay + 0.25)
                        continue
                    errors.append(f"{provider}: {message}")
                    logger.warning("provider %s failed: %s", provider, message)
                    break

        with self._lock:
            self.stats.failures += 1
        raise LLMError("all providers failed -> " + " | ".join(errors))

    def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        images: list[str] | None = None,
        default: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, LLMResponse]:
        """Completion that must return JSON. Returns (parsed, response)."""
        guard = (
            "\n\nRespond with valid JSON only. No prose, no markdown fences, "
            "no explanation before or after the JSON."
        )
        response = self.complete(prompt + guard, system=system, images=images, **kwargs)
        return response.as_json(default), response


def clear_cache() -> int:
    """Drop every cached completion. Returns the number of rows removed."""
    return _cache.clear()
