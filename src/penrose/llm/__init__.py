"""Model abstraction layer.

All LLM roles default to GLM 5x. Recommendation:
one OpenAI-compatible adapter behind a role seam + budget caps + content-hash
cache. Defer multi-provider routing, fallback, eval harness until the first model
swap (its only trigger).

Swap providers by editing config (role -> provider+model) or env vars. No code
changes needed for any OpenAI-compatible endpoint (Artificial Analysis, OpenAI,
GLM's compat endpoint, Ollama, LiteLLM proxy).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .. import config


# --- response types --------------------------------------------------------- #

@dataclass
class LLMResponse:
    text: str
    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    elapsed_s: float
    cached: bool = False
    finish_reason: str | None = None
    independent_verifier: bool = False


# --- budget enforcement ----------------------------------------------------- #

@dataclass
class _Budget:
    spent_today_usd: float = 0.0
    day: str = time.strftime("%Y-%m-%d", time.gmtime())

    def _rollover(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if self.day != today:
            self.day = today
            self.spent_today_usd = 0.0

    def charge(self, usd: float) -> None:
        self._rollover()
        self.spent_today_usd += usd
        if self.spent_today_usd > config.LLM_BUDGET["max_usd_per_day"]:
            raise RuntimeError(
                f"LLM daily budget exceeded: ${self.spent_today_usd:.3f} > "
                f"${config.LLM_BUDGET['max_usd_per_day']:.3f} (per F6 budget cap)"
            )


_BUDGET = _Budget()


def reset_budget() -> None:
    """Test hook."""
    _BUDGET.spent_today_usd = 0.0


# --- content-hash cache ----------------------------------------------------- #

def _cache_key(model: str, messages: list[dict], **kw) -> str:
    blob = json.dumps({"model": model, "messages": messages, **kw}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    p = config.LLM_CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def _cache_put(key: str, payload: dict) -> None:
    config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = config.LLM_CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps(payload, default=str))


# --- the OpenAI-compatible adapter ------------------------------------------ #

class OpenAICompatProvider:
    """One adapter for everything OpenAI-shaped.

    Works for: Artificial Analysis, OpenAI, z.ai's compat endpoint, Ollama,
    LiteLLM proxy, vLLM, etc. Configure via env vars or config.LLM_PROVIDER.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("PENROSE_LLM_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("PENROSE_LLM_BASE_URL")
                         or "https://api.artificialanalysis.ai/v1").rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, model: str, messages: list[dict],
                 max_tokens: int = 2000, temperature: float = 0.2,
                 response_format: Optional[dict] = None, timeout: int = 90,
                 extra_body: Optional[dict] = None) -> dict:
        if not self.available:
            raise RuntimeError(
                "no PENROSE_LLM_API_KEY set; copy .env.example to .env and add a key "
                "(F6-lite needs an OpenAI-compatible endpoint to call)"
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            body["response_format"] = response_format
        if extra_body:
            # Some providers (Gemini 2.5 thinking models) accept extra knobs via
            # an OpenAI-extension `extra_body`. Penrose uses this to cap thinking
            # tokens when needed; harmless on providers that ignore it.
            body.update(extra_body)

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "User-Agent": "penrose/0.2"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:500] if hasattr(e, "read") else str(e)
            raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e

        usage = data.get("usage", {})
        choice = (data.get("choices") or [{}])[0]
        return {
            "text": (choice.get("message", {}) or {}).get("content") or "",
            "model": data.get("model", model),
            "in_tokens": usage.get("prompt_tokens", 0),
            "out_tokens": usage.get("completion_tokens", 0),
            "finish_reason": choice.get("finish_reason"),
        }


# --- role-driven public API ------------------------------------------------- #

_PROVIDER: Optional[OpenAICompatProvider] = None


def _provider() -> OpenAICompatProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = OpenAICompatProvider()
    return _PROVIDER


def _default_provider_config() -> tuple[str, str]:
    api_key = os.environ.get("PENROSE_LLM_API_KEY") or ""
    base_url = (os.environ.get("PENROSE_LLM_BASE_URL")
                or "https://api.artificialanalysis.ai/v1").rstrip("/")
    return base_url, api_key


def _verifier_provider_config() -> tuple[str, str]:
    base_url = (getattr(config, "VERIFIER_LLM_BASE_URL", "")
                or os.environ.get("PENROSE_LLM_VERIFIER_BASE_URL", "")).rstrip("/")
    api_key = (getattr(config, "VERIFIER_LLM_API_KEY", "")
               or os.environ.get("PENROSE_LLM_VERIFIER_API_KEY", ""))
    return base_url, api_key


def _fidelity_verifier_provider() -> tuple[OpenAICompatProvider, bool, str | None]:
    """Return provider, whether it is genuinely independent, and a cache namespace."""
    verifier_base, verifier_key = _verifier_provider_config()
    if not (verifier_base and verifier_key):
        return _provider(), False, None
    default_base, _default_key = _default_provider_config()
    # Independence is a PROVIDER property, not a billing-key one: a different api_key at the SAME
    # base_url and the same default model is still the model judging its own work. Only a different
    # provider endpoint OR a different verifier model counts as genuinely independent.
    independent = (verifier_base != default_base) or (
        config.VERIFIER_LLM_MODEL != config.DEFAULT_LLM_MODEL)
    namespace = hashlib.sha256(f"{verifier_base}\0{verifier_key}".encode()).hexdigest()[:12]
    return OpenAICompatProvider(api_key=verifier_key, base_url=verifier_base), independent, namespace


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    """Rough cost in USD; per-model pricing in config.LLM_PRICING."""
    p = config.LLM_PRICING.get(model, config.LLM_PRICING["__default__"])
    return (in_tok * p["in_per_m"] + out_tok * p["out_per_m"]) / 1e6


def call(role: str, messages: list[dict], *,
         max_tokens: Optional[int] = None, temperature: float = 0.2,
         json_mode: bool = False, use_cache: bool = True,
         timeout: Optional[int] = None, extra_body: Optional[dict] = None) -> LLMResponse:
    """Run one LLM call for a named role.

    role: one of config.LLM_ROLES (claim_extractor, falsifiability_classifier,
    deep_reader, dreamer, synthesizer, qual_lens_*, module_spec_generator)

    max_tokens: optional cap. Defaults to the role's configured max_tokens.
    Cannot exceed the role cap — calling with a higher value is silently capped.

    extra_body: provider-specific extensions (e.g. Gemini 2.5 thinking_budget).
    """
    if role not in config.LLM_ROLES:
        raise KeyError(f"unknown LLM role {role!r}; defined roles: {list(config.LLM_ROLES)}")

    role_cfg = config.LLM_ROLES[role]
    model = role_cfg["model"]
    if timeout is None:
        timeout = int(config.LLM_TIMEOUTS.get(role, config.LLM_TIMEOUTS.get("default", 90)))
    role_cap = role_cfg.get("max_tokens", 2000)
    if max_tokens is None:
        max_tokens = role_cap
    else:
        max_tokens = min(max_tokens, role_cap)

    provider = _provider()
    independent_verifier = False
    verifier_cache_namespace = None
    verifier_provider_configured = False
    if role == "fidelity_refuter":
        provider, independent_verifier, verifier_cache_namespace = _fidelity_verifier_provider()
        verifier_provider_configured = verifier_cache_namespace is not None

    cache_extra = ({"verifier_provider": verifier_cache_namespace}
                   if verifier_provider_configured else {})
    key = _cache_key(model, messages, max_tokens=max_tokens, temperature=temperature,
                     json_mode=json_mode, **cache_extra)
    if use_cache:
        cached = _cache_get(key)
        if cached:
            return LLMResponse(**cached)

    t0 = time.time()
    verifier_fallback = False
    try:
        payload = provider.complete(
            model, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
            timeout=timeout,
            extra_body=extra_body,
        )
    except Exception:  # noqa: BLE001 - verifier provider must degrade to the default path
        if role != "fidelity_refuter" or not verifier_provider_configured:
            raise
        verifier_fallback = True
        independent_verifier = False
        payload = _provider().complete(
            model, messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
            timeout=timeout,
            extra_body=extra_body,
        )
    elapsed = time.time() - t0
    cost = _estimate_cost(payload["model"], payload["in_tokens"], payload["out_tokens"])

    # per-run cap
    if cost > role_cfg.get("max_cost_per_call", 1.0):
        raise RuntimeError(
            f"role {role} call cost ${cost:.4f} exceeds per-call cap "
            f"${role_cfg['max_cost_per_call']:.4f}"
        )

    _BUDGET.charge(cost)

    resp = LLMResponse(
        text=payload["text"], model=payload["model"],
        in_tokens=payload["in_tokens"], out_tokens=payload["out_tokens"],
        cost_usd=cost, elapsed_s=round(elapsed, 3), cached=False,
        finish_reason=payload.get("finish_reason"),
        independent_verifier=bool(independent_verifier),
    )
    if use_cache and not verifier_fallback and (resp.text or "").strip():   # never cache an empty reply (would poison retries)
        _cache_put(key, {"text": resp.text, "model": resp.model,
                         "in_tokens": resp.in_tokens, "out_tokens": resp.out_tokens,
                         "cost_usd": resp.cost_usd, "elapsed_s": resp.elapsed_s,
                         "cached": False, "independent_verifier": resp.independent_verifier})
    return resp


def _parse_or_repair(text: str):
    """Return a parsed dict from JSON text, attempting a truncation repair, else None."""
    txt = (text or "").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(text)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None
    return None


def call_json(role: str, messages: list[dict], *, _max_attempts: int = 3,
              **kw) -> tuple[dict, LLMResponse]:
    """Convenience: role call with json_mode=True, parsed response.

    glm-5.2 is a thinking model and intermittently returns an EMPTY or truncated body
    (reasoning tokens crowd out the content). So we retry: attempt 0 may read cache;
    every retry forces a fresh call (use_cache=False) so a transient empty/bad reply
    can't be re-served. Empty replies are never cached (see call()). Repairs truncated
    JSON when possible. Raises RuntimeError only after all attempts fail."""
    kw.setdefault("max_tokens", None)
    kw.pop("use_cache", None)
    last_text, last_finish, last_err = "", None, None
    for attempt in range(max(1, _max_attempts)):
        try:
            r = call(role, messages, json_mode=True, use_cache=(attempt == 0), **kw)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # Transient network / read-timeout (e.g. a slow z.ai extraction call). A single
            # hiccup must NOT kill the whole paper -> retry with a fresh (uncached) call and
            # linear backoff; only raise if EVERY attempt fails. (P2-timeout hardening)
            last_err = e
            if attempt < _max_attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"role {role} network error after {_max_attempts} attempts: {e}") from e
        except RuntimeError as e:
            # call() wraps HTTPError as 'LLM HTTP <code>'. Retry transient 429/5xx; a 4xx
            # (auth / bad request) is not transient -> re-raise immediately.
            msg = str(e)
            if ("LLM HTTP 5" in msg or "LLM HTTP 429" in msg) and attempt < _max_attempts - 1:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        parsed = _parse_or_repair(r.text)
        if parsed is not None:
            return parsed, r
        last_text, last_finish = (r.text or "").strip(), r.finish_reason
    hint = (" (finish_reason=length -> raise the role's max_tokens; reasoning tokens "
            "crowded out the JSON content)" if last_finish == "length" else "")
    raise RuntimeError(
        f"role {role} returned un-parseable/empty JSON after {_max_attempts} attempts "
        f"(last body {len(last_text)} chars, finish_reason={last_finish}){hint}: {last_text[:300]}"
    )


def _cache_evict(key: str) -> None:
    p = config.LLM_CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def _repair_truncated_json(text: str) -> Optional[str]:
    """Best-effort repair of LLM JSON truncated at max_tokens.

    Strategy: scan the text left-to-right tracking bracket depth while respecting
    string literals (so a literal "}," inside a string value can't fool us — the
    A-034 bug in the old rfind("},") approach). We remember the position of the
    last point where a complete top-level container item closed at depth 1 (i.e.
    a fully-formed element of the outer array/object). We cut there and then close
    the remaining open containers in correct nesting order.

    Works for the common case "returned 3 claims instead of 5 because token cap
    hit". Returns None if it can't produce something plausibly closeable.
    """
    s = text.strip()
    if not s:
        return None

    stack: list[str] = []          # open container chars, in nesting order
    saw_container = False          # did we ever open a {/[ ?
    in_str = False
    escaped = False
    # index (exclusive) just after the last container element that closed cleanly
    # at the top item level, plus the stack snapshot to close from there.
    cut_at = -1
    cut_stack: list[str] = []

    for i, ch in enumerate(s):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
            saw_container = True
        elif ch in "}]":
            if not stack:
                # unbalanced close outside any container; nothing sane to do
                break
            stack.pop()
            # A container just closed cleanly. If at least one container is still
            # open, we just completed a nested element (e.g. one claim object in
            # the "claims" array) — a safe point to cut and close the rest.
            if stack:
                cut_at = i + 1
                cut_stack = list(stack)
        elif ch == "," and stack and not in_str:
            # separator after a complete value inside an open container — also a
            # safe cut point (drops the trailing incomplete value after it).
            cut_at = i  # exclude the comma
            cut_stack = list(stack)

    if not saw_container:
        # no JSON container at all — nothing repairable
        return None

    # If the whole thing is already balanced (all containers closed), no repair
    # needed; hand it back for the caller to json.loads.
    if not stack and not in_str:
        return s

    if cut_at <= 0:
        # never saw a complete top-level element; can't repair safely.
        # Last resort: if we're mid-string or shallow, just close current stack.
        if in_str or not stack:
            return None
        repair = s
        for ch in reversed(stack):
            repair += "}" if ch == "{" else "]"
        # validate-ability is checked by the caller (json.loads); return best effort
        return repair

    truncated = s[:cut_at]
    repair = truncated
    for ch in reversed(cut_stack):
        repair += "}" if ch == "{" else "]"
    return repair


# --- embeddings (for P5 dedup) ---------------------------------------------- #

_EMBED_ENDPOINT = os.environ.get("PENROSE_EMBED_BASE_URL",
                                 "http://localhost:8088/v1")  # optional legacy accelerator
_EMBED_MODEL = os.environ.get("PENROSE_EMBED_MODEL", "bge-small-en-v1.5")
_LOCAL_EMBED_MODEL = os.environ.get("PENROSE_LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_EMBED_AVAILABLE_CACHE: Optional[bool] = None
_LOCAL_EMBEDDER: Any | None = None
_LOCAL_EMBEDDER_FAILED = False


def _local_embedder() -> Any | None:
    """Return the optional in-process FastEmbed model, or None if unavailable."""
    global _LOCAL_EMBEDDER, _LOCAL_EMBEDDER_FAILED
    if _LOCAL_EMBEDDER is not None:
        return _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER_FAILED:
        return None
    try:
        from fastembed import TextEmbedding  # type: ignore

        cache_dir = config.ROOT / ".embed_cache" / "fastembed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache_dir))
        try:
            _LOCAL_EMBEDDER = TextEmbedding(model_name=_LOCAL_EMBED_MODEL, cache_dir=str(cache_dir))
        except TypeError:
            # Older fastembed versions use FASTEMBED_CACHE_PATH but do not accept cache_dir.
            _LOCAL_EMBEDDER = TextEmbedding(model_name=_LOCAL_EMBED_MODEL)
        return _LOCAL_EMBEDDER
    except Exception:  # noqa: BLE001
        _LOCAL_EMBEDDER_FAILED = True
        return None


def embed_local_available() -> bool:
    """True when the optional in-process FastEmbed path can be constructed."""
    return _local_embedder() is not None


def embed_local(text: str) -> Optional[list[float]]:
    """Embed a single string in-process with optional fastembed; never uses network services."""
    if not text:
        return None
    key = _cache_key("embed-local", [{"text": text}], model=_LOCAL_EMBED_MODEL)
    cached = _cache_get(key)
    if cached and "vector" in cached:
        return cached["vector"]
    model = _local_embedder()
    if model is None:
        return None
    try:
        vecs = list(model.embed([text]))
        if not vecs:
            return None
        vec = [float(x) for x in vecs[0]]
        _cache_put(key, {"vector": vec, "model": _LOCAL_EMBED_MODEL, "backend": "fastembed"})
        return vec
    except Exception:  # noqa: BLE001
        return None


def _server_embed_available() -> bool:
    """Probe the optional legacy embeddings endpoint once per session."""
    global _EMBED_AVAILABLE_CACHE
    if _EMBED_AVAILABLE_CACHE is not None:
        return _EMBED_AVAILABLE_CACHE
    try:
        url = f"{_EMBED_ENDPOINT.rstrip('/')}/models"
        req = urllib.request.Request(url, headers={"User-Agent": "penrose/0.2"})
        with urllib.request.urlopen(req, timeout=3) as r:
            _EMBED_AVAILABLE_CACHE = (r.status == 200)
    except Exception:  # noqa: BLE001
        _EMBED_AVAILABLE_CACHE = False
    return _EMBED_AVAILABLE_CACHE


def embed_available() -> bool:
    """Return whether the legacy embeddings endpoint is live."""
    return _server_embed_available()


def embed(text: str) -> Optional[list[float]]:
    """Embed a single string with the optional legacy server.

    Returns None if the server backend is unavailable so callers can fall back safely.
    Embeddings are cached with the same content-hash scheme as LLM calls.
    """
    if not text:
        return None
    key = _cache_key("embed", [{"text": text}])
    cached = _cache_get(key)
    if cached and "vector" in cached:
        return cached["vector"]
    if not _server_embed_available():
        return None
    try:
        body = json.dumps({"model": _EMBED_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            f"{_EMBED_ENDPOINT.rstrip('/')}/embeddings",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local",
                     "User-Agent": "penrose/0.2"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        vec = data["data"][0]["embedding"]
        _cache_put(key, {"vector": vec, "model": _EMBED_MODEL})
        return vec
    except Exception:  # noqa: BLE001
        return None


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
