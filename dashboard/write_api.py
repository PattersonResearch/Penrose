"""F15 — pre-flight + submit endpoints for the chat tab.

Two endpoints that augment the read-only dashboard server:

  POST /preflight       — runs P3 (falsifiability) + P4 (fee curve) + P5 (dedup)
                          against a draft thesis. READ-ONLY: no writes anywhere.
                          Returns JSON the chat tab renders as diagnostic cards.

  POST /submit-claim    — writes the refined thesis to inbox/ as a .txt file
                          for the next scan cadence to pick up. WRITE, but only
                          to the inbox queue — never to the brain, archives,
                          decisions log, or review queue. The HUB_DESIGN_SPEC
                          "server never writes" invariant applies to project
                          state; inbox/ is a queue, not state.

Origin validation: only same-origin requests (file:// or http://127.0.0.1:*)
are accepted. A random per-launch token in $PENROSE_ROOT/.dash-token must also
be supplied via X-Penrose-Token header. Defense against browser-origin attacks
on localhost (binding to 127.0.0.1 alone is not sufficient).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

PENROSE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PENROSE_ROOT / "src"))

TOKEN_FILE = PENROSE_ROOT / ".dash-token"
CHAT_LOG = PENROSE_ROOT / "chat_log"   # conversations persisted locally, one JSON per session
_SESSION_TOKEN: str | None = None


def _ensure_token() -> str:
    """Generate once per server process and return the per-launch token. The dashboard reads this
    via a separate endpoint and includes it in X-Penrose-Token on submits."""
    global _SESSION_TOKEN
    if _SESSION_TOKEN is None:
        _SESSION_TOKEN = secrets.token_urlsafe(32)
        TOKEN_FILE.write_text(_SESSION_TOKEN)
        TOKEN_FILE.chmod(0o600)
    return _SESSION_TOKEN


def _check_origin(headers) -> bool:
    """Strict origin validation. Allow file:// (local HTML) and same-host HTTP."""
    origin = headers.get("Origin", "")
    if not origin:
        return True  # non-browser clients (curl, internal scripts)
    if origin.startswith("file://"):
        return True
    # http(s)://127.0.0.1:any or localhost:any
    m = re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin)
    return m is not None


def _read_json_body(handler, max_bytes: int = 65536) -> dict:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length == 0 or length > max_bytes:
        raise ValueError("body too large or empty")
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# Preflight: read-only checks against a draft thesis
# --------------------------------------------------------------------------- #
def run_preflight(thesis_text: str) -> dict:
    """Run P3/P4/P5 against a draft thesis. READ-ONLY. Returns JSON for the chat tab."""
    from penrose.brain import BrainReader, Claim
    from penrose.pipeline import stages, extract
    from penrose import config

    text = (thesis_text or "").strip()
    if not text or len(text) < 20:
        return {"ok": False, "error": "thesis too short (need at least 20 chars)"}

    # Fabricate a transient Claim for the cheap filters. Not persisted anywhere.
    claim = Claim(
        claim_id="preflight-draft",
        statement=text[:500],
        mechanism="",
        scope="",
        horizon="",
        source_id="chat",
        source_span=text[:200],   # the draft is its own span
        claimed_metric_quote="",
        applicable_strategy_class="chat-draft",
    )

    reader = BrainReader()
    checks: list[dict] = []

    # P3 — falsifiability. Use stub for speed/determinism in v1 (LLM cost + latency
    # for live brainstorming isn't worth it).
    p3 = extract.classify_claim_stub(claim)
    p3["phase"] = "P3"
    p3["name"] = "Falsifiability"
    p3["description"] = ("Claim must specify a directional, testable prediction "
                         "with a measurable horizon.")
    checks.append(p3)

    # P4 — fee curve (deterministic; uses default edge estimate since user hasn't
    # provided one. The point is to show whether the claim WOULD survive at 50c.)
    try:
        p4 = stages.p4_fee_curve(claim, expected_edge=0.02)
        p4["phase"] = "P4"
        p4["name"] = "Fee-curve sanity"
        p4["description"] = (
            f"Edge must survive {p4.get('binary_fee_at_50c', 0.018):.2%} taker fee at 50c."
        )
        checks.append(p4)
    except Exception as e:  # noqa: BLE001
        checks.append({"phase": "P4", "name": "Fee-curve sanity", "v": "warn",
                       "error": str(e)[:200]})

    # P5 — dedup against committed brain atoms
    try:
        p5 = stages.p5_dedup(claim, reader)
        p5["phase"] = "P5"
        p5["name"] = "Dedup vs brain"
        p5["description"] = "No existing committed claim covers this hypothesis."
        checks.append(p5)
    except Exception as e:  # noqa: BLE001
        # brain unavailable — soft warn, don't block submission
        checks.append({"phase": "P5", "name": "Dedup vs brain", "v": "warn",
                       "error": str(e)[:200]})

    # overall: any check can be killed/warn; only P3 hard-kills (unfalsifiable)
    overall = "ok" if all(not c.get("killed") for c in checks) else "blocked"
    if any(c.get("v") == "warn" or c.get("error") for c in checks):
        overall = "warn" if overall == "ok" else overall

    return {"ok": True, "overall": overall, "checks": checks,
            "thesis": text, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# --------------------------------------------------------------------------- #
# Conversational assistant: discuss & steer toward a testable hypothesis
# --------------------------------------------------------------------------- #
_CHAT_SYSTEM = (
    "You are Pennie, Penrose's research assistant — a knowledgeable, skeptical "
    "quant-research collaborator embedded in Penrose, a falsification-first research "
    "engine. (If asked your name, you are Pennie.) You discuss trading-strategy "
    "ideas (signals, mechanisms, data sources, transaction costs, regimes) with the "
    "user and STEER them toward a hypothesis penrose can actually test.\n\n"
    "penrose can only test a hypothesis that meets ALL of these criteria — guide the "
    "user toward them:\n"
    "  1. A SINGLE falsifiable directional claim (X predicts the sign/direction of a "
    "forward return), not a vague theme or a bundle of claims.\n"
    "  2. A clear signal -> forward-return MECHANISM with NO look-ahead (the signal "
    "must be observable strictly before the return window it predicts).\n"
    "  3. DATA that plausibly exists: crypto prices/funding rates, Kalshi macro "
    "signals, weather series, or standard vendor market data. If the data almost "
    "certainly does not exist, say so plainly.\n"
    "  4. A measurable HORIZON with NON-overlapping trades (so observations are "
    "independent and the sample isn't inflated).\n"
    "  5. An explicit FALSIFIER: the concrete result that would prove the claim wrong.\n\n"
    "Be HONEST and skeptical. NEVER claim an idea 'works', 'is profitable', or "
    "'makes money' — you cannot know that; the backtester decides. Your job is to "
    "SHARPEN a rough idea into something testable, not to hype it. Surface look-ahead "
    "bias, overlapping-window inflation, missing data, and cost problems when you see "
    "them. Keep replies CONCISE and Socratic: ask the one clarifying question that "
    "moves the idea closest to being testable. When the idea already meets the "
    "criteria, say so and suggest the user click 'Prepare Hypothesis'."
)

_PREPARE_INSTRUCTION = (
    "The conversation above is a brainstorm about a trading-strategy idea. Rewrite the "
    "WHOLE discussion into ONE clean, self-contained, falsifiable thesis that a "
    "downstream extractor can parse. Use plain prose / markdown — NO JSON, NO "
    "conversational framing, NO meta commentary. Resolve ambiguity using the best "
    "interpretation supported by the conversation. Include exactly these labeled "
    "sections:\n"
    "  Claim: one sentence, a single falsifiable directional prediction.\n"
    "  Mechanism: why the signal should lead the forward return, with no look-ahead.\n"
    "  Scope: the assets / universe it applies to.\n"
    "  Horizon: the forward window, with non-overlapping trades.\n"
    "  Data: the specific time series it needs.\n"
    "  Falsifier: the concrete result that would prove the claim wrong.\n"
    "Output ONLY the thesis."
)


def retrieve_corpus_context(convo: list[dict]) -> str:
    """Return discovery-safe corpus context for the recent user turns. Fails soft."""
    try:
        recent = [
            str(m.get("content") or "").strip()
            for m in (convo or [])
            if isinstance(m, dict) and m.get("role") == "user"
            and str(m.get("content") or "").strip()
        ][-3:]
        query = "\n".join(recent).strip()
        if not query:
            return ""
        from penrose import retrieval

        nodes = retrieval.retrieve(query, k=6, hops=1)
        return retrieval.format_context(nodes)
    except Exception:  # noqa: BLE001
        return ""


def _persist_chat(session_id, messages) -> None:
    """Save the running conversation to chat_log/<session>.json (local only; never raises)."""
    if not session_id:
        return
    try:
        sid = re.sub(r"[^A-Za-z0-9._-]+", "-", str(session_id)).strip("-")[:80] or "session"
        CHAT_LOG.mkdir(parents=True, exist_ok=True)
        payload = {"session_id": sid,
                   "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "messages": [{"role": m.get("role"), "content": m.get("content", "")}
                                for m in messages if isinstance(m, dict)]}
        (CHAT_LOG / f"{sid}.json").write_text(json.dumps(payload, indent=2, default=str))
    except Exception:  # noqa: BLE001
        pass


def extract_paper_text(filename: str, content_b64: str, max_chars: int = 6000) -> dict:
    """Claude-style attach: extract a paper's text for IN-CONVERSATION discussion. Does NOT queue
    to inbox; returns truncated text the assistant can read. Fails soft."""
    import tempfile
    name = os.path.basename(filename or "").strip() or "paper"
    try:
        data = base64.b64decode(content_b64 or "", validate=True)
    except (binascii.Error, ValueError):
        return {"ok": False, "error": "invalid base64 content"}
    if len(data) > 60 * 1024 * 1024:
        return {"ok": False, "error": "file too large (>60MB)"}
    suffix = ".md" if name.lower().endswith((".md", ".txt")) else ".pdf"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        from penrose.pipeline import p1_ingest
        src = p1_ingest.sanitize(tmp, source_id="chat-attach")
        text = (getattr(src, "text", "") or "").strip()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"could not read paper: {e}"}
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if not text:
        return {"ok": False, "error": "no extractable text (scanned/image PDF?)"}
    return {"ok": True, "title": name, "chars": len(text),
            "truncated": len(text) > max_chars, "text": text[:max_chars]}


def chat_reply(messages: list[dict], session_id=None) -> dict:
    """Continue the brainstorm conversation. Never raises — fails soft. Persists to chat_log/."""
    try:
        from penrose import llm
        convo = [m for m in (messages or [])
                 if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                 and (m.get("content") or "").strip()]
        if not convo:
            return {"ok": False, "error": "no conversation provided"}
        ctx = retrieve_corpus_context(convo)
        full = [{"role": "system", "content": _CHAT_SYSTEM}]
        if ctx:
            full.append({"role": "system", "content":
                         "Relevant prior Penrose findings (committed, discovery-safe - "
                         "for grounding only, do not treat as proof):\n" + ctx})
        full += convo
        resp = llm.call("chat_assistant", full, temperature=0.4, timeout=120)
        text = (resp.text or "").strip()
        if not text:
            return {"ok": False, "error": "assistant returned an empty reply"}
        _persist_chat(session_id, convo + [{"role": "assistant", "content": text}])
        return {"ok": True, "reply": text, "session_id": session_id}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def prepare_hypothesis(messages: list[dict]) -> dict:
    """Rewrite the whole conversation into one clean, falsifiable, ingestable
    thesis. Returns the text for the user to confirm — does NOT submit. Fails soft."""
    try:
        from penrose import llm
        convo = [m for m in (messages or [])
                 if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                 and (m.get("content") or "").strip()]
        if not convo:
            return {"ok": False, "error": "no conversation to prepare"}
        full = ([{"role": "system", "content": _CHAT_SYSTEM}] + convo +
                [{"role": "user", "content": _PREPARE_INSTRUCTION}])
        resp = llm.call("chat_assistant", full, temperature=0.2, timeout=120)
        text = (resp.text or "").strip()
        if not text:
            return {"ok": False, "error": "could not draft a hypothesis (empty reply)"}
        return {"ok": True, "hypothesis": text}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------------- #
# Submit: write the refined thesis to inbox/
# --------------------------------------------------------------------------- #
def write_to_inbox(thesis_text: str, source_label: str = "chat") -> dict:
    """Write a thesis to inbox/ for the next scan cadence to pick up. Returns
    the path + content hash. No brain/archive/decision writes."""
    text = (thesis_text or "").strip()
    if not text:
        return {"ok": False, "error": "empty thesis"}

    inbox = PENROSE_ROOT / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"{ts}-{source_label}-{digest}.txt"
    path = inbox / fname

    body = f"# penrose thesis (source: {source_label})\n# submitted: {ts}\n# sha: {digest}\n\n{text}\n"
    path.write_text(body)
    return {"ok": True, "path": str(path), "sha": digest, "submitted_at": ts,
            "note": "queued in inbox/; next scan will ingest via P1"}


def save_paper_to_inbox(filename: str, content_b64: str) -> dict:
    """Save an uploaded PDF to inbox/ for the next scan to ingest. inbox/ is a queue,
    not state — no brain/archive/decision writes. Validates it's a real PDF."""
    name = os.path.basename(filename or "").strip()
    if not name.lower().endswith(".pdf"):
        return {"ok": False, "error": "only .pdf files are accepted"}
    # sanitize to a safe filename
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "paper.pdf"
    try:
        data = base64.b64decode(content_b64 or "", validate=True)
    except (binascii.Error, ValueError):
        return {"ok": False, "error": "invalid base64 content"}
    if not data[:5] == b"%PDF-":
        return {"ok": False, "error": "not a valid PDF (missing %PDF header)"}
    if len(data) > 60 * 1024 * 1024:
        return {"ok": False, "error": "file too large (>60MB)"}
    inbox = PENROSE_ROOT / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / safe
    if path.exists():  # don't clobber; suffix with a short hash
        digest = hashlib.sha256(data).hexdigest()[:6]
        path = inbox / f"{path.stem}-{digest}.pdf"
    path.write_bytes(data)
    return {"ok": True, "path": str(path.relative_to(PENROSE_ROOT)),
            "bytes": len(data), "note": "queued in inbox/; next scan ingests via P1"}


# --------------------------------------------------------------------------- #
# Submit: append a user enhancement/bug suggestion to feedback.jsonl
# --------------------------------------------------------------------------- #
def write_feedback(text: str, kind: str = "enhancement") -> dict:
    """Append a user suggestion to feedback.jsonl. Local-only append log the operator reviews —
    no brain/archive/decision writes, never enters the pipeline."""
    body = (text or "").strip()
    if len(body) < 8:
        return {"ok": False, "error": "please add a few more words"}
    body = body[:8000]
    kind = kind if kind in {"enhancement", "bug", "question"} else "enhancement"
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    digest = hashlib.sha256(body.encode()).hexdigest()[:8]
    row = {"ts": ts, "kind": kind, "sha": digest, "text": body}
    path = PENROSE_ROOT / "feedback.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return {"ok": True, "sha": digest, "submitted_at": ts, "note": "logged to feedback.jsonl"}


# --------------------------------------------------------------------------- #
# Handler glue — used by live_server.py's Handler to dispatch POSTs
# --------------------------------------------------------------------------- #
def handle_post(handler, path: str) -> bool:
    """Returns True if the path was handled here (and a response sent).

    Caller (live_server.do_POST) should set up CORS headers + token check first.
    """
    try:
        if path == "/preflight":
            payload = _read_json_body(handler)
            result = run_preflight(payload.get("thesis", ""))
            handler._serve(json.dumps(result, default=str).encode(),
                           "application/json; charset=utf-8")
            return True
        if path == "/chat":
            payload = _read_json_body(handler, max_bytes=2 * 1024 * 1024)  # may carry attached text
            result = chat_reply(payload.get("messages", []), payload.get("session_id"))
            handler._serve(json.dumps(result, default=str).encode(),
                           "application/json; charset=utf-8")
            return True
        if path == "/attach-paper":
            payload = _read_json_body(handler, max_bytes=90 * 1024 * 1024)  # PDFs (base64)
            result = extract_paper_text(payload.get("filename", ""),
                                        payload.get("content_b64", ""))
            handler._serve(json.dumps(result, default=str).encode(),
                           "application/json; charset=utf-8")
            return True
        if path == "/prepare-hypothesis":
            payload = _read_json_body(handler)
            result = prepare_hypothesis(payload.get("messages", []))
            handler._serve(json.dumps(result, default=str).encode(),
                           "application/json; charset=utf-8")
            return True
        if path == "/submit-claim":
            payload = _read_json_body(handler)
            result = write_to_inbox(payload.get("thesis", ""),
                                    payload.get("source", "chat"))
            handler._serve(json.dumps(result, default=str).encode(),
                           "application/json; charset=utf-8")
            return True
        if path == "/upload-paper":
            payload = _read_json_body(handler, max_bytes=90 * 1024 * 1024)  # PDFs (base64)
            result = save_paper_to_inbox(payload.get("filename", ""),
                                         payload.get("content_b64", ""))
            handler._serve(json.dumps(result, default=str).encode(),
                           "application/json; charset=utf-8")
            return True
        if path == "/submit-feedback":
            payload = _read_json_body(handler)
            result = write_feedback(payload.get("text", ""), payload.get("kind", "enhancement"))
            handler._serve(json.dumps(result, default=str).encode(),
                           "application/json; charset=utf-8")
            return True
        if path == "/dash-token":
            # client requests the token (only same-origin allowed by _check_origin)
            handler._serve(json.dumps({"token": _ensure_token()}).encode(),
                           "application/json; charset=utf-8")
            return True
    except ValueError as e:
        handler.send_error(400, str(e))
        return True
    except Exception as e:  # noqa: BLE001
        handler.send_error(500, f"preflight/submit failed: {e}")
        return True
    return False
