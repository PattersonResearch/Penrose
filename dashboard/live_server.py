#!/usr/bin/env python3
"""
penrose v1 dashboard — READ-ONLY live server.

Serves dashboard/index.html with `window.__LIVE__` injected from three on-disk
data sources at request time. It NEVER writes anything anywhere.

Architecture:
  - index.html ships with realistic placeholder data and renders standalone.
  - This server overlays real values one key at a time (per-key fallback): a
    failing/missing source is simply omitted, so its placeholder survives and
    the page never blanks.
  - stdlib only (http.server). No framework, no build step, no node_modules.
  - Caches built HTML and refreshes every 60s on a daemon thread.

Sources (all read-only):
  1. dashboard/live.json        -> stats, decisions, provenance, synthetic flag
  2. review_queue.jsonl         -> Action Required queue (one JSON obj per line)
  3. reports/2604.01431v1.md    -> Sources-tab report markdown

Run:  python3 live_server.py        (default port 8077)
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# F15 endpoints (preflight + submit + token) — writes are scoped to inbox/ only
sys_imported = False
try:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import write_api  # noqa: E402
    sys_imported = True
except Exception as e:  # noqa: BLE001
    write_api = None
    print("[penrose] write_api unavailable (F15 disabled):", e)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # penrose/
INDEX = os.path.join(HERE, "index.html")
LIVE_JSON = os.path.join(HERE, "live.json")
REVIEW_QUEUE = os.path.join(ROOT, "review_queue.jsonl")
REPORT_MD = os.path.join(ROOT, "reports", "2604.01431v1.md")

PORT = int(os.environ.get("PENROSE_PORT", "8077"))
REFRESH_SECS = 60

_cache_lock = threading.Lock()
_cache_html = b""


# ---------------------------------------------------------------- source readers
def _read_live_json():
    """live.json -> the bulk of window.__LIVE__ (per-key)."""
    with open(LIVE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k in ("updated_at", "status_badge", "synthetic_warning",
              "data_provenance", "stats", "decisions",
              "pipeline_status", "source_title", "data_requests", "notice"):
        if k in d:
            out[k] = d[k]
    return out


def _read_recent_runs(limit: int = 8):
    """runs.jsonl -> compact summaries of the last N papers (newest first) for the
    Overview history: title, when, verdict counts, off-domain flag."""
    runs_path = os.path.join(ROOT, "runs.jsonl")
    if not os.path.exists(runs_path):
        return []
    out = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts = {}
            for d in r.get("decisions", []):
                v = d.get("verdict", "?")
                counts[v] = counts.get(v, 0) + 1
            out.append({
                "source_id": r.get("source_id", "?"),
                "title": r.get("source_title", r.get("source_id", "?")),
                "run_at": r.get("run_at", ""),
                "off_domain": bool(r.get("off_domain")),
                "relevance_reason": (r.get("relevance") or {}).get("reason", ""),
                "claims": r.get("claims_extracted", len(r.get("decisions", []))),
                "counts": counts,
            })
    return list(reversed(out))[:limit]


def _read_analysis_reports(limit: int = 30):
    """analysis_index.jsonl -> recent backtested outcomes (newest first) for the
    Analysis Reports page: title, verdict, metrics, chart filename."""
    p = os.path.join(ROOT, "reports", "analysis_index.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(out))[:limit]


def _read_review_queue():
    """review_queue.jsonl -> list of proposal objects for Action Required."""
    items = []
    with open(REVIEW_QUEUE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a bad line; never blank the whole queue
    return items


def _read_report_md():
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        return f.read()


def _source_title(report_md):
    """Pull a human title from the report's bold line, fallback to id."""
    for line in report_md.splitlines():
        s = line.strip()
        if s.startswith("**") and s.endswith("**") and len(s) > 4:
            return "2604.01431v1 — " + s.strip("*").strip()
    return "2604.01431v1 — Do Prediction Markets Forecast Cryptocurrency Volatility?"


def build_live():
    """Assemble window.__LIVE__. Each source is guarded independently so one
    broken source never blanks the page (its placeholder survives)."""
    live = {}

    try:
        live.update(_read_live_json())
    except Exception as e:  # noqa: BLE001
        print("[penrose] live.json unavailable:", e)

    try:
        live["review_queue"] = _read_review_queue()
    except Exception as e:  # noqa: BLE001
        print("[penrose] review_queue.jsonl unavailable:", e)

    try:
        live["recent_runs"] = _read_recent_runs()
    except Exception as e:  # noqa: BLE001
        print("[penrose] runs.jsonl unavailable:", e)

    try:
        live["analysis_reports"] = _read_analysis_reports()
    except Exception as e:  # noqa: BLE001
        print("[penrose] analysis_index.jsonl unavailable:", e)

    try:
        md = _read_report_md()
        live["report_md"] = md
        live["source_title"] = _source_title(md)
    except Exception as e:  # noqa: BLE001
        print("[penrose] report markdown unavailable:", e)

    # live per-stage progress (separate file, written frequently DURING a run)
    try:
        pjson = os.path.join(HERE, "progress.json")
        if os.path.exists(pjson):
            with open(pjson, "r", encoding="utf-8") as f:
                live["progress"] = json.load(f)
    except Exception as e:  # noqa: BLE001
        print("[penrose] progress.json unavailable:", e)

    # brain connection-discovery (advisory; written by scripts/brain_connections.py)
    try:
        cjson = os.path.join(HERE, "connections.json")
        if os.path.exists(cjson):
            with open(cjson, "r", encoding="utf-8") as f:
                live["connections"] = json.load(f)
    except Exception as e:  # noqa: BLE001
        print("[penrose] connections.json unavailable:", e)

    # self-calibration controls (refreshed from the calib-* battery; see scripts/calibration_*.py)
    try:
        kjson = os.path.join(HERE, "calibration.json")
        if os.path.exists(kjson):
            with open(kjson, "r", encoding="utf-8") as f:
                live["calibration"] = json.load(f)
    except Exception as e:  # noqa: BLE001
        print("[penrose] calibration.json unavailable:", e)

    return live


def build_html():
    """Read index.html and inject the live blob just before the boot <script>."""
    with open(INDEX, "r", encoding="utf-8") as f:
        raw = f.read()

    live = build_live()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    inject = ("<script>window.__LIVE__ = "
              + json.dumps(live, default=float)
              + "; window.__LIVE_STAMP__ = " + json.dumps(stamp) + ";</script>\n")

    # Inject before the first inline boot script. The placeholder data + per-key
    # overlay live in that same script, and it reads window.__LIVE__ at parse
    # time, so __LIVE__ must be defined first.
    marker = "<script>\n/* ============================ placeholder data"
    if marker in raw:
        html = raw.replace(marker, inject + marker, 1)
    else:
        html = raw.replace("</head>", inject + "</head>", 1)
    return html.encode("utf-8")


def refresh_loop():
    global _cache_html
    while True:
        try:
            html = build_html()
            with _cache_lock:
                _cache_html = html
        except Exception as e:  # noqa: BLE001
            print("[penrose] refresh failed (keeping last good):", e)
        time.sleep(REFRESH_SECS)


# ---------------------------------------------------------------- http handler
class Handler(BaseHTTPRequestHandler):
    def _serve(self, body, ctype="text/html; charset=utf-8"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "*"))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with _cache_lock:
                body = _cache_html
            if not body:  # cold start before first cache build
                try:
                    body = build_html()
                except Exception as e:  # noqa: BLE001
                    self.send_error(500, "build failed: %s" % e)
                    return
            self._serve(body)
        elif path == "/live.json":
            # convenience: expose the assembled live blob (read-only)
            body = json.dumps(build_live(), default=float, indent=2).encode("utf-8")
            self._serve(body, "application/json; charset=utf-8")
        elif path.startswith("/charts/"):
            # serve a backtest chart PNG (read-only, sandboxed to reports/charts)
            name = os.path.basename(path[len("/charts/"):])
            fp = os.path.join(ROOT, "reports", "charts", name)
            if name.endswith(".png") and os.path.exists(fp):
                with open(fp, "rb") as f:
                    self._serve(f.read(), "image/png")
            else:
                self.send_error(404, "chart not found")
        elif path.startswith("/assets/"):
            # serve a static dashboard asset (read-only, sandboxed to dashboard/assets)
            name = os.path.basename(path[len("/assets/"):])
            fp = os.path.join(HERE, "assets", name)
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            ctype = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                     "webp": "image/webp", "svg": "image/svg+xml"}.get(ext)
            if ctype and os.path.exists(fp):
                with open(fp, "rb") as f:
                    self._serve(f.read(), ctype)
            else:
                self.send_error(404, "asset not found")
        elif path.startswith("/report/"):
            # serve a run's report markdown (read-only, sandboxed to reports/)
            name = os.path.basename(path[len("/report/"):])
            fp = os.path.join(ROOT, "reports", name if name.endswith(".md") else name + ".md")
            if name and os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    self._serve(f.read().encode("utf-8"), "text/markdown; charset=utf-8")
            else:
                self.send_error(404, "report not found")
        elif path == "/healthz":
            self._serve(b"ok", "text/plain")
        elif path == "/dash-token" and write_api is not None:
            # client requests the per-launch token (same-origin only via Origin check)
            body = json.dumps({"token": write_api._ensure_token()}).encode()
            self._serve(body, "application/json; charset=utf-8")
        else:
            self.send_error(404, "not found")

    do_HEAD = do_GET

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if write_api is None:
            self.send_error(503, "write_api unavailable (F15 disabled)")
            return
        # Origin validation first (send_error writes its own status line).
        if not write_api._check_origin(self.headers):
            self.send_error(403, "origin not allowed")
            return
        # OPTIONS preflight — status line FIRST, then headers (correct ordering).
        if self.command == "OPTIONS":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "*"))
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Penrose-Token")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.end_headers()
            return
        # token check (skip for /dash-token itself)
        if path != "/dash-token":
            token = self.headers.get("X-Penrose-Token", "")
            if not (token and token == write_api._ensure_token()):
                self.send_error(401, "missing or invalid X-Penrose-Token")
                return
        # dispatch to write_api (it responds via self._serve, which carries CORS)
        handled = write_api.handle_post(self, path)
        if not handled:
            self.send_error(404, "not found")

    def do_OPTIONS(self):
        self.do_POST()

    def log_message(self, fmt, *args):
        pass  # quiet


def main():
    global _cache_html
    # warm the cache once at boot so the first request is instant
    try:
        _cache_html = build_html()
    except Exception as e:  # noqa: BLE001
        print("[penrose] initial build failed:", e)

    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:%d/" % PORT
    print("=" * 60)
    print("  penrose v1 dashboard — READ-ONLY research engine + F15 inbox writes")
    print("  serving:  " + url)
    print("  sources:  live.json · review_queue.jsonl · reports/*.md")
    print("  refresh:  every %ds · writes go to inbox/ only (never brain/state)" % REFRESH_SECS)
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[penrose] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
