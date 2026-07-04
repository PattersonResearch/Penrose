"""Container sandbox for executing UNTRUSTED (auto-implemented, model-written) module code.

Three swarm passes proved that scanning LLM-written Python in-process always has escape paths
(reflection, stdlib gadgets, look-ahead). The only categorical fix is not running it in penrose's
process. So **auto-generated modules run in a Docker container** with: no network, read-only root
fs, no host filesystem beyond a bind-mounted work dir, no secrets in the env, and memory / CPU /
pids / wall-clock limits, as a non-root user. Operator-written (trusted) modules run in-process.

Requirement (operator's call): if Docker is unavailable, auto-implementation is DISABLED — we
NEVER fall back to unsandboxed exec of model-written code.

IPC is pickle-free in BOTH directions of trust that matter: the untrusted child returns results
only as parquet + JSON, so it cannot RCE the parent via deserialization.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

from .. import config

IMAGE = "penrose-sandbox:py311"
_HERE = Path(__file__).resolve().parent
# Work dir lives UNDER the repo (~/Development/...) so Docker Desktop on macOS can bind-mount it;
# /var/folders tmp is not shared by default and would make the mount fail.
_WORK_ROOT = config.ROOT / ".sandbox_work"


def docker_available() -> bool:
    """True iff the docker CLI + a running daemon are reachable."""
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=12).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def ensure_image() -> bool:
    """Build the sandbox image if missing (one-time; the build is the only step that needs network)."""
    try:
        if subprocess.run(["docker", "image", "inspect", IMAGE],
                          capture_output=True, timeout=15).returncode == 0:
            return True
        d = _HERE / "_sandbox_docker"
        d.mkdir(exist_ok=True)
        (d / "Dockerfile").write_text(
            "FROM python:3.11-slim\n"
            "RUN pip install --no-cache-dir numpy pandas pyarrow\n"
            "RUN useradd -m sandbox\nUSER sandbox\n")
        return subprocess.run(["docker", "build", "-t", IMAGE, str(d)],
                              capture_output=True, timeout=900, text=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


# Self-contained runner executed INSIDE the container. Rebuilds a shim bundle (no penrose import),
# imports the module, calls run(), writes results as parquet/JSON. No network, no host fs.
_RUNNER = r'''
import json, importlib.util
import pandas as pd
IN, OUT = "/work/in", "/work/out"
man = json.load(open(IN + "/manifest.json"))

class _S:
    def __init__(self, data, prov):
        self.data, self.available, self.provenance = data, True, prov
class _Bundle:
    def __init__(self): self.series = {}
    def get(self, name): return self.series.get(name)

b = _Bundle()
for nm, meta in man["series"].items():
    df = pd.read_parquet(IN + "/series__%d.parquet" % meta["idx"])
    b.series[nm] = _S(pd.Series(df["v"].values, index=pd.to_datetime(df["t"])), meta.get("provenance", ""))

cj = json.load(open(IN + "/claim.json"))
claim = type("Claim", (), {})()
for k, v in cj.items():
    setattr(claim, k, v)

spec = importlib.util.spec_from_file_location("usermod", "/mod/impl.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
res = m.run(b, claim, man["cost_frac"])

out = {"ok": bool(isinstance(res, dict) and res.get("ok"))}
if out["ok"]:
    for key in ("net", "positions", "payoff", "position_signed"):
        v = res.get(key)
        if v is not None:
            v = pd.Series(v)
            pd.DataFrame({"t": pd.Series(v.index).astype(str), "v": v.values}).to_parquet(OUT + "/%s.parquet" % key)
    wf = res.get("wf_frame")
    if wf is not None:
        pd.DataFrame(wf).to_parquet(OUT + "/wf_frame.parquet")
    out["bars_per_year"] = float(res.get("bars_per_year", 0) or 0)
    out["n_trades"] = int(res.get("n_trades", 0) or 0)
else:
    out["reason"] = str((res or {}).get("reason", "module returned not-ok"))[:300] if isinstance(res, dict) else "module returned non-dict"
json.dump(out, open(OUT + "/result.json", "w"))
'''


def run_in_container(module_path: str, bundle, claim, cost_frac: float, timeout: int = 120) -> dict:
    """Execute module.run() for an untrusted module inside the sandbox container.

    Returns the contract result dict (with net/positions etc. as pd.Series), or
    {"ok": False, "reason": "..."} on ANY failure. Never raises into the caller,
    never runs the module in this process."""
    if not docker_available() or not ensure_image():
        return {"ok": False, "reason": "engine_error: docker sandbox unavailable (auto-impl requires Docker)"}
    _WORK_ROOT.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sbx_", dir=_WORK_ROOT))
    try:
        (work / "in").mkdir(); (work / "out").mkdir()
        man = {"series": {}, "cost_frac": float(cost_frac)}
        for i, (nm, v) in enumerate(bundle.series.items()):
            data = getattr(v, "data", None)
            if data is None or not getattr(v, "available", False):
                continue
            pd.DataFrame({"t": pd.Series(data.index).astype(str), "v": data.values}).to_parquet(
                work / f"in/series__{i}.parquet")
            man["series"][nm] = {"idx": i, "provenance": getattr(v, "provenance", "")}
        json.dump(man, open(work / "in/manifest.json", "w"))
        cj = asdict(claim) if is_dataclass(claim) else {
            k: getattr(claim, k) for k in vars(claim) if isinstance(getattr(claim, k), (str, int, float))}
        json.dump(cj, open(work / "in/claim.json", "w"), default=str)
        (work / "runner.py").write_text(_RUNNER)
        cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "512m", "--cpus", "1",
               "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp",
               "-v", f"{work}:/work", "-v", f"{Path(module_path).parent}:/mod:ro",
               IMAGE, "python", "/work/runner.py"]
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        rp = work / "out/result.json"
        if not rp.exists():
            return {"ok": False, "reason": f"engine_error: sandbox run failed rc={r.returncode}: {(r.stderr or '')[:200]}"}
        out = json.load(open(rp))
        if not out.get("ok"):
            return {"ok": False, "reason": out.get("reason", "module returned not-ok in sandbox (no reason given)")}
        res = {"ok": True, "bars_per_year": out.get("bars_per_year"), "n_trades": out.get("n_trades")}
        for key in ("net", "positions", "payoff", "position_signed"):
            p = work / f"out/{key}.parquet"
            if p.exists():
                d = pd.read_parquet(p)
                res[key] = pd.Series(d["v"].values, index=pd.to_datetime(d["t"]))
        wfp = work / "out/wf_frame.parquet"
        if wfp.exists():
            res["wf_frame"] = pd.read_parquet(wfp)
        return res
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "engine_error: sandbox timeout"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"engine_error: sandbox error {type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
