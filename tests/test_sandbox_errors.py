import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from penrose.brain import Claim
from penrose.data.contract import DataBundle


def _module_path(tmp_path) -> Path:
    mod = tmp_path / "module"
    mod.mkdir()
    impl = mod / "impl.py"
    impl.write_text("def run(bundle, claim, cost_frac):\n    return {'ok': False}\n")
    return impl


def _claim():
    return Claim("sandbox", "sandbox claim", "", "", "", "unit", "span", "")


def _work_from_cmd(cmd) -> Path:
    for i, part in enumerate(cmd):
        if part == "-v" and i + 1 < len(cmd) and cmd[i + 1].endswith(":/work"):
            return Path(cmd[i + 1].split(":", 1)[0])
    raise AssertionError("sandbox work volume missing")


def test_sandbox_rc_failure_is_engine_error(tmp_path, monkeypatch):
    from penrose.pipeline import sandbox

    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "ensure_image", lambda: True)
    monkeypatch.setattr(sandbox.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stderr="crashed"))

    res = sandbox.run_in_container(str(_module_path(tmp_path)), DataBundle(), _claim(), 0.0)

    assert res["ok"] is False
    assert res["reason"].startswith("engine_error: sandbox run failed rc=1: crashed")


def test_sandbox_timeout_is_engine_error(tmp_path, monkeypatch):
    from penrose.pipeline import sandbox

    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=1)

    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "ensure_image", lambda: True)
    monkeypatch.setattr(sandbox.subprocess, "run", raise_timeout)

    res = sandbox.run_in_container(str(_module_path(tmp_path)), DataBundle(), _claim(), 0.0)

    assert res == {"ok": False, "reason": "engine_error: sandbox timeout"}


def test_sandbox_module_data_unavailable_passes_through(tmp_path, monkeypatch):
    from penrose.pipeline import sandbox

    def write_data_reason(cmd, **kwargs):
        work = _work_from_cmd(cmd)
        (work / "out/result.json").write_text(json.dumps({
            "ok": False,
            "reason": "data_unavailable: eth_spot_daily",
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "ensure_image", lambda: True)
    monkeypatch.setattr(sandbox.subprocess, "run", write_data_reason)

    res = sandbox.run_in_container(str(_module_path(tmp_path)), DataBundle(), _claim(), 0.0)

    assert res == {"ok": False, "reason": "data_unavailable: eth_spot_daily"}


def test_sandbox_not_ok_without_reason_is_contract_error(tmp_path, monkeypatch):
    from penrose.pipeline import sandbox

    def write_no_reason(cmd, **kwargs):
        work = _work_from_cmd(cmd)
        (work / "out/result.json").write_text(json.dumps({"ok": False}))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "ensure_image", lambda: True)
    monkeypatch.setattr(sandbox.subprocess, "run", write_no_reason)

    res = sandbox.run_in_container(str(_module_path(tmp_path)), DataBundle(), _claim(), 0.0)

    assert res == {"ok": False, "reason": "module returned not-ok in sandbox (no reason given)"}
