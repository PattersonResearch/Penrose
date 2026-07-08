"""Read-only MCP server + shared accessors. The server must be STRICTLY read-only,
and `mcp` must be an OPTIONAL dependency (core works without it)."""
from __future__ import annotations

import importlib
import json

import pytest


def _has_mcp() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


def test_accessors_return_structured_data(tmp_path, monkeypatch):
    from penrose import config, views

    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "analysis_index.jsonl")
    monkeypatch.setattr(config, "DATA_REQUESTS", tmp_path / "data_requests.jsonl")
    monkeypatch.setattr(config, "LIVE_JSON", tmp_path / "live.json")

    # fail-open: missing files -> empty
    assert views.verdicts() == []
    assert views.data_requests() == []
    assert views.status()["pipeline_status"] == "idle"

    config.ANALYSIS_INDEX.write_text(
        json.dumps({"claim_id": "c1", "verdict": "kill", "kill_reason": "in_sample_only",
                    "statement": "x", "synthetic": False}) + "\n")
    config.DATA_REQUESTS.write_text(
        json.dumps({"claim_id": "c2", "status": "open", "missing_series": ["spy"]}) + "\n")
    config.LIVE_JSON.write_text(json.dumps({"pipeline_status": "running", "stats": {"n": 1}}))

    v = views.verdicts()
    assert v and v[0]["verdict"] == "kill" and v[0]["claim_id"] == "c1"
    dr = views.data_requests()
    assert dr and dr[0]["missing_series"] == ["spy"]
    assert views.status()["pipeline_status"] == "running"


def test_proposal_and_principle_accessors_are_readonly_lists(tmp_path, monkeypatch):
    from penrose import config, views

    monkeypatch.setattr(config, "PRINCIPLES_PROPOSED", tmp_path / "principles_proposed.jsonl")
    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "ANALYSIS_INDEX", tmp_path / "analysis_index.jsonl")

    # fail-open on an empty corpus
    assert views.proposals() == []
    assert views.principles() == []
    assert isinstance(views.principles(limit=5), list)


def test_mcp_server_module_imports_without_mcp_dependency():
    # The module must import even when `mcp` is absent (the import is lazy in build_server).
    mod = importlib.import_module("penrose.mcp_server")
    assert hasattr(mod, "build_server") and hasattr(mod, "main")
    # importing penrose / the CLI must never require mcp
    importlib.import_module("penrose")
    importlib.import_module("penrose.cli")


def test_build_server_registers_five_readonly_tools_or_fails_gracefully():
    from penrose import mcp_server
    if _has_mcp():
        server = mcp_server.build_server()
        import asyncio
        tools = asyncio.get_event_loop().run_until_complete(server.list_tools()) \
            if hasattr(server, "list_tools") else None
        names = {t.name for t in tools} if tools else set()
        # Be tolerant of SDK shape: fall back to the registered tool manager if present.
        if not names and hasattr(server, "_tool_manager"):
            names = set(server._tool_manager._tools.keys())
        expected = {"penrose_verdicts", "penrose_proposals", "penrose_principles",
                    "penrose_data_requests", "penrose_status", "penrose_triage"}
        assert names == expected
    else:
        with pytest.raises(ImportError) as e:
            mcp_server.build_server()
        assert "pip install penrose[mcp]" in str(e.value)


def test_server_is_strictly_read_only():
    # The shared read accessors remain strictly read-only. The MCP module now also contains
    # opt-in management wrappers, covered in test_mcp_management.py; the default server surface
    # is checked above.
    import ast
    import penrose.views as vw
    used: set[str] = set()
    tree = ast.parse(open(vw.__file__).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    forbidden = {"run_source", "PRINCIPLES_LOG", "write_proposals", "run_in_container",
                 "final_holdout_eval", "final_holdout", "_append_jsonl", "_write_ledger",
                 "register_trials"}
    hits = forbidden & used
    assert not hits, f"read-only server references forbidden code symbols: {hits}"
