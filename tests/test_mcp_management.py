"""Opt-in MCP management surface.

Management tools may run guarded proposal/bookkeeping paths, but they must never
reach the human P9 commit surface.
"""
from __future__ import annotations

import ast
import importlib

import pytest


READ_ONLY = {
    "penrose_verdicts",
    "penrose_proposals",
    "penrose_principles",
    "penrose_data_requests",
    "penrose_status",
}
MANAGEMENT = {
    "penrose_fetch_verdict",
    "penrose_register_cohort",
    "penrose_run_claim",
}


def _has_mcp() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


def _tool_names(server) -> set[str]:
    if hasattr(server, "_tool_manager"):
        return set(server._tool_manager._tools.keys())
    if hasattr(server, "list_tools"):
        import asyncio
        tools = asyncio.get_event_loop().run_until_complete(server.list_tools())
        return {t.name for t in tools}
    return set()


@pytest.mark.skipif(not _has_mcp(), reason="mcp optional extra is not installed")
def test_default_build_server_exposes_exactly_readonly_tools():
    from penrose import mcp_server

    names = _tool_names(mcp_server.build_server())

    assert names == READ_ONLY
    assert not (MANAGEMENT & names)


@pytest.mark.skipif(not _has_mcp(), reason="mcp optional extra is not installed")
def test_management_build_server_adds_exactly_guarded_tools():
    from penrose import mcp_server

    names = _tool_names(mcp_server.build_server(management=True))

    assert names == READ_ONLY | MANAGEMENT
    assert not any("approve" in n or "promote" in n for n in names)


def test_mcp_server_never_references_p9_commit_surface():
    import penrose.mcp_server as ms

    source = open(ms.__file__).read()
    assert "PromotionClient" not in source
    assert "confirm_run" not in source

    tree = ast.parse(source)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)

    forbidden_exact = {
        "PromotionClient",
        "confirm_run",
        "cmd_approve",
        "put_atom",
        "_put",
        "_link",
        "write_proposals",
        "PRINCIPLES_LOG",
        "final_holdout_eval",
    }
    assert not (forbidden_exact & used)
    assert not [name for name in used if "promote" in name.lower()]
    assert not [name for name in used if "approve" in name.lower()]


def test_run_claim_routes_through_run_source_and_returns_proposals(monkeypatch):
    ms = importlib.import_module("penrose.mcp_server")
    called = {}

    def stub_run_source(path, **kwargs):
        called["path"] = path
        called["kwargs"] = kwargs
        return {
            "decisions": [
                {"claim_id": "c1", "verdict": "watch"},
                {"claim_id": "c2", "verdict": "kill", "kill_reason": "fragile"},
            ],
            "claims": [{"claim_id": "ignored"}],
        }

    monkeypatch.setattr(ms, "views", object())
    monkeypatch.setattr(ms, "run_source", stub_run_source)
    monkeypatch.setattr(ms, "register_trials", lambda rows: None)

    out = ms._run_claim(claim={"claim_id": "c1", "statement": "test claim"}, max_claims=1)

    assert out["ok"] is True
    assert called["kwargs"]["claims_override"][0].claim_id == "c1"
    assert called["kwargs"]["max_claims"] == 1
    assert [p["claim_id"] for p in out["proposals"]] == ["c1", "c2"]
    assert {p["status"] for p in out["proposals"]} == {"proposed"}
    assert all(p.get("status") != "approved" for p in out["proposals"])


def test_register_cohort_validation_and_ledger_only(monkeypatch):
    ms = importlib.import_module("penrose.mcp_server")
    calls = []

    monkeypatch.setattr(ms, "views", object())
    monkeypatch.setattr(ms, "run_source", lambda *a, **k: {})
    monkeypatch.setattr(ms, "register_trials", lambda rows: calls.append(rows))

    assert ms._register_cohort("", "family", 1)["ok"] is False
    assert ms._register_cohort("cohort", "", 1)["ok"] is False
    assert ms._register_cohort("cohort", "family", 0)["ok"] is False
    assert ms._register_cohort("cohort", "family", 1, "not-list")["ok"] is False
    # audit fix: a huge denominator would permanently over-deflate the family -> capped, no ledger write
    assert ms._register_cohort("cohort", "family", 10**12)["ok"] is False
    assert calls == [], "a rejected cohort must not touch the ledger"

    out = ms._register_cohort("cohort", "family", 3, ["s1", "s2"])

    assert out == {"ok": True, "registered": 3, "cohort_id": "cohort"}
    assert calls == [[
        {
            "strategy": "s1",
            "family": "family",
            "generation_source": "mcp_management",
            "search_cohort_id": "cohort",
            "search_denominator": 3,
        },
        {
            "strategy": "s2",
            "family": "family",
            "generation_source": "mcp_management",
            "search_cohort_id": "cohort",
            "search_denominator": 3,
        },
    ]]


def test_run_claim_refuses_paper_path_outside_inbox(monkeypatch):
    ms = importlib.import_module("penrose.mcp_server")
    called = []
    monkeypatch.setattr(ms, "run_source", lambda *a, **k: called.append(a) or {})
    # an out-of-inbox path (arbitrary-file-read primitive) must be refused BEFORE run_source
    out = ms._run_claim(paper_path="/etc/passwd")
    assert out["ok"] is False and "inbox" in out["error"].lower()
    assert called == [], "run_source must not be invoked for a path outside the inbox"


def test_management_tools_fail_gracefully(monkeypatch):
    ms = importlib.import_module("penrose.mcp_server")

    class BadViews:
        @staticmethod
        def verdicts(limit):
            raise RuntimeError("boom")

    def bad_run_source(*args, **kwargs):
        raise RuntimeError("run boom")

    def bad_register_trials(rows):
        raise RuntimeError("ledger boom")

    monkeypatch.setattr(ms, "views", BadViews)
    monkeypatch.setattr(ms, "run_source", bad_run_source)
    monkeypatch.setattr(ms, "register_trials", bad_register_trials)

    for out in (
        ms._single_verdict("c1"),
        ms._register_cohort("cohort", "family", 1),
        ms._run_claim(claim={"claim_id": "c1", "statement": "x"}, max_claims=1),
    ):
        assert out["ok"] is False
        assert isinstance(out["error"], str) and out["error"]

    assert ms._single_verdict("")["ok"] is False
    assert ms._run_claim(paper_path=None, claim=None)["ok"] is False
    assert ms._run_claim(paper_path="paper.pdf", claim={"statement": "x"})["ok"] is False
    assert ms._run_claim(claim={"statement": "x"}, max_claims=999)["ok"] is False
