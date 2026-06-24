from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _disable_local_embed(monkeypatch) -> None:
    from penrose import llm

    monkeypatch.setattr(llm, "embed_local_available", lambda: False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")


def test_put_get_roundtrip(tmp_path, monkeypatch):
    from penrose.brainstore import BrainStore

    _disable_local_embed(monkeypatch)
    store = BrainStore(tmp_path / "atoms.db", auto_rebuild=False)
    content = "---\ntype: atom\nkind: decision\ntrust: 0.7\n---\n\nFunding carries returns.\n"

    assert store._put("atoms/penrose/decision/d1", content) is True
    assert store.get("atoms/penrose/decision/d1") == content


def test_search_lexical_fallback(tmp_path, monkeypatch):
    from penrose.brainstore import BrainStore

    _disable_local_embed(monkeypatch)
    store = BrainStore(tmp_path / "atoms.db", auto_rebuild=False)
    store._put("atoms/penrose/decision/funding", "---\nkind: decision\n---\n\nFunding carry predicts BTC returns.\n")
    store._put("atoms/penrose/decision/weather", "---\nkind: decision\n---\n\nWeather affects crop yields.\n")

    assert store.search("btc funding carry", n=2).splitlines()[0].startswith(
        "atoms/penrose/decision/funding ::")


def test_search_includes_non_vectored_atoms_when_vectors_exist(tmp_path, monkeypatch):
    from penrose import llm
    from penrose.brainstore import BrainStore

    embed_enabled = {"value": False}
    monkeypatch.setattr(llm, "embed_local_available", lambda: embed_enabled["value"])
    monkeypatch.setattr(llm, "embed_local", lambda text: [1.0, 0.0])
    monkeypatch.setattr(llm, "cosine", lambda left, right: 1.0)

    store = BrainStore(tmp_path / "atoms.db", auto_rebuild=False)
    store._put("atoms/penrose/decision/no-vector", "---\nkind: decision\n---\n\nNeedlealpha only here.\n")

    embed_enabled["value"] = True
    store._put("atoms/penrose/decision/vector", "---\nkind: decision\n---\n\nUnrelated vectorized atom.\n")

    hits = store.search("needlealpha", n=2)

    assert "atoms/penrose/decision/no-vector" in hits


def test_search_lexical_fallback_for_nonpositive_vector_score(tmp_path, monkeypatch):
    from penrose import llm
    from penrose.brainstore import BrainStore

    monkeypatch.setattr(llm, "embed_local_available", lambda: True)
    monkeypatch.setattr(llm, "embed_local", lambda text: [1.0, 0.0])
    monkeypatch.setattr(llm, "cosine", lambda left, right: 0.0)

    store = BrainStore(tmp_path / "atoms.db", auto_rebuild=False)
    store._put("atoms/penrose/decision/vector-zero", "---\nkind: decision\n---\n\nNeedlebeta only here.\n")

    hits = store.search("needlebeta", n=1)

    assert "atoms/penrose/decision/vector-zero" in hits


def test_graph_traversal(tmp_path, monkeypatch):
    from penrose.brainstore import BrainStore

    _disable_local_embed(monkeypatch)
    store = BrainStore(tmp_path / "atoms.db", auto_rebuild=False)
    for name in ("a", "b", "c"):
        store._put(f"atoms/penrose/decision/{name}", f"---\nkind: decision\n---\n\n{name}\n")
    assert store._link("atoms/penrose/decision/a", "atoms/penrose/decision/b", "supports")
    assert store._link("atoms/penrose/decision/b", "atoms/penrose/decision/c", "supports")
    assert store._link("atoms/penrose/decision/c", "atoms/penrose/decision/a", "supports")

    out = store.graph("atoms/penrose/decision/a", depth=2)
    assert "atoms/penrose/decision/b" in out
    assert "atoms/penrose/decision/c" in out
    assert len(out.splitlines()) == len(set(out.splitlines()))


def test_reader_cannot_write():
    from penrose import brainstore
    from penrose.brain import BrainReader, PromotionClient
    from penrose.brainstore import BrainStore

    reader = BrainReader()
    assert not hasattr(reader, "put_atom")
    assert not hasattr(reader, "link")
    assert not hasattr(BrainStore, "put")
    assert not hasattr(BrainStore, "link")
    assert not hasattr(brainstore, "put")
    assert not hasattr(brainstore, "link")
    try:
        PromotionClient("")
    except PermissionError:
        pass
    else:
        raise AssertionError("PromotionClient accepted a missing approver")
    try:
        PromotionClient(" ")
    except PermissionError:
        pass
    else:
        raise AssertionError("PromotionClient accepted a blank approver")


def test_empty_flat_files_rebuilds_once_then_rebuilds_when_valid_rows_arrive(tmp_path, monkeypatch):
    from penrose import brainstore, config
    from penrose.brainstore import BrainStore

    _disable_local_embed(monkeypatch)
    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "CONCEPTS", tmp_path / "reports" / "concepts.jsonl")
    monkeypatch.setattr(config, "PRINCIPLES_LOG", tmp_path / "principles.jsonl")
    _write_jsonl(config.DECISIONS_LOG, [{"missing_id": "d1"}])
    _write_jsonl(config.CONCEPTS, [])
    _write_jsonl(config.PRINCIPLES_LOG, [])

    calls = {"count": 0}
    real_rebuild = brainstore.rebuild_from_flat_files

    def counting_rebuild_from_flat_files(**kwargs):
        calls["count"] += 1
        return real_rebuild(**kwargs)

    monkeypatch.setattr(brainstore, "rebuild_from_flat_files", counting_rebuild_from_flat_files)
    store = BrainStore(tmp_path / "atoms.db")

    assert store.search("anything", n=5) == ""
    assert store.list(n=5) == ""
    assert store.get("missing") is None
    assert calls["count"] == 1
    with store._connect() as conn:
        first_sig = conn.execute("SELECT value FROM _meta WHERE key = ?", ("flat_sig",)).fetchone()["value"]

    _write_jsonl(config.DECISIONS_LOG, [{
        "decision_id": "d1",
        "claim_id": "c1",
        "verdict": "kill",
        "kill_reason": "dedup",
        "rationale": "Funding carry predicts returns.",
        "metrics": {},
    }])

    hits = store.search("funding carry", n=5)

    assert "atoms/penrose/decision/d1" in hits
    assert "atoms/penrose/decision/d1" in store.list(n=5)
    assert calls["count"] == 2
    with store._connect() as conn:
        second_sig = conn.execute("SELECT value FROM _meta WHERE key = ?", ("flat_sig",)).fetchone()["value"]
    assert second_sig != first_sig
    assert store.get("atoms/penrose/decision/d1") is not None
    assert calls["count"] == 2


def test_rebuild_from_flat_files(tmp_path, monkeypatch):
    from penrose import brainstore, config
    from penrose.brainstore import BrainStore

    _disable_local_embed(monkeypatch)
    decisions = tmp_path / "decisions.jsonl"
    concepts = tmp_path / "reports" / "concepts.jsonl"
    principles = tmp_path / "principles.jsonl"
    db = tmp_path / "atoms.db"
    monkeypatch.setattr(config, "DECISIONS_LOG", decisions)
    monkeypatch.setattr(config, "CONCEPTS", concepts)
    monkeypatch.setattr(config, "PRINCIPLES_LOG", principles)
    _write_jsonl(decisions, [{
        "decision_id": "d1",
        "claim_id": "c1",
        "verdict": "kill",
        "kill_reason": "dedup",
        "rationale": "Funding duplicate.",
        "metrics": {},
    }])
    _write_jsonl(concepts, [{
        "concept_id": "o1",
        "source_claim_id": "c1",
        "statement": "Funding observation.",
        "mechanism": "carry",
    }])
    _write_jsonl(principles, [{
        "principle_id": "p1",
        "statement": "Repeated funding duplicates should be killed.",
        "supporting_kills": ["d1"],
        "applicable_strategy_classes": ["vol"],
        "n_observations": 1,
        "confidence": 0.6,
    }])

    first = brainstore.rebuild_from_flat_files(db_path=db)
    second = brainstore.rebuild_from_flat_files(db_path=db)
    store = BrainStore(db, auto_rebuild=False)

    assert first["decisions"] == second["decisions"] == 1
    assert "atoms/penrose/decision/d1" in store.list(n=10)
    assert "atoms/penrose/observation/o1" in store.list(n=10)
    assert "atoms/penrose/principle/p1" in store.list(n=10)
    assert "evaluated_in" in store.graph("atoms/penrose/decision/d1", depth=1)


def test_dedup_finds_committed_decision(tmp_path, monkeypatch):
    from penrose import brainstore, config
    from penrose.brain import BrainReader, Claim

    _disable_local_embed(monkeypatch)
    monkeypatch.setattr(config, "BRAINSTORE_DB", tmp_path / "atoms.db")
    monkeypatch.setattr(config, "DECISIONS_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(config, "CONCEPTS", tmp_path / "reports" / "concepts.jsonl")
    monkeypatch.setattr(config, "PRINCIPLES_LOG", tmp_path / "principles.jsonl")
    _write_jsonl(config.DECISIONS_LOG, [{
        "decision_id": "funding-d1",
        "claim_id": "funding-c1",
        "verdict": "kill",
        "kill_reason": "dedup",
        "rationale": "Perpetual funding rate carry predicts BTC returns.",
        "metrics": {},
    }])
    _write_jsonl(config.CONCEPTS, [])
    _write_jsonl(config.PRINCIPLES_LOG, [])
    brainstore.rebuild_from_flat_files(db_path=config.BRAINSTORE_DB)

    claim = Claim(
        claim_id="new",
        statement="BTC perpetual funding carry predicts returns",
        mechanism="carry",
        scope="btc",
        horizon="daily",
        source_id="test",
        source_span="BTC perpetual funding carry predicts returns",
        claimed_metric_quote="returns",
    )
    hits = BrainReader().search(claim.statement, limit=5)

    assert "atoms/penrose/decision/funding-d1" in hits


def test_determinism(tmp_path, monkeypatch):
    from penrose.brainstore import BrainStore

    _disable_local_embed(monkeypatch)
    store = BrainStore(tmp_path / "atoms.db", auto_rebuild=False)
    store._put("atoms/penrose/decision/b", "---\nkind: decision\n---\n\nFunding carry edge.\n")
    store._put("atoms/penrose/decision/a", "---\nkind: decision\n---\n\nFunding carry edge.\n")

    first = store.search("funding carry", n=2)
    second = store.search("funding carry", n=2)
    assert first == second
    assert [line.split(" :: ", 1)[0] for line in first.splitlines()] == [
        "atoms/penrose/decision/a",
        "atoms/penrose/decision/b",
    ]


def test_fail_soft(tmp_path, monkeypatch):
    from penrose.brainstore import BrainStore

    _disable_local_embed(monkeypatch)
    store = BrainStore(tmp_path / "missing" / "atoms.db", auto_rebuild=False)
    assert store.search("funding", n=5) == ""
    assert store.get("missing") is None

    store._put("atoms/penrose/decision/d1", "---\nkind: decision\n---\n\nFunding carry.\n")
    with store._connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO vectors(slug, dim, vec_json, text_hash) VALUES (?, ?, ?, ?)",
            ("atoms/penrose/decision/d1", 3, "{bad", "x"),
        )
    assert "atoms/penrose/decision/d1" in store.search("funding", n=5)
