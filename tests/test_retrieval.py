from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_corpus(path: Path, nodes: list[dict], edges: list[dict] | None = None) -> None:
    path.write_text(json.dumps({"nodes": nodes, "edges": edges or []}))


def test_loads_missing_corpus_softfails(tmp_path):
    from penrose import retrieval

    assert retrieval.load_corpus(tmp_path / "missing.json") == {"nodes": [], "edges": []}


def test_lexical_fallback_ranks(tmp_path, monkeypatch):
    from penrose import config, llm, retrieval

    corpus_path = tmp_path / "corpus.json"
    _write_corpus(corpus_path, [
        {"node_id": "a", "level": "observation", "direction": "positive",
         "statement": "Funding rate carry predicts bitcoin returns.",
         "source_type": "external_source"},
        {"node_id": "b", "level": "observation", "direction": "negative",
         "statement": "Weather surprises affect harvest yields.",
         "source_type": "external_source"},
    ])
    monkeypatch.setattr(config, "CORPUS_JSON", corpus_path)
    monkeypatch.setattr(llm, "embed_available", lambda: False)
    monkeypatch.setattr(llm, "embed_local_available", lambda: False)

    nodes = retrieval.retrieve("bitcoin funding carry", k=2)
    assert [n["node_id"] for n in nodes][0] == "a"


def test_graph_expansion_pulls_neighbor(tmp_path, monkeypatch):
    from penrose import config, llm, retrieval

    corpus_path = tmp_path / "corpus.json"
    _write_corpus(
        corpus_path,
        [
            {"node_id": "seed", "level": "observation", "direction": "positive",
             "statement": "Funding carry predicts returns.", "source_type": "external_source"},
            {"node_id": "neighbor", "level": "family_principle", "direction": "positive",
             "statement": "Positive evidence recurs across carry signals.",
             "source_type": "external_source"},
        ],
        [{"from": "seed", "to": "neighbor", "type": "supports"}],
    )
    monkeypatch.setattr(config, "CORPUS_JSON", corpus_path)
    monkeypatch.setattr(llm, "embed_local_available", lambda: False)

    assert "neighbor" in [n["node_id"] for n in retrieval.retrieve("funding", k=2, hops=1)]


def test_firewall_excludes_reserved(tmp_path, monkeypatch):
    from penrose import config, llm, retrieval

    corpus_path = tmp_path / "corpus.json"
    _write_corpus(corpus_path, [
        {"node_id": "reserved", "level": "observation", "direction": "positive",
         "statement": "Funding carry predicts returns.", "source_type": "external_source",
         "data_provenance": {"data_domains": ["crypto"], "periods": [
             {"start": "2020-01-01", "end": "2020-12-31"}]}},
        {"node_id": "safe", "level": "observation", "direction": "positive",
         "statement": "Funding carry has independent support.", "source_type": "external_source",
         "data_provenance": {"data_domains": ["rates"], "periods": [
             {"start": "2021-01-01", "end": "2021-12-31"}]}},
    ])
    reserve = {"epochs": [{"epoch_id": "r1", "data_domains": ["crypto"],
                           "periods": [{"start": "2020-06-01", "end": "2020-06-30"}]}]}
    monkeypatch.setattr(config, "CORPUS_JSON", corpus_path)
    monkeypatch.setattr(llm, "embed_local_available", lambda: False)

    assert "reserved" not in [n["node_id"] for n in retrieval.retrieve("funding carry", reserve=reserve)]


def test_firewall_excludes_unanchored(tmp_path, monkeypatch):
    from penrose import config, llm, retrieval

    corpus_path = tmp_path / "corpus.json"
    _write_corpus(corpus_path, [
        {"node_id": "generated", "level": "observation", "direction": "positive",
         "statement": "Funding carry predicts returns.",
         "source_type": "synthesized_hypothesis"},
        {"node_id": "generated-aggregate", "level": "family_principle",
         "direction": "positive",
         "statement": "Funding carry recurs across generated observations.",
         "provenance": ["generated"]},
        {"node_id": "external", "level": "observation", "direction": "positive",
         "statement": "Funding carry external evidence.",
         "source_type": "external_source"},
        {"node_id": "external-aggregate", "level": "family_principle",
         "direction": "positive",
         "statement": "Funding carry recurs across external observations.",
         "provenance": ["external"]},
    ])
    monkeypatch.setattr(config, "CORPUS_JSON", corpus_path)
    monkeypatch.setattr(llm, "embed_local_available", lambda: False)

    node_ids = [n["node_id"] for n in retrieval.retrieve("funding carry", k=4)]
    assert "generated" not in node_ids
    assert "generated-aggregate" not in node_ids
    assert "external" in node_ids


def test_determinism(tmp_path, monkeypatch):
    from penrose import config, llm, retrieval

    corpus_path = tmp_path / "corpus.json"
    _write_corpus(corpus_path, [
        {"node_id": "b", "level": "observation", "direction": "positive",
         "statement": "Funding carry edge.", "source_type": "external_source"},
        {"node_id": "a", "level": "observation", "direction": "positive",
         "statement": "Funding carry edge.", "source_type": "external_source"},
    ])
    monkeypatch.setattr(config, "CORPUS_JSON", corpus_path)
    monkeypatch.setattr(llm, "embed_local_available", lambda: False)

    first = [n["node_id"] for n in retrieval.retrieve("funding carry", k=2)]
    second = [n["node_id"] for n in retrieval.retrieve("funding carry", k=2)]
    assert first == second == ["a", "b"]


def test_format_context_caps_length():
    from penrose import retrieval

    nodes = [
        {"node_id": str(i), "level": "observation", "direction": "positive",
         "statement": "x" * 200, "support_count": 1}
        for i in range(20)
    ]
    assert retrieval.format_context([]) == ""
    assert len(retrieval.format_context(nodes)) <= 1200


def test_chat_unchanged_when_empty(monkeypatch):
    from dashboard import write_api
    from penrose import llm

    captured = {}

    def fake_call(role, messages, **kwargs):
        captured["role"] = role
        captured["messages"] = messages

        class Resp:
            text = "reply"

        return Resp()

    monkeypatch.setattr(write_api, "retrieve_corpus_context", lambda convo: "")
    monkeypatch.setattr(llm, "call", fake_call)
    result = write_api.chat_reply([{"role": "user", "content": "funding idea"}])

    assert result["ok"] is True
    assert captured["role"] == "chat_assistant"
    assert captured["messages"] == [
        {"role": "system", "content": write_api._CHAT_SYSTEM},
        {"role": "user", "content": "funding idea"},
    ]
