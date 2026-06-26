from __future__ import annotations

import sys

import pytest

from penrose import config
from penrose.brain import Claim
from penrose.pipeline import relevance, spec_gen
from penrose.pipeline.p1_ingest import IngestedSource


def _clear_catalog_caches() -> None:
    sys.modules.pop("loader", None)


@pytest.fixture(autouse=True)
def _catalog_isolation():
    _clear_catalog_caches()
    yield
    _clear_catalog_caches()


def test_catalog_awareness_fail_open_without_catalog(monkeypatch, tmp_path):
    monkeypatch.delenv("PENROSE_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "missing")
    assert relevance._data_domains() == relevance._BASE_DOMAINS
    assert spec_gen._catalog_vocab() == ""


def test_catalog_domains_and_vocab_light_up(monkeypatch, tmp_path):
    (tmp_path / "loader.py").write_text(
        "def domains():\n"
        "    return ['equity', 'inflation']\n\n"
        "def available():\n"
        "    return ['equity_spy', 'us_breakeven_10y']\n\n"
        "def describe_brief(name):\n"
        "    meta = {\n"
        "        'equity_spy': 'equity_spy [equity, stooq, vendor]',\n"
        "        'us_breakeven_10y': 'us_breakeven_10y [inflation, fred, vendor]',\n"
        "    }\n"
        "    return meta[name]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    def fake_screen_call(role, messages, *, temperature):
        assert role == "falsifiability_classifier"
        assert "equity indices / ETFs" in messages[0]["content"]
        assert "inflation expectations / breakevens" in messages[0]["content"]
        return {"relevant": True, "domains": ["equity", "inflation"], "reason": "testable"}, None

    monkeypatch.setattr(relevance.llm, "call_json", fake_screen_call)
    screened = relevance.screen(
        "Crypto and equity risk premia",
        "A crypto-vs-equity note comparing BTC returns with SPY and inflation breakevens.",
    )
    assert screened["relevant"] is True

    def fake_spec_call(role, messages, *, temperature):
        assert role == "module_spec_generator"
        assert "AVAILABLE DATA SERIES" in messages[1]["content"]
        assert "equity_spy [equity, stooq, vendor]" in messages[1]["content"]
        assert "us_breakeven_10y [inflation, fred, vendor]" in messages[1]["content"]
        return {
            "module_id": "crypto_equity_inflation",
            "strategy_class": "equity_macro",
            "claim_translation": "Compare crypto with equity and inflation proxies.",
            "inputs": ["equity_spy", "us_breakeven_10y"],
            "signal_logic": "Use listed catalog inputs.",
            "kill_criterion": "OOS DSR below threshold.",
            "unknowns": [],
        }, None

    monkeypatch.setattr(spec_gen.llm, "call_json", fake_spec_call)
    claim = Claim(
        claim_id="c1",
        statement="Crypto returns are explained by equity beta and inflation expectations.",
        mechanism="risk premia",
        scope="US liquid markets",
        horizon="daily",
        source_id="s1",
        source_span="Crypto returns are explained by equity beta and inflation expectations.",
        claimed_metric_quote="",
        applicable_strategy_class="equity_macro",
    )
    source = IngestedSource("s1", "source", "text", 1, 4, "sha", [])
    spec = spec_gen._llm_spec(claim, source)
    assert spec["inputs"] == ["equity_spy", "us_breakeven_10y"]


def test_catalog_loader_is_per_call_and_restores_sys_path(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "loader.py").write_text("def domains():\n    return ['equity']\n", encoding="utf-8")
    (second / "loader.py").write_text(
        "def domains():\n    return ['inflation']\n",
        encoding="utf-8",
    )
    before = list(sys.path)

    monkeypatch.setattr(config, "DATA_DIR", first)
    assert any("equity indices / ETFs" in d for d in relevance._catalog_domains())
    assert str(first) not in sys.path

    monkeypatch.setattr(config, "DATA_DIR", second)
    assert any("inflation expectations / breakevens" in d for d in relevance._catalog_domains())
    assert str(second) not in sys.path
    assert sys.path == before
