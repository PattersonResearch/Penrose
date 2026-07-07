from __future__ import annotations

from copy import deepcopy


GRAPH = {"nodes": [{"node_id": "cross-1", "level": "cross_family_mechanism"}]}
CAPABILITIES = {"series": ["funding_btc", "spot_btc"]}


def _candidate() -> dict:
    return {
        "statement": "Funding divergence predicts next-period BTC returns",
        "mechanism": "funding pressure can mark crowded positioning",
        "scope": "BTC perpetuals",
        "horizon": "3d",
        "strategy_class": "funding-divergence",
        "candidate_class": "testable_now",
        "inspired_by": ["cross-1"],
        "falsifier": "no OOS lift after funding divergence",
        "spec": {
            "signal": "zscore(funding_btc, window) - ma(funding_btc, slow_window)",
            "series": ["funding_btc"],
            "params": {"window": 20, "slow_window": 60},
            "param_grid": {"window": [10, 20, 40], "slow_window": [40, 60, 120]},
            "conditioning": None,
            "entry_exit": "enter when signal > 1; exit after horizon or signal <= 0",
            "horizon": "3d",
        },
    }


def test_complete_grounded_structured_candidate_is_reconstructable_and_admitted():
    from penrose.synthesize import normalize

    _claims, normalized = normalize("synth-r", [_candidate()], GRAPH, CAPABILITIES)

    assert normalized[0]["reconstructable"] is True
    assert normalized[0]["reconstructable_reason"] == ""
    assert normalized[0]["admitted"] is True


def test_missing_required_spec_field_is_not_reconstructable_or_admitted():
    from penrose.synthesize import normalize

    raw = _candidate()
    del raw["spec"]["param_grid"]
    _claims, normalized = normalize("synth-r", [raw], GRAPH, CAPABILITIES)

    assert normalized[0]["reconstructable"] is False
    assert normalized[0]["admitted"] is False
    assert "spec.param_grid" in normalized[0]["reconstructable_reason"]


def test_series_name_outside_capabilities_is_not_reconstructable():
    from penrose.synthesize import normalize

    raw = _candidate()
    raw["spec"]["series"] = ["fabricated_funding"]
    raw["spec"]["signal"] = "zscore(fabricated_funding, window)"
    _claims, normalized = normalize("synth-r", [raw], GRAPH, CAPABILITIES)

    assert normalized[0]["reconstructable"] is False
    assert normalized[0]["admitted"] is False
    assert "unknown series" in normalized[0]["reconstructable_reason"]


def test_signal_with_undeclared_identifier_is_not_reconstructable():
    from penrose.synthesize import normalize

    raw = _candidate()
    raw["spec"]["signal"] = "zscore(funding_btc, window) - undeclared_factor"
    _claims, normalized = normalize("synth-r", [raw], GRAPH, CAPABILITIES)

    assert normalized[0]["reconstructable"] is False
    assert normalized[0]["admitted"] is False
    assert "undeclared signal identifier" in normalized[0]["reconstructable_reason"]


def test_missing_capabilities_fails_open_for_series_resolution_only():
    from penrose.synthesize import normalize

    raw = deepcopy(_candidate())
    raw["spec"]["series"] = ["not_in_a_manifest"]
    raw["spec"]["signal"] = "zscore(not_in_a_manifest, window)"
    _claims, normalized = normalize("synth-r", [raw], GRAPH, {})

    assert normalized[0]["reconstructable"] is True
    assert normalized[0]["admitted"] is True


def test_reconstructability_rejects_semantically_vacuous_specs():
    """N-1: a syntactically-valid-but-vacuous spec must NOT pass the gate (was passing then pending_module)."""
    from penrose.synthesize import _reconstructability as R
    caps = {"series": {"funding_btc": "x", "eth_spot_daily": "y"}}
    ok = {"signal": "zscore(funding_btc, window) - ma(funding_btc, w2)", "series": ["funding_btc"],
          "params": {"window": 20, "w2": 60}, "param_grid": {"window": [10, 20], "w2": [30, 60]},
          "entry_exit": "long top decile", "horizon": "1d", "conditioning": None}
    assert R({"spec": ok}, caps)[0] is True
    def mut(**kw):
        s = dict(ok); s.update(kw); return R({"spec": s}, caps)[0]
    assert mut(signal="funding_btc") is False                    # bare series
    assert mut(signal="zscore(funding_btc)") is False            # wrong arity
    assert mut(signal="funding_btc eth_spot_daily") is False     # malformed
    assert mut(params={"window": "twenty"}) is False             # non-numeric param value
    assert mut(param_grid={"unrelated": [1, 2]}) is False         # grids an undeclared param
