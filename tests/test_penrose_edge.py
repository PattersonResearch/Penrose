import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _concept(i, family, domain, statement="positive recurring observation"):
    return {"concept_id": f"c{i}", "source_claim_id": f"q{i}", "statement": statement,
            "created_at": "2026-01-01T00:00:00+00:00",
            "evidence_direction": "positive",
            "data_provenance": {"strategy_family": family, "data_domain": domain}}


def test_kill_cannot_emit_supported_surviving_explanation():
    from penrose.concepts import extract
    c = extract({"claim_id": "q", "statement": "claim", "verdict": "kill",
                 "competing_explanations": [{"explanation": "carry survived",
                                             "verdict": "survives"}]}, use_llm=False)
    assert c is not None
    assert c.surviving_explanation == ""


def test_concept_extraction_is_fail_soft():
    from penrose import concepts
    assert concepts.extract({"statement": object()}, use_llm=False) is None


def test_cross_family_needs_two_domains_and_is_deterministic():
    from penrose.corpus import build
    one = [_concept(i, "carry", "crypto") for i in range(3)]
    a = build(one, min_support=3, current_year=2026)
    assert not [n for n in a["nodes"] if n["level"] == "cross_family_mechanism"]
    two = one + [_concept(i + 3, "yield", "rates") for i in range(3)]
    b = build(two, min_support=3, current_year=2026)
    c = build(list(reversed(two)), min_support=3, current_year=2026)
    scrub = lambda x: {k: v for k, v in x.items() if k != "generated_at"}
    assert [n for n in b["nodes"] if n["level"] == "cross_family_mechanism"]
    assert scrub(b) == scrub(c)


def test_synthesized_source_is_capped_and_population_registered(tmp_path):
    from penrose import config
    from penrose.brain import Claim, source_is_unanchored
    from penrose.dream import create_manifest, record_candidates, register_search
    from penrose.pipeline import p7_backtest as p7
    from penrose.synthesize import normalize
    graph = {"nodes": [{"node_id": "m1", "level": "cross_family_mechanism"}]}
    raw = [{"statement": "candidate", "strategy_class": "x",
            "candidate_class": "testable_now", "inspired_by": ["m1"],
            "spec": {"signal": "zscore(edge_series, window)", "series": ["edge_series"],
                     "params": {"window": 20}, "param_grid": {"window": [10, 20, 60]},
                     "conditioning": None,
                     "entry_exit": "enter when signal > 1; exit after horizon",
                     "horizon": "1d"}},
           {"statement": "blocked", "strategy_class": "x",
            "candidate_class": "conceptual_only", "inspired_by": ["m1"]}]
    claims, norm = normalize("s", raw, graph)
    assert all(source_is_unanchored(c.source_type) for c in claims)
    manifest = create_manifest(run_id="s", generation_budget=5, model="x",
                               corpus_snapshot_hash="h", root=tmp_path / "s")
    manifest["source_type"] = "synthesized_hypothesis"
    manifest = record_candidates(manifest, raw)
    old = p7.LEDGER
    p7.LEDGER = tmp_path / "ledger.tsv"
    try:
        manifest = register_search(manifest, claims, norm)
    finally:
        p7.LEDGER = old
    assert manifest["effective_search_denominator"] == 5
    assert manifest["candidates_admitted"] == 1


def test_confirmation_refuses_reserve_overlap():
    from penrose.confirmation import validate_firewall
    reserve = {"epoch_id": "r", "start": "2024-01-01", "end": "2025-01-01",
               "data_domains": ["reserved"], "datasets": []}
    ok, reason = validate_firewall(
        {"data_provenance": {"data_domains": ["reserved"],
                             "periods": [{"start": "2020-01-01", "end": "2021-01-01"}]}},
        reserve)
    assert not ok and "intersects" in reason


def test_exposure_decomposition_recovers_beta():
    import numpy as np
    import pandas as pd
    from penrose.explanations import exposure_decomposition
    x = pd.Series(np.linspace(-1, 1, 100))
    y = 0.002 + 2.0 * x
    out = exposure_decomposition(y, market=x)
    assert out["applicable"]
    assert abs(out["betas"]["market"] - 2.0) < 1e-9
    assert abs(out["intercept"] - 0.002) < 1e-9
    assert out["survives"] is True


def test_explanation_inputs_never_see_holdout_tail():
    import numpy as np
    import pandas as pd
    from penrose.explanations import visible_inputs
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    base = pd.Series(np.arange(100, dtype=float), index=idx)
    changed = base.copy()
    changed.iloc[80:] = 1e9
    assert visible_inputs(base)["net"].equals(visible_inputs(changed)["net"])


def test_grounding_drops_common_overclaims():
    from penrose.concepts import ground_draft
    draft, flags = ground_draft(
        {"statement": "observation", "mechanism": "reliable alpha edge",
         "rejected_explanations": [], "boundary": {}},
        {"verdict": "kill", "competing_explanations": []})
    assert draft["mechanism"] == ""
    assert "mechanism:overclaim" in flags


def test_unknown_direction_is_not_promoted():
    from penrose.corpus import build
    rows = [{**_concept(i, "carry", "crypto"), "evidence_direction": "unknown"}
            for i in range(3)]
    graph = build(rows, min_support=3, current_year=2026)
    assert not [n for n in graph["nodes"] if n["level"] == "family_principle"]


def test_concept_append_is_concurrency_safe(tmp_path):
    from penrose.concepts import Concept, append
    path = tmp_path / "concepts.jsonl"
    concepts = [Concept(concept_id=f"c{i}", source_claim_id=f"q{i}",
                        statement=f"observation {i}") for i in range(12)]
    threads = [threading.Thread(target=append, args=(c, path)) for c in concepts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert {x["concept_id"] for x in rows} == {c.concept_id for c in concepts}


def test_confirmation_loads_distinct_epoch_bundles(tmp_path, monkeypatch):
    from penrose import config
    from penrose.confirmation import confirm_run
    from penrose.data.contract import DataBundle
    from penrose.data import client as dataclient
    from penrose.pipeline import run as runmod

    synth_root = tmp_path / "syntheses"
    run_root = synth_root / "r1"
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(json.dumps({
        "status": "generated_only", "generation_budget": 2, "candidates_generated": 2}))
    rows = []
    for i in range(2):
        rows.append({
            "claim_id": f"r1-c{i}", "raw_hypothesis_id": f"raw-{i}", "admitted": True,
            "raw": {"statement": f"candidate {i}", "strategy_class": "x"},
            "data_provenance": {
                "periods": [{"start": "2020-01-01", "end": "2021-01-01"}]}})
    (run_root / "candidates.normalized.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rows))
    (run_root / "source.md").write_text("# frozen synthesis")
    monkeypatch.setattr(config, "SYNTHESIS_ARCHIVES", synth_root)
    monkeypatch.setattr(config, "CONFIRMATION_RESERVE", {
        "reserve_id": "reserve",
        "epochs": [
            {"epoch_id": "e1", "start": "2024-01-01", "end": "2024-06-30"},
            {"epoch_id": "e2", "start": "2024-07-01", "end": "2024-12-31"},
        ]})
    loaded, invoked = [], []
    def fake_fetch(start, end):
        loaded.append((start, end))
        return DataBundle(requested_window=(start, end))
    def fake_run(source, **kwargs):
        invoked.append((kwargs["claims_override"][0].claim_id,
                        kwargs["bundle_override"].requested_window))
        return {"decisions": []}
    monkeypatch.setattr(dataclient, "fetch_bundle", fake_fetch)
    monkeypatch.setattr(runmod, "run_source", fake_run)
    result = confirm_run("r1")
    assert result["status"] == "complete"
    assert loaded == [("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31")]
    assert len({window for _, window in invoked}) == 2


def test_confirmation_dataset_allowlist_is_enforced():
    import pandas as pd
    from penrose.confirmation import _restrict_bundle
    from penrose.data.contract import DataBundle, Series
    idx = pd.date_range("2024-01-01", periods=3)
    bundle = DataBundle(series={
        "reserved": Series("reserved", pd.Series([1, 2, 3], index=idx), "reserve", "x"),
        "discovery": Series("discovery", pd.Series([4, 5, 6], index=idx), "discovery", "x"),
    })
    out = _restrict_bundle(bundle, {"datasets": ["reserved"]})
    assert set(out.series) == {"reserved"}


def test_missing_concept_timestamp_is_flagged():
    from penrose.concepts import extract
    concept = extract(
        {"claim_id": "legacy", "statement": "legacy observation", "verdict": "kill"},
        use_llm=False)
    assert concept is not None
    assert concept.created_at == "1970-01-01T00:00:00+00:00"
    assert "created_at:missing_source_timestamp" in concept.grounding_flags


def test_synthesis_summary_upsert_is_idempotent(tmp_path, monkeypatch):
    from penrose import config
    from penrose.synthesize import _upsert_synthesis_summary
    path = tmp_path / "synthesis_runs.jsonl"
    monkeypatch.setattr(config, "SYNTHESIS_RUNS", path)
    _upsert_synthesis_summary({"synthesis_run_id": "s1", "status": "registered"})
    _upsert_synthesis_summary({"synthesis_run_id": "s1", "status": "complete"})
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows == [{"synthesis_run_id": "s1", "status": "complete"}]


def test_confirmation_failure_before_holdout_is_retryable(tmp_path, monkeypatch):
    from penrose import config
    from penrose.confirmation import confirm_run
    from penrose.data.contract import DataBundle
    from penrose.data import client as dataclient
    from penrose.pipeline import run as runmod

    synth_root = tmp_path / "syntheses"
    run_root = synth_root / "r2"
    run_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text(json.dumps({
        "status": "generated_only", "generation_budget": 1, "candidates_generated": 1}))
    row = {
        "claim_id": "r2-c1", "raw_hypothesis_id": "raw-1", "admitted": True,
        "raw": {"statement": "candidate", "strategy_class": "x"},
        "data_provenance": {
            "periods": [{"start": "2020-01-01", "end": "2021-01-01"}]}}
    (run_root / "candidates.normalized.jsonl").write_text(json.dumps(row) + "\n")
    (run_root / "source.md").write_text("# frozen synthesis")
    monkeypatch.setattr(config, "SYNTHESIS_ARCHIVES", synth_root)
    monkeypatch.setattr(config, "CONFIRMATION_RESERVE", {
        "reserve_id": "reserve",
        "epochs": [{"epoch_id": "e1", "start": "2024-01-01", "end": "2024-12-31"}]})
    monkeypatch.setattr(
        dataclient, "fetch_bundle",
        lambda start, end: DataBundle(requested_window=(start, end)))
    monkeypatch.setattr(runmod, "run_source",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = confirm_run("r2")
    assert result["status"] == "refused"
    assert result["holdout_consumed"] is False
    assert "may be retried" in result["reason"]
    assert not (run_root / "confirmation_locks" / "e1.lock").exists()
