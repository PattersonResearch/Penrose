from penrose.brain import Claim
from penrose.pipeline import stages


def _claim(**kwargs) -> Claim:
    base = dict(
        claim_id="post",
        statement="post sample test",
        mechanism="",
        scope="",
        horizon="",
        source_id="unit",
        source_span="span",
        claimed_metric_quote="",
    )
    base.update(kwargs)
    return Claim(**base)


def _survivor_bt(**post_sample):
    return {
        "psr": 0.99,
        "dsr": 0.99,
        "edge_t": 3.0,
        "n_oos": 1200,
        "bars_per_year": 252.0,
        "three_fold": {"folds": [1.0, 1.0, 1.0], "consistent": True},
        "capacity_usd": 1e6,
        "bootstrap": {},
        "permutation": {},
        "regime": {},
        "post_sample": post_sample,
    }


def test_no_post_sample_caps_supported_to_watch_and_flags():
    claim = _claim(sample_period={"start": "2020-01-01", "end": "2023-12-31"})
    bt = _survivor_bt(sample_end="2023-12-31", data_end="2023-12-31",
                      post_years=0.0, declared=True)
    dec = stages.p8_verdict(claim, bt, {"holdout_sharpe": 1.0, "holdout_psr": 0.99}, False)
    assert dec.verdict == "watch"
    assert dec.metrics["no_post_sample_data"] is True
    assert "no post-sample evidence" in dec.rationale


def test_sufficient_post_sample_is_unaffected():
    from penrose import config

    claim = _claim(sample_period={"start": "2020-01-01", "end": "2022-01-01"})
    holdout = {"holdout_sharpe": 1.0, "holdout_psr": 0.99}
    base = dict(sample_end="2022-01-01", data_end="2024-01-02",
                post_years=2.0, declared=True)
    old = config.COST_PROVENANCE
    try:
        config.COST_PROVENANCE = "measured"
        dec = stages.p8_verdict(claim, _survivor_bt(**base), holdout, False)
    finally:
        config.COST_PROVENANCE = old
    assert dec.verdict == "research-supported"
    assert dec.metrics["no_post_sample_data"] is False


def test_post_sample_cap_does_not_touch_kills_or_underpowered():
    claim = _claim(sample_period={"start": "2020-01-01", "end": "2023-12-31"})
    bt = _survivor_bt(sample_end="2023-12-31", data_end="2023-12-31",
                      post_years=0.0, declared=True)
    bt["dsr"] = 0.1
    bt["n_oos"] = 1200
    dec = stages.p8_verdict(claim, bt, {}, False)
    assert dec.verdict == "kill"
    assert dec.metrics["no_post_sample_data"] is False


def test_claim_sample_period_validation():
    assert _claim(sample_period=None).sample_period is None
    try:
        _claim(sample_period={"start": "2024-01-01", "end": "2023-01-01"})
    except ValueError as e:
        assert "sample_period.start must be before" in str(e)
    else:
        raise AssertionError("bad sample_period should fail clearly")
