from penrose.brain import Claim
from penrose.pipeline import stages


def _claim(expected_edge):
    return Claim(
        claim_id="fee-edge",
        statement="Test statement",
        mechanism="Test mechanism",
        scope="BTC",
        horizon="1 day",
        source_id="unit",
        source_span="Test statement",
        claimed_metric_quote="",
        expected_edge=expected_edge,
    )


def test_pen16_fee_curve_uses_claim_expected_edge():
    # PEN-16: fee_curve is only evaluated when the claim states a numeric edge.
    low = stages.p4_fee_curve(_claim(0.0004), expected_edge=0.0004)
    assert low["killed"] is True
    assert low["reason"] == "fee_curve"
    assert low["evaluated"] is True

    unstated = stages.p4_fee_curve(_claim(None), expected_edge=None)
    assert unstated["killed"] is False
    assert unstated["reason"] is None
    assert unstated["evaluated"] is False

    high = stages.p4_fee_curve(_claim(0.02), expected_edge=0.02)
    assert high["killed"] is False
    assert high["reason"] is None
    assert high["evaluated"] is True
