import pandas as pd

from penrose.pipeline import event_study as ES


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


def test_es1_negative_start_window_estimation_strictly_before_event_window():
    """ES-1: for a negative-start event window ([-2,+5], the standard pre-event-drift convention), the
    baseline estimation window must end strictly BEFORE the event window starts — not at the event date —
    so pre-event-leg returns never contaminate the baseline they are compared against."""
    idx = _idx(100)
    ret = pd.Series(0.0, index=idx)
    ret.iloc[58] = 1.0
    ret.iloc[59] = 1.0  # positions 58,59 = the [-2] pre-event leg of a window at event position 60
    net, rows = ES._event_car_rows(
        ret, pd.DatetimeIndex([idx[60]]), event_window=(-2, 5),
        estimation_window=20, baseline="mean_adjusted")
    kept = [r for r in rows if not r.get("skipped")]
    assert len(kept) == 1
    # faithful CAR captures the full +2.0 bump; the contaminated-baseline bug attenuated it to 1.2
    assert abs(float(net.iloc[0]) - 2.0) < 1e-9
    # estimation window ends strictly before the event window start (position 58)
    assert pd.Timestamp(kept[0]["estimation_end"]) < idx[58]


def test_es_no_lookahead_baseline_unaffected_by_post_event_returns():
    """Mutating returns at/after the event window never changes a prior event's baseline."""
    idx = _idx(100)
    ret = pd.Series(0.0, index=idx)
    net_a, _ = ES._event_car_rows(ret, pd.DatetimeIndex([idx[60]]), event_window=(0, 5),
                                  estimation_window=20, baseline="mean_adjusted")
    ret2 = ret.copy()
    ret2.iloc[70:] = 5.0  # far after the event window
    net_b, _ = ES._event_car_rows(ret2, pd.DatetimeIndex([idx[60]]), event_window=(0, 5),
                                  estimation_window=20, baseline="mean_adjusted")
    assert abs(float(net_a.iloc[0]) - float(net_b.iloc[0])) < 1e-12
