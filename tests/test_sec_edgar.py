import gzip
import os

import pandas as pd
import pytest

from penrose.data import sec_edgar
from penrose.data.panel import Panel


def _facts(namespace, tag, unit, rows):
    return {"facts": {namespace: {tag: {"units": {unit: rows}}}}}


def test_available_true_no_raise_and_grade(monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_UA", "")
    assert sec_edgar.available() is True
    assert sec_edgar.PROVENANCE_GRADE == "point_in_time"
    assert sec_edgar.NAME == "sec_edgar"


def test_get_handles_gzip_without_network(monkeypatch):
    class Response:
        headers = {"Content-Encoding": "gzip"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, n=-1):
            return gzip.compress(b'{"ok": true}')

    monkeypatch.setattr(sec_edgar.time, "sleep", lambda _: None)
    monkeypatch.setattr(sec_edgar.urllib.request, "urlopen", lambda req, timeout: Response())

    assert sec_edgar._get("https://example.test/companyfacts.json") == {"ok": True}


def test_ticker_cik_map_fetches_and_zero_pads_fail_open_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_edgar, "_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        sec_edgar,
        "_get",
        lambda url: {
            "0": {"ticker": "aapl", "cik_str": 320193},
            "1": {"ticker": "msft", "cik_str": "789019"},
        },
    )

    assert sec_edgar.ticker_cik_map() == {"AAPL": "0000320193", "MSFT": "0000789019"}

    (tmp_path / "ticker_cik.json").write_text("{not json")
    monkeypatch.setattr(sec_edgar, "_get", lambda url: None)
    assert sec_edgar.ticker_cik_map() == {}


def test_concept_records_tag_fallback_tz_aware_and_drops_incomplete(monkeypatch):
    fallback_facts = {
        "facts": {
            "us-gaap": {
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": {
                    "units": {
                        "USD": [
                            {"end": "2020-12-31", "filed": "2021-02-10", "val": 100.0},
                            {"end": "2021-12-31", "filed": None, "val": 200.0},
                            {"end": "2022-12-31", "filed": "2023-02-10"},
                            {"filed": "2024-02-10", "val": 400.0},
                        ]
                    }
                }
            }
        }
    }
    monkeypatch.setattr(sec_edgar, "_companyfacts", lambda cik: fallback_facts)

    rec = sec_edgar.concept_records("AAA", "book_equity", cik_map={"AAA": "1"})

    assert rec is not None
    assert list(rec.columns) == ["end", "filed", "val"]
    assert len(rec) == 1
    assert rec.loc[0, "val"] == 100.0
    assert str(rec["end"].dt.tz) == "UTC"
    assert str(rec["filed"].dt.tz) == "UTC"


def test_concept_records_first_candidate_with_data_wins(monkeypatch):
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [{"end": "2020-12-31", "filed": "2021-01-31", "val": 10.0}]}
                },
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [{"end": "2020-12-31", "filed": "2021-01-31", "val": 999.0}]}
                },
            }
        }
    }
    monkeypatch.setattr(sec_edgar, "_companyfacts", lambda cik: facts)

    rec = sec_edgar.concept_records("AAA", "revenue", cik_map={"AAA": "1"})

    assert rec is not None
    assert rec["val"].to_list() == [10.0]


def test_fundamentals_panel_point_in_time_absent_entities_and_empty(monkeypatch):
    dates = pd.DatetimeIndex(["2021-01-31", "2021-02-10", "2021-03-15", "2021-04-30"])
    cik_map = {"AAA": "1", "FAIL": "2", "BAD": "3"}
    facts_by_cik = {
        "0000000001": _facts(
            "us-gaap",
            "Assets",
            "USD",
            [
                {"end": "2020-12-31", "filed": "2021-02-10", "val": 100.0},
                {"end": "2021-03-31", "filed": "2021-04-15", "val": 150.0},
            ],
        ),
        "0000000002": None,
        "0000000003": {"facts": {"us-gaap": {"Assets": {"units": {"USD": "not rows"}}}}},
    }
    monkeypatch.setattr(sec_edgar, "_companyfacts", lambda cik: facts_by_cik.get(cik))

    panel = sec_edgar.fundamentals_panel(["AAA", "MISS", "FAIL", "BAD"], "assets", dates, cik_map=cik_map)

    assert isinstance(panel, Panel)
    assert panel.kind == "characteristic"
    assert panel.provenance.startswith("sec_edgar:assets:point_in_time")
    assert str(panel.data.index.tz) == "UTC"
    assert list(panel.data.columns) == ["AAA"]
    assert pd.isna(panel.data.loc[pd.Timestamp("2021-01-31", tz="UTC"), "AAA"])
    assert panel.data.loc[pd.Timestamp("2021-02-10", tz="UTC"), "AAA"] == 100.0
    assert panel.data.loc[pd.Timestamp("2021-03-15", tz="UTC"), "AAA"] == 100.0
    assert panel.data.loc[pd.Timestamp("2021-04-30", tz="UTC"), "AAA"] == 150.0

    empty = sec_edgar.fundamentals_panel([], "assets", dates, cik_map=cik_map)
    assert isinstance(empty, Panel)
    assert empty.data.empty
    assert str(empty.data.index.tz) == "UTC"


def test_fundamentals_panel_lag_days_delays_availability(monkeypatch):
    dates = pd.DatetimeIndex(["2021-02-10", "2021-02-12"], tz="UTC")
    monkeypatch.setattr(
        sec_edgar,
        "_companyfacts",
        lambda cik: _facts("us-gaap", "Assets", "USD", [{"end": "2020-12-31", "filed": "2021-02-10", "val": 7}]),
    )

    panel = sec_edgar.fundamentals_panel(["AAA"], "assets", dates, cik_map={"AAA": "1"}, lag_days=2)

    assert pd.isna(panel.data.loc[pd.Timestamp("2021-02-10", tz="UTC"), "AAA"])
    assert panel.data.loc[pd.Timestamp("2021-02-12", tz="UTC"), "AAA"] == 7.0


def test_malformed_companyfacts_blob_absent_no_raise(monkeypatch):
    monkeypatch.setattr(sec_edgar, "_companyfacts", lambda cik: {"facts": None})

    rec = sec_edgar.concept_records("AAA", "assets", cik_map={"AAA": "1"})
    panel = sec_edgar.fundamentals_panel(["AAA"], "assets", pd.date_range("2021-01-01", periods=2), cik_map={"AAA": "1"})

    assert rec is None
    assert panel.data.empty


@pytest.mark.skipif(not os.environ.get("PENROSE_SEC_LIVE"), reason="PENROSE_SEC_LIVE not set")
def test_sec_edgar_live_aapl_book_equity_panel():
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="YE", tz="UTC")

    panel = sec_edgar.fundamentals_panel(["AAPL"], "book_equity", dates)

    assert isinstance(panel, Panel)
    assert "AAPL" in panel.data.columns
    assert panel.data["AAPL"].dropna().size > 0
