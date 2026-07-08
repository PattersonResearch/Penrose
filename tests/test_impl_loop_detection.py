from types import SimpleNamespace

from penrose.brain import Claim


def _claim():
    return Claim(
        "impl-loop-c1",
        "A daily momentum strategy buys after positive trailing returns.",
        "time-series momentum",
        "BTC spot",
        "10d",
        "unit",
        "span",
        "positive DSR",
    )


def _spec():
    return {
        "module_id": "loop-test",
        "strategy_class": "time_series_momentum",
        "claim_type": "trading_strategy",
        "claim_translation": "Daily time-series momentum.",
        "inputs": ["btc_spot_daily"],
        "signal_logic": "Compute trailing returns without look-ahead.",
        "kill_criterion": "fails robustness",
    }


def _code(label):
    return f"""
__module_id__ = 'unused'
__strategy_class__ = 'time_series_momentum'
VERSION = {label!r}

def run(bundle, claim, cost_frac):
    return {{'ok': False, 'reason': 'data_unavailable: unit test'}}
"""


def _install_validation_fakes(tmp_path, monkeypatch, validation_results):
    from penrose import config
    from penrose.pipeline import impl_gen, sandbox

    monkeypatch.setattr(config, "AUTO_MODULES", tmp_path / "modules" / "_auto")
    monkeypatch.setattr(config, "FIDELITY_CHECK", False)
    monkeypatch.setattr(config, "IMPL_NO_PROGRESS_LIMIT", 2)

    gen_calls = []
    validation_calls = []

    def fake_generate(spec, available, feedback=None):
        gen_calls.append({"spec": dict(spec), "feedback": feedback})
        return _code(f"attempt_{len(gen_calls)}")

    def fake_validate(impl_path, mid, bundle, claim, cost_frac, prerun_result=None,
                      validation_meta=None):
        source = impl_path.read_text()
        validation_calls.append(source)
        result = validation_results[len(validation_calls) - 1]
        if result is True:
            if validation_meta is not None:
                validation_meta["unit_test"] = len(validation_calls)
            return True, SimpleNamespace(
                __file__=str(impl_path),
                module_id=mid,
                source=source,
                validation=len(validation_calls),
            )
        return False, result

    monkeypatch.setattr(impl_gen, "_generate_code", fake_generate)
    monkeypatch.setattr(impl_gen, "_validate_module", fake_validate)
    monkeypatch.setattr(sandbox, "run_in_container", lambda *a, **k: {"ok": True})
    return gen_calls, validation_calls


def test_same_failure_signature_short_circuits_to_needs_review(tmp_path, monkeypatch):
    from penrose.pipeline import impl_gen

    gen_calls, validation_calls = _install_validation_fakes(
        tmp_path,
        monkeypatch,
        [
            "data gate rejected: missing 12 rows from btc_spot_daily at 2026-07-07T12:00:00Z",
            "data gate rejected: missing 49 rows from btc_spot_daily at 2026-07-08T12:00:00Z",
            "should not be reached",
        ],
    )

    out = impl_gen.try_implement(_spec(), _claim(), bundle=object(), cost_frac=0.001,
                                 use_llm=True, max_attempts=3)

    assert out["ok"] is False
    assert out["needs_review"] is True
    assert out["attempts"] == 2
    assert "auto-impl made no progress: 2 attempts" in out["reason"]
    assert out["no_progress"]["attempts_tried"] == 2
    assert out["no_progress"]["consecutive_attempts"] == 2
    assert len(gen_calls) == 2
    assert len(validation_calls) == 2


def test_different_failure_signatures_run_to_normal_bound(tmp_path, monkeypatch):
    from penrose.pipeline import impl_gen

    gen_calls, validation_calls = _install_validation_fakes(
        tmp_path,
        monkeypatch,
        [
            "data gate rejected: missing btc_spot_daily",
            "run() not ok and not a clean data blocker: KeyError signal",
            "generated net has non-datetime index",
        ],
    )

    out = impl_gen.try_implement(_spec(), _claim(), bundle=object(), cost_frac=0.001,
                                 use_llm=True, max_attempts=3)

    assert out["ok"] is False
    assert not out.get("needs_review")
    assert out["reason"].startswith("validation failed after 3 attempts")
    assert len(gen_calls) == 3
    assert len(validation_calls) == 3


def test_successful_attempt_is_unaffected(tmp_path, monkeypatch):
    from penrose.pipeline import impl_gen

    gen_calls, validation_calls = _install_validation_fakes(tmp_path, monkeypatch, [True])

    out = impl_gen.try_implement(_spec(), _claim(), bundle=object(), cost_frac=0.001,
                                 use_llm=True, max_attempts=3)

    assert out["ok"] is True
    assert out["attempts"] == 1
    assert "attempt_1" in out["module"].source
    assert len(gen_calls) == 1
    assert len(validation_calls) == 1


def test_failure_signature_is_deterministic_and_ignores_volatile_numbers():
    from penrose.pipeline import impl_gen

    one = impl_gen._failure_signature(
        _spec(), "Data gate rejected: missing 12 rows at 2026-07-07T12:00:00Z line 44")
    two = impl_gen._failure_signature(
        _spec(), "data gate rejected: missing 99 rows at 2026-01-02T08:30:00Z line 7")
    again = impl_gen._failure_signature(
        _spec(), "Data gate rejected: missing 12 rows at 2026-07-07T12:00:00Z line 44")

    assert one["signature"] == two["signature"]
    assert one["signature"] == again["signature"]
    assert "<num>" in one["category"]
    assert "12" not in one["category"]


def test_normalize_collapses_quoted_values_ld2():
    """LD-2: the SAME dead end feeding different quoted values must collide (guard fires)."""
    from penrose.pipeline.impl_gen import _normalize_failure_reason as N
    assert N("ValueError: invalid literal for int() with base 10: 'abc'") == \
           N("ValueError: invalid literal for int() with base 10: 'xyz'")
    assert N('KeyError: "foo"') == N('KeyError: "bar"')


def test_no_progress_limit_clamps_degenerate_values_ld3():
    """LD-3: limit=1 would fire on the first failure and disable self-repair; it clamps to 2. 0=disabled."""
    def clamp(raw):
        return 0 if raw <= 0 else max(2, raw)
    assert clamp(1) == 2 and clamp(0) == 0 and clamp(-5) == 0 and clamp(3) == 3


def test_normalize_quote_strip_survives_contractions_ld2r2():
    """LD2-1: an apostrophe inside a word (doesn't) must not break collision of a following quoted value."""
    from penrose.pipeline.impl_gen import _normalize_failure_reason as N
    assert N("module doesn't implement 'momentum' signal") == N("module doesn't implement 'trend' signal")
