from types import SimpleNamespace

from penrose.brain import Claim


def _claim():
    return Claim(
        "impl-fidelity-c1",
        "A daily 10-day time-series momentum rule buys after positive trailing returns.",
        "time-series momentum",
        "BTC spot",
        "10d",
        "unit",
        "span",
        "positive DSR",
    )


def _spec():
    return {
        "module_id": "tsmom",
        "strategy_class": "time_series_momentum",
        "claim_type": "trading_strategy",
        "claim_translation": "Daily 10-day time-series momentum.",
        "inputs": ["btc_spot_daily"],
        "signal_logic": "Use daily observations; compute a 10-day trailing return lookback.",
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


def _install_common_fakes(tmp_path, monkeypatch, generated, fid_results=None):
    from penrose import config
    from penrose.pipeline import fidelity, impl_gen, sandbox

    monkeypatch.setattr(config, "AUTO_MODULES", tmp_path / "modules" / "_auto")
    monkeypatch.setattr(config, "FIDELITY_CHECK", True)
    monkeypatch.setattr(config, "FIDELITY_KILL_CONFIDENCE", 0.6)

    gen_calls = []
    validation_calls = []
    fidelity_calls = []

    def fake_generate(spec, available, feedback=None):
        gen_calls.append({"spec": dict(spec), "feedback": feedback})
        return generated[len(gen_calls) - 1]

    def fake_validate(impl_path, mid, bundle, claim, cost_frac, prerun_result=None,
                      validation_meta=None):
        source = impl_path.read_text()
        validation_calls.append(source)
        if validation_meta is not None:
            validation_meta["unit_test"] = len(validation_calls)
        return True, SimpleNamespace(
            __file__=str(impl_path),
            module_id=mid,
            source=source,
            validation=len(validation_calls),
        )

    def fake_assess(claim, module_code, spec=None):
        fidelity_calls.append({"claim": claim, "module_code": module_code, "spec": spec})
        return fid_results[len(fidelity_calls) - 1]

    monkeypatch.setattr(impl_gen, "_generate_code", fake_generate)
    monkeypatch.setattr(impl_gen, "_validate_module", fake_validate)
    monkeypatch.setattr(sandbox, "run_in_container", lambda *a, **k: {"ok": True})
    if fid_results is not None:
        monkeypatch.setattr(fidelity, "assess", fake_assess)

    return gen_calls, validation_calls, fidelity_calls


def _unfaithful(divergences):
    return {
        "faithful": False,
        "verified": True,
        "confidence": 0.9,
        "divergences": divergences,
        "note": "unfaithful",
    }


def test_impl_fidelity_self_correction_retries_then_accepts_faithful(tmp_path, monkeypatch):
    from penrose.pipeline import impl_gen

    divergences = ["sampled every 10th day instead of using a 10-day trailing lookback"]
    gen_calls, _validation_calls, fidelity_calls = _install_common_fakes(
        tmp_path,
        monkeypatch,
        [_code("attempt_1"), _code("attempt_2")],
        [
            _unfaithful(divergences),
            {"faithful": True, "verified": True, "confidence": 0.95,
             "divergences": [], "note": "faithful"},
        ],
    )

    out = impl_gen.try_implement(_spec(), _claim(), bundle=object(), cost_frac=0.001,
                                 use_llm=True, max_attempts=3)

    assert out["ok"] is True
    assert out["attempts"] == 2
    assert "attempt_2" in out["module"].source
    assert len(gen_calls) == 2
    assert len(fidelity_calls) == 2
    assert gen_calls[0]["feedback"] is None
    assert "FIDELITY:" in gen_calls[1]["feedback"][1]
    assert divergences[0] in gen_calls[1]["feedback"][1]
    assert out["fidelity_attempts"][0]["divergences"] == divergences
    assert out["fidelity_attempts"][1]["faithful"] is True


def test_impl_fidelity_self_correction_exhausted_returns_last_validated(tmp_path, monkeypatch):
    from penrose.pipeline import impl_gen

    max_attempts = 3
    gen_calls, _validation_calls, fidelity_calls = _install_common_fakes(
        tmp_path,
        monkeypatch,
        [_code("attempt_1"), _code("attempt_2"), _code("attempt_3")],
        [
            _unfaithful(["uses a 1-day return instead of the claimed 10-day lookback"]),
            _unfaithful(["trades the inverse direction of the claimed momentum rule"]),
            _unfaithful(["omits the required BTC spot input from the signal"]),
        ],
    )

    out = impl_gen.try_implement(_spec(), _claim(), bundle=object(), cost_frac=0.001,
                                 use_llm=True, max_attempts=max_attempts)

    assert out["ok"] is True
    assert out["attempts"] == max_attempts
    assert "attempt_3" in out["module"].source
    assert len(gen_calls) == max_attempts
    assert len(fidelity_calls) == max_attempts
    assert len(out["fidelity_attempts"]) == max_attempts
    assert all(a["faithful"] is False for a in out["fidelity_attempts"])


def test_impl_fidelity_self_correction_fidelity_off_preserves_existing_behavior(
    tmp_path, monkeypatch
):
    from penrose import config
    from penrose.pipeline import fidelity, impl_gen

    monkeypatch.setattr(config, "FIDELITY_CHECK", False)
    gen_calls, _validation_calls, fidelity_calls = _install_common_fakes(
        tmp_path,
        monkeypatch,
        [_code("attempt_1")],
        fid_results=None,
    )

    def fail_assess(*args, **kwargs):
        raise AssertionError("fidelity should not be called")

    monkeypatch.setattr(config, "FIDELITY_CHECK", False)
    monkeypatch.setattr(fidelity, "assess", fail_assess)

    out = impl_gen.try_implement(_spec(), _claim(), bundle=object(), cost_frac=0.001,
                                 use_llm=True, max_attempts=3)

    assert out["ok"] is True
    assert out["attempts"] == 1
    assert "attempt_1" in out["module"].source
    assert len(gen_calls) == 1
    assert fidelity_calls == []
    assert "fidelity_attempts" not in out
