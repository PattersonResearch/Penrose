import json
import os


def _claim():
    from penrose.brain import Claim

    return Claim(
        claim_id="fid-c1",
        statement="unit claim",
        mechanism="unit",
        scope="unit",
        horizon="1d",
        source_id="unit",
        source_span="unit claim",
        claimed_metric_quote="",
    )


def _install_fake_provider(monkeypatch, tmp_path):
    from penrose import config, llm

    calls = []

    class FakeProvider:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key or os.environ.get("PENROSE_LLM_API_KEY", "")
            self.base_url = (base_url or os.environ.get("PENROSE_LLM_BASE_URL")
                             or "https://default.example/v1").rstrip("/")

        @property
        def available(self):
            return bool(self.api_key)

        def complete(self, model, messages, **kwargs):
            calls.append({"base_url": self.base_url, "api_key": self.api_key, "model": model})
            if "fail-verifier" in self.base_url:
                raise RuntimeError("verifier unavailable")
            return {
                "text": json.dumps({
                    "faithful": True,
                    "confidence": 0.91,
                    "divergences": [],
                    "note": "ok",
                }),
                "model": model,
                "in_tokens": 10,
                "out_tokens": 10,
                "finish_reason": "stop",
            }

    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / ".llm_cache")
    monkeypatch.setitem(config.LLM_ROLES["fidelity_refuter"], "model", "verifier-model")
    monkeypatch.setattr(llm, "OpenAICompatProvider", FakeProvider)
    monkeypatch.setattr(llm, "_PROVIDER", None)
    llm.reset_budget()
    return calls


def test_fidelity_refuter_routes_to_configured_independent_provider(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import fidelity

    calls = _install_fake_provider(monkeypatch, tmp_path)
    monkeypatch.setenv("PENROSE_LLM_BASE_URL", "https://default.example/v1")
    monkeypatch.setenv("PENROSE_LLM_API_KEY", "default-key")
    monkeypatch.setattr(config, "VERIFIER_LLM_BASE_URL", "https://verifier.example/v1")
    monkeypatch.setattr(config, "VERIFIER_LLM_API_KEY", "verifier-key")

    result = fidelity.assess(_claim(), "def run(bundle, claim, cost): return {}")

    assert calls == [{
        "base_url": "https://verifier.example/v1",
        "api_key": "verifier-key",
        "model": "verifier-model",
    }]
    assert result["independent_verifier"] is True
    assert result["verified"] is True


def test_fidelity_refuter_defaults_to_same_provider_when_unset(tmp_path, monkeypatch):
    from penrose import config, llm

    calls = _install_fake_provider(monkeypatch, tmp_path)
    monkeypatch.setenv("PENROSE_LLM_BASE_URL", "https://default.example/v1")
    monkeypatch.setenv("PENROSE_LLM_API_KEY", "default-key")
    monkeypatch.setattr(config, "VERIFIER_LLM_BASE_URL", "")
    monkeypatch.setattr(config, "VERIFIER_LLM_API_KEY", "")

    response = llm.call(
        "fidelity_refuter",
        [{"role": "user", "content": "check"}],
        json_mode=True,
        use_cache=False,
    )

    assert calls == [{
        "base_url": "https://default.example/v1",
        "api_key": "default-key",
        "model": "verifier-model",
    }]
    assert response.independent_verifier is False


def test_fidelity_refuter_verifier_failure_falls_back_same_provider(tmp_path, monkeypatch):
    from penrose import config
    from penrose.pipeline import fidelity

    calls = _install_fake_provider(monkeypatch, tmp_path)
    monkeypatch.setenv("PENROSE_LLM_BASE_URL", "https://default.example/v1")
    monkeypatch.setenv("PENROSE_LLM_API_KEY", "default-key")
    monkeypatch.setattr(config, "VERIFIER_LLM_BASE_URL", "https://fail-verifier.example/v1")
    monkeypatch.setattr(config, "VERIFIER_LLM_API_KEY", "verifier-key")

    result = fidelity.assess(_claim(), "def run(bundle, claim, cost): return {}")

    assert calls == [
        {"base_url": "https://fail-verifier.example/v1", "api_key": "verifier-key",
         "model": "verifier-model"},
        {"base_url": "https://default.example/v1", "api_key": "default-key",
         "model": "verifier-model"},
    ]
    assert result["independent_verifier"] is False
    assert result["verified"] is True
