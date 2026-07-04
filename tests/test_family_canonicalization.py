import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose.brain import Claim
from penrose.pipeline import run as runmod


class _NovelModule:
    __strategy_class__ = "novel-module-class"


class _TrustedModule:
    __strategy_class__ = "crypto_funding_carry"


def _claim(claim_id: str, statement: str, cls: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        statement=statement,
        mechanism="",
        scope="",
        horizon="",
        source_id="paper",
        source_span="",
        claimed_metric_quote="",
        applicable_strategy_class=cls,
    )


def test_unregistered_paper_classes_share_external_domain_family(monkeypatch):
    monkeypatch.setattr(runmod, "_REGISTRY_CANONICAL_OWNERS", {})
    a = _claim("a", "BTC funding effect", "paper-invented-carry")
    b = _claim("b", "Bitcoin perp funding effect", "fresh-paper-label")

    assert runmod._family(a, _NovelModule()) == "external::crypto"
    assert runmod._family(a, _NovelModule()) == runmod._family(b, _NovelModule())


def test_registered_operator_class_keeps_class_scoped_family(monkeypatch):
    monkeypatch.setattr(
        runmod,
        "_REGISTRY_CANONICAL_OWNERS",
        {"crypto_funding_carry": "crypto_funding_carry"},
    )
    claim = _claim("trusted", "BTC funding carry", "crypto_funding_carry")

    assert runmod._family(claim, _TrustedModule()) == "crypto_funding_carry::crypto"
