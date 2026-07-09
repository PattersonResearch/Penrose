"""Corpus-commons client: reproducibility receipt + share/pull round-trip (#6f)."""
import json

import pandas as pd
import pytest

from penrose import commons


def _series(vals):
    idx = pd.date_range("2021-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx)


def test_merkle_root_deterministic_and_binds_bytes():
    s = _series([1.0, 2.0, 3.0, 4.0])
    r1 = commons.merkle_root(s)
    r2 = commons.merkle_root(_series([1.0, 2.0, 3.0, 4.0]))
    assert r1 == r2 and isinstance(r1, str) and r1
    # a single-value edit changes the root (data-binding / challengeable)
    assert commons.merkle_root(_series([1.0, 2.0, 3.0, 4.5])) != r1


def test_canonical_json_is_order_stable():
    assert commons.canonical_json({"b": 1, "a": 2}) == commons.canonical_json({"a": 2, "b": 1})


def _receipt(**over):
    decision = {"claim_statement": "S", "verdict": "kill", "kill_reason": "in_sample_only",
                "metrics": {"dsr": 0.1, "edge_t": 0.2}}
    decision.update(over.get("decision", {}))
    spec = over.get("spec", {"claim_type": "formulaic_signal", "signal": "sign(x)"})
    footprint = over.get("footprint", {"data_domains": ["crypto"], "series_count": 1,
                                       "periods": [{"start": "2021-01-01", "end": "2022-01-01"}]})
    roots = over.get("roots", {"x": "abc123"})
    return commons.build_receipt(decision, spec, footprint, roots)


def test_receipt_hash_deterministic_and_tamper_evident():
    r1 = _receipt()
    r2 = _receipt()
    assert r1["receipt_hash"] == r2["receipt_hash"]
    # changing any bound field changes the hash
    assert _receipt(spec={"claim_type": "formulaic_signal", "signal": "sign(y)"})["receipt_hash"] != r1["receipt_hash"]
    assert _receipt(roots={"x": "DIFFERENT"})["receipt_hash"] != r1["receipt_hash"]
    # receipt records engine version + data binding but NOT raw values
    assert "engine_version" in r1 and "series_merkle_roots" in r1


def test_share_exports_only_kills():
    with pytest.raises(ValueError):
        commons.kill_record_from_row({"claim_id": "c1", "verdict": "watch"})


def test_pull_ingests_advisory_concept_and_rejects_tamper(tmp_path):
    row = {"claim_id": "c1", "verdict": "kill", "kill_reason": "in_sample_only",
           "claim_statement": "momentum dies OOS", "claim_type": "formulaic_signal",
           "strategy_family": "time_series_momentum",
           "spec": {"claim_type": "formulaic_signal", "signal": "sign(returns(btc, 20))",
                    "strategy_family": "time_series_momentum"},
           "data_provenance": {"data_domains": ["crypto"], "periods": [{"start": "2021-01-01", "end": "2022-01-01"}]}}
    rec = commons.kill_record_from_row(row)
    assert rec["verdict"] == "kill" and rec["receipt"]["receipt_hash"]

    # a valid record ingests as an ADVISORY negative concept (a prior, never a verdict)
    concept = commons.advisory_concept_from_record(rec)
    assert concept["evidence_direction"] == "negative"
    assert "commons" in json.dumps(concept).lower()

    # write a dir with one valid + one tampered record; pull accepts the first, rejects the tamper
    (tmp_path / "good.killrecord.json").write_text(json.dumps(rec))
    bad = json.loads(json.dumps(rec))
    bad["receipt"]["receipt_hash"] = "0" * 64  # tamper: hash no longer matches contents
    (tmp_path / "bad.killrecord.json").write_text(json.dumps(bad))
    concepts_path = tmp_path / "concepts.jsonl"
    out = commons.commons_pull_dir(tmp_path, concepts_path=concepts_path)
    assert out.get("ingested", 0) >= 1
    assert out.get("invalid", 0) >= 1  # the tampered record was rejected

    # dedup: pulling again ingests 0 new
    out2 = commons.commons_pull_dir(tmp_path, concepts_path=concepts_path)
    assert out2.get("ingested", 0) == 0


def test_signed_record_verifies_end_to_end():
    """S6F-1: a signed receipt must carry its public key so a recipient can verify it (else every signed
    record is rejected). Skips gracefully if cryptography is unavailable (signing is optional)."""
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    r = commons.build_receipt({"claim_statement": "s", "verdict": "kill"},
                              {"claim_type": "formulaic_signal", "signal": "sign(x)"},
                              {"data_domains": ["crypto"]}, {"x": "abc"})
    r = commons.sign_receipt(r, pem)
    assert "public_key_pem" in r
    ok, _ = commons._verify_signature(r)
    assert ok
    r_tampered = dict(r); r_tampered["receipt_hash"] = "0" * 64
    assert not commons._verify_signature(r_tampered)[0]
