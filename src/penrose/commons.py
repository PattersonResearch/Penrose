"""Opt-in local corpus commons client.

Kill records imported from the commons are advisory concepts only. They never
create or alter Penrose verdicts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__, config

SCHEMA_VERSION = 1
KILL_RECORD_REQUIRED = {
    "schema_version",
    "claim_statement",
    "claim_type",
    "strategy_family",
    "spec",
    "verdict",
    "kill_reason",
    "receipt",
    "shared_at",
}
KILL_RECORD = {
    "schema_version": SCHEMA_VERSION,
    "required": sorted(KILL_RECORD_REQUIRED),
    "verdict": "kill",
}
GATE_CONFIG_KEYS = (
    "FEE_CURVE",
    "VOL_TRADE_COST",
    "IMPACT_COEF_BPS_PER_1M",
    "IMPLAUSIBILITY",
    "DSR_DECISION",
    "HOLDOUT_CONFIRM_PSR",
    "POWER",
    "POST_SAMPLE",
    "DEFLATION_PRIOR",
    "BOOTSTRAP",
    "PERMUTATION",
    "REGIME_FRAGILITY",
    "FRAGILITY_GATE",
    "WALK_FORWARD",
    "CPCV",
    "ROBUSTNESS_GATES",
    "REGIME_ADHERENCE_MIN",
    "COST_SENSITIVITY_GATE",
    "TAIL_RISK_GATE",
    "COST_PROVENANCE",
    "FIDELITY_CHECK",
    "FIDELITY_KILL_CONFIDENCE",
)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_normalize(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return _normalize(value.item())
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _iso_ts(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _format_obs_value(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.10g}"


def _series_items(series: Any) -> list[tuple[Any, Any]]:
    if hasattr(series, "items") and not isinstance(series, dict):
        return list(series.items())
    if isinstance(series, dict):
        return sorted(series.items(), key=lambda kv: str(kv[0]))
    return list(enumerate(series or []))


def merkle_root(series: Any) -> str:
    """Deterministic Merkle root over ``iso_ts|value`` observation leaves.

    Odd nodes are duplicated at each level. Empty inputs return the sha256 of an
    empty byte string so the commitment is still deterministic.
    """
    leaves = [
        hashlib.sha256(f"{_iso_ts(ts)}|{_format_obs_value(value)}".encode("utf-8")).hexdigest()
        for ts, value in _series_items(series)
    ]
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = leaves
    while len(level) > 1:
        if len(level) % 2 == 1:
            level = [*level, level[-1]]
        level = [
            hashlib.sha256((level[i] + level[i + 1]).encode("ascii")).hexdigest()
            for i in range(0, len(level), 2)
        ]
    return level[0]


def gate_config_block() -> dict:
    return {key: getattr(config, key) for key in GATE_CONFIG_KEYS if hasattr(config, key)}


def config_hash() -> str:
    return sha256_json(gate_config_block())


def _get(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _gate_outputs(decision: Any) -> dict:
    metrics = _get(decision, "metrics", {}) or {}
    if not isinstance(metrics, dict):
        metrics = {}
    out = {
        key: metrics.get(key)
        for key in ("dsr", "psr", "edge_t", "n_oos")
        if metrics.get(key) is not None
    }
    for key in (
        "three_fold",
        "regime",
        "power_sufficient",
        "mde_ic",
        "bootstrap_edge_ci",
        "permutation_p",
        "walk_forward_is",
        "cpcv_overfit_prob",
        "tail_risk",
    ):
        if metrics.get(key) is not None:
            out[key] = metrics.get(key)
    summary = _get(decision, "gate_summary", None) or metrics.get("gate_summary")
    if summary is not None:
        out["gate_summary"] = summary
    return out


def build_receipt(
    decision: Any,
    spec: dict,
    bundle_footprint: dict,
    series_hashes: dict[str, str],
) -> dict:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "claim_statement": str(
            _get(decision, "claim_statement")
            or _get(decision, "statement")
            or _get(decision, "rationale")
            or ""
        ),
        "spec_hash": sha256_json(spec or {}),
        "engine_version": __version__,
        "config_hash": config_hash(),
        "data_footprint": _normalize(bundle_footprint or {}),
        "series_merkle_roots": _normalize(series_hashes or {}),
        "verdict": str(_get(decision, "verdict", "")),
        "kill_reason": _get(decision, "kill_reason"),
        "gate_outputs": _gate_outputs(decision),
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def sign_receipt(receipt: dict, private_key_pem: str | bytes | None) -> dict:
    """Optionally attach a signature over ``receipt_hash``.

    The core commons client only needs hashlib. If cryptography is unavailable,
    the receipt is returned unsigned with a note rather than making signing a
    required dependency.
    """
    out = dict(receipt)
    if not private_key_pem:
        return out
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    except Exception:  # noqa: BLE001
        out["signature_note"] = "cryptography unavailable; receipt left unsigned"
        return out
    try:
        key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8") if isinstance(private_key_pem, str) else private_key_pem,
            password=None,
        )
        payload = str(out.get("receipt_hash", "")).encode("ascii")
        if isinstance(key, rsa.RSAPrivateKey):
            sig = key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
            alg = "rsa-pkcs1v15-sha256"
        elif isinstance(key, ec.EllipticCurvePrivateKey):
            sig = key.sign(payload, ec.ECDSA(hashes.SHA256()))
            alg = "ecdsa-sha256"
        else:
            out["signature_note"] = "unsupported private key type; receipt left unsigned"
            return out
        out["signature"] = sig.hex()
        out["signature_alg"] = alg
        # S6F-1: attach the public key so a recipient can actually verify the signature. (Whether to TRUST
        # this key is the reputation layer's job; attaching it makes the origin/integrity primitive functional
        # end-to-end instead of every signed record being rejected as missing_public_key.)
        out["public_key_pem"] = key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        out["signature_note"] = f"signing failed: {type(exc).__name__}; receipt left unsigned"
    return out


def _verify_signature(receipt: dict) -> tuple[bool, str]:
    signature = receipt.get("signature")
    public_key_pem = receipt.get("public_key_pem")
    if not signature:
        return True, "unsigned"
    if not public_key_pem:
        return False, "missing_public_key"
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    except Exception:  # noqa: BLE001
        return True, "signature_present_but_cryptography_unavailable"
    try:
        key = serialization.load_pem_public_key(str(public_key_pem).encode("utf-8"))
        payload = str(receipt.get("receipt_hash", "")).encode("ascii")
        sig = bytes.fromhex(str(signature))
        if isinstance(key, rsa.RSAPublicKey):
            key.verify(sig, payload, padding.PKCS1v15(), hashes.SHA256())
        elif isinstance(key, ec.EllipticCurvePublicKey):
            key.verify(sig, payload, ec.ECDSA(hashes.SHA256()))
        else:
            return False, "unsupported_public_key_type"
        return True, "signature_ok"
    except Exception:  # noqa: BLE001
        return False, "signature_invalid"


def data_footprint_from_row(row: dict) -> dict:
    provenance = row.get("data_provenance") if isinstance(row.get("data_provenance"), dict) else {}
    domains = provenance.get("data_domains") or (
        [provenance.get("data_domain")] if provenance.get("data_domain") else []
    )
    datasets = sorted(str(x) for x in (provenance.get("datasets") or []) if str(x or ""))
    bundle = provenance.get("bundle") if isinstance(provenance.get("bundle"), dict) else {}
    vendors = set()
    for key in ("vendors", "vendor_names", "sources"):
        raw = provenance.get(key) or bundle.get(key)
        if isinstance(raw, (list, tuple, set)):
            vendors.update(str(x) for x in raw if str(x or ""))
        elif raw:
            vendors.add(str(raw))
    for value in bundle.values():
        if isinstance(value, dict):
            vendor = value.get("vendor") or value.get("source")
            if vendor:
                vendors.add(str(vendor))
    series_count = provenance.get("series_count")
    if series_count is None:
        series_count = len(datasets)
    return {
        "domains": sorted(str(x) for x in domains if str(x or "")),
        "periods": list(provenance.get("periods") or []),
        "datasets": datasets,
        "vendors": sorted(vendors),
        "series_count": int(series_count or 0),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        text = path.read_text()
    except Exception:  # noqa: BLE001
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_spec_for_claim(row: dict) -> dict:
    for key in ("spec", "claim_spec"):
        if isinstance(row.get(key), dict):
            return dict(row[key])
    provenance = row.get("data_provenance") if isinstance(row.get("data_provenance"), dict) else {}
    for key in ("formulaic_signal_spec", "spec", "claim_spec"):
        if isinstance(provenance.get(key), dict):
            return dict(provenance[key])
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    spec_path = row.get("spec_path") or metrics.get("spec_path")
    candidates = []
    if spec_path:
        candidates.append(Path(spec_path))
    claim_id = str(row.get("claim_id") or "").strip()
    if claim_id:
        candidates.append(config.MODULES / "_specs" / f"{claim_id}.yaml")
    for path in candidates:
        try:
            if not path.exists():
                continue
            import yaml  # type: ignore

            loaded = yaml.safe_load(path.read_text()) or {}
            return loaded if isinstance(loaded, dict) else {}
        except Exception:  # noqa: BLE001
            continue
    return {}


def _series_roots_from_row(row: dict) -> dict[str, str]:
    roots = row.get("series_merkle_roots") or row.get("series_hashes")
    if isinstance(roots, dict):
        return {str(k): str(v) for k, v in roots.items()}
    provenance = row.get("data_provenance") if isinstance(row.get("data_provenance"), dict) else {}
    roots = provenance.get("series_merkle_roots") or provenance.get("series_hashes")
    if isinstance(roots, dict):
        return {str(k): str(v) for k, v in roots.items()}
    return {}


def _record_path_name(claim_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", claim_id or "claim").strip("-")
    return f"{slug or 'claim'}.killrecord.json"


def _commons_contributor() -> str:
    return os.environ.get("PENROSE_COMMONS_CONTRIBUTOR", "anonymous") or "anonymous"


def kill_record_from_row(row: dict, *, anonymize: bool = False) -> dict:
    if str(row.get("verdict") or "") != "kill":
        raise ValueError("commons share exports only kill verdicts")
    spec = _read_spec_for_claim(row)
    decision = {
        **row,
        # S6F-3: str-coerce so the stored record and the receipt (which str-coerces) agree — a non-string
        # claim_statement must not produce a record that fails its own validate_kill_record.
        "claim_statement": str(row.get("claim_statement") or row.get("statement") or ""),
    }
    receipt = build_receipt(
        decision,
        spec,
        data_footprint_from_row(row),
        _series_roots_from_row(row),
    )
    private_key = os.environ.get("PENROSE_COMMONS_PRIVATE_KEY", "")
    receipt = sign_receipt(receipt, private_key)
    record = {
        "schema_version": SCHEMA_VERSION,
        "claim_statement": decision["claim_statement"],
        "claim_type": str(row.get("claim_type") or spec.get("claim_type") or "unknown"),
        "strategy_family": row.get("strategy_family") or spec.get("strategy_family") or "",
        "spec": spec,
        "verdict": "kill",
        "kill_reason": row.get("kill_reason"),
        "receipt": receipt,
        "shared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not anonymize:
        record["contributor_id"] = _commons_contributor()
    return record


def export_kill_records(out_dir: str | Path, *, anonymize: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(Path(config.ANALYSIS_INDEX))
    if not rows:
        rows = _read_jsonl(Path(config.DECISIONS_LOG))
    exported = 0
    skipped_non_kill = 0
    invalid = 0
    files: list[str] = []
    for row in rows:
        if str(row.get("verdict") or "") != "kill":
            skipped_non_kill += 1
            continue
        try:
            record = kill_record_from_row(row, anonymize=anonymize)
            path = out / _record_path_name(str(row.get("claim_id") or record["receipt"]["receipt_hash"]))
            path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
            exported += 1
            files.append(str(path))
        except Exception:  # noqa: BLE001
            invalid += 1
    return {
        "exported": exported,
        "skipped_non_kill": skipped_non_kill,
        "invalid": invalid,
        "out": str(out),
        "files": files,
    }


def _receipt_without_hash(receipt: dict) -> dict:
    return {
        k: v for k, v in receipt.items()
        if k not in {"receipt_hash", "signature", "signature_alg", "signature_note", "public_key_pem"}
    }


def validate_kill_record(record: dict) -> tuple[bool, str]:
    if not isinstance(record, dict):
        return False, "record_not_object"
    missing = sorted(KILL_RECORD_REQUIRED - set(record))
    if missing:
        return False, f"missing_fields:{','.join(missing)}"
    if record.get("schema_version") != SCHEMA_VERSION:
        return False, "unsupported_schema_version"
    if str(record.get("verdict") or "") != "kill":
        return False, "non_kill_record"
    receipt = record.get("receipt")
    if not isinstance(receipt, dict):
        return False, "receipt_not_object"
    if receipt.get("schema_version") != SCHEMA_VERSION:
        return False, "receipt_schema_mismatch"
    expected_hash = sha256_json(_receipt_without_hash(receipt))
    if receipt.get("receipt_hash") != expected_hash:
        return False, "receipt_hash_mismatch"
    if receipt.get("claim_statement") != record.get("claim_statement"):
        return False, "claim_statement_mismatch"
    if receipt.get("verdict") != record.get("verdict"):
        return False, "verdict_mismatch"
    if receipt.get("kill_reason") != record.get("kill_reason"):
        return False, "kill_reason_mismatch"
    if receipt.get("spec_hash") != sha256_json(record.get("spec") or {}):
        return False, "spec_hash_mismatch"
    sig_ok, sig_reason = _verify_signature(receipt)
    if not sig_ok:
        return False, sig_reason
    return True, sig_reason


def _claim_spec_key(record: dict) -> str:
    receipt = record.get("receipt") or {}
    return sha256_json({
        "claim_statement": record.get("claim_statement") or receipt.get("claim_statement") or "",
        "spec_hash": receipt.get("spec_hash") or sha256_json(record.get("spec") or {}),
    })


def _existing_commons_keys(path: Path) -> tuple[set[str], set[str]]:
    receipt_hashes: set[str] = set()
    claim_spec_keys: set[str] = set()
    for row in _read_jsonl(path):
        provenance = row.get("data_provenance") if isinstance(row.get("data_provenance"), dict) else {}
        commons = provenance.get("commons") if isinstance(provenance.get("commons"), dict) else {}
        if commons.get("receipt_hash"):
            receipt_hashes.add(str(commons["receipt_hash"]))
        if commons.get("claim_spec_key"):
            claim_spec_keys.add(str(commons["claim_spec_key"]))
    return receipt_hashes, claim_spec_keys


def advisory_concept_from_record(record: dict) -> dict:
    receipt = record["receipt"]
    receipt_hash = str(receipt["receipt_hash"])
    claim_spec_key = _claim_spec_key(record)
    footprint = receipt.get("data_footprint") or {}
    return {
        "concept_id": f"commons-{receipt_hash[:16]}",
        "source_claim_id": f"commons:{receipt_hash[:16]}",
        "statement": str(record.get("claim_statement") or "")[:1000],
        "mechanism": "",
        "surviving_explanation": "",
        "rejected_explanations": [str(record.get("kill_reason") or "kill")],
        "boundary": {"kill_reason": record.get("kill_reason"), "commons_advisory": True},
        "reusable_principle": (
            f"Commons advisory negative evidence: an external Penrose run killed this claim "
            f"with reason '{record.get('kill_reason')}'. Treat as a prior only; independently "
            "test any new claim."
        )[:1000],
        "implementation_consequence": "Advisory prior only; never a verdict or automatic gate.",
        "evidence_strength": dict(receipt.get("gate_outputs") or {}),
        "data_provenance": {
            "source": "commons",
            "source_type": "external_source",
            "data_domains": list(footprint.get("domains") or []),
            "datasets": list(footprint.get("datasets") or []),
            "vendors": list(footprint.get("vendors") or []),
            "periods": list(footprint.get("periods") or []),
            "series_count": footprint.get("series_count", 0),
            "series_merkle_roots": dict(receipt.get("series_merkle_roots") or {}),
            "strategy_family": record.get("strategy_family") or "",
            "commons": {
                "receipt_hash": receipt_hash,
                "claim_spec_key": claim_spec_key,
                "spec_hash": receipt.get("spec_hash"),
                "schema_version": record.get("schema_version"),
                "contributor_id": record.get("contributor_id", "anonymous"),
            },
        },
        "source_type": "external_source",
        "abstraction_level": "observation",
        "created_at": str(record.get("shared_at") or "1970-01-01T00:00:00+00:00"),
        "seed": int(receipt_hash[:8], 16),
        "grounding_flags": ["commons:advisory_only"],
        "source_verdict": "kill",
        "evidence_direction": "negative",
    }


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _opportunistic_merkle_mismatches(
    record: dict,
    local_series_roots: dict[str, str] | None = None,
) -> list[str]:
    if not local_series_roots:
        return []
    expected = (record.get("receipt") or {}).get("series_merkle_roots") or {}
    mismatches = []
    for name, root in expected.items():
        if name in local_series_roots and str(local_series_roots[name]) != str(root):
            mismatches.append(str(name))
    return sorted(mismatches)


def commons_pull_dir(
    directory: str | Path,
    *,
    concepts_path: str | Path | None = None,
    local_series_roots: dict[str, str] | None = None,
) -> dict:
    root = Path(directory)
    concept_path = Path(concepts_path or config.CONCEPTS)
    existing_receipts, existing_claim_specs = _existing_commons_keys(concept_path)
    ingested = duplicate = invalid = merkle_mismatch = 0
    invalid_files: list[str] = []
    mismatch_files: list[str] = []
    for path in sorted(root.glob("*.killrecord.json")):
        try:
            record = json.loads(path.read_text())
            ok, reason = validate_kill_record(record)
            if not ok:
                invalid += 1
                invalid_files.append(f"{path.name}:{reason}")
                continue
            receipt_hash = str(record["receipt"]["receipt_hash"])
            claim_spec_key = _claim_spec_key(record)
            if receipt_hash in existing_receipts or claim_spec_key in existing_claim_specs:
                duplicate += 1
                continue
            mismatches = _opportunistic_merkle_mismatches(record, local_series_roots)
            if mismatches:
                merkle_mismatch += 1
                mismatch_files.append(f"{path.name}:{','.join(mismatches)}")
            concept = advisory_concept_from_record(record)
            _append_jsonl(concept_path, concept)
            existing_receipts.add(receipt_hash)
            existing_claim_specs.add(claim_spec_key)
            ingested += 1
        except Exception as exc:  # noqa: BLE001
            invalid += 1
            invalid_files.append(f"{path.name}:{type(exc).__name__}")
            continue
    return {
        "ingested": ingested,
        "duplicate": duplicate,
        "invalid": invalid,
        "merkle_mismatch": merkle_mismatch,
        "invalid_files": invalid_files,
        "mismatch_files": mismatch_files,
    }
