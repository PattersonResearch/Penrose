"""Human-review explanations for non-verdict routing states."""
from __future__ import annotations

import re


def _clean_text(value, *, fallback: str = "not specified", limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return fallback
    if "Traceback (most recent call last)" in text:
        text = text.split("Traceback (most recent call last)", 1)[0].strip() or fallback
    return text[:limit]


def _series_name(value) -> str:
    if isinstance(value, dict):
        if value.get("kind") == "derived_series":
            transform = str(value.get("transform") or "").strip()
            base = str(value.get("base_series") or "").strip()
            window = value.get("window")
            if transform == "realized_vol":
                return f"realized_vol({base}, {window})"
            if transform:
                return f"{transform}({base})"
            return base or "unresolved"
        return str(
            value.get("series")
            or value.get("id")
            or value.get("name")
            or value.get("key")
            or "unresolved"
        ).strip()
    return str(value or "").strip() or "unresolved"


def _candidate_from_provenance(prov: dict, value) -> str:
    kind = str((prov or {}).get("kind") or "").lower()
    if kind in {"derived", "derived_series"}:
        return _series_name({
            "kind": "derived_series",
            "transform": prov.get("transform"),
            "base_series": prov.get("base_series"),
            "window": prov.get("window"),
        })
    return str((prov or {}).get("series") or _series_name(value) or "").strip()


def _confidence_from_provenance(prov: dict) -> float | None:
    if not isinstance(prov, dict):
        return None
    if str(prov.get("kind") or "").lower() in {"derived", "derived_series"}:
        prov = prov.get("base_resolution") or {}
    try:
        return float(prov.get("score"))
    except (TypeError, ValueError):
        return None


def _matched_tokens_from_provenance(prov: dict) -> list[str]:
    if not isinstance(prov, dict):
        return []
    if str(prov.get("kind") or "").lower() in {"derived", "derived_series"}:
        prov = prov.get("base_resolution") or {}
    out = []
    for token in prov.get("matched_tokens") or []:
        text = str(token or "").strip()
        if text:
            out.append(text)
    return out


def _binding_role_sentence(role: str, detail: dict) -> str:
    provenance = detail.get("binding_provenance") or {}
    prov = provenance.get(role) if isinstance(provenance, dict) else {}
    prov = prov if isinstance(prov, dict) else {}
    value = detail.get(role)
    prose = _clean_text(
        prov.get("description") or detail.get(f"{role}_prose"),
        fallback=f"{role} prose was not isolated",
        limit=220,
    )
    confirmed = bool((detail.get("confirmed") or {}).get(role))
    if str(prov.get("kind") or "").lower() == "unresolved" or _series_name(value) == "unresolved":
        return (
            f"{role.title()} prose '{prose}' did not resolve to a catalog series "
            f"and is not confirmed."
        )
    candidate = _candidate_from_provenance(prov, value)
    confidence = _confidence_from_provenance(prov)
    matched = _matched_tokens_from_provenance(prov)
    confidence_text = f"confidence {confidence:.2f}" if confidence is not None else "confidence unavailable"
    matched_text = f", matched: {', '.join(matched)}" if matched else ", matched tokens unavailable"
    status = "confirmed" if confirmed else "not confirmed"
    return (
        f"{role.title()} prose '{prose}' -> candidate `{candidate}` "
        f"({confidence_text}{matched_text}) - {status}."
    )


def _binding_action(detail: dict) -> str:
    predictor = _series_name(detail.get("predictor"))
    target = _series_name(detail.get("target"))
    predictor_text = predictor if predictor != "unresolved" else "the correct predictor catalog series"
    target_text = target if target != "unresolved" else "the correct target catalog series"
    return (
        f"Confirm predictor={predictor_text} and target={target_text}, or supply the "
        "correct catalog series, then re-run."
    )


def _factor_binding_action(detail: dict) -> str:
    candidate = _series_name(detail.get("candidate_factor"))
    candidate_text = (
        candidate if candidate != "unresolved" else "the correct candidate factor catalog series"
    )
    benchmark_set = _clean_text(detail.get("benchmark_set"), fallback="the declared benchmark set")
    return (
        f"Confirm candidate_factor={candidate_text} and benchmark_set={benchmark_set}, "
        "or supply the correct catalog series, then re-run."
    )


def _panel_name(value) -> str:
    if isinstance(value, dict):
        return str(
            value.get("path")
            or value.get("table")
            or value.get("table_path")
            or value.get("panel_path")
            or "unresolved"
        ).strip()
    return str(value or "").strip() or "unresolved"


def human_review_explanation(kind: str, detail: dict | None = None) -> dict:
    """Return operator-readable what/why/action strings for human review stops.

    The helper is intentionally fail-soft: a malformed detail payload still returns
    useful strings and never exposes stack traces as the primary operator message.
    """
    detail = detail or {}
    try:
        if kind == "predictive_regression_binding_uncertain":
            return {
                "what": (
                    "Routed to human review - I could not confirm the data binding "
                    "for this predictive-regression claim."
                ),
                "why": " ".join([
                    _binding_role_sentence("predictor", detail),
                    _binding_role_sentence("target", detail),
                ]),
                "action": _binding_action(detail),
            }
        if kind == "factor_spanning_binding_uncertain":
            candidate_detail = {
                "binding_provenance": {
                    "candidate_factor": (
                        (detail.get("binding_provenance") or {}).get("candidate_factor")
                        if isinstance(detail.get("binding_provenance"), dict) else {}
                    )
                },
                "candidate_factor": detail.get("candidate_factor"),
                "confirmed": detail.get("confirmed") or {},
            }
            candidate_sentence = _binding_role_sentence("candidate_factor", candidate_detail)
            benchmark_set = _clean_text(detail.get("benchmark_set"), fallback="unresolved")
            benchmarks = ", ".join(
                _series_name(x) for x in (detail.get("benchmark_factors") or [])
            ) or "unresolved"
            return {
                "what": (
                    "Routed to human review - I could not confirm the data binding "
                    "for this factor-spanning claim."
                ),
                "why": (
                    f"{candidate_sentence} Benchmark set `{benchmark_set}` maps to "
                    f"`{benchmarks}`."
                ),
                "action": _factor_binding_action(detail),
            }
        if kind == "cross_sectional_sort_binding_uncertain":
            panel_inputs = detail.get("panel_inputs") or {}
            returns_panel = _panel_name(panel_inputs.get("returns"))
            characteristic_panel = _panel_name(panel_inputs.get("characteristic"))
            characteristic = _clean_text(detail.get("characteristic"), fallback="unresolved")
            confirmed = detail.get("confirmed") or {}
            return {
                "what": (
                    "Routed to human review - I could not confirm the panel binding "
                    "for this cross-sectional-sort claim."
                ),
                "why": (
                    f"Characteristic `{characteristic}` confirmed={bool(confirmed.get('characteristic'))}. "
                    f"Returns panel `{returns_panel}` confirmed={bool(confirmed.get('returns_panel'))}. "
                    f"Characteristic panel `{characteristic_panel}` confirmed="
                    f"{bool(confirmed.get('characteristic_panel'))}."
                ),
                "action": (
                    "Confirm the characteristic, universe, returns panel, and point-in-time "
                    "characteristic panel; ensure the returns panel declares survivorship=corrected, "
                    "then re-run."
                ),
            }
        if kind == "auto_impl_no_progress":
            reason = _clean_text(detail.get("reason"), fallback="auto-implementation repeated the same failure")
            attempts = detail.get("attempts") or detail.get("attempts_tried")
            attempt_text = f" after {attempts} attempts" if attempts else ""
            return {
                "what": (
                    "Routed to human review - auto-implementation stopped because it "
                    "was not making progress."
                ),
                "why": f"The generated module kept failing with the same validation shape{attempt_text}: {reason}.",
                "action": (
                    "Inspect the ModuleSpec and validation failure, implement or correct "
                    "the module, then re-run."
                ),
            }
    except Exception:  # noqa: BLE001
        pass
    reason = _clean_text(detail.get("reason") if isinstance(detail, dict) else "", fallback="manual review is required")
    return {
        "what": "Routed to human review - the claim cannot be tested automatically yet.",
        "why": reason,
        "action": "Inspect the review item, correct the blocker, then re-run.",
    }
