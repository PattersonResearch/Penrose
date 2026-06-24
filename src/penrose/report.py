"""P8 report one-pager. Stat-block philosophy (HUB_DESIGN_SPEC §4.2): pair every
flattering metric with its deflating one — DSR next to trial count, OOS Sharpe
next to capacity, edge next to vol-trade cost."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import config


def write_report(source_id, title, claims, decisions, provenance, principle) -> Path:
    by_id = {c.claim_id: c for c in claims}
    lines = [
        f"# penrose report — {source_id}",
        f"\n**{title}**",
        f"\n_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"`● RESEARCH ENGINE — NO LIVE TRADING`_\n",
        "## Data provenance\n",
        "| series | provenance | window | n |",
        "|---|---|---|---|",
    ]
    for k, v in provenance.items():
        if v.get("provenance") == "unavailable":
            lines.append(f"| {k} | **unavailable** | — | {v.get('reason','')} |")
        else:
            lines.append(f"| {k} | {v['provenance']} | {v.get('from')}→{v.get('to')} | {v.get('n')} |")
    if any(v.get("provenance") == "synthetic" for v in provenance.values()):
        lines.append("\n> ⚠️ Some inputs are **synthetic** — verdicts below are provisional, "
                     "pending live Kalshi macro-signal collection.\n")

    lines.append("\n## Verdicts (per-claim)\n")
    for d in decisions:
        c = by_id.get(d.claim_id)
        m = d.metrics or {}
        lines += [
            f"### {d.verdict.upper()} — {d.claim_id}",
            f"\n> {c.statement if c else ''}\n",
            f"- **Source span:** _{c.source_span}_" if c else "",
            f"- **Claimed:** {c.claimed_metric_quote}" if c else "",
            f"- **kill_reason:** `{d.kill_reason}`" if d.kill_reason else "- **kill_reason:** —",
            f"- **rationale:** {d.rationale}",
            f"- **evidence provenance:** `{m.get('fidelity_provenance', 'unknown')}`",
            "",
            "| OOS PSR | DSR | OOS Sharpe | edge_t | 3-fold | capacity $ | trades |",
            "|---|---|---|---|---|---|---|",
            f"| {m.get('psr')} | {m.get('dsr')} | {m.get('oos_sharpe')} | {m.get('edge_t')} "
            f"| {m.get('three_fold')} | {m.get('capacity_usd')} | {m.get('n_trades')} |",
        ]
        ho = (m.get("holdout") or {})
        if ho:
            lines.append("\n- **Single-use holdout:** "
                         f"Sharpe {ho.get('holdout_sharpe')}, "
                         f"PSR {ho.get('holdout_psr')}, nbars {ho.get('nbars')}")
        try:
            psr = float(m.get("psr"))
            dsr = float(m.get("dsr"))
        except (TypeError, ValueError):
            psr = dsr = None
        if psr is not None and dsr is not None and psr > 0 and abs(dsr - psr) < 1e-9:
            lines.append("- **Deflation note:** scored by PSR; deflation engages once the "
                         "search family has multiple trials.")
        res = m.get("resolution")
        if res:
            edge = res.get("current_mde_ic")
            bars = res.get("needed_oos_bars")
            breadth = res.get("needed_breadth_n")
            target = config.POWER["realistic_ic_floor"]
            lines.append(f"- **Resolution:** current MDE IC ~{edge}; to resolve a {target} IC edge: "
                         f"~{bars} OOS trades, or breadth >= {breadth} names")
        cs = m.get("cost_sensitivity") or {}
        if cs:
            cfg_bps = 1e4 * (cs.get("configured_cost_frac") or 0.0)
            be = cs.get("breakeven_cost_frac")
            if be is None:
                lines.append(f"- **Cost sensitivity:** robust through tested grid; configured cost "
                             f"{cfg_bps:.1f} bps")
            else:
                margin = cs.get("margin")
                lines.append(f"- **Cost sensitivity:** survives to ~{1e4*be:.1f} bps round-trip; "
                             f"configured cost {cfg_bps:.1f} bps"
                             f"{f' (margin {margin}x)' if margin is not None else ''}")
        ci = m.get("corpus_isolation") or {}
        if ci:
            lines.append(f"- **Corpus isolation:** {ci.get('advisory')} "
                         f"(neighbors={ci.get('neighbor_count')}, score={ci.get('isolation_score')})")
        lines.append("")

    lines.append("## Principle proposal (P8 → P9)\n")
    if principle:
        lines.append(f"- **{principle.statement}** (N={principle.n_observations}, "
                     f"conf={principle.confidence}) — *pending human approval*")
    else:
        lines.append("- None. BTC-only run yields ≤2 kills; principle extraction needs "
                     "N≥3 with a consistent kill_reason (O9). Sub-threshold.")

    lines.append("\n## Commit status\n")
    lines.append("All verdicts above are **proposals** queued in Action Required. "
                 "Nothing is committed to the brain until approved at P9 (the firewall).")

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS / f"{source_id}.md"
    path.write_text("\n".join(x for x in lines if x is not None))
    return path
