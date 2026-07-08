"""Test/run the brain connection-discovery on penrose's real verdict corpus.

Loads analysis_index.jsonl (the live verdict corpus) -> normalized Records -> advisory connections
(clusters / cross-domain links / power-aware principles / similarity edges). Writes
dashboard/connections.json for the UI and prints a summary. INFORM-NEVER-GATE: pure read-only.

Run:  python scripts/brain_connections.py [--cz]
  --cz : also run the SAME discovery on the richer 212-anomaly Chen-Zimmermann corpus (diverse
         verdicts + categories) to show what the engine finds when the corpus has real variety.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from penrose import brain_connect as BC      # noqa: E402
from penrose.brain_connect import Record     # noqa: E402


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_live_corpus() -> list[Record]:
    rows = _read_jsonl(ROOT / "reports" / "analysis_index.jsonl")
    rows = list({r.get("claim_id"): r for r in rows}.values())   # dedup re-runs (keep latest)
    recs = []
    for r in rows:
        kr = r.get("kill_reason")
        verdict = r.get("verdict")
        text = f"{r.get('statement','')} {r.get('source_title','')}"
        metrics = r.get("metrics") or {}
        data_prov = r.get("data_provenance") if isinstance(r.get("data_provenance"), dict) else {}
        recs.append(Record(
            id=str(r.get("claim_id")), domain=BC.infer_domain(text),
            verdict=verdict, kill_reason=kr, statement=r.get("statement", "")[:240],
            structural=(verdict == "kill" and kr in BC.STRUCTURAL_KILLS),
            power_sufficient=metrics.get("power_sufficient"),
            date=r.get("run_at", ""), synthetic=bool(r.get("synthetic")),
            strategy_family=data_prov.get("strategy_family_structured")))
    return recs


def load_cz_corpus() -> list[Record]:
    """Richer test: run the 212 CZ anomalies through penrose, group by published category."""
    import os, tempfile
    os.environ["PENROSE_HOLDOUT_LOCK"] = os.path.join(tempfile.gettempdir(), "penrose_calib_holdout.lock")
    import pandas as pd
    from penrose import config
    config.COST_PROVENANCE = "measured"
    from penrose.pipeline import p7_backtest as P7, stages
    from penrose.brain import Claim
    lit = Path.home() / "Development/penrose-data/literature/chen_zimmermann"
    if not (lit / "ls_panel.parquet").exists():
        print(f"[connections --cz] Chen-Zimmermann data not found at {lit}")
        print("  The --cz corpus needs the published-anomaly panel, a separate data download.")
        print("  See notebooks/penrose_demo.ipynb. Skipping the --cz corpus.")
        return []
    panel = pd.read_parquet(lit / "ls_panel.parquet") / 100.0
    doc = pd.read_parquet(lit / "signal_doc.parquet").set_index("Acronym")
    catcol = "Cat.Economic" if "Cat.Economic" in doc.columns else None
    names = [c for c in panel.columns if panel[c].notna().sum() >= 30]
    tmp = Path("/tmp/_bc_cz.tsv"); old = P7.LEDGER; P7.LEDGER = tmp; tmp.unlink(missing_ok=True)
    recs = []
    try:
        for nm in names:
            net = panel[nm].dropna()
            P7.run_backtest(nm, net, pd.Series(1.0, index=net.index), 12.0, cost_frac=0.0,
                            family="cz::x", log=True)
        for nm in names:
            net = panel[nm].dropna()
            bt = P7.run_backtest(nm, net, pd.Series(1.0, index=net.index), 12.0, cost_frac=0.0,
                                 family="cz::x", log=False)
            c = Claim(claim_id=nm, statement="", mechanism="", scope="", horizon="",
                      source_id="cz", source_span="", claimed_metric_quote="", applicable_strategy_class="cz")
            d = stages.p8_verdict(c, bt, {}, False)
            dom = str(doc.loc[nm, catcol]) if (catcol and nm in doc.index) else "factor"
            recs.append(Record(id=nm, domain=f"cz:{dom}", verdict=d.verdict, kill_reason=d.kill_reason,
                               statement=nm, structural=(d.verdict == "kill" and d.kill_reason in BC.STRUCTURAL_KILLS),
                               power_sufficient=d.metrics.get("power_sufficient"), date="2024-12-31"))
    finally:
        P7.LEDGER = old; tmp.unlink(missing_ok=True)
    return recs


def _report(label: str, recs: list[Record]):
    c = BC.discover(recs)
    print(f"\n===== CONNECTIONS: {label} =====")
    print(f"corpus: {c.stats['n_records']} records | {c.stats['n_structural_kills']} structural kills | "
          f"{c.stats['n_underpowered']} underpowered | verdicts={c.stats['verdicts']}")
    print(f"domains: {c.stats['domains']}")
    print(f"\nFAILURE-MODE CLUSTERS ({len(c.failure_clusters)}):")
    for cl in c.failure_clusters[:8]:
        print(f"  [{cl['kill_reason']} × {cl['domain']}] n={cl['n']}  e.g. {cl['members'][:2]}")
    print(f"\nCROSS-DOMAIN LINKS ({len(c.cross_domain)}):")
    for cd in c.cross_domain[:6]:
        print(f"  '{cd['kill_reason']}' spans {cd['n_domains']} domains {cd['domains']} (total {cd['total']})")
    print(f"\nPRINCIPLES ({len(c.principles)}):")
    for p in c.principles[:6]:
        print(f"  [{p['n_observations']} obs, conf {p['confidence']}] {p['principle_id']}")
    print(f"\nSIMILARITY LINKS ({len(c.similarity_links)}): top {min(4,len(c.similarity_links))}")
    for s in c.similarity_links[:4]:
        print(f"  {s['similarity']}: {s['a'][:22]} ({s['a_verdict']}) ~ {s['b'][:22]} ({s['b_verdict']})")
    return c


def main():
    live = load_live_corpus()
    c = _report("LIVE penrose corpus (analysis_index)", live)
    (ROOT / "dashboard" / "connections.json").write_text(json.dumps({
        "failure_clusters": c.failure_clusters, "cross_domain": c.cross_domain,
        "principles": c.principles, "similarity_links": c.similarity_links, "stats": c.stats}, indent=2))
    print("\nwrote dashboard/connections.json (live corpus)")
    if "--cz" in sys.argv:
        _report("Chen-Zimmermann 212-anomaly corpus (richer)", load_cz_corpus())
    print("\nINFORM-NEVER-GATE check: this is advisory output; no verdict was created or changed.")


if __name__ == "__main__":
    main()
