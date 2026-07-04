"""P9 — human review gate (the COMMIT half of the firewall).

This is the ONLY place a read-write PromotionClient is constructed, and only when
a human passes --approver. `list` and `show` are read-only; `approve` promotes a
proposal to the brain as a committed atom with typed-edge provenance; `reject`
returns it with a reason.
"""
from __future__ import annotations

import argparse
import json

from .. import config
from ..brain import PromotionClient, slug


def _load_queue() -> list[dict]:
    if not config.REVIEW_QUEUE.exists():
        return []
    return [json.loads(l) for l in config.REVIEW_QUEUE.read_text().splitlines() if l.strip()]


def _save_queue(rows) -> None:
    config.REVIEW_QUEUE.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")


def _append_principle(row: dict) -> None:
    config.PRINCIPLES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with config.PRINCIPLES_LOG.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def cmd_list(_args) -> None:
    rows = _load_queue()
    pending = [r for r in rows if r.get("status", "pending") == "pending"]
    print(f"Action Required — {len(pending)} pending proposal(s):\n")
    for i, r in enumerate(rows):
        if r.get("status", "pending") != "pending":
            continue
        if r["type"] == "decision":
            print(f"  [{i}] decision  {r['verdict'].upper():18s} {r['claim_id']}  "
                  f"(kill_reason={r.get('kill_reason')})")
        elif r["type"] == "engine_error":
            print(f"  [{i}] ENGINE ERROR {r.get('claim_id')}  "
                  f"{str(r.get('rationale', ''))[:60]}")
        else:
            print(f"  [{i}] principle {r.get('principle_id')}  N={r.get('n_observations')}")


def cmd_show(args) -> None:
    rows = _load_queue()
    print(json.dumps(rows[args.idx], indent=2, default=str))


def cmd_approve(args) -> None:
    if not args.approver:
        raise SystemExit("refuse: --approver required (this is the human commit gate)")
    rows = _load_queue()
    r = rows[args.idx]
    if r["type"] == "engine_error":
        raise SystemExit("refuse: engine errors are fixed in code, not approved")
    pc = PromotionClient(approved_by=args.approver)

    if r["type"] == "decision":
        body = (f"Verdict: **{r['verdict']}** (kill_reason={r.get('kill_reason')})\n\n"
                f"{r.get('rationale','')}\n\nClaim: {r.get('claim_statement','')}\n\n"
                f"Metrics: {json.dumps(r.get('metrics', {}), default=str)}")
        res = pc.put_atom("decision", r["decision_id"], body,
                          verdict=r["verdict"], kill_reason=r.get("kill_reason"),
                          claim_id=r["claim_id"], trust=0.7, verified_by_human=True)
        # typed-edge provenance: decision killed_by/derived_from its claim & source
        claim_slug = slug("claim", r["claim_id"])
        pc.link(res["slug"], claim_slug, "evaluated_in")
        print(f"committed decision {res['slug']} (ok={res['ok']}) + edge -> {claim_slug}")
    else:
        body = (f"{r['statement']}\n\nSupporting kills: {r.get('supporting_kills')}\n"
                f"N={r.get('n_observations')} confidence={r.get('confidence')}")
        res = pc.put_atom("principle", r["principle_id"], body,
                          n_observations=r.get("n_observations"),
                          confidence=r.get("confidence"),
                          applicable_strategy_classes=r.get("applicable_strategy_classes"),
                          trust=0.6)
        for k in r.get("supporting_kills", []):
            pc.link(res["slug"], slug("decision", k), "supports_principle")
        _append_principle({
            "principle_id": r["principle_id"],
            "statement": r["statement"],
            "supporting_kills": r.get("supporting_kills", []),
            "applicable_strategy_classes": r.get("applicable_strategy_classes", []),
            "n_observations": r.get("n_observations"),
            "confidence": r.get("confidence"),
            "status": "approved",
            "approved_by": args.approver,
        })
        print(f"committed principle {res['slug']} (ok={res['ok']})")

    r["status"] = "approved"
    r["approved_by"] = args.approver
    _save_queue(rows)


def cmd_reject(args) -> None:
    rows = _load_queue()
    rows[args.idx]["status"] = "rejected"
    rows[args.idx]["reject_reason"] = args.reason or ""
    _save_queue(rows)
    print(f"rejected [{args.idx}] — returned to P8 (not committed)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="penrose-review")
    sub = ap.add_subparsers(required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    s = sub.add_parser("show"); s.add_argument("idx", type=int); s.set_defaults(func=cmd_show)
    s = sub.add_parser("approve"); s.add_argument("idx", type=int)
    s.add_argument("--approver", required=True); s.set_defaults(func=cmd_approve)
    s = sub.add_parser("reject"); s.add_argument("idx", type=int)
    s.add_argument("--reason", default=""); s.set_defaults(func=cmd_reject)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
