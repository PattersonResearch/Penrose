"""Download + build the Chen-Zimmermann long-short anomaly panel that `cz_referee.py` referees.

One-time fetch so `make cz-referee` works from a fresh clone. Pulls the public Open Source Asset
Pricing "original paper" (op) long-short portfolios via the openassetpricing package (by the CZ
authors themselves), pivots them into a months x anomalies return panel signed to each published
direction, and writes it to penrose-data/literature/chen_zimmermann/ls_panel.parquet.

openassetpricing is an OPTIONAL, reproduction-only dependency (it is not in the core install, and it
pins an older pandas — install it in a throwaway/scratch env if you want to keep your penrose env
pristine). This script only needs it at fetch time; refereeing the built parquet does not.

Run:  python scripts/fetch_cz_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OUT_DIR = Path.home() / "Development/penrose-data/literature/chen_zimmermann"
OUT = OUT_DIR / "ls_panel.parquet"


def main() -> int:
    try:
        import pandas as pd
        import openassetpricing as oap
    except ImportError:
        print("This fetch needs openassetpricing + pandas. Install with:\n"
              "  pip install openassetpricing\n"
              "(optional, reproduction-only — note it pins an older pandas, so a scratch venv is safest.)",
              file=sys.stderr)
        return 1

    print("downloading Open Source Asset Pricing 'op' long-short portfolios (~20 MB public data)...")
    m = oap.OpenAP()
    df = m.dl_port("op", "pandas")
    ls = df[df["port"] == "LS"]
    panel = (ls.assign(date=pd.to_datetime(ls["date"]))
               .pivot_table(index="date", columns="signalname", values="ret"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT)
    print(f"wrote {OUT}  ({panel.shape[0]} months x {panel.shape[1]} anomalies)")
    if panel.shape[1] < 200:
        print(f"warning: expected ~212 anomalies, got {panel.shape[1]} — upstream data may have changed",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
