"""M0 gate harness (V3_SPEC 4): score ANY prediction file against the
pre-registered csfix game set with the G1 protocol.

    python scripts/gate_v3.py --pred <csv> [--col p_v3] [--json out.json]
    python scripts/gate_v3.py --smoke              # re-score p_us: delta == 0

The pred CSV needs columns: game_id + the probability column (default 'p_v3').
Joins on game_id to data/capstone_pergame_csfix.csv (columns season, game_id,
game_date, home, away, y, p_us, p_mkt, ...); paired_bootstrap_delta per season
+ pooled + the October slice (D16 territory, always inspected for M1).
Ship rule: keep iff the pooled 95% CI excludes zero (G1).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from nbapred.eval.ablate import paired_bootstrap_delta
from nbapred.eval.metrics import log_loss

BASE = REPO / "data" / "capstone_pergame_csfix.csv"


def gate(pred_df: pd.DataFrame, col: str = "p_v3", base_path=BASE,
         base_col: str = "p_us") -> dict:
    base = pd.read_csv(base_path, dtype={"game_id": str})
    df = base.merge(pred_df[["game_id", col]].astype({"game_id": str}),
                    on="game_id", how="inner", validate="1:1")
    if len(df) != len(base):
        print(f"WARNING: joined {len(df)}/{len(base)} baseline games "
              "(gate runs on the intersection; a full-coverage pred file "
              "is required for the registered verdict)")
    out = {"n": int(len(df)), "col": col,
           "logloss_base": log_loss(df.y, df[base_col]),
           "logloss_new": log_loss(df.y, df[col]),
           "logloss_mkt": log_loss(df.y, df.p_mkt)}
    out["pooled"] = paired_bootstrap_delta(df.y.values, df[base_col].values,
                                           df[col].values)
    for season, g in df.groupby("season"):
        out[season] = paired_bootstrap_delta(g.y.values, g[base_col].values,
                                             g[col].values)
        out[season]["logloss_base"] = log_loss(g.y, g[base_col])
        out[season]["logloss_new"] = log_loss(g.y, g[col])
    oct_mask = pd.to_datetime(df.game_date).dt.month == 10
    if oct_mask.any():
        g = df[oct_mask]
        out["october"] = paired_bootstrap_delta(g.y.values, g[base_col].values,
                                                g[col].values)
        out["october"]["n"] = int(len(g))
    return out


def pretty(res: dict) -> None:
    print(f"n={res['n']}  base={res['logloss_base']:.4f}  "
          f"new={res['logloss_new']:.4f}  mkt={res['logloss_mkt']:.4f}")
    for k in ("pooled", "2023-24", "2024-25", "2025-26", "october"):
        if k not in res:
            continue
        r = res[k]
        lo, hi = r["ci95"]
        extra = f" n={r['n']}" if "n" in r else ""
        print(f"  {k:9s} delta={r['delta_logloss']:+.5f} "
              f"CI({lo:+.5f},{hi:+.5f}) keep={r['keep']}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="prediction CSV (game_id + prob column)")
    ap.add_argument("--col", default="p_v3")
    ap.add_argument("--json", help="write full result JSON here")
    ap.add_argument("--smoke", action="store_true",
                    help="re-score p_us against itself (delta must be 0)")
    args = ap.parse_args()
    if args.smoke:
        base = pd.read_csv(BASE, dtype={"game_id": str})
        res = gate(base.rename(columns={"p_us": "p_v3"}), "p_v3")
        pretty(res)
        assert abs(res["pooled"]["delta_logloss"]) < 1e-12, "smoke test FAILED"
        print("smoke OK: delta == 0")
        return
    if not args.pred:
        ap.error("--pred required (or --smoke)")
    res = gate(pd.read_csv(args.pred, dtype={"game_id": str}), args.col)
    pretty(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, default=float))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
