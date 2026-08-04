#!/usr/bin/env python3
"""Fit the hierarchical Bayesian shooting model (II.1/II.2) against 2K ratings
as the prior, and report the learned per-dimension trust (beta). The FT-highest-
beta result is the handoff's built-in sanity check.

Leakage note: for a PRODUCTION/backtest fit, join 2K/DARKO as-of (nbapred/pit.py)
and use a trailing outcome window. This script uses the current corpus + current
2K scrape for the in-sample sanity check only.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.model.shooting import fit_dimension

DIMS = {  # dim: (makes_col, att_col, 2K attribute)
    "fg3": ("fg3m", "fg3a", "Three-Point Shot"),
    "rim": ("rimm", "rima", "Close Shot"),
    "mid": ("midm", "mida", "Mid-Range Shot"),
    "ft": ("ftm", "fta", "Free Throw"),
}


def main(min_minutes=30, min_att=5):
    con = connect(read_only=True)
    stats = con.execute("""
        SELECT player_id, sum(fg3a) fg3a, sum(fg3m) fg3m, sum(rima) rima, sum(rimm) rimm,
               sum(mida) mida, sum(midm) midm, sum(fta) fta, sum(ftm) ftm,
               sum(seconds)/60.0 min
        FROM player_game_stats WHERE game_id LIKE '002%' GROUP BY player_id""").fetchdf()
    xw = con.execute("SELECT nba_player_id, name_2k FROM player_xwalk "
                     "WHERE name_2k IS NOT NULL").fetchdf()
    r2k = con.execute("SELECT player_name, attributes FROM ratings_2k "
                      "WHERE scrape_date=(SELECT max(scrape_date) FROM ratings_2k)").fetchdf()
    con.close()

    name2attr = {r.player_name: json.loads(r.attributes) for r in r2k.itertuples()}
    pid2attr = {r.nba_player_id: name2attr.get(r.name_2k) for r in xw.itertuples()}
    stats = stats[stats["min"] >= min_minutes].reset_index(drop=True)

    print(f"players: {len(stats)}")
    print(f"\n{'dim':4} {'2K attr':16} {'beta (trust)':>16} {'sigma':>7}  n")
    betas = {}
    for d, (mk, at, akey) in DIMS.items():
        sub = stats[stats[at] >= min_att].copy()
        sub["r"] = sub["player_id"].map(lambda p: (pid2attr.get(p) or {}).get(akey))
        sub = sub.dropna(subset=["r"])
        z = ((sub["r"] - sub["r"].mean()) / sub["r"].std()).to_numpy()
        res = fit_dimension(z, sub[at].to_numpy(), sub[mk].to_numpy(),
                            num_warmup=400, num_samples=800)
        betas[d] = res["beta_mean"]
        print(f"{d:4} {akey:16} {res['beta_mean']:8.3f}+-{res['beta_sd']:.3f} "
              f"{res['sigma_mean']:7.3f}  {len(sub)}")
    ok = betas["ft"] == max(betas.values())
    print(f"\nSanity check (II.1: FT highest beta): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
