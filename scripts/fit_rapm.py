#!/usr/bin/env python3
"""Player RAPM: sign-fixed classic vs prior-informed (DARKO), judged OUT OF
SAMPLE. Validation is held-out stint-margin prediction, NOT correlation with
DARKO (circular when DARKO is the prior). Also prints top two-way + top
defenders as a sanity check.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.model.rapm import RAPM, SEC_PER_POSS, load_stints


def wrmse(model, stints):
    """Weighted RMSE of predicted vs actual offense rate on held-out stints."""
    se, wsum = 0.0, 0.0
    for s in stints:
        poss = s["seconds"] / SEC_PER_POSS
        if poss < 1:
            continue
        for off5, def5, pts in ((s["home"], s["away"], s["home_pts"]),
                                (s["away"], s["home"], s["away_pts"])):
            actual = 100.0 * pts / poss
            pred = model.mu + sum(model.off.get(p, 0.0) for p in off5) \
                - sum(model.deff.get(p, 0.0) for p in def5)
            se += poss * (actual - pred) ** 2
            wsum += poss
    return (se / wsum) ** 0.5


class DarkoOnly:
    def __init__(self, mu, off, deff):
        self.mu, self.off, self.deff = mu, off, deff


def main(ridge=2000.0):
    con = connect(read_only=True)
    stints = load_stints(con)
    names = dict(con.execute("SELECT player_id, full_name FROM nba_players").fetchall())
    drk = con.execute("SELECT nba_player_id, o_dpm, d_dpm FROM darko_dpm "
                      "WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall()
    con.close()
    off_prior = {p: o for p, o, d in drk}
    def_prior = {p: d for p, o, d in drk}
    print(f"stints: {len(stints)}  darko priors: {len(off_prior)}")

    # sign-fixed classic RAPM (shrink to zero) — sanity table
    r0 = RAPM(ridge=ridge).fit(stints)
    print(f"\nTop 12 two-way (classic, sign-fixed):")
    for pid, o, d, net in r0.table(12):
        print(f"  {names.get(pid,str(pid))[:22]:22} OFF {o:5.2f}  DEF {d:5.2f}  NET {net:5.2f}")
    defs = sorted(((r0.deff[p], p) for p in r0.deff), reverse=True)[:8]
    print("Top 8 DEFENDERS (should be rim protectors/stoppers):")
    for d, p in defs:
        print(f"  {names.get(p,str(p))[:22]:22} DEF {d:5.2f}")

    # OUT-OF-SAMPLE: split stints, fit on train, score held-out
    rng = np.random.default_rng(0)
    order = rng.permutation(len(stints))
    cut = int(0.7 * len(stints))
    train = [stints[i] for i in order[:cut]]
    test = [stints[i] for i in order[cut:]]
    league_mu = np.average([100 * (s["home_pts"] + s["away_pts"]) / (2 * s["seconds"] / SEC_PER_POSS)
                            for s in train if s["seconds"] > SEC_PER_POSS])

    m_zero = RAPM(ridge=ridge).fit(train)
    m_prior = RAPM(ridge=ridge, off_prior=off_prior, def_prior=def_prior).fit(train)
    m_darko = DarkoOnly(league_mu, off_prior, def_prior)
    m_null = DarkoOnly(league_mu, {}, {})

    print(f"\n=== OUT-OF-SAMPLE stint-rate prediction (weighted RMSE, lower=better) ===")
    print(f"  null (league mean only) : {wrmse(m_null, test):.3f}")
    print(f"  DARKO only              : {wrmse(m_darko, test):.3f}")
    print(f"  classic RAPM (->0)      : {wrmse(m_zero, test):.3f}")
    print(f"  PRIOR-INFORMED RAPM     : {wrmse(m_prior, test):.3f}")
    print("\nprior-informed should beat BOTH classic RAPM and DARKO-only if the")
    print("stint data adds signal on top of the box/DARKO prior.")


if __name__ == "__main__":
    main()
