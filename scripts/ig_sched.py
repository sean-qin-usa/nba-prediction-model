"""IG probe F+G: schedule layer estimator + cost of the live path dropping b2b.

F) fit_schedule_layer at 3 cutoffs/season: applied (shrunk) vs raw coefficients,
   shrink weight w = n/(n+600). Is the 600-game prior mass distorting anything
   with n~2400?
G) The live entrypoint (predict_today.py) never passes b2b flags; the backtest
   always does. Reconstruct the counterfactual "no-b2b" probabilities from the
   headline CSV (margin = 7.2*logit(p)) minus the season's fitted b2b terms and
   measure the log-loss cost live would pay.
Read-only.
"""
import sys, warnings, datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect
from nbapred.model.production import fit_schedule_layer, SCHED_PRIOR, SCHED_SHRINK, SCALE

def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def main():
    con = connect(read_only=True)
    print("== F: schedule layer applied vs raw ==")
    coefs = {}
    for season, y0 in (("2023-24", 2024), ("2024-25", 2025), ("2025-26", 2026)):
        for cut in (dt.date(y0, 1, 15),):
            n = con.execute("""SELECT count(*)/2 FROM nba_games WHERE game_id LIKE '002%'
                AND pts IS NOT NULL AND game_date < ? AND game_date >= ?""",
                [cut, cut - dt.timedelta(days=730)]).fetchone()[0]
            ap = fit_schedule_layer(con, cut)
            w = n / (n + SCHED_SHRINK)
            raw = tuple((ap[i] - (1 - w) * SCHED_PRIOR[i]) / w for i in range(5))
            coefs[season] = ap
            print(f" {season} cut {cut}: n={int(n)} w={w:.3f}")
            print(f"   applied he={ap[0]:+.2f} hb2b={ap[1]:+.2f} ab2b={ap[2]:+.2f} "
                  f"hdead={ap[3]:+.2f} adead={ap[4]:+.2f}")
            print(f"   raw     he={raw[0]:+.2f} hb2b={raw[1]:+.2f} ab2b={raw[2]:+.2f} "
                  f"hdead={raw[3]:+.2f} adead={raw[4]:+.2f}")

    print("\n== G: live-path cost of dropping b2b flags ==")
    cap = pd.read_csv("/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_carry2.csv",
                      parse_dates=["game_date"])
    # team dates per season for b2b flags (same construction as harness)
    ab2id = {}
    tdates = {}
    for season in cap.season.unique():
        rows = con.execute("""SELECT team_id, team_abbrev, game_date FROM nba_games
            WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL""", [season]).fetchall()
        for tid, ab, gd in rows:
            ab2id[(season, ab)] = tid
            tdates.setdefault((season, ab), set()).add(gd)
    def b2b(season, ab, d):
        return (d - dt.timedelta(days=1)) in tdates.get((season, ab), set())
    cap["gd"] = cap.game_date.dt.date
    cap["hb"] = [b2b(s, h, d) for s, h, d in zip(cap.season, cap.home, cap.gd)]
    cap["ab_"] = [b2b(s, a, d) for s, a, d in zip(cap.season, cap.away, cap.gd)]
    cap["margin"] = SCALE * np.log(cap.p_us / (1 - cap.p_us))
    print(f" b2b frequency: home {cap.hb.mean():.3f} away {cap.ab_.mean():.3f}")
    d_all = []
    for season, sub in cap.groupby("season"):
        apc = coefs[season]
        dm = apc[1] * sub.hb.values + apc[2] * sub.ab_.values
        p_nob2b = 1 / (1 + np.exp(-(sub.margin.values - dm) / SCALE))
        d = ll(sub.y.values, p_nob2b) - ll(sub.y.values, sub.p_us.values)
        d_all.append(d)
        aff = (dm != 0)
        print(f" {season}: games with a b2b flag {aff.mean():.1%}; "
              f"live-minus-backtest ll delta {d.mean():+.5f}/game "
              f"({d[aff].mean():+.5f} on affected games)")
    d_all = np.concatenate(d_all)
    rng = np.random.default_rng(0)
    bs = [d_all[rng.integers(0, len(d_all), len(d_all))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f" pooled cost of live dropping b2b: {d_all.mean():+.5f} CI95({lo:+.5f},{hi:+.5f})")
    con.close()

if __name__ == "__main__":
    main()
