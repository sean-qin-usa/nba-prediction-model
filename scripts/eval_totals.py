"""Totals (over/under): a full-game market we haven't tested. Walk-forward
predicted total (team scoring for/against, recent 25 games) vs the market's
total line. Reports MAE, blend weights (does our total add info?), and O/U
disagreement hit rates.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect


def main(season="2025-26", season_end=2026, warm=15, window=25):
    con = connect(read_only=True)
    rows = con.execute("""SELECT o.game_date, o.total, o.score_home sh, o.score_away sa,
        gh.team_id hid, ga.team_id aid FROM odds_market o
        JOIN (SELECT DISTINCT game_date, team_abbrev, team_id FROM nba_games WHERE season=?) gh
          ON gh.game_date=o.game_date AND gh.team_abbrev=o.home
        JOIN (SELECT DISTINCT game_date, team_abbrev, team_id FROM nba_games WHERE season=?) ga
          ON ga.game_date=o.game_date AND ga.team_abbrev=o.away
        WHERE o.season_end=? AND o.total IS NOT NULL ORDER BY o.game_date""",
        [season, season, season_end]).fetchdf()
    con.close()

    hist = {}
    ours, mkt, actual = [], [], []
    for r in rows.itertuples():
        fh, fa = hist.get(r.hid, []), hist.get(r.aid, [])
        if len(fh) >= warm and len(fa) >= warm:
            hf = np.mean([x[0] for x in fh[-window:]]); hg = np.mean([x[1] for x in fh[-window:]])
            af = np.mean([x[0] for x in fa[-window:]]); ag = np.mean([x[1] for x in fa[-window:]])
            ours.append((hf + ag) / 2 + (af + hg) / 2)
            mkt.append(r.total); actual.append(r.sh + r.sa)
        hist.setdefault(r.hid, []).append((r.sh, r.sa))
        hist.setdefault(r.aid, []).append((r.sa, r.sh))
    ours, mkt, actual = map(np.array, (ours, mkt, actual))
    print(f"{season}: games {len(ours)}")
    print(f"TOTALS MAE:  ours {np.abs(ours-actual).mean():.2f}   market {np.abs(mkt-actual).mean():.2f}")
    print(f"bias: ours {np.mean(ours-actual):+.2f}  market {np.mean(mkt-actual):+.2f}")
    X = np.c_[ours, mkt, np.ones(len(ours))]
    beta = np.linalg.lstsq(X, actual, rcond=None)[0]
    print(f"blend weights: ours {beta[0]:+.3f}  market {beta[1]:+.3f}  (ours~0 = no added info)")
    for thr in (3, 5, 8):
        m = np.abs(ours - mkt) > thr
        push = actual == mkt
        sel = m & ~push
        if sel.sum() > 10:
            win = np.where(ours > mkt, actual > mkt, actual < mkt)[sel]
            print(f"O/U disagree>{thr}: n={sel.sum()} our-side hit {np.mean(win):.3f} (breakeven .524)")


if __name__ == "__main__":
    main()
    print()
    main(season="2024-25", season_end=2025)
