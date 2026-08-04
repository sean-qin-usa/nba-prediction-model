"""IG probe A: carry hard-stop handover discontinuity + ff.ready-from-day-1.

Read-only. For each season:
  1. Is ff.ready True at the season's FIRST game date (carry active)? If yes the
     entire fallback path (w_comp blend, ratings, D16 prior, D20 home devs,
     linear-20 fade) is dead code in the shipped capstone.
  2. Find the handover: last cutoff c1 with <200 current factor rows (carry
     active) and first cutoff c2 with >=200 (carry dropped). Measure margin
     discontinuity on the SAME matchup set (next 14 days of real games) between
     fits at c1 and c2, vs a placebo pair (same calendar gap, both post-handover)
     to net out normal data drift.
  3. Report the carry effective-weight share just before handover.
"""
import sys, warnings, datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.model.production import fit_production, SCALE, sigmoid
from nbapred.model.four_factors import factor_game_rows

def game_dates(con, season):
    return [r[0] for r in con.execute("""SELECT DISTINCT game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL ORDER BY 1""", [season]).fetchall()]

def matchups_between(con, season, d0, d1):
    rows = con.execute("""SELECT game_id, game_date, matchup, team_abbrev, team_id
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        AND game_date >= ? AND game_date < ? ORDER BY game_date""", [season, d0, d1]).fetchall()
    by = {}
    for gid, gd, m, ab, tid in rows:
        by.setdefault(gid, []).append((m, ab, tid, gd))
    out = []
    for gid, recs in by.items():
        if len(recs) != 2:
            continue
        m = recs[0][0]
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x[1] == host), None)
        a = next((x for x in recs if x[1] != host), None)
        if h and a:
            out.append((int(h[2]), int(a[2]), h[3]))
    return out

def margins(model, mus):
    return np.array([model.margin(h, a, None, None, gd) for h, a, gd in mus])

def main():
    con = connect(read_only=True)
    for season in ("2023-24", "2024-25", "2025-26"):
        dates = game_dates(con, season)
        d1 = dates[0]
        # 1) day-1 readiness: rebuild ff exactly as fit_production does
        from nbapred.model.production import continuity_map as _cm, CARRY_W0 as _w0, \
            CARRY_CONT_DEFAULT as _cd, _prev_season as _ps
        from nbapred.model.four_factors import FourFactors
        cutoff1 = d1  # BEFORE the first game: zero current rows
        _cont = _cm(con, season, before=cutoff1)
        _prev = factor_game_rows(con, _ps(season), before=None)
        _cw = [_w0 * _cont.get(x["tid"], _cd) for x in _prev] if _cont else None
        ff_day1 = FourFactors().fit(con, season, before=cutoff1,
                                    carry_rows=_prev if _cont else None, carry_weights=_cw)
        nrows_d1 = len(factor_game_rows(con, season, before=cutoff1))
        # find handover cutoffs
        c1 = c2 = None
        for d in dates:
            n = len(factor_game_rows(con, season, before=d))
            if n < 200:
                c1 = d
            elif c2 is None:
                c2 = d
                break
        # carry weight share just before handover
        from nbapred.model.production import continuity_map, CARRY_W0, CARRY_CONT_DEFAULT, _prev_season
        cont = continuity_map(con, season, before=c1)
        prev_rows = factor_game_rows(con, _prev_season(season), before=None)
        cur_n = len(factor_game_rows(con, season, before=c1))
        wsum_carry = sum(CARRY_W0 * cont.get(x["tid"], CARRY_CONT_DEFAULT) for x in prev_rows)
        share = wsum_carry / (wsum_carry + cur_n)
        # 2) handover discontinuity on next-14d games
        mus = matchups_between(con, season, c2, c2 + dt.timedelta(days=14))
        A = fit_production(con, season, before=c1)
        B = fit_production(con, season, before=c2)
        dmar = margins(B, mus) - margins(A, mus)
        pA = sigmoid(margins(A, mus) / SCALE); pB = sigmoid(margins(B, mus) / SCALE)
        # placebo: same gap, both sides post-handover (c2+7 vs c2+7+gap)
        gap = (c2 - c1).days
        p1 = c2 + dt.timedelta(days=7); p2 = p1 + dt.timedelta(days=gap)
        P1 = fit_production(con, season, before=p1)
        P2 = fit_production(con, season, before=p2)
        mus2 = matchups_between(con, season, p2, p2 + dt.timedelta(days=14))
        dmar_pl = margins(P2, mus2) - margins(P1, mus2)
        dp_pl = sigmoid(margins(P2, mus2) / SCALE) - sigmoid(margins(P1, mus2) / SCALE)
        print(f"\n== {season} ==")
        print(f" first game {d1}; current factor rows at cutoff = {nrows_d1}; "
              f"ff.ready(day1, opening tip) = {ff_day1.ready}")
        print(f" handover c1={c1} (n={cur_n}, carry share {share:.1%}) -> c2={c2} gap {gap}d")
        print(f" HANDOVER  n={len(mus)} mean|dM|={np.abs(dmar).mean():.3f} max|dM|={np.abs(dmar).max():.3f}"
              f" mean|dP|={np.abs(pB-pA).mean():.4f} max|dP|={np.abs(pB-pA).max():.4f}")
        print(f" PLACEBO   n={len(mus2)} mean|dM|={np.abs(dmar_pl).mean():.3f} max|dM|={np.abs(dmar_pl).max():.3f}"
              f" mean|dP|={np.abs(dp_pl).mean():.4f} max|dP|={np.abs(dp_pl).max():.4f}")
    con.close()

if __name__ == "__main__":
    main()
