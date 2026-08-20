#!/usr/bin/env python3
"""D256 — WAS D255's PAIR PERSISTENCE ACTUALLY A *TEAM* EFFECT?

D255 left one thing open: across-season persistence of the ratings residual was
strongest in the pairs that meet LEAST (cross-conference, 1.94 meetings,
r +0.0461) and absent in the pairs that meet most (divisional, 3.88 meetings,
r +0.0252 ns). I flagged travel/venue as the candidate. There is a duller and
much more likely explanation that I failed to control for.

**THE DEFECT.** D255 correlated raw PAIR-MEAN residuals across seasons. A pair
mean for (i, j) contains, additively:

    grand + row_i + col_j + interaction_ij

`row_i` is anything persistently unmodelled about team i's OFFENCE and `col_j`
the same for team j's DEFENCE. Both recur next season for the same reasons
(scheme, personnel, home court). And `d255_matchup_residual.fit_additive` fits
only `mu + off_i - def_j + home` — it omits the per-team home deviation that
production actually carries, so team-level home effects sit in the residual by
construction.

So a purely TEAM-level persistent misfit produces apparent PAIR persistence:
every pair (i, ·) inherits row_i, and it is there again next year. Only
`interaction_ij` is a matchup effect in the sense Sean asked about.

**THE FIX.** Double-centre the 30x30 pair matrix within each season — subtract
row means and column means, add back the grand mean — then correlate across
seasons. That is the standard two-way decomposition and it removes row_i and
col_j exactly, leaving the interaction.

Reported for each layer so the arithmetic is visible:
  L0  raw pair means                     (what D255 correlated)
  L1  row-centred only                   (offence-side team effect removed)
  L2  double-centred = INTERACTION ONLY  (the honest matchup estimand)

and the row/column persistence themselves, since if THOSE are the whole story
they are a finding in their own right: it would say the ratings leave a
persistent, per-team unmodelled component — which is a statement about the
rank-1 model that has nothing to do with matchups.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import importlib.util                                             # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

spec = importlib.util.spec_from_file_location(
    "d255", ROOT / "scripts" / "d255_matchup_residual.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
spec2 = importlib.util.spec_from_file_location(
    "d255b", ROOT / "scripts" / "d255b_matchup_value.py")
B = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(B)


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def matrices(m):
    """Per season: the 30x30 pair-mean matrix and its centred versions."""
    out = {}
    for s, g in m.groupby("season"):
        teams = sorted(set(g.ab) | set(g.opp_ab))
        idx = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        S = np.full((T, T), np.nan)
        pm = g.groupby(["ab", "opp_ab"]).resid.mean()
        for (a, b), v in pm.items():
            S[idx[a], idx[b]] = v
        M = np.ma.masked_invalid(S)
        row = M.mean(axis=1).filled(np.nan)[:, None]
        col = M.mean(axis=0).filled(np.nan)[None, :]
        grand = float(M.mean())
        out[s] = dict(teams=teams, raw=S, row1=S - row,
                      dbl=S - row - col + grand,
                      rowvec=row.ravel(), colvec=col.ravel())
    return out


def persist(mats, key, mask_fn=None):
    """Correlate season s with season s+1 on the given layer."""
    seasons = sorted(mats)
    rs = []
    for a, b in zip(seasons[:-1], seasons[1:]):
        ta, tb = mats[a]["teams"], mats[b]["teams"]
        common = [t for t in ta if t in tb]
        ia = [ta.index(t) for t in common]
        ib = [tb.index(t) for t in common]
        A = mats[a][key][np.ix_(ia, ia)]
        Bm = mats[b][key][np.ix_(ib, ib)]
        iu = ~np.eye(len(common), dtype=bool)
        if mask_fn is not None:
            iu = iu & mask_fn(common)
        x, y = A[iu], Bm[iu]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 30:
            rs.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
    return np.array(rs)


def main():
    m = D.build_team_games().sort_values(["season", "game_date"])
    parts = []
    for s, g in m.groupby("season"):
        g = g.copy(); g["resid"] = D.fit_additive(g); parts.append(g)
    m = pd.concat(parts)
    mats = matrices(m)
    print(f"{len(mats)} seasons, {len(mats[sorted(mats)[0]]['teams'])} teams\n")

    print("=" * 78)
    print("ACROSS-SEASON PERSISTENCE BY LAYER")
    print("=" * 78)
    out = {}
    for key, lab in (("raw", "L0 raw pair means (what D255 used)"),
                     ("row1", "L1 row-centred (offence team effect out)"),
                     ("dbl", "L2 DOUBLE-CENTRED = interaction only")):
        v = persist(mats, key)
        mn, lo, hi, k = clus(v)
        flag = "SIG" if lo > 0 else "ns"
        print(f"  {lab:44} r {mn:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{int((v>0).sum())}/{k}  {flag}")
        out[key] = dict(r=float(mn), ci=[float(lo), float(hi)], k=k)

    print("\n  The row/column effects themselves, correlated across seasons:")
    seasons = sorted(mats)
    for nm, vec in (("row (team offence residual)", "rowvec"),
                    ("col (team defence residual)", "colvec")):
        rs = []
        for a, b in zip(seasons[:-1], seasons[1:]):
            ta, tb = mats[a]["teams"], mats[b]["teams"]
            common = [t for t in ta if t in tb]
            x = mats[a][vec][[ta.index(t) for t in common]]
            y = mats[b][vec][[tb.index(t) for t in common]]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 20:
                rs.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
        mn, lo, hi, k = clus(rs)
        print(f"    {nm:32} r {mn:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{int((np.array(rs)>0).sum())}/{k}  "
              f"{'SIG' if lo > 0 else 'ns'}")
        out[vec] = dict(r=float(mn), ci=[float(lo), float(hi)], k=k)

    print("\n" + "=" * 78)
    print("THE OPEN QUESTION: cross-conference vs divisional, BY LAYER")
    print("=" * 78)
    EAST, DIV = B.EAST, B.DIV
    def mk(kind):
        def f(common):
            n = len(common)
            M = np.zeros((n, n), dtype=bool)
            for i, a in enumerate(common):
                for j, b in enumerate(common):
                    same_conf = (DIV.get(a) in EAST) == (DIV.get(b) in EAST)
                    same_div = DIV.get(a) == DIV.get(b)
                    M[i, j] = (same_div if kind == "div"
                               else (not same_conf) if kind == "cross"
                               else (same_conf and not same_div))
            return M
        return f
    print(f"  {'layer':28} {'divisional':>18} {'cross-conference':>20}")
    for key, lab in (("raw", "L0 raw"), ("dbl", "L2 interaction")):
        cells = []
        for kind in ("div", "cross"):
            v = persist(mats, key, mk(kind))
            mn, lo, hi, k = clus(v)
            cells.append(f"{mn:+.4f} {'SIG' if lo>0 else 'ns ':>4}")
            out[f"{key}_{kind}"] = dict(r=float(mn), ci=[float(lo), float(hi)])
        print(f"  {lab:28} {cells[0]:>18} {cells[1]:>20}")

    print("\n  READ: if L2 collapses to ~0 while L0 was +0.030, D255's")
    print("  'pair persistence' was TEAM persistence — every pair (i,.) carries")
    print("  team i's own unmodelled residual, and it recurs next season for")
    print("  reasons that have nothing to do with the opponent.")
    json.dump(out, open(ROOT / "data" / "d256_pair_interaction.json", "w"),
              default=float)
    print("\nwrote data/d256_pair_interaction.json")


if __name__ == "__main__":
    main()
