#!/usr/bin/env python3
"""D255b — IS THE PERSISTENT PAIR EFFECT WORTH ANYTHING, AND IS IT REAL?

D255 found the ratings residual carries a pair effect that does NOT replicate
within a season (r -0.018) but DOES persist across seasons (r +0.0303, CI
[+0.0122, +0.0484], 14/18). That combination is what a stylistic propensity
would look like — teams adjust between meetings, the underlying scheme mismatch
recurs the following year — but r = 0.030 explains 0.09% of pair-mean variance
and two things have to be settled before it means anything.

**(1) IS IT SCHEDULING RATHER THAN STYLE?** Divisional opponents meet four
times a season, conference opponents three or four, non-conference twice. If
the additive model mis-fits frequent opponents differently — through the
rest/travel patterns that come with divisional scheduling, or simply because
more meetings means a better-estimated pair mean — persistent pair structure
appears with no stylistic content at all. Split the persistence by division and
conference. **If it lives only in divisional pairs, it is the schedule.**

**(2) IS IT WORTH ANY MONEY?** The only number that matters: does the PRIOR
season's pair residual predict THIS season's margin against the market? That is
strictly out-of-sample, uses nothing from the current season, and is exactly how
the feature would have to be used in production. Endpoint is the market residual
`margin_actual - close_margin`, and secondarily `- open_margin`, since the
opener is the price we actually beat.

An effect can be real, persistent and significant in R2 and still be worth zero
here, which is the outcome the register keeps producing. Both are reported.

The estimand for (2) is a coefficient in points of margin per point of prior-
season pair residual, season-clustered, with MDE80 alongside.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

import importlib.util                                             # noqa: E402
spec = importlib.util.spec_from_file_location(
    "d255", ROOT / "scripts" / "d255_matchup_residual.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

DIV = {
    "BOS": "ATL", "BKN": "ATL", "NYK": "ATL", "PHI": "ATL", "TOR": "ATL",
    "CHI": "CEN", "CLE": "CEN", "DET": "CEN", "IND": "CEN", "MIL": "CEN",
    "ATL": "SE", "CHA": "SE", "MIA": "SE", "ORL": "SE", "WAS": "SE",
    "DEN": "NW", "MIN": "NW", "OKC": "NW", "POR": "NW", "UTA": "NW",
    "GSW": "PAC", "LAC": "PAC", "LAL": "PAC", "PHX": "PAC", "SAC": "PAC",
    "DAL": "SW", "HOU": "SW", "MEM": "SW", "NOP": "SW", "SAS": "SW",
}
EAST = {"ATL", "CEN", "SE"}


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def mde80(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return 2.80 * v.std(ddof=1) / np.sqrt(len(v))


def main():
    m = D.build_team_games()
    m = m.sort_values(["season", "game_date"]).reset_index(drop=True)
    parts = []
    for s, g in m.groupby("season"):
        g = g.copy(); g["resid"] = D.fit_additive(g)
        parts.append(g)
    m = pd.concat(parts)
    pm = m.groupby(["season", "ab", "opp_ab"]).resid.agg(["mean", "size"])
    pm = pm.reset_index().rename(columns={"mean": "pres", "size": "nmeet"})
    seasons = sorted(m.season.unique())
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    pm["nseason"] = pm.season.map(nxt)
    j = pm.merge(pm, left_on=["nseason", "ab", "opp_ab"],
                 right_on=["season", "ab", "opp_ab"], suffixes=("", "_n"))
    j["div_same"] = [DIV.get(a) == DIV.get(b) for a, b in zip(j.ab, j.opp_ab)]
    j["conf_same"] = [(DIV.get(a) in EAST) == (DIV.get(b) in EAST)
                      for a, b in zip(j.ab, j.opp_ab)]

    print("=" * 76)
    print("(1) IS THE PERSISTENCE SCHEDULING, OR STYLE?")
    print("=" * 76)
    print(f"  {'group':22} {'pairs/season':>13} {'meetings':>9} {'mean r':>9} "
          f"{'CI':>22}")
    out = {}
    groups = {
        "ALL pairs": j.index == j.index,
        "same DIVISION (4x)": j.div_same,
        "same conf, diff div": j.conf_same & ~j.div_same,
        "cross-conference (2x)": ~j.conf_same,
    }
    for lab, mask in groups.items():
        sub = j[mask]
        rows = []
        for s, g in sub.groupby("season"):
            if len(g) > 30:
                rows.append(float(np.corrcoef(g.pres, g.pres_n)[0, 1]))
        if len(rows) < 3:
            continue
        mn, lo, hi, t, k = clus(rows)
        print(f"  {lab:22} {len(sub)/max(sub.season.nunique(),1):13.0f} "
              f"{sub.nmeet.mean():9.2f} {mn:+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  {'SIG' if lo>0 else ''}")
        out[lab] = dict(mean_r=float(mn), ci=[float(lo), float(hi)], k=k)
    print("\n  If persistence sits ONLY in the 4-meeting divisional pairs, it is")
    print("  the schedule (a better-estimated mean and shared travel/rest), not")
    print("  a stylistic matchup.")

    # ---------------- (2) is it worth anything ----------------------
    print("\n" + "=" * 76)
    print("(2) DOES THE PRIOR SEASON'S PAIR RESIDUAL PREDICT THIS SEASON'S")
    print("    MARGIN AGAINST THE MARKET?  (strictly out-of-sample)")
    print("=" * 76)
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = (f.game_id.astype(str).str.replace(r"\.0$", "", regex=True)
                    .str.zfill(10))
    f = f.dropna(subset=["margin_actual", "close_margin", "open_margin"]).copy()
    prev = pm.set_index(["nseason", "ab", "opp_ab"]).pres
    # home team's prior-season residual vs this away team, minus the reverse
    hk = list(zip(f.season, f.home, f.away))
    ak = list(zip(f.season, f.away, f.home))
    f["pair_h"] = [prev.get(k, np.nan) for k in hk]
    f["pair_a"] = [prev.get(k, np.nan) for k in ak]
    f["pair_signal"] = f.pair_h - f.pair_a
    d = f.dropna(subset=["pair_signal"]).copy()
    print(f"  {len(d):,} games with a prior-season pair signal "
          f"({d.season.nunique()} seasons), sd {d.pair_signal.std():.3f}")
    for price, lab in (("close_margin", "vs CLOSE"), ("open_margin", "vs OPEN")):
        d["mres"] = d.margin_actual - d[price]
        per = []
        for s, g in d.groupby("season"):
            if len(g) < 100:
                continue
            b = np.polyfit(g.pair_signal, g.mres, 1)[0]
            per.append(b)
        mn, lo, hi, t, k = clus(per)
        md = mde80(per)
        flag = ("SIG" if (lo > 0 or hi < 0) else
                f"ns (MDE80 {md:.4f} = {md/abs(mn) if mn else np.inf:.1f}x est)")
        print(f"  {lab}: slope {mn:+.5f} pts per pt of pair signal  "
              f"CI [{lo:+.5f}, {hi:+.5f}]  k={k}  {flag}")
        print(f"    implied margin swing across +-2sd of the signal: "
              f"{4*d.pair_signal.std()*mn:+.4f} pts")
        out[f"value_{lab}"] = dict(slope=float(mn), ci=[float(lo), float(hi)],
                                   mde80=float(md), k=k)

    json.dump(out, open(ROOT / "data" / "d255b_matchup_value.json", "w"),
              default=float)
    print("\nwrote data/d255b_matchup_value.json")


if __name__ == "__main__":
    main()
