"""TRAVEL/CIRCADIAN/DENSITY — margin-scale coefficients in POINTS.

Pre-registered readout #2 of data/travel_prereg.md
(sha256 d3d334b92665af13dae7914133af626c9d8c1993982a67023a40810c2fbb5a3e).
This is the WELL-POWERED half of the design: the endpoint gate cannot resolve a
0.25-pt effect but the margin regression at n=8,279 can bound it.

Design matrix, exactly the pre-registered control discipline:
  [1, hb2b, ab2b, hdead, adead] (the SHIPPED D46 regressors)
  + arm regressors
  + qd = wpct_home - wpct_away  (D46 wpct-diff CONTROL, fit-only never applied)
Response: home margin. Read-only DB.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nbapred.db import connect  # noqa: E402
from nbapred.model.production import DEAD_GP, DEAD_WPCT  # noqa: E402
from nbapred.model.travel import build_state  # noqa: E402


def frame(con, seasons=None):
    """Home-margin frame + every regressor, for the requested seasons."""
    st = build_state(con)
    sq = ""
    params = []
    if seasons:
        sq = " AND season IN (" + ",".join("?" * len(seasons)) + ")"
        params = list(seasons)
    g = con.execute(f"""
        WITH t AS (SELECT season, game_id, game_date, team_id, is_home, pts
                   FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL {sq})
        SELECT h.season, h.game_date, h.team_id ht, a.team_id at_, h.pts - a.pts m
        FROM t h JOIN t a USING (game_id) WHERE h.is_home AND NOT a.is_home
        ORDER BY h.game_date""", params).fetchall()
    # PIT standings as-of each game (for dead flags + the wpct control)
    wl = con.execute(f"""SELECT season, team_id, game_date, wl FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL {sq} ORDER BY game_date""",
        params).fetchall()
    gp, wins, stand = {}, {}, {}
    for season, t, d0, w0 in wl:
        d0 = d0.date() if hasattr(d0, "date") else d0
        k = (season, t)
        stand[(t, d0)] = (gp.get(k, 0), wins.get(k, 0) / max(gp.get(k, 1), 1))
        gp[k] = gp.get(k, 0) + 1
        wins[k] = wins.get(k, 0) + (w0 == "W")

    rows, y, seas = [], [], []
    for season, d, ht, at_, m in g:
        d = d.date() if hasattr(d, "date") else d
        sh, sa = st.get((ht, d)), st.get((at_, d))
        if sh is None or sa is None:
            continue
        wh, wa = stand.get((ht, d)), stand.get((at_, d))
        dead_h = wh is not None and wh[0] >= DEAD_GP and wh[1] < DEAD_WPCT
        dead_a = wa is not None and wa[0] >= DEAD_GP and wa[1] < DEAD_WPCT
        rows.append(dict(
            hb2b=float(sh["b2b"]), ab2b=float(sa["b2b"]),
            hdead=float(dead_h), adead=float(dead_a),
            # ARM A/B gated (antisymmetric) + separate-entry diagnostic
            dtrav_kkm=(sh["travel_km"] - sa["travel_km"]) / 1000.0,
            trav_h=sh["travel_km"] / 1000.0, trav_a=sa["travel_km"] / 1000.0,
            dtz_east=sh["tz_east"] - sa["tz_east"],
            tze_h=sh["tz_east"], tze_a=sa["tz_east"],
            # ARM C (naturally one-sided)
            hret_h=sh["home_return"],
            rlen_extra_a=max(sa["road_len"] - 1.0, 0.0),
            aret_a=sa["home_return"],
            rlen_extra_h=max(sh["road_len"] - 1.0, 0.0),
            # ARM D
            d3in4=sh["is_3in4"] - sa["is_3in4"], d5in7=sh["is_5in7"] - sa["is_5in7"],
            i34_h=sh["is_3in4"], i34_a=sa["is_3in4"],
            i57_h=sh["is_5in7"], i57_a=sa["is_5in7"],
            # control
            qd=(wh[1] if wh else 0.5) - (wa[1] if wa else 0.5),
            # exploratory, NOT gated
            delev_km=(sh["elev_gain_m"] - sa["elev_gain_m"]) / 1000.0,
        ))
        y.append(float(m))
        seas.append(season)
    return rows, np.array(y), np.array(seas)


BASE = ["hb2b", "ab2b", "hdead", "adead"]
ARMS = {
    "A": ["dtrav_kkm"],
    "B": ["dtz_east"],
    "C": ["hret_h", "rlen_extra_a"],
    "D": ["d3in4", "d5in7"],
}
ARMS_SEP = {   # free-sign asymmetry diagnostic (D46 home/away treatment)
    "A": ["trav_h", "trav_a"],
    "B": ["tze_h", "tze_a"],
    "C": ["hret_h", "rlen_extra_a", "aret_a", "rlen_extra_h"],
    "D": ["i34_h", "i34_a", "i57_h", "i57_a"],
}
# pre-registered sign predictions on HOME MARGIN (+1 expect positive, -1 negative)
PRED = {"dtrav_kkm": -1, "dtz_east": +1, "hret_h": -1, "rlen_extra_a": +1,
        "d3in4": -1, "d5in7": -1}
PRED_SEP = {"trav_h": -1, "trav_a": +1, "tze_h": +1, "tze_a": -1,
            "hret_h": -1, "rlen_extra_a": +1, "aret_a": +1, "rlen_extra_h": -1,
            "i34_h": -1, "i34_a": +1, "i57_h": -1, "i57_a": +1}


def ols(rows, y, cols):
    X = np.c_[np.ones(len(y)), np.array([[r[c] for c in cols] for r in rows])]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(resid @ resid) / dof
    XtXi = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtXi) * s2)
    return beta, se, float(np.sqrt(s2)), len(y)


def report(tag, rows, y, cols, pred):
    beta, se, sig, n = ols(rows, y, cols + ["qd"])
    out = {"tag": tag, "n": n, "resid_sd": round(sig, 3), "terms": {}}
    print(f"\n### {tag}  n={n}  resid_sd={sig:.3f}")
    print(f"  {'term':16s} {'beta(pts)':>10s} {'se':>7s} {'95% CI':>20s} "
          f"{'t':>6s}  pred  sign?")
    for i, c in enumerate(["const"] + cols + ["qd"]):
        b, s = beta[i], se[i]
        lo, hi = b - 1.96 * s, b + 1.96 * s
        p = pred.get(c)
        ok = "" if p is None else ("MATCH" if np.sign(b) == p else "**MISS**")
        pr = "" if p is None else ("neg" if p < 0 else "pos")
        star = "SIG" if lo * hi > 0 else "ns "
        print(f"  {c:16s} {b:+10.4f} {s:7.4f} ({lo:+8.4f},{hi:+8.4f}) "
              f"{b/s:+6.2f} {star} {pr:>4s}  {ok}")
        out["terms"][c] = dict(beta=round(float(b), 5), se=round(float(s), 5),
                               lo=round(float(lo), 5), hi=round(float(hi), 5),
                               t=round(float(b / s), 3), sig=bool(lo * hi > 0),
                               pred=p, sign_match=(None if p is None
                                                   else bool(np.sign(b) == p)))
    return out


if __name__ == "__main__":
    con = connect(read_only=True)
    res = {}
    for label, seasons in (
            ("FULL 2019-20..2025-26", None),
            ("SCORED 2021-22..2025-26", ["2021-22", "2022-23", "2023-24",
                                         "2024-25", "2025-26"]),
            ("DEV 2023-24..2025-26", ["2023-24", "2024-25", "2025-26"]),
            ("HOLDOUT 2021-22..2022-23", ["2021-22", "2022-23"])):
        rows, y, seas = frame(con, seasons)
        print("\n" + "=" * 78)
        print(f"UNIVERSE: {label}")
        print("=" * 78)
        # shipped layer alone (the control)
        res[f"{label}|SHIPPED"] = report("SHIPPED D46 layer (control)", rows, y,
                                         BASE, PRED)
        for a in "ABCD":
            res[f"{label}|arm{a}"] = report(
                f"ARM {a} gated (+ shipped layer + wpct control)",
                rows, y, BASE + ARMS[a], PRED)
        res[f"{label}|portfolio"] = report(
            "PORTFOLIO ABCD gated", rows, y,
            BASE + ARMS["A"] + ARMS["B"] + ARMS["C"] + ARMS["D"], PRED)
        for a in "ABCD":
            res[f"{label}|arm{a}_sep"] = report(
                f"ARM {a} SEPARATE-ENTRY asymmetry diagnostic",
                rows, y, BASE + ARMS_SEP[a], PRED_SEP)
    con.close()
    json.dump(res, open("data/tv_margin_fit.json", "w"), indent=1)
    print("\nwrote data/tv_margin_fit.json")
