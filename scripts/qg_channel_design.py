#!/usr/bin/env python3
"""GATE 3 DESIGN DIAGNOSTIC — per-channel props ramp.

CHANNEL-MEAN LEVEL ONLY. No CRPS / PIT endpoint number is produced here (the
D133/D145 protocol: run and disclose the design diagnostic, pre-register after).

D133 open item 3 / D145 open item 17(b): the D133 games-played ramp and the
D145 absence ramp both subtract minutes from ONE `proj_min`, and both were
fitted so that POINTS is unbiased. Rebounds scale ~linearly in minutes and
points do not, so the points-optimal subtraction over-corrects rebounds
(D133 measured October rebound PIT 0.4833 -> 0.5229).

What this script measures, on the shipped estimator (PROPS_MIN_RAMP=1,
PROPS_ABSENCE_RAMP=1 — the D145 §15 control convention):
  (a) the per-channel MEAN bias  E[rate_c * expo_c] - E[y_c]  before and after
      the ramps, on ramp-active rows, split by ramp size;
  (b) the moment-fit lam_c that zeroes the post-ramp bias of channel c:
          lam_c = (E[rate_c * m0] - E[y_c]) / E[rate_c * D]
      i.e. the fraction of the shared subtraction D that channel c should
      actually receive (lam = 1 is today's behaviour);
  (c) the walk-forward stationarity of lam_c across all five fit cutoffs;
  (d) the STRUCTURAL separability fact: which channels can be treated at all.

`player_rates_from_stats` is reproduced exactly (EWMA hl=10 over >=720s rows,
D133 gp ramp, D145 miss10 ramp) via an O(n) recursion per player; the
reproduction is asserted against the live function on a sample of rows.

READ-ONLY. Writes data/qg_channel_design.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine.props import (absence_ramp, minutes_ramp,
                                  player_rates_from_stats)

HL = 10.0
R = 0.5 ** (1.0 / HL)
SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
           "2025-26")
SCORED = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
COLS = ("rima", "mida", "thra", "fta", "reb", "ast")


def ewma_prefix(x, w_mins):
    """S[i] = sum_{j<i} r^{i-1-j} * x[j]  (exactly props.py's EWMA numerator)."""
    n = len(x)
    out = np.empty(n + 1)
    out[0] = 0.0
    acc = 0.0
    for i in range(n):
        acc = x[i] + R * acc
        out[i + 1] = acc
    return out


def main():
    con = connect(read_only=True)
    df = con.execute("""
        SELECT s.player_id, s.team_id, g.season, g.game_date, s.seconds/60.0 AS mins,
               s.pts, s.oreb + s.dreb AS reb, s.ast, s.rima, s.mida, s.thra, s.fta
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date""").fetchdf()
    df["ord"] = df["game_date"].values.astype("datetime64[D]").astype(int)
    sched = defaultdict(list)
    for s, t, d in con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%'""").fetchall():
        sched[(s, int(t))].append(np.datetime64(d).astype("datetime64[D]").astype(int))
    for k in sched:
        sched[k] = np.array(sorted(set(sched[k])))
    print(f"loaded {len(df)} 002/>=720 player-games", flush=True)

    bigdates = defaultdict(set)
    for r in df.itertuples():
        bigdates[(r.season, int(r.team_id), int(r.player_id))].add(int(r.ord))

    recs = []
    for pid, sub in df.groupby("player_id", sort=False):
        o = sub["ord"].to_numpy()
        m = sub["mins"].to_numpy(float)
        seas = sub["season"].to_numpy(object)
        team = sub["team_id"].to_numpy()
        y_reb = sub["reb"].to_numpy(float)
        y_ast = sub["ast"].to_numpy(float)
        Sw = ewma_prefix(np.ones_like(m), m)          # sum of weights
        Sm = ewma_prefix(m, m)                        # sum w*mins
        Sc = {c: ewma_prefix(sub[c].to_numpy(float), m) for c in ("reb", "ast")}
        for i in range(len(m)):
            if i < 8:
                continue
            proj_raw = Sm[i] / Sw[i]
            if proj_raw < 20:
                continue
            season = seas[i]
            gp = int((seas[:i] == season).sum())
            d_gp = minutes_ramp(gp)
            pm1 = max(proj_raw - d_gp, 0.0)
            # absence axis (props.games_missed_last10 semantics: team of the
            # player's most recent >=12-min game before the cutoff)
            lt = int(team[i - 1])
            sc = sched.get((season, lt))
            miss10 = 0
            if sc is not None:
                prior = sc[sc < o[i]][-10:]
                own = {d for d, s_, t_ in zip(o[:i], seas[:i], team[:i])
                       if s_ == season and int(t_) == lt}
                if own and prior.size:
                    first = min(own)
                    miss10 = int(sum(1 for d in prior if d >= first and d not in own))
            d_ab = absence_ramp(miss10)
            pm = max(pm1 - d_ab, 0.0)
            delta = proj_raw - pm
            r_reb = Sc["reb"][i] / Sm[i]
            r_ast = Sc["ast"][i] / Sm[i]
            recs.append((season, int(pd.Timestamp(o[i], unit="D").month), gp,
                         miss10, proj_raw, pm, delta, r_reb, r_ast,
                         y_reb[i], y_ast[i], m[i], int(pid)))
    d = pd.DataFrame(recs, columns=["season", "month", "gp", "miss10", "m0",
                                    "pm", "delta", "r_reb", "r_ast", "y_reb",
                                    "y_ast", "y_min", "player_id"])
    print(f"eval universe rows: {len(d)}", flush=True)

    # ---- reproduction assertion vs the live function ------------------------
    chk, bad = 0, []
    for r in df[df["season"] == "2024-25"].iloc[::811].itertuples():
        live = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if live is None or live["n_games"] < 8 or live["proj_min"] < 20:
            continue
        row = d[(d.player_id == int(r.player_id))
                & (d.m0 - d.delta > 0)
                & (np.abs(d.pm - live["proj_min"]) < 1e-9)]
        got = d[(d.player_id == int(r.player_id))]
        if not len(got):
            continue
        # match on the projection value (unique per player-date in practice)
        near = got.iloc[(got.pm - live["proj_min"]).abs().argsort()[:1]]
        e_pm = float(abs(near.pm.iloc[0] - live["proj_min"]))
        e_rb = float(abs(near.r_reb.iloc[0] - live["reb_per_min"]))
        e_as = float(abs(near.r_ast.iloc[0] - live["ast_per_min"]))
        e_dl = float(abs(near.delta.iloc[0] - live.get("ramp_delta", 0.0)))
        bad.append(max(e_pm, e_rb, e_as, e_dl))
        chk += 1
        if chk >= 25:
            break
    con.close()
    max_err = float(max(bad)) if bad else float("nan")
    print(f"reproduction check on {chk} live rows: max|err| = {max_err:.3e}")
    assert chk >= 10 and max_err < 1e-6, f"reproduction failed ({max_err})"

    out = {"n_rows": int(len(d)), "reproduction_max_err": max_err,
           "n_checked": chk}
    dv = d[d.season.isin(SCORED)].copy()
    dv["pred_reb"] = dv.r_reb * dv.pm
    dv["pred_ast"] = dv.r_ast * np.clip(dv.pm, 10, 44)
    dv["pred_reb0"] = dv.r_reb * dv.m0
    dv["pred_ast0"] = dv.r_ast * np.clip(dv.m0, 10, 44)
    dv["active"] = dv.delta > 0

    def bias_table(sub, label):
        t = {}
        for c in ("reb", "ast"):
            t[c] = {
                "n": int(len(sub)),
                "bias_before_ramp": round(float((sub[f"pred_{c}0"] - sub[f"y_{c}"]).mean()), 4),
                "bias_after_ramp": round(float((sub[f"pred_{c}"] - sub[f"y_{c}"]).mean()), 4),
                "mean_y": round(float(sub[f"y_{c}"].mean()), 3),
            }
        t["minutes"] = {
            "bias_before_ramp": round(float((sub.m0 - sub.y_min).mean()), 4),
            "bias_after_ramp": round(float((sub.pm - sub.y_min).mean()), 4)}
        t["mean_delta"] = round(float(sub.delta.mean()), 4)
        out[label] = t
        print(f"\n-- {label} (n={len(sub)}) --")
        print("   ", json.dumps(t))

    bias_table(dv, "ALL scored rows")
    bias_table(dv[dv.active], "RAMP-ACTIVE rows (delta>0)")
    bias_table(dv[dv.active & (dv.delta >= 2)], "RAMP-ACTIVE delta>=2")
    bias_table(dv[dv.active & dv.month.isin((10, 11))], "RAMP-ACTIVE Oct-Nov")
    bias_table(dv[~dv.active], "ramp-inactive (must be untouched)")

    # ---- (b) moment-fit lam_c ----------------------------------------------
    def fit_lam(sub):
        a = sub[sub.delta > 0]
        lam = {}
        for c, expo0, ex in (("reb", a.m0, a.r_reb),
                             ("ast", np.clip(a.m0, 10, 44), a.r_ast)):
            num = float((ex * expo0 - a[f"y_{c}"]).mean())
            den = float((ex * a.delta).mean())
            lam[c] = round(num / den, 4) if den else 0.0
        lam["n"] = int(len(a))
        return lam

    out["lam_full_corpus"] = fit_lam(d)          # ALL seasons incl. fit fuel
    out["lam_scored_only"] = fit_lam(dv)
    print("\nmoment-fit lam (fraction of the shared ramp the channel should take):")
    print("   FULL corpus 2019-26:", json.dumps(out["lam_full_corpus"]))
    print("   scored 2021-26     :", json.dumps(out["lam_scored_only"]))

    # ---- (c) walk-forward stationarity -------------------------------------
    wf = {}
    for s in SCORED:
        prior = [x for x in SEASONS if x < s]
        wf[s] = fit_lam(d[d.season.isin(prior)])
        wf[s]["fit_seasons"] = prior
    out["lam_walk_forward"] = wf
    print("\nwalk-forward lam by scored season (fit on strictly-prior only):")
    for k, v in wf.items():
        print(f"    {k}: reb {v['reb']:+.4f}  ast {v['ast']:+.4f}  n={v['n']}")

    # ---- (d) separability ---------------------------------------------------
    out["separability"] = {
        "points": "NOT separable — points is read off the same zone-attempt "
                  "draws as threes; a channel-specific minutes exposure for "
                  "either necessarily moves the other. Points/threes are held "
                  "BITWISE FIXED by both arms.",
        "threes": "NOT separable (same reason).",
        "rebounds": "separable — rng.poisson(reb_per_min * mins), own draw.",
        "assists": "separable — rng.poisson(ast_per_min * ast_expo), own draw, "
                   "but ast_expo is a SCALAR so arms A and B coincide for it "
                   "up to the [10,44] clip.",
    }

    # ---- universe sizing ----------------------------------------------------
    out["universe"] = {
        "scored_rows": int(len(dv)),
        "ramp_active": int(dv.active.sum()),
        "ramp_active_share": round(float(dv.active.mean()), 4),
        "delta_ge_2": int((dv.delta >= 2).sum()),
        "max_delta": round(float(dv.delta.max()), 4),
        "mean_delta_active": round(float(dv[dv.active].delta.mean()), 4),
        "players": int(dv.player_id.nunique()),
    }
    print("\nuniverse:", json.dumps(out["universe"]))
    dv.to_csv("data/qg_channel_rows.csv.gz", index=False, compression="gzip")
    json.dump(out, open("data/qg_channel_design.json", "w"), indent=1, default=float)
    print("\nwrote data/qg_channel_design.json")
    print("QG_CHANNEL_DESIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
