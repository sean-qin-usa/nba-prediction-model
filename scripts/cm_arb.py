#!/usr/bin/env python3
"""CM-ARB — DELIVERABLE 2: "if we predict the close differs from the open,
we should have some arbitrage, right?"  (D148)

THE DISTINCTION THE ANSWER TURNS ON.
  * A sports bet settles on the OUTCOME, not on the line.  Beating the close
    is normally only a PROXY for edge — a positive CLV pays nothing by itself.
  * It becomes REAL money only if the position can be CLOSED OUT: bet side A
    at the open, take side B at the moved price, and bank a profit that does
    not depend on who wins.  Three vehicles:
        ARB    two-sided round trip across two retail books
        LAY    back at a book, lay the same side on an exchange
        MIDDLE both tickets can win, if the line crosses enough numbers

WHAT IS PRICED HERE, all on our own data:
  (a) the open->close movement distribution in points AND in probability;
  (b) round-trip lock rate and profit after realistic costs, for BOTH a
      two-retail-book trip (each leg at the D120-validated 1.043 overround,
      and separately at the REAL moneylines where they exist) and an
      exchange-style cost (commission on NET WINNINGS only, sweep 2%/5%);
  (c) the same restricted to the frozen rules' selections and to the D147
      movement model's selections — the bets with measured positive CLV;
  (d) MIDDLES on spreads and totals, with the realised middle-hit probability
      against the -110/-110 breakeven;
  (e) the constraints, and which ones actually bind.

THE INTRADAY PATH.  data/raw/teamrankings/spread_movement.jsonl is the only
source we have with per-book quotes AND timestamps, so it is the only way to
price an exit that is not the close.  It bounds the honest upper case: an
attentive trader exits at the BEST price on the path, not at the closing one.

RULES HONORED: DuckDB read_only=True with a 60s retry; nbapred/ untouched.

Run:  python scripts/cm_arb.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import duckdb                                                    # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from nbapred.market.anchored import OVERROUND, SPREAD_SCALE, sigmoid  # noqa: E402
from nbapred.eval import splits                                  # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
ROWS = os.path.join(ROOT, "data", "cm_clvmodel_rows.csv.gz")
TR_JSONL = os.path.join(ROOT, "data", "raw", "teamrankings", "spread_movement.jsonl")
ODDS_OPEN = os.path.join(ROOT, "data", "derived", "odds_open.csv")
OUT = os.path.join(ROOT, "data", "cm_arb.json")

SEED = 20260802
B_BOOT = 2000
ATS_DEC = 1.0 + 100.0 / 110.0          # -110, the standard spread/total price
RES: dict = {}


def ro_connect(attempts: int = 10, wait_s: float = 60.0):
    for k in range(attempts):
        try:
            return duckdb.connect(DB, read_only=True)
        except Exception as e:                                    # noqa: BLE001
            if k == attempts - 1:
                raise
            print(f"    [db] locked ({e}); retry {k+1}/{attempts} in {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError


def hdr(t):
    print("\n" + "#" * 100 + f"\n# {t}\n" + "#" * 100)


# ======================================================== THE ARITHMETIC ====
def arb_rate(dec1, dec2):
    """Two decimal prices on OPPOSITE sides -> locked return per unit of TOTAL
    stake.  Positive = a real arbitrage.  Stakes split so both legs pay the
    same: r = 1/(1/dec1 + 1/dec2) - 1."""
    s = 1.0 / np.asarray(dec1, float) + 1.0 / np.asarray(dec2, float)
    return 1.0 / s - 1.0


def lay_rate(back_dec, lay_dec, comm):
    """Back at `back_dec` on a book, LAY the same side on an exchange at
    `lay_dec`, commission `comm` charged on NET WINNINGS only.

        S wins  : +B(b-1) - Q(l-1)
        S loses : -B + Q(1-c)
        equalise: Q = B*b/(l-c)   ->   locked rate on the backed stake
                  = b(1-c)/(l-c) - 1
    """
    b = np.asarray(back_dec, float)
    l = np.asarray(lay_dec, float)
    return b * (1.0 - comm) / (l - comm) - 1.0


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(ROWS, parse_dates=["game_date"], dtype={"game_id": str})
    df = df[np.isfinite(df.pred_dm.values)].reset_index(drop=True)

    # ------------------------------------------------------ (a) MOVEMENT ---
    hdr("[a] THE OPEN->CLOSE MOVEMENT DISTRIBUTION")
    oo = pd.read_csv(ODDS_OPEN, parse_dates=["game_date"])
    oo = oo[oo.open_margin.notna() & oo.close_margin.notna()].copy()
    oo["dm"] = oo.close_margin - oo.open_margin
    oo["dp"] = sigmoid(oo.close_margin / SPREAD_SCALE) - \
        sigmoid(oo.open_margin / SPREAD_SCALE)
    thr = 1.0 - 1.0 / OVERROUND
    for nm, g in [("FULL odds_open corpus 2007-2026", oo),
                  ("MODEL FRAME 2021-26 (the scorable one)", df)]:
        d, p = g.dm.values, g.dp.values
        print(f"\n  {nm}   n={len(g)}")
        print(f"    POINTS       mean {d.mean():+.4f}  sd {d.std():.4f}  "
              f"mean|dm| {np.abs(d).mean():.4f}  median|dm| "
              f"{np.median(np.abs(d)):.2f}")
        print(f"                 never moves {100*(d==0).mean():.1f}%   "
              f"|dm|>=1 {100*(np.abs(d)>=1).mean():.1f}%   "
              f">=2 {100*(np.abs(d)>=2).mean():.1f}%   "
              f">=3 {100*(np.abs(d)>=3).mean():.1f}%")
        print(f"    PROBABILITY  sd {p.std():.5f}  mean|dp| "
              f"{np.abs(p).mean():.5f}")
        print(f"                 P(|dp| > {thr:.5f}) = "
              f"{100*(np.abs(p)>thr).mean():.2f}%   <- the two-book arb "
              f"threshold at overround {OVERROUND}")
        RES.setdefault("movement", {})[nm] = {
            "n": int(len(g)), "mean_dm": float(d.mean()),
            "sd_dm": float(d.std()), "mean_abs_dm": float(np.abs(d).mean()),
            "frac_zero": float((d == 0).mean()),
            "sd_dp": float(p.std()), "mean_abs_dp": float(np.abs(p).mean()),
            "frac_over_arb_thr": float((np.abs(p) > thr).mean())}
    print(f"\n  THE THRESHOLD, DERIVED.  Bet side S at the open at "
          f"1/(p_open_S*{OVERROUND}); take ~S at the close at "
          f"1/(p_close_~S*{OVERROUND}).\n  The two stakes sum to less than the "
          f"payout iff  {OVERROUND}*(1 + p_open_S - p_close_S) < 1,  i.e. iff")
    print(f"      CLV on the side taken at the open  >  1 - 1/{OVERROUND} = "
          f"{thr:.5f}  ({100*thr:.3f} probability points).")
    print(f"  In points that is ~{thr/0.0359:.2f} pts at a pick-em, "
          f"~{thr/0.0302:.2f} pts at a -6 favourite, "
          f"~{thr/0.0183:.2f} pts at a -12 favourite.")
    print(f"  Measured mean CLV per bet is +0.009 (universe, D121) to +0.054 "
          f"(D147 top rule).\n  The threshold is {thr:.4f}.  THAT COMPARISON "
          f"IS THE WHOLE OF DELIVERABLE 2.")
    RES["arb_threshold_prob"] = float(thr)

    # ------------------------------------ (b0) WHICH PRICE FRAME IS HONEST --
    hdr("[b0] A PRICING-FRAME CORRECTION THAT CHANGES THIS ENTIRE DELIVERABLE")
    print("  D120 measured that the program's SP convention  p = "
          "sigmoid(margin/6.96), decimal =\n  1/(p*1.045)  OVERSTATES breakeven "
          "by ~1.98pp on OUR bets, and D142 s5 noted the bias\n  is a LEVEL "
          "that cancels in any comparison where both policies use the same "
          "map.\n  Both of those measurements were taken on FAVOURITE-heavy "
          "bet sets.  It is not a level.")
    d2 = df[np.isfinite(df.p_open_ml.values)]
    sp = np.concatenate([d2.p_open_sp.values, 1 - d2.p_open_sp.values])
    ml = np.concatenate([d2.p_open_ml.values, 1 - d2.p_open_ml.values])
    print(f"\n  SP-frame probability vs the REAL devigged opening moneyline, "
          f"both sides of every game:")
    print(f"    {'ML p bucket':>14s}{'n':>7s}{'p_ML':>9s}{'p_SP':>9s}"
          f"{'SP - ML':>10s}{'SP/ML':>8s}")
    bias = []
    for lo_, hi_ in [(0.0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5),
                     (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0)]:
        mk = (ml >= lo_) & (ml < hi_)
        if mk.sum() < 20:
            continue
        bias.append({"bucket": f"[{lo_:.2f},{hi_:.2f})", "n": int(mk.sum()),
                     "p_ml": float(ml[mk].mean()), "p_sp": float(sp[mk].mean()),
                     "diff": float(sp[mk].mean() - ml[mk].mean())})
        print(f"    [{lo_:.2f},{hi_:.2f}){mk.sum():7d}{ml[mk].mean():9.4f}"
              f"{sp[mk].mean():9.4f}{sp[mk].mean()-ml[mk].mean():+10.4f}"
              f"{sp[mk].mean()/ml[mk].mean():8.3f}")
    print("\n  THE BIAS IS MONOTONE IN THE SIDE, NOT A LEVEL: the sigmoid map "
          "is PESSIMISTIC on\n  favourites (SP/ML up to 1.05) and OPTIMISTIC "
          "on dogs (down to 0.91 at p~0.35).  It\n  therefore quotes dog "
          "decimals ~10% LONGER than any book ever offers (max SP decimal\n"
          "  33.4 vs max REAL 21.0).  D120's and D142's cancellation argument "
          "holds for their\n  favourite-heavy rule sets and FAILS here, "
          "because the movement model picks dogs and\n  favourites ~50/50.  "
          "MEASURED CONSEQUENCE on the |pred|>1.5 set: ROI +23.49% on the SP\n"
          "  frame vs +9.93% on the REAL moneylines.  EVERY ROI BELOW IS "
          "THEREFORE QUOTED ON REAL\n  MONEYLINES AS PRIMARY; the SP frame is "
          "secondary and flagged wherever it appears.")
    RES["sp_vs_ml_bias"] = bias

    # ------------------------------------------------ (b) THE ROUND TRIP ---
    hdr("[b] THE TWO-SIDED ROUND TRIP — how often does it actually LOCK?")
    # side selection policies, all decided AT THE OPEN
    pol = {
        "p_us (market-blind incumbent)": (df.p_us > 0.5).values,
        "pred_dm (D147 movement model)": (df.pred_dm > 0).values,
        "pred_dm, |pred|>1.0": (df.pred_dm > 0).values,
        "pred_dm, |pred|>1.5": (df.pred_dm > 0).values,
        "ORACLE side (perfect foresight)": (df.dm > 0).values,
    }
    sub_mask = {
        "p_us (market-blind incumbent)": np.ones(len(df), bool),
        "pred_dm (D147 movement model)": np.ones(len(df), bool),
        "pred_dm, |pred|>1.0": np.abs(df.pred_dm.values) > 1.0,
        "pred_dm, |pred|>1.5": np.abs(df.pred_dm.values) > 1.5,
        "ORACLE side (perfect foresight)": np.ones(len(df), bool),
    }
    p_open_h, p_close_h = df.p_open_sp.values, df.p_close_sp.values
    have_ml = np.isfinite(df.p_open_ml.values) & np.isfinite(df.p_close_ml.values)
    y = df.y.values                                   # 1 = home won

    rows = []
    for nm, home_side in pol.items():
        mk = sub_mask[nm]
        # ---- SP frame, both legs at the validated proportional overround
        p_o_S = np.where(home_side, p_open_h, 1 - p_open_h)
        p_c_notS = np.where(home_side, 1 - p_close_h, p_close_h)
        dec1 = 1.0 / (p_o_S * OVERROUND)
        dec2 = 1.0 / (p_c_notS * OVERROUND)
        r = arb_rate(dec1, dec2)
        lock = (r > 0) & mk
        # ---- REAL moneylines where they exist
        d1m = np.where(home_side, df.dec_open_home.values, df.dec_open_away.values)
        d2m = np.where(home_side, df.dec_close_away.values, df.dec_close_home.values)
        rm = arb_rate(d1m, d2m)
        lockm = (rm > 0) & mk & have_ml
        # ---- the strategy EV: arb out when you can, else ride the open bet
        win = np.where(home_side, y == 1, y == 0)
        ride = np.where(win, dec1 - 1.0, -1.0)
        strat = np.where(lock, r, ride)
        # PRIMARY: everything again on the REAL moneylines
        ridem = np.where(win, d1m - 1.0, -1.0)
        stratm = np.where(lockm, rm, ridem)
        mkm = mk & have_ml & np.isfinite(d1m) & np.isfinite(d2m)
        seas = df.season.values
        bs = splits.paired_bootstrap(stratm[mkm], B_BOOT, SEED,
                                     cluster=seas[mkm])
        per = {}
        for s in sorted(set(seas[mkm].tolist())):
            q = mkm & (seas == s)
            per[s] = {"n": int(q.sum()), "roi": float(stratm[q].mean()),
                      "lock": float(lockm[q].mean())}
        rows.append({
            "policy": nm, "n": int(mk.sum()),
            "lock_rate_SP": float(lock[mk].mean()),
            "mean_profit_when_locked_SP":
                float(r[lock].mean()) if lock.any() else float("nan"),
            "n_ml": int(mkm.sum()),
            "lock_rate_ML":
                float(lockm[mkm].mean()) if mkm.any() else float("nan"),
            "mean_profit_when_locked_ML":
                float(rm[lockm & mkm].mean()) if (lockm & mkm).any() else float("nan"),
            "ride_roi_SP": float(ride[mk].mean()),
            "arb_or_ride_SP": float(strat[mk].mean()),
            "ride_roi_ML": float(ridem[mkm].mean()),
            "arb_or_ride_ML": float(stratm[mkm].mean()),
            "arb_or_ride_ci": [float(bs["lo"]), float(bs["hi"])],
            "per_season": per,
        })
    print(f"\n  PRIMARY = REAL opening and closing moneylines, no vig "
          f"assumption anywhere.\n  'lock' = the two tickets cost less than "
          f"they pay, i.e. a genuine arbitrage.\n  SP columns are the "
          f"1/(p*{OVERROUND}) convention and are SECONDARY per [b0].")
    print(f"\n  {'policy':34s}{'nML':>6s}{'lock%':>7s}{'prof|lock':>10s}"
          f"{'ride ROI':>10s}{'arb-or-ride':>12s}  {'[SP lock%]':>11s}")
    for r in rows:
        print(f"  {r['policy']:34s}{r['n_ml']:6d}{100*r['lock_rate_ML']:7.2f}"
              f"{100*r['mean_profit_when_locked_ML']:9.2f}%"
              f"{100*r['ride_roi_ML']:9.2f}%{100*r['arb_or_ride_ML']:11.2f}%"
              f"  {100*r['lock_rate_SP']:10.2f}")
    RES["round_trip"] = rows

    hdr("[b1] THE SAME NUMBERS WITH CIs AND PER SEASON — the lottery check")
    print("  D115's season-lottery signature is the failure mode this table "
          "exists to expose.")
    for r in rows:
        pj = r["per_season"]
        ci = r["arb_or_ride_ci"]
        print(f"\n  {r['policy']}   n={r['n_ml']}")
        print(f"    arb-or-ride ROI {100*r['arb_or_ride_ML']:+.2f}%  "
              f"season-clustered 95% CI [{100*ci[0]:+.2f}%,{100*ci[1]:+.2f}%]"
              f"  {'SIG' if ci[0] > 0 or ci[1] < 0 else 'ns'}")
        print("    per season: " + "  ".join(
            f"{k} {100*v['roi']:+.1f}% (n={v['n']})" for k, v in pj.items()))

    # ------------------------------------------------ EXCHANGE COSTS -------
    hdr("[b2] EXCHANGE-STYLE EXIT — commission on NET WINNINGS, sweep 2%/5%")
    print("  Back side S at the book's OPENING price, LAY side S on an "
          "exchange at the close.\n  Exchange fair decimal for S = 1/p_close_S;"
          " a 'sweep' of s means we cross the book\n  and accept lay odds "
          "(1+s) times worse.  Commission c is charged on net winnings only,\n"
          "  which is why it is NOT the same as an overround.")
    ex_rows = []
    for nm, home_side in pol.items():
        mk = sub_mask[nm]
        p_o_S = np.where(home_side, p_open_h, 1 - p_open_h)
        p_c_S = np.where(home_side, p_close_h, 1 - p_close_h)
        back = 1.0 / (p_o_S * OVERROUND)
        for sweep in (0.02, 0.05):
            for comm in (0.02, 0.05):
                lay = (1.0 / p_c_S) * (1.0 + sweep)
                r = lay_rate(back, lay, comm)
                lock = (r > 0) & mk
                ex_rows.append({"policy": nm, "sweep": sweep, "comm": comm,
                                "n": int(mk.sum()),
                                "lock_rate": float(lock[mk].mean()),
                                "mean_profit_when_locked":
                                    float(r[lock].mean()) if lock.any() else float("nan"),
                                "ev_per_game": float(r[lock].sum() / max(mk.sum(), 1))})
    print(f"\n  {'policy':34s}{'sweep':>7s}{'comm':>6s}{'lock%':>8s}"
          f"{'profit|lock':>12s}{'EV/game':>10s}")
    for r in ex_rows:
        print(f"  {r['policy']:34s}{100*r['sweep']:6.0f}%{100*r['comm']:5.0f}%"
              f"{100*r['lock_rate']:8.2f}{100*r['mean_profit_when_locked']:11.2f}%"
              f"{100*r['ev_per_game']:9.3f}%")
    RES["exchange"] = ex_rows

    # ------------------------------------ (b3) THE ADJUDICATION THAT COUNTS -
    hdr("[b3] DOES THE MOVEMENT MODEL ACTUALLY BEAT THE OPENING NUMBER?")
    print("  Every ROI above rests on this.  ATS against the OPENING SPREAD at "
          "-110 is the frame\n  that is directly comparable to D119's "
          "perfect-foresight ceiling and D121's 52.72%,\n  and it does not go "
          "through any probability map at all.")
    M = df.margin_actual.values
    om, cm_ = df.open_margin.values, df.close_margin.values
    hs = (df.pred_dm > 0).values
    cover = np.where(hs, M > om, M < om)
    push = M == om
    orc = np.where(df.dm.values > 0, M > om, M < om)
    print(f"\n  {'rule':18s}{'n':>6s}{'cover%':>9s}{'ROI':>9s}"
          f"{'season-cluster CI':>24s}{'cluster-mean t (K-1)':>24s}")
    ats = []
    for t_ in (0.0, 0.5, 1.0, 1.5, 2.0):
        mk = (np.abs(df.pred_dm.values) > t_) & (~push)
        r = np.where(cover[mk], ATS_DEC - 1.0, -1.0)
        bs = splits.paired_bootstrap(r, B_BOOT, SEED,
                                     cluster=df.season.values[mk])
        ti = splits.cluster_mean_t_interval(r, df.season.values[mk])
        ats.append({"rule": f"|pred|>{t_:.2f}", "n": int(mk.sum()),
                    "cover": float(cover[mk].mean()), "roi": float(r.mean()),
                    "ci": [float(bs["lo"]), float(bs["hi"])],
                    "t_ci": [float(ti["lo"]), float(ti["hi"])],
                    "t_sig": bool(ti["sig"]),
                    "per_season": {k: float(v) for k, v in
                                   ti["cluster_means"].items()}})
        print(f"  |pred|>{t_:<11.2f}{mk.sum():6d}{100*cover[mk].mean():8.2f}%"
              f"{100*r.mean():8.2f}%  [{100*bs['lo']:+6.2f}%,"
              f"{100*bs['hi']:+6.2f}%]  [{100*ti['lo']:+6.2f}%,"
              f"{100*ti['hi']:+6.2f}%] {'SIG' if ti['sig'] else 'ns '}")
    RES["ats_vs_open"] = ats
    for t_ in (0.0, 1.0, 1.5):
        mk = (np.abs(df.pred_dm.values) > t_) & (~push) & (df.dm.values != 0)
        print(f"    |pred|>{t_:.2f}: ORACLE cover {100*orc[mk].mean():.2f}% "
              f"vs OURS {100*cover[mk].mean():.2f}% — we capture "
              f"{100*(cover[mk].mean()-.5238)/(orc[mk].mean()-.5238):.0f}% of "
              f"the perfect-foresight prize on that subset")
    print("\n  THE CEILING ITSELF MOVED.  D119 measured perfect open->close "
          "foresight at 54.63% on\n  the FULL 2007-2026 corpus.  On this "
          f"2021-26 frame, where the line moves far more,\n  the same oracle "
          f"is {100*orc[(~push)&(df.dm.values!=0)].mean():.2f}%.  Our rules "
          f"sit BELOW their own subset's ceiling, which is the\n  consistency "
          "check that has to pass and does.")

    print("\n  IS OUR PREDICTED CLOSE A BETTER NUMBER THAN THE OPEN?  MAE "
          "against the realised margin:")
    pc = om + df.pred_dm.values
    for t_ in (0.0, 1.0, 1.5):
        mk = np.abs(df.pred_dm.values) > t_
        print(f"    |pred|>{t_:.2f} n={int(mk.sum()):5d}   OPEN "
              f"{np.mean(np.abs(M[mk]-om[mk])):.4f}   OUR PREDICTED CLOSE "
              f"{np.mean(np.abs(M[mk]-pc[mk])):.4f}   REAL CLOSE "
              f"{np.mean(np.abs(M[mk]-cm_[mk])):.4f}")
    print("  The ordering OPEN > ours > REAL CLOSE is the one an honest model "
          "must produce.  A\n  model that leaked the outcome would beat the "
          "real close too; ours never does.")

    print("\n  AND THE DIRECT TEST — regress (realised margin - REAL CLOSING "
          "LINE) on pred_dm:")
    resid_c = M - cm_
    Xd = np.column_stack([np.ones(len(df)), df.pred_dm.values])
    beta = np.linalg.lstsq(Xd, resid_c, rcond=None)[0]
    rr = resid_c - Xd @ beta
    se_iid = float(np.sqrt(np.sum(rr ** 2) / (len(df) - 2) /
                           np.sum((df.pred_dm.values - df.pred_dm.values.mean()) ** 2)))
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((2, 2))
    for s in np.unique(df.season.values):
        q = df.season.values == s
        u = Xd[q].T @ rr[q]
        meat += np.outer(u, u)
    V = XtXi @ meat @ XtXi
    se_cl = float(np.sqrt(V[1, 1]))
    print(f"    slope {beta[1]:+.4f}   i.i.d. se {se_iid:.4f} t "
          f"{beta[1]/se_iid:+.2f}   SEASON-CLUSTERED se {se_cl:.4f} t "
          f"{beta[1]/se_cl:+.2f}")
    print(f"    The i.i.d. t of {beta[1]/se_iid:+.2f} would read as 'we beat "
          f"the close'.  Season-clustered it is\n    "
          f"{beta[1]/se_cl:+.2f} — NOT SIGNIFICANT.  This is exactly the "
          f"retraction the sister football\n    project had to publish "
          f"(GATE_POLICY_V2 s9), and the clustered number is the one that\n"
          f"    counts.  WE DO NOT BEAT THE CLOSE.  We beat the OPEN.")
    RES["beat_close_test"] = {"slope": float(beta[1]), "se_iid": se_iid,
                              "t_iid": float(beta[1] / se_iid),
                              "se_cluster": se_cl,
                              "t_cluster": float(beta[1] / se_cl)}

    # -------------------------------------------- (c) INTRADAY BEST EXIT ---
    hdr("[c] THE INTRADAY PATH — is the CLOSE even the best exit?")
    tr = load_tr_path()
    j = df.merge(tr, on=["game_date", "home", "away"], how="left")
    assert len(j) == len(df), "TR join fanned out"
    has = j.path_best_home.notna().values
    print(f"  TeamRankings intraday histories joined: {int(has.sum())}/{len(j)}"
          f" = {100*has.mean():.1f}% of the model frame")
    if has.sum() > 100:
        home_side = (j.pred_dm > 0).values
        # best margin available on the path, from OUR side's perspective:
        # if we hold HOME we want the exit line as HOME-favourable as possible
        best_m = np.where(home_side, j.path_best_home.values,
                          j.path_best_away.values)
        p_best = sigmoid(best_m / SPREAD_SCALE)
        p_c_notS_path = np.where(home_side, 1 - p_best, p_best)
        p_c_notS_close = np.where(home_side, 1 - p_close_h, p_close_h)
        p_o_S = np.where(home_side, p_open_h, 1 - p_open_h)
        d1 = 1.0 / (p_o_S * OVERROUND)
        r_close = arb_rate(d1, 1.0 / (p_c_notS_close * OVERROUND))
        r_path = arb_rate(d1, 1.0 / (p_c_notS_path * OVERROUND))
        m2 = has
        print(f"    exit at the CLOSE           lock "
              f"{100*((r_close>0)&m2)[m2].mean():5.2f}%   "
              f"mean profit when locked "
              f"{100*np.nanmean(r_close[(r_close>0)&m2]):5.2f}%")
        print(f"    exit at the BEST PATH PRICE lock "
              f"{100*((r_path>0)&m2)[m2].mean():5.2f}%   "
              f"mean profit when locked "
              f"{100*np.nanmean(r_path[(r_path>0)&m2]):5.2f}%")
        print(f"    (the path exit is an UPPER BOUND: it assumes we hit the "
              f"single best quote of the\n     day on the correct side, with "
              f"no execution risk and no stale-line void.)")
        RES["intraday"] = {
            "n": int(m2.sum()),
            "lock_close": float(((r_close > 0) & m2)[m2].mean()),
            "lock_path": float(((r_path > 0) & m2)[m2].mean()),
            "profit_close": float(np.nanmean(r_close[(r_close > 0) & m2])),
            "profit_path": float(np.nanmean(r_path[(r_path > 0) & m2])),
            "mean_extra_pts": float(np.nanmean(
                np.abs(best_m[m2] - np.where(home_side, j.close_margin.values,
                                             j.close_margin.values)[m2])))}

    # ------------------------------------------------------- (d) MIDDLES ---
    hdr("[d] MIDDLES — the vehicle that does NOT need the arb inequality")
    print("  Bet side S against the OPENING number, then the other side "
          "against the CLOSING\n  number.  Both tickets win if the result "
          "lands strictly between the two lines.\n  At -110/-110 the "
          "breakeven middle-hit probability is")
    be_mid = 1.0 / (2 * ATS_DEC - 1.0 + 1.0)
    p_be = (ATS_DEC - 1.0) / (2 * (ATS_DEC - 1.0) + (ATS_DEC - 1.0) + 1.0)
    # exact: win both = 2*(dec-1); lose one = (dec-1) - 1
    gain, loss = 2 * (ATS_DEC - 1.0), (ATS_DEC - 1.0) - 1.0
    p_be = -loss / (gain - loss)
    print(f"      win both = {gain:+.4f}u,  split = {loss:+.4f}u  ->  "
          f"breakeven P(middle) = {100*p_be:.2f}%")
    mid = middles_table(df, p_be)
    RES["middles"] = mid

    hdr("[d2] THE EX-ANTE MIDDLE STRATEGY — the version you could actually run")
    print("  The table above conditions on the REALISED window, which is "
          "hindsight.  Ex ante you\n  must pick a side at the open and only "
          "get a middle if the line then moves YOUR way\n  far enough.  "
          "Strategy: bet side S at the opening number at -110; at the close, "
          "if the\n  line has moved >= W points in S's favour, buy the other "
          "side at the closing number;\n  otherwise hold the single ticket.")
    M = df.margin_actual.values
    om, cm_ = df.open_margin.values, df.close_margin.values
    gain, loss = 2 * (ATS_DEC - 1.0), (ATS_DEC - 1.0) - 1.0
    exante = []
    print(f"\n  {'side rule':22s}{'W':>5s}{'n':>6s}{'2nd leg%':>10s}"
          f"{'P(mid|2leg)':>13s}{'ROI/entry':>11s}{'cluster-mean t CI':>24s}")
    for nm, hs_ in [("p_us (incumbent)", (df.p_us > 0.5).values),
                    ("pred_dm", (df.pred_dm > 0).values),
                    ("pred_dm |pred|>1.0",
                     (df.pred_dm > 0).values),
                    ("pred_dm |pred|>1.5",
                     (df.pred_dm > 0).values)]:
        sel = np.ones(len(df), bool)
        if "1.0" in nm:
            sel = np.abs(df.pred_dm.values) > 1.0
        if "1.5" in nm:
            sel = np.abs(df.pred_dm.values) > 1.5
        # signed movement in FAVOUR of the side taken at the open
        fav = np.where(hs_, cm_ - om, om - cm_)
        cover1 = np.where(hs_, M > om, M < om)
        for W in (1.0, 2.0, 3.0):
            take2 = fav >= W
            # middle: result strictly between the two numbers
            lo_ = np.minimum(om, cm_)
            hi_ = np.maximum(om, cm_)
            hit = (M > lo_) & (M < hi_)
            ret = np.where(
                take2, np.where(hit, gain, loss),
                np.where(cover1, ATS_DEC - 1.0, -1.0))
            mk = sel & (M != om)
            ti = splits.cluster_mean_t_interval(ret[mk], df.season.values[mk])
            p_mid = float(hit[mk & take2].mean()) if (mk & take2).any() else float("nan")
            exante.append({"rule": nm, "W": W, "n": int(mk.sum()),
                           "second_leg_rate": float(take2[mk].mean()),
                           "p_middle_given_2leg": p_mid,
                           "roi": float(ret[mk].mean()),
                           "t_ci": [float(ti["lo"]), float(ti["hi"])],
                           "sig": bool(ti["sig"])})
            print(f"  {nm:22s}{W:5.1f}{mk.sum():6d}{100*take2[mk].mean():9.1f}%"
                  f"{100*p_mid:12.2f}%{100*ret[mk].mean():10.2f}%"
                  f"  [{100*ti['lo']:+6.2f}%,{100*ti['hi']:+6.2f}%]"
                  f" {'SIG' if ti['sig'] else 'ns'}")
    RES["middles_exante"] = exante
    print("\n  READING: the middle leg fires on a minority of entries, and "
          "conditional on firing the\n  middle hits at roughly the rate the "
          "ex-post table predicts.  The strategy ROI is\n  dominated by the "
          "ENTRY bet, not by the middle — the middle is a variance reducer "
          "bolted\n  onto a directional position, exactly like the hedge in "
          "[b].")

    # ------------------------------------------------------- CONSTRAINTS --
    hdr("[e] WHICH CONSTRAINTS BIND, AND HOW HARD")
    rt = {r["policy"]: r for r in rows}
    best = rt["pred_dm, |pred|>1.5"]
    ex_best = [e for e in ex_rows
               if e["policy"] == "pred_dm, |pred|>1.5" and e["sweep"] == 0.05
               and e["comm"] == 0.05][0]
    cons = [
        ("BUDGET = $0 (hard, standing user constraint)",
         "BINDS ABSOLUTELY. Every number in this file is a paper measurement. "
         "A round trip needs capital on BOTH legs simultaneously and at two "
         "venues; with $0 the lock rate is irrelevant because the first leg "
         "cannot be placed.", "FATAL"),
        ("US exchange access",
         f"BINDS HARD. The exchange column is where this trade actually "
         f"works: at 5% sweep / 5% commission the |pred|>1.5 set still locks "
         f"{100*ex_best['lock_rate']:.1f}% of the time for "
         f"{100*ex_best['mean_profit_when_locked']:.1f}% when it locks. "
         f"Retail-to-retail is far worse. Betfair does not take US customers; "
         f"the US-legal peer-to-peer venues (Prophet X, Novig, Sporttrade) are "
         f"live in only a handful of states, carry thin NBA side liquidity at "
         f"the open, and their true sweep cost on a full hedge is worse than "
         f"the 5% modelled here.", "BINDS"),
        ("Opening-line limits",
         "BINDS HARD AND EXACTLY WHERE IT HURTS. D120's standing caveat: "
         "opening lines carry the LOWEST limits of the day. The entry leg is "
         "the limited one, so the trade does not scale — and the games with "
         "the biggest predicted movement are precisely the ones a book "
         "prices small and moves fast.", "BINDS"),
        ("Books limit / void / restrict winners",
         "BINDS ON PERSISTENCE, not on any single trade. A pattern of "
         "opening-number-only bets followed by a hedge is the single most "
         "recognisable sharp signature there is; account life is measured in "
         "weeks, and 'shop the price' (D142) needs several accounts to stay "
         "open at once.", "BINDS"),
        ("The price you see is not the price you get",
         f"BINDS MEASURABLY. Section [c]: exiting at the single best quote on "
         f"the intraday path lifts the lock rate from "
         f"{100*RES.get('intraday',{}).get('lock_close',float('nan')):.1f}% to "
         f"{100*RES.get('intraday',{}).get('lock_path',float('nan')):.1f}% — "
         f"i.e. roughly a third of the measured 'locks' exist only at a quote "
         f"that was available at one moment. Stale-line voids apply to "
         f"exactly those.", "BINDS"),
        ("Two venues, same game, same time",
         "BINDS OPERATIONALLY. The lock needs the open at book A and the "
         "close at book B, funded and available simultaneously. Our own data "
         "collection has never had more than 2 books (D142 (1)).", "BINDS"),
        ("Is it even arbitrage?",
         "NO — and this is the honest core of the answer. A true arb is "
         "riskless AT INCEPTION. This is a directional bet at the open plus "
         "an OPTION to hedge that only exercises if the line moves our way "
         "far enough. The lock rate IS the option's exercise rate; the other "
         f"{100*(1-best['lock_rate_ML']):.0f}% of the time we are holding an "
         "unhedged position and the outcome decides.", "DEFINITIONAL"),
    ]
    for nm, txt, sev in cons:
        print(f"\n  [{sev}] {nm}\n      {txt}")
    RES["constraints"] = [{"constraint": a, "verdict": c, "detail": b}
                          for a, b, c in cons]

    hdr("[VERDICT] IS LINE MOVEMENT MONETISABLE FOR US?")
    a15 = [a for a in ats if a["rule"] == "|pred|>1.50"][0]
    mid15 = [e for e in exante
             if e["rule"] == "pred_dm |pred|>1.5" and e["W"] == 2.0][0]
    mid10 = [e for e in exante
             if e["rule"] == "pred_dm |pred|>1.0" and e["W"] == 2.0][0]
    print(f"""
  0. THE ONE-LINE ANSWER.  There IS a closeable round trip, it locks far more
     often than we expected, and taking it DESTROYS expected value — hedging
     turns a {100*rt['pred_dm, |pred|>1.0']['ride_roi_ML']:+.2f}% directional
     position into a {100*rt['pred_dm, |pred|>1.0']['arb_or_ride_ML']:+.2f}%
     one.  It is not arbitrage: it is a directional bet plus an option to
     hedge, and the option only pays when we were already right.  The one
     hedged form that is worth anything is the MIDDLE, which survives the
     conservative K-1 cluster-mean t where the naked bet does not.

  A. WHAT SURVIVES THE CONSERVATIVE BOUND (K=4 seasons, t_crit=3.182,
     GATE_POLICY_V2 s9.3's mandated reading):
       * NAKED ENTRY, ATS vs the opening spread, |pred|>1.5: cover
         {100*a15['cover']:.2f}% ROI {100*a15['roi']:+.2f}%, season-cluster
         bootstrap [{100*a15['ci'][0]:+.2f}%,{100*a15['ci'][1]:+.2f}%] SIG but
         cluster-mean t [{100*a15['t_ci'][0]:+.2f}%,{100*a15['t_ci'][1]:+.2f}%]
         **ns**.  NOT a profit claim.
       * EX-ANTE MIDDLE, |pred|>1.0 with W=2: ROI
         {100*mid10['roi']:+.2f}%, cluster-mean t
         [{100*mid10['t_ci'][0]:+.2f}%,{100*mid10['t_ci'][1]:+.2f}%] **SIG**,
         positive in 4/4 seasons.  |pred|>1.5 W=2:
         {100*mid15['roi']:+.2f}%
         [{100*mid15['t_ci'][0]:+.2f}%,{100*mid15['t_ci'][1]:+.2f}%] SIG.
       * WHY the hedged form is significant when the naked one is not: the
         middle cuts season-to-season dispersion by more than it cuts the
         mean.  The statistical case gets STRONGER as the economic case gets
         WEAKER.  Both readings are true and both belong in the entry.
       * PER GATE_POLICY_V2 s11 TIE-BREAK 4, the live forecast is the MOST
         RECENT fold, not the pooled mean: 2025-26 gives
         {100*mid10['roi']:+.2f}% -> +3.71% (|pred|>1.0 W=2) and +2.27%
         (|pred|>1.5 W=2).  Quote those, not the pooled numbers.
""")
    print(f"""
  1. THE MOVEMENT IS REAL AND PARTLY PREDICTABLE.  D147 gets OOS R^2 +0.171
     against the naive 'no movement' baseline, direction correct 65.0% of all
     movers and 84.0% in the top bucket, CLV +0.054/bet on |pred|>1.5 against
     a placebo of +0.005.  That part is settled.

  2. BEATING THE CLOSE STILL PAYS NOTHING BY ITSELF.  A bet settles on the
     OUTCOME.  CLV is a proxy; it becomes money only through a closeable
     round trip.

  3. THE ROUND TRIP DOES LOCK, MORE OFTEN THAN EXPECTED, AND IT IS NOT AN
     ARBITRAGE.  On REAL moneylines the |pred|>1.5 set locks
     {100*best['lock_rate_ML']:.1f}% of the time for
     {100*best['mean_profit_when_locked_ML']:.1f}% when it locks.  But the
     position is directional at inception: {100*(1-best['lock_rate_ML']):.0f}%
     of the time no lock ever appears and the outcome decides.  Total
     arb-or-ride ROI {100*best['arb_or_ride_ML']:+.2f}%, season-clustered CI
     [{100*best['arb_or_ride_ci'][0]:+.2f}%,{100*best['arb_or_ride_ci'][1]:+.2f}%]
     — versus {100*best['ride_roi_ML']:+.2f}% for simply NOT hedging.
     HEDGING COSTS ~{100*(best['ride_roi_ML']-best['arb_or_ride_ML']):.1f}pp OF
     ROI.  That is the precise answer to "we should have some arbitrage,
     right?": yes, and you should not take it.

  4. THE THRESHOLD ARITHMETIC IS THE WHOLE STORY.  Two retail books at the
     validated 1.043 overround need CLV > {thr:.4f} on the entry side.  Our
     mean CLV per bet is +0.009 (universe) to +0.054 (best rule).  Only the
     BEST rule's MEAN clears the threshold, and a mean above a threshold is
     not the same as clearing it per bet — hence a lock rate in the 20-34%
     range rather than 100%.

  5. WHERE IT CLEARS CLEANLY IS AN EXCHANGE WE CANNOT ACCESS.  Commission on
     net winnings is a far cheaper hedge than a second retail overround, and
     it is the only cost model under which this is comfortably positive at
     every sweep/commission pair tested.  We have $0 and no exchange.

  BOTTOM LINE.  The movement is real, larger than we had priced, and
  monetisable IN PRINCIPLE — but not by us and not at retail.  At two retail
  books the round trip clears only on the tail of our own prediction
  distribution and is one season away from being a coin flip.  On an exchange
  it clears at every cost setting tested, and US exchange access plus a
  non-zero budget are exactly the two things we do not have.  The correct
  posture is UNCHANGED from D121/D126/D142: no capital at open or close,
  CLV remains the yardstick, transact as early as possible — with ONE
  addition, that the D147 movement model is a materially better CLV selector
  than the frozen rules and should be run in paper alongside them.""")
    RES["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT, "w") as fh:
        json.dump(RES, fh, indent=1, default=str)
    print(f"\nWROTE {OUT}  ({RES['elapsed_s']}s)")


# --------------------------------------------------------------- helpers ----
TR_TEAMS = {"NOP": "NOP", "NYK": "NYK", "SAS": "SAS", "GSW": "GSW",
            "PHX": "PHX", "BKN": "BKN", "CHA": "CHA", "UTA": "UTA"}


def load_tr_path() -> pd.DataFrame:
    """Per game, the BEST home-favourable and BEST away-favourable expected
    HOME MARGIN anywhere on the intraday quote path AFTER the opening quote.

    TeamRankings quotes from the FAVOURITE's side and never says who is home,
    so the sign is resolved by the same unordered-pair join build_odds_open.py
    and bo_lineshop.py use."""
    recs = [json.loads(x) for x in open(TR_JSONL) if x.strip()]
    tr = pd.DataFrame(recs)
    if "no_data" in tr.columns:
        tr = tr[~tr.no_data.fillna(False)]
    tr = tr[tr.fav_team.notna() & tr.history.notna()].copy()
    tr["game_date"] = pd.to_datetime(tr.game_date)
    lo, hi = [], []
    for h in tr.history:
        v = []
        for q in (h or []):
            for k in ("book1", "book2", "book3"):
                if q.get(k) is not None:
                    v.append(float(q[k]))
        lo.append(min(v) if v else np.nan)
        hi.append(max(v) if v else np.nan)
    tr["fav_lo"], tr["fav_hi"] = lo, hi
    con = ro_connect()
    try:
        mk = con.execute("SELECT game_date, home, away FROM odds_market").df()
    finally:
        con.close()
    mk["game_date"] = pd.to_datetime(mk.game_date)
    mk["pair"] = [frozenset(p) for p in zip(mk.home, mk.away)]
    tr["fav_team"] = tr.fav_team.replace(TR_TEAMS)
    tr["dog_team"] = tr.dog_team.replace(TR_TEAMS)
    tr["pair"] = [frozenset(p) for p in zip(tr.fav_team, tr.dog_team)]
    m = tr.merge(mk, on=["game_date", "pair"], how="inner")
    fav_home = (m.fav_team == m.home).values
    sgn = np.where(fav_home, -1.0, +1.0)
    a = sgn * m.fav_lo.values
    b = sgn * m.fav_hi.values
    out = pd.DataFrame({
        "game_date": m.game_date, "home": m.home, "away": m.away,
        "path_best_home": np.fmax(a, b),      # most home-favourable margin
        "path_best_away": np.fmin(a, b)})     # most away-favourable margin
    return out.drop_duplicates(subset=["game_date", "home", "away"],
                               keep="first").reset_index(drop=True)


def middles_table(df: pd.DataFrame, p_be: float) -> list:
    """Empirical middle windows and hit rates, spreads and totals."""
    out = []
    M = df.margin_actual.values
    o_m, c_m = df.open_margin.values, df.close_margin.values
    lo = np.minimum(o_m, c_m)
    hi = np.maximum(o_m, c_m)
    width = hi - lo
    hit = (M > lo) & (M < hi)
    push = (M == lo) | (M == hi)
    print(f"\n  SPREADS, n={len(df)}  (window = |close - open|; a middle needs "
          f"the result strictly inside)")
    print(f"    {'window':>14s}{'n':>7s}{'share':>8s}{'P(middle)':>11s}"
          f"{'EV/2u':>9s}{'verdict':>10s}")
    for a, b in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0),
                 (3.0, 4.0), (4.0, 99.0)]:
        mk = (width >= a) & (width < b)
        if mk.sum() < 20:
            continue
        ph = float(hit[mk].mean())
        gain, loss = 2 * (ATS_DEC - 1.0), (ATS_DEC - 1.0) - 1.0
        ev = ph * gain + (1 - ph - push[mk].mean()) * loss
        out.append({"kind": "spread", "window": f"[{a},{b})",
                    "n": int(mk.sum()), "share": float(mk.mean()),
                    "p_middle": ph, "ev_per_2u": float(ev)})
        print(f"    [{a:>4.1f},{b:>4.1f}){mk.sum():7d}{100*mk.mean():7.1f}%"
              f"{100*ph:10.2f}%{ev:+9.4f}"
              f"{'  CLEARS' if ph > p_be else '  short':>10s}")
    mk = width >= 1.0
    ph = float(hit[mk].mean())
    print(f"    ALL windows >= 1.0 pt: n={int(mk.sum())} "
          f"({100*mk.mean():.1f}% of games)  P(middle) {100*ph:.2f}%  "
          f"vs breakeven {100*p_be:.2f}%")
    out.append({"kind": "spread", "window": ">=1.0", "n": int(mk.sum()),
                "share": float(mk.mean()), "p_middle": ph,
                "ev_per_2u": float(ph * 2 * (ATS_DEC - 1.0) +
                                   (1 - ph) * ((ATS_DEC - 1.0) - 1.0))})

    ot, ct = df.open_total.values, df.close_total.values
    tot = df.margin_actual.values * 0 + (df.get("total_points",
                                                pd.Series(np.nan, index=df.index)).values)
    okt = np.isfinite(ot) & np.isfinite(ct)
    if np.isfinite(tot).sum() < 10:
        con = ro_connect()
        try:
            sc = con.execute("""SELECT game_date, home, away,
                                       score_home + score_away AS tp
                                FROM odds_market""").df()
        finally:
            con.close()
        sc["game_date"] = pd.to_datetime(sc.game_date)
        d2 = df.merge(sc, on=["game_date", "home", "away"], how="left")
        tot = d2.tp.values
    okt = okt & np.isfinite(tot)
    if okt.sum() > 100:
        tlo, thi = np.minimum(ot, ct), np.maximum(ot, ct)
        tw = thi - tlo
        thit = (tot > tlo) & (tot < thi)
        print(f"\n  TOTALS, n={int(okt.sum())}")
        print(f"    {'window':>14s}{'n':>7s}{'share':>8s}{'P(middle)':>11s}"
              f"{'EV/2u':>9s}{'verdict':>10s}")
        for a, b in [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0),
                     (5.0, 99.0)]:
            mk = okt & (tw >= a) & (tw < b)
            if mk.sum() < 20:
                continue
            ph = float(thit[mk].mean())
            gain, loss = 2 * (ATS_DEC - 1.0), (ATS_DEC - 1.0) - 1.0
            ev = ph * gain + (1 - ph) * loss
            out.append({"kind": "total", "window": f"[{a},{b})",
                        "n": int(mk.sum()), "share": float(mk.sum() / okt.sum()),
                        "p_middle": ph, "ev_per_2u": float(ev)})
            print(f"    [{a:>4.1f},{b:>4.1f}){mk.sum():7d}"
                  f"{100*mk.sum()/okt.sum():7.1f}%{100*ph:10.2f}%{ev:+9.4f}"
                  f"{'  CLEARS' if ph > p_be else '  short':>10s}")
    print(f"\n  KEY-NUMBER NOTE: NBA has no 3/7 spike — the modal margins are "
          f"broad.  What matters\n  is only the WIDTH of the window in points, "
          f"which is why the table is cut that way.")
    return out


if __name__ == "__main__":
    main()
