#!/usr/bin/env python3
"""ATS19 — AGAINST-THE-SPREAD BETTING AT THE OPENING NUMBER, 19 SEASONS.

THE FIRST OPEN-PRICE TEST IN THIS PROJECT WITH REAL STATISTICAL POWER.

Every open-price verdict the register holds (D121, D142, D148, D159, D161's
open arm) is stuck at K=3 seasons = 2 dof, because real opening MONEYLINES
exist only for 2023-24..2025-26.  Opening SPREADS exist for 19 seasons, and an
ATS bet at a stated -110 needs NO devig and NO synthetic spread->probability
map (D155: that map is optimistic on dogs by 3.1pp; D148: +23.49% synthetic vs
+9.93% real on the same bets).  Our model's native output is an expected
MARGIN, so margin-vs-spread is a direct comparison with no probability
conversion at all.

EVERYTHING IS PRE-REGISTERED IN data/ats19_prereg.md
(sha256 d5d500800006790bcab357a8abdf9ed2f0d7fa7dbcfcbb19cf88d8302e8421f0),
WRITTEN AND HASHED BEFORE THIS FILE WAS RUN.

AVAILABILITY TIER: **BLIND ON ALL 19 SEASONS** (empty OUT sets, one constant
tier, D158).  Every level here is a LOWER BOUND.  No played-set oracle.

Read-only.  nbapred/ and scripts/bet_engine.py UNTOUCHED.  No default changed.
scripts/prod_by_season.py is NOT run and data/capstone_pergame.csv is NOT
written.

  python3 scripts/ats19_score.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import warnings
from math import sqrt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import nbapred.threads                                          # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from lb_longshot import cluster_boot, cluster_mean_t, icc_oneway  # noqa: E402
from nbapred.eval.splits import (Panel, block_bootstrap, loso,   # noqa: E402
                                 rolling_origin)

# --- PATH OVERRIDES ONLY (added for the D173 re-run on the D170/D171 backfilled
# --- data).  Defaults are byte-identical to D162's; no statistic is touched.
K19 = os.environ.get("ATS19_K19") or os.path.join(ROOT, "data",
                                                  "k19_pergame.csv")
_TAG = os.environ.get("ATS19_TAG", "")
ODDS = os.path.join(ROOT, "data", "derived", "odds_open.csv")
CAPSTONE = os.path.join(ROOT, "data", "capstone_pergame.csv")
OUT = os.path.join(ROOT, "data", f"ats19{_TAG}.json")
FRAME_OUT = os.path.join(ROOT, "data", f"ats19_frame{_TAG}.csv.gz")

SEED = 20260804
N_BOOT = 4000
N_PERM = 400

# ---- PRE-REGISTERED CONSTANTS (data/ats19_prereg.md §2 §3 §4) --------------
SCALE_NATIVE = 7.2            # nbapred.model.production.SCALE — PRIMARY
SCALE_D121 = 6.96             # bo_openbacktest.SPREAD_SCALE — reconciliation
THRESHOLDS = [0.0, 1.0, 2.0, 3.0]        # points; DECLARED, NOT TUNED
JUICE = {"-105": 1.0 + 100.0 / 105.0,
         "-110": 1.0 + 100.0 / 110.0,    # HEADLINE
         "-115": 1.0 + 100.0 / 115.0}
DEC = JUICE["-110"]
BE = 1.0 / DEC                            # 0.523810

SEASONS19 = ["%d-%02d" % (y, (y + 1) % 100) for y in range(2007, 2026)]
COVID = {"2019-20", "2020-21"}
LOCKOUT = {"2011-12"}
DEV5 = set(SEASONS19[14:])                # 2021-22..2025-26 (development era)
OOS14 = set(SEASONS19[:14])               # 2007-08..2020-21  <- THE REAL TEST
OOS_DEEP15 = set(SEASONS19[:15])          # D161's block, continuity
WINDOWS = [("POOL19", set(SEASONS19)),
           ("OOS14", OOS14),
           ("DEV5", DEV5),
           ("NOCOVID17", set(SEASONS19) - COVID),
           ("OOS_DEEP15", OOS_DEEP15)]
# OWNER-SCOPED ADDITIONAL WINDOW (D173), OFF BY DEFAULT so D162's five
# pre-registered windows are reproduced byte-identically when it is unset.
# It is a FRAME the owner named in advance ("focus after 2018-12-17, when the
# injury reports begin"), NOT a window selected by looking at results.
REPORT8 = set(SEASONS19[11:])             # 2018-19..2025-26, the report era
if os.environ.get("ATS19_REPORT_ERA"):
    WINDOWS.append(("REPORT8", REPORT8))

K19_ERA = {}
for _s in SEASONS19[0:4]:
    K19_ERA[_s] = "K-A"
for _s in SEASONS19[4:7]:
    K19_ERA[_s] = "K-B"
for _s in SEASONS19[7:12]:
    K19_ERA[_s] = "K-C"
for _s in SEASONS19[12:14]:
    K19_ERA[_s] = "K-D"
for _s in SEASONS19[14:]:
    K19_ERA[_s] = "K-E"
ERA_ORDER = ["K-A", "K-B", "K-C", "K-D", "K-E"]
ERA_DESC = {"K-A": "2007-08..2010-11 pre-lockout CBA",
            "K-B": "2011-12..2013-14 post-lockout",
            "K-C": "2014-15..2018-19 3PT ramp",
            "K-D": "2019-20..2020-21 COVID",
            "K-E": "2021-22..2025-26 development era"}
# |open_spread| strata for the D155 matched control (pre-registered §8.3)
SP_BINS = [0.0, 2.0, 4.0, 6.0, 8.0, 11.0, 1e9]

FRANCHISE = {"SEA": "OKC", "NJN": "BKN", "NOH": "NOP", "NOK": "NOP",
             "VAN": "MEM", "CHH": "CHA"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# =========================================================== inference =======
def clustered(vals, clus, seed=SEED, boot=True):
    vals = np.asarray(vals, float)
    clus = np.asarray(clus)
    K = len(np.unique(clus))
    if len(vals) < 3 or K < 2:
        return dict(n=int(len(vals)), K=int(K),
                    mean=float(vals.mean()) if len(vals) else float("nan"),
                    lo=float("nan"), hi=float("nan"), se=float("nan"),
                    tlo=float("nan"), thi=float("nan"), icc=float("nan"),
                    deff=float("nan"), iid_se=float("nan"), sig=False)
    if boot:
        lo, hi, se = cluster_boot(vals, clus, n_boot=N_BOOT, seed=seed)
    else:
        lo = hi = se = float("nan")
    tlo, thi, K = cluster_mean_t(vals, clus)
    icc, deff = icc_oneway(vals, clus)
    iid = float(vals.std(ddof=1) / sqrt(len(vals)))
    return dict(n=int(len(vals)), K=int(K), mean=float(vals.mean()),
                lo=lo, hi=hi, se=se, tlo=float(tlo), thi=float(thi),
                icc=float(icc), deff=float(deff), iid_se=iid,
                deff_boot=float(se / iid) if (boot and iid) else float("nan"),
                sig=bool(np.isfinite(tlo) and (tlo > 0 or thi < 0)))


def tag(d):
    if not np.isfinite(d.get("tlo", np.nan)):
        return "  . "
    if d["tlo"] > 0:
        return " SIG+"
    if d["thi"] < 0:
        return " SIG-"
    return "  ns "


def mde80_iid(d):
    d = np.asarray(d, float)
    return float(2.802 * d.std(ddof=1) / sqrt(len(d))) if len(d) > 2 else np.nan


def mde80_clustered(d, clus):
    """(t_{.975,K-1} + z_{.80}) * se(cluster means) — the honest MDE on the
    statistic this run actually ships."""
    s = pd.DataFrame({"v": np.asarray(d, float), "c": np.asarray(clus)})
    mu = s.groupby("c").v.mean().values
    K = len(mu)
    if K < 2:
        return float("nan")
    # D173: the table was sparse and fell back to 2.101 (dof 18) for any dof it
    # did not list — which UNDERSTATES MDE80 at the K=8 report-era frame
    # (dof 7 -> 2.365).  Completed to match lb_longshot.cluster_mean_t exactly.
    # Affects the MDE80 reporting statistic only; no point estimate or CI moves.
    tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
          13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
          18: 2.101, 19: 2.093, 20: 2.086}.get(K - 1, 2.101)
    return float((tq + 0.8416) * mu.std(ddof=1) / sqrt(K))


# =============================================================== frame =======
def build():
    df = pd.read_csv(K19, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    oo = pd.read_csv(ODDS, parse_dates=["game_date"])
    keep = ["season", "game_date", "home", "away", "open_spread",
            "close_spread", "open_margin", "close_margin", "open_total",
            "close_total", "score_home", "score_away", "source"]
    n0 = len(df)
    m = df.merge(oo[keep], on=["season", "game_date", "home", "away"],
                 how="left")
    assert len(m) == n0, "odds_open join fanned out — not 1:1"
    m["frame_all"] = True
    m = m[m.open_margin.notna() & m.score_home.notna()].copy()
    m["margin_actual"] = (m.score_home - m.score_away).astype(float)
    m["era"] = m.season.map(K19_ERA)
    m["m_us"] = SCALE_NATIVE * logit(m.p_us.values)
    m["m_us696"] = SCALE_D121 * logit(m.p_us.values)
    m["edge"] = m.m_us - m.open_margin
    m["edge696"] = m.m_us696 - m.open_margin
    return m.sort_values(["game_date", "game_id"]).reset_index(drop=True), n0


def ats(m, sel, edge, line, dec=DEC, actual=None):
    """Score an ATS bet set.  Returns per-bet arrays.
    pick_home = edge > 0; push = (actual - line) == 0."""
    e = np.asarray(edge)[sel]
    ln = np.asarray(line)[sel]
    act = (m.margin_actual.values if actual is None else np.asarray(actual))[sel]
    pick_home = e > 0
    diff = act - ln
    push = diff == 0
    cover = np.where(pick_home, diff > 0, diff < 0)
    pnl = np.where(push, 0.0, np.where(cover, dec - 1.0, -1.0))
    return pick_home, cover, push, pnl


def cell(m, sel, edge, line, dec=DEC, seed=SEED, boot=True):
    if sel.sum() < 5:
        return {"n": int(sel.sum())}
    pick_home, cover, push, pnl = ats(m, sel, edge, line, dec)
    live = ~push
    clus = m.season.values[sel]
    r = clustered(pnl, clus, seed, boot)
    hit = float(cover[live].mean())
    return {"n": int(sel.sum()), "n_live": int(live.sum()),
            "n_push": int(push.sum()), "push_pct": float(100 * push.mean()),
            "hit": hit, "breakeven": float(1.0 / dec),
            "edge_pp": float(hit - 1.0 / dec),
            "pct_home": float(pick_home.mean()),
            "roi": r, "mde80_iid": mde80_iid(pnl),
            "mde80_clustered": mde80_clustered(pnl, clus)}


def fmt(name, c):
    if "roi" not in c:
        return f"    {name:<26}{c.get('n',0):>7}   (too few)"
    r = c["roi"]
    return (f"    {name:<26}{c['n']:>7}{c['n_push']:>6}{100*c['hit']:>8.3f}"
            f"{100*c['breakeven']:>8.3f}{100*c['edge_pp']:>+8.3f}"
            f"{100*r['mean']:>+8.2f}"
            f"  [{100*r['lo']:>+6.2f},{100*r['hi']:>+6.2f}]"
            f"  [{100*r['tlo']:>+6.2f},{100*r['thi']:>+6.2f}]{tag(r)}"
            f"{r['K']:>4}")


HDR = (f"    {'arm':<26}{'n':>7}{'push':>6}{'cover%':>8}{'be%':>8}{'edge':>8}"
       f"{'ROI%':>8}{'  [95% cluster boot]':>22}{'  [K-1 cluster-mean t]':>24}"
       f"{'sig':>5}{'K':>4}")


# ================================================================ main =======
def main():
    res = {"design": "ATS19 — model expected margin vs the OPENING SPREAD, "
                     "19 contiguous seasons, availability-BLIND (LOWER BOUND)",
           "prereg_sha256": sha256(os.path.join(ROOT, "data",
                                                "ats19_prereg.md")),
           "md5": {os.path.basename(p): md5(p) for p in (K19, ODDS, CAPSTONE)},
           "constants": {"SCALE_NATIVE": SCALE_NATIVE,
                         "SCALE_D121": SCALE_D121,
                         "thresholds_pts": THRESHOLDS,
                         "juice": {k: [v, 1.0 / v] for k, v in JUICE.items()},
                         "seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM},
           "era_coding": K19_ERA}

    print("=" * 138)
    print("ATS19 — OUR EXPECTED MARGIN vs THE OPENING SPREAD, 19 CONTIGUOUS "
          "SEASONS (2007-08..2025-26).  AVAILABILITY-BLIND: EVERY LEVEL IS A "
          "LOWER BOUND.")
    print(f"  PRE-REGISTRATION data/ats19_prereg.md sha256 "
          f"{res['prereg_sha256']}")
    print("  Headline juice -110 (breakeven 52.3810%).  Shipping statistic = "
          "the K-1 = 18 dof season-cluster-mean t interval.")
    print("=" * 138)
    for k, v in res["md5"].items():
        print(f"    {v}  {k}")

    m, n0 = build()
    print(f"\n[0] FRAME.  k19_pergame rows {n0} -> joined 1:1 -> "
          f"{len(m)} with an OPENING SPREAD and a final score "
          f"({100*len(m)/n0:.2f}%).")
    fx_rows = int(((m.home != m.home_nba) | (m.away != m.away_nba)).sum())
    print(f"    D161 FRANCHISE-CODE FIX, INHERITED AND RE-VERIFIED: "
          f"{fx_rows} rows carry a season abbreviation != the modern code "
          f"(SEA/NJN/NOH).  Without it they drop.")
    cov = m.groupby("season").apply(lambda g: pd.Series({
        "n": len(g), "era": K19_ERA[g.name],
        "open_sp": int(g.open_margin.notna().sum()),
        "close_sp": int(g.close_margin.notna().sum()),
        "open_tot": int(g.open_total.notna().sum()),
        "push": int((g.margin_actual == g.open_margin).sum()),
        "push_pct": round(100 * float((g.margin_actual == g.open_margin).mean()), 3),
        "int_line_pct": round(100 * float((g.open_margin == g.open_margin.round()).mean()), 1),
        "src": "/".join(sorted(g.source.dropna().unique())),
    }), include_groups=False)
    print(cov.to_string())
    tot_push = int((m.margin_actual == m.open_margin).sum())
    print(f"    TOTAL n={len(m)}   PUSHES {tot_push} = "
          f"{100*tot_push/len(m):.4f}%   (close-line push rate "
          f"{100*float((m.margin_actual==m.close_margin).mean()):.4f}%)")
    print(f"    OPENING TOTALS present on {int(m.open_total.notna().sum())} of "
          f"{len(m)} — never bet, see the totals paragraph.")
    res["coverage"] = cov.reset_index().to_dict("records")
    res["push"] = {"n_push_open": tot_push, "pct_open": 100 * tot_push / len(m),
                   "pct_close": 100 * float((m.margin_actual ==
                                             m.close_margin).mean()),
                   "int_line_pct": 100 * float((m.open_margin ==
                                                m.open_margin.round()).mean())}
    res["n_frame"] = int(len(m))
    res["franchise_fix_rows"] = fx_rows

    okm = m.edge.values != 0.0
    print(f"    edge == 0.0 exactly (no bet): {int((~okm).sum())}")

    # ================================================ [1] THE PRIMARY ARM ====
    print(f"\n{'='*138}\n[1] THE ALL-GAMES ARM — NO SELECTION AT ALL.  Bet HOME "
          f"if 7.2*logit(p_us) > open_margin, else AWAY, every game, -110.\n"
          f"    THIS IS THE PRIMARY, HONEST BASELINE: does our margin beat the "
          f"opening spread across 19 seasons, unconditionally?\n{'='*138}")
    print(f"\n    PER SEASON (all games, -110):")
    print(f"    {'season':<10}{'era':<6}{'n':>6}{'push':>6}{'cover%':>9}"
          f"{'vs 52.381':>11}{'ROI%':>9}{'CLV pts':>10}{'':<4}")
    per_season = []
    for s in SEASONS19:
        sel = (m.season.values == s) & okm
        ph, cv, pu, pn = ats(m, sel, m.edge.values, m.open_margin.values)
        live = ~pu
        h = float(cv[live].mean())
        clvp = np.where(ph, 1.0, -1.0) * (m.close_margin.values[sel]
                                          - m.open_margin.values[sel])
        clvp = clvp[np.isfinite(clvp)]
        r = {"season": s, "era": K19_ERA[s], "n": int(sel.sum()),
             "n_push": int(pu.sum()), "hit": h, "edge_pp": h - BE,
             "roi": float(pn.mean()), "clv_pts": float(np.mean(clvp)),
             "pct_home": float(ph.mean())}
        per_season.append(r)
        print(f"    {s:<10}{r['era']:<6}{r['n']:>6}{r['n_push']:>6}"
              f"{100*h:>8.2f}%{100*(h-BE):>+10.2f}{100*r['roi']:>+9.2f}"
              f"{r['clv_pts']:>+10.4f}"
              f"   {'COVER' if h > BE else ''}")
    res["per_season_allgames"] = per_season
    npos = sum(1 for r in per_season if r["hit"] > BE)
    n50 = sum(1 for r in per_season if r["hit"] > 0.5)
    print(f"    SEASONS ABOVE THE -110 BREAKEVEN (52.381%): {npos}/19.   "
          f"SEASONS ABOVE 50.000%: {n50}/19.")
    res["seasons_above_be"] = npos
    res["seasons_above_50"] = n50

    # ------------------------------------------------- pooled, all windows --
    print(f"\n    POOLED, BY WINDOW (-110).  The K-1 cluster-mean t interval "
          f"is the shipping statistic.")
    print(HDR)
    allg = {}
    for wn, ws in WINDOWS:
        sel = okm & m.season.isin(ws).values
        c = cell(m, sel, m.edge.values, m.open_margin.values)
        allg[wn] = c
        print(fmt(f"ALL GAMES  {wn}", c))
    res["allgames_windows"] = allg
    p19 = allg["POOL19"]
    print(f"\n    MDE80 on POOL19: i.i.d. {100*p19['mde80_iid']:.2f}% of ROI; "
          f"CLUSTERED (the shipping scale) {100*p19['mde80_clustered']:.2f}% "
          f"of ROI = {100*p19['mde80_clustered']/DEC:.2f}pp of cover rate.")
    print(f"    ICC {p19['roi']['icc']:+.6f}  DEFF_anova "
          f"{p19['roi']['deff']:.2f}  DEFF_boot {p19['roi']['deff_boot']:.2f}"
          f"  (i.i.d. se {100*p19['roi']['iid_se']:.3f}pp vs clustered se "
          f"{100*p19['roi']['se']:.3f}pp)")

    # ------------------------------------------------------ juice sweep -----
    print(f"\n[2] JUICE SENSITIVITY.  Real books vary by side, by number and by "
          f"era; this corpus carries NO per-game ATS price, so -110 is the "
          f"headline\n    and nothing better than -110 is assumed anywhere.")
    print(f"    {'price':<8}{'decimal':>10}{'breakeven%':>12}"
          f"{'ALL-GAMES cover%':>18}{'ROI%':>9}{'  [K-1 t]':>22}{'sig':>5}")
    jz = {}
    for pr, d in JUICE.items():
        sel = okm
        c = cell(m, sel, m.edge.values, m.open_margin.values, dec=d,
                 boot=False)
        jz[pr] = c
        print(f"    {pr:<8}{d:>10.6f}{100/d:>11.4f}%{100*c['hit']:>17.3f}%"
              f"{100*c['roi']['mean']:>+9.2f}"
              f"  [{100*c['roi']['tlo']:>+7.2f},{100*c['roi']['thi']:>+7.2f}]"
              f"{tag(c['roi']):>5}")
    res["juice"] = jz

    # ================================================= [3] THRESHOLD ARMS ===
    print(f"\n{'='*138}\n[3] THE PRE-DECLARED, UNTUNED THRESHOLD ARMS — "
          f"|edge| >= T points, T in {THRESHOLDS} (declared in the prereg "
          f"before scoring).\n{'='*138}")
    thr = {}
    for wn, ws in WINDOWS:
        print(f"\n   window {wn}")
        print(HDR)
        thr[wn] = {}
        for T in THRESHOLDS:
            sel = okm & (np.abs(m.edge.values) >= T) & m.season.isin(ws).values
            c = cell(m, sel, m.edge.values, m.open_margin.values)
            thr[wn][str(T)] = c
            print(fmt(f"|edge| >= {T:.1f}", c))
    res["thresholds"] = thr

    # ================================================ [4] V3 FULL BATTERY ===
    print(f"\n{'='*138}\n[4] THE FULL V3 BATTERY ON THE PRIMARY (ALL-GAMES) ARM"
          f" — GATE_POLICY_V2 §8.\n{'='*138}")
    sel = okm
    _, _, _, pnl = ats(m, sel, m.edge.values, m.open_margin.values)
    pan = Panel(m.season.values[sel], pnl,
                date=m.game_date.dt.strftime("%Y-%m-%d").values[sel],
                label="ATS19 all-games @open -110")
    bb = block_bootstrap(pan.d, pan.date, B=N_BOOT, seed=SEED, block_days=7)
    print(f"    BLOCK BOOTSTRAP (7-day calendar blocks): ROI "
          f"{100*bb['est']:+.3f}% [{100*bb['lo']:+.3f},{100*bb['hi']:+.3f}] "
          f"se {100*bb['se']:.3f}pp  {'SIG' if bb['sig'] else 'ns'}")
    ro = rolling_origin(pan, B=1000, seed=SEED, min_train=1)
    print(f"    ROLLING-ORIGIN (expanding window, train<=k test k+1), "
          f"{ro['n_folds']} folds: sign consistency "
          f"{ro['sign_consistency']}, mean {100*ro['mean_of_folds']:+.3f}%, "
          f"sd {100*ro['sd_of_folds']:.3f}pp, drift "
          f"{100*ro['drift_per_season']:+.4f}pp/season")
    print(f"      final cumulative ROI "
          f"{100*ro['final_cumulative']['est']:+.3f}% "
          f"[{100*ro['final_cumulative']['lo']:+.3f},"
          f"{100*ro['final_cumulative']['hi']:+.3f}] (i.i.d. within fold)")
    ls = loso(pan, B=1000, seed=SEED)
    print(f"    LOSO (stability diagnostic ONLY — {ls['k']} folds share "
          f"{ls['pairwise_data_overlap']:.0%} of their data, NOT {ls['k']} "
          f"proofs): test-on sign {ls['sign_consistency']}, "
          f"fold range [{100*ls['min_fold']:+.2f},{100*ls['max_fold']:+.2f}]%, "
          f"jackknife range "
          f"[{100*ls['jackknife_range'][0]:+.3f},"
          f"{100*ls['jackknife_range'][1]:+.3f}]%")
    infl = sorted(ls["folds"], key=lambda f: -abs(f["influence"]))[:3]
    for f in infl:
        print(f"      most influential: drop {f['left_out']} -> pooled "
              f"{100*f['jackknife']['est']:+.3f}% "
              f"(influence {100*f['influence']:+.3f}pp)")
    res["battery"] = {"block_bootstrap": bb,
                      "rolling_origin": {k: v for k, v in ro.items()
                                         if k != "folds"},
                      "rolling_origin_folds": [
                          {"test": f["test"], "est": f["fold"]["est"],
                           "cum": f["cumulative"]["est"]} for f in ro["folds"]],
                      "loso": {k: v for k, v in ls.items() if k != "folds"},
                      "loso_influence": [{"left_out": f["left_out"],
                                          "test_on": f["test_on"]["est"],
                                          "jackknife": f["jackknife"]["est"],
                                          "influence": f["influence"]}
                                         for f in ls["folds"]]}

    # ---------------------------------------------------- era decomposition -
    print(f"\n    ERA DECOMPOSITION (K19-ERA coding, DerSimonian-Laird on the "
          f"era means):")
    print(f"    {'era':<6}{'seasons':<32}{'K':>3}{'n':>7}{'cover%':>9}"
          f"{'ROI%':>9}{'  [K-1 t]':>24}{'sig':>5}")
    eras = {}
    for e in ERA_ORDER:
        s = okm & (m.era.values == e)
        c = cell(m, s, m.edge.values, m.open_margin.values)
        eras[e] = c
        print(f"    {e:<6}{ERA_DESC[e]:<32}{c['roi']['K']:>3}{c['n']:>7}"
              f"{100*c['hit']:>8.2f}%{100*c['roi']['mean']:>+9.2f}"
              f"  [{100*c['roi']['tlo']:>+8.2f},{100*c['roi']['thi']:>+8.2f}]"
              f"{tag(c['roi']):>5}")
    pts = np.array([eras[e]["roi"]["mean"] for e in ERA_ORDER])
    ses = np.array([eras[e]["roi"]["se"] for e in ERA_ORDER])
    w = 1 / ses ** 2
    mu = (w * pts).sum() / w.sum()
    Q = float((w * (pts - mu) ** 2).sum())
    dfq = len(pts) - 1
    I2 = max(0.0, 100 * (Q - dfq) / Q) if Q > 0 else 0.0
    from statistics import NormalDist
    z = ((Q / dfq) ** (1 / 3) - (1 - 2 / (9 * dfq))) / sqrt(2 / (9 * dfq))
    pq = 1 - NormalDist().cdf(z)
    verdict = ("ERA-STABLE" if I2 < 50 and pq > 0.10 else
               "ERA-CONDITIONAL" if pts.min() * pts.max() > 0 else
               "ERA-SPECIFIC")
    print(f"    DL: Q={Q:.2f} df={dfq} I2={I2:.1f}% p={pq:.4f}  -> "
          f"**{verdict}**   (sign: {int((pts>0).sum())}/5 eras positive)")
    res["by_era"] = eras
    res["era_heterogeneity"] = {"Q": Q, "df": dfq, "I2": I2, "p": pq,
                                "verdict": verdict}

    # =============================================== [5] DEV vs OOS =========
    print(f"\n{'='*138}\n[5] DEV vs OOS.  The model and its rules were "
          f"developed on 2021-26.  2007-08..2020-21 (14 seasons) is a genuine "
          f"out-of-sample block\n    and IS THE REAL EVIDENCE.\n{'='*138}")
    print(HDR)
    devoos = {}
    for lab, ws in (("DEV5 2021-22..2025-26", DEV5),
                    ("OOS14 2007-08..2020-21", OOS14),
                    ("OOS_DEEP15 (D161)", OOS_DEEP15)):
        for T in THRESHOLDS:
            s = okm & (np.abs(m.edge.values) >= T) & m.season.isin(ws).values
            c = cell(m, s, m.edge.values, m.open_margin.values)
            devoos[f"{lab}|{T}"] = c
            print(fmt(f"{lab.split()[0]} T>={T:.1f}", c))
        print()
    res["dev_vs_oos"] = devoos

    # =============================================== [6] CLV IN POINTS ======
    print(f"\n{'='*138}\n[6] CLV IN SPREAD POINTS — open number vs close "
          f"number, signed toward the side we took.  RAW, CONVENTION-FREE: no "
          f"devig,\n    no overround assumption, no probability map.  This is "
          f"the cleanest CLV statement in the project.\n{'='*138}")
    have = okm & m.close_margin.notna().values
    clv_all = {}
    print(f"    {'arm':<26}{'n':>7}{'CLV pts':>11}"
          f"{'  [95% cluster boot]':>24}{'  [K-1 cluster-mean t]':>26}"
          f"{'sig':>5}{'K':>4}{'  %>0':>8}")
    for T in THRESHOLDS:
        s = have & (np.abs(m.edge.values) >= T)
        ph = m.edge.values[s] > 0
        cp = np.where(ph, 1.0, -1.0) * (m.close_margin.values[s]
                                        - m.open_margin.values[s])
        r = clustered(cp, m.season.values[s], SEED + 7)
        clv_all[str(T)] = {"n": int(s.sum()), **r,
                           "pct_positive": float((cp > 0).mean()),
                           "pct_zero": float((cp == 0).mean())}
        print(f"    {'|edge| >= %.1f' % T:<26}{int(s.sum()):>7}"
              f"{r['mean']:>+11.5f}  [{r['lo']:>+9.5f},{r['hi']:>+9.5f}]"
              f"  [{r['tlo']:>+10.5f},{r['thi']:>+10.5f}]{tag(r)}{r['K']:>4}"
              f"{100*(cp>0).mean():>7.1f}%")
    print(f"\n    PER SEASON (all games): " + ", ".join(
        f"{r['season']} {r['clv_pts']:+.3f}" for r in per_season))
    npos_clv = sum(1 for r in per_season if r["clv_pts"] > 0)
    print(f"    SEASONS WITH POSITIVE POINT-CLV: {npos_clv}/19")
    # windows
    print(f"\n    CLV pts by window (all games):")
    clv_w = {}
    for wn, ws in WINDOWS:
        s = have & m.season.isin(ws).values
        ph = m.edge.values[s] > 0
        cp = np.where(ph, 1.0, -1.0) * (m.close_margin.values[s]
                                        - m.open_margin.values[s])
        r = clustered(cp, m.season.values[s], SEED + 7)
        clv_w[wn] = {"n": int(s.sum()), **r}
        print(f"    {wn:<26}{int(s.sum()):>7}{r['mean']:>+11.5f}"
              f"  [{r['lo']:>+9.5f},{r['hi']:>+9.5f}]"
              f"  [{r['tlo']:>+10.5f},{r['thi']:>+10.5f}]{tag(r)}{r['K']:>4}")
    res["clv_points"] = {"by_threshold": clv_all, "by_window": clv_w,
                         "seasons_positive": npos_clv}

    # =============================================== [7] CONTROLS ===========
    print(f"\n{'='*138}\n[7] CONTROLS.\n{'='*138}")

    # --- 7a within-date permutation placebo (D147/D161) ---------------------
    print(f"\n  (a) WITHIN-DATE PERMUTATION PLACEBO (D147 method), {N_PERM} "
          f"draws, seed {SEED}.  p_us is permuted ACROSS GAMES ON THE SAME "
          f"DATE, so the slate,\n      the price distribution and the "
          f"selection mechanism all survive while the model's information is "
          f"destroyed.")
    rng = np.random.default_rng(SEED)
    dcode = pd.factorize(m.game_date.values)[0]
    order = np.argsort(dcode, kind="stable")
    starts = np.searchsorted(dcode[order], np.arange(dcode.max() + 1))
    ends = np.searchsorted(dcode[order], np.arange(dcode.max() + 1), "right")
    groups = [order[a:b] for a, b in zip(starts, ends) if b - a > 1]
    p_us = m.p_us.values
    line = m.open_margin.values
    act = m.margin_actual.values
    clo = m.close_margin.values
    seas = m.season.values
    plc = {str(T): {"roi": [], "hit": [], "clv": []} for T in THRESHOLDS}
    for b in range(N_PERM):
        pp = p_us.copy()
        for g in groups:
            pp[g] = pp[rng.permutation(g)]
        me = SCALE_NATIVE * logit(pp) - line
        for T in THRESHOLDS:
            s = (me != 0) & (np.abs(me) >= T)
            ph = me[s] > 0
            diff = act[s] - line[s]
            push = diff == 0
            cover = np.where(ph, diff > 0, diff < 0)
            pn = np.where(push, 0.0, np.where(cover, DEC - 1.0, -1.0))
            plc[str(T)]["roi"].append(float(pn.mean()))
            plc[str(T)]["hit"].append(float(cover[~push].mean()))
            cp = np.where(ph, 1.0, -1.0) * (clo[s] - line[s])
            plc[str(T)]["clv"].append(float(np.nanmean(cp)))
    print(f"      {'arm':<16}{'real ROI%':>11}{'placebo ROI%':>14}"
          f"{'sd':>7}{'p(plc>=real)':>14}{'real CLV':>11}{'plc CLV':>10}"
          f"{'sd':>8}{'p':>8}")
    plac = {}
    for T in THRESHOLDS:
        k = str(T)
        rr = thr["POOL19"][k]["roi"]["mean"]
        rc = clv_all[k]["mean"]
        a = np.array(plc[k]["roi"])
        c = np.array(plc[k]["clv"])
        pr = float((a >= rr).mean())
        pc = float((c >= rc).mean())
        plac[k] = {"real_roi": rr, "placebo_roi_mean": float(a.mean()),
                   "placebo_roi_sd": float(a.std(ddof=1)), "p_roi": pr,
                   "real_clv": rc, "placebo_clv_mean": float(c.mean()),
                   "placebo_clv_sd": float(c.std(ddof=1)), "p_clv": pc,
                   "roi_gain_pp": 100 * (rr - a.mean())}
        print(f"      {'|edge|>=%.1f' % T:<16}{100*rr:>+11.2f}"
              f"{100*a.mean():>+14.2f}{100*a.std(ddof=1):>7.2f}{pr:>14.3f}"
              f"{rc:>+11.5f}{c.mean():>+10.5f}{c.std(ddof=1):>8.5f}{pc:>8.3f}")
    res["placebo"] = plac

    # --- 7b family-wise ----------------------------------------------------
    print(f"\n  (b) FAMILY-WISE (D121's 9-vs-14.4 standard).  PRIMARY FAMILY "
          f"pre-registered as 4 thresholds x 5 windows = 20 ROI cells at -110.")
    fam_pos = fam_neg = 0
    for wn, _ in WINDOWS:
        for T in THRESHOLDS:
            c = thr[wn][str(T)]
            if "roi" not in c:
                continue
            if c["roi"]["tlo"] > 0:
                fam_pos += 1
            elif c["roi"]["thi"] < 0:
                fam_neg += 1
    fam2_pos = sum(1 for e in ERA_ORDER
                   for T in THRESHOLDS
                   if (lambda cc: "roi" in cc and cc["roi"]["tlo"] > 0)(
                       cell(m, okm & (m.era.values == e) &
                            (np.abs(m.edge.values) >= T),
                            m.edge.values, m.open_margin.values, boot=False)))
    era_cells = {}
    fam2_neg = 0
    for e in ERA_ORDER:
        for T in THRESHOLDS:
            c = cell(m, okm & (m.era.values == e) &
                     (np.abs(m.edge.values) >= T),
                     m.edge.values, m.open_margin.values, boot=False)
            era_cells[f"{e}|{T}"] = {"n": c.get("n"),
                                     "roi": c.get("roi", {}).get("mean"),
                                     "tlo": c.get("roi", {}).get("tlo"),
                                     "thi": c.get("roi", {}).get("thi")}
            if "roi" in c and np.isfinite(c["roi"]["thi"]) and c["roi"]["thi"] < 0:
                fam2_neg += 1
    print(f"      PRIMARY FAMILY  20 cells, expected 1.0 significant under a "
          f"global null.  OBSERVED SIG POSITIVE: {fam_pos}.  "
          f"SIG NEGATIVE: {fam_neg}.")
    print(f"      SECONDARY (4 T x 5 K19-eras) 20 cells, expected 1.0.  "
          f"OBSERVED SIG POSITIVE: {fam2_pos}.  SIG NEGATIVE: {fam2_neg}.")
    print(f"      Cells overlap heavily (the same bets re-scored), so both "
          f"counts are UPPER BOUNDS on the surprise — as D121 and D161 said "
          f"of their own.")
    res["family_wise"] = {"primary_cells": 20, "primary_expected": 1.0,
                          "primary_sig_pos": fam_pos,
                          "primary_sig_neg": fam_neg,
                          "secondary_cells": 20, "secondary_expected": 1.0,
                          "secondary_sig_pos": fam2_pos,
                          "secondary_sig_neg": fam2_neg,
                          "era_cells": era_cells}

    # --- 7c D155 matched control -------------------------------------------
    print(f"\n  (c) THE D155 MATCHED-CONTROL ANALOGUE FOR ATS.  Control = bet "
          f"the MARKET FAVOURITE against the OPENING SPREAD at -110, from the "
          f"same\n      (season x |open_spread| bin) strata as the arm's own "
          f"bets, reweighted to the arm's stratum distribution.  It removes "
          f"SIDE EXPOSURE,\n      it does NOT remove hypothesis selection "
          f"(D111).")
    sp_bin = np.digitize(np.abs(m.open_margin.values), SP_BINS[1:-1])
    fav_home = m.open_margin.values > 0          # market favourite is HOME
    # control pnl on EVERY game: bet the favourite ATS
    diff_all = m.margin_actual.values - m.open_margin.values
    push_all = diff_all == 0
    cov_fav = np.where(fav_home, diff_all > 0, diff_all < 0)
    pnl_fav = np.where(push_all, 0.0, np.where(cov_fav, DEC - 1.0, -1.0))
    pnl_home = np.where(push_all, 0.0, np.where(diff_all > 0, DEC - 1.0, -1.0))
    pnl_away = np.where(push_all, 0.0, np.where(diff_all < 0, DEC - 1.0, -1.0))
    print(f"      {'arm':<20}{'n':>7}{'armROI%':>10}{'ctrlROI%':>10}"
          f"{'alpha pp':>10}{'  [K-1 t on alpha]':>26}{'sig':>5}")
    mc = {}
    strata = pd.DataFrame({"season": m.season.values, "bin": sp_bin})
    for T in THRESHOLDS:
        s = okm & (np.abs(m.edge.values) >= T)
        _, _, _, pn = ats(m, s, m.edge.values, m.open_margin.values)
        # per-(season,bin) control mean, reweighted to the arm's own counts
        key = list(zip(strata.season.values, strata["bin"].values))
        kd = pd.DataFrame({"k": key, "pnl": pnl_fav, "season": m.season.values})
        cmean = kd.groupby("k").pnl.mean()
        arm_k = pd.Series(key)[s]
        wts = arm_k.value_counts(normalize=True)
        common = [k for k in wts.index if k in cmean.index]
        ctrl = float(sum(wts[k] * cmean[k] for k in common)
                     / sum(wts[k] for k in common))
        # season-level alpha for the clustered t
        rows_a, rows_c = [], []
        for ss in sorted(set(m.season.values[s])):
            sm = s & (m.season.values == ss)
            _, _, _, pns = ats(m, sm, m.edge.values, m.open_margin.values)
            ak = pd.Series(key)[sm]
            wk = ak.value_counts(normalize=True)
            kk = [k for k in wk.index if k in cmean.index]
            rows_a.append(float(pns.mean()))
            rows_c.append(float(sum(wk[k] * cmean[k] for k in kk)
                                / sum(wk[k] for k in kk)))
        alpha = np.array(rows_a) - np.array(rows_c)
        K = len(alpha)
        tq = {13: 2.160, 14: 2.145, 16: 2.120, 18: 2.101, 4: 2.776}.get(
            K - 1, 2.101)
        se = alpha.std(ddof=1) / sqrt(K)
        tlo, thi = alpha.mean() - tq * se, alpha.mean() + tq * se
        mc[str(T)] = {"n": int(s.sum()), "arm_roi": float(pn.mean()),
                      "ctrl_roi": ctrl, "alpha": float(pn.mean()) - ctrl,
                      "alpha_season_mean": float(alpha.mean()),
                      "tlo": float(tlo), "thi": float(thi), "K": K,
                      "sig": bool(tlo > 0 or thi < 0)}
        print(f"      {'|edge|>=%.1f' % T:<20}{int(s.sum()):>7}"
              f"{100*pn.mean():>+10.2f}{100*ctrl:>+10.2f}"
              f"{100*(pn.mean()-ctrl):>+10.2f}"
              f"  [{100*tlo:>+10.2f},{100*thi:>+10.2f}]"
              f"{' SIG' if (tlo>0 or thi<0) else '  ns':>5}")
    for nm, pv in (("ALWAYS FAVOURITE", pnl_fav), ("ALWAYS HOME", pnl_home),
                   ("ALWAYS AWAY", pnl_away)):
        r = clustered(pv, m.season.values, SEED + 11)
        cvr = (cov_fav if nm == "ALWAYS FAVOURITE" else
               (diff_all > 0) if nm == "ALWAYS HOME" else (diff_all < 0))
        mc[nm] = {"n": int(len(pv)), "hit": float(cvr[~push_all].mean()), **r}
        print(f"      {nm:<20}{len(pv):>7}  cover "
              f"{100*cvr[~push_all].mean():>6.3f}%  ROI "
              f"{100*r['mean']:>+7.2f}%  [K-1 t "
              f"{100*r['tlo']:+.2f},{100*r['thi']:+.2f}]{tag(r)}")
    res["matched_control"] = mc

    # =============================================== [8] RECONCILIATION =====
    print(f"\n{'='*138}\n[8] RECONCILIATION TO D121 AND D148.\n{'='*138}")
    print(f"    D121 registered: 'ATS vs the opening spread 52.72% "
          f"CI(51.2,54.2) vs 52.38% breakeven, p=0.325' — measured with "
          f"m_us = 6.96*logit(p_us)\n    on the 2021-26 FULL-FEED frame.  "
          f"Reproduced here on the same convention (BLIND tier, so not "
          f"bit-identical):")
    print(HDR)
    rec = {}
    for lab, ws, ed in (("D121 conv 6.96 POOL19", set(SEASONS19), "edge696"),
                        ("D121 conv 6.96 DEV5", DEV5, "edge696"),
                        ("native 7.2   DEV5", DEV5, "edge"),
                        ("native 7.2   POOL19", set(SEASONS19), "edge")):
        e = m[ed].values
        s = (e != 0) & m.season.isin(ws).values
        c = cell(m, s, e, m.open_margin.values)
        rec[lab] = c
        print(fmt(lab, c))
    agree = float(((m.edge.values > 0) == (m.edge696.values > 0)).mean())
    print(f"    SIDE AGREEMENT between the 7.2 and 6.96 conventions: "
          f"{100*agree:.2f}% of games.")
    rec["side_agreement_7_2_vs_6_96"] = agree
    print(f"    D148 §8 registered on 2021-26 (n=4,849): |pred_dm|>0 cover "
          f"51.91% ROI -0.90%; |pred_dm|>1.0 55.96% +6.83%; |pred_dm|>1.5 "
          f"57.99% +10.70%,\n    ALL ns under the K-1 cluster-mean t at K=4.  "
          f"THAT IS A DIFFERENT SELECTOR: `pred_dm` is D147's predicted "
          f"OPEN->CLOSE LINE MOVEMENT,\n    not our margin.  This run's DEV5 "
          f"row above is the margin-based analogue on the same seasons and is "
          f"NOT the same bet set.")
    res["reconciliation"] = rec

    # frozen-rule bet sets re-priced ATS (secondary, no new selection)
    print(f"\n    SECONDARY, NO NEW SELECTION: the four FROZEN F4 bet sets "
          f"re-priced at -110 ATS against the OPENING SPREAD (same games, "
          f"same side).")
    try:
        import ba_intersection as bi
        import bo_openbacktest as bo
        _ts, _so = bi.team_schedule, bi.star_out_map

        def _tsf(con):
            d = _ts(con)
            d["team"] = d.team.map(lambda t: FRANCHISE.get(t, t))
            return d

        def _sof(con):
            d = _so(con)
            d["team"] = d.team.map(lambda t: FRANCHISE.get(t, t))
            return d
        bo.team_schedule, bo.star_out_map = _tsf, _sof
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mm = bo.build(K19, "p_us", "ATS19", {})
        ps, dec_, ok_ = bo.price_cols(mm, "open", "SP")
        masks, _, _ = bo.registry_masks(mm, ps, "open")
        u = np.zeros(len(mm), bool)
        for v in masks.values():
            u |= np.asarray(v)
        masks["UNION"] = u
        mm["margin_actual"] = mm.score_home - mm.score_away
        # the rule's OWN side (p_us > 0.5), scored ATS at the open
        e_rule = np.where(mm.pick_home.values, 1.0, -1.0)
        fz = {}
        print(HDR)
        for rn in ["R4_LOWT", "T20_D03_10_W", "T20_D03_10",
                   "STAR_FAV_SHARPER", "UNION"]:
            s = np.asarray(masks[rn]) & ok_
            c = cell(mm, s, e_rule, mm.open_margin.values)
            fz[rn] = c
            print(fmt(f"FROZEN {rn}", c))
        res["frozen_rules_ats"] = fz
    except Exception as ex:                                      # pragma: no cover
        print(f"    (frozen-rule ATS arm skipped: {ex})")
        res["frozen_rules_ats_error"] = str(ex)

    # =============================================== [9] TOTALS =============
    tt = m[m.open_total.notna()]
    tot = {"n_with_open_total": int(len(tt)),
           "n_with_both": int((m.open_total.notna() &
                               m.close_total.notna()).sum()),
           "seasons": int(tt.season.nunique()),
           "mean_open_total": float(tt.open_total.mean()),
           "mean_abs_move": float((tt.close_total - tt.open_total).abs().mean()),
           "pct_no_move": float(((tt.close_total - tt.open_total) == 0).mean()),
           "push_rate_open": float(((tt.score_home + tt.score_away)
                                    == tt.open_total).mean()),
           "over_rate_open": float(((tt.score_home + tt.score_away)
                                    > tt.open_total).mean()),
           "sd_total_error": float(((tt.score_home + tt.score_away)
                                    - tt.open_total).std(ddof=1))}
    print(f"\n{'='*138}\n[9] OPENING TOTALS — MEASURED, NOT MODELLED.\n{'='*138}")
    print(f"    n with an opening total {tot['n_with_open_total']} over "
          f"{tot['seasons']} seasons; both open+close "
          f"{tot['n_with_both']}.  mean opening total "
          f"{tot['mean_open_total']:.2f}; mean |open->close move| "
          f"{tot['mean_abs_move']:.4f} pts; never moves "
          f"{100*tot['pct_no_move']:.1f}%.")
    print(f"    Naive OVER at the opening total covers "
          f"{100*tot['over_rate_open']:.3f}% (push "
          f"{100*tot['push_rate_open']:.3f}%); sd of "
          f"(actual total - opening total) = {tot['sd_total_error']:.3f} pts.")
    print(f"    NO TOTALS MODEL IS BUILT.  See the totals paragraph.")
    res["totals"] = tot

    # ---------------------------------------------------------- persist -----
    keepcols = ["season", "era", "game_id", "game_date", "home", "away", "y",
                "p_us", "m_us", "m_us696", "open_margin", "close_margin",
                "open_spread", "close_spread", "open_total", "close_total",
                "margin_actual", "edge", "edge696"]
    m[keepcols].to_csv(FRAME_OUT, index=False, compression="gzip")
    res["capstone_md5_after"] = md5(CAPSTONE)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(f"\nwrote {OUT} and {FRAME_OUT}")
    print(f"capstone_pergame.csv md5 AFTER the run: "
          f"{res['capstone_md5_after']}  "
          f"({'UNCHANGED' if res['capstone_md5_after'] == res['md5']['capstone_pergame.csv'] else '*** CHANGED ***'})")


if __name__ == "__main__":
    main()
