#!/usr/bin/env python3
"""K19-RULES — THE FOUR FROZEN BETTING RULES SCORED ON 19 CONTIGUOUS SEASONS.

**THE QUESTION K=3 COULD NOT ANSWER: DO THESE RULES HAVE AN EDGE AT ALL?**

Every registered verdict on the rules (D121, D126, D142, D155, D159) was taken
on 3-4 seasons = 2-3 degrees of freedom on the season-clustered statistic — a
test that cannot reject anything.  D160 landed 19 contiguous seasons that are
both scorable and carry market odds.  This is the first frame with power.

MODEL INPUT: `data/k19_pergame.csv` (scripts/k19_model.py), the certified
production stack scored **AVAILABILITY-BLIND (EMPTY OUT SETS) ON ALL 19
SEASONS** — one constant, honest tier.  It is STRICTLY WEAKER than what
October ships (T2 on every game), so **every number here is a LOWER BOUND**;
the three full-feed seasons (D158/D159) remain the live estimate.  No
played-set oracle is used anywhere.

EVERYTHING THAT TOUCHES A REGISTERED BET SET IS IMPORTED VERBATIM from
`bo_openbacktest.py` (rules, prices, scoring) and `lb_exploit.py` (the D155
matched favourite control).  The ONLY harness change is a FRANCHISE-CODE
CROSSWALK on the join keys (SEA->OKC, NJN->BKN, NOH->NOP): `nba_games` carries
the abbreviation in force that season and both odds tables carry the modern
code, so without it three franchises' entire schedules drop out of the
historical frame.  That is a JOIN FIX, not a rule change; no operator, no
threshold and no price convention is touched.

ARMS (price convention is NEVER mixed inside a cell):
  CLOSE|ML|19   fire on edge-vs-CLOSE, transact at the REAL closing moneyline.
                19 seasons: odds_open.close_ml_* covers 2007-08..2025-26.
  OPEN|SP|19    fire on edge-vs-OPEN, transact at the SP@1.045 convention.
                19 seasons: open_margin covers the whole frame.  CLV is
                p_close_sp - p_open_sp.  THE 19-SEASON CLV ARM.
  OPEN|ML|3     fire on edge-vs-OPEN at the REAL opening moneyline.  Real
                opening MLs exist ONLY in 2023-24..2025-26 — this is D159's
                frame, carried as the level anchor.

WINDOWS.  The rules were SELECTED on 2023-24 + 2024-25, so:
  DEV       2023-24, 2024-25                        (where they were chosen)
  NONDEV    2022-23, 2025-26                        (D115's other half)
  OOS_DEEP  2007-08 .. 2021-22   **15 SEASONS**     (no gate has ever seen
                                                     these; THE REAL EVIDENCE)

K19-ERA CODING (this run's own; `splits.py::ERAS` is flagged too coarse and
D160 §13 left the sub-structure for whoever re-registers):
  K-A 2007-08..2010-11  K-B 2011-12..2013-14  K-C 2014-15..2018-19
  K-D 2019-20..2020-21 (COVID, both D160 separate strata, never pooled into a
  headline)             K-E 2021-22..2025-26

Read-only on data/nba.duckdb.  nbapred/, scripts/bet_engine.py and the frozen
registry are UNTOUCHED.  No production default changed, no gate re-run, the
eval corpus is not widened.

  python3 scripts/k19_rules.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import warnings
from math import comb, sqrt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import nbapred.threads                                          # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

import ba_intersection as bi                                    # noqa: E402
import bo_openbacktest as bo                                    # noqa: E402
import lb_exploit as lx                                         # noqa: E402
from lb_longshot import cluster_boot, cluster_mean_t, icc_oneway  # noqa: E402

# --- PATH OVERRIDES ONLY (added for the D173 re-run on the D170/D171 backfilled
# --- data).  Defaults are byte-identical to D161's; no rule, threshold, price
# --- convention or statistic is touched.
K19 = os.environ.get("K19_PERGAME") or os.path.join(ROOT, "data",
                                                    "k19_pergame.csv")
_TAG = os.environ.get("K19_RULES_TAG", "")
LB_EXPLOIT = os.path.join(ROOT, "data", "lb_exploit.json")
OUT = os.path.join(ROOT, "data", f"k19_rules{_TAG}.json")

SEED = 20260803
N_BOOT = 4000
N_PERM = 400
RULES = ["R4_LOWT", "T20_D03_10_W", "T20_D03_10", "STAR_FAV_SHARPER", "UNION"]

SEASONS19 = ["%d-%02d" % (y, (y + 1) % 100) for y in range(2007, 2026)]
DEV = {"2023-24", "2024-25"}
NONDEV = {"2022-23", "2025-26"}
OOS_DEEP = set(SEASONS19[:15])                 # 2007-08..2021-22
COVID = {"2019-20", "2020-21"}

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

WINDOWS = [("POOL19", set(SEASONS19)),
           ("OOS_DEEP", OOS_DEEP),
           ("DEV", DEV),
           ("NONDEV", NONDEV),
           ("NOCOVID", set(SEASONS19) - COVID)]
# OWNER-SCOPED ADDITIONAL WINDOW (D173), OFF BY DEFAULT so D161's five windows
# reproduce byte-identically when unset.  2018-19..2025-26 is the era in which
# injury reports exist (they begin 2018-12-17) — a frame the owner named in
# advance, NOT one chosen by looking at results.
REPORT8 = set(SEASONS19[11:])
if os.environ.get("K19_REPORT_ERA"):
    WINDOWS.append(("REPORT8", REPORT8))

TQ = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
      14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
      20: 2.086}


# ---------------------------------------------------------------- crosswalk --
FRANCHISE = {"SEA": "OKC", "NJN": "BKN", "NOH": "NOP", "NOK": "NOP",
             "VAN": "MEM", "CHH": "CHA"}


def _patch_joins():
    """`bo.build` merges `team_schedule` (nba_games abbrevs) and
    `star_out_map` (nba_games abbrevs) onto the model frame's home/away, which
    k19_model.py emits as MODERN franchise codes.  Normalise the DB side so
    the join is 1:1 on all 19 seasons.  Nothing about the rules changes."""
    _ts, _so = bi.team_schedule, bi.star_out_map

    def ts(con):
        d = _ts(con)
        d["team"] = d.team.map(lambda t: FRANCHISE.get(t, t))
        return d

    def so(con):
        d = _so(con)
        d["team"] = d.team.map(lambda t: FRANCHISE.get(t, t))
        return d

    bo.team_schedule = ts
    bo.star_out_map = so


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **kw)
    return r, buf.getvalue()


# --------------------------------------------------------------- inference ---
def clustered(vals, clus, seed=SEED):
    vals = np.asarray(vals, float)
    clus = np.asarray(clus)
    if len(vals) < 2 or len(np.unique(clus)) < 2:
        return dict(n=int(len(vals)),
                    mean=float(vals.mean()) if len(vals) else float("nan"),
                    lo=float("nan"), hi=float("nan"), se=float("nan"),
                    tlo=float("nan"), thi=float("nan"),
                    K=int(len(np.unique(clus))), icc=float("nan"),
                    deff=float("nan"), iid_se=float("nan"), sig=False)
    lo, hi, se = cluster_boot(vals, clus, n_boot=N_BOOT, seed=seed)
    tlo, thi, K = cluster_mean_t(vals, clus)
    icc, deff = icc_oneway(vals, clus)
    return dict(n=int(len(vals)), mean=float(vals.mean()), lo=lo, hi=hi, se=se,
                tlo=float(tlo), thi=float(thi), K=int(K), icc=float(icc),
                deff=float(deff),
                iid_se=float(vals.std(ddof=1) / sqrt(len(vals))),
                sig=bool(np.isfinite(tlo) and (tlo > 0 or thi < 0)))


def sig_tag(d):
    if not np.isfinite(d.get("tlo", np.nan)):
        return "  . "
    return " SIG" if (d["tlo"] > 0 or d["thi"] < 0) else "  ns"


# ------------------------------------------------------------------ frames ---
def build_arm(m, when, src):
    """(masks incl. UNION, p_side, dec, ok, clv, clv_ok) for one price arm."""
    p_side, dec, ok = bo.price_cols(m, when, src)
    masks, edge, same = bo.registry_masks(m, p_side, when)
    u = np.zeros(len(m), bool)
    for v in masks.values():
        u |= np.asarray(v)
    masks = {k: np.asarray(v) for k, v in masks.items()}
    masks["UNION"] = u
    # CLV is always measured in the arm's OWN price convention: never mixed.
    po, _, oko = bo.price_cols(m, "open", src)
    pc, _, okc = bo.price_cols(m, "close", src)
    return masks, p_side, dec, ok, (pc - po), (oko & okc)


def score_cell(m, sel, dec, clv, clv_ok, seed=SEED):
    """n / hit / breakeven / ROI / CLV, each season-clustered with the K-1
    cluster-mean t interval as the shipping statistic (GATE_POLICY_V2 §9.1)."""
    out = {"n": int(sel.sum())}
    if sel.sum() < 5:
        return out
    hit = m.hit.values[sel].astype(float)
    d = dec[sel]
    pnl = np.where(hit > 0, d - 1.0, -1.0)
    be = 1.0 / d
    clus = m.season.values[sel]
    out["hit"] = float(hit.mean())
    out["breakeven"] = float(be.mean())
    out["edge_pp"] = float(hit.mean() - be.mean())
    out["mean_dec"] = float(d.mean())
    out["mean_implied"] = float(be.mean())
    out["roi"] = clustered(pnl, clus, seed)
    # ROI net of the per-bet breakeven — the "does it clear the vig" endpoint
    out["roi_excess"] = clustered(pnl - (0.0), clus, seed)   # pnl is already EV0
    s2 = sel & clv_ok
    if s2.sum() >= 5:
        out["clv"] = clustered(clv[s2], m.season.values[s2], seed + 3)
        out["clv_n"] = int(s2.sum())
    return out


def hdr_line():
    return (f"    {'rule':<18}{'n':>6}{'hit%':>7}{'be%':>7}{'edge':>7}"
            f"{'ROI%':>8}{'  [K-1 t CI]':>22}{'sig':>5}"
            f"{'n_clv':>7}{'CLV':>10}{'  [K-1 t CI]':>24}{'sig':>5}{'K_clv':>6}")


def fmt_cell(rule, c):
    if not c.get("n") or "roi" not in c:
        return f"    {rule:<18}{c.get('n', 0):>6}   (too few)"
    r = c["roi"]
    s = (f"    {rule:<18}{c['n']:>6}{100*c['hit']:>7.2f}"
         f"{100*c['breakeven']:>7.2f}{100*c['edge_pp']:>+7.2f}"
         f"{100*r['mean']:>+8.2f}"
         f"  [{100*r['tlo']:>+7.2f},{100*r['thi']:>+7.2f}]{sig_tag(r):>5}")
    if "clv" in c:
        v = c["clv"]
        s += (f"{c['clv_n']:>7}{v['mean']:>+10.5f}"
              f"  [{v['tlo']:>+9.5f},{v['thi']:>+9.5f}]{sig_tag(v):>5}"
              f"{v['K']:>4}")
    else:
        s += f"{'-':>7}{'-':>10}{'-':>24}{'-':>5}{'-':>4}"
    return s


# ==================================================================== main ===
def main():
    res = {"design": "K19 — four frozen rules on 19 contiguous seasons, "
                     "availability-BLIND model (LOWER BOUND)",
           "md5": {os.path.basename(p): md5(p) for p in
                   (K19, os.path.join(ROOT, "data", "derived", "odds_open.csv"),
                    os.path.join(ROOT, "data", "capstone_pergame.csv"))},
           "era_coding": K19_ERA, "windows": {w: sorted(s) for w, s in WINDOWS}}

    print("=" * 132)
    print("K19-RULES — THE FOUR FROZEN RULES ON 19 CONTIGUOUS SEASONS "
          "(2007-08..2025-26), AVAILABILITY-BLIND MODEL")
    print("  EVERY NUMBER IS A LOWER BOUND: the model runs BLIND on all 19 "
          "seasons (constant tier), which is strictly weaker than the T2")
    print("  production ships.  The 3 full-feed seasons (D158/D159) remain the "
          "live estimate.  No played-set oracle anywhere.")
    print("=" * 132)
    for k, v in res["md5"].items():
        print(f"    {v}  {k}")

    _patch_joins()
    m, txt = quiet(bo.build, K19, "p_us", "K19-BLIND", {})
    print(f"\n[0] FRAME: n={len(m)} rows with BOTH an open and a close price; "
          f"seasons={len(m.season.unique())}")
    cov = m.groupby("season").apply(lambda g: pd.Series({
        "n": len(g),
        "close_ML": int(np.isfinite(bo.am2dec(g.close_ml_home)).sum()),
        "open_ML": int(np.isfinite(bo.am2dec(g.open_ml_home)).sum()),
        "star_flag": int(g.fav_star_out_close.sum()),
        "era": K19_ERA.get(g.name, "?"),
    }), include_groups=False)
    print(cov.to_string())
    res["frame_coverage"] = cov.reset_index().to_dict("records")
    print("\n    ERA-AVAILABILITY (GATE_POLICY_V2 §10.2) — STATED BEFORE ANY "
          "RESULT:")
    print("      * real CLOSING moneylines exist in ALL 19 seasons.")
    print("      * real OPENING moneylines exist ONLY in 2023-24..2025-26; the "
          "19-season open arm therefore uses the SP@1.045 convention.")
    print("      * **STAR_FAV_SHARPER's trigger is `game_inactives`, which "
          "STARTS 2022-23.**  The rule is STRUCTURALLY INERT on 15 of the 19")
    print("        seasons — it cannot fire, so its OOS block is UNTESTABLE, "
          "not failed (the D110 §1a cold-estimator error, generalised).")

    frames = {}
    for lab, when, src, seasons in (
            ("CLOSE|ML|19", "close", "ML", set(SEASONS19)),
            ("OPEN|SP|19", "open", "SP", set(SEASONS19)),
            ("CLOSE|SP|19", "close", "SP", set(SEASONS19)),
            ("OPEN|ML|3", "open", "ML", {"2023-24", "2024-25", "2025-26"})):
        masks, ps, dec, ok, clv, clvok = build_arm(m, when, src)
        keep = m.season.isin(seasons).values
        frames[lab] = dict(masks=masks, p_side=ps, dec=dec, ok=ok & keep,
                           clv=clv, clv_ok=clvok, when=when, src=src)

    # =============================================== THE HEADLINE TABLES =====
    print(f"\n{'='*132}\n[1] THE HEADLINE — FOUR FROZEN RULES + UNION, PER ARM, "
          f"PER WINDOW.  Shipping statistic = the K-1 season-cluster-mean t "
          f"interval.\n{'='*132}")
    cells = []
    table = {}
    for lab, F in frames.items():
        print(f"\n  {'-'*128}\n  ARM {lab}   (fire on edge-vs-{F['when'].upper()}"
              f", transact at {F['src']}; CLV in the {F['src']} convention)")
        table[lab] = {}
        for wname, wset in WINDOWS:
            wsel = m.season.isin(wset).values
            if (F["ok"] & wsel).sum() < 50:
                continue
            print(f"\n   window {wname}  ({len(sorted(set(m.season[F['ok'] & wsel])))} seasons)")
            print(hdr_line())
            table[lab][wname] = {}
            for r in RULES:
                sel = F["masks"][r] & F["ok"] & wsel
                c = score_cell(m, sel, F["dec"], F["clv"], F["clv_ok"])
                table[lab][wname][r] = c
                print(fmt_cell(r, c))
                if c.get("n", 0) >= 5 and "roi" in c:
                    cells.append({"arm": lab, "window": wname, "rule": r,
                                  "n": c["n"], "roi": c["roi"]["mean"],
                                  "roi_sig": c["roi"]["sig"],
                                  "clv": c.get("clv", {}).get("mean"),
                                  "clv_sig": c.get("clv", {}).get("sig")})
    res["table"] = table
    res["cells"] = cells

    # ====================================================== ERA CONDITIONALITY
    print(f"\n{'='*132}\n[2] ERA CONDITIONALITY — is the edge era-stable or "
          f"concentrated?  A rule that only works post-2015 is a different "
          f"product.\n{'='*132}")
    era_out = {}
    for lab in ("CLOSE|ML|19", "OPEN|SP|19"):
        F = frames[lab]
        print(f"\n  ARM {lab}")
        print(f"    {'rule':<18}{'era':<6}{'seasons':<10}{'n':>6}{'hit%':>7}"
              f"{'be%':>7}{'ROI%':>8}{'  [K-1 t]':>20}{'sig':>5}"
              f"{'CLV':>10}{'  [K-1 t]':>22}{'sig':>5}")
        era_out[lab] = {}
        for r in RULES:
            era_out[lab][r] = {}
            for e in ERA_ORDER:
                wset = {s for s, c in K19_ERA.items() if c == e}
                sel = F["masks"][r] & F["ok"] & m.season.isin(wset).values
                c = score_cell(m, sel, F["dec"], F["clv"], F["clv_ok"])
                era_out[lab][r][e] = c
                if not c.get("n") or "roi" not in c:
                    print(f"    {r:<18}{e:<6}{len(wset):<10}{c.get('n',0):>6}"
                          f"   (inert / too few)")
                    continue
                v = c["roi"]
                cl = c.get("clv", {})
                print(f"    {r:<18}{e:<6}{len(wset):<10}{c['n']:>6}"
                      f"{100*c['hit']:>7.2f}{100*c['breakeven']:>7.2f}"
                      f"{100*v['mean']:>+8.2f}"
                      f"  [{100*v['tlo']:>+6.2f},{100*v['thi']:>+6.2f}]"
                      f"{sig_tag(v):>5}"
                      + (f"{cl['mean']:>+10.5f}  [{cl['tlo']:>+8.5f},"
                         f"{cl['thi']:>+8.5f}]{sig_tag(cl):>5}"
                         if cl else f"{'-':>10}{'-':>22}{'-':>5}"))
            print()
    res["era"] = era_out

    # DerSimonian-Laird on the era means (I^2 / Q), union only
    print("    ERA HETEROGENEITY (DerSimonian-Laird on the era means, UNION):")
    het = {}
    for lab in ("CLOSE|ML|19", "OPEN|SP|19"):
        for metric in ("roi", "clv"):
            pts, ses, labs = [], [], []
            for e in ERA_ORDER:
                c = era_out[lab]["UNION"].get(e, {})
                if metric not in c or not np.isfinite(c[metric].get("se", np.nan)):
                    continue
                se = c[metric]["se"]
                if se <= 0:
                    continue
                pts.append(c[metric]["mean"])
                ses.append(se)
                labs.append(e)
            if len(pts) < 3:
                continue
            pts, ses = np.array(pts), np.array(ses)
            w = 1 / ses ** 2
            mu = (w * pts).sum() / w.sum()
            Q = (w * (pts - mu) ** 2).sum()
            dfq = len(pts) - 1
            I2 = max(0.0, 100 * (Q - dfq) / Q) if Q > 0 else 0.0
            from statistics import NormalDist
            # chi2 upper tail via Wilson-Hilferty
            z = ((Q / dfq) ** (1 / 3) - (1 - 2 / (9 * dfq))) / sqrt(2 / (9 * dfq))
            p = 1 - NormalDist().cdf(z)
            verdict = ("ERA-STABLE" if I2 < 50 and p > 0.10 else
                       "ERA-CONDITIONAL" if min(pts) * max(pts) > 0 else
                       "ERA-SPECIFIC (sign flips)")
            het[f"{lab}|{metric}"] = {"eras": labs, "pts": pts.tolist(),
                                      "Q": float(Q), "df": dfq, "I2": float(I2),
                                      "p": float(p), "verdict": verdict}
            print(f"      {lab:<13}{metric:<5} Q={Q:7.2f} df={dfq} "
                  f"I2={I2:5.1f}% p={p:.3f}  -> {verdict}   "
                  f"({', '.join(f'{a}={b:+.4f}' for a, b in zip(labs, pts))})")
    res["era_heterogeneity"] = het

    # ================================================== PER-SEASON UNION =====
    print(f"\n{'='*132}\n[3] PER-SEASON UNION — the 19 cluster means the K-1 t "
          f"interval is built from\n{'='*132}")
    ps_out = {}
    for lab in ("CLOSE|ML|19", "OPEN|SP|19"):
        F = frames[lab]
        sel = F["masks"]["UNION"] & F["ok"]
        d = pd.DataFrame({
            "season": m.season.values[sel],
            "pnl": np.where(m.hit.values[sel] > 0, F["dec"][sel] - 1.0, -1.0)})
        g = d.groupby("season").pnl.agg(["size", "mean"])
        s2 = sel & F["clv_ok"]
        d2 = pd.DataFrame({"season": m.season.values[s2],
                           "clv": F["clv"][s2]})
        g2 = d2.groupby("season").clv.agg(["size", "mean"])
        print(f"\n  ARM {lab}")
        print(f"    {'season':<10}{'era':<6}{'block':<10}{'n':>6}{'ROI%':>9}"
              f"{'n_clv':>7}{'CLV':>11}")
        rowsx = []
        for s in SEASONS19:
            if s not in g.index:
                continue
            blk = ("DEV" if s in DEV else "NONDEV" if s in NONDEV
                   else "OOS_DEEP")
            cl = g2.loc[s, "mean"] if s in g2.index else float("nan")
            nc = int(g2.loc[s, "size"]) if s in g2.index else 0
            print(f"    {s:<10}{K19_ERA[s]:<6}{blk:<10}{int(g.loc[s,'size']):>6}"
                  f"{100*g.loc[s,'mean']:>+9.2f}{nc:>7}{cl:>+11.5f}")
            rowsx.append({"season": s, "era": K19_ERA[s], "block": blk,
                          "n": int(g.loc[s, "size"]),
                          "roi": float(g.loc[s, "mean"]),
                          "n_clv": nc, "clv": float(cl)})
        ps_out[lab] = rowsx
    res["per_season_union"] = ps_out

    # ================================================ MATCHED CONTROL ========
    print(f"\n{'='*132}\n[4] THE D155 MATCHED FAVOURITE CONTROL ON THE FULL "
          f"FRAME.  Control = bet the MARKET favourite from the same\n"
          f"    (season x implied-probability) strata as the rules' own bets, "
          f"at the real moneyline.  D155 open UNION +8.22% SIG at K=3;\n"
          f"    D159 honest open UNION +4.93% ns at K=3.  **At K=19 (18 dof) "
          f"it is testable for the first time.**\n{'='*132}")
    lbj = json.load(open(LB_EXPLOIT))
    sub = {"longrun_fav_roi": lbj["longrun_fav_roi"]}
    for when in ("close", "open"):
        print(f"\n  >>> @{when}")
        _, t = quiet(lx.matched_control, m, bo, when, sub, f"matched|{when}")
        print(t)
    res["matched_control"] = {k: v for k, v in sub.items()
                              if k.startswith("matched|")}
    print(f"\n    ALPHA SUMMARY — note `lb_exploit.matched_control`'s t-quantile "
          f"table falls back to 2.1 above K-1=4, which at K-1=18 is 2.101 to "
          f"3 dp: EXACT.")
    print(f"    {'when':<7}{'set':<18}{'n':>6}{'K':>4}{'ruleROI':>10}"
          f"{'ctrl':>9}{'alpha':>9}{'  [K-1 t CI]':>24}{'sig':>5}")
    for when in ("close", "open"):
        for r in RULES:
            v = sub[f"matched|{when}"].get(r)
            if not v:
                continue
            sg = " SIG" if (np.isfinite(v["alpha_tlo"])
                            and v["alpha_tlo"] > 0) else "  ns"
            print(f"    {when:<7}{r:<18}{v['n']:>6}{v['K']:>4}"
                  f"{100*v['rule_roi']:>+10.2f}"
                  f"{100*v['ctrl_contemporaneous']:>+9.2f}"
                  f"{100*v['alpha_vs_contemp']:>+9.2f}"
                  f"  [{100*v['alpha_tlo']:>+9.2f},{100*v['alpha_thi']:>+9.2f}]"
                  f"{sg:>5}")

    # ========================================== WITHIN-DATE PERMUTATION ======
    print(f"\n{'='*132}\n[5] WITHIN-DATE PERMUTATION PLACEBO (D147's method) — "
          f"p_us permuted ACROSS GAMES ON THE SAME DATE, so the slate, the "
          f"selection\n    mechanism and the price distribution all survive but "
          f"the model's information is destroyed.  Rules re-fire on the "
          f"placebo.\n    If the placebo earns what we earn, we are harvesting "
          f"price noise, not information.  {N_PERM} draws, seed {SEED}.\n"
          f"{'='*132}")
    plac = {}
    dates = m.game_date.values
    uniq_d = {d: np.where(dates == d)[0] for d in np.unique(dates)}
    for lab in ("CLOSE|ML|19", "OPEN|SP|19"):
        F = frames[lab]
        when, src = F["when"], F["src"]
        real = {}
        for r in RULES:
            sel = F["masks"][r] & F["ok"]
            pnl = np.where(m.hit.values[sel] > 0, F["dec"][sel] - 1.0, -1.0)
            s2 = sel & F["clv_ok"]
            real[r] = {"n": int(sel.sum()), "roi": float(pnl.mean()),
                       "clv": float(F["clv"][s2].mean()) if s2.sum() else np.nan}
        rng = np.random.default_rng(SEED)
        draws = {r: {"roi": [], "clv": [], "n": []} for r in RULES}
        base = m.copy()
        for _ in range(N_PERM):
            pp = m.p_us.values.copy()
            for ix in uniq_d.values():
                if len(ix) > 1:
                    pp[ix] = pp[rng.permutation(ix)]
            base["p_us"] = pp
            base["pick_home"] = pp > 0.5
            base["p_us_side"] = np.where(pp > 0.5, pp, 1 - pp)
            base["conf_us"] = np.abs(pp - 0.5)
            base["hit"] = np.where(pp > 0.5, m.y.values == 1,
                                   m.y.values == 0).astype(int)
            mk, ps2, dc2, ok2, clv2, cok2 = build_arm(base, when, src)
            ok2 = ok2 & F["ok"]           # same season restriction
            for r in RULES:
                s = mk[r] & ok2
                if s.sum() < 5:
                    continue
                pnl = np.where(base.hit.values[s] > 0, dc2[s] - 1.0, -1.0)
                draws[r]["roi"].append(float(pnl.mean()))
                draws[r]["n"].append(int(s.sum()))
                s2 = s & cok2
                if s2.sum() >= 5:
                    draws[r]["clv"].append(float(clv2[s2].mean()))
        print(f"\n  ARM {lab}")
        print(f"    {'rule':<18}{'n_real':>7}{'ROI real':>10}"
              f"{'ROI placebo(sd)':>20}{'p':>7}   "
              f"{'CLV real':>10}{'CLV placebo(sd)':>20}{'p':>7}")
        plac[lab] = {}
        for r in RULES:
            dr = np.array(draws[r]["roi"])
            dc = np.array(draws[r]["clv"])
            p_roi = float((dr >= real[r]["roi"]).mean()) if len(dr) else np.nan
            p_clv = float((dc >= real[r]["clv"]).mean()) if len(dc) else np.nan
            plac[lab][r] = {"real_n": real[r]["n"], "real_roi": real[r]["roi"],
                            "real_clv": real[r]["clv"],
                            "placebo_roi_mean": float(dr.mean()) if len(dr) else None,
                            "placebo_roi_sd": float(dr.std()) if len(dr) else None,
                            "placebo_clv_mean": float(dc.mean()) if len(dc) else None,
                            "placebo_clv_sd": float(dc.std()) if len(dc) else None,
                            "p_roi": p_roi, "p_clv": p_clv,
                            "placebo_mean_n": float(np.mean(draws[r]["n"]))
                            if draws[r]["n"] else None}
            print(f"    {r:<18}{real[r]['n']:>7}{100*real[r]['roi']:>+10.2f}"
                  f"{100*dr.mean():>+13.2f}({100*dr.std():>4.2f})"
                  f"{p_roi:>7.3f}   {real[r]['clv']:>+10.5f}"
                  f"{dc.mean():>+13.5f}({dc.std():>5.5f})"
                  f"{p_clv:>7.3f}" if len(dc) else
                  f"    {r:<18}{real[r]['n']:>7}{100*real[r]['roi']:>+10.2f}"
                  f"{100*dr.mean():>+13.2f}({100*dr.std():>4.2f}){p_roi:>7.3f}")
    res["placebo_withindate"] = plac

    # ================================================== FAMILY-WISE ==========
    print(f"\n{'='*132}\n[6] FAMILY-WISE ARITHMETIC (D121's 9-vs-14.4 standard)"
          f"\n{'='*132}")
    fw = {}
    for scope, keep in (("PRE-SPECIFIED (4 arms x 5 windows x 5 sets, ROI)",
                         lambda c: True),):
        k = len(cells)
        sig_roi = [c for c in cells if c["roi_sig"] and c["roi"] > 0]
        sig_roi_neg = [c for c in cells if c["roi_sig"] and c["roi"] < 0]
        sig_clv = [c for c in cells if c.get("clv_sig") and (c.get("clv") or 0) > 0]
        n_clv = len([c for c in cells if c.get("clv") is not None])
        p_at = (1.0 - sum(comb(k, i) * 0.05 ** i * 0.95 ** (k - i)
                          for i in range(len(sig_roi)))) if k else float("nan")
        print(f"\n  -- {scope}")
        print(f"     ROI cells = {k}; expected significant at 95% under a "
              f"global null = {0.05*k:.1f}")
        print(f"     OBSERVED significant POSITIVE ROI = {len(sig_roi)}   "
              f"-> P(chance >= {len(sig_roi)}) = {p_at:.4f}")
        print(f"     OBSERVED significant NEGATIVE ROI = {len(sig_roi_neg)}")
        print(f"     CLV cells = {n_clv}; expected {0.05*n_clv:.1f}; "
              f"OBSERVED significant POSITIVE CLV = {len(sig_clv)}")
        for c in sig_roi:
            print(f"       SIG+ROI: {c['arm']:<12}{c['window']:<10}"
                  f"{c['rule']:<18}n={c['n']:<6}ROI {100*c['roi']:+.2f}%")
        for c in sig_roi_neg:
            print(f"       SIG-ROI: {c['arm']:<12}{c['window']:<10}"
                  f"{c['rule']:<18}n={c['n']:<6}ROI {100*c['roi']:+.2f}%")
        fw[scope] = {"cells": k, "expected": 0.05 * k,
                     "observed_pos": len(sig_roi),
                     "observed_neg": len(sig_roi_neg),
                     "clv_cells": n_clv, "clv_observed_pos": len(sig_clv),
                     "p_atleast": p_at}
    print("\n     Cells OVERLAP heavily (the same bets re-scored across arms "
          "and windows), so these are UPPER BOUNDS on the surprise.")
    res["familywise"] = fw

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\nwrote {OUT}")
    return res


if __name__ == "__main__":
    main()
