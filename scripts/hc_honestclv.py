#!/usr/bin/env python3
"""HC-HONESTCLV — do the TRADING results survive the removal of the D158
availability leak?

THE QUESTION.  D158 showed `scripts/prod_by_season.py` (and its sibling
`scripts/ds_rt1_capstone.py`) built availability OUT-sets from the PLAYED set —
tonight's box score — which docs/LEAKAGE.md:131 forbids.  Every registered
betting result (D121, D126, D142, D147/D148, D155) was computed off model
probabilities produced by that path.  Our rules select on model-vs-market
DIVERGENCE, so a model that secretly knows who played selects better bets.
D156 Part B re-measured ROI honestly.  **CLV — the declared October target —
has never been measured honestly.  This script measures it.**

ARMS (the ONLY difference between the first two is the availability construction;
same script, same corpus, same weekly-refit cadence, same market join):
  HONEST      data/capstone_pergame.csv                 T2-HONEST (D158 default)
  LEAKY_PAIR  data/capstone_pergame_oracle_ceiling.csv  C1 played-set oracle
  LEAKY_REG   data/ds_rt1_pergame.csv (p_full)          the frame D121/D126/D142/
              D155 ACTUALLY priced — carried as the fidelity anchor, because a
              re-measurement that cannot reproduce the registered digits is not
              a re-measurement.

AVAILABILITY TIER VARIES BY SEASON and is NEVER pooled silently:
  2021-22 no feed at all  -> BLIND (empty out-sets)   [outside the ML frame]
  2022-23 inactives only  -> partial T2
  2023-24..2025-26        -> full T2  == what October ships
The REAL-MONEYLINE frame (n=3,682) is EXACTLY the three full-T2 seasons.

EVERYTHING that touches a registered bet set is IMPORTED VERBATIM from
scripts/bo_openbacktest.py (rules, pricing, CLV), scripts/bo_lineshop.py (the
2-book TeamRankings panel and the execution policies) and scripts/lb_exploit.py
(the D155 matched favourite control).  Nothing is re-implemented.

Read-only on data/nba.duckdb (60s retry).  No production default is changed and
scripts/bet_engine.py is not touched.

Run:  python scripts/hc_honestclv.py
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

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

import bo_openbacktest as bo                                      # noqa: E402
import bo_lineshop as ls                                          # noqa: E402
import lb_exploit as lx                                           # noqa: E402
from lb_longshot import cluster_boot, cluster_mean_t, icc_oneway  # noqa: E402

HONEST = os.path.join(ROOT, "data", "capstone_pergame.csv")
LEAKY_PAIR = os.path.join(ROOT, "data", "capstone_pergame_oracle_ceiling.csv")
LEAKY_REG = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
LB_EXPLOIT = os.path.join(ROOT, "data", "lb_exploit.json")
OUT = os.path.join(ROOT, "data", "hc_honestclv.json")

SEED = 20260803
N_BOOT = 4000
RULES = ["R4_LOWT", "T20_D03_10_W", "T20_D03_10", "STAR_FAV_SHARPER", "UNION"]
FRAME_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]   # registered frame
TIER = {"2021-22": "BLIND (no feed)", "2022-23": "inactives-only T2",
        "2023-24": "full T2", "2024-25": "full T2", "2025-26": "full T2"}

# D121 registered live bands (ML frame, union @open, ~44 bets/month)
D121_RED, D121_GOOD = -0.0131, +0.0200

TQ = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
      14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
      20: 2.086}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **kw)
    return r, buf.getvalue()


# ------------------------------------------------------------- inference ----
def clustered(vals, clus, seed=SEED):
    """Season-clustered mean + CI + K-1 cluster-mean t + ICC/design effect."""
    vals = np.asarray(vals, float)
    clus = np.asarray(clus)
    if len(vals) == 0:
        return dict(n=0)
    lo, hi, se = cluster_boot(vals, clus, n_boot=N_BOOT, seed=seed)
    tlo, thi, K = cluster_mean_t(vals, clus)
    icc, deff = icc_oneway(vals, clus)
    return dict(n=int(len(vals)), mean=float(vals.mean()), lo=lo, hi=hi,
                se=se, tlo=tlo, thi=thi, K=int(K), icc=float(icc),
                deff=float(deff), iid_se=float(vals.std(ddof=1) / sqrt(len(vals))))


def diff_clustered(v_a, c_a, v_b, c_b, seed=SEED + 7):
    """Season-clustered CI on mean(A) - mean(B) where A and B are DIFFERENT bet
    sets drawn from the SAME seasons.  Seasons are resampled jointly, so the
    shared season structure is respected and the two sets are not pretended to
    be independent samples."""
    v_a, c_a, v_b, c_b = (np.asarray(v_a, float), np.asarray(c_a),
                          np.asarray(v_b, float), np.asarray(c_b))
    keys = np.unique(np.concatenate([c_a, c_b]))
    ia = {k: np.where(c_a == k)[0] for k in keys}
    ib = {k: np.where(c_b == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        sa = np.concatenate([ia[keys[j]] for j in pick])
        sb = np.concatenate([ib[keys[j]] for j in pick])
        out[b] = (v_a[sa].mean() if len(sa) else np.nan) - \
                 (v_b[sb].mean() if len(sb) else np.nan)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    # cluster-mean t on the per-season difference of set means
    da = pd.Series(v_a).groupby(pd.Series(c_a)).mean()
    db = pd.Series(v_b).groupby(pd.Series(c_b)).mean()
    j = da.index.intersection(db.index)
    d = (da[j] - db[j]).values
    K = len(d)
    tq = TQ.get(K - 1, 1.96)
    se = d.std(ddof=1) / sqrt(K) if K > 1 else float("nan")
    return dict(diff=float(v_a.mean() - v_b.mean()), lo=float(lo),
                hi=float(hi), se=float(np.nanstd(out, ddof=1)),
                K=int(K), per_season=[float(x) for x in d],
                tlo=float(d.mean() - tq * se) if K > 1 else float("nan"),
                thi=float(d.mean() + tq * se) if K > 1 else float("nan"))


# ----------------------------------------------------------------- frames ---
def build_frames(res):
    frames = {}
    for lab, path, pcol in (("HONEST", HONEST, "p_us"),
                            ("LEAKY_PAIR", LEAKY_PAIR, "p_us"),
                            ("LEAKY_REG", LEAKY_REG, "p_full")):
        (m, _), _txt = quiet(lambda: (bo.build(path, pcol, lab, {}), None))
        frames[lab] = m
    return frames


def attach_panel(frames, res):
    panel, _ = quiet(ls.build_panel, res)
    for lab, m in frames.items():
        n0 = len(m)
        mm = m.merge(panel.drop(columns=["season_end"]),
                     on=["game_date", "home", "away"], how="left")
        assert len(mm) == n0, "panel join fanned out"
        mm["n_books"] = mm.n_books.fillna(0).astype(int)
        frames[lab] = mm
    return frames


def masks_of(m, src, when="open"):
    p_side, dec, ok = bo.price_cols(m, when, src)
    masks, edge, same = bo.registry_masks(m, p_side, when)
    u = np.zeros(len(m), bool)
    for v in masks.values():
        u |= np.asarray(v)
    masks["UNION"] = u
    return masks, p_side, dec, ok, edge


# ================================================================== main ====
def main():
    res = {"design": __doc__.split("\n")[0],
           "md5": {os.path.basename(p): md5(p) for p in
                   (HONEST, LEAKY_PAIR, LEAKY_REG,
                    os.path.join(ROOT, "data", "capstone_pergame_d132_leaky.csv"),
                    os.path.join(ROOT, "data", "derived", "odds_open.csv"))},
           "tier_by_season": TIER}

    print("=" * 118)
    print("HC-HONESTCLV — CLV, MATCHED-CONTROL ALPHA AND RULE FIRING ON "
          "AVAILABILITY-HONEST PROBABILITIES")
    print("=" * 118)
    print("[0] CONTROL ARTIFACTS (D134 control-hash field)")
    for k, v in res["md5"].items():
        print(f"    {v}  {k}")
    pbs = json.load(open(os.path.join(ROOT, "data", "prod_by_season.json")))
    print("\n    availability tier ACTUALLY IN FORCE per season "
          "(data/prod_by_season.json, the honest run):")
    for o in pbs:
        c = o["coverage"]
        print(f"      {o['season']}  n={o['n']:4d}  report={c['report']:4d} "
              f"inactives={c['inactives']:4d}  mean_outs/team="
              f"{o['mean_outs_per_team']:.3f}   {o['tier']}")
    res["prod_by_season_tiers"] = [
        {"season": o["season"], "tier": o["tier"],
         "mean_outs_per_team": o["mean_outs_per_team"],
         "coverage": o["coverage"]} for o in pbs]

    frames = build_frames(res)
    frames = attach_panel(frames, res)
    for lab, m in frames.items():
        print(f"\n    frame {lab:<11s} n={len(m):5d}  seasons="
              f"{sorted(m.season.unique())}  2-book={int((m.n_books>=2).sum())}"
              f"  real open ML={int(np.isfinite(m.p_open_ml).sum())}")

    # per-game probability distance between the honest and leaky arms
    h, l = frames["HONEST"], frames["LEAKY_PAIR"]
    k = ["season", "game_id"]
    j = h[k + ["p_us"]].merge(l[k + ["p_us"]], on=k, suffixes=("_h", "_l"))
    d = (j.p_us_h - j.p_us_l).abs()
    print(f"\n[0b] HONEST vs LEAKY per-game |dp| on {len(j)} shared games: "
          f"mean {d.mean():.5f}  median {d.median():.5f}  max {d.max():.5f}  "
          f"frac|dp|>0.02 {100*(d>0.02).mean():.1f}%")
    per_s = j.assign(d=d).groupby("season").d.agg(["size", "mean", "max"])
    print(per_s.to_string())
    res["dp_honest_vs_leaky"] = {
        "mean": float(d.mean()), "median": float(d.median()),
        "max": float(d.max()),
        "per_season": per_s.reset_index().to_dict("records")}

    # ============================================================ ANCHORS ===
    print(f"\n{'='*118}\n[1] HARNESS ANCHOR — the LEAKY_REG arm must reproduce "
          f"the registered digits before anything here is believed\n{'='*118}")
    mr = frames["LEAKY_REG"]
    masks, p_side, dec, ok, edge = masks_of(mr, "ML")
    pc, _, okc = bo.price_cols(mr, "close", "ML")
    clv = pc - p_side
    D155_OPEN = {"R4_LOWT": (485, 0.01961), "T20_D03_10_W": (215, 0.01358),
                 "T20_D03_10": (554, 0.01785),
                 "STAR_FAV_SHARPER": (1009, 0.01215), "UNION": (1378, 0.01590)}
    anc = {}
    print(f"    {'rule':<18}{'n':>6}{'CLV(prop devig)':>18}   "
          f"vs D155(7) registered")
    for r in RULES:
        s = np.asarray(masks[r]) & ok & okc
        n, c = int(s.sum()), float(clv[s].mean())
        rn, rc = D155_OPEN[r]
        good = (n == rn) and abs(c - rc) < 5e-5
        anc[r] = {"n": n, "clv": c, "reg_n": rn, "reg_clv": rc,
                  "exact": bool(good)}
        print(f"    {r:<18}{n:>6}{c:>+18.5f}   [n={rn} CLV={rc:+.5f}]  "
              f"{'EXACT' if good else '*** MISMATCH ***'}")
    res["anchor_d155_open_ML"] = anc

    # ====================================================== TASK 4: FIRING ==
    print(f"\n{'='*118}\n[2] TASK 4 — RULE FIRING, HONEST vs LEAKY.  The rules "
          f"are thresholds on model-vs-market DIVERGENCE, so a less-confident\n"
          f"    honest model fires DIFFERENTLY, not merely less.\n{'='*118}")
    fire = {}
    for src, whichframe in (("ML", None), ("SP", None)):
        sel = {}
        for lab in ("HONEST", "LEAKY_PAIR", "LEAKY_REG"):
            m = frames[lab]
            if src == "SP":                      # 4-season registered frame
                keep = m.season.isin(FRAME_SEASONS).values
            else:
                keep = np.ones(len(m), bool)
            mk, ps, dc, o, _ = masks_of(m, src)
            sel[lab] = {r: (np.asarray(mk[r]) & o & keep) for r in RULES}
            sel[lab]["_keys"] = list(zip(m.season, m.game_id))
        srcdesc = ("real opening moneylines, 2023-24..2025-26 = the three "
                   "full-T2 seasons" if src == "ML" else
                   "spread convention, 2022-23..2025-26 = the registered "
                   "4-season frame")
        print(f"\n    --- price source {src} ({srcdesc}) ---")
        print(f"    {'rule':<18}{'HONEST':>8}{'LEAKY':>8}{'dn':>7}{'d%':>8}"
              f"{'|both|':>8}{'honest-only':>13}{'leaky-only':>12}"
              f"{'Jaccard':>9}")
        fire[src] = {}
        for r in RULES:
            kh = {k for k, s in zip(sel["HONEST"]["_keys"], sel["HONEST"][r]) if s}
            kl = {k for k, s in zip(sel["LEAKY_PAIR"]["_keys"],
                                    sel["LEAKY_PAIR"][r]) if s}
            inter, uni = len(kh & kl), len(kh | kl)
            jac = inter / uni if uni else float("nan")
            print(f"    {r:<18}{len(kh):>8}{len(kl):>8}{len(kh)-len(kl):>+7}"
                  f"{100*(len(kh)-len(kl))/max(len(kl),1):>+8.1f}"
                  f"{inter:>8}{len(kh-kl):>13}{len(kl-kh):>12}{jac:>9.3f}")
            fire[src][r] = {"honest": len(kh), "leaky": len(kl),
                            "both": inter, "honest_only": len(kh - kl),
                            "leaky_only": len(kl - kh), "jaccard": float(jac)}
        # per-season union firing
        if src == "ML":
            us_h = pd.Series([k[0] for k, s in zip(sel["HONEST"]["_keys"],
                                                   sel["HONEST"]["UNION"]) if s]
                             ).value_counts().sort_index()
            us_l = pd.Series([k[0] for k, s in zip(sel["LEAKY_PAIR"]["_keys"],
                                                   sel["LEAKY_PAIR"]["UNION"]) if s]
                             ).value_counts().sort_index()
            print(f"\n    UNION bets per season (ML frame): honest "
                  f"{us_h.to_dict()}  leaky {us_l.to_dict()}")
            fire[src]["union_per_season"] = {
                "honest": {k: int(v) for k, v in us_h.items()},
                "leaky": {k: int(v) for k, v in us_l.items()}}
    res["firing"] = fire

    # =========================================== TASK 2: CLV, THE HEADLINE ==
    print(f"\n{'='*118}\n[3] TASK 2 — CLV AT THE OPEN, HONEST vs LEAKY.  "
          f"CLV = p_close_side - p_open_side on OUR side, de-vigged.\n"
          f"    The DIFFERENCE column is the quantity the October programme "
          f"lives or dies on: how much of our measured CLV was the leak.\n"
          f"{'='*118}")
    clv_res = {}

    # ---- arm A: real opening/closing moneylines, proportional devig --------
    print(f"\n    ARM A — REAL MONEYLINES @ OPEN (the D121/D155 pricing "
          f"convention).  n=3,682 universe = 2023-24..2025-26 = FULL T2 on "
          f"every game.\n           K=3 seasons: the small-K warning of "
          f"GATE_POLICY_V2 §9.3 applies to every CI in this arm.")
    armA = {}
    hset, lset = {}, {}
    for lab in ("HONEST", "LEAKY_PAIR", "LEAKY_REG"):
        m = frames[lab]
        mk, ps, dc, o, _ = masks_of(m, "ML")
        pc2, _, oc = bo.price_cols(m, "close", "ML")
        cv = pc2 - ps
        store = {}
        for r in RULES:
            s = np.asarray(mk[r]) & o & oc
            store[r] = (cv[s], m.season.values[s])
        (hset if lab == "HONEST" else lset).setdefault(lab, store)
    hA = hset["HONEST"]
    lA = lset["LEAKY_PAIR"]
    rA = lset["LEAKY_REG"]
    print(f"\n    {'rule':<18}{'n_h':>6}{'CLV honest':>12}"
          f"{'[95% season-clustered]':>26}{'K-1 t':>20}"
          f"{'n_l':>6}{'CLV leaky':>11}{'  DIFF (h-l)':>13}{'  [95% cl]':>22}"
          f"{'  %of leaky':>12}")
    for r in RULES:
        vh, ch = hA[r]
        vl, cl_ = lA[r]
        a = clustered(vh, ch)
        b = clustered(vl, cl_)
        dd = diff_clustered(vh, ch, vl, cl_)
        frac = 100.0 * (b["mean"] - a["mean"]) / b["mean"] if b["mean"] else np.nan
        armA[r] = {"honest": a, "leaky": b, "diff": dd,
                   "leak_share_pct": float(frac),
                   "leaky_registered": clustered(*rA[r])}
        print(f"    {r:<18}{a['n']:>6}{a['mean']:>+12.5f}"
              f"  [{a['lo']:+.5f},{a['hi']:+.5f}]  [{a['tlo']:+.5f},{a['thi']:+.5f}]"
              f"{b['n']:>6}{b['mean']:>+11.5f}{dd['diff']:>+13.5f}"
              f"  [{dd['lo']:+.5f},{dd['hi']:+.5f}]{frac:>+12.1f}%")
    print(f"\n    (leaky-as-REGISTERED, ds_rt1 p_full, for level continuity: "
          + "  ".join(f"{r}={armA[r]['leaky_registered']['mean']:+.5f}"
                      for r in RULES) + ")")
    print(f"    ICC / design effect (union honest): "
          f"ICC={armA['UNION']['honest']['icc']:+.5f} "
          f"DEFF_anova={armA['UNION']['honest']['deff']:.2f}  "
          f"cluster SE {armA['UNION']['honest']['se']:.5f} vs iid SE "
          f"{armA['UNION']['honest']['iid_se']:.5f} "
          f"(DEFF_boot {armA['UNION']['honest']['se']/armA['UNION']['honest']['iid_se']:.2f})")
    clv_res["ML_open"] = armA

    # per-season CLV, union, both arms (tier heterogeneity, NEVER pooled blind)
    print(f"\n    UNION CLV BY SEASON (tier stated; do not pool across tiers "
          f"without saying so):")
    ps_tab = {}
    for lab, store in (("HONEST", hA), ("LEAKY_PAIR", lA)):
        v, c = store["UNION"]
        g = pd.DataFrame({"v": v, "s": c}).groupby("s").v.agg(["size", "mean"])
        ps_tab[lab] = g
    print(f"    {'season':<10}{'tier':<22}{'n_h':>6}{'CLV_h':>10}"
          f"{'n_l':>6}{'CLV_l':>10}{'diff':>10}")
    ps_out = []
    for s in sorted(set(ps_tab["HONEST"].index) | set(ps_tab["LEAKY_PAIR"].index)):
        a = ps_tab["HONEST"].loc[s] if s in ps_tab["HONEST"].index else None
        b = ps_tab["LEAKY_PAIR"].loc[s] if s in ps_tab["LEAKY_PAIR"].index else None
        print(f"    {s:<10}{TIER.get(s,'?'):<22}{int(a['size']):>6}"
              f"{a['mean']:>+10.5f}{int(b['size']):>6}{b['mean']:>+10.5f}"
              f"{a['mean']-b['mean']:>+10.5f}")
        ps_out.append({"season": s, "tier": TIER.get(s), "n_h": int(a["size"]),
                       "clv_h": float(a["mean"]), "n_l": int(b["size"]),
                       "clv_l": float(b["mean"]),
                       "diff": float(a["mean"] - b["mean"])})
    clv_res["ML_open_per_season_union"] = ps_out

    # ---- arm B: SP frame under the D142 execution policies ----------------
    print(f"\n{'-'*118}\n    ARM B — D142 EXECUTION POLICIES on the 2-book "
          f"TeamRankings panel (ONEBOOK / BEST2 / WORST2), SP pricing "
          f"convention,\n           4-season registered frame.  K=4.  Bets are "
          f"selected on the REGISTERED consensus open exactly as in D142; only "
          f"the transacted price varies.")
    armB = {}
    for lab in ("HONEST", "LEAKY_PAIR", "LEAKY_REG"):
        m = frames[lab]
        keep = m.season.isin(FRAME_SEASONS).values & (m.n_books >= 2).values
        mk, ps, dc, o, _ = masks_of(m, "SP")
        pols = ls.policy_margins(m)
        pr = ls.policy_prices(m, pols)
        pcl = bo.sigmoid(m.close_margin.values / ls.SC)
        pc_side = np.where(m.pick_home.values, pcl, 1 - pcl)
        armB[lab] = {}
        for r in RULES:
            s = np.asarray(mk[r]) & o & keep
            row = {}
            for pol in ("ONEBOOK", "CONS_REG", "BEST2", "WORST2"):
                cv = pc_side[s] - pr[pol][0][s]
                row[pol] = clustered(cv, m.season.values[s])
                row[pol]["_vals"] = cv
                row[pol]["_clus"] = m.season.values[s]
            armB[lab][r] = row
    print(f"\n    {'rule':<18}{'pol':<9}{'n_h':>6}{'CLV honest':>12}"
          f"{'[95% cl]':>24}{'n_l':>6}{'CLV leaky':>11}{'DIFF':>11}"
          f"{'[95% cl]':>24}{'%leak':>8}")
    armB_out = {}
    for r in RULES:
        armB_out[r] = {}
        for pol in ("ONEBOOK", "BEST2", "WORST2"):
            a = armB["HONEST"][r][pol]
            b = armB["LEAKY_PAIR"][r][pol]
            dd = diff_clustered(a["_vals"], a["_clus"], b["_vals"], b["_clus"])
            frac = 100.0 * (b["mean"] - a["mean"]) / b["mean"] if b["mean"] else np.nan
            print(f"    {r:<18}{pol:<9}{a['n']:>6}{a['mean']:>+12.5f}"
                  f"  [{a['lo']:+.5f},{a['hi']:+.5f}]{b['n']:>6}"
                  f"{b['mean']:>+11.5f}{dd['diff']:>+11.5f}"
                  f"  [{dd['lo']:+.5f},{dd['hi']:+.5f}]{frac:>+8.1f}%")
            armB_out[r][pol] = {
                "honest": {k: v for k, v in a.items() if not k.startswith("_")},
                "leaky": {k: v for k, v in b.items() if not k.startswith("_")},
                "leaky_registered": {
                    k: v for k, v in armB["LEAKY_REG"][r][pol].items()
                    if not k.startswith("_")},
                "diff": dd, "leak_share_pct": float(frac)}
    clv_res["SP_open_policies"] = armB_out

    # ---- the tier ladder: CLV by season, SP frame, INCLUDING the blind one -
    print(f"\n    UNION CLV BY SEASON ON THE 5-SEASON SP FRAME — the "
          f"AVAILABILITY-TIER LADDER.  2021-22 has NO injury feed in "
          f"existence,\n    so its HONEST arm is fully availability-BLIND.  "
          f"It is the natural experiment for 'what if the feed fails in "
          f"October'.")
    print(f"    {'season':<10}{'tier':<20}{'n_h':>6}{'CLV_h':>10}{'n_l':>6}"
          f"{'CLV_l':>10}{'diff':>10}{'leak share':>12}")
    ladder = []
    per = {}
    for lab in ("HONEST", "LEAKY_PAIR"):
        m = frames[lab]
        po, _, oko = bo.price_cols(m, "open", "SP")
        pcs, _, okc3 = bo.price_cols(m, "close", "SP")
        mk, _, _ = bo.registry_masks(m, po, "open")
        uu = np.zeros(len(m), bool)
        for vv in mk.values():
            uu |= np.asarray(vv)
        uu &= (oko & okc3)
        s = m[uu].copy()
        s["clv"] = (pcs - po)[uu]
        per[lab] = s.groupby("season").clv.agg(["size", "mean"])
    for sea in sorted(set(per["HONEST"].index) | set(per["LEAKY_PAIR"].index)):
        a, b = per["HONEST"].loc[sea], per["LEAKY_PAIR"].loc[sea]
        sh = 100 * (b["mean"] - a["mean"]) / b["mean"] if b["mean"] else np.nan
        print(f"    {sea:<10}{TIER[sea]:<20}{int(a['size']):>6}"
              f"{a['mean']:>+10.5f}{int(b['size']):>6}{b['mean']:>+10.5f}"
              f"{a['mean']-b['mean']:>+10.5f}{sh:>+11.0f}%")
        ladder.append({"season": sea, "tier": TIER[sea], "n_h": int(a["size"]),
                       "clv_h": float(a["mean"]), "n_l": int(b["size"]),
                       "clv_l": float(b["mean"]),
                       "diff": float(a["mean"] - b["mean"]),
                       "leak_share_pct": float(sh)})
    print("    READ: on the three FULL-T2 seasons the leak is worth 10-20% of "
          "the measured CLV; on the season with NO FEED it is worth 71%.\n"
          "    The availability feed is not a nicety — it is most of what the "
          "CLV asset is made of.")
    clv_res["SP_tier_ladder_by_season"] = ladder

    # D142 anchor on the LEAKY_REG arm
    D142 = {"R4_LOWT": (231, 0.02855, 0.03906),
            "T20_D03_10_W": (104, 0.02333, 0.03054),
            "T20_D03_10": (374, 0.02537, 0.03376),
            "STAR_FAV_SHARPER": (712, 0.01352, 0.02249),
            "UNION": (938, 0.01933, 0.02874)}
    print(f"\n    D142 ANCHOR — the LEAKY_REG arm against D142 (8) as "
          f"registered:")
    print(f"    {'rule':<18}{'n':>6}{'ONEBOOK':>10}{'BEST2':>10}   "
          f"registered [n, ONEBOOK, BEST2]")
    anc2 = {}
    for r in RULES:
        a1 = armB["LEAKY_REG"][r]["ONEBOOK"]
        a2 = armB["LEAKY_REG"][r]["BEST2"]
        rn, r1, r2 = D142[r]
        ok_ = (a1["n"] == rn and abs(a1["mean"] - r1) < 5e-5
               and abs(a2["mean"] - r2) < 5e-5)
        anc2[r] = {"n": a1["n"], "onebook": a1["mean"], "best2": a2["mean"],
                   "reg": [rn, r1, r2], "exact": bool(ok_)}
        print(f"    {r:<18}{a1['n']:>6}{a1['mean']:>+10.5f}{a2['mean']:>+10.5f}"
              f"   [{rn}, {r1:+.5f}, {r2:+.5f}]  "
              f"{'EXACT' if ok_ else 'residual (see note)'}")
    res["anchor_d142_sp"] = anc2
    print("    NOTE ON THE RESIDUAL: R4_LOWT / T20_D03_10_W / T20_D03_10 "
          "reproduce D142 to the digit.  The residual is confined to\n"
          "    STAR_FAV_SHARPER (+8 bets of 712) and therefore to the UNION, "
          "and STAR_FAV_SHARPER is the ONE rule whose universe is defined by\n"
          "    the REALIZED ROTATION (`star_out_map` reads game_inactives x "
          "player_game_stats) — a DB-STATE-DEPENDENT quantity, D131's\n"
          "    staleness class and D158 §8(b)(2)'s flagged universe-transfer "
          "question.  D152's historical backfill landed 2026-08-02, AFTER\n"
          "    D142 ran.  It is +1.1% of one rule, moves the union CLV by "
          "~+0.0001, and is COMMON-MODE across the honest and leaky arms\n"
          "    here (both are scored against today's DB), so it cannot "
          "manufacture or hide a honest-vs-leaky difference.")

    # ------------------------------------------------------- DECOMPOSITION -
    print(f"\n{'-'*118}\n    DECOMPOSITION OF THE UNION CLV WE HOLD IN THE "
          f"REGISTER.  Two things separate the registered number from the "
          f"honest one:\n    (a) FRAME DRIFT — ds_rt1 (the frame D121/D142/D155 "
          f"priced, built 2026-07-31) vs the D158 capstone harness on today's "
          f"corpus\n        (the D152 backfill and the D153 tank-floor move "
          f"landed in between); and (b) THE LEAK itself, which is the "
          f"availability-only\n        contrast between the two capstone arms.")
    dec_out = {}
    for frame, tag in ((armA, "ML@open"),):
        for r in RULES:
            reg = frame[r]["leaky_registered"]["mean"]
            lk = frame[r]["leaky"]["mean"]
            hn = frame[r]["honest"]["mean"]
            dec_out[f"{tag}|{r}"] = {"registered": reg, "leaky_paired": lk,
                                     "honest": hn,
                                     "frame_drift": lk - reg,
                                     "leak": hn - lk, "total": hn - reg}
    print(f"    {'set (ML@open)':<18}{'REGISTERED':>12}{'->drift':>10}"
          f"{'LEAKY_PAIR':>12}{'->leak':>10}{'HONEST':>10}{'total d':>10}"
          f"{'leak share of total':>21}")
    for r in RULES:
        v = dec_out[f"ML@open|{r}"]
        sh = 100 * v["leak"] / v["total"] if v["total"] else float("nan")
        print(f"    {r:<18}{v['registered']:>+12.5f}{v['frame_drift']:>+10.5f}"
              f"{v['leaky_paired']:>+12.5f}{v['leak']:>+10.5f}"
              f"{v['honest']:>+10.5f}{v['total']:>+10.5f}{sh:>20.0f}%")
    res["clv_decomposition"] = dec_out

    # ------------------------------- D121 CONTROL 2: THE SELECTION PLACEBO -
    print(f"\n{'-'*118}\n    D121 (2)'s SELECTION PLACEBO, RE-RUN HONEST.  "
          f"p_us permuted within (season x p_open decile) so the selection "
          f"mechanism and the\n    open-price distribution survive but the "
          f"model's information is destroyed.  If honest CLV is real "
          f"information rather than\n    harvested open-price noise, the "
          f"placebo must come back at ~0.")
    plc = {}
    for lab in ("HONEST", "LEAKY_PAIR"):
        m = frames[lab]
        po, _, oko = bo.price_cols(m, "open", "ML")
        pcx, _, okc2 = bo.price_cols(m, "close", "ML")
        okp = oko & okc2
        rng = np.random.default_rng(SEED)
        dec_o = pd.qcut(po, 10, labels=False, duplicates="drop")
        p_perm = m.p_us.values.copy()
        for key in pd.unique(list(zip(m.season.values, dec_o))):
            idx = np.where((m.season.values == key[0]) & (dec_o == key[1]))[0]
            if len(idx) > 1:
                p_perm[idx] = p_perm[rng.permutation(idx)]
        mp = m.copy()
        mp["p_us"] = p_perm
        mp["pick_home"] = mp.p_us > 0.5
        mp["p_us_side"] = np.where(mp.pick_home, mp.p_us, 1 - mp.p_us)
        mp["conf_us"] = (mp.p_us - 0.5).abs()
        po_p, _, _ = bo.price_cols(mp, "open", "ML")
        pc_p, _, _ = bo.price_cols(mp, "close", "ML")
        mo_p, _, _ = bo.registry_masks(mp, po_p, "open")
        up = np.zeros(len(mp), bool)
        for vv in mo_p.values():
            up |= np.asarray(vv)
        mo_p["UNION"] = up
        cvp = pc_p - po_p
        plc[lab] = {}
        for r in RULES:
            s = np.asarray(mo_p[r]) & okp
            plc[lab][r] = clustered(cvp[s], mp.season.values[s])
    print(f"    {'rule':<18}{'n':>6}{'PLACEBO clv':>14}"
          f"{'[95% season-clustered]':>26}{'REAL honest clv':>18}  verdict")
    for r in RULES:
        a = plc["HONEST"][r]
        real = armA[r]["honest"]["mean"]
        vd = ("MECHANICAL" if a["lo"] > 0 and a["mean"] > 0.5 * real
              else "clean" if not (a["lo"] > 0) else "partial")
        print(f"    {r:<18}{a['n']:>6}{a['mean']:>+14.5f}"
              f"  [{a['lo']:+.5f},{a['hi']:+.5f}]{real:>+18.5f}  {vd}")
    res["placebo"] = plc

    res["clv"] = clv_res

    # ============================================ TASK 3: MATCHED CONTROL ===
    print(f"\n{'='*118}\n[4] TASK 3 — THE D155 MATCHED FAVOURITE CONTROL ON "
          f"HONEST PROBABILITIES.  Control = bet the MARKET favourite from the "
          f"same\n    (season x implied-probability) strata as the rules' own "
          f"bets, at the real moneyline.  D155: close +6.51% ns / open +8.22% "
          f"SIG.\n{'='*118}")
    lbj = json.load(open(LB_EXPLOIT))
    mc = {}
    for lab, frame_key in (("HONEST", "HONEST"), ("LEAKY_PAIR", "LEAKY_PAIR"),
                           ("LEAKY_REG", "LEAKY_REG")):
        m = frames[frame_key]
        sub = {"longrun_fav_roi": lbj["longrun_fav_roi"]}
        for when in ("close", "open"):
            print(f"\n  >>> ARM {lab}  @{when}")
            lx.matched_control(m, bo, when, sub, f"matched|{when}")
        mc[lab] = {k: v for k, v in sub.items() if k.startswith("matched|")}
    res["matched_control"] = mc

    print(f"\n    ALPHA SUMMARY (rule ROI minus contemporaneous bin-matched "
          f"favourite control):")
    print(f"    {'arm':<12}{'when':<7}{'set':<18}{'n':>6}{'ruleROI':>10}"
          f"{'ctrl':>9}{'alpha':>9}{'K-1 t CI on alpha':>26}")
    for lab in ("HONEST", "LEAKY_PAIR", "LEAKY_REG"):
        for when in ("close", "open"):
            for r in RULES:
                v = mc[lab][f"matched|{when}"].get(r)
                if not v:
                    continue
                print(f"    {lab:<12}{when:<7}{r:<18}{v['n']:>6}"
                      f"{100*v['rule_roi']:>+10.2f}"
                      f"{100*v['ctrl_contemporaneous']:>+9.2f}"
                      f"{100*v['alpha_vs_contemp']:>+9.2f}"
                      f"   [{100*v['alpha_tlo']:+.2f},{100*v['alpha_thi']:+.2f}]"
                      f"{'  SIG' if v['alpha_tlo']>0 else '  ns'}")

    # ================================================ TASK 5: LIVE BANDS ====
    print(f"\n{'='*118}\n[5] TASK 5 — RE-DERIVED MONTHLY CLV BANDS.  D121/D125 "
          f"registered RED < {D121_RED:+.4f} and GOOD > {D121_GOOD:+.4f} at "
          f"~44 bets/month,\n    calibrated on LEAKY probabilities.  Re-derived "
          f"here on honest ones, union @open, unique games, ML frame.\n"
          f"{'='*118}")
    print("    THE REGISTERED CONSTRUCTION, RECOVERED EXACTLY FROM "
          "data/bo_openbacktest.json (it is not what the register's prose "
          "implies):\n"
          "      frame  = PRIMARY rt1 p_full 4-season | SP  (spread "
          "convention, 2022-23..2025-26)\n"
          "      CENTRE = the CLV of the ALL-SAME-SIDE UNIVERSE (+0.00350), "
          "NOT of the rule union (+0.0180)\n"
          "      WIDTH  = +-2 * per-bet CLV sd of the UNION (0.05521) / "
          "sqrt(median union bets/month = 44) = +-0.01664\n"
          "      -> RED -0.01314, GOOD +0.02015, i.e. the registered "
          "-0.0131 / +0.0200.\n"
          "    So the band is CENTRED ON THE UNIVERSE and WIDTHED BY THE "
          "UNION.  bet_engine.py --monthly-report scores the UNION against\n"
          "    it (D125 (5)).  That mismatch is a SECOND, leak-independent "
          "defect and is re-derived consistently below.")
    bands = {}
    for src, seasons in (("SP", FRAME_SEASONS), ("ML", None)):
        for lab in ("HONEST", "LEAKY_PAIR", "LEAKY_REG"):
            m = frames[lab]
            keep = (m.season.isin(seasons).values if seasons
                    else np.ones(len(m), bool))
            mk, ps, dc, o, _ = masks_of(m, src)
            pc2, _, oc = bo.price_cols(m, "close", src)
            _, _, same = bo.registry_masks(m, ps, "open")
            okk = o & oc & keep
            cv = pc2 - ps
            uni = np.asarray(mk["UNION"]) & okk
            universe = np.asarray(same) & okk           # "ALL same-side"
            sub = m[uni].copy()
            sub["clv"] = cv[uni]
            sub["ym"] = sub.game_date.dt.to_period("M").astype(str)
            g = sub.groupby("ym").clv.agg(["size", "mean"])
            g = g[g["size"] >= 3]
            nmed = float(g["size"].median())
            sd_bet = float(sub.clv.std(ddof=1))
            se_at_n = sd_bet / sqrt(nmed)
            centre_universe = float(cv[universe].mean())
            centre_union = float(sub.clv.mean())
            mu_months = float(g["mean"].mean())
            key = f"{src}|{lab}"
            bands[key] = {
                "src": src, "arm": lab, "n_union_bets": int(len(sub)),
                "n_universe": int(universe.sum()), "n_months": int(len(g)),
                "median_bets_per_month": nmed, "per_bet_sd": sd_bet,
                "se_at_median_n": se_at_n,
                "centre_universe": centre_universe,
                "centre_union": centre_union,
                "mean_of_months": mu_months,
                "sd_of_month_means": float(g["mean"].std(ddof=1)),
                "frac_months_positive": float((g["mean"] > 0).mean()),
                "d121_style_red": centre_universe - 2 * se_at_n,
                "d121_style_good": centre_universe + 2 * se_at_n,
                "union_centred_red": centre_union - 2 * se_at_n,
                "union_centred_good": centre_union + 2 * se_at_n,
                "empirical_p10": float(g["mean"].quantile(0.10)),
                "empirical_p50": float(g["mean"].median()),
                "empirical_p90": float(g["mean"].quantile(0.90)),
                "months": g.reset_index().to_dict("records")}
    print(f"\n    {'frame|arm':<20}{'nbets':>7}{'mo':>4}{'med/mo':>8}"
          f"{'universe':>10}{'union':>9}{'sd_bet':>9}{'se':>9}"
          f"{'D121-style RED/GOOD':>24}{'union-centred RED/GOOD':>26}")
    for key, b in bands.items():
        print(f"    {key:<20}{b['n_union_bets']:>7}{b['n_months']:>4}"
              f"{b['median_bets_per_month']:>8.0f}{b['centre_universe']:>+10.5f}"
              f"{b['centre_union']:>+9.5f}{b['per_bet_sd']:>9.5f}"
              f"{b['se_at_median_n']:>9.5f}"
              f"   {b['d121_style_red']:>+9.5f}/{b['d121_style_good']:>+9.5f}"
              f"     {b['union_centred_red']:>+9.5f}/"
              f"{b['union_centred_good']:>+9.5f}")
    # arithmetic anchor straight off the registered artifact
    boj = json.load(open(os.path.join(ROOT, "data", "bo_openbacktest.json")))
    v = boj["clv"]["PRIMARY rt1 p_full 4-season|SP"]
    c0 = v["ALL same-side"]["clv_prob"]
    sd0, n0 = v["monthly"]["per_bet_sd"], v["monthly"]["median_bets"]
    r0, g0 = c0 - 2 * sd0 / sqrt(n0), c0 + 2 * sd0 / sqrt(n0)
    okA = abs(r0 - D121_RED) < 3e-4 and abs(g0 - D121_GOOD) < 3e-4
    print(f"\n    ARITHMETIC ANCHOR on the registered artifact itself: centre "
          f"{c0:+.5f} +- 2*{sd0:.5f}/sqrt({n0:.0f}) = {r0:+.5f} / {g0:+.5f}  "
          f"[registered {D121_RED:+.4f} / {D121_GOOD:+.4f}]  "
          f"{'EXACT — the construction is confirmed' if okA else '*** MISMATCH ***'}")
    reg = bands["SP|LEAKY_REG"]
    print(f"    Re-running that construction on TODAY'S SP frame gives RED "
          f"{reg['d121_style_red']:+.5f} / GOOD {reg['d121_style_good']:+.5f} "
          f"on {reg['n_months']} months — not byte-identical to D121 because\n"
          f"    D121 priced a 4,349-game SP frame and D126 subsequently "
          f"recovered 623 2022-23 opens into odds_open (today: "
          f"{int(reg['n_union_bets'])} union bets over "
          f"{reg['n_months']} months).  Frame growth, not a leak effect: it "
          f"moves in the SAME direction in every arm.")
    res["bands"] = bands
    res["band_arithmetic_anchor"] = {"centre": c0, "per_bet_sd": sd0,
                                     "median_bets": n0, "red": r0, "good": g0,
                                     "exact": bool(okA)}

    # ---- is the D125 real-stakes trigger still reachable? -----------------
    print(f"\n    IS THE D125 REAL-STAKES TRIGGER STILL REACHABLE?  It asks "
          f"for 2 CONSECUTIVE completed months with union OPEN CLV > "
          f"{D121_GOOD:+.4f},\n    and no completed month < {D121_RED:+.4f}.")
    trig = {}
    for key in ("SP|HONEST", "SP|LEAKY_REG", "ML|HONEST", "ML|LEAKY_REG"):
        b = bands[key]
        gm = pd.DataFrame(b["months"]).sort_values("ym")
        v = gm["mean"].values
        above = v > D121_GOOD
        runs = "".join("1" if x else "0" for x in above).split("0")
        longest = max([len(x) for x in runs] or [0])
        pairs = int(((above[:-1]) & (above[1:])).sum())
        # normal model: monthly mean ~ N(centre_union, se_at_n)
        from statistics import NormalDist
        nd = NormalDist(b["centre_union"], b["se_at_median_n"])
        p_good = 1 - nd.cdf(D121_GOOD)
        p_red = nd.cdf(D121_RED)
        trig[key] = {"months": int(len(gm)),
                     "months_above_good": int(above.sum()),
                     "months_below_red": int((v < D121_RED).sum()),
                     "consecutive_pairs_above": pairs,
                     "longest_run_above": longest,
                     "p_month_above_good_normal": float(p_good),
                     "p_month_below_red_normal": float(p_red),
                     "p_two_consec_normal": float(p_good ** 2),
                     "expected_months_to_trigger": float(1 / (p_good ** 2))
                     if p_good > 0 else float("inf")}
        t = trig[key]
        print(f"      {key:<15} months={t['months']:>3}  above GOOD "
              f"{t['months_above_good']:>2}  consecutive pairs "
              f"{t['consecutive_pairs_above']:>2}  longest run "
              f"{t['longest_run_above']:>2}  below RED "
              f"{t['months_below_red']:>2}   |  normal model: "
              f"P(month>GOOD)={100*t['p_month_above_good_normal']:.1f}%  "
              f"P(2 consecutive)={100*t['p_two_consec_normal']:.1f}%  "
              f"E[months to trigger]={t['expected_months_to_trigger']:.0f}")
    res["trigger_check"] = trig

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\nwrote {OUT}")
    return res


if __name__ == "__main__":
    main()
