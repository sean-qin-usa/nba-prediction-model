#!/usr/bin/env python3
"""PART 3 (b) — RE-PRICE D162's ATS RESULT AND D159's CLV UNDER *MEASURED*
BEST-OF-N EXECUTION.

Pre-registered in data/multibook_prereg.md
sha256 b7d93a57d513b7ce23d9053cbe31baf2d2afafc4cacd79f1a93a809914c6717f.

EXECUTION CONVENTION (prereg §3, = D142 §4): PURE EXECUTION. The bet set is
frozen by the REGISTERED consensus open under D162's untuned rule
    HOME iff m_us > open_margin,      m_us = 7.2*logit(p_us)
and ONLY the transacted handicap varies. HOME takes the MINIMUM home-margin
handicap in the subset, AWAY the MAXIMUM.

ONE DECLARED DEVIATION FROM THE PREREG, AND IT IS A STRICT IMPROVEMENT.
The prereg specified 200 Monte-Carlo k-subsets. The expectation over ALL
k-subsets is available in closed form: for values sorted ascending v_(1)..v_(n),
    P(min of a random k-subset = v_(i)) = C(n-i, k-1) / C(n, k)
    P(max of a random k-subset = v_(i)) = C(i-1, k-1) / C(n, k)
so E[f(best-of-k)] = sum_i w_i f(v_(i)) EXACTLY, for f = handicap, cover, ROI
or CLV alike. Same estimand, zero sampling error, no seed. The Monte-Carlo
ladder is computed too and the two are printed side by side as a check.

Read-only. No production default touched.
"""
from __future__ import annotations

import nbapred.threads  # noqa: F401
nbapred.threads.pin(1)

import collections
import csv
import gzip
import json
import math
import os
from datetime import datetime

import numpy as np

ROOT = "/hdd/steveqin/sean_dev/nba_model"
OUT = f"{ROOT}/data/mb_score.json"

DP_PER_PT = 0.3989422804014327 / 12.574     # D162 §6
BE110 = 110.0 / 210.0
WIN110 = 100.0 / 110.0
KS = [1, 2, 3, 4, 5, 8]

R = {}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def fnum(x):
    if x in (None, "", "None"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def breakeven(am):
    if am is None:
        return None
    a = float(am)
    return (-a) / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)


def juice_pts(am):
    b = breakeven(am)
    return None if b is None else (b - BE110) / DP_PER_PT


def subset_weights(n, k, which):
    """w[i] = P(the min (or max) of a uniformly random k-subset is the i-th
    order statistic), i ascending 0-based."""
    tot = math.comb(n, k)
    w = np.zeros(n)
    for i in range(n):
        if which == "min":
            c = math.comb(n - i - 1, k - 1) if n - i - 1 >= k - 1 else 0
        else:
            c = math.comb(i, k - 1) if i >= k - 1 else 0
        w[i] = c / tot
    return w


def clustered_t(vals, groups, alpha=0.05):
    """K-1 cluster-mean t interval (§9.1(4)). Returns (mean, lo, hi, K)."""
    import statistics
    g = collections.defaultdict(list)
    for v, k in zip(vals, groups):
        g[k].append(v)
    means = [float(np.mean(v)) for v in g.values()]
    K = len(means)
    m = float(np.mean(means))
    if K < 2:
        return m, float("nan"), float("nan"), K
    sd = float(np.std(means, ddof=1))
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(1 - alpha / 2, K - 1))
    except Exception:
        TAB = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
               8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
               14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101}
        tcrit = TAB.get(K - 1, 1.96)
    h = tcrit * sd / math.sqrt(K)
    return m, m - h, m + h, K


def load_ats19():
    with gzip.open(f"{ROOT}/data/ats19_frame.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("p_us", "m_us", "open_margin", "close_margin", "margin_actual"):
            r[k] = fnum(r[k])
    return {r["game_id"]: r for r in rows}


# --------------------------------------------------------------- scoring
def score_panel(panel, ats, label, phase="open", use_juice=False,
                haircut=None, mc_check=False):
    """panel: game_id -> {operator: {m, jh, ja, close}}"""
    rows = []
    for gid, d in panel.items():
        r = ats.get(gid)
        if r is None or r["open_margin"] is None or r["margin_actual"] is None:
            continue
        if r["m_us"] is None:
            continue
        side = "H" if r["m_us"] > r["open_margin"] else "A"
        vh, va = [], []
        ok = True
        for op in sorted(d):
            q = d[op]
            m = q["close"] if phase == "close" else q["m"]
            if m is None:
                ok = False
                break
            if use_juice:
                pj, pa = juice_pts(q["jh"]), juice_pts(q["ja"])
                if pj is None or pa is None:
                    ok = False
                    break
                vh.append(m + pj)
                va.append(m - pa)
            else:
                vh.append(m)
                va.append(m)
        if not ok or len(vh) < 2:
            continue
        v = np.array(sorted(vh)) if side == "H" else np.array(sorted(va))
        rows.append(dict(gid=gid, season=r["season"], side=side, v=v,
                         actual=r["margin_actual"], close=r["close_margin"],
                         open_c=r["open_margin"]))
    if not rows:
        say(f"  {label}: no scorable games")
        return None

    say(f"\n{'='*72}\n{label}   phase={phase}  juice={use_juice}"
        f"{'  haircut=' + haircut if haircut else ''}\n{'='*72}")
    say(f"n games {len(rows)}   panel size: " +
        ", ".join(f"{k}:{v}" for k, v in
                  sorted(collections.Counter(len(r['v']) for r in rows).items())))

    out = []
    per_k_game = {}
    for k in KS:
        hand = []      # E[transacted handicap]
        cov = []       # E[cover] (0/1, push excluded from cover but kept in ROI)
        roi = []
        clv = []
        gain = []
        seas = []
        push = []
        for r in rows:
            n = len(r["v"])
            kk = min(k, n)
            w = subset_weights(n, kk, "min" if r["side"] == "H" else "max")
            v = r["v"]
            if haircut == "outlier":
                # D142 §5(ii): a price more than 1.5 pts from the panel median
                # is not realistically transactable. Games are NEVER dropped —
                # dropping breaks the paired arrays — the weight is
                # redistributed onto the surviving quotes.
                med = float(np.median(v))
                keep = (np.abs(v - med) <= 1.5).astype(float)
                if keep.sum() == 0 or (w * keep).sum() <= 0:
                    w = np.zeros(n)
                    w[int(np.argmin(np.abs(v - med)))] = 1.0
                else:
                    w = w * keep
                    w = w / w.sum()
            elif haircut == "cap":
                med = float(np.median(v))
                v = (np.minimum(np.maximum(v, med - 0.75), med + 0.75))
            # outcome per candidate handicap
            if r["side"] == "H":
                c = np.where(r["actual"] > v, 1.0, np.where(r["actual"] < v, 0.0, np.nan))
                cl = (r["close"] - v) if r["close"] is not None else np.full(n, np.nan)
            else:
                c = np.where(r["actual"] < v, 1.0, np.where(r["actual"] > v, 0.0, np.nan))
                cl = (v - r["close"]) if r["close"] is not None else np.full(n, np.nan)
            ispush = np.isnan(c)
            pr = np.where(ispush, 0.0, np.where(c == 1.0, WIN110, -1.0))
            hand.append(float((w * v).sum()))
            push.append(float((w * ispush).sum()))
            nz = w * (~ispush)
            cov.append(float((nz * np.nan_to_num(c)).sum()) / max(nz.sum(), 1e-12)
                       if nz.sum() > 0 else np.nan)
            roi.append(float((w * pr).sum()))
            clv.append(float((w * np.nan_to_num(cl)).sum()))
            seas.append(r["season"])
            gain.append(float((w * v).sum()))
        hand = np.array(hand); cov = np.array(cov); roi = np.array(roi)
        clv = np.array(clv); push = np.array(push)
        per_k_game[k] = dict(hand=hand, cov=cov, roi=roi, clv=clv, seas=seas)
        m_cov, lo_c, hi_c, K = clustered_t(cov[~np.isnan(cov)],
                                           [s for s, c in zip(seas, cov) if not np.isnan(c)])
        m_roi, lo_r, hi_r, _ = clustered_t(roi, seas)
        m_clv, lo_v, hi_v, _ = clustered_t(clv, seas)
        out.append(dict(k=k, n=len(roi), K=K,
                        hand=float(hand.mean()), push=float(push.mean()),
                        cover=float(np.nanmean(cov)), cover_t=[lo_c, hi_c],
                        roi=float(roi.mean()) * 100, roi_t=[lo_r * 100, hi_r * 100],
                        clv=float(clv.mean()), clv_t=[lo_v, hi_v]))

    base = per_k_game[1]
    say(f"{'k':>3s} {'n':>6s} {'E[hand]':>8s} {'gain pt':>8s} {'cover%':>7s} "
        f"{'dcover':>7s} {'ROI%':>7s} {'K-1 t on ROI':>22s} {'CLVpt':>7s} {'dCLV':>7s}")
    for o, k in zip(out, KS):
        pk = per_k_game[k]
        g = float(np.mean(np.abs(base["hand"] - pk["hand"])))
        dc = float(np.nanmean(pk["cov"] - base["cov"])) * 100
        dv = float(np.mean(pk["clv"] - base["clv"]))
        o["gain_pts"] = g
        o["dcover_pp"] = dc
        o["dclv_pts"] = dv
        o["conv_ratio"] = (dc / 100.0) / (DP_PER_PT * g) if g > 0 else None
        say(f"{k:3d} {o['n']:6d} {o['hand']:8.3f} {g:8.4f} {100*o['cover']:7.3f} "
            f"{dc:+7.3f} {o['roi']:7.3f} "
            f"[{o['roi_t'][0]:+7.2f},{o['roi_t'][1]:+7.2f}] {o['clv']:7.4f} {dv:+7.4f}")
    say(f"season clusters K={out[0]['K']}")
    say("conversion check — realised dcover / (0.0317276 * gain):  " +
        "  ".join(f"k{o['k']}:{(o['conv_ratio'] if o['conv_ratio'] else float('nan')):.3f}"
                  for o in out if o["k"] > 1))

    if mc_check:
        rng = np.random.default_rng(20260804)
        say("MC cross-check of the exact weights (200 draws), gain pts:")
        for k in KS:
            if k == 1:
                continue
            gs = []
            for r in rows:
                n = len(r["v"])
                kk = min(k, n)
                idx = [rng.choice(n, kk, replace=False) for _ in range(200)]
                if r["side"] == "H":
                    gs.append(np.mean([r["v"][i].min() for i in idx]))
                else:
                    gs.append(np.mean([r["v"][i].max() for i in idx]))
            say(f"  k={k}  MC {float(np.mean(np.abs(base['hand'] - np.array(gs)))):.4f}"
                f"   exact {out[KS.index(k)]['gain_pts']:.4f}")
    return dict(table=out, n=len(rows))


def adverse_selection(panel, ats, label, k=8):
    """D142 §5(i) re-run at the panel's largest N: does the SIZE of the shop
    gain predict a lower cover rate? (Is the best price the stale price?)"""
    say(f"\n--- ADVERSE SELECTION AT BEST-OF-{k} — {label} ---")
    rec = []
    for gid, d in panel.items():
        r = ats.get(gid)
        if r is None or r["m_us"] is None or r["open_margin"] is None:
            continue
        v = np.array(sorted(q["m"] for q in d.values() if q["m"] is not None))
        if len(v) < k:
            continue
        side = "H" if r["m_us"] > r["open_margin"] else "A"
        best = v.min() if side == "H" else v.max()
        gain = abs(v.mean() - best)
        a = r["margin_actual"]
        c = 1.0 if ((side == "H" and a > best) or (side == "A" and a < best)) else (
            0.0 if ((side == "H" and a < best) or (side == "A" and a > best)) else np.nan)
        rec.append((gain, c, r["season"]))
    if len(rec) < 100:
        say("  too few games")
        return None
    g = np.array([x[0] for x in rec])
    c = np.array([x[1] for x in rec])
    m = ~np.isnan(c)
    r_ = float(np.corrcoef(g[m], c[m])[0, 1])
    say(f"n={len(rec)}  corr(shop gain, cover) = {r_:+.4f}")
    qs = np.quantile(g, [0, .25, .5, .75, 1.0])
    say(f"{'gain quartile':>16s} {'n':>6s} {'cover%':>7s}")
    tab = []
    for i in range(4):
        sel = (g >= qs[i]) & (g <= qs[i + 1]) & m
        if sel.sum() == 0:
            continue
        say(f"{f'[{qs[i]:.2f},{qs[i+1]:.2f}]':>16s} {int(sel.sum()):6d} "
            f"{100*float(c[sel].mean()):7.3f}")
        tab.append(dict(lo=float(qs[i]), hi=float(qs[i + 1]), n=int(sel.sum()),
                        cover=float(c[sel].mean())))
    return dict(n=len(rec), corr=r_, quartiles=tab)


def main():
    say(f"mb_score.py  start {datetime.utcnow().isoformat()}Z")
    say("prereg data/multibook_prereg.md sha256 "
        "b7d93a57d513b7ce23d9053cbe31baf2d2afafc4cacd79f1a93a809914c6717f")
    ats = load_ats19()
    espn = json.load(open(f"{ROOT}/data/mb_panel_espn.json"))
    kag = json.load(open(f"{ROOT}/data/mb_panel_kag.json"))
    say(f"ats19 games {len(ats)}   espn panel {len(espn)}   kag panel {len(kag)}")

    # ---- prereg §2: KAG snapshot validation, BEFORE any endpoint uses it
    say("\n" + "=" * 72)
    say("PREREG §2 GATE — WHAT IS THE KAGGLE SNAPSHOT? (open, close, neither)")
    say("=" * 72)
    eo = ec = n = 0
    do, dc = [], []
    for gid, d in kag.items():
        r = ats.get(gid)
        if r is None or r["open_margin"] is None or r["close_margin"] is None:
            continue
        med = float(np.median([q["m"] for q in d.values()]))
        n += 1
        eo += (med == r["open_margin"])
        ec += (med == r["close_margin"])
        do.append(abs(med - r["open_margin"]))
        dc.append(abs(med - r["close_margin"]))
    say(f"n={n}  panel MEDIAN == our OPEN  {100*eo/n:.2f}%  mean|diff| {np.mean(do):.4f}")
    say(f"      panel MEDIAN == our CLOSE {100*ec/n:.2f}%  mean|diff| {np.mean(dc):.4f}")
    verdict = ("CLOSE" if ec > eo and ec / n > 0.60 else
               "OPEN" if eo / n > 0.60 else "NEITHER (dispersion-only)")
    say(f"=> KAG snapshot is: {verdict}")
    R["kag_snapshot"] = dict(n=n, exact_open=eo / n, exact_close=ec / n,
                             mad_open=float(np.mean(do)), mad_close=float(np.mean(dc)),
                             verdict=verdict)

    # ---- ESPN, the modern US retail panel
    R["espn_open"] = score_panel(espn, ats, "ESPN23 US RETAIL — OPENING lines",
                                 phase="open", use_juice=False, mc_check=True)
    R["espn_open_juice"] = score_panel(espn, ats, "ESPN23 — OPENING, JUICE-ADJUSTED",
                                       phase="open", use_juice=True)
    R["espn_close"] = score_panel(espn, ats,
                                  "ESPN23 — CLOSING lines (THE SIMULTANEITY ARM)",
                                  phase="close", use_juice=False)
    R["espn_open_outlier"] = score_panel(espn, ats,
                                         "ESPN23 — OPENING, OUTLIER-REALISM HAIRCUT",
                                         phase="open", use_juice=False, haircut="outlier")
    R["espn_open_cap"] = score_panel(espn, ats,
                                     "ESPN23 — OPENING, GAIN CAPPED AT 0.75 FROM MEDIAN",
                                     phase="open", use_juice=False, haircut="cap")
    R["espn_adverse"] = adverse_selection(espn, ats, "ESPN23", k=8)

    # ---- KAG, the 11-season offshore panel (K=11, the shipping statistic)
    say("\nPREREG §2 GATE OUTCOME: the KAG snapshot is identified as the CLOSE")
    say("(80.27% exact, mean|diff| 0.110, against 18.07% / 1.052 for the open).")
    say("It is therefore a CLOSING-line panel. The SIDE is still frozen by the")
    say("registered consensus OPEN under D162's rule (no new selection), but the")
    say("transacted number is a CLOSING shop and every KAG row is labelled so.")
    R["kag"] = score_panel(kag, ats,
                           "KAG OFFSHORE 2007-08..2017-18 — CLOSING-panel shop, K=11",
                           phase="open", use_juice=False)
    R["kag_juice"] = score_panel(kag, ats,
                                 "KAG OFFSHORE — CLOSING-panel shop, JUICE-ADJUSTED",
                                 phase="open", use_juice=True)
    R["kag_adverse"] = adverse_selection(kag, ats, "KAG", k=8)

    # ---- THE ARITHMETIC THE PREREG §9 DECISION RULE NEEDS
    say("\n" + "=" * 72)
    say("PREREG §9 DECISION ARITHMETIC — DOES THE SHOP CLOSE THE 0.545-pt GAP?")
    say("=" * 72)
    say("D162 §6: effective edge 0.206 pts, -110 needs 0.751 pts, GAP = 0.545.")
    say(f"{'arm':52s} {'k=2':>7s} {'k=5':>7s} {'k=8':>7s} {'k8 vs gap':>10s}")
    summary = []
    for nm, key in (("ESPN23 open, raw handicap", "espn_open"),
                    ("ESPN23 open, JUICE-ADJUSTED (headline haircut)", "espn_open_juice"),
                    ("ESPN23 CLOSE panel (simultaneity arm)", "espn_close"),
                    ("ESPN23 open, OUTLIER-REALISM haircut", "espn_open_outlier"),
                    ("ESPN23 open, gain capped 0.75 from median", "espn_open_cap"),
                    ("KAG offshore close panel, raw handicap", "kag"),
                    ("KAG offshore close panel, JUICE-ADJUSTED", "kag_juice")):
        d = R.get(key)
        if not d:
            continue
        g = {t["k"]: t["gain_pts"] for t in d["table"]}
        say(f"{nm:52s} {g.get(2, float('nan')):7.4f} {g.get(5, float('nan')):7.4f} "
            f"{g.get(8, float('nan')):7.4f} {g.get(8, float('nan'))/0.545:9.2f}x")
        summary.append(dict(arm=nm, k2=g.get(2), k5=g.get(5), k8=g.get(8),
                            k8_over_gap=(g.get(8) / 0.545 if g.get(8) else None)))
    R["decision_arithmetic"] = summary

    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(f"{ROOT}/data/logs/mb_score.log", "w") as f:
        f.write("\n".join(LOG))
    say(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
