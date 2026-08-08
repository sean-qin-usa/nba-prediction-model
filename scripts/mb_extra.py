#!/usr/bin/env python3
"""PART 3 (c) — the three things the main scoring run leaves open:

  (1) THE COMBINED HAIRCUT. Prereg §7 lists JUICE (H1) and OUTLIER REALISM (H2)
      as separate haircuts. Applying them together is the honest worst case and
      is what the verdict is written against.
  (2) THE ERA COMPARISON, APPLES TO APPLES. ESPN23 and KAG differ in BOTH era
      and price phase. ESPN23's CLOSE panel vs KAG's CLOSE panel is the same
      phase, so the difference is era/market alone.
  (3) THE PER-SEASON KAG LADDER (K=11) — is the shop gain trending, and what is
      the season-clustered interval on the gain itself?

Same prereg, same conventions, same exact-subset weights.
"""
from __future__ import annotations

import nbapred.threads  # noqa: F401
nbapred.threads.pin(1)

import collections
import json
import math
import numpy as np

import mb_score as S       # reuse every convention verbatim

ROOT = "/hdd/steveqin/sean_dev/nba_model"
R = {}


def say(*a):
    S.say(*a)


def main():
    S.LOG.clear()
    say("mb_extra.py — combined haircut, era comparison, per-season ladder")
    say("prereg sha256 b7d93a57d513b7ce23d9053cbe31baf2d2afafc4cacd79f1a93a809914c6717f")
    ats = S.load_ats19()
    espn = json.load(open(f"{ROOT}/data/mb_panel_espn.json"))
    kag = json.load(open(f"{ROOT}/data/mb_panel_kag.json"))

    # (1) combined haircut, both phases
    R["espn_open_juice_outlier"] = S.score_panel(
        espn, ats, "ESPN23 OPEN — JUICE **AND** OUTLIER-REALISM (the worst case)",
        phase="open", use_juice=True, haircut="outlier")
    R["espn_close_juice_outlier"] = S.score_panel(
        espn, ats, "ESPN23 CLOSE — JUICE **AND** OUTLIER-REALISM",
        phase="close", use_juice=True, haircut="outlier")
    R["kag_juice_outlier"] = S.score_panel(
        kag, ats, "KAG CLOSE — JUICE **AND** OUTLIER-REALISM, K=11",
        phase="open", use_juice=True, haircut="outlier")

    # (2) era comparison at a common phase (CLOSE), raw handicap
    say("\n" + "=" * 72)
    say("ERA COMPARISON AT A COMMON PRICE PHASE (CLOSE, raw handicap)")
    say("=" * 72)
    a = {t["k"]: t["gain_pts"] for t in R and json.load(
        open(f"{ROOT}/data/mb_score.json"))["espn_close"]["table"]}
    b = {t["k"]: t["gain_pts"] for t in json.load(
        open(f"{ROOT}/data/mb_score.json"))["kag"]["table"]}
    say(f"{'k':>3s} {'ESPN23 close (US retail 23-24)':>32s} "
        f"{'KAG close (offshore 07-18)':>28s} {'ratio':>7s}")
    era = []
    for k in S.KS:
        if k == 1:
            continue
        say(f"{k:3d} {a[k]:32.4f} {b[k]:28.4f} {a[k]/max(b[k],1e-9):7.2f}x")
        era.append(dict(k=k, espn_close=a[k], kag_close=b[k], ratio=a[k] / b[k]))
    R["era"] = era

    # (3) per-season KAG ladder + season-clustered interval ON THE GAIN
    say("\n" + "=" * 72)
    say("PER-SEASON KAG LADDER (K=11) — is the shop gain era-stable?")
    say("=" * 72)
    rows = []
    for gid, d in kag.items():
        r = ats.get(gid)
        if r is None or r["m_us"] is None or r["open_margin"] is None:
            continue
        v = np.array(sorted(q["m"] for q in d.values() if q["m"] is not None))
        if len(v) < 2:
            continue
        side = "H" if r["m_us"] > r["open_margin"] else "A"
        rows.append((r["season"], side, v))
    say(f"{'season':>9s} {'n':>6s} {'k=2':>7s} {'k=5':>7s} {'k=8':>7s}")
    bysea = collections.defaultdict(list)
    for s, side, v in rows:
        bysea[s].append((side, v))
    tab = []
    gains_by_k = {2: [], 5: [], 8: []}
    for s in sorted(bysea):
        g = {}
        for k in (2, 5, 8):
            vals = []
            for side, v in bysea[s]:
                n = len(v)
                kk = min(k, n)
                w1 = S.subset_weights(n, 1, "min" if side == "H" else "max")
                wk = S.subset_weights(n, kk, "min" if side == "H" else "max")
                vals.append(abs(float((wk * v).sum() - (w1 * v).sum())))
            g[k] = float(np.mean(vals))
            gains_by_k[k].append(g[k])
        say(f"{s:>9s} {len(bysea[s]):6d} {g[2]:7.4f} {g[5]:7.4f} {g[8]:7.4f}")
        tab.append(dict(season=s, n=len(bysea[s]), **{f"k{k}": g[k] for k in (2, 5, 8)}))
    R["kag_by_season"] = tab
    say("")
    for k in (2, 5, 8):
        v = np.array(gains_by_k[k])
        K = len(v)
        try:
            from scipy import stats
            tc = float(stats.t.ppf(0.975, K - 1))
        except Exception:
            tc = 2.228
        h = tc * v.std(ddof=1) / math.sqrt(K)
        say(f"  k={k}: season-clustered mean gain {v.mean():.4f} pts "
            f"K-1 t [{v.mean()-h:.4f},{v.mean()+h:.4f}] at {K-1} dof "
            f"(min {v.min():.4f} max {v.max():.4f})")
        R[f"kag_gain_t_k{k}"] = dict(mean=float(v.mean()), lo=float(v.mean() - h),
                                     hi=float(v.mean() + h), K=K)

    # (4) the final arithmetic
    say("\n" + "=" * 72)
    say("FINAL ARITHMETIC — D162 §6 UNITS. edge 0.206, need 0.751, GAP 0.545")
    say("=" * 72)
    sc = json.load(open(f"{ROOT}/data/mb_score.json"))
    arms = [
        ("ESPN23 open, raw", sc["espn_open"]),
        ("ESPN23 open, +juice", sc["espn_open_juice"]),
        ("ESPN23 close, raw (simultaneous)", sc["espn_close"]),
        ("ESPN23 open, +outlier", sc["espn_open_outlier"]),
        ("ESPN23 open, +juice +outlier  <-- WORST CASE", R["espn_open_juice_outlier"]),
        ("ESPN23 close, +juice +outlier", R["espn_close_juice_outlier"]),
        ("KAG close, raw (K=11)", sc["kag"]),
        ("KAG close, +juice +outlier (K=11)", R["kag_juice_outlier"]),
    ]
    say(f"{'arm':46s} {'k5 gain':>8s} {'0.206+g':>8s} {'k8 gain':>8s} "
        f"{'0.206+g':>8s} {'clears .751?':>13s}")
    fin = []
    for nm, d in arms:
        if not d:
            continue
        g = {t["k"]: t["gain_pts"] for t in d["table"]}
        e5, e8 = 0.206 + g[5], 0.206 + g[8]
        say(f"{nm:46s} {g[5]:8.4f} {e5:8.4f} {g[8]:8.4f} {e8:8.4f} "
            f"{('YES at k=5' if e5 >= 0.751 else ('YES at k=8' if e8 >= 0.751 else 'NO')):>13s}")
        fin.append(dict(arm=nm, k5=g[5], k8=g[8], edge_k5=e5, edge_k8=e8,
                        clears_k5=e5 >= 0.751, clears_k8=e8 >= 0.751))
    R["final"] = fin

    with open(f"{ROOT}/data/mb_extra.json", "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(f"{ROOT}/data/logs/mb_extra.log", "w") as f:
        f.write("\n".join(S.LOG))
    say(f"\nwrote {ROOT}/data/mb_extra.json")


if __name__ == "__main__":
    main()
