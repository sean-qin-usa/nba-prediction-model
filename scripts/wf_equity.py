#!/usr/bin/env python3
"""THE TRADING PATH OF THE BEST NO-LOOKAHEAD CALIBRATION STRATEGY, PRICED AT THE
OPEN, UNDER MULTI-BOOK EXECUTION.

Reuses, does not rebuild:
  scripts/oc_capacity.py   D164 harness  (frame, 600-cell masks, per-season agg, CI)
  scripts/as_adaptive.py   D165 loop     (window selection; nulls read from JSON)
  data/mb_panel_espn.json  D163 ESPN 9-operator US retail panel  (2023-24)
  data/mb_panel_kag.json   D163 Kaggle 9-operator offshore panel (2006-07..2017-18)

DIAGNOSTIC. Nothing ships. No production default touched. No DB opened.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import oc_capacity as oc  # noqa: E402

import os  # noqa: E402
# PATH OVERRIDES ONLY (D173 re-run on the D170/D171 backfilled data).  Defaults
# are byte-identical to D166's; no selection rule, band or tier is touched.
_TAG = os.environ.get("WF_TAG", "")
OUT = ROOT / "data" / f"wf_equity{_TAG}.json"
AS_JSON = ROOT / "data" / f"as_adaptive{os.environ.get('AS_TAG', '')}.json"
LOG = []
WIN = 100.0 / 110.0
K_COMMON = 5
TIERS = [1, 2, 5, 8, 9]   # D181: 9 = MAX books observed at the open (2023-24 panel)
DP_PER_PT = 0.3989422804014327 / 12.574     # D162 §6 linearisation constant

# era assignment for the multi-book execution model (declared in equity_notes.md)
OFFSHORE_SEASONS = tuple(f"20{y:02d}-{y+1:02d}" for y in range(6, 18))   # ..2017-18
GAP_SEASONS = ("2018-19", "2019-20", "2020-21", "2021-22", "2022-23")
RETAIL_MEASURED = ("2023-24",)
RETAIL_EXTRAP = ("2024-25", "2025-26")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ selection
def select(c_win, p_win, guard):
    """D164/D165 selection rule verbatim."""
    elig = c_win >= guard
    if not elig.any():
        return None
    roi = np.where(c_win > 0, p_win / np.maximum(c_win, 1e-9), np.nan)
    cand = np.where(elig, roi, -np.inf)
    best = int(np.argmax(cand))
    ties = np.where(np.isclose(cand, cand[best], rtol=0, atol=1e-12))[0]
    if len(ties) > 1:
        best = int(ties[np.lexsort((ties, -c_win[ties]))[0]])
    return best


def arm_window(cnt, pay, K, W, k_start=K_COMMON):
    """Season-window walk-forward selection. W=None -> all history."""
    steps = []
    for k in range(k_start, K):
        lo = 0 if W is None else max(0, k - W)
        nseas = k - lo
        c = cnt[:, lo:k].sum(1)
        p = pay[:, lo:k].sum(1)
        best = select(c, p, 100.0 * nseas)
        if best is None:
            continue
        steps.append({"k": k, "cfg": best, "lo": lo, "nseas": nseas,
                      "sel_roi": float(p[best] / c[best]),
                      "sel_n": float(c[best]),
                      "test_roi": float(pay[best, k] / cnt[best, k])
                      if cnt[best, k] > 0 else np.nan,
                      "test_n": float(cnt[best, k]),
                      "test_pay": float(pay[best, k])})
    return steps


# ------------------------------------------------------- shop-gain machinery
def subset_weights(n, k, which):
    """P(min|max of a uniformly random k-subset is the i-th order stat), ascending."""
    tot = math.comb(n, k)
    w = np.zeros(n)
    for i in range(n):
        if which == "min":
            c = math.comb(n - i - 1, k - 1) if n - i - 1 >= k - 1 else 0
        else:
            c = math.comb(i, k - 1) if i >= k - 1 else 0
        w[i] = c / tot
    return w


def gain_dist(v, side, k, haircut):
    """Exact distribution of the shop gain g (points, toward our side) for one game.

    v          sorted ascending array of the panel's home-margin handicaps
    side       'H' (we want the MINIMUM home-margin bar) or 'A' (the MAXIMUM)
    returns    (g values, weights summing to 1)
    """
    n = len(v)
    kk = min(k, n)
    w = subset_weights(n, kk, "min" if side == "H" else "max")
    if haircut:
        # D142 §5(ii) as tightened by D163 §10: a quote more than 1.5 pts from the
        # panel MEDIAN is not realistically transactable; weight is redistributed
        # onto the survivors and no game is ever dropped.
        med = float(np.median(v))
        keep = (np.abs(v - med) <= 1.5).astype(float)
        if keep.sum() == 0 or (w * keep).sum() <= 0:
            w = np.zeros(n)
            w[int(np.argmin(np.abs(v - med)))] = 1.0
        else:
            w = w * keep
            w = w / w.sum()
    cons = float(v.mean())
    g = (cons - v) if side == "H" else (v - cons)
    return g, w


def load_panels():
    """game_id -> sorted quote vector, for both D163 panels."""
    out = {}
    for nm, f in (("KAG", "mb_panel_kag.json"), ("ESPN", "mb_panel_espn.json")):
        d = json.load(open(ROOT / "data" / f))
        p = {}
        for gid, ops in d.items():
            vs = sorted(q["m"] for q in ops.values() if q.get("m") is not None)
            if vs:
                p[gid] = np.array(vs, float)
        out[nm] = p
    return out


def pooled_dist(panel, k, haircut, side, rng_max=None):
    """Pool the per-game gain distributions of a whole panel into one empirical law.

    rng_max: if set, restrict to games whose best-worst range <= rng_max (the
    'offside tail excluded' limits arm) and also return the kept fraction.
    """
    gs, ws = [], []
    kept = tot = 0
    for gid, v in panel.items():
        tot += 1
        if rng_max is not None and (v.max() - v.min()) > rng_max:
            continue
        kept += 1
        g, w = gain_dist(v, side, k, haircut)
        gs.append(g)
        ws.append(w)
    G = np.concatenate(gs)
    W = np.concatenate(ws) / len(ws)
    o = np.argsort(G, kind="stable")
    return G[o], W[o], (kept / tot if tot else 1.0)


def payoff_from_dist(d, G, cw):
    """E[payoff] and P(win)/P(loss)/P(push) for one bet, given a gain law.

    Bet wins iff d + g > 0, pushes iff = 0, loses iff < 0.  G ascending, cw = cumsum(W).
    """
    t = -d
    lo = np.searchsorted(G, t, side="left")
    hi = np.searchsorted(G, t, side="right")
    tw = float(cw[-1])
    p_lt = float(cw[lo - 1]) if lo > 0 else 0.0
    p_le = float(cw[hi - 1]) if hi > 0 else 0.0
    p_eq = p_le - p_lt
    p_gt = tw - p_le
    return WIN * p_gt - p_lt, p_gt, p_lt, p_eq


# ---------------------------------------------------------------------- main
def main():
    say("=" * 78)
    say("WALK-FORWARD EQUITY PATH AT THE OPEN, UNDER MEASURED MULTI-BOOK EXECUTION")
    say("=" * 78)

    df, seasons = oc.load()
    K = len(seasons)
    st = oc.build_static(df)
    m_us = df["m_us"].to_numpy(float)
    p_us = df["p_us"].to_numpy(float)
    payoff, M, keys, win, push, bet_home = oc.payoff_and_masks(m_us, p_us, st)
    cnt, pay = oc.agg(M, payoff, st)
    say(f"frame {st['n']} games, {K} seasons, {len(keys)} configs")

    # anchor: D162 cell 1 of 600
    ref = 0
    say(f"ANCHOR D162 POOL19  ROI {100*pay[ref].sum()/cnt[ref].sum():+.3f}%  "
        f"cover {100*(M[ref]*win).sum()/((M[ref]).sum()-(M[ref]*push).sum()):.3f}%  "
        f"pushes {int((M[ref]*push).sum())}")

    R = {"seasons": seasons, "n_games": int(st["n"])}

    # ================================================== STEP 1: window choice
    ad = json.load(open(AS_JSON))
    WINDOWS = [("RECENCY-1", 1, "A_REC1"), ("RECENCY-2", 2, "B_REC2"),
               ("RECENCY-3", 3, "B_REC3"), ("RECENCY-5", 5, "B_REC5"),
               ("ALL-HISTORY", None, "C_ALL")]
    say("\n" + "-" * 78)
    say("STEP 1  THE WINDOW-CHOICE RULE (declared in data/equity_notes.md first)")
    say("-" * 78)
    say(f"{'window':14s} {'ROI%':>7s} {'n':>6s} {'null%':>7s} {'net':>7s} "
        f"{'decay':>7s} {'cfg chg':>8s}  anchor-vs-D165")
    winrows = []
    steps_by_w = {}
    for nm, W, key in WINDOWS:
        steps = arm_window(cnt, pay, K, W)
        steps_by_w[nm] = steps
        n = sum(s["test_n"] for s in steps)
        p = sum(s["test_pay"] for s in steps)
        roi = p / n
        per = [s["test_roi"] for s in steps]
        d165 = ad["arms"][key]
        ok = np.allclose(per, d165["per_season_roi"], atol=1e-9)
        nullm = ad["null"][key]["mean"]
        chg = sum(1 for i in range(1, len(steps))
                  if steps[i]["cfg"] != steps[i - 1]["cfg"])
        winrows.append(dict(name=nm, W=(W if W else 99), key=key, roi=roi, n=n,
                            null=nullm, net=roi - nullm,
                            decay=d165["decay"], changes=chg,
                            ci=oc.cluster_mean_t(per), per=per,
                            mde80=d165["mde80"], fwer_p=d165["fwer_p"]))
        say(f"{nm:14s} {100*roi:+7.2f} {n:6.0f} {100*nullm:+7.2f} "
            f"{100*(roi-nullm):+7.2f} {100*d165['decay']:+7.2f} "
            f"{chg:3d}/{len(steps)-1:<4d}  "
            f"{'EXACT' if ok else 'MISMATCH'}")
    assert all(np.allclose(winrows[i]["roi"],
                           ad["arms"][winrows[i]["key"]]["pooled_roi"])
               for i in range(len(winrows))), "arm reproduction failed"

    BAND = 0.0213      # D165 §7 measured family-wise burden, +2.13 ROI points
    best_net = max(r["net"] for r in winrows)
    elig = [r for r in winrows if r["net"] >= best_net - BAND]
    chosen = max(elig, key=lambda r: r["W"])
    runner = max([r for r in winrows if r is not chosen], key=lambda r: r["net"])
    say(f"\nbest net-of-null: {max(winrows, key=lambda r: r['net'])['name']} "
        f"{100*best_net:+.2f}")
    say(f"inside the {100*BAND:.2f}-point band: "
        + ", ".join(f"{r['name']}({100*r['net']:+.2f})" for r in elig))
    say(f"RULE (iv) longest window inside the band  ->  CHOSEN: {chosen['name']}")
    say(f"runner-up by net-of-null: {runner['name']} {100*runner['net']:+.2f}")
    R["window_choice"] = {"band": BAND, "rows": [
        {k: v for k, v in r.items() if k != "per"} | {"per_season_roi": r["per"]}
        for r in winrows], "chosen": chosen["name"], "runner_up": runner["name"]}

    # ============================================ STEP 2: the chosen bet sets
    steps = steps_by_w[chosen["name"]]
    scored = [seasons[s["k"]] for s in steps]
    say("\n" + "-" * 78)
    say(f"STEP 2  BET SETS — {chosen['name']}, scored track "
        f"{scored[0]}..{scored[-1]} ({len(scored)} seasons)")
    say("-" * 78)
    gid = df["game_id"].astype(str).to_numpy()
    gdate = df["game_date"].astype(str).to_numpy()
    d_signed = np.where(bet_home, st["actual"] - st["open_margin"],
                        st["open_margin"] - st["actual"])
    bets = []          # per scored season: dict(season, gids, d)
    for s in steps:
        k = s["k"]
        sel = (M[s["cfg"]] > 0) & (st["s_i"] == k)
        idx = np.where(sel)[0]
        bets.append(dict(season=seasons[k], cfg=list(map(str, keys[s["cfg"]])),
                         idx=idx, gids=gid[idx], dates=gdate[idx],
                         d=d_signed[idx],
                         home=bet_home[idx], sel_roi=s["sel_roi"],
                         sel_n=s["sel_n"]))
        say(f"  {seasons[k]}  cfg {'/'.join(map(str,keys[s['cfg']])):22s} "
            f"n={len(idx):4d}  tier-1 ROI {100*payoff[idx].mean():+7.2f}%")

    # ================================================= STEP 3: gain machinery
    panels = load_panels()
    say(f"\npanels loaded: KAG {len(panels['KAG'])} games, "
        f"ESPN {len(panels['ESPN'])} games")

    def era_of(season):
        if season in OFFSHORE_SEASONS:
            return "KAG", "MEASURED"
        if season in GAP_SEASONS:
            return "KAG", "EXTRAPOLATED"
        if season in RETAIL_MEASURED:
            return "ESPN", "MEASURED"
        return "ESPN", "EXTRAPOLATED"

    # pooled laws, cached: (panel, k, haircut, side, rngmax) -> (G, cw, keptfrac)
    pcache = {}

    def pooled(pn, k, hc, side, rngmax=None):
        key = (pn, k, hc, side, rngmax)
        if key not in pcache:
            G, W, kept = pooled_dist(panels[pn], k, hc, side, rngmax)
            pcache[key] = (G, np.cumsum(W), kept)
        return pcache[key]

    PERBET = {}          # D173: tier -> [ {season,gid,date,ev,keep,tag}, ... ]

    def score_tier(k, hc, rngmax=None, mix=None, exch=None, pb_name=None):
        """Score every scored season at tier k.

        rngmax  offside-tail exclusion (limits arm a)
        mix     stake-cap fill fraction f at the best price, 1-f at consensus (arm b)
        exch    exchange commission c: price the tier-1 handicap at (1-c)/-1
        """
        rows = []
        for b in bets:
            pn, tag = era_of(b["season"])
            P = panels[pn]
            n_meas = 0
            tot_pay = tot_w = tot_l = tot_p = 0.0
            n_kept = 0.0
            for j in range(len(b["idx"])):
                d = float(b["d"][j])
                side = "H" if b["home"][j] else "A"
                if exch is not None:
                    w_ = 1.0 if d > 0 else 0.0
                    l_ = 1.0 if d < 0 else 0.0
                    pu = 1.0 - w_ - l_
                    if pb_name is not None:
                        PERBET.setdefault(pb_name, []).append(
                            {"season": b["season"], "gid": str(b["gids"][j]),
                             "date": str(b["dates"][j]),
                             "ev": float((1.0 - exch) * w_ - l_),
                             "keep": 1.0, "tag": tag})
                    tot_pay += (1.0 - exch) * w_ - l_
                    tot_w += w_; tot_l += l_; tot_p += pu
                    n_kept += 1.0
                    continue
                if k == 1:
                    ev = (WIN if d > 0 else (0.0 if d == 0 else -1.0))
                    w_ = 1.0 if d > 0 else 0.0
                    l_ = 1.0 if d < 0 else 0.0
                    pu = 1.0 - w_ - l_
                    keep = 1.0
                else:
                    g_id = b["gids"][j]
                    if g_id in P and tag == "MEASURED":
                        v = P[g_id]
                        if rngmax is not None and (v.max() - v.min()) > rngmax:
                            continue                      # not bet at all
                        gg, ww = gain_dist(v, side, k, hc)
                        o = np.argsort(gg, kind="stable")
                        G, cw = gg[o], np.cumsum(ww[o])
                        keep = 1.0
                        n_meas += 1
                    else:
                        G, cw, kf = pooled(pn, k, hc, side, rngmax)
                        keep = kf if rngmax is not None else 1.0
                    ev, w_, l_, pu = payoff_from_dist(d, G, cw)
                if mix is not None and k > 1:
                    ev0 = (WIN if d > 0 else (0.0 if d == 0 else -1.0))
                    w0 = 1.0 if d > 0 else 0.0
                    l0 = 1.0 if d < 0 else 0.0
                    ev = mix * ev + (1 - mix) * ev0
                    w_ = mix * w_ + (1 - mix) * w0
                    l_ = mix * l_ + (1 - mix) * l0
                    pu = 1.0 - w_ - l_
                if pb_name is not None:
                    PERBET.setdefault(pb_name, []).append(
                        {"season": b["season"], "gid": str(b["gids"][j]),
                         "date": str(b["dates"][j]), "ev": float(ev),
                         "keep": float(keep), "tag": tag})
                tot_pay += keep * ev
                tot_w += keep * w_
                tot_l += keep * l_
                tot_p += keep * pu
                n_kept += keep
            rows.append(dict(season=b["season"], n=n_kept, pay=tot_pay,
                             roi=(tot_pay / n_kept if n_kept else np.nan),
                             cover=(tot_w / (tot_w + tot_l))
                             if (tot_w + tot_l) > 0 else np.nan,
                             push=tot_p, measured_frac=(n_meas / max(len(b["idx"]), 1)),
                             tag=tag))
        n = sum(r["n"] for r in rows)
        p = sum(r["pay"] for r in rows)
        w = sum(r["cover"] * (r["n"] - r["push"]) for r in rows)
        nz = sum(r["n"] - r["push"] for r in rows)
        return dict(rows=rows, n=n, pay=p, roi=p / n, cover=w / nz,
                    ci=oc.cluster_mean_t([r["roi"] for r in rows]),
                    cum=list(np.cumsum([r["pay"] for r in rows])),
                    measured_seasons=sum(1 for r in rows if r["tag"] == "MEASURED"))

    # =============================================== STEP 4: the tier ladder
    say("\n" + "-" * 78)
    say("STEP 4  EXECUTION TIERS.  raw = D163 measured ladder; hc = + D163 §10")
    say("        outlier-realism haircut (>1.5 pts off the panel median gets no weight)")
    say("-" * 78)
    say(f"{'tier':22s} {'n':>7s} {'cover%':>7s} {'ROI%':>7s} "
        f"{'K-1 CI (13 dof)':>22s} {'cum u':>8s} {'meas sns':>9s}")
    T = {}
    for k in TIERS:
        for hc in ([False] if k == 1 else [False, True]):
            nm = f"k={k}" + (" +haircut" if hc else " raw")
            r = score_tier(k, hc, pb_name=nm if os.environ.get("WF_PERBET")
                           else None)
            T[nm] = r
            say(f"{nm:22s} {r['n']:7.0f} {100*r['cover']:7.3f} {100*r['roi']:+7.2f} "
                f"[{100*r['ci']['lo']:+6.2f},{100*r['ci']['hi']:+6.2f}]{'SIG' if r['ci']['sig'] else '   '}"
                f"  {r['cum'][-1]:+8.1f} {r['measured_seasons']:5d}/14")
    for c in (0.02, 0.05):
        nm = f"exchange c={int(100*c)}%"
        r = score_tier(1, False, exch=c)
        T[nm] = r
        say(f"{nm:22s} {r['n']:7.0f} {100*r['cover']:7.3f} {100*r['roi']:+7.2f} "
            f"[{100*r['ci']['lo']:+6.2f},{100*r['ci']['hi']:+6.2f}]{'SIG' if r['ci']['sig'] else '   '}"
            f"  {r['cum'][-1]:+8.1f}   ARITHMETIC")

    # realised mean gain per tier, as an anchor against D163's published ladder
    say("\nrealised mean shop gain (pts) on the scored bet set, vs D163's ladder:")
    for k in TIERS[1:]:
        for hc in (False, True):
            gsum = gn = 0.0
            for b in bets:
                pn, tag = era_of(b["season"])
                for j in range(len(b["idx"])):
                    side = "H" if b["home"][j] else "A"
                    g_id = b["gids"][j]
                    if g_id in panels[pn] and tag == "MEASURED":
                        gg, ww = gain_dist(panels[pn][g_id], side, k, hc)
                        gsum += float((gg * ww).sum())
                    else:
                        G, cw, _ = pooled(pn, k, hc, side)
                        W = np.diff(np.concatenate([[0.0], cw]))
                        gsum += float((G * W).sum())
                    gn += 1
            say(f"  k={k} {'haircut' if hc else 'raw    '}  mean gain "
                f"{gsum/gn:+.4f} pts")

    # ============================================ STEP 5: firm-grade friction
    say("\n" + "-" * 78)
    say("STEP 5  FIRM FRICTION on the DEFAULT tier (k=5, measured, +haircut)")
    say("-" * 78)
    FR = {}
    base = T["k=5 +haircut"]
    say(f"{'arm':46s} {'n':>7s} {'ROI%':>7s} {'cum u':>8s}")
    say(f"{'DEFAULT k=5 +haircut':46s} {base['n']:7.0f} "
        f"{100*base['roi']:+7.2f} {base['cum'][-1]:+8.1f}")
    r = score_tier(5, True, rngmax=3.0)
    FR["offside_tail_excluded"] = r
    say(f"{'(a) offside tail excluded (panel range > 3.0 pts)':46s} {r['n']:7.1f} "
        f"{100*r['roi']:+7.2f} {r['cum'][-1]:+8.1f}")
    for f in (0.75, 0.5, 0.25):
        r = score_tier(5, True, mix=f)
        FR[f"stake_cap_f{f}"] = r
        say(f"{f'(b) stake cap: f={f:.2f} at best price, rest at consensus':46s} "
            f"{r['n']:7.0f} {100*r['roi']:+7.2f} {r['cum'][-1]:+8.1f}")
    r = score_tier(5, True, rngmax=3.0, mix=0.5)
    FR["both"] = r
    say(f"{'(a)+(b) both, f=0.50':46s} {r['n']:7.1f} "
        f"{100*r['roi']:+7.2f} {r['cum'][-1]:+8.1f}")

    # ============================================ STEP 5b: honesty diagnostics
    say("\n" + "-" * 78)
    say("STEP 5b  PER-TIER DIAGNOSTICS.  MDE80 = 2.802*sd/sqrt(14); K_resolve = the")
    say("         number of seasons at which a 95%% CI half-width equals the point est.")
    say("-" * 78)
    say(f"{'tier':22s} {'+sns':>5s} {'sd pp':>6s} {'MDE80':>6s} {'Kres':>5s} "
        f"{'ex-24/25 ROI':>12s} {'ex-24/25 cum':>12s} {'max |infl|':>10s} {'conv':>6s}")
    i2425 = scored.index("2024-25")
    base_cov = T["k=1 raw"]["cover"]
    for nm, r in T.items():
        per = np.array([x["roi"] for x in r["rows"]])
        npr = np.array([x["n"] for x in r["rows"]])
        pays = np.array([x["pay"] for x in r["rows"]])
        sd = float(per.std(ddof=1))
        mde = 2.802 * sd / np.sqrt(len(per))
        kres = (1.96 * sd / abs(r["roi"])) ** 2 if r["roi"] != 0 else np.inf
        keep = np.arange(len(per)) != i2425
        ex_roi = pays[keep].sum() / npr[keep].sum()
        infl = max(abs(r["roi"] - (pays[np.arange(len(per)) != j].sum()
                                   / npr[np.arange(len(per)) != j].sum()))
                   for j in range(len(per)))
        gainmap = {"k=2 raw": 0.1889, "k=2 +haircut": 0.1350, "k=5 raw": 0.3999,
                   "k=5 +haircut": 0.2852, "k=8 raw": 0.5069, "k=8 +haircut": 0.3278}
        cv = ((r["cover"] - base_cov) / (DP_PER_PT * gainmap[nm])
              if nm in gainmap else np.nan)
        r["diag"] = dict(sd=sd, mde80=mde, k_resolve=kres, ex2425_roi=ex_roi,
                         ex2425_cum=float(pays[keep].sum()),
                         max_influence=infl, conv_ratio=cv,
                         seasons_positive=int((per > 0).sum()))
        say(f"{nm:22s} {int((per>0).sum()):3d}/14 {100*sd:6.2f} {100*mde:6.2f} "
            f"{kres:5.0f} {100*ex_roi:+12.2f} {pays[keep].sum():+12.1f} "
            f"{100*infl:10.2f} {cv:6.3f}")

    # gap-season ladder sensitivity: 2018-19..2022-23 on the RETAIL ladder instead
    say("\nSENSITIVITY — the 5 undatable gap seasons (2018-19..2022-23) on the "
        "RETAIL\n  ladder instead of the conservative offshore one:")
    global GAP_SEASONS
    _gap = GAP_SEASONS
    GAP_SEASONS = ()
    RET = {}
    for k in (5, 8):
        for hc in (True, False):
            rr = score_tier(k, hc)
            RET[f"gapretail_k{k}_{'hc' if hc else 'raw'}"] = rr
            say(f"  k={k} {'haircut' if hc else 'raw    '}  ROI "
                f"{100*rr['roi']:+6.2f}%  cum {rr['cum'][-1]:+7.1f}u   "
                f"(conservative: {100*T[f'k={k}' + (' +haircut' if hc else ' raw')]['roi']:+6.2f}%, "
                f"{T[f'k={k}' + (' +haircut' if hc else ' raw')]['cum'][-1]:+7.1f}u)")
    GAP_SEASONS = _gap
    R["gap_retail_sensitivity"] = RET

    # ============================================== STEP 6: the season path
    say("\n" + "-" * 78)
    say("STEP 6  THE PER-SEASON PATH — the headline object")
    say("-" * 78)
    hdr = (f"{'season':9s} {'cfg':22s} "
           f"{'n':>4s} {'cov%':>6s} {'ROI%':>7s} {'cum u':>7s} | "
           f"{'n':>5s} {'cov%':>6s} {'ROI%':>7s} {'cum u':>7s} | "
           f"{'ROI%':>7s} {'cum u':>7s} | tag")
    say(f"{'':32s} {'---- 1 BOOK (retail ref) ----':>28s}   "
        f"{'-- k=5 +haircut (DEFAULT) --':>30s}   {'k=8 raw (bound)':>16s}")
    say(hdr)
    t1, t5, t8 = T["k=1 raw"], T["k=5 +haircut"], T["k=8 raw"]
    for i, b in enumerate(bets):
        a, c, e = t1["rows"][i], t5["rows"][i], t8["rows"][i]
        say(f"{b['season']:9s} {'/'.join(b['cfg']):22s} "
            f"{a['n']:4.0f} {100*a['cover']:6.2f} {100*a['roi']:+7.2f} "
            f"{t1['cum'][i]:+7.1f} | "
            f"{c['n']:5.0f} {100*c['cover']:6.2f} {100*c['roi']:+7.2f} "
            f"{t5['cum'][i]:+7.1f} | "
            f"{100*e['roi']:+7.2f} {t8['cum'][i]:+7.1f} | {c['tag']}")

    R["chosen"] = chosen["name"]
    R["scored_seasons"] = scored
    R["cfgs"] = [b["cfg"] for b in bets]
    R["tiers"] = {k: {kk: vv for kk, vv in v.items()} for k, v in T.items()}
    R["friction"] = FR
    R["runner_up_paths"] = {r["name"]: r["per"] for r in winrows}
    if os.environ.get("WF_PERBET"):
        pbo = ROOT / "data" / f"wf_perbet{_TAG}.json"
        json.dump(PERBET, open(pbo, "w"))
        say(f"wrote {pbo}  ({sum(len(v) for v in PERBET.values())} bet-rows)")
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    (ROOT / "data" / "logs").mkdir(exist_ok=True)
    (ROOT / "data" / "logs" / f"wf_equity{_TAG}.log").write_text("\n".join(LOG))
    say(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
