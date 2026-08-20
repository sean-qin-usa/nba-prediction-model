#!/usr/bin/env python3
"""EL-ERALOCAL — does an ERA-LOCAL model beat the GLOBAL one when each is
scored on held-out seasons WITHIN its own era?

PRE-REGISTERED in data/eralocal_prereg.md sha256
7f0fa5dae1edda2cab90b35b5e158bcbb36631a094d25ff5b519ef6b23888aea

REUSES, DOES NOT REBUILD
  scripts/oc_capacity.py      D164 harness (frame loader, 600-cell masks, agg,
                              cluster-mean t, the within-date permutation null)
  scripts/wf_equity.py        D166 machinery (select, gain_dist, pooled_dist,
                              payoff_from_dist, load_panels, era map, haircut)
  data/sl_frames/frame_V*.csv.gz   D168's six paired model-variant frames
  data/mb_panel_{kag,espn}.json    D163's measured operator panels

ARMS (all scored on the SAME within-era held-out blocks)
  EL  era-local MODEL (6 rungs) x era-local CONFIG (600 cells) = 3600 cells,
      selected on that era's SELECTION BLOCK only
  EC  shipped V0 + era-local CONFIG only
  GF  shipped V0 + config selected on ALL prior history, frozen at the era
      boundary (the matched comparator)
  GW  shipped V0 + config re-selected walk-forward each held-out season
      (D166's literal procedure)
  WLADDER  V0 + config selected on the W seasons immediately preceding each
      held-out season, W in {1,2,3,5,ALL} — the capacity-vs-window mechanism

Every arm is re-run through the IDENTICAL procedure on 200 within-date
permutations (seed 20260804) so every arm carries its OWN null.

DIAGNOSTIC.  Nothing ships.  No default changed.  The DB is never opened.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import oc_capacity as oc  # noqa: E402
import wf_equity as wq  # noqa: E402

OUT = ROOT / "data" / "el_eralocal.json"
LOG: list[str] = []
WIN = 100.0 / 110.0
SEED = 20260804
NDRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
GUARD_PER_SEASON = 100.0
PREREG = "7f0fa5dae1edda2cab90b35b5e158bcbb36631a094d25ff5b519ef6b23888aea"

VARIANTS = ["V0_FULL", "V1_noTANK", "V2_noTANK_noBRIDGE",
            "V3_noTANK_noBRIDGE_noCARRY", "V4_STRIPPED", "V5_FF_ONLY"]
VSHORT = ["V0 FULL", "V1 -tank", "V2 -tank-bridge", "V3 -tank-bridge-carry",
          "V4 STRIPPED", "V5 4F+home"]
NV = len(VARIANTS)

# ---- pre-registered eras (data/eralocal_prereg.md §1), season INDEX form
ERAS = [
    ("E_OLD", "2007-08..2013-14", [0, 1, 2, 3, 4], [5, 6]),
    ("E_MID", "2014-15..2018-19", [7, 8, 9], [10, 11]),
    ("E_MOD", "2021-22..2025-26", [14, 15, 16], [17, 18]),
]
ERA_SENS = ("E_MOD_E5SPLIT", "2021-22..2025-26 split at the E5 boundary",
            [14, 15], [16, 17, 18])
WLADDER = [1, 2, 3, 5, None]          # None = all prior history


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------- execution tier
PANELS = None
_pcache: dict = {}
_gcache: dict = {}


def era_of(season):
    if season in wq.OFFSHORE_SEASONS:
        return "KAG", "MEASURED"
    if season in wq.GAP_SEASONS:
        return "KAG", "EXTRAPOLATED"
    if season in wq.RETAIL_MEASURED:
        return "ESPN", "MEASURED"
    return "ESPN", "EXTRAPOLATED"


def pooled(pn, k, hc, side):
    key = (pn, k, hc, side)
    if key not in _pcache:
        G, W, _ = wq.pooled_dist(PANELS[pn], k, hc, side)
        _pcache[key] = (G, np.cumsum(W))
    return _pcache[key]


def bet_law(season, gid, side, k, hc):
    """D166's exact per-game or pooled shop-gain law for one bet."""
    if k == 1:
        return None, None
    pn, tag = era_of(season)
    if tag == "MEASURED" and gid in PANELS[pn]:
        key = (gid, side, k, hc)
        r = _gcache.get(key)
        if r is None:
            gg, ww = wq.gain_dist(PANELS[pn][gid], side, k, hc)
            o = np.argsort(gg, kind="stable")
            r = (gg[o], np.cumsum(ww[o]))
            _gcache[key] = r
        return r
    return pooled(pn, k, hc, side)


# ------------------------------------------------------------------ the frames
def load_all():
    frames, order = [], None
    files = sorted((ROOT / "data" / "sl_frames").glob("frame_V*.csv.gz"))
    byname = {p.name.split("frame_")[1].rsplit(".csv.gz")[0]: p for p in files}
    st = gid = None
    for name in VARIANTS:
        oc.FRAME = byname[name]
        df, seasons = oc.load()
        if order is None:
            order = df["game_id"].to_numpy()
            st = oc.build_static(df)
            gid = df["gid_str"].astype(str).to_numpy()
            SEAS = seasons
        else:
            assert (df["game_id"].to_numpy() == order).all(), \
                f"{name} game ordering differs"
        frames.append((df["m_us"].to_numpy(float), df["p_us"].to_numpy(float)))
    return frames, st, gid, SEAS


def build(frames, st):
    """payoff / masks / per-season aggregates for every variant."""
    P, MB, CNT, PAY, BH, DSG, WINF, PSH = [], [], [], [], [], [], [], []
    for m_us, p_us in frames:
        payoff, M, keys, win, push, bh = oc.payoff_and_masks(m_us, p_us, st)
        cnt, pay = oc.agg(M, payoff, st)
        Mw = M * win.astype(np.float32)
        Mp = M * push.astype(np.float32)
        P.append(payoff)
        MB.append(M > 0)
        CNT.append(cnt)
        PAY.append(pay)
        BH.append(bh)
        DSG.append(np.where(bh, st["actual"] - st["open_margin"],
                            st["open_margin"] - st["actual"]))
        WINF.append(Mw @ st["S"])
        PSH.append(Mp @ st["S"])
        del M, Mw, Mp
    return dict(payoff=P, M=MB, cnt=CNT, pay=PAY, bet_home=BH, d=DSG,
                wcnt=WINF, pcnt=PSH, keys=keys)


# -------------------------------------------------------------------- scoring
def score_season(B, v, cfg, s_idx, st, gid, seasons, k, hc):
    """Score one (variant, config) on one season at execution tier k."""
    sel = B["M"][v][cfg] & (st["s_i"] == s_idx)
    idx = np.where(sel)[0]
    season = seasons[s_idx]
    d = B["d"][v]
    bh = B["bet_home"][v]
    tot = w = l = p = 0.0
    if k == 1:
        dd = d[idx]
        w = float((dd > 0).sum())
        l = float((dd < 0).sum())
        p = float((dd == 0).sum())
        tot = WIN * w - l
    else:
        for i in idx:
            G, cw = bet_law(season, gid[i], "H" if bh[i] else "A", k, hc)
            ev, w_, l_, pu = wq.payoff_from_dist(float(d[i]), G, cw)
            tot += ev
            w += w_
            l += l_
            p += pu
    n = float(len(idx))
    return dict(season=season, n=n, pay=tot, roi=(tot / n) if n else np.nan,
                w=w, l=l, push=p)


def pooled_rows(rows):
    n = sum(r["n"] for r in rows)
    p = sum(r["pay"] for r in rows)
    w = sum(r["w"] for r in rows)
    l = sum(r["l"] for r in rows)
    per = [r["roi"] for r in rows]
    return dict(n=n, pay=p, roi=(p / n) if n else np.nan,
                cover=(w / (w + l)) if (w + l) else np.nan,
                per_season_roi=per, ci=oc.cluster_mean_t(per),
                seasons=[r["season"] for r in rows])


# ------------------------------------------------------------------ selection
def sel_stats(B, vlist, blocks):
    """Flattened (variant, config) selection ROI over a block of seasons."""
    c = np.concatenate([B["cnt"][v][:, blocks].sum(1) for v in vlist])
    p = np.concatenate([B["pay"][v][:, blocks].sum(1) for v in vlist])
    return c, p


def pick(B, vlist, blocks):
    c, p = sel_stats(B, vlist, blocks)
    best = wq.select(c, p, GUARD_PER_SEASON * len(blocks))
    if best is None:
        return None
    v = vlist[best // 600]
    cfg = best % 600
    return dict(v=v, cfg=cfg, sel_roi=float(p[best] / c[best]),
                sel_n=float(c[best]))


# ---------------------------------------------------------------- the arms
def run_arms(B, st, gid, seasons, tiers):
    """Every arm, every era.  Returns nested dict keyed arm -> era -> tier."""
    out = {}
    allv = list(range(NV))
    for arm in ("EL", "EC", "GF", "GW"):
        out[arm] = {}
        for ename, ewin, sel, hold in ERAS + [ERA_SENS]:
            picks = []
            if arm == "EL":
                pk = pick(B, allv, sel)
                picks = [(pk, s) for s in hold]
            elif arm == "EC":
                pk = pick(B, [0], sel)
                picks = [(pk, s) for s in hold]
            elif arm == "GF":
                pk = pick(B, [0], list(range(hold[0])))
                picks = [(pk, s) for s in hold]
            else:                       # GW: re-select before each season
                picks = [(pick(B, [0], list(range(s))), s) for s in hold]
            if any(p is None for p, _ in picks):
                out[arm][ename] = None
                continue
            rec = {"picks": [{"season": seasons[s], "v": VARIANTS[p["v"]],
                              "cfg": list(map(str, B["keys"][p["cfg"]])),
                              "sel_roi": p["sel_roi"], "sel_n": p["sel_n"]}
                             for p, s in picks],
                   "v_idx": [p["v"] for p, _ in picks],
                   "sel_roi": float(np.mean([p["sel_roi"] for p, _ in picks])),
                   "sel_n": float(np.mean([p["sel_n"] for p, _ in picks]))}
            for k, hc, tag in tiers:
                rows = [score_season(B, p["v"], p["cfg"], s, st, gid,
                                     seasons, k, hc) for p, s in picks]
                rec[tag] = pooled_rows(rows)
                rec[tag]["rows"] = rows
            out[arm][ename] = rec
    # ---- the capacity ladder: window length W, on the SAME held-out seasons
    hold_all = [s for _n, _w, _sl, hd in ERAS for s in hd]
    out["WLADDER"] = {}
    for W in WLADDER:
        picks = []
        for s in hold_all:
            lo = 0 if W is None else max(0, s - W)
            blocks = list(range(lo, s))
            picks.append((pick(B, [0], blocks), s))
        if any(p is None for p, _ in picks):
            out["WLADDER"][str(W)] = None
            continue
        rec = {"sel_roi": float(np.mean([p["sel_roi"] for p, _ in picks])),
               "sel_n": float(np.mean([p["sel_n"] for p, _ in picks])),
               "n_distinct_cfg": len({p["cfg"] for p, _ in picks})}
        for k, hc, tag in tiers:
            rows = [score_season(B, p["v"], p["cfg"], s, st, gid, seasons,
                                 k, hc) for p, s in picks]
            rec[tag] = pooled_rows(rows)
        out["WLADDER"][str(W)] = rec
    return out


# ------------------------------------------------------------------ the null
def permute(st, rng, order, bounds):
    perm = order.copy()
    for gi in range(len(bounds) - 1):
        a, b = bounds[gi], bounds[gi + 1]
        if b - a > 1:
            perm[a:b] = rng.permutation(perm[a:b])
        idx = None
    idx = np.empty_like(perm)
    idx[order] = perm
    return idx


def main():
    global PANELS
    t00 = time.time()
    say("=" * 92)
    say("EL-ERALOCAL — era-local vs global selection, held out WITHIN era")
    say(f"prereg sha256 {PREREG}")
    say("=" * 92)

    frames, st, gid, seasons = load_all()
    say(f"frames: {NV} variants x {st['n']} games x {len(seasons)} seasons")
    PANELS = wq.load_panels()
    say(f"panels: KAG {len(PANELS['KAG'])} games, ESPN {len(PANELS['ESPN'])}")

    B = build(frames, st)
    TIERS = [(1, False, "t1"), (5, True, "t5")]

    # ---------------------------------------------------------- ANCHORS
    ref = 0
    cnt0, pay0 = B["cnt"][0], B["pay"][0]
    a_roi = 100 * pay0[ref].sum() / cnt0[ref].sum()
    a_cov = 100 * B["wcnt"][0][ref].sum() / (B["wcnt"][0][ref].sum()
                                             + (cnt0[ref].sum()
                                                - B["wcnt"][0][ref].sum()
                                                - B["pcnt"][0][ref].sum()))
    say(f"\nANCHOR 1  D162 POOL19  ROI {a_roi:+.3f}%  cover {a_cov:.3f}%  "
        f"pushes {int(B['pcnt'][0][ref].sum())}  n {int(cnt0[ref].sum())}")
    assert abs(a_roi + 3.245) < 0.002 and abs(a_cov - 50.654) < 0.002

    # ANCHOR 2: D166/D168 firm default on V0 through the unchanged loop
    steps = wq.arm_window(cnt0, pay0, len(seasons), None, wq.K_COMMON)
    rows1, rows5 = [], []
    for s in steps:
        rows1.append(score_season(B, 0, s["cfg"], s["k"], st, gid, seasons,
                                  1, False))
        rows5.append(score_season(B, 0, s["cfg"], s["k"], st, gid, seasons,
                                  5, True))
    p1, p5 = pooled_rows(rows1), pooled_rows(rows5)
    say(f"ANCHOR 2  D166 firm default on V0: n {p5['n']:.0f}  "
        f"1-book ROI {100*p1['roi']:+.2f}%  k=5+HC ROI {100*p5['roi']:+.2f}%  "
        f"cover {100*p5['cover']:.3f}%  cum {p5['pay']:+.1f}u  "
        f"CI [{100*p5['ci']['lo']:+.2f},{100*p5['ci']['hi']:+.2f}]")
    assert abs(100 * p5["roi"] - 3.54) < 0.02 and abs(p5["n"] - 1553) < 0.5

    R = {"prereg_sha256": PREREG, "seasons": seasons,
         "n_games": int(st["n"]), "variants": VARIANTS,
         "eras": [{"name": n, "window": w,
                   "selection": [seasons[i] for i in s],
                   "holdout": [seasons[i] for i in h]}
                  for n, w, s, h in ERAS + [ERA_SENS]],
         "anchors": {"pool19_roi": a_roi, "pool19_cover": a_cov,
                     "d166_default_roi": 100 * p5["roi"],
                     "d166_default_n": p5["n"],
                     "d166_1book_roi": 100 * p1["roi"]},
         "d166_reference": {"t1": {k: p1[k] for k in
                                   ("n", "roi", "cover", "per_season_roi",
                                    "seasons")},
                            "t5": {k: p5[k] for k in
                                   ("n", "roi", "cover", "per_season_roi",
                                    "seasons")}}}

    # ------------------------------------------------------------- REAL ARMS
    say("\n" + "-" * 92)
    say("REAL ARMS")
    say("-" * 92)
    real = run_arms(B, st, gid, seasons, TIERS)
    R["real"] = real

    for arm in ("EL", "EC", "GF", "GW"):
        say(f"\n{arm}")
        for ename, _w, _s, _h in ERAS + [ERA_SENS]:
            r = real[arm][ename]
            if r is None:
                say(f"  {ename:14s}  NO ELIGIBLE CELL")
                continue
            pk = " | ".join(f"{p['season']}:{p['v'].split('_')[0]}/"
                            f"{'/'.join(p['cfg'])}" for p in r["picks"])
            say(f"  {ename:14s} selROI {100*r['sel_roi']:+6.2f}% "
                f"(n {r['sel_n']:6.0f})  held-out n {r['t1']['n']:5.0f}  "
                f"1bk {100*r['t1']['roi']:+7.2f}%  k5+HC "
                f"{100*r['t5']['roi']:+7.2f}%  cover {100*r['t5']['cover']:6.2f}%")
            say(f"                 picks: {pk}")

    say("\nWLADDER (V0, config from the W seasons before each held-out season)")
    for W in WLADDER:
        r = real["WLADDER"][str(W)]
        lbl = "ALL" if W is None else str(W)
        if r is None:
            say(f"  W={lbl:3s}  NO ELIGIBLE CELL")
            continue
        say(f"  W={lbl:3s} selROI {100*r['sel_roi']:+6.2f}% "
            f"(n {r['sel_n']:6.0f})  held-out 1bk {100*r['t1']['roi']:+7.2f}% "
            f" k5+HC {100*r['t5']['roi']:+7.2f}%  ncfg {r['n_distinct_cfg']}")

    # ------------------------------------------------------------------ NULL
    say("\n" + "-" * 92)
    say(f"NULL — {NDRAWS} within-date permutations of (m_us, p_us), seed {SEED},"
        " identical procedure")
    say("-" * 92)
    rng = np.random.default_rng(SEED)
    slate = st["slate"]
    order = np.argsort(slate, kind="stable")
    bounds = np.searchsorted(slate[order], np.arange(slate.max() + 2))
    draws = []
    t0 = time.time()
    for d in range(NDRAWS):
        idx = permute(st, rng, order, bounds)
        fr = [(m[idx], p[idx]) for m, p in frames]
        Bn = build(fr, st)
        draws.append(run_arms(Bn, st, gid, seasons, TIERS))
        del Bn, fr
        if (d + 1) % 20 == 0:
            el = time.time() - t0
            say(f"  draw {d+1}/{NDRAWS}  ({el:.0f}s, "
                f"eta {el/(d+1)*(NDRAWS-d-1):.0f}s)")
    R["null_draws"] = NDRAWS
    R["null_seed"] = SEED

    # --------------------------------------------------------- summarise null
    def dist(x):
        x = np.asarray([v for v in x if np.isfinite(v)], float)
        if not len(x):
            return None
        return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1
                else 0.0, "p05": float(np.percentile(x, 5)),
                "p50": float(np.percentile(x, 50)),
                "p95": float(np.percentile(x, 95)),
                "min": float(x.min()), "max": float(x.max()), "n": len(x)}

    NUL = {}
    for arm in ("EL", "EC", "GF", "GW"):
        NUL[arm] = {}
        for ename, _w, _s, _h in ERAS + [ERA_SENS]:
            cell = {}
            for tag in ("t1", "t5"):
                cell[tag] = dist([dd[arm][ename][tag]["roi"] for dd in draws
                                  if dd[arm][ename]])
                cell[tag + "_perseason"] = np.array(
                    [dd[arm][ename][tag]["per_season_roi"] for dd in draws
                     if dd[arm][ename]], float).mean(0).tolist()
            cell["sel_roi"] = dist([dd[arm][ename]["sel_roi"] for dd in draws
                                    if dd[arm][ename]])
            NUL[arm][ename] = cell
    NUL["WLADDER"] = {}
    for W in WLADDER:
        cell = {}
        for tag in ("t1", "t5"):
            cell[tag] = dist([dd["WLADDER"][str(W)][tag]["roi"] for dd in draws
                              if dd["WLADDER"][str(W)]])
        cell["sel_roi"] = dist([dd["WLADDER"][str(W)]["sel_roi"]
                                for dd in draws if dd["WLADDER"][str(W)]])
        NUL["WLADDER"][str(W)] = cell

    # ---- POOLED across the three primary eras, real and null, per tier
    def pool_eras(res, tag, enames):
        rows = []
        for e in enames:
            r = res[e]
            if r is None:
                continue
            rows += r[tag]["rows"] if "rows" in r[tag] else []
        return pooled_rows(rows) if rows else None

    PRIM = [e[0] for e in ERAS]
    R["pooled_real"] = {}
    for arm in ("EL", "EC", "GF", "GW"):
        R["pooled_real"][arm] = {t: pool_eras(real[arm], t, PRIM)
                                 for t in ("t1", "t5")}

    # pooled null: per draw, pool the three eras then take the distribution
    def pooled_null(arm, tag):
        vals, pers = [], []
        for dd in draws:
            rows = []
            for e in PRIM:
                if dd[arm][e]:
                    rows += dd[arm][e][tag]["rows"]
            if rows:
                pr = pooled_rows(rows)
                vals.append(pr["roi"])
                pers.append(pr["per_season_roi"])
        return dist(vals), np.array(pers, float)

    R["pooled_null"] = {}
    NULL_PER = {}
    for arm in ("EL", "EC", "GF", "GW"):
        R["pooled_null"][arm] = {}
        NULL_PER[arm] = {}
        for tag in ("t1", "t5"):
            d_, per = pooled_null(arm, tag)
            R["pooled_null"][arm][tag] = d_
            NULL_PER[arm][tag] = per

    # ---- PRIMARY STATISTIC: paired per-season delta EL - GF, net of null
    R["paired"] = {}
    for tag in ("t1", "t5"):
        blk = {}
        for a, b in (("EL", "GF"), ("EC", "GF"), ("EL", "EC"), ("EL", "GW"),
                     ("EC", "GW")):
            ra = R["pooled_real"][a][tag]["per_season_roi"]
            rb = R["pooled_real"][b][tag]["per_season_roi"]
            dl = [x - y for x, y in zip(ra, rb)]
            ci = oc.cluster_mean_t(dl)
            na = NULL_PER[a][tag]
            nb = NULL_PER[b][tag]
            nd = (na - nb).mean(1)          # per-draw mean paired delta
            blk[f"{a}_minus_{b}"] = {
                "per_season": dl, "ci": ci,
                "mde80": float(2.802 * np.std(dl, ddof=1) / np.sqrt(len(dl))),
                "null": dist(nd.tolist()),
                "net_of_null": float(ci["mean"] - float(np.mean(nd))),
                "p_own_null": float(np.mean(nd >= ci["mean"]))}
        R["paired"][tag] = blk
    R["null_summary"] = NUL

    # ---- which MODEL rung does era-local selection pick, real vs null?
    VP = {}
    for ename, _w, _s, _h in ERAS + [ERA_SENS]:
        cr = real["EL"][ename]["v_idx"][0] if real["EL"][ename] else None
        hist = [0] * NV
        for dd in draws:
            if dd["EL"][ename]:
                hist[dd["EL"][ename]["v_idx"][0]] += 1
        VP[ename] = {"real": VARIANTS[cr] if cr is not None else None,
                     "null_hist": hist}
    R["variant_picks"] = VP
    say("\nWHICH MODEL RUNG DOES ERA-LOCAL SELECTION PICK?")
    say(f"  {'era':14s} {'REAL':28s} " + " ".join(f"{v:>7s}" for v in VSHORT))
    for ename, _w, _s, _h in ERAS + [ERA_SENS]:
        h = VP[ename]["null_hist"]
        say(f"  {ename:14s} {str(VP[ename]['real']):28s} "
            + " ".join(f"{100.0*x/max(1,sum(h)):6.1f}%" for x in h))

    # ---- per-era paired point estimates (K=2, no interval is claimed)
    say("\nPER-ERA PAIRED POINT ESTIMATES (K=2 held-out seasons: NO interval "
        "is claimable and none is quoted)")
    PE = {}
    for tag in ("t1", "t5"):
        PE[tag] = {}
        for ename, _w, _s, _h in ERAS + [ERA_SENS]:
            cell = {}
            for a, b in (("EL", "GF"), ("EC", "GF"), ("EL", "EC")):
                if real[a][ename] is None or real[b][ename] is None:
                    continue
                ra = real[a][ename][tag]["per_season_roi"]
                rb = real[b][ename][tag]["per_season_roi"]
                dl = float(np.mean([x - y for x, y in zip(ra, rb)]))
                nd = np.array([np.mean(np.array(dd[a][ename][tag]["per_season_roi"])
                                       - np.array(dd[b][ename][tag]["per_season_roi"]))
                               for dd in draws if dd[a][ename] and dd[b][ename]])
                cell[f"{a}_minus_{b}"] = {
                    "delta": dl, "null_mean": float(nd.mean()),
                    "null_p95": float(np.percentile(nd, 95)),
                    "net": float(dl - nd.mean()),
                    "p": float(np.mean(nd >= dl))}
            PE[tag][ename] = cell
    R["per_era_paired"] = PE
    for tag in ("t5", "t1"):
        say(f"  [{tag}]")
        for ename in PE[tag]:
            for kk, vv in PE[tag][ename].items():
                say(f"    {ename:14s} {kk:12s} {100*vv['delta']:+7.2f}  "
                    f"null {100*vv['null_mean']:+6.2f} "
                    f"(p95 {100*vv['null_p95']:+6.2f})  "
                    f"NET {100*vv['net']:+6.2f}  p={vv['p']:.3f}")

    # ------------------------------------------------------------- REPORT
    say("\n" + "=" * 92)
    say("HELD-OUT RESULTS, PER ERA, ALL ARMS, NET OF EACH ARM'S OWN NULL")
    say("=" * 92)
    for tag, tname in (("t5", "FIRM DEFAULT k=5 measured + haircut"),
                       ("t1", "1-BOOK k=1")):
        say(f"\n### {tname}")
        say(f"{'era':14s} {'arm':4s} {'n':>6s} {'ROI%':>8s} {'cover%':>7s} "
            f"{'null%':>8s} {'net':>7s} {'p':>6s} {'selROI%':>8s} {'decay':>7s}")
        for ename, _w, _s, _h in ERAS + [ERA_SENS]:
            for arm in ("EL", "EC", "GF", "GW"):
                r = real[arm][ename]
                if r is None:
                    continue
                nu = NUL[arm][ename][tag]
                roi = 100 * r[tag]["roi"]
                nl = 100 * nu["mean"]
                pv = float(np.mean([dd[arm][ename][tag]["roi"] >= r[tag]["roi"]
                                    for dd in draws if dd[arm][ename]]))
                dec = 100 * (r["sel_roi"] - r["t1"]["roi"])
                say(f"{ename:14s} {arm:4s} {r[tag]['n']:6.0f} {roi:+8.2f} "
                    f"{100*r[tag]['cover']:7.2f} {nl:+8.2f} {roi-nl:+7.2f} "
                    f"{pv:6.3f} {100*r['sel_roi']:+8.2f} {dec:+7.2f}")
        say("")
        say(f"{'POOLED (3 eras)':14s} {'arm':4s} {'n':>6s} {'ROI%':>8s} "
            f"{'cover%':>7s} {'null%':>8s} {'net':>7s} {'5-dof CI':>19s} "
            f"{'MDE80':>6s}")
        for arm in ("EL", "EC", "GF", "GW"):
            pr = R["pooled_real"][arm][tag]
            nu = R["pooled_null"][arm][tag]
            per = pr["per_season_roi"]
            say(f"{'':14s} {arm:4s} {pr['n']:6.0f} {100*pr['roi']:+8.2f} "
                f"{100*pr['cover']:7.2f} {100*nu['mean']:+8.2f} "
                f"{100*(pr['roi']-nu['mean']):+7.2f} "
                f"[{100*pr['ci']['lo']:+8.2f},{100*pr['ci']['hi']:+8.2f}] "
                f"{100*2.802*np.std(per,ddof=1)/np.sqrt(len(per)):6.2f}")
        say("\n  PAIRED per-season deltas over the 6 held-out seasons "
            "(5 dof), each net of the SAME paired delta under the null:")
        for kk, vv in R["paired"][tag].items():
            say(f"    {kk:14s} {100*vv['ci']['mean']:+7.2f} "
                f"[{100*vv['ci']['lo']:+7.2f},{100*vv['ci']['hi']:+7.2f}]"
                f"{'  SIG' if vv['ci']['sig'] else '  ns '}"
                f"  null {100*vv['null']['mean']:+6.2f} "
                f"(p95 {100*vv['null']['p95']:+6.2f})  "
                f"NET {100*vv['net_of_null']:+6.2f}  p={vv['p_own_null']:.3f}"
                f"  MDE80 {100*vv['mde80']:.2f}")

    say("\n" + "=" * 92)
    say("CAPACITY — null level and decay by SELECTION-WINDOW LENGTH "
        "(V0, 600 cells, the same 6 held-out seasons)")
    say("=" * 92)
    say(f"{'W (seasons)':12s} {'sel n':>8s} {'in-window%':>11s} "
        f"{'held-out%':>10s} {'DECAY':>7s} | {'null in-win%':>13s} "
        f"{'null held%':>11s} {'null DECAY':>11s} {'ncfg':>5s}")
    for W in WLADDER:
        r = real["WLADDER"][str(W)]
        nu = NUL["WLADDER"][str(W)]
        if r is None:
            continue
        lbl = "ALL" if W is None else str(W)
        say(f"{lbl:12s} {r['sel_n']:8.0f} {100*r['sel_roi']:+11.2f} "
            f"{100*r['t1']['roi']:+10.2f} "
            f"{100*(r['sel_roi']-r['t1']['roi']):+7.2f} | "
            f"{100*nu['sel_roi']['mean']:+13.2f} "
            f"{100*nu['t1']['mean']:+11.2f} "
            f"{100*(nu['sel_roi']['mean']-nu['t1']['mean']):+11.2f} "
            f"{r['n_distinct_cfg']:5d}")

    R["runtime_s"] = round(time.time() - t00, 1)
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    (ROOT / "data" / "logs").mkdir(exist_ok=True)
    (ROOT / "data" / "logs" / "el_eralocal.log").write_text("\n".join(LOG))
    say(f"\nwrote {OUT}  ({R['runtime_s']}s)")


if __name__ == "__main__":
    main()
