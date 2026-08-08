#!/usr/bin/env python3
"""SL-SCORE — the three questions on D166's headline.

Q1  STRUCTURAL LOOKAHEAD.  Re-run D166's walk-forward loop on progressively
    more primitive MODEL STRUCTURES and report what fraction of +3.54%
    survives when the terms whose gates are most era-specific are removed.
Q2  Is 2024-25 special in the MODEL's own accuracy, or only in the betting
    outcome?  Full leave-one-season-out influence profile, all 14 seasons.
Q3  Confidence-scaled staking on the walk-forward bet stream, every fitted
    parameter estimated WALK-FORWARD.

PRE-REGISTERED in data/structural_prereg.md sha256
76d79823bd45e0e6d1339ce7d22cc3b711ece3ec7ee5f727f42f4e526b06cddc

REUSES, DOES NOT REBUILD:
  scripts/oc_capacity.py   D164 harness (frame loader, 600-cell masks, agg, CI)
  scripts/wf_equity.py     D166 machinery (arm_window, gain_dist, pooled_dist,
                           payoff_from_dist, load_panels, era map, haircut)
  data/mb_panel_*.json     D163's measured operator panels
  data/ats19_frame.csv.gz  D162's frame (the V0 anchor and the odds columns)

DIAGNOSTIC.  Nothing ships.  No default changed.  DB never opened here.
"""
from __future__ import annotations

import json
import sys
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

import os  # noqa: E402
# PATH OVERRIDES ONLY (D173 re-run on the D170/D171 backfilled data).  Defaults
# are byte-identical to D168's; no variant, rung, stake rule or constant moves.
_TAG = os.environ.get("SL_SCORE_TAG", "")
SL_COMP_CSV = ROOT / "data" / f"sl_components{os.environ.get('SL_TAG','')}.csv.gz"
SL_COMP_JSON = ROOT / "data" / f"sl_components{os.environ.get('SL_TAG','')}.json"
OUT = ROOT / "data" / f"sl_score{_TAG}.json"
SCRATCH = ROOT / "data" / f"sl_frames{_TAG}"
LOG = []
WIN = 100.0 / 110.0
SCALE = 7.2
SD_MARGIN = 12.574            # D162 §6 measured sd(actual - opening line)
SEED = 20260804
NDRAW_MC = 200
KELLY_FRAC = 0.25             # f4_shrinkage.KELLY_FRAC, declared convention
KELLY_BANK = 100.0            # f4_shrinkage.BANKROLL_REF
STAKE_CAP = 3.0               # declared convention (prereg §3)
D121_OPEN_A = -0.0037733442091709493      # D121 registered OPEN calibration
D121_OPEN_B = +0.5684830302091815         # (fit on 2022-23..2025-26 = LOOKAHEAD)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ variants
# margin = 0.5*ff + 0.5*cm + he + b2b + tank (+ bridge residual)
VARIANTS = [
    ("V0_FULL", "shipped stack, verbatim",
     lambda c: c["full"]),
    ("V1_noTANK", "- D73 tank term",
     lambda c: c["full"] - c["tk"]),
    ("V2_noTANK_noBRIDGE", "- D73 tank - D84A October bridge",
     lambda c: c["full"] - c["tk"] - c["bridge"]),
    ("V3_noTANK_noBRIDGE_noCARRY", "- D73 tank - D84A bridge - D62 carry",
     lambda c: 0.5 * c["fm_nocarry"] + 0.5 * c["cm"] + c["he"] + c["b2b"]),
    ("V4_STRIPPED", "four-factors + composition + home constant ONLY",
     lambda c: 0.5 * c["fm_nocarry"] + 0.5 * c["cm"] + c["he"]),
    ("V5_FF_ONLY", "four-factors + home constant (no composition, no blend)",
     lambda c: c["fm_nocarry"] + c["he"]),
    ("V6_NOHOME", "V4 with NO home term at all (sanity floor)",
     lambda c: 0.5 * c["fm_nocarry"] + 0.5 * c["cm"]),
]


def build_frames():
    """One frame per variant, identical games/ordering, only m_us/p_us differ."""
    SCRATCH.mkdir(exist_ok=True)
    comp = pd.read_csv(SL_COMP_CSV, dtype={"game_id": str})
    ats = pd.read_csv(oc.FRAME)
    ats["game_id"] = ats["game_id"].astype(np.int64)
    comp["gid_int"] = comp["game_id"].astype(np.int64)
    keep = ats[["season", "game_id", "game_date", "open_margin",
                "margin_actual", "p_us", "m_us"]].rename(
        columns={"p_us": "p_us_ref", "m_us": "m_us_ref"})
    m = keep.merge(comp.drop(columns=["game_date", "season"]),
                   left_on="game_id", right_on="gid_int", how="left",
                   suffixes=("", "_c"))
    assert len(m) == len(ats), "component join fanned out"
    assert m["full"].notna().all(), "component rows missing for some games"
    dp = np.abs(m["p_us_ref"].to_numpy() - m["p_us"].to_numpy()).max()
    say(f"ANCHOR component pass vs D162 frame: max |dp| = {dp:.3e} "
        f"on {len(m)} games  {'EXACT' if dp < 1e-9 else '*** MISMATCH ***'}")
    assert dp < 1e-9

    # the game_id string form the D163 panels are keyed by
    m["gid_str"] = m["game_id_c"].astype(str)
    paths = {}
    for name, _desc, fn in VARIANTS:
        mg = fn(m).to_numpy(float)
        d = pd.DataFrame({
            "season": m["season"], "game_id": m["game_id"],
            "gid_str": m["gid_str"],
            "game_date": m["game_date"], "open_margin": m["open_margin"],
            "margin_actual": m["margin_actual"],
            "m_us": mg, "p_us": 1.0 / (1.0 + np.exp(-mg / SCALE))})
        p = SCRATCH / f"frame_{name}.csv.gz"
        d.to_csv(p, index=False, compression="gzip")
        paths[name] = p
    return paths, m


# --------------------------------------------------------- execution scoring
PANELS = None


def era_of(season):
    if season in wq.OFFSHORE_SEASONS:
        return "KAG", "MEASURED"
    if season in wq.GAP_SEASONS:
        return "KAG", "EXTRAPOLATED"
    if season in wq.RETAIL_MEASURED:
        return "ESPN", "MEASURED"
    return "ESPN", "EXTRAPOLATED"


_pcache = {}


def pooled(pn, k, hc, side):
    key = (pn, k, hc, side)
    if key not in _pcache:
        G, W, kept = wq.pooled_dist(PANELS[pn], k, hc, side)
        _pcache[key] = (G, np.cumsum(W))
    return _pcache[key]


def bet_law(b, j, k, hc):
    """(G, cw) for one bet — D166's exact per-game or pooled law."""
    pn, tag = era_of(b["season"])
    side = "H" if b["home"][j] else "A"
    gid = b["gids"][j]
    if k == 1:
        return None, None
    if gid in PANELS[pn] and tag == "MEASURED":
        gg, ww = wq.gain_dist(PANELS[pn][gid], side, k, hc)
        o = np.argsort(gg, kind="stable")
        return gg[o], np.cumsum(ww[o])
    return pooled(pn, k, hc, side)


def score_bets(bets, k, hc):
    """D166's score_tier, per-bet detail retained.  Returns rows + per-bet arrays."""
    rows, ev_all, pw_all, pl_all, sea_all = [], [], [], [], []
    for b in bets:
        tot_pay = tot_w = tot_l = tot_p = 0.0
        n = len(b["idx"])
        for j in range(n):
            d = float(b["d"][j])
            if k == 1:
                ev = WIN if d > 0 else (0.0 if d == 0 else -1.0)
                w_ = 1.0 if d > 0 else 0.0
                l_ = 1.0 if d < 0 else 0.0
                pu = 1.0 - w_ - l_
            else:
                G, cw = bet_law(b, j, k, hc)
                ev, w_, l_, pu = wq.payoff_from_dist(d, G, cw)
            tot_pay += ev
            tot_w += w_
            tot_l += l_
            tot_p += pu
            ev_all.append(ev)
            pw_all.append(w_)
            pl_all.append(l_)
            sea_all.append(b["season"])
        rows.append(dict(season=b["season"], n=float(n), pay=tot_pay,
                         roi=tot_pay / n,
                         cover=tot_w / (tot_w + tot_l),
                         push=tot_p, tag=era_of(b["season"])[1]))
    n = sum(r["n"] for r in rows)
    p = sum(r["pay"] for r in rows)
    w = sum(r["cover"] * (r["n"] - r["push"]) for r in rows)
    nz = sum(r["n"] - r["push"] for r in rows)
    return dict(rows=rows, n=n, pay=p, roi=p / n, cover=w / nz,
                ci=oc.cluster_mean_t([r["roi"] for r in rows]),
                cum=list(np.cumsum([r["pay"] for r in rows])),
                ev=np.array(ev_all), pw=np.array(pw_all), pl=np.array(pl_all),
                season_of_bet=np.array(sea_all))


def run_variant(path):
    """D164/D165/D166's loop, unchanged, on one variant frame."""
    oc.FRAME = path
    df, seasons = oc.load()
    K = len(seasons)
    st = oc.build_static(df)
    m_us = df["m_us"].to_numpy(float)
    p_us = df["p_us"].to_numpy(float)
    payoff, M, keys, win, push, bet_home = oc.payoff_and_masks(m_us, p_us, st)
    cnt, pay = oc.agg(M, payoff, st)
    steps = wq.arm_window(cnt, pay, K, None, wq.K_COMMON)
    gid = df["gid_str"].astype(str).to_numpy()
    d_signed = np.where(bet_home, st["actual"] - st["open_margin"],
                        st["open_margin"] - st["actual"])
    edge = np.abs(m_us - st["open_margin"])
    conf = np.abs(p_us - 0.5)
    bets = []
    for s in steps:
        k = s["k"]
        sel = (M[s["cfg"]] > 0) & (st["s_i"] == k)
        idx = np.where(sel)[0]
        bets.append(dict(season=seasons[k], cfg=list(map(str, keys[s["cfg"]])),
                         cfg_idx=s["cfg"], idx=idx, gids=gid[idx],
                         d=d_signed[idx], home=bet_home[idx],
                         edge=edge[idx], conf=conf[idx], k=k,
                         sel_roi=s["sel_roi"], sel_n=s["sel_n"]))
    ref_roi = [float(pay[0, j] / cnt[0, j]) for j in range(K)]
    ref_cov = [float((M[0] * win)[st["s_i"] == j].sum()
                     / ((M[0])[st["s_i"] == j].sum()
                        - (M[0] * push)[st["s_i"] == j].sum()))
               for j in range(K)]
    return dict(seasons=seasons, K=K, bets=bets, steps=steps, cnt=cnt, pay=pay,
                M=M, keys=keys, st=st, payoff=payoff, win=win, push=push,
                ref_roi=ref_roi, ref_cov=ref_cov, edge=edge, conf=conf,
                d_signed=d_signed, m_us=m_us, p_us=p_us, df=df)


def mde80(per):
    per = np.asarray(per, float)
    return 2.802 * per.std(ddof=1) / np.sqrt(len(per))


def loso(rows):
    """Leave-one-season-out influence on the pooled ROI."""
    n = np.array([r["n"] for r in rows], float)
    p = np.array([r["pay"] for r in rows], float)
    pooled = p.sum() / n.sum()
    out = []
    for j in range(len(rows)):
        keep = np.arange(len(rows)) != j
        rj = p[keep].sum() / n[keep].sum()
        out.append(dict(season=rows[j]["season"], roi_drop=rj,
                        influence=pooled - rj,
                        cum_drop=float(p[keep].sum()),
                        season_roi=rows[j]["roi"], n=rows[j]["n"]))
    return pooled, out


# ----------------------------------------------------------------- Q3 staking
def kelly_stake(p_cal, dec=1.0 + 100.0 / 110.0):
    f = (p_cal * dec - 1.0) / (dec - 1.0)
    return max(0.0, f)


def staking_arms(V, tier):
    """Every arm's stake vector on the SAME bet stream.  Fitted parameters are
    estimated WALK-FORWARD from the selection window of that step only."""
    bets = V["bets"]
    cnt, pay = V["cnt"], V["pay"]
    M, st = V["M"], V["st"]
    edge_all, conf_all = V["edge"], V["conf"]
    d_all = V["d_signed"]
    arms = {n: [] for n in ("S0_FLAT", "S1_EDGE", "S1_EDGE_UNCAP", "S2_KELLY_WF",
                            "S2X_KELLY_D121", "S3_CONF_LADDER", "S3_CONF_WF")}
    params = []
    for b in bets:
        k, cfgi = b["k"], b["cfg_idx"]
        # the SELECTION WINDOW of this step: seasons 0..k-1, the config's own bets
        hist = (M[cfgi] > 0) & (st["s_i"] < k)
        hidx = np.where(hist)[0]
        he, hc_ = edge_all[hidx], conf_all[hidx]
        hd = d_all[hidx]
        hcov = np.where(hd > 0, 1.0, np.where(hd < 0, 0.0, np.nan))
        n = len(b["idx"])
        e, c = b["edge"], b["conf"]
        # ---- S0 flat
        arms["S0_FLAT"].append(np.ones(n))
        # ---- S1 edge-proportional, normaliser = walk-forward mean |edge|
        Ek = float(he.mean()) if len(he) else 1.0
        s1 = e / Ek
        arms["S1_EDGE_UNCAP"].append(s1)
        arms["S1_EDGE"].append(np.clip(s1, 0.0, STAKE_CAP))
        # ---- S2 capped fractional Kelly, (a,b) refit WALK-FORWARD
        from math import erf, sqrt
        def phi(x):
            return 0.5 * (1.0 + np.vectorize(lambda z: erf(z / sqrt(2.0)))(x))
        pc_h = phi(he / SD_MARGIN)
        ok = np.isfinite(hcov)
        if ok.sum() >= 50:
            bb, aa = np.polyfit(pc_h[ok] - 0.5, hcov[ok] - 0.5, 1)
        else:
            aa, bb = 0.0, 0.0
        pc = phi(e / SD_MARGIN)
        est = np.maximum(0.0, aa + bb * (pc - 0.5))
        # stake in UNITS of a 100u reference bankroll: 0.25 * f * 100
        s2 = np.array([KELLY_FRAC * kelly_stake(0.5 + x) * KELLY_BANK
                       for x in est])
        arms["S2_KELLY_WF"].append(np.clip(s2, 0.0, STAKE_CAP))
        est_x = np.maximum(0.0, D121_OPEN_A + D121_OPEN_B * (pc - 0.5))
        s2x = np.array([KELLY_FRAC * kelly_stake(0.5 + x) * KELLY_BANK
                        for x in est_x])
        arms["S2X_KELLY_D121"].append(np.clip(s2x, 0.0, STAKE_CAP))
        # ---- S3 confidence terciles, breakpoints WALK-FORWARD
        if len(hc_) >= 30:
            q1, q2 = np.percentile(hc_, [33.3333, 66.6667])
        else:
            q1, q2 = np.percentile(c, [33.3333, 66.6667])
        tier_idx = np.where(c <= q1, 0, np.where(c <= q2, 1, 2))
        arms["S3_CONF_LADDER"].append(np.array([0.5, 1.0, 1.5])[tier_idx])
        # walk-forward band weights: each tercile's realised cover excess
        wts = []
        htier = np.where(hc_ <= q1, 0, np.where(hc_ <= q2, 1, 2))
        for t in range(3):
            sel = (htier == t) & ok
            wts.append(max(0.0, (np.nanmean(hcov[sel]) - 1.0 / (1.0 + WIN))
                           if sel.sum() >= 20 else 0.0))
        wts = np.array(wts)
        wts = (wts / wts.mean()) if wts.mean() > 0 else np.ones(3)
        arms["S3_CONF_WF"].append(np.clip(wts, 0.0, STAKE_CAP)[tier_idx])
        params.append(dict(season=b["season"], sel_n=int(len(hidx)),
                           mean_edge_window=Ek, kelly_a=float(aa),
                           kelly_b=float(bb), conf_q1=float(q1),
                           conf_q2=float(q2),
                           conf_wf_weights=[float(x) for x in wts]))
    return {k: np.concatenate(v) for k, v in arms.items()}, params


def stake_metrics(stake, tier, bets, mcG=None, label=""):
    """ROI on staked units, volatility, cumulative units, drawdown, RA."""
    ev, pw, pl = tier["ev"], tier["pw"], tier["pl"]
    seas = tier["season_of_bet"]
    tot_stake = float(stake.sum())
    pnl = stake * ev
    roi = float(pnl.sum() / tot_stake)
    # UNIT VOLATILITY = sd of the REALISED per-bet P&L in units.  At the k=1
    # tier the outcomes are realised, so this is exact; at k>1 the outcome is a
    # draw from the shop-gain law and it is taken from the MC below.
    sd_bet = float(np.std(stake * np.where(pw > pl, WIN, np.where(pl > pw, -1.0, 0.0)),
                          ddof=1))
    # per-season ROI (stake-weighted)
    per, per_u, per_s = [], [], []
    for s in sorted(set(seas.tolist()), key=lambda x: x):
        m = seas == s
        per.append(float(pnl[m].sum() / stake[m].sum()))
        per_u.append(float(pnl[m].sum()))
        per_s.append(float(stake[m].sum()))
    ci = oc.cluster_mean_t(per)
    sd_season = float(np.std(per, ddof=1))
    sd_season_u = float(np.std(per_u, ddof=1))
    out = dict(label=label, roi=roi, total_stake=tot_stake,
               mean_stake=tot_stake / len(stake),
               max_stake=float(stake.max()),
               cum_units=float(pnl.sum()),
               cum_units_equalstake=float(roi * len(stake)),
               sd_bet=sd_bet, sd_season=sd_season, sd_season_units=sd_season_u,
               ci=ci,
               mde80=mde80(per), per_season_roi=per,
               per_season_units=per_u, per_season_stake=per_s,
               ra_per_season_sd=float(roi / sd_season) if sd_season else np.nan,
               ra_units=float(pnl.sum() / sd_season_u) if sd_season_u else np.nan,
               sharpe_bet=float(pnl.sum() / (sd_bet * np.sqrt(len(stake))))
               if sd_bet else np.nan,
               seasons_positive=int(sum(1 for x in per if x > 0)))
    if mcG is not None:
        dd, fin, sdb = [], [], []
        rng = np.random.default_rng(SEED)
        for _ in range(NDRAW_MC):
            u = rng.random(len(stake))
            realised = np.empty(len(stake))
            for i, (G, cw) in enumerate(mcG):
                if G is None:
                    realised[i] = 1.0 if pw[i] > 0.5 else (
                        -1.0 if pl[i] > 0.5 else 0.0)
                else:
                    j = int(np.searchsorted(cw, u[i] * cw[-1]))
                    realised[i] = G[min(j, len(G) - 1)]
            dsig = tier["d_signed_bets"] + realised
            x = np.where(dsig > 0, WIN, np.where(dsig < 0, -1.0, 0.0))
            p = np.cumsum(stake * x)
            dd.append(float((np.maximum.accumulate(np.concatenate([[0.0], p]))
                             - np.concatenate([[0.0], p])).max()))
            fin.append(float(p[-1]))
            sdb.append(float(np.std(stake * x, ddof=1)))
        out["maxdd_mc"] = dict(median=float(np.median(dd)),
                               p05=float(np.percentile(dd, 5)),
                               p95=float(np.percentile(dd, 95)),
                               mean_final=float(np.mean(fin)))
        out["sd_bet"] = sd_bet = float(np.mean(sdb))
        out["sharpe_bet"] = float(pnl.sum() / (sd_bet * np.sqrt(len(stake))))
        out["mc_roi_check"] = float(np.mean(fin) / tot_stake)
    else:
        p = np.cumsum(pnl)
        out["maxdd_exact"] = float(
            (np.maximum.accumulate(np.concatenate([[0.0], p]))
             - np.concatenate([[0.0], p])).max())
    return out


def main():
    global PANELS
    say("=" * 84)
    say("SL-SCORE — structural lookahead, 2024-25 influence, walk-forward staking")
    say("=" * 84)
    paths, merged = build_frames()
    PANELS = wq.load_panels()
    say(f"panels: KAG {len(PANELS['KAG'])} games, ESPN {len(PANELS['ESPN'])}")

    R = {"prereg_sha256":
         "76d79823bd45e0e6d1339ce7d22cc3b711ece3ec7ee5f727f42f4e526b06cddc"}

    # ==================================================================== Q1
    say("\n" + "-" * 84)
    say("Q1  ABLATION LADDER — D166's loop on progressively more primitive models")
    say("-" * 84)
    say(f"{'variant':30s} {'n':>5s} {'1bk ROI%':>9s} {'DEF ROI%':>9s} "
        f"{'cover%':>7s} {'cum u':>7s} {'13-dof CI':>19s} {'MDE80':>6s} "
        f"{'surv':>6s} {'+sns':>5s}")
    VR, base_def = {}, None
    for name, desc, _fn in VARIANTS:
        V = run_variant(paths[name])
        t1 = score_bets(V["bets"], 1, False)
        t5 = score_bets(V["bets"], 5, True)
        per5 = [r["roi"] for r in t5["rows"]]
        if base_def is None:
            base_def = t5["roi"]
        surv = t5["roi"] / base_def
        say(f"{name:30s} {t1['n']:5.0f} {100*t1['roi']:+9.2f} "
            f"{100*t5['roi']:+9.2f} {100*t5['cover']:7.3f} "
            f"{t5['cum'][-1]:+7.1f} "
            f"[{100*t5['ci']['lo']:+6.2f},{100*t5['ci']['hi']:+6.2f}] "
            f"{100*mde80(per5):6.2f} {surv:6.2f} "
            f"{sum(1 for x in per5 if x>0):3d}/14")
        VR[name] = dict(
            desc=desc, n=t1["n"],
            roi_1book=t1["roi"], roi_default=t5["roi"], cover=t5["cover"],
            cum_units=t5["cum"][-1], ci=t5["ci"], mde80=float(mde80(per5)),
            survival=float(surv),
            per_season_roi_1book=[r["roi"] for r in t1["rows"]],
            per_season_roi_default=per5,
            per_season_n=[r["n"] for r in t1["rows"]],
            per_season_pay_default=[r["pay"] for r in t5["rows"]],
            cum_path=[float(x) for x in t5["cum"]],
            cum_path_1book=[float(x) for x in t1["cum"]],
            seasons=[r["season"] for r in t5["rows"]],
            cfgs=[b["cfg"] for b in V["bets"]],
            steps_cfg=[s["cfg"] for s in V["steps"]],
            ref_roi_per_season=V["ref_roi"], ref_cov_per_season=V["ref_cov"],
            seasons_positive=int(sum(1 for x in per5 if x > 0)))
        if name == "V0_FULL":
            V0, T1_0, T5_0 = V, t1, t5
        else:
            del V["M"]
    # paired deltas against V0 on the identical 14 seasons
    say("\nPAIRED per-season delta vs V0 (firm default), 13 dof:")
    for name, _d, _f in VARIANTS[1:]:
        dl = [a - b for a, b in zip(VR[name]["per_season_roi_default"],
                                    VR["V0_FULL"]["per_season_roi_default"])]
        ci = oc.cluster_mean_t(dl)
        VR[name]["paired_vs_V0"] = ci
        say(f"  {name:30s} {100*ci['mean']:+7.2f} "
            f"[{100*ci['lo']:+7.2f},{100*ci['hi']:+7.2f}]"
            f"{'  SIG' if ci['sig'] else '  ns'}")
    R["Q1_variants"] = VR

    # ---- Q1(b) backward generalisation: the MODEL's own accuracy per variant,
    # on D161's full 22,804-game frame, split into the block the features were
    # chosen on (2021-26) and the future-informed block scored by D166.
    sj = json.load(open(SL_COMP_JSON))
    say("\n" + "-" * 84)
    say("Q1(b)  BACKWARD GENERALISATION — the MODEL's own normalized gap "
        "(D161's statistic)")
    say("       lower = better; the market wins whenever it is positive")
    say("-" * 84)
    cc = pd.read_csv(SL_COMP_CSV, dtype={"game_id": str})
    LN2 = 0.6931471805599453
    yv = cc["y"].to_numpy(float)
    pm = np.clip(cc["p_mkt"].to_numpy(float), 1e-9, 1 - 1e-9)
    seas_c = cc["season"].to_numpy()
    say(f"{'variant':30s} {'POOL19':>8s} {'D166 scored 14':>15s} "
        f"{'FUTURE-INFORMED 9':>18s} {'CHOSEN-ON 5':>12s}  gap")
    BW = {}
    blk_a = [s for s in sorted(set(seas_c.tolist())) if "2012-13" <= s <= "2020-21"]
    blk_b = [s for s in sorted(set(seas_c.tolist())) if s >= "2021-22"]
    for name, _d, fn in VARIANTS:
        mg = fn(cc).to_numpy(float)
        pu = np.clip(1.0 / (1.0 + np.exp(-mg / SCALE)), 1e-9, 1 - 1e-9)

        def ng(sel):
            u = -(yv[sel] * np.log(pu[sel]) + (1 - yv[sel]) * np.log(1 - pu[sel])).mean()
            k = -(yv[sel] * np.log(pm[sel]) + (1 - yv[sel]) * np.log(1 - pm[sel])).mean()
            return 100.0 * (u - k) / (LN2 - k)
        per = {s: float(ng(seas_c == s)) for s in sorted(set(seas_c.tolist()))}
        allm = np.ones(len(yv), bool)
        m14 = np.isin(seas_c, blk_a + blk_b)
        m9 = np.isin(seas_c, blk_a)
        m5 = np.isin(seas_c, blk_b)
        BW[name] = dict(pool19=float(ng(allm)), scored14=float(ng(m14)),
                        future_informed9=float(ng(m9)), chosen_on5=float(ng(m5)),
                        per_season=per)
        say(f"{name:30s} {BW[name]['pool19']:+8.2f} {BW[name]['scored14']:+15.2f} "
            f"{BW[name]['future_informed9']:+18.2f} {BW[name]['chosen_on5']:+12.2f}"
            f"  {BW[name]['future_informed9']-BW[name]['chosen_on5']:+6.2f}")
    R["Q1_backward"] = BW

    # ==================================================================== Q2
    say("\n" + "-" * 84)
    say("Q2  IS 2024-25 SPECIAL IN THE MODEL, OR ONLY IN THE BETTING OUTCOME?")
    say("-" * 84)
    seas19 = [s["season"] for s in sj["seasons"]]
    g = np.array([s["norm_gap_pct"] for s in sj["seasons"]])
    i25 = seas19.index("2024-25")
    oth = np.delete(g, i25)
    z_gap = (g[i25] - oth.mean()) / oth.std(ddof=1)
    cov = np.array(VR["V0_FULL"]["ref_cov_per_season"]) * 100
    roi19 = np.array(VR["V0_FULL"]["ref_roi_per_season"]) * 100
    z_cov = (cov[i25] - np.delete(cov, i25).mean()) / np.delete(cov, i25).std(ddof=1)
    z_roi = (roi19[i25] - np.delete(roi19, i25).mean()) / np.delete(roi19, i25).std(ddof=1)
    say(f"  normalized model gap  2024-25 {g[i25]:+.2f}%  others mean "
        f"{oth.mean():+.2f}% sd {oth.std(ddof=1):.2f}  ->  z = {z_gap:+.2f}"
        f"   rank {int((g > g[i25]).sum())+1}/19 (1 = worst gap)")
    say(f"  ALL-GAMES cover rate  2024-25 {cov[i25]:.3f}%  others mean "
        f"{np.delete(cov,i25).mean():.3f}% sd {np.delete(cov,i25).std(ddof=1):.3f}"
        f"  ->  z = {z_cov:+.2f}")
    say(f"  ALL-GAMES ROI         2024-25 {roi19[i25]:+.2f}%  ->  z = {z_roi:+.2f}")
    rr = float(np.corrcoef(g, cov)[0, 1])
    rr14 = float(np.corrcoef(g[5:], cov[5:])[0, 1])
    say(f"  corr(model normalized gap, ATS cover rate) over 19 seasons = "
        f"{rr:+.3f}   over the 14 scored = {rr14:+.3f}")
    say("  (a NEGATIVE correlation would mean a more accurate model covers "
        "more; it is the sign of\n   the mechanism the contamination story "
        "needs)")
    R["Q2_2425"] = dict(norm_gap=float(g[i25]), z_norm_gap=float(z_gap),
                        cover=float(cov[i25]), z_cover=float(z_cov),
                        roi=float(roi19[i25]), z_roi=float(z_roi),
                        norm_gap_all=g.tolist(), cover_all=cov.tolist(),
                        roi_all=roi19.tolist(), seasons19=seas19)

    say("\n  FULL LEAVE-ONE-SEASON-OUT INFLUENCE PROFILE (all 14 scored seasons)")
    say(f"  {'season':9s} {'n':>5s} {'1bk ROI%':>9s} {'drop->pooled':>13s} "
        f"{'infl pp':>8s} | {'DEF ROI%':>9s} {'drop->pooled':>13s} "
        f"{'infl pp':>8s} {'cum u':>8s}")
    p1, l1 = loso(T1_0["rows"])
    p5, l5 = loso(T5_0["rows"])
    for a, b in zip(l1, l5):
        say(f"  {a['season']:9s} {a['n']:5.0f} {100*a['season_roi']:+9.2f} "
            f"{100*a['roi_drop']:+13.2f} {100*a['influence']:+8.2f} | "
            f"{100*b['season_roi']:+9.2f} {100*b['roi_drop']:+13.2f} "
            f"{100*b['influence']:+8.2f} {b['cum_drop']:+8.1f}")
    R["Q2_loso"] = {"pooled_1book": p1, "pooled_default": p5,
                    "rows_1book": l1, "rows_default": l5}

    # ==================================================================== Q3
    say("\n" + "-" * 84)
    say("Q3  STAKING ON THE FIRM-DEFAULT BET STREAM (k=5 measured + haircut)")
    say("-" * 84)
    stakes, params = staking_arms(V0, T5_0)
    say("  walk-forward parameters, per step (fitting window = seasons 1..k):")
    for p in params:
        say(f"    {p['season']}  histbets={p['sel_n']:5d} "
            f"E|edge|={p['mean_edge_window']:.3f}  kelly a={p['kelly_a']:+.5f} "
            f"b={p['kelly_b']:+.4f}  conf q33={p['conf_q1']:.4f} "
            f"q67={p['conf_q2']:.4f}  bandw="
            f"{[round(x,3) for x in p['conf_wf_weights']]}")
    # per-bet gain laws for the MC drawdown
    mcG, dsb = [], []
    for b in V0["bets"]:
        for j in range(len(b["idx"])):
            mcG.append(bet_law(b, j, 5, True))
            dsb.append(float(b["d"][j]))
    T5_0["d_signed_bets"] = np.array(dsb)
    T1_0["d_signed_bets"] = np.array(dsb)
    say(f"\n  {'arm':20s} {'mean u':>7s} {'maxu':>5s} {'staked':>8s} {'ROI%':>7s} "
        f"{'cum u':>7s} {'eq-u':>7s} {'sd/bet':>7s} {'sd/sns':>7s} "
        f"{'sdU/sns':>8s} {'RA':>6s} {'RAu':>6s} {'Shrp':>6s} "
        f"{'maxDD (p05-p95)':>22s} {'+sns':>5s} {'13-dof CI':>19s}")
    ST = {}
    for nm in ("S0_FLAT", "S1_EDGE", "S1_EDGE_UNCAP", "S2_KELLY_WF",
               "S2X_KELLY_D121", "S3_CONF_LADDER", "S3_CONF_WF"):
        s = stakes[nm]
        r = stake_metrics(s, T5_0, V0["bets"], mcG=mcG, label=nm)
        ST[nm] = r
        say(f"  {nm:20s} {r['mean_stake']:7.3f} {r['max_stake']:5.2f} "
            f"{r['total_stake']:8.1f} "
            f"{100*r['roi']:+7.2f} {r['cum_units']:+7.1f} "
            f"{r['cum_units_equalstake']:+7.1f} {r['sd_bet']:7.3f} "
            f"{100*r['sd_season']:7.2f} {r['sd_season_units']:8.2f} "
            f"{r['ra_per_season_sd']:6.3f} {r['ra_units']:6.3f} "
            f"{r['sharpe_bet']:6.3f} "
            f"{r['maxdd_mc']['median']:7.1f} "
            f"({r['maxdd_mc']['p05']:.1f}-{r['maxdd_mc']['p95']:.1f})".ljust(23)
            + f"{r['seasons_positive']:3d}/14 "
            f"[{100*r['ci']['lo']:+6.2f},{100*r['ci']['hi']:+6.2f}]")
    # the same arms at the 1-book tier, where outcomes are REALISED
    say("\n  1-BOOK tier (realised outcomes, EXACT drawdown):")
    ST1 = {}
    for nm in ("S0_FLAT", "S1_EDGE", "S2_KELLY_WF", "S3_CONF_LADDER"):
        s = stakes[nm]
        r = stake_metrics(s, T1_0, V0["bets"], mcG=None, label=nm)
        ST1[nm] = r
        say(f"  {nm:20s} {r['mean_stake']:7.3f} {r['total_stake']:8.1f} "
            f"{100*r['roi']:+7.2f} {r['cum_units']:+7.1f} "
            f"{r['cum_units_equalstake']:+7.1f} {r['sd_bet']:7.3f} "
            f"{100*r['sd_season']:7.2f} {r['sd_season_units']:8.2f} "
            f"{r['ra_per_season_sd']:6.3f} {r['ra_units']:6.3f} "
            f"{r['sharpe_bet']:6.3f} {r['maxdd_exact']:7.1f} "
            f"{r['seasons_positive']:3d}/14")
    say("\n  PAIRED per-season delta vs FLAT (firm default), 13 dof:")
    for nm in ST:
        if nm == "S0_FLAT":
            continue
        dl = [a - b for a, b in zip(ST[nm]["per_season_roi"],
                                    ST["S0_FLAT"]["per_season_roi"])]
        ci = oc.cluster_mean_t(dl)
        ST[nm]["paired_vs_flat"] = ci
        say(f"    {nm:20s} {100*ci['mean']:+7.2f} "
            f"[{100*ci['lo']:+7.2f},{100*ci['hi']:+7.2f}]"
            f"{'  SIG' if ci['sig'] else '  ns'}")
    R["Q3_staking_default"] = ST
    R["Q3_staking_1book"] = ST1
    R["Q3_params"] = params

    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    (ROOT / "data" / "logs").mkdir(exist_ok=True)
    (ROOT / "data" / "logs" / f"sl_score{_TAG}.log").write_text("\n".join(LOG))
    say(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
