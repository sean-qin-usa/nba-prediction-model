#!/usr/bin/env python3
"""BO-OPEN2223-OOS — the D121 out-of-sample extension on recovered 2022-23
post-January opening lines (TeamRankings scrape).

WHY THIS EXISTS.  D120/D121's paired open-vs-close cells could not score
2022-23's late window: the SBR archive dies 2023-01-16, so that season's
post-January games had no opening line, and the gp>=55 late window (R4_LOWT,
T20_D03_10_W) starts ~February — the hole removed 100% of 2022-23's
late-window games.  The TeamRankings scrape recovered 623 post-2023-01-16
regular-season opens (spread-only, no MLs).  2022-23 never entered rule
selection at any stage (rules frozen from dev 23-24+24-25, OOS 25-26), so
this is the first genuinely untouched season for the PRIMARY late-season
rule at open prices.

NOTHING IS RE-CHOSEN.  Frame construction, pricing, rule operators, scoring
and bootstrap machinery are IMPORTED from scripts/bo_openbacktest.py (the
D120/D121 artifact generator) and run on the frozen registry verbatim.

PRICING CONVENTION (stated per the task):
  p_open  = sigmoid(open_margin / 6.96)   — the program's spread->prob map,
            the exact frame frozen in bet_engine.OPEN_SHRUNK
            ("SP open, sigmoid(m/6.96)"), i.e. D120's PRIMARY|SP|OPEN frame.
  decimal = max(1 / (p_side * 1.045), 1.01) — the D75/D78 proportional
            overround, which D120 VALIDATED against real opening MLs
            (measured 1.0431/1.0433).  No MLs exist for the TR rows, so the
            SP frame is the only computable one — and D120 established this
            convention is ~2pp PESSIMISTIC on breakeven vs real MLs (the
            spread->prob map, not the vig).  Every negative number below
            therefore carries a known ~+2pp haircut in our favor.
  close   = odds_market.home_exp_margin (the model's own close source),
            same map/vig.  TR's own close is carried as a robustness check.

RULES HONORED: DuckDB read_only=True (60s retry on lock); new file; nothing
in nbapred/ or the frozen registry touched.

Run:  python scripts/bo_open2223_oos.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import duckdb                                                    # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402

import bo_openbacktest as bo                                     # noqa: E402
from ba_intersection import star_out_map                         # noqa: E402
from bet_sim3 import MIN_DEC, OVERROUND                          # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
RT1 = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
REG = os.path.join(ROOT, "data", "bo_openbacktest.json")
OUT = os.path.join(ROOT, "data", "bo_open2223_oos.json")

CUT = pd.Timestamp("2023-01-16")     # SBR archive death; recovered window is >
SEED = bo.SEED


def _ro_connect(attempts: int = 10, wait_s: float = 60.0):
    for i in range(attempts):
        try:
            return duckdb.connect(DB, read_only=True)
        except Exception as e:                                   # noqa: BLE001
            msg = str(e).lower()
            if ("lock" not in msg and "held" not in msg) or i == attempts - 1:
                raise
            print(f"reader blocked by writer lock, retry {i+1}/{attempts} "
                  f"in {wait_s:.0f}s")
            time.sleep(wait_s)


def binom_ci(k: int, n: int):
    """Exact Clopper-Pearson 95% CI on a hit rate."""
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main() -> None:
    res: dict = {"convention": {
        "p_open": "sigmoid(open_margin/6.96)  [bet_engine.OPEN_SHRUNK frame, "
                  "D120 PRIMARY|SP|OPEN]",
        "decimal": f"max(1/(p_side*{OVERROUND}), {MIN_DEC})  "
                   f"[D75/D78 proportional overround, D120-validated 1.0431]",
        "close_source": "odds_market.home_exp_margin (primary) / "
                        "TR close_margin (robustness)",
        "known_bias": "SP map ~2pp pessimistic on breakeven vs real MLs (D120)",
        "seed": SEED}}

    # ---- frame: the exact D120/D121 build, PRIMARY frame --------------------
    m = bo.build(RT1, "p_full", "PRIMARY rt1 p_full 4-season", res)

    # ---- odds_market close for 2022-23 --------------------------------------
    con = _ro_connect()
    try:
        om = con.execute("""
            SELECT game_date, home, away, home_exp_margin AS om_close_margin
            FROM odds_market WHERE season_end = 2023
        """).fetchdf()
        so = star_out_map(con)
    finally:
        con.close()
    om["game_date"] = pd.to_datetime(om.game_date)

    m22 = m[(m.season == "2022-23") & (m.game_date > CUT)].copy()
    src = m22.source.value_counts().to_dict()
    m22 = m22.merge(om, on=["game_date", "home", "away"], how="left")
    n_tr = len(m22)
    m22 = m22[m22.om_close_margin.notna()].reset_index(drop=True)

    print(f"\n{'=' * 116}\nD121 OOS EXTENSION — 2022-23 AFTER {CUT.date()} "
          f"(recovered TeamRankings opens)\n{'=' * 116}")
    print(f"  recovered-open games in frame: {n_tr} (source: {src}); with an "
          f"odds_market close: {len(m22)}")
    d_tr_om = (m22.close_margin - m22.om_close_margin)
    fav_flip = int(((m22.close_margin >= 0) !=
                    (m22.om_close_margin >= 0)).sum())
    print(f"  TR close vs odds_market close: mean|diff| = "
          f"{d_tr_om.abs().mean():.3f} pts, corr = "
          f"{np.corrcoef(m22.close_margin, m22.om_close_margin)[0,1]:.4f}, "
          f"favourite flips = {fav_flip}")
    print(f"  window: {m22.game_date.min().date()} .. "
          f"{m22.game_date.max().date()};   late (gp>=55) games: "
          f"{int(m22.late.sum())}/{len(m22)}")
    res["frame"] = {"n_recovered": n_tr, "n_scored": len(m22),
                    "sources": src, "tr_vs_om_close_mad": float(
                        d_tr_om.abs().mean()), "fav_flips": fav_flip,
                    "n_late": int(m22.late.sum())}

    # keep TR close aside, then make odds_market the close of record
    m22["close_margin_tr"] = m22.close_margin
    m22["p_close_tr"] = m22.p_close_sp
    m22["close_margin"] = m22.om_close_margin
    m22["p_close_sp"] = bo.sigmoid(m22.close_margin / bo.SPREAD_SCALE)
    # re-derive the close-favourite star-out flag on the odds_market close
    som = so.rename(columns={"team": "fav_team"})[["game_id", "fav_team",
                                                   "star_out"]]
    m22["fav_team"] = np.where(m22.p_close_sp >= 0.5, m22.home, m22.away)
    m22 = m22.drop(columns=["fav_star_out_close"]).merge(
        som, on=["game_id", "fav_team"], how="left")
    m22["fav_star_out_close"] = m22.star_out.fillna(False).astype(bool)
    m22 = m22.drop(columns=["star_out", "fav_team"])

    # ---- prices (verbatim price_cols; SP source only — no MLs exist) --------
    po, deco, oko = bo.price_cols(m22, "open", "SP")
    pc, decc, okc = bo.price_cols(m22, "close", "SP")
    pctr = np.where(m22.pick_home, m22.p_close_tr, 1 - m22.p_close_tr)
    dectr = np.maximum(1.0 / (pctr * OVERROUND), MIN_DEC)
    ok = oko & okc
    mo, edge_o, same_o = bo.registry_masks(m22, po, "open")
    mc, edge_c, _ = bo.registry_masks(m22, pc, "close")

    print(f"\n  universe: n priced = {int(ok.sum())}   mean edge vs OPEN "
          f"{edge_o[ok].mean():+.5f}   vs CLOSE {edge_c[ok].mean():+.5f}   "
          f"same-side@open {100 * same_o[ok].mean():.1f}%")

    # ---- registered D121 numbers for side-by-side ---------------------------
    reg = json.load(open(REG))
    FR = "PRIMARY rt1 p_full 4-season"
    reg_cells = {(c["set"], c["arm"], c["window"]): c for c in reg["cells"]
                 if c["frame"] == FR and c["src"] == "SP"
                 and c.get("scope") == "rule"}
    reg_pair = {c["set"]: c for c in reg["cells"]
                if c["frame"] == FR and c["src"] == "SP"
                and c.get("scope") == "paired"}
    reg_clv = reg["clv"][FR + "|SP"]

    # ---- the four frozen rules ---------------------------------------------
    print(f"\n[RULES] fired on edge-vs-OPEN, scored at OPEN (A) and at the "
          f"odds_market CLOSE (B, same bets); C = fired-and-scored at close.")
    print(bo.hdr(14))
    clv_om_all = pc - po                     # our side, de-vigged, OM close
    clv_tr_all = pctr - po                   # our side, TR's own close
    res["rules"] = {}
    for rule in mo:
        so_, sc_ = mo[rule] & ok, mc[rule] & ok
        n_o = int(so_.sum())
        out = {"n_open": n_o, "n_close": int(sc_.sum()),
               "n_overlap": int((so_ & sc_).sum())}
        print(f"\n  RULE {rule}   fires@open n={n_o}   fires@close "
              f"n={int(sc_.sum())}   overlap n={out['n_overlap']}")
        if n_o == 0:
            print("    no bets — rule cannot be scored in this window")
            res["rules"][rule] = out
            continue
        for tag, sel, dec, pf in (("A OPEN", so_, deco, po),
                                  ("B OPEN@CLS", so_, decc, pc),
                                  ("C CLOSE", sc_, decc, pc)):
            r = bo.score_ml(m22, sel, dec, pf)
            if not r.get("n"):
                continue
            k = int(round(r["hit"] * r["n"]))
            blo, bhi = binom_ci(k, r["n"])
            r["hit_exact_lo"], r["hit_exact_hi"] = blo, bhi
            r["binom_p_vs_be"] = float(stats.binomtest(
                k, r["n"], r["breakeven"], alternative="greater").pvalue)
            print(f"    {tag:<14}{'2223-L':>8}{bo.fmt(r)}  "
                  f"binomCI[{100*blo:.1f},{100*bhi:.1f}] "
                  f"pbin={r['binom_p_vs_be']:.3f}")
            out[tag] = r
        # registered D121 rows for the same rule (context)
        for w in ("POOL", "DEV", "NONDEV"):
            for arm in ("A OPEN", "B OPEN@CLS"):
                c = reg_cells.get((rule, arm, w))
                if c:
                    print(f"      [D121 reg] {arm:<11}{w:<7} n={c['n']:<4} "
                          f"hit={100*c['hit']:.1f} be={100*c['breakeven']:.2f} "
                          f"ROI={100*c['roi']:+.2f} "
                          f"[{100*c['roi_lo']:+.1f},{100*c['roi_hi']:+.1f}]")
        # paired pure-price dROI on the identical bet set
        ho = m22.hit.values[so_].astype(bool)
        d_om = (np.where(ho, deco[so_] - 1, -1.0) -
                np.where(ho, decc[so_] - 1, -1.0))
        d_tr = (np.where(ho, deco[so_] - 1, -1.0) -
                np.where(ho, dectr[so_] - 1, -1.0))
        for nm, d in (("dROI om", d_om), ("dROI tr", d_tr)):
            mn, lo, hi, t = bo.boot_mean(d)
            sig = "SIG" if (lo > 0 or hi < 0) else "NS"
            rp = reg_pair.get(rule, {})
            extra = (f"   [D121 reg POOL {100*rp.get('mean',np.nan):+.2f}pp "
                     f"[{100*rp.get('lo',np.nan):+.2f},"
                     f"{100*rp.get('hi',np.nan):+.2f}] n={rp.get('n','-')}]"
                     if nm == "dROI om" else "")
            print(f"    PAIRED {nm}: {100*mn:+.2f}pp [{100*lo:+.2f},"
                  f"{100*hi:+.2f}] {sig} n={n_o}{extra}")
            out[nm.replace(" ", "_")] = {"mean": mn, "lo": lo, "hi": hi,
                                         "t": t, "n": n_o}
        # CLV of the open-fired bets
        for nm, cl in (("CLV om", clv_om_all), ("CLV tr", clv_tr_all)):
            a, alo, ahi, at = bo.boot_mean(cl[so_])
            sig = "SIG" if (alo > 0 or ahi < 0) else "NS"
            rc = reg_clv.get(rule, {})
            extra = (f"   [D121 reg {rc.get('clv_prob',np.nan):+.5f} "
                     f"[{rc.get('lo',np.nan):+.5f},{rc.get('hi',np.nan):+.5f}]"
                     f" t={rc.get('t',np.nan):+.1f} n={rc.get('n','-')}]"
                     if nm == "CLV om" else "")
            print(f"    {nm}: {a:+.5f} [{alo:+.5f},{ahi:+.5f}] t={at:+.2f} "
                  f"{sig}  %>0={100*(cl[so_] > 0).mean():.1f}{extra}")
            out[nm.replace(" ", "_")] = {"mean": a, "lo": alo, "hi": ahi,
                                         "t": at,
                                         "frac_pos": float(
                                             (cl[so_] > 0).mean())}
        res["rules"][rule] = out

    # ---- universe CLV + all-same-side, both closes --------------------------
    print(f"\n[CLV] whole-window universe (our side), both close sources:")
    res["clv_universe"] = {}
    for nm, sel in (("ALL games", ok), ("ALL same-side", ok & same_o)):
        for cn, cl in (("om", clv_om_all), ("tr", clv_tr_all)):
            a, alo, ahi, at = bo.boot_mean(cl[sel])
            print(f"    {nm:<15}[{cn}] n={int(sel.sum()):<4} clv={a:+.5f} "
                  f"[{alo:+.5f},{ahi:+.5f}] t={at:+.2f}")
            res["clv_universe"][f"{nm}|{cn}"] = {
                "n": int(sel.sum()), "clv": a, "lo": alo, "hi": ahi, "t": at}

    # ---- selection placebo (D120's control, single-season strata) -----------
    rng = np.random.default_rng(SEED)
    dec10 = pd.qcut(po, 10, labels=False, duplicates="drop")
    p_perm = m22.p_us.values.copy()
    for b in np.unique(dec10):
        idx = np.where(dec10 == b)[0]
        if len(idx) > 1:
            p_perm[idx] = p_perm[rng.permutation(idx)]
    mp = m22.copy()
    mp["p_us"] = p_perm
    mp["pick_home"] = mp.p_us > 0.5
    mp["p_us_side"] = np.where(mp.pick_home, mp.p_us, 1 - mp.p_us)
    mp["conf_us"] = (mp.p_us - 0.5).abs()
    po_p, _, _ = bo.price_cols(mp, "open", "SP")
    pc_p, _, _ = bo.price_cols(mp, "close", "SP")
    mo_p, _, same_p = bo.registry_masks(mp, po_p, "open")
    clv_p = pc_p - po_p
    print(f"\n[PLACEBO] p_us permuted within p_open decile (selection "
          f"mechanism kept, model information destroyed):")
    res["placebo"] = {}
    for nm, selr in [("ALL same-side", ok & same_p)] + \
                    [(r, mo_p[r] & ok) for r in mo_p]:
        if selr.sum() < 5:
            print(f"    {nm:<18} n={int(selr.sum())} — too few, skipped")
            continue
        a, alo, ahi, at = bo.boot_mean(clv_p[selr])
        real = res["rules"].get(nm, {}).get("CLV_om", {}).get("mean", np.nan) \
            if nm != "ALL same-side" else \
            res["clv_universe"]["ALL same-side|om"]["clv"]
        print(f"    {nm:<18} n={int(selr.sum()):<4} placebo clv={a:+.5f} "
              f"[{alo:+.5f},{ahi:+.5f}] t={at:+.2f}   real={real:+.5f}")
        res["placebo"][nm] = {"n": int(selr.sum()), "clv": a, "lo": alo,
                              "hi": ahi, "t": at, "real": float(real)}

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
