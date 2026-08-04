#!/usr/bin/env python3
"""HISTORICAL EVALUATION — ANALYSIS (D153).

Consumes `data/history_pergame.csv` (scripts/history_eval.py) and produces:

  (A) the per-season headline table: n, our log loss, market log loss, raw gap,
      NORMALIZED gap = (ll_us - ll_mkt)/(ln2 - ll_mkt);
  (B) what drives the gap — season-level correlation of the gap against the
      measured era signatures (docs/ERAS.md / data/era_signatures.json):
      home margin, 3PA share, pace, travel, b2b rate, and the market's own
      skill (ln2 - ll_mkt);
  (C) the per-shipped-term ablation, per season and PER ERA, through the V3
      harness `nbapred.eval.splits` (era decomposition with Q / I2 / tau,
      season-clustered bootstrap, cluster-mean t at K-1 dof, rolling origin),
      with a mechanical ERA-STABLE / ERA-CONDITIONAL / ERA-SPECIFIC verdict
      per GATE_POLICY_V2 §10.3 and an explicit SIGN-FLIP flag.

Writes data/history_analysis.json.  READ-ONLY apart from that file.

  python scripts/history_analyze.py [--csv data/history_pergame.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbapred import threads  # noqa: E402
threads.pin(1)

import numpy as np  # noqa: E402

from nbapred.eval import splits as S  # noqa: E402

LN2 = float(np.log(2.0))
CERT = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")   # D132 corpus
STRATA = ("2011-12", "2019-20", "2020-21")     # scorable, never pooled
TERMS = {"no_sched": "D46 schedule layer", "no_carry": "D62 carry",
         "no_tank": "D73 tank", "no_bridge": "D91 October bridge",
         "no_ff": "D21 four-factors leg", "no_comp": "D19 composition leg",
         "ratings_only": "D19+D21 both blend legs",
         "no_prior": "D16 cold-start prior",
         "add_late": "D90 late-state (ADD-BACK diagnostic)"}
B = 2000
SEED = 20260801


def ll(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in list(r):
            if k.startswith("p_") or k in ("y",) or k.startswith("act_"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    pass
    rows.sort(key=lambda r: (r["game_date"], r["game_id"]))
    return rows


def season_table(rows):
    out = {}
    for s in sorted({r["season"] for r in rows}):
        rr = [r for r in rows if r["season"] == s]
        y = np.array([r["y"] for r in rr])
        u = float(ll(y, [r["p_prod"] for r in rr]).mean())
        m = float(ll(y, [r["p_mkt"] for r in rr]).mean())
        out[s] = dict(season=s, era=S.era_of(s), n=len(rr),
                      ll_us=round(u, 5), ll_mkt=round(m, 5),
                      raw_gap=round(u - m, 5),
                      mkt_skill=round(LN2 - m, 5),
                      norm_gap_pct=round(100 * (u - m) / (LN2 - m), 2),
                      in_cert_corpus=s in CERT, stratum=s in STRATA)
    return out


def pooled(rows, seasons, label):
    rr = [r for r in rows if r["season"] in seasons]
    if not rr:
        return None
    y = np.array([r["y"] for r in rr])
    u = float(ll(y, [r["p_prod"] for r in rr]).mean())
    m = float(ll(y, [r["p_mkt"] for r in rr]).mean())
    d = ll(y, [r["p_prod"] for r in rr]) - ll(y, [r["p_mkt"] for r in rr])
    pan = S.Panel(np.array([r["season"] for r in rr]), d,
                  date=np.array([r["game_date"] for r in rr]), label=label)
    cl = S.clustering_report(pan, B, SEED)
    return dict(label=label, seasons=list(seasons), n=len(rr),
                ll_us=round(u, 5), ll_mkt=round(m, 5),
                raw_gap=round(u - m, 5),
                norm_gap_pct=round(100 * (u - m) / (LN2 - m), 2),
                iid_ci=[round(cl["iid"]["lo"], 5), round(cl["iid"]["hi"], 5)],
                season_cluster_ci=[round(cl["season_cluster_boot"]["lo"], 5),
                                   round(cl["season_cluster_boot"]["hi"], 5)],
                season_mean_t_ci=[round(cl["season_mean_t"]["lo"], 5),
                                  round(cl["season_mean_t"]["hi"], 5)],
                icc_season=round(cl["icc_season"]["icc"], 6),
                design_effect_season=round(cl["design_effect_season"], 3))


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), 0
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def _spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        return r
    return _pearson(rank(x), rank(y))


def drivers(tab):
    sig = json.load(open(ROOT / "data" / "era_signatures.json"))["per_season"]
    feats = ["home_margin", "fg3a_share", "poss_per_tg", "travel_km_per_tg",
             "b2b_rate", "core_dnp_rate", "pts_per_tg"]
    seasons = [s for s in tab if s in sig]
    out = {"seasons_used": sorted(seasons), "features": {}}
    ng = [tab[s]["norm_gap_pct"] for s in sorted(seasons)]
    rg = [tab[s]["raw_gap"] for s in sorted(seasons)]
    mk = [tab[s]["ll_mkt"] for s in sorted(seasons)]
    us = [tab[s]["ll_us"] for s in sorted(seasons)]
    for f in feats:
        v = [sig[s].get(f) for s in sorted(seasons)]
        rn, n = _pearson(v, ng)
        rr, _ = _pearson(v, rg)
        rs, _ = _spearman(v, ng)
        out["features"][f] = dict(n=n, r_vs_norm_gap=round(rn, 3),
                                  r_vs_raw_gap=round(rr, 3),
                                  spearman_vs_norm_gap=round(rs, 3),
                                  values={s: sig[s].get(f) for s in sorted(seasons)})
    r, n = _pearson(mk, ng)
    out["market_ll_vs_norm_gap"] = dict(r=round(r, 3), n=n)
    r, _ = _pearson(mk, rg)
    out["market_ll_vs_raw_gap"] = dict(r=round(r, 3))
    r, _ = _pearson(mk, us)
    out["market_ll_vs_our_ll"] = dict(r=round(r, 3))
    # season index (chronology) — is the gap trending?
    idx = list(range(len(sorted(seasons))))
    r, _ = _pearson(idx, ng)
    out["chronology_vs_norm_gap"] = dict(r=round(r, 3))
    return out


def classify(rep, per_era):
    """GATE_POLICY_V2 §10.3 verdict + explicit sign-flip detection."""
    ests = {e["era"]: e["est"] for e in per_era}
    sigs = {e["era"]: (e["sig"], e["est"]) for e in per_era}
    pos_sig = [e for e, (sg, v) in sigs.items() if sg and v > 0]
    neg_sig = [e for e, (sg, v) in sigs.items() if sg and v < 0]
    point_signs = {e: (1 if v > 0 else (-1 if v < 0 else 0)) for e, v in ests.items()}
    point_flip = len({s for s in point_signs.values() if s}) > 1
    era_stable = rep["era"]["era_stable"]
    high_i2 = rep["era"]["high_I2"]
    if pos_sig and neg_sig:
        verdict = "ERA-SPECIFIC"
        why = (f"SIGN FLIP, both directions SIGNIFICANT: helps in {pos_sig}, "
               f"HARMS in {neg_sig}")
    elif not era_stable and point_flip:
        verdict = "ERA-SPECIFIC"
        why = (f"era means differ by more than sampling noise (Q p="
               f"{rep['era']['p_heterogeneity']:.3f}) AND the point-estimate "
               f"sign is not consistent across eras")
    elif not era_stable or high_i2:
        verdict = "ERA-CONDITIONAL"
        why = (f"magnitude heterogeneity (I2={rep['era']['I2']:.0%}, Q p="
               f"{rep['era']['p_heterogeneity']:.3f}) with a consistent sign"
               if not point_flip else
               f"heterogeneous (I2={rep['era']['I2']:.0%}) with a "
               f"non-significant point-sign flip")
    else:
        verdict = "ERA-STABLE"
        why = (f"Q not significant (p={rep['era']['p_heterogeneity']:.3f}), "
               f"I2={rep['era']['I2']:.0%}")
    return dict(verdict=verdict, why=why,
                point_sign_flip=bool(point_flip),
                significant_sign_flip=bool(pos_sig and neg_sig),
                eras_sig_positive=pos_sig, eras_sig_negative=neg_sig,
                era_point_estimates={e: round(v, 5) for e, v in ests.items()})


def term_report(rows, arm, seasons, label):
    rr = [r for r in rows if r["season"] in seasons]
    y = np.array([r["y"] for r in rr])
    d = ll(y, [r[f"p_{arm}"] for r in rr]) - ll(y, [r["p_base"] for r in rr])
    pan = S.Panel(np.array([r["season"] for r in rr]), d,
                  date=np.array([r["game_date"] for r in rr]),
                  label=f"{TERMS[arm]} — {label}")
    rep = S.full_report(pan, B, SEED)
    moved = np.array([abs(r[f"p_{arm}"] - r["p_base"]) > 1e-12 for r in rr])
    out = dict(
        arm=arm, term=TERMS[arm], group=label, n=len(rr),
        n_moved=int(moved.sum()), moved_frac=round(float(moved.mean()), 4),
        pooled=dict(est=round(rep["pooled"]["est"], 6),
                    lo=round(rep["pooled"]["lo"], 6),
                    hi=round(rep["pooled"]["hi"], 6),
                    sig=rep["pooled"]["sig"]),
        mde80=round(rep["pooled_mde80"], 6),
        season_cluster=dict(lo=round(rep["clustering"]["season_cluster_boot"]["lo"], 6),
                            hi=round(rep["clustering"]["season_cluster_boot"]["hi"], 6),
                            sig=rep["clustering"]["season_cluster_boot"]["sig"]),
        season_mean_t=dict(lo=round(rep["clustering"]["season_mean_t"]["lo"], 6),
                           hi=round(rep["clustering"]["season_mean_t"]["hi"], 6),
                           sig=rep["clustering"]["season_mean_t"]["sig"],
                           dof=rep["clustering"]["season_mean_t"]["dof"]),
        icc_season=round(rep["clustering"]["icc_season"]["icc"], 6),
        design_effect_season=round(rep["clustering"]["design_effect_season"], 3),
        per_season={p["season"]: dict(est=round(p["est"], 6),
                                      lo=round(p["lo"], 6), hi=round(p["hi"], 6),
                                      sig=p["sig"], n=p["n"],
                                      era=S.era_of(p["season"]))
                    for p in rep["per_season"]},
        per_era={e["era"]: dict(est=round(e["est"], 6), lo=round(e["lo"], 6),
                                hi=round(e["hi"], 6), sig=e["sig"], n=e["n"],
                                seasons=e["seasons"])
                 for e in rep["era"]["per_era"]},
        heterogeneity=dict(Q=round(rep["era"]["Q"], 3), dof=rep["era"]["dof"],
                           p=round(rep["era"]["p_heterogeneity"], 4),
                           I2=round(rep["era"]["I2"], 4),
                           tau=round(rep["era"]["tau"], 6),
                           between_share=round(rep["era"]["between_share"], 4)),
        rolling_origin=dict(sign=rep["rolling_origin"]["sign_consistency"],
                            drift=round(rep["rolling_origin"]["drift_per_season"], 6),
                            folds=[dict(test=f["test"], era=f["era"],
                                        est=round(f["fold"]["est"], 6),
                                        sig=f["fold"]["sig"])
                                   for f in rep["rolling_origin"]["folds"]]),
        block_bootstrap=(dict(lo=round(rep["block_bootstrap"]["lo"], 6),
                              hi=round(rep["block_bootstrap"]["hi"], 6),
                              sig=rep["block_bootstrap"]["sig"])
                         if "block_bootstrap" in rep else None),
        flags=rep["verdict"]["flags"] if isinstance(rep["verdict"], dict) else None,
    )
    out["classification"] = classify(rep, rep["era"]["per_era"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "history_pergame.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "history_analysis.json"))
    a = ap.parse_args()
    rows = load(a.csv)
    tab = season_table(rows)
    allseasons = sorted(tab)
    hist_pool = [s for s in allseasons if s not in CERT and s not in STRATA]
    pool_all = [s for s in allseasons if s not in STRATA]

    res = {"csv": a.csv, "per_season": tab,
           "pooled": {k: v for k, v in {
               "certified_5": pooled(rows, CERT, "certified corpus 2021-26"),
               "historical_new": pooled(rows, hist_pool, "new historical seasons"),
               "all_poolable": pooled(rows, pool_all, "all poolable seasons"),
           }.items() if v},
           "strata": {s: pooled(rows, [s], f"stratum {s}") for s in STRATA
                      if s in tab},
           "drivers": drivers(tab)}

    res["ablation"] = {}
    groups = {"ALL": pool_all, "HISTORICAL_NEW": hist_pool, "CERTIFIED_5": list(CERT)}
    for arm in TERMS:
        if f"p_{arm}" not in rows[0]:
            continue
        res["ablation"][arm] = {}
        for g, ss in groups.items():
            ss = [s for s in ss if s in tab]
            if len(ss) < 2:
                continue
            res["ablation"][arm][g] = term_report(rows, arm, ss, g)
        print(f"\n=== {TERMS[arm]} ===", flush=True)
        for g in groups:
            r = res["ablation"][arm].get(g)
            if not r:
                continue
            c = r["classification"]
            print(f"  {g:15s} {r['pooled']['est']:+.5f} "
                  f"CI({r['pooled']['lo']:+.5f},{r['pooled']['hi']:+.5f}) "
                  f"{'SIG' if r['pooled']['sig'] else 'ns '} | "
                  f"seasclust {'SIG' if r['season_cluster']['sig'] else 'ns '} | "
                  f"t{r['season_mean_t']['dof']} "
                  f"{'SIG' if r['season_mean_t']['sig'] else 'ns '} | "
                  f"I2={r['heterogeneity']['I2']:.0%} Qp={r['heterogeneity']['p']:.3f} | "
                  f"RO {r['rolling_origin']['sign']} | {c['verdict']}"
                  + ("  ** SIGN FLIP **" if c["significant_sign_flip"] else ""),
                  flush=True)
        r = res["ablation"][arm].get("ALL")
        if r:
            print("     per-era: " + "  ".join(
                f"{e}{'*' if v['sig'] else ''}={v['est']:+.5f}"
                for e, v in sorted(r["per_era"].items(),
                                   key=lambda kv: S.ERA_ORDER.index(kv[0])
                                   if kv[0] in S.ERA_ORDER else 99)), flush=True)

    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(f"\nwrote {a.out}")

    print("\n=== PER-SEASON HEADLINE ===")
    print(f"{'season':9s} {'era':4s} {'n':>5s} {'ll_us':>8s} {'ll_mkt':>8s} "
          f"{'raw':>9s} {'norm%':>7s}  note")
    for s in allseasons:
        t = tab[s]
        note = ("CERTIFIED CORPUS" if t["in_cert_corpus"]
                else ("STRATUM — not pooled" if t["stratum"] else "NEW (never gated on)"))
        print(f"{s:9s} {t['era']:4s} {t['n']:5d} {t['ll_us']:8.5f} "
              f"{t['ll_mkt']:8.5f} {t['raw_gap']:+9.5f} {t['norm_gap_pct']:+7.2f}  {note}")
    for k, v in res["pooled"].items():
        print(f"{v['label']:32s} n={v['n']:5d} ll_us={v['ll_us']:.5f} "
              f"ll_mkt={v['ll_mkt']:.5f} raw={v['raw_gap']:+.5f} "
              f"norm={v['norm_gap_pct']:+.2f}%")


if __name__ == "__main__":
    main()
