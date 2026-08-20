"""CF-2 — what the corpus-floor relaxation did to the CAPSTONE, and the
warm-coefficient per-point efficacy (D112 JOB 1b / JOB 2 secondary).

Two arms of scripts/cf_holdout_retest.py differ ONLY in TANK_SEASON_FLOOR
(2022-23 = pre-D112 shipped, 2021-22 = derived). Both base arms are anchored
to the literal fit_production of their own arm, so pairing them game-by-game
is a clean same-code / same-corpus / one-knob comparison.

REPORTS
  (1) ARM_OLD reproduces the registered D110 run (data/of_transfer_pergame.csv)
      -> proves nothing else in the corpus moved under us.
  (2) per-season ll / gap / normalized gap, OLD vs NEW, plus the paired
      bootstrap on the per-game log-loss delta (positive = the floor fix HELPS).
      This is the honest answer to "is the current capstone unchanged?".
  (3) per-point EFFICACY (scripts/of_term_efficacy.py logic, verbatim) on the
      NEW arm — the D110 normalization, now with warm coefficients.

Usage: python scripts/cf_floor_effect.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from of_transfer_ablation import SEED, TERMS, paired_bootstrap  # noqa: E402
from of_transfer_table import DEV, GATE, HELD, HELD1  # noqa: E402

SCALE = 7.2
LN2 = 0.6931471805599453
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k.startswith(("p_", "act_", "gp_")) or k == "y":
                r[k] = float(r[k])
    return rows


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def ratio_boot(num, den, n_boot=2000, seed=SEED):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0 or den.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return {"n_active": int(len(num)), "efficacy": float(num.sum() / den.sum()),
            "lo": float(np.percentile(r, 2.5)),
            "hi": float(np.percentile(r, 97.5)),
            "mean_abs_term_pts": float(np.abs(den).mean()),
            "effect_per_active_game": float(num.mean())}


def main():
    old = load(ROOT / "data" / "cf_holdout_old_pergame.csv")
    new = load(ROOT / "data" / "cf_holdout_new_pergame.csv")
    res = {}

    # --- (1) does ARM_OLD reproduce the registered D110 run? ---------------
    d110 = {}
    p = ROOT / "data" / "of_transfer_pergame.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            d110[(r["season"], r["game_id"])] = float(r["p_base"])
    diffs = [abs(r["p_base"] - d110[(r["season"], r["game_id"])])
             for r in old if (r["season"], r["game_id"]) in d110]
    res["arm_old_vs_registered_D110"] = {
        "n": len(diffs), "max_abs_dp": max(diffs) if diffs else None}

    # --- (2) capstone: OLD vs NEW, per season -----------------------------
    oi = {(r["season"], r["game_id"]): r for r in old}
    res["capstone"] = {}
    pooled = []
    for s in SEASONS:
        ns = [r for r in new if r["season"] == s]
        pairs = [(oi[(s, r["game_id"])], r) for r in ns
                 if (s, r["game_id"]) in oi]
        y = np.array([r["y"] for _, r in pairs])
        po = np.array([o["p_base"] for o, _ in pairs])
        pn = np.array([r["p_base"] for _, r in pairs])
        pm = np.array([r["p_mkt"] for _, r in pairs])
        lo_, ln_, lm = ll(y, po).mean(), ll(y, pn).mean(), ll(y, pm).mean()
        d = ll(y, po) - ll(y, pn)          # positive = NEW (floor fix) helps
        pooled.append(d)
        res["capstone"][s] = {
            "n": len(pairs),
            "ll_old": round(float(lo_), 5), "ll_new": round(float(ln_), 5),
            "ll_mkt": round(float(lm), 5),
            "gap_old": round(float(lo_ - lm), 5),
            "gap_new": round(float(ln_ - lm), 5),
            "norm_gap_old_pct": round(100 * float(lo_ - lm) / float(LN2 - lm), 2),
            "norm_gap_new_pct": round(100 * float(ln_ - lm) / float(LN2 - lm), 2),
            "max_abs_dp": float(np.abs(po - pn).max()),
            "n_moved": int((np.abs(po - pn) > 1e-12).sum()),
            "paired": paired_bootstrap(d)}
        c = res["capstone"][s]
        print(f"{s}: ll {c['ll_old']:.5f} -> {c['ll_new']:.5f} | mkt {c['ll_mkt']:.5f}"
              f" | norm {c['norm_gap_old_pct']:.2f}% -> {c['norm_gap_new_pct']:.2f}%"
              f" | delta {c['paired']['mean']:+.5f} "
              f"CI({c['paired']['lo']:+.5f},{c['paired']['hi']:+.5f})"
              f" {c['paired']['verdict']} moved={c['n_moved']}", flush=True)
    for name, ss in (("POOLED_ALL", SEASONS), ("DEV_24_26", DEV),
                     ("HELDOUT_21_23", HELD), ("HELDOUT_2223", HELD1),
                     ("GATE_2324", GATE)):
        idx = [i for i, s in enumerate(SEASONS) if s in ss]
        d = np.concatenate([pooled[i] for i in idx])
        res.setdefault("capstone_groups", {})[name] = paired_bootstrap(d)
        b = res["capstone_groups"][name]
        print(f"{name:15s} {b['mean']:+.5f} CI({b['lo']:+.5f},{b['hi']:+.5f})"
              f" n={b['n']} {b['verdict']}", flush=True)

    # --- (3) warm-coefficient per-point efficacy on the NEW arm -----------
    groups = [("DEV 24-26", DEV), ("HELD-OUT 22-23", HELD1),
              ("HELD-OUT 21-23", HELD), ("gate 23-24", GATE)]
    res["efficacy_new"] = {}
    print("\nEFFICACY (nats per POINT of margin applied, active games only) "
          "— NEW arm, warm coefficients")
    for arm, name in TERMS.items():
        if arm not in ("no_tank", "no_late", "no_sched"):
            continue
        print(f"{name}")
        res["efficacy_new"][arm] = {"term": name, "groups": {}}
        for gname, ss in groups:
            rs = [r for r in new if r["season"] in ss]
            y = np.array([r["y"] for r in rs])
            pb = np.array([r["p_base"] for r in rs])
            pa = np.array([r[f"p_{arm}"] for r in rs])
            t = SCALE * (logit(pb) - logit(pa))
            gain = ll(y, pa) - ll(y, pb)
            act = np.abs(t) > 1e-12
            out = ratio_boot(gain[act], np.abs(t[act]))
            res["efficacy_new"][arm]["groups"][gname] = out
            if out is None:
                print(f"  {gname:16s} term never applied — UNTESTABLE")
                continue
            print(f"  {gname:16s} n_act={out['n_active']:5d} "
                  f"mean|term|={out['mean_abs_term_pts']:.3f}pts  "
                  f"EFFICACY={out['efficacy']:+.5f} "
                  f"({out['lo']:+.5f},{out['hi']:+.5f})")
        d = res["efficacy_new"][arm]["groups"]["DEV 24-26"]
        h = res["efficacy_new"][arm]["groups"]["HELD-OUT 22-23"]
        if d and h and d["efficacy"]:
            res["efficacy_new"][arm]["transfer_ratio_2223_over_dev"] = \
                h["efficacy"] / d["efficacy"]
            print(f"  -> held-out efficacy is "
                  f"{100*h['efficacy']/d['efficacy']:.0f}% of dev efficacy")

    # --- (4) difference-in-differences, NEW arm (dev minus held-out) ------
    # independent resamples: the two groups are disjoint game sets
    rng = np.random.default_rng(SEED)
    res["did_new"] = {}
    print("\nDiD (dev effect - held-out effect), NEW arm — positive = the term "
          "helped MORE on its development seasons")
    for arm, name in TERMS.items():
        def eff(ss):
            rs = [r for r in new if r["season"] in ss]
            y = np.array([r["y"] for r in rs])
            return (ll(y, np.array([r[f"p_{arm}"] for r in rs]))
                    - ll(y, np.array([r["p_base"] for r in rs])))
        dd, hh = eff(DEV), eff(HELD)
        if not len(dd) or not len(hh):
            continue
        bd = dd[rng.integers(0, len(dd), size=(2000, len(dd)))].mean(axis=1)
        bh = hh[rng.integers(0, len(hh), size=(2000, len(hh)))].mean(axis=1)
        diff = bd - bh
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        res["did_new"][arm] = {"term": name, "dev": float(dd.mean()),
                               "held": float(hh.mean()),
                               "did": float(dd.mean() - hh.mean()),
                               "lo": lo, "hi": hi,
                               "verdict": "SIG" if lo > 0 or hi < 0 else "NS"}
        e = res["did_new"][arm]
        print(f"  {name:24s} dev {e['dev']:+.5f}  held {e['held']:+.5f}  "
              f"DiD {e['did']:+.5f} CI({lo:+.5f},{hi:+.5f}) {e['verdict']}")

    # --- (5) candidate SHIPPING STACKS, per season (JOB 3 headline) -------
    # every arm of the NEW run is an exact same-run counterfactual capstone,
    # so the surviving stack's headline needs no extra walk-forward.
    res["stacks"] = {}
    print("\nCANDIDATE STACKS (NEW arm, per-season ll / gap / normalized gap)")
    for arm, label in (("p_base", "full (D46+D62+D73+D90+D91)"),
                       ("p_no_late", "D90 late-state REMOVED"),
                       ("p_no_tank", "D73 tank REMOVED")):
        res["stacks"][arm] = {"label": label, "seasons": {}}
        line = []
        for s in SEASONS:
            rs = [r for r in new if r["season"] == s]
            y = np.array([r["y"] for r in rs])
            lu = float(ll(y, np.array([r[arm] for r in rs])).mean())
            lm = float(ll(y, np.array([r["p_mkt"] for r in rs])).mean())
            res["stacks"][arm]["seasons"][s] = {
                "n": len(rs), "ll": round(lu, 5), "ll_mkt": round(lm, 5),
                "gap": round(lu - lm, 5),
                "norm_gap_pct": round(100 * (lu - lm) / (LN2 - lm), 2)}
            line.append(f"{s} {lu:.5f}({100*(lu-lm)/(LN2-lm):.1f}%)")
        y = np.array([r["y"] for r in new])
        lu = float(ll(y, np.array([r[arm] for r in new])).mean())
        lm = float(ll(y, np.array([r["p_mkt"] for r in new])).mean())
        res["stacks"][arm]["pooled"] = {
            "n": len(new), "ll": round(lu, 5), "ll_mkt": round(lm, 5),
            "gap": round(lu - lm, 5),
            "norm_gap_pct": round(100 * (lu - lm) / (LN2 - lm), 2)}
        print(f"  {label:32s} " + "  ".join(line)
              + f"  | pooled gap {lu-lm:+.5f}")

    json.dump(res, open(ROOT / "data" / "cf_floor_effect.json", "w"), indent=1)
    print("\nwrote data/cf_floor_effect.json")


if __name__ == "__main__":
    main()
