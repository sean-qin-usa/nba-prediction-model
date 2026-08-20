#!/usr/bin/env python3
"""D245d — L3 (composition margin) and L4 (full stack). Prereg 1d145732...

THE DECISIVE COMPARISON AT BOTH LAYERS IS PRIMARY vs N1, NOT PRIMARY vs
CONTROL. Otherwise total-minute normalisation collects credit as tier alpha --
D245b measured that N1 alone supplies 43% of the L2 gain.

  L3  composition margin vs realised margin, each arm affine-recalibrated on
      training folds only (the D242 scale trap: a 240-conserving allocation
      makes sum m_i/48 == 5 exactly, so level and dispersion change for reasons
      unrelated to the hypothesis)
  L4  FULL STACK. Every downstream layer refitted per arm per fold -- the
      composition blend weight, the offset ridge and the probability link --
      because an arm whose upstream representation changed cannot be judged
      against a downstream fitted on the old one (D235's lesson).

Arms: CONTROL, N1, PRIMARY(alpha), FIXED 160/80 (diagnostic, ineligible).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402
from scipy.optimize import minimize_scalar                        # noqa: E402

SCR = Path(sys.argv[1])
ARMS = {"ctrl": "ch_ctrl", "n1": "ch_B_N1", "prim": "ch_ALPHA", "fix": "ch_B_N3"}
LAM_OFFSET = 3000.0


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def main():
    base = None
    for tag, fn in ARMS.items():
        d = pd.read_csv(SCR / f"{fn}.csv")
        d["game_id"] = zf(d["game_id"])
        keep = d[["game_id", "season", "game_date", "m_ff", "m_comp",
                  "m_sched", "m_tank"]].rename(
            columns={"m_comp": f"comp_{tag}"})
        base = keep if base is None else base.merge(
            keep[["game_id", f"comp_{tag}"]], on="game_id")
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    pit["game_id"] = zf(pit["game_id"])
    f = base.merge(pit[["game_id", "margin_actual", "open_margin",
                        "rest_home", "rest_away"]], on="game_id")
    f = f.dropna(subset=["margin_actual", "open_margin"]).copy()
    f["y"] = (f["margin_actual"] > 0).astype(float)
    f["rest_diff"] = (f["rest_home"].clip(upper=7).fillna(0)
                      - f["rest_away"].clip(upper=7).fillna(0))
    f = f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(f.season.unique())
    print(f"frame {len(f):,} games, {len(seasons)} seasons")

    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        ytr, yte = tr.y.to_numpy(float), te.y.to_numpy(float)
        r = {"season": s, "n": len(te)}
        for tag in ARMS:
            c_tr = tr[f"comp_{tag}"].to_numpy(float)
            c_te = te[f"comp_{tag}"].to_numpy(float)
            # ---- L3: affine recalibration of the composition margin only
            b, a = np.polyfit(c_tr, tr.margin_actual.to_numpy(float), 1)
            r[f"l3_{tag}"] = float(np.sqrt(
                ((te.margin_actual - (a + b * c_te)) ** 2).mean()))

            # ---- L4: refit BLEND, OFFSET and LINK per arm per fold
            # blend: m_blind = w*ff + (1-w)*comp + sched + tank, w fitted
            X = np.column_stack([tr.m_ff, c_tr])
            tgt = (tr.margin_actual - tr.m_sched - tr.m_tank).to_numpy(float)
            w = np.linalg.lstsq(X, tgt, rcond=None)[0]
            mb_tr = X @ w + tr.m_sched + tr.m_tank
            mb_te = (np.column_stack([te.m_ff, c_te]) @ w
                     + te.m_sched + te.m_tank)
            # offset: ridge on (edge, rest, |open|), refit on this arm
            def des(mb, d):
                return np.column_stack([mb - d.open_margin.to_numpy(float),
                                        d.rest_diff.to_numpy(float),
                                        d.open_margin.abs().to_numpy(float)])
            Xo = des(mb_tr, tr)
            resid = (tr.margin_actual - tr.open_margin).to_numpy(float)
            bo = np.linalg.solve(Xo.T @ Xo + LAM_OFFSET * np.eye(3), Xo.T @ resid)
            mo_tr = tr.open_margin.to_numpy(float) + Xo @ bo
            mo_te = te.open_margin.to_numpy(float) + des(mb_te, te) @ bo
            sc = fit_scale(mo_tr, ytr)               # link refit per arm
            r[f"l4_{tag}"] = float(nll(1 / (1 + np.exp(-mo_te / sc)), yte).mean())
            r[f"w_{tag}"] = float(w[0] / (w[0] + w[1])) if (w[0] + w[1]) else np.nan
            r[f"edge_{tag}"] = float(bo[0])
        rows.append(r)
    d = pd.DataFrame(rows)

    print("\n=== L3 composition-margin RMSE (affine-recalibrated per fold) ===")
    print(d[["season", "n"] + [f"l3_{t}" for t in ARMS]].to_string(
        index=False, float_format=lambda v: f"{v:8.4f}"))
    print("\n=== L4 FULL-STACK opening-line log loss (blend+offset+link refit) ===")
    print(d[["season"] + [f"l4_{t}" for t in ARMS]].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))
    print("\n  fitted four-factor blend share / offset edge coefficient:")
    for t in ARMS:
        print(f"    {t:5}  w_ff {d[f'w_{t}'].mean():.3f}   "
              f"edge {d[f'edge_{t}'].mean():.4f}")

    out = {}
    for lvl in ("l3", "l4"):
        print(f"\n=== {lvl.upper()} COMPARISONS ===")
        for a, b, lab in (("prim", "ctrl", "PRIMARY vs CONTROL (norm + tier)"),
                          ("n1", "ctrl", "N1 vs CONTROL (240-normalisation)"),
                          ("prim", "n1", "**PRIMARY vs N1 (tier increment)**"),
                          ("fix", "prim", "FIXED vs PRIMARY (diagnostic)")):
            v = (d[f"{lvl}_{a}"] - d[f"{lvl}_{b}"]).to_numpy()
            m, lo, hi, t, k = clus(v)
            flag = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "ns")
            print(f"  {lab:38} {m:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  "
                  f"t {t:+6.2f}  better {int((v<0).sum())}/{k}  {flag}")
            out[f"{lvl}_{a}_vs_{b}"] = dict(mean=float(m), ci=[float(lo), float(hi)],
                                            t=float(t), better=int((v < 0).sum()),
                                            k=k, flag=flag)
    json.dump({"rows": rows, "comparisons": out},
              open(ROOT / "data" / "d245d_l3l4.json", "w"), default=float)
    print("\nwrote data/d245d_l3l4.json")


if __name__ == "__main__":
    main()
