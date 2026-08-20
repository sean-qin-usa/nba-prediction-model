#!/usr/bin/env python3
"""D246b — CLEAN COMPLETION of the movement teacher. Six defects repaired.

THE CENTRAL ONE. Ridge is LINEAR in the target, so

    beta_M + beta_F = R_lam[(C-O) + (Y-C)] = R_lam(Y-O)

Verified numerically at 2.7e-15. With gamma ~ eta ~ 1 the D246 "teacher" was
algebraically a plain ridge on (Y-O) and the closing-line decomposition
CANCELLED. The fitted gamma 1.007 / eta 1.004 were that identity, not evidence
of two distilled signals. **So D246's T3 never tested whether learning close
movement improves the offset.** The clean test is M-ONLY: O + predicted(C-O).

Also repaired:
  * T5 (signed CLV) was PRE-REGISTERED AND NEVER COMPUTED.
  * T4 used an in-sample correlation, not held-out incremental loss.
  * lambda=50 was applied to UNSTANDARDISED features spanning points, tens of
    points, and hundredths -- so head F's null was not a fair test.
  * No CI or MDE80 for head F: "nothing measurable" was asserted from a point
    estimate.
  * "Already captured by the offset" was never ablated against edge-only.

Arms: OFFSET (shipped) · M-ONLY · DIRECT (ridge on Y-O) · M+F ·
      EDGE-ONLY-M and M-WITHOUT-EDGE (diagnostics).
Only M-ONLY and M+F are teacher candidates.
Meta-weights are CROSS-FITTED on inner walk-forward predictions.
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

FROM, LAM = "2019-20", 50.0
FEATS = ["edge", "abs_open", "rest_diff", "eo_diff", "tot_c", "days_c", "mkl_c"]


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


def std_ridge(Xtr, ytr, Xte, lam=LAM):
    """Standardise on TRAINING statistics only; penalise in standardised space;
    intercept unpenalised. D246 penalised raw columns spanning three orders of
    magnitude, which shrank the small-scale features far harder."""
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Ztr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    Zte = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    P = lam * np.eye(Ztr.shape[1]); P[0, 0] = 0.0
    b = np.linalg.solve(Ztr.T @ Ztr + P, Ztr.T @ ytr)
    return Ztr @ b, Zte @ b, b


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f.season >= FROM].copy()
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f["game_date"])
    cap = pd.read_csv(ROOT / "data" / "capstone_2019_26.csv"); cap["game_id"] = zf(cap["game_id"])
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz"); pit["game_id"] = zf(pit["game_id"])
    f = (f.merge(cap[["game_id", "eo_home", "eo_away"]], on="game_id", how="left")
           .merge(pit[["game_id", "rest_home", "rest_away"]], on="game_id", how="left"))
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual", "m_us_blind"])
    f["edge"] = f["m_us_blind"] - f["open_margin"]
    f["abs_open"] = f["open_margin"].abs()
    f["eo_diff"] = (f["eo_home"] - f["eo_away"]).fillna(0.0)
    f["rest_diff"] = (f["rest_home"].clip(upper=7).fillna(0)
                      - f["rest_away"].clip(upper=7).fillna(0))
    t = pd.to_numeric(f["open_total"], errors="coerce").fillna(225.0)
    f["tot_c"] = np.where((t < 150) | (t > 290), 225.0, t) - 225.0
    f["days_c"] = ((f["game_date"] - f.groupby("season")["game_date"]
                    .transform("min")).dt.days) / 100.0
    p = np.clip(1 / (1 + np.exp(-f["open_margin"] / 6.96)), 1e-9, 1 - 1e-9)
    yb = (f["margin_actual"] > 0).astype(float)
    llo = -(yb * np.log(p) + (1 - yb) * np.log(1 - p))
    roll = llo.groupby(f["game_date"]).mean().rolling(60, min_periods=20).mean().shift(1)
    f["mkl_c"] = f["game_date"].map(roll).fillna(0.61) - 0.61
    f["y"] = yb
    f = f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(f.season.unique())
    print(f"frame {len(f):,} games, {len(seasons)} seasons")

    rows, fdiff = [], []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        Xtr, Xte = tr[FEATS].to_numpy(float), te[FEATS].to_numpy(float)
        dM_tr = (tr.close_margin - tr.open_margin).to_numpy(float)
        dF_tr = (tr.margin_actual - tr.close_margin).to_numpy(float)
        dM_te = (te.close_margin - te.open_margin).to_numpy(float)
        dF_te = (te.margin_actual - te.close_margin).to_numpy(float)
        Otr, Ote = tr.open_margin.to_numpy(float), te.open_margin.to_numpy(float)
        ytr, yte = tr.y.to_numpy(float), te.y.to_numpy(float)

        _, pM, _ = std_ridge(Xtr, dM_tr, Xte)
        _, pF, _ = std_ridge(Xtr, dF_tr, Xte)
        _, pD, _ = std_ridge(Xtr, (tr.margin_actual - Otr).to_numpy(float), Xte)
        # diagnostics: edge-only and everything-but-edge
        ei = [FEATS.index("edge")]
        oi = [j for j in range(len(FEATS)) if j != ei[0]]
        _, pM_edge, _ = std_ridge(Xtr[:, ei], dM_tr, Xte[:, ei])
        _, pM_noedge, _ = std_ridge(Xtr[:, oi], dM_tr, Xte[:, oi])

        # CROSS-FITTED meta-weights: inner walk-forward heads, never in-sample
        inner = sorted(tr.season.unique())
        A_rows, A_y = [], []
        for j, s2 in enumerate(inner):
            if j == 0:
                continue
            itr, ite = tr[tr.season.isin(inner[:j])], tr[tr.season == s2]
            Xi, Xj = itr[FEATS].to_numpy(float), ite[FEATS].to_numpy(float)
            _, qM, _ = std_ridge(Xi, (itr.close_margin - itr.open_margin).to_numpy(float), Xj)
            _, qF, _ = std_ridge(Xi, (itr.margin_actual - itr.close_margin).to_numpy(float), Xj)
            A_rows.append(np.column_stack([qM, qF]))
            A_y.append((ite.margin_actual - ite.open_margin).to_numpy(float))
        if A_rows:
            A = np.vstack(A_rows); ay = np.concatenate(A_y)
            ge = np.linalg.lstsq(A, ay, rcond=None)[0]
        else:
            ge = np.array([1.0, 1.0])

        def ll_of(m_te, m_tr_):
            sc = fit_scale(m_tr_, ytr)
            return float(nll(1 / (1 + np.exp(-m_te / sc)), yte).mean())
        _, _, _ = None, None, None
        mtr_M, _, _ = std_ridge(Xtr, dM_tr, Xtr)
        mtr_D, _, _ = std_ridge(Xtr, (tr.margin_actual - Otr).to_numpy(float), Xtr)
        ll_off = ll_of(te.m_us.to_numpy(float), tr.m_us.to_numpy(float))
        ll_M = ll_of(Ote + pM, Otr + mtr_M)
        ll_D = ll_of(Ote + pD, Otr + mtr_D)
        ll_MF = ll_of(Ote + ge[0] * pM + ge[1] * pF, Otr + mtr_D)

        # T5: SIGNED CLV — pre-registered in D246 and never computed
        side_off = np.sign(te.m_us.to_numpy(float) - Ote)
        side_M = np.sign(pM)
        clv_off = float(np.mean(side_off * dM_te))
        clv_M = float(np.mean(side_M * dM_te))

        # head F paired dMSE against predicting zero
        d_i = (dF_te - pF) ** 2 - dF_te ** 2
        fdiff.append(dict(season=s, mean=float(d_i.mean()),
                          n=len(d_i), sd=float(d_i.std(ddof=1))))
        rows.append(dict(
            season=s, n=len(te),
            r2_M=float(1 - ((dM_te - pM) ** 2).sum() / (dM_te ** 2).sum()),
            r2_M_edge=float(1 - ((dM_te - pM_edge) ** 2).sum() / (dM_te ** 2).sum()),
            r2_M_noedge=float(1 - ((dM_te - pM_noedge) ** 2).sum() / (dM_te ** 2).sum()),
            r2_F=float(1 - ((dF_te - pF) ** 2).sum() / (dF_te ** 2).sum()),
            gamma=float(ge[0]), eta=float(ge[1]),
            ll_off=ll_off, ll_M=ll_M, ll_D=ll_D, ll_MF=ll_MF,
            clv_off=clv_off, clv_M=clv_M))
    d = pd.DataFrame(rows)
    print("\n" + d[["season", "r2_M", "r2_M_edge", "r2_M_noedge", "r2_F",
                    "gamma", "eta"]].to_string(index=False,
                                               float_format=lambda v: f"{v:8.4f}"))
    print("\n" + d[["season", "ll_off", "ll_M", "ll_D", "ll_MF",
                    "clv_off", "clv_M"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:9.5f}"))

    print("\n=== THE TEST D246 NEVER RAN: M-ONLY vs the SHIPPED OFFSET ===")
    for a, b, lab in (("ll_M", "ll_off", "M-ONLY vs OFFSET"),
                      ("ll_MF", "ll_M", "M+F vs M-ONLY (does F add?)"),
                      ("ll_D", "ll_off", "DIRECT ridge(Y-O) vs OFFSET"),
                      ("ll_MF", "ll_D", "M+F vs DIRECT (identity check)")):
        m, lo, hi, t_, k = clus(d[a] - d[b])
        print(f"  {lab:34} {m:+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
              f"better {int(((d[a]-d[b])<0).sum())}/{k}  "
              f"{'BETTER' if hi < 0 else ('WORSE' if lo > 0 else 'ns')}")

    print("\n=== T5, PRE-REGISTERED AND PREVIOUSLY UNCOMPUTED: SIGNED CLV ===")
    m, lo, hi, t_, k = clus(d.clv_M - d.clv_off)
    print(f"  offset mean CLV {d.clv_off.mean():+.4f} pts;  "
          f"M-only mean CLV {d.clv_M.mean():+.4f} pts")
    print(f"  difference {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"better {int(((d.clv_M-d.clv_off)<0).sum())}/{k}  "
          f"{'M-ONLY BETTER' if lo > 0 else ('OFFSET BETTER' if hi < 0 else 'ns')}")

    print("\n=== HEAD F, WITH POWER — not a bare point estimate ===")
    fd = pd.DataFrame(fdiff)
    m, lo, hi, t_, k = clus(fd["mean"])
    pooled_sd = float(np.sqrt((fd.sd ** 2 * fd.n).sum() / fd.n.sum()))
    mde = (stats.t.ppf(0.975, k - 1) + stats.t.ppf(0.80, k - 1)) * \
        fd["mean"].std(ddof=1) / np.sqrt(k)
    print(f"  paired dMSE vs predicting zero: {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  MDE80 {mde:.4f}   mean r2_F {d.r2_F.mean():+.5f}")
    print(f"  -> {'F improves' if hi < 0 else 'no point-estimate improvement under this spec'}")

    print("\n=== ABLATION: is movement prediction just the offset's edge? ===")
    print(f"  full 7 features   r2 {d.r2_M.mean():+.4f}")
    print(f"  edge only         r2 {d.r2_M_edge.mean():+.4f}")
    print(f"  everything BUT edge r2 {d.r2_M_noedge.mean():+.4f}")
    json.dump({"rows": rows, "f_paired": fdiff},
              open(ROOT / "data" / "d246b_completion.json", "w"), default=float)
    print("\nwrote data/d246b_completion.json")


if __name__ == "__main__":
    main()
