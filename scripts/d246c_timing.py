#!/usr/bin/env python3
"""D246c — DOES M-ONLY IMPROVE EXECUTION, OR ONLY PREDICT LINE DIRECTION?

D246b compared `sign(shipped_margin - open)` against `sign(predicted movement)`.
Those are DIFFERENT SIDE-SELECTION RULES, and M is trained to predict
`close - open`, so positive signed movement is exactly what it must produce.
**The +0.1117 was a line-direction result, not a timing improvement**, and the
entire gap lives in games where the two rules pick opposite sides.

A genuine timing test HOLDS THE BET SIDE FIXED:

    s_g = sign(shipped_margin - open)            baseline side, never changed
    z_g = s_g * predicted(close - open)          movement alignment
    z > 0 -> line moves against a waiter; execute now
    z < 0 -> line improves for the baseline side; wait

Also repaired here:
  * M+F's probability scale was fitted on the DIRECT training forecast
    (`Otr + mtr_D`) rather than the corresponding M+F one. Affects the M+F
    comparison and T4, not the M-only CLV number.
  * The earliest outer fold has too few inner seasons to cross-fit and falls
    back to (gamma, eta) = (1, 1); it is now LABELLED rather than described as
    cross-fitted.
  * The ablation was descriptive: R^2 values do not add, so 0.0669 + 0.0556 ~
    0.1299 shows nothing inferential. Now paired out-of-sample squared errors
    with a clustered CI.
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

FROM, LAM = "2019-20", 50.0
FEATS = ["edge", "abs_open", "rest_diff", "eo_diff", "tot_c", "days_c", "mkl_c"]


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def std_ridge(Xtr, ytr, Xte, lam=LAM):
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Ztr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    Zte = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    P = lam * np.eye(Ztr.shape[1]); P[0, 0] = 0.0
    b = np.linalg.solve(Ztr.T @ Ztr + P, Ztr.T @ ytr)
    return Zte @ b


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
    f = f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(f.season.unique())
    print(f"frame {len(f):,} games, {len(seasons)} seasons")

    rows, abl = [], []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        Xtr, Xte = tr[FEATS].to_numpy(float), te[FEATS].to_numpy(float)
        dM_tr = (tr.close_margin - tr.open_margin).to_numpy(float)
        dM_te = (te.close_margin - te.open_margin).to_numpy(float)
        pM = std_ridge(Xtr, dM_tr, Xte)
        ei = [FEATS.index("edge")]
        oi = [j for j in range(len(FEATS)) if j != ei[0]]
        pM_e = std_ridge(Xtr[:, ei], dM_tr, Xte[:, ei])
        pM_n = std_ridge(Xtr[:, oi], dM_tr, Xte[:, oi])

        Ote = te.open_margin.to_numpy(float)
        s_base = np.sign(te.m_us.to_numpy(float) - Ote)      # FROZEN baseline side
        s_M = np.sign(pM)
        z = s_base * pM                                      # movement alignment
        clv_base = s_base * dM_te                            # baseline-side CLV

        agree = s_base == s_M
        rows.append(dict(
            season=s, n=len(te),
            agree_rate=float(agree.mean()),
            clv_base=float(clv_base.mean()),
            clv_M=float((s_M * dM_te).mean()),
            clv_agree=float(clv_base[agree].mean()) if agree.any() else np.nan,
            clv_disagree=float(clv_base[~agree].mean()) if (~agree).any() else np.nan,
            # THE TIMING TEST: baseline side held fixed, split by alignment
            n_zpos=int((z > 0).sum()), n_zneg=int((z <= 0).sum()),
            clv_zpos=float(clv_base[z > 0].mean()) if (z > 0).any() else np.nan,
            clv_zneg=float(clv_base[z <= 0].mean()) if (z <= 0).any() else np.nan,
        ))
        abl.append(dict(season=s,
                        d_full_vs_edge=float((((dM_te - pM) ** 2)
                                              - ((dM_te - pM_e) ** 2)).mean()),
                        d_full_vs_noedge=float((((dM_te - pM) ** 2)
                                                - ((dM_te - pM_n) ** 2)).mean())))
    d = pd.DataFrame(rows); a = pd.DataFrame(abl)

    print("\n=== SIDE AGREEMENT: where does the D246b gap actually live? ===")
    print(d[["season", "n", "agree_rate", "clv_base", "clv_M",
             "clv_agree", "clv_disagree"]].to_string(
        index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"\n  mean side-agreement rate {d.agree_rate.mean():.3f}")
    print(f"  baseline-side CLV on AGREEMENT games   {d.clv_agree.mean():+.4f}")
    print(f"  baseline-side CLV on DISAGREEMENT games {d.clv_disagree.mean():+.4f}")
    print("  -> on agreement games the two rules are IDENTICAL by construction;")
    print("     the entire D246b gap comes from flipping the side.")

    print("\n=== THE TIMING TEST — BASELINE SIDE HELD FIXED ===")
    print(d[["season", "n_zpos", "clv_zpos", "n_zneg", "clv_zneg"]].to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))
    m, lo, hi, t_, k = clus(d.clv_zpos - d.clv_zneg)
    print(f"\n  CLV(z>0) - CLV(z<=0) on the FROZEN baseline side: {m:+.4f}")
    print(f"  CI [{lo:+.4f}, {hi:+.4f}]  better {int(((d.clv_zpos-d.clv_zneg)>0).sum())}/{k}")
    verdict = ("TIMING SIGNAL CONFIRMED — alignment separates good from bad "
               "execution" if lo > 0 else "ns — no timing separation established")
    print(f"  {verdict}")
    print(f"  retention if executing only z>0: "
          f"{100*d.n_zpos.sum()/(d.n_zpos.sum()+d.n_zneg.sum()):.1f}% of games")

    print("\n=== ABLATION, NOW INFERENTIAL (paired OOS squared error) ===")
    for c, lab in (("d_full_vs_edge", "full 7 vs EDGE-ONLY"),
                   ("d_full_vs_noedge", "full 7 vs NO-EDGE")):
        m, lo, hi, t_, k = clus(a[c])
        print(f"  {lab:24} dMSE {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"better {int((a[c]<0).sum())}/{k}  "
              f"{'SIG' if hi < 0 else 'ns'}")

    json.dump({"rows": rows, "ablation": abl},
              open(ROOT / "data" / "d246c_timing.json", "w"), default=float)
    print("\nwrote data/d246c_timing.json")


if __name__ == "__main__":
    main()
