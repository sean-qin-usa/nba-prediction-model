#!/usr/bin/env python3
"""D230 — CHANNEL-WISE TRUST IN THE OFFSET LAYER. Prereg sha256 6ce46dcf...

    incumbent   m_off = m_open + b*(m_blind - m_open) + g*rest + d*|m_open|
    challenger  m_off = m_open + b*(m_blind - m_open) + SUM_k dk*c_k
                              + g*rest + d*|m_open|

with an L2 penalty on the dk ONLY. At lam -> inf every dk -> 0 and the
challenger IS the incumbent, so the null is the shipped model rather than zero
(D198's rule). Both arms are refitted walk-forward by the same code on the same
folds, so the comparison cannot be won by giving one arm fresher coefficients.

Everything is fitted on seasons 1..k and scored on k+1 alone.
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
from scipy.optimize import minimize_scalar                        # noqa: E402

CH = ["m_ff", "m_comp", "m_sched", "m_tank"]        # m_late is identically 0
LAMS = [1e9, 3e4, 1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3e0, 1e0]


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def load() -> pd.DataFrame:
    chan = pd.read_csv(sys.argv[1] if len(sys.argv) > 1
                       else ROOT / "data" / "channel_pergame.csv")
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    # GAME-ID JOIN GUARD. This project has already lost a whole analysis to an
    # int-vs-zero-padded-VARCHAR join that silently matched nothing (D197), so
    # both sides are normalised and the match rate is ASSERTED, not inspected.
    for d in (chan, pit):
        d["game_id"] = d["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        d["game_id"] = d["game_id"].str.zfill(10)
    keep = ["game_id", "open_margin", "margin_actual", "rest_home", "rest_away",
            "close_margin"]
    f = chan.merge(pit[keep], on="game_id", how="inner", validate="one_to_one")
    rate = len(f) / len(chan)
    print(f"join: {len(chan)} channel rows x {len(pit)} pit rows -> {len(f)} "
          f"({rate:.1%})")
    assert rate > 0.90, f"join collapsed at {rate:.1%} — check game_id format"
    f = f.dropna(subset=["open_margin", "margin_actual"] + CH).copy()
    f["rest_diff"] = (f["rest_home"].clip(upper=7).fillna(0)
                      - f["rest_away"].clip(upper=7).fillna(0))
    f["m_blind"] = f[CH].sum(axis=1)
    f["y"] = (f["margin_actual"] > 0).astype(float)
    f["target"] = f["margin_actual"] - f["open_margin"]      # opener's error
    return f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)


def design(f: pd.DataFrame, channels: bool):
    """[edge, rest, |open|] (+ the channel deviations when `channels`)."""
    cols = [(f["m_blind"] - f["open_margin"]).to_numpy(float),
            f["rest_diff"].to_numpy(float),
            f["open_margin"].abs().to_numpy(float)]
    if channels:
        cols += [f[c].to_numpy(float) for c in CH]
    return np.column_stack(cols)


def ridge_fit(X, y, lam, n_free=3):
    """L2 on the trailing (channel-deviation) columns only."""
    P = np.eye(X.shape[1]) * lam
    P[:n_free, :n_free] = 0.0
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def placebo_mde80(f, seasons, n_rep=12, seed=230):
    """sd of per-season deltas under a within-season permutation null."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    deltas = []
    for rep in range(n_rep):
        g = f.copy()
        for s in seasons:                      # permute channels within season
            m = (g["season"] == s).to_numpy()
            idx = rng.permutation(int(m.sum()))
            for c in CH:
                g.loc[m, c] = g.loc[m, c].to_numpy()[idx]
        g["m_blind"] = g[CH].sum(axis=1)
        for i, s in enumerate(seasons):
            if i == 0:
                continue
            tr, te = g["season"].isin(seasons[:i]), g["season"] == s
            gtr, gte = g[tr], g[te]
            ytr = gtr["y"].to_numpy(float)
            yte = gte["y"].to_numpy(float)
            Xi_tr, Xi_te = design(gtr, False), design(gte, False)
            Xc_tr, Xc_te = design(gtr, True), design(gte, True)
            bi = ridge_fit(Xi_tr, gtr["target"].to_numpy(float), 0.0, n_free=3)
            bc = ridge_fit(Xc_tr, gtr["target"].to_numpy(float), 1e3)
            o_tr = gtr["open_margin"].to_numpy(float)
            o_te = gte["open_margin"].to_numpy(float)
            si = fit_scale(o_tr + Xi_tr @ bi, ytr)
            sc = fit_scale(o_tr + Xc_tr @ bc, ytr)
            lli = nll(1 / (1 + np.exp(-(o_te + Xi_te @ bi) / si)), yte).mean()
            llc = nll(1 / (1 + np.exp(-(o_te + Xc_te @ bc) / sc)), yte).mean()
            deltas.append(llc - lli)
    k = len(seasons) - 1
    sd = float(np.std(deltas, ddof=1))
    return (stats.t.ppf(0.975, k - 1) + stats.t.ppf(0.80, k - 1)) * sd / np.sqrt(k)


def main():
    f = load()
    seasons = sorted(f["season"].unique())
    print(f"frame {len(f)} games, {len(seasons)} seasons {seasons[0]}..{seasons[-1]}")

    # ---- MDE80, STATED BEFORE THE ENDPOINT IS READ -----------------------
    # From the incumbent's own per-game loss dispersion on the scored block:
    # the smallest per-season mean delta an 80%-power test could resolve at
    # K=6 folds. Printed before any challenger number exists below it.
    # The null's OWN dispersion, not a guess: permute the channel block WITHIN
    # each season (destroying any link to the target while preserving every
    # marginal), refit both arms on the same folds, and take the sd of the
    # resulting per-season deltas. That is the dispersion a no-effect arm
    # produces here, which is what a power statement has to be scaled by.
    mde80 = placebo_mde80(f, seasons)
    print(f"MDE80 (stated first, placebo null): {mde80:.5f} nats")

    rows, coefs = [], []
    for i, s in enumerate(seasons):
        if i == 0:
            continue                       # first season is fit-only
        tr = f["season"].isin(seasons[:i])
        te = f["season"] == s
        ftr, fte = f[tr], f[te]

        # incumbent: 3 unpenalised columns
        Xi_tr, Xi_te = design(ftr, False), design(fte, False)
        bi = ridge_fit(Xi_tr, ftr["target"].to_numpy(float), 0.0, n_free=3)
        mi = fte["open_margin"].to_numpy(float) + Xi_te @ bi

        # challenger: pick lam on an INNER split of the training block only
        Xc_tr, Xc_te = design(ftr, True), design(fte, True)
        inner = ftr["season"] != seasons[i - 1]
        best, best_ll = None, np.inf
        if inner.sum() > 100 and (~inner).sum() > 100:
            Xa, Xb = design(ftr[inner], True), design(ftr[~inner], True)
            ya = ftr.loc[inner, "target"].to_numpy(float)
            fb = ftr[~inner]
            for lam in LAMS:
                bb = ridge_fit(Xa, ya, lam)
                mb = fb["open_margin"].to_numpy(float) + Xb @ bb
                sc = fit_scale(mb, fb["y"].to_numpy(float))
                ll = nll(1 / (1 + np.exp(-mb / sc)), fb["y"].to_numpy(float)).mean()
                if ll < best_ll:
                    best_ll, best = ll, lam
        lam = best if best is not None else 1e9
        bc = ridge_fit(Xc_tr, ftr["target"].to_numpy(float), lam)
        mc = fte["open_margin"].to_numpy(float) + Xc_te @ bc

        yte = fte["y"].to_numpy(float)
        # each arm gets its OWN walk-forward scale, fitted on the training
        # block, so neither is handicapped by the other's calibration (D193)
        si = fit_scale(ftr["open_margin"].to_numpy(float) + Xi_tr @ bi,
                       ftr["y"].to_numpy(float))
        sc_ = fit_scale(ftr["open_margin"].to_numpy(float) + Xc_tr @ bc,
                        ftr["y"].to_numpy(float))
        lli = nll(1 / (1 + np.exp(-mi / si)), yte).mean()
        llc = nll(1 / (1 + np.exp(-mc / sc_)), yte).mean()
        rows.append(dict(season=s, n=int(te.sum()), lam=lam,
                         ll_inc=lli, ll_chal=llc, delta=llc - lli))
        coefs.append(dict(season=s, lam=lam, b=bc[0], g=bc[1], d=bc[2],
                          **{f"d_{c}": v for c, v in zip(CH, bc[3:])}))

    r = pd.DataFrame(rows)
    print("\n--- per-season (delta < 0 = challenger better) ---")
    print(r.to_string(index=False, float_format=lambda x: f"{x:9.5f}"))

    dl = r["delta"].to_numpy(float)
    k = len(dl)
    mean = dl.mean()
    se = dl.std(ddof=1) / np.sqrt(k)
    from scipy import stats
    tcrit = stats.t.ppf(0.975, k - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se
    t = mean / se if se > 0 else np.nan
    print(f"\nseason-clustered mean delta {mean:+.6f} nats")
    print(f"95% CI ({k-1} dof)          [{lo:+.6f}, {hi:+.6f}]")
    print(f"t                           {t:+.3f}")
    print(f"better in                   {int((dl<0).sum())}/{k} seasons")
    print(f"MDE80 (stated first)        {mde80:.5f}")
    print(f"\nVERDICT: {'SHIP' if (lo<0 and hi<0) else 'NO SHIP — CI includes zero'}")

    c = pd.DataFrame(coefs)
    print("\n--- channel deviations d_k (DIAGNOSTIC; prereg §6) ---")
    print(c.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print("\nmean d_k across folds:")
    for ch in CH:
        print(f"  d_{ch:9} {c[f'd_{ch}'].mean():+.4f}")
    json.dump({"per_season": rows, "coefs": coefs,
               "mean": mean, "ci": [lo, hi], "t": t, "mde80": mde80},
              open(ROOT / "data" / "d230_channel_offset.json", "w"), default=float)


if __name__ == "__main__":
    main()
