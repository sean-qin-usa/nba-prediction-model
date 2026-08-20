#!/usr/bin/env python3
"""D235 — INTEGRATION AUDIT: does D232 survive the offset layer?

THE ORDERING PROBLEM D232 DID NOT TEST.  D232 gated the absence term on the
MARKET-BLIND margin.  Production does not forecast with that margin — it passes
it through the market-offset layer:

    m_final = m_open + 0.3564*(m_blind - m_open) + 0.0417*rest - 0.0114*|m_open|

Layers do not commute: `offset(m + absence) != offset(m) + absence`.  The
absence term is added BEFORE the offset, so its effect on the forecast that is
actually made is multiplied by the edge coefficient:

    -0.8284 pts per absence  ->  -0.2953 pts per absence      2.81x attenuation

So D232's -0.002174 nats is an improvement to an INTERMEDIATE quantity. This
script measures the improvement to the thing production actually emits.

THE OFFSET IS REFIT PER ARM.  When an upstream representation changes, a
downstream learned layer fitted on the old one is no longer the right
comparator — leaving it fixed would credit or penalise the challenger for a
stale downstream fit rather than for its own content. Each arm therefore gets
its own walk-forward offset AND its own walk-forward link scale.

THIS IS AN AUDIT, NOT FRESH CONFIRMATION.  These seasons were used to develop
D232, so a favourable result here is an integration check — evidence that the
term is not destroyed by the layer above it — and NOT independent confirmation.
The clean test is the frozen 2026-27 shadow.

Also runs the CLEANER CONFOUND CONTROL: D232 controlled for `m_total`, which
already contains the availability-sensitive composition channel and is therefore
partly the treatment (a "bad control"). `m_ff` — the four-factors channel — takes
no availability input at all, so it is the pre-availability strength estimate the
control should have used.
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

LAM_OFFSET = 3000.0          # the production offset's shrinkage (D204/D225)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def load():
    chan = pd.read_csv(ROOT / "data" / "channel_pergame.csv")   # PRE-absence
    cap = pd.read_csv(ROOT / "data" / "capstone_2019_26.csv")
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    for d in (chan, cap, pit):
        d["game_id"] = zf(d["game_id"])
    f = (chan.merge(cap[["game_id", "eo_home", "eo_away"]], on="game_id",
                    validate="one_to_one")
             .merge(pit[["game_id", "open_margin", "margin_actual",
                         "rest_home", "rest_away"]], on="game_id",
                    validate="one_to_one"))
    f = f.dropna(subset=["open_margin", "margin_actual", "m_total"]).copy()
    f["out_diff"] = f["eo_home"] - f["eo_away"]
    f["rest_diff"] = (f["rest_home"].clip(upper=7).fillna(0)
                      - f["rest_away"].clip(upper=7).fillna(0))
    f["y"] = (f["margin_actual"] > 0).astype(float)
    return f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)


def offset_design(blind, f):
    return np.column_stack([blind - f["open_margin"].to_numpy(float),
                            f["rest_diff"].to_numpy(float),
                            f["open_margin"].abs().to_numpy(float)])


def through_offset(tr, te, blind_tr, blind_te):
    """Fit the offset on the training block with THIS arm's blind margin."""
    X = offset_design(blind_tr, tr)
    r = (tr["margin_actual"] - tr["open_margin"]).to_numpy(float)
    b = np.linalg.solve(X.T @ X + LAM_OFFSET * np.eye(3), X.T @ r)
    m_tr = tr["open_margin"].to_numpy(float) + X @ b
    m_te = te["open_margin"].to_numpy(float) + offset_design(blind_te, te) @ b
    return m_tr, m_te, b


def main():
    f = load()
    seasons = sorted(f["season"].unique())
    print(f"frame {len(f)} games, seasons {seasons[0]}..{seasons[-1]}")

    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f["season"].isin(seasons[:i])], f[f["season"] == s]
        ytr, yte = tr["y"].to_numpy(float), te["y"].to_numpy(float)

        # walk-forward absence beta, exactly as D232 fitted it
        x = tr["out_diff"].to_numpy(float)
        resid = (tr["margin_actual"] - tr["m_total"]).to_numpy(float)
        beta = float((x @ resid) / (x @ x))

        blind_i_tr = tr["m_total"].to_numpy(float)
        blind_i_te = te["m_total"].to_numpy(float)
        blind_c_tr = blind_i_tr + beta * x
        blind_c_te = blind_i_te + beta * te["out_diff"].to_numpy(float)

        # --- BLIND LAYER (what D232 gated) --------------------------------
        s_i = fit_scale(blind_i_tr, ytr)
        s_c = fit_scale(blind_c_tr, ytr)
        ll_bi = nll(1 / (1 + np.exp(-blind_i_te / s_i)), yte).mean()
        ll_bc = nll(1 / (1 + np.exp(-blind_c_te / s_c)), yte).mean()

        # --- FINAL STACK (what production emits) --------------------------
        mi_tr, mi_te, bi = through_offset(tr, te, blind_i_tr, blind_i_te)
        mc_tr, mc_te, bc = through_offset(tr, te, blind_c_tr, blind_c_te)
        si = fit_scale(mi_tr, ytr)
        sc = fit_scale(mc_tr, ytr)
        ll_fi = nll(1 / (1 + np.exp(-mi_te / si)), yte).mean()
        ll_fc = nll(1 / (1 + np.exp(-mc_te / sc)), yte).mean()

        rows.append(dict(season=s, n=len(te), beta=beta,
                         edge_inc=bi[0], edge_chal=bc[0],
                         d_blind=ll_bc - ll_bi, d_final=ll_fc - ll_fi,
                         p_inc=float((1 / (1 + np.exp(-mi_te / si))).mean()),
                         p_chal=float((1 / (1 + np.exp(-mc_te / sc))).mean()),
                         base=float(yte.mean())))

    r = pd.DataFrame(rows)
    print("\n--- per season (negative = challenger better) ---")
    print(r[["season", "n", "beta", "edge_inc", "edge_chal",
             "d_blind", "d_final"]].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))

    def clus(col):
        d = r[col].to_numpy(float)
        k = len(d)
        se = d.std(ddof=1) / np.sqrt(k)
        tc = stats.t.ppf(0.975, k - 1)
        return d.mean(), d.mean() - tc * se, d.mean() + tc * se, d.mean() / se, k

    out = {}
    for col, tag in (("d_blind", "BLIND LAYER (what D232 gated)"),
                     ("d_final", "FINAL STACK (what production emits)")):
        m, lo, hi, t, k = clus(col)
        better = int((r[col] < 0).sum())
        print(f"\n=== {tag} ===")
        print(f"  season-clustered mean delta {m:+.6f}")
        print(f"  95% CI ({k-1} dof)          [{lo:+.6f}, {hi:+.6f}]")
        print(f"  t {t:+.2f}   better in {better}/{k}")
        print(f"  VERDICT: {'CI excludes zero' if hi < 0 else 'CI INCLUDES ZERO'}")
        out[col] = dict(mean=m, ci=[lo, hi], t=t, better=better, k=k)

    di = (r["p_inc"] - r["base"]).abs().mean()
    dc = (r["p_chal"] - r["base"]).abs().mean()
    print(f"\ncalibration drift (final stack): incumbent {di:.4f} "
          f"challenger {dc:.4f} -> {'PASS' if dc <= di + 1e-6 else 'VETO'}")
    print(f"attenuation: mean |d_blind| {r['d_blind'].abs().mean():.6f} -> "
          f"mean |d_final| {r['d_final'].abs().mean():.6f}")

    # ---- cleaner confound control: m_ff takes NO availability input ------
    print("\n=== CLEANER CONFOUND CONTROL (m_ff, availability-insensitive) ===")
    for name, ctrl in (("m_total (D232, a BAD control — contains the treatment)",
                        "m_total"),
                       ("m_ff (pre-availability strength)", "m_ff")):
        bs = []
        for s in seasons:
            sub = f[f["season"] == s]
            X = np.column_stack([sub["out_diff"].to_numpy(float),
                                 sub[ctrl].to_numpy(float),
                                 np.ones(len(sub))])
            yv = (sub["margin_actual"] - sub["m_total"]).to_numpy(float)
            bs.append(np.linalg.lstsq(X, yv, rcond=None)[0][0])
        bs = np.array(bs)
        k = len(bs)
        se = bs.std(ddof=1) / np.sqrt(k)
        tc = stats.t.ppf(0.975, k - 1)
        print(f"  control = {name}")
        print(f"    beta {bs.mean():+.4f}  95% CI [{bs.mean()-tc*se:+.4f}, "
              f"{bs.mean()+tc*se:+.4f}]  same sign {int((bs<0).sum())}/{k}")
        out[f"ctrl_{ctrl}"] = dict(beta=float(bs.mean()),
                                   ci=[float(bs.mean()-tc*se),
                                       float(bs.mean()+tc*se)])

    json.dump({"rows": rows, **out},
              open(ROOT / "data" / "d235_final_stack.json", "w"), default=float)
    print("\nwrote data/d235_final_stack.json")


if __name__ == "__main__":
    main()
