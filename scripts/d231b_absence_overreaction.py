#!/usr/bin/env python3
"""D231b — DOES THE MODEL OVER-REACT TO ABSENCES? The bet-time consequence.

D231(A) measured that per-minute production RISES with minutes: each minute a
player logs above his trailing norm is worth +0.0787 pts MORE than the naive
`rate x minutes` bridge predicts, monotone across all ten deciles.

That is the OPPOSITE sign to what D133 arm C and D144 concluded ("a promoted
player's per-minute rates do not survive being scaled to starter minutes"), and
the difference matters directionally for the shipped model.

THE MECHANISM, IF THE SIGN IS RIGHT. When a player is out his minutes are
absorbed by team-mates, whose minutes then sit ABOVE their trailing norms. The
composition leg drops the absent player and carries everyone else at their
unchanged trailing rate, so it credits the replacements too LITTLE. The model
should therefore OVER-PENALISE absences, and the size of the error should scale
with how many players are out.

THE TEST IS BET-TIME CLEAN, WHICH D231(B) WAS NOT. Expected outs come from the
walk-forward P(out) artifact (as-of-open, D201) and are known before tip; the
residual is the realised margin minus the model's margin. Nothing here uses
tonight's minutes.

    residual = margin_actual - m_model
    H0: residual is unrelated to (expected outs home - expected outs away)
    H1: coefficient > 0 -- more HOME absences than away => model too LOW on home

A positive coefficient means the model over-penalises the side with more outs.
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

CAP = sys.argv[1] if len(sys.argv) > 1 else "data/capstone_2019_26.csv"
CHAN = sys.argv[2] if len(sys.argv) > 2 else "data/channel_pergame.csv"


def zfill(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def main():
    cap = pd.read_csv(ROOT / CAP)          # n_out_home/away = EXPECTED outs
    chan = pd.read_csv(ROOT / CHAN)        # per-channel margins + m_total
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    for d in (cap, chan, pit):
        d["game_id"] = zfill(d["game_id"])

    f = (chan.merge(cap[["game_id", "n_out_home", "n_out_away"]], on="game_id",
                    validate="one_to_one")
             .merge(pit[["game_id", "margin_actual", "open_margin"]],
                    on="game_id", validate="one_to_one"))
    assert len(f) > 0.9 * len(chan), f"join collapsed: {len(f)}/{len(chan)}"
    f = f.dropna(subset=["margin_actual", "m_total"]).copy()

    f["out_diff"] = f["n_out_home"] - f["n_out_away"]
    f["resid"] = f["margin_actual"] - f["m_total"]
    print(f"frame {len(f)} games, seasons {f.season.min()}..{f.season.max()}")
    print(f"expected outs: home {f.n_out_home.mean():.2f} away "
          f"{f.n_out_away.mean():.2f}   out_diff sd {f.out_diff.std():.2f}")

    # ---- pooled association -------------------------------------------
    x = f["out_diff"].to_numpy(float)
    y = f["resid"].to_numpy(float)
    b, a = np.polyfit(x, y, 1)
    n = len(x)
    yhat = a + b * x
    se = np.sqrt(((y - yhat) ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    t = b / se
    print(f"\n=== POOLED ===")
    print(f"  residual = {a:+.4f} {b:+.4f} * out_diff   (se {se:.4f}, t {t:+.2f})")
    print(f"  -> each extra EXPECTED absence on the home side costs the model "
          f"{b:+.3f} pts of margin it should not have charged"
          if b > 0 else
          f"  -> sign is NEGATIVE: the model UNDER-penalises absences")

    print("\n=== BY OUT-DIFFERENTIAL BUCKET ===")
    q = pd.cut(f["out_diff"], [-99, -2, -1, -0.5, 0.5, 1, 2, 99])
    print(f"{'bucket':>14} {'n':>6} {'mean resid':>11}")
    for k, sub in f.groupby(q, observed=True):
        print(f"{str(k):>14} {len(sub):6d} {sub['resid'].mean():11.3f}")

    # ---- season-clustered, the shipping statistic ----------------------
    per = []
    for s, sub in f.groupby("season"):
        if len(sub) < 200:
            continue
        bb = np.polyfit(sub["out_diff"], sub["resid"], 1)[0]
        per.append(dict(season=s, n=len(sub), b=float(bb)))
    r = pd.DataFrame(per)
    bs = r["b"].to_numpy(float)
    k = len(bs)
    sec = bs.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    lo, hi = bs.mean() - tc * sec, bs.mean() + tc * sec
    print(f"\n=== SEASON-CLUSTERED (K={k}) ===")
    print(r.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"  mean slope {bs.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"t {bs.mean()/sec:+.2f}")
    print(f"  same sign in {int((bs > 0).sum() if bs.mean() > 0 else (bs < 0).sum())}/{k} seasons")
    sig = (lo > 0) or (hi < 0)
    print(f"  VERDICT: {'REAL — CI excludes zero' if sig else 'ns — CI includes zero'}")

    json.dump({"pooled_b": float(b), "pooled_t": float(t), "n": int(n),
               "per_season": per, "mean_b": float(bs.mean()),
               "ci": [float(lo), float(hi)], "significant": bool(sig)},
              open(ROOT / "data" / "d231b_absence.json", "w"), default=float)
    print("\nwrote data/d231b_absence.json")


if __name__ == "__main__":
    main()
