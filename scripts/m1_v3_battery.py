"""M1 (3P-luck DEFENSE-ONLY) — full V3 battery, DIAGNOSTIC RE-SCORING ONLY.

Re-analyses the ALREADY-ON-DISK D138 artifact `data/nsport_joint_pergame.csv`
under GATE_POLICY_V2 (=V3) SS8-SS11.  No model is re-run; the per-game deltas are
bit-identical to what D138 scored.  This is the same operation D139 SS12 applied
to the JOINT arm (`scripts/cv_rescore.py`), extended to the M1 MARGINAL, which
is the arm D138 SS12 recommended gating solo.

READ-ONLY.  Writes only data/m1_v3_battery.json + data/logs/m1_v3_battery.log.

IT IS NOT A NEW TEST.  It adds zero new data and cannot resolve selection; the
6,148 games are the SAME ones D138 spent.  What it does is run the mechanical
checks D138 left unrun (SS12 (a) and (b)) plus the V3 split/clustering battery
that did not exist when D138 ran.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.eval import splits as S

ART = ROOT / "data" / "nsport_joint_pergame.csv"
CAP = ROOT / "data" / "capstone_pergame_d132.csv"
OUT = ROOT / "data" / "m1_v3_battery.json"


def main():
    df = pd.read_csv(ART, dtype={"game_id": str})
    print(f"artifact {ART.name}: n={len(df)} seasons={sorted(df.season.unique())}",
          flush=True)

    # ---------- D134 CONTROL-HASH FIELD -----------------------------------
    cap = pd.read_csv(CAP, dtype={"game_id": str})
    j = cap.merge(df, on=["season", "game_id"], suffixes=("_cap", ""))
    dp = np.abs(j["p_us"].values - j["p_ctl"].values)
    ctrl = dict(matched=int(len(j)), of_capstone=int(len(cap)),
                max_abs_dp=float(dp.max()),
                frac_games_moved=float((dp > 1e-9).mean()),
                per_season={s: float(np.abs(g["p_us"] - g["p_ctl"]).max())
                            for s, g in j.groupby("season")})
    print(f"CONTROL-HASH: {ctrl['matched']}/{ctrl['of_capstone']} matched, "
          f"max|dp|={ctrl['max_abs_dp']:.3e}, moved={ctrl['frac_games_moved']:.4f}",
          flush=True)

    # ---------- the panel --------------------------------------------------
    panel = S.Panel.from_logloss(
        season=df["season"].values, y=df["y"].values,
        p_ctrl=df["p_ctl"].values, p_treat=df["p_defonly"].values,
        date=df["game_date"].values,
        label="M1 3P-luck defense-only (defonly) vs D132 control")
    rep = S.full_report(panel)
    print(S.format_report(rep), flush=True)

    # ---------- holdout / dev arms ----------------------------------------
    sub = {}
    for name, seas in (("holdout_21_23", ("2021-22", "2022-23")),
                       ("dev_23_26", ("2023-24", "2024-25", "2025-26"))):
        p = panel.by_seasons(seas)
        r = S.paired_bootstrap(p.d)
        r["mde80"] = S.mde80(p.d)
        r["season_mean_t"] = S.cluster_mean_t_interval(p.d, p.season)
        sub[name] = r
        print(f"{name:14s} n={r['n']} {r['est']:+.5f} "
              f"CI({r['lo']:+.5f},{r['hi']:+.5f}) {'SIG' if r['sig'] else 'ns'}",
              flush=True)

    # ---------- M2 decay, for the register (report-only) -------------------
    p2 = S.Panel.from_logloss(
        season=df["season"].values, y=df["y"].values,
        p_ctrl=df["p_ctl"].values, p_treat=df["p_blend6040"].values,
        date=df["game_date"].values, label="M2 comp-heavy 60/40")
    m2 = dict(pooled=S.paired_bootstrap(p2.d),
              per_season=S.per_season(p2),
              rolling_origin=S.rolling_origin(p2),
              clustering=S.clustering_report(p2))
    m2["verdict"] = S.adjudicate(S.full_report(p2))
    ms = [r["est"] for r in m2["per_season"]]
    ss = [r["season"] for r in m2["per_season"]]
    # OLS trend per season
    x = np.arange(len(ms), dtype=float)
    slope = float(np.polyfit(x, ms, 1)[0])
    m2["trend_per_season"] = slope
    print("M2 per-season " + " ".join(f"{s}:{v:+.5f}" for s, v in zip(ss, ms))
          + f"  trend {slope:+.5f}/season", flush=True)

    json.dump(dict(control_hash=ctrl, m1=rep, m1_subsets=sub, m2=m2),
              open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
