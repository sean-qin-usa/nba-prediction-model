#!/usr/bin/env python3
"""D170: put the three K19 arms side by side and answer the owner's question —
how much of 2024-25's apparent superiority was DATA COMPLETENESS?

  A  = blind, PRE-backfill everything          (= D161, k19_blindA.json)
  B1 = blind, + DARKO backfill                 (isolates DARKO in the FIT)
  B2 = blind, + DARKO + report backfill        (isolates the report feed's
                                                effect on the tank term)
  C  = T2,    + DARKO + report backfill        (isolates the AVAILABILITY TIER)

Each contrast changes exactly one thing. Ordering matters and is deliberate:
darko_history and injury_reports_pit are both READ INSIDE nbapred/ (composition
talent; tanking's shutdown component), so they move the FIT and must be loaded
before the blind reference is taken. `game_inactives` is read NOWHERE inside
nbapred/ — it enters only through the caller's out-set construction — so B2->C
cannot move the fit and is a pure tier contrast.

ADVANTAGE metric, stated up front so it is not chosen after seeing the answer:
    ADV(s) = mean(norm_gap of the other 18 seasons) - norm_gap(s)
i.e. how much better than the rest of the frame season s looks. Positive =
better than the field. The question "how much of 2024-25 was data" is then
    1 - ADV_C(2024-25) / ADV_A(2024-25).

Per-season uncertainty on the raw gap is the paired per-game SE (the model's
per-game log loss minus the market's, same games), which is the only interval
this design supports at the season level.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LN2 = 0.6931471805599453


def load(tag):
    j = json.load(open(ROOT / "data" / f"k19_{tag}.json"))
    return {s["season"]: s for s in j["seasons"]}


def paired_se(tag):
    """Per-season SE of (model log loss - market log loss), paired per game."""
    df = pd.read_csv(ROOT / "data" / f"k19_{tag}_pergame.csv")
    out = {}
    for s, g in df.groupby("season"):
        y = g.y.values
        lu = -(y * np.log(g.p_us.values) + (1 - y) * np.log(1 - g.p_us.values))
        lm = -(y * np.log(g.p_mkt.values) + (1 - y) * np.log(1 - g.p_mkt.values))
        d = lu - lm
        out[s] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d))
    return out


def main():
    A, B1, B2, C = (load("blindA"), load("blindB1"),
                    load("blindB2"), load("t2"))
    seA, seC = paired_se("blindA"), paired_se("t2")
    seasons = sorted(A)

    print("=" * 122)
    print("D170 — 19 SEASONS: NORMALIZED GAP vs MARKET, THREE ARMS "
          "(lower is better; % of the market's own headroom to ln2)")
    print("=" * 122)
    print(f"{'season':8s} {'tier(C)':13s} {'A':>7s} {'B1':>7s} {'B2':>7s} "
          f"{'C':>7s} | {'B1-A':>6s} {'B2-B1':>6s} {'C-B2':>6s} {'C-A':>7s} | "
          f"{'darkoA':>7s} {'darkoC':>7s} | {'out/tm':>6s} {'+-SE':>7s}")
    print("-" * 122)
    for s in seasons:
        a, b1, b2, c = A[s], B1[s], B2[s], C[s]
        m, se, n = seC[s]
        print(f"{s:8s} {c['tier_label']:13s} {a['norm_gap_pct']:+7.2f} "
              f"{b1['norm_gap_pct']:+7.2f} {b2['norm_gap_pct']:+7.2f} "
              f"{c['norm_gap_pct']:+7.2f} | "
              f"{b1['norm_gap_pct']-a['norm_gap_pct']:+6.2f} "
              f"{b2['norm_gap_pct']-b1['norm_gap_pct']:+6.2f} "
              f"{c['norm_gap_pct']-b2['norm_gap_pct']:+6.2f} "
              f"{c['norm_gap_pct']-a['norm_gap_pct']:+7.2f} | "
              f"{100*a['darko_frac_roster_nonzero']:6.1f}% "
              f"{100*c['darko_frac_roster_nonzero']:6.1f}% | "
              f"{c['mean_outs_per_team']:6.2f} {se:7.5f}")

    def pooled(D, tag):
        df = pd.read_csv(ROOT / "data" / f"k19_{tag}_pergame.csv")
        y = df.y.values
        lu = -(y * np.log(df.p_us.values) + (1 - y) * np.log(1 - df.p_us.values)).mean()
        lm = -(y * np.log(df.p_mkt.values) + (1 - y) * np.log(1 - df.p_mkt.values)).mean()
        return lu, lm, 100.0 * (lu - lm) / (LN2 - lm)

    print("-" * 122)
    for tag, nm in (("blindA", "A blind (=D161)"), ("blindB1", "B1 blind+DARKO"),
                    ("blindB2", "B2 blind+DARKO+rep"), ("t2", "C T2 (payoff)")):
        lu, lm, ng = pooled(None, tag)
        print(f"POOLED {nm:18s} ll_us={lu:.5f} ll_mkt={lm:.5f} "
              f"norm_gap={ng:+.2f}%")

    # ---- the owner's question ------------------------------------------------
    print("\n" + "=" * 122)
    print("WAS 2024-25 BEST BECAUSE IT WAS THE SEASON, OR BECAUSE IT WAS THE "
          "ONLY ERA WITH COMPLETE DATA?")
    print("=" * 122)

    def adv(D, s):
        others = [D[k]["norm_gap_pct"] for k in D if k != s]
        return float(np.mean(others)) - D[s]["norm_gap_pct"]

    def rank(D, s):
        order = sorted(D, key=lambda k: D[k]["norm_gap_pct"])
        return order.index(s) + 1

    def zsc(D, s):
        others = np.array([D[k]["norm_gap_pct"] for k in D if k != s])
        return (others.mean() - D[s]["norm_gap_pct"]) / others.std(ddof=1)

    for s in ("2024-25", "2016-17", "2015-16", "2025-26", "2023-24"):
        rows = []
        for nm, D in (("A", A), ("B2", B2), ("C", C)):
            rows.append(f"{nm}: gap={D[s]['norm_gap_pct']:+6.2f} "
                        f"rank={rank(D, s):2d}/19 adv={adv(D, s):+6.2f} "
                        f"z={zsc(D, s):+5.2f}")
        print(f"  {s}   " + "   |   ".join(rows))

    s = "2024-25"
    aA, aB1, aB2, aC = adv(A, s), adv(B1, s), adv(B2, s), adv(C, s)
    print(f"\n  ADV(2024-25) = mean(other 18) - 2024-25:")
    print(f"    arm A  (D161, blind, nothing backfilled) : {aA:+.2f} pp")
    print(f"    arm B1 (blind, + DARKO)                  : {aB1:+.2f} pp   "
          f"(DARKO removes {aA-aB1:+.2f} pp)")
    print(f"    arm B2 (blind, + DARKO + reports)        : {aB2:+.2f} pp   "
          f"(reports remove {aB1-aB2:+.2f} pp)")
    print(f"    arm C  (T2, everything)                  : {aC:+.2f} pp   "
          f"(tier removes {aB2-aC:+.2f} pp)")
    print(f"\n  SHARE OF 2024-25's APPARENT SUPERIORITY THAT WAS DATA "
          f"COMPLETENESS: {100*(aA-aC)/aA:.1f}%")
    print(f"    of which DARKO coverage      : {100*(aA-aB1)/aA:.1f}%")
    print(f"    of which report feed         : {100*(aB1-aB2)/aA:.1f}%")
    print(f"    of which availability tier   : {100*(aB2-aC)/aA:.1f}%")
    print(f"  RESIDUAL, genuinely the season : {100*aC/aA:.1f}% "
          f"({aC:+.2f} pp of the original {aA:+.2f} pp)")

    # correlation of the improvement with how much data each season gained
    print("\n" + "=" * 122)
    print("DOES THE IMPROVEMENT TRACK HOW MUCH DATA THE SEASON GAINED? "
          "(the D153 confound, re-measured)")
    print("=" * 122)
    dgap_BA = np.array([B1[s]["norm_gap_pct"] - A[s]["norm_gap_pct"] for s in seasons])
    ddark = np.array([100 * (B1[s]["darko_frac_roster_nonzero"]
                             - A[s]["darko_frac_roster_nonzero"]) for s in seasons])
    dgap_CB = np.array([C[s]["norm_gap_pct"] - B2[s]["norm_gap_pct"] for s in seasons])
    outs = np.array([C[s]["mean_outs_per_team"] for s in seasons])
    print(f"  corr(DARKO coverage GAINED, gap change B-A)      = "
          f"{np.corrcoef(ddark, dgap_BA)[0,1]:+.3f}")
    print(f"  corr(mean OUTs per team-game at T2, gap change C-B2) = "
          f"{np.corrcoef(outs, dgap_CB)[0,1]:+.3f}")
    print(f"  seasons improved by DARKO (B1-A)      : "
          f"{int((dgap_BA<0).sum())}/19")
    print(f"  seasons improved by the T2 tier (C-B2): "
          f"{int((dgap_CB<0).sum())}/19")


if __name__ == "__main__":
    main()
