#!/usr/bin/env python3
"""BET-DIAGNOSE — win/loss intuition diagnosis vs the market (Deliverable 2).

QUESTIONS
  WHERE WE WIN: profile the sharper-and-hit games (same-side, our prob more
  extreme on the shared side, pick lands; n=663 over 3 seasons) plus the
  2025-26 mid-season rolling-beat window (D66: -0.0086, unexplained) against
  baselines. What over-represents: star-outs, post-event windows (trades /
  star returns), b2b, dead teams, phase, home/away, favorite strength?
  Two contrasts, because they answer different questions:
    (B1) SHARPER (n=981) vs ALL games   — where does the model GO sharper;
    (B2) sharper-HIT (663) vs sharper-MISS (318) — where is sharper RIGHT.

  WHERE WE LOSE: beyond the known late-season collapse cluster (D65),
  profile the flatter-and-favorite-lands bleed (same-side, market more
  extreme, favorite lands; n=1675): is our margin-magnitude shortfall
  uniform or concentrated (favorite strength, phase, specific teams)?

DATA: data/capstone_pergame_carry2.csv + data/nba.duckdb (read_only).
d = logloss(p_us) - logloss(p_mkt) per game (positive = we lose nats).

DEFINITIONS (all imported/reused from existing scripts — no redefinition):
  star-out: inactive with trailing avg minutes >= 28 over last 10 games with
    12+ min, strictly before the game (ba_intersection.star_out_map,
    game_inactives join).
  post-event window: team within 15 games after a detected trade arrival
    (>=25 trailing min/g, first game for a new team) or star return (>=30
    min/g after >=15 days out) — ba_portfolio.detect_events (verbatim
    exp_eventrecency construction), WINDOW_GAMES=15.
  dead team: entering wpct < .35 with gp >= 60 (ba_intersection convention).
  phase: early = W6 (either team gp < 20, nbapred.market.windows), late =
    month in {3,4}, mid = the rest.
  favorite orientation: market favorite side (p_mkt >= 0.5).
  2025-26 mid-season beat window: the contiguous 300-game (date-ordered)
  window of 2025-26 with the most negative rolling mean d; 200/400-game
  windows reported as robustness (transparent, no tuning on features).

STATS: iid game-level bootstrap 2000x seed 7 (game-level iid validated by
the methodology audit; ba_intersection.boot_diff).

COVERAGE CAVEAT (checked in-run): game_inactives covers 2023-24 fully,
2024-25 for 1195/1230 games, and 2025-26 for only 20/1230 games. All
star-out-flag rows are therefore computed on the covered seasons only
(star_cov mask = 2023-24 + 2024-25) and reported n/a where a contrast group
has no coverage (e.g. the 2025-26 window profile). n_out_home/away come
from the capstone CSV (oracle OUT sets) and are unaffected.

RULES HONORED: DuckDB read_only=True; new file scripts/bet_diagnose.py only;
nothing in nbapred/ or existing scripts edited (helpers IMPORTED).

Run:  python scripts/bet_diagnose.py
"""
from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ba_intersection import (team_schedule, star_out_map, logloss,  # noqa: E402
                             boot_diff, DEAD_WPCT, DEAD_GP)
from ba_portfolio import player_logs, detect_events, WINDOW_GAMES   # noqa: E402
from nbapred.market.windows import EARLY_SEASON_GP                  # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
CSV = os.path.join(ROOT, "data", "capstone_pergame_carry2.csv")
ROLL_W = 300
SEASONS = ("2023-24", "2024-25", "2025-26")


def postevent_flags(con, ts: pd.DataFrame) -> dict[tuple, bool]:
    """(season, team_abbrev, game_date) -> team within WINDOW_GAMES games
    after its most recent detected trade/star-return event."""
    id2ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games").fetchall())
    dates = {(s, t): g.game_date.dt.date.values
             for (s, t), g in ts.groupby(["season", "team"], sort=False)}
    flags = {}
    for season in SEASONS:
        ev_by_team, _ = detect_events(player_logs(con, season))
        for tid, evs in ev_by_team.items():
            ab = id2ab.get(tid)
            if ab is None or (season, ab) not in dates:
                continue
            ds = dates[(season, ab)]
            for gd in ds:
                prior = [e for e in evs if e < gd]
                if not prior:
                    continue
                e = prior[-1]
                n_since = int(((ds >= e) & (ds < gd)).sum())
                if n_since < WINDOW_GAMES:
                    flags[(season, ab, gd)] = True
    return flags


def build_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_date"] = pd.to_datetime(df.game_date)
    df["ll_us"] = logloss(df.p_us, df.y)
    df["ll_mkt"] = logloss(df.p_mkt, df.y)
    df["d"] = df.ll_us - df.ll_mkt

    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
        so = star_out_map(con)
        pe = postevent_flags(con, ts)
    finally:
        con.close()

    keys = ["season", "game_id"]
    tcols = ["gp_before", "wpct_before", "b2b"]
    for side in ("home", "away"):
        m = ts.rename(columns={"team": side})
        df = df.merge(m[keys + [side] + tcols], on=keys + [side], how="left")
        df = df.rename(columns={c: f"{side[0]}_{c}" for c in tcols})
        s = so.rename(columns={"team": side})
        df = df.merge(s[["game_id", side, "star_out"]], on=["game_id", side],
                      how="left")
        df = df.rename(columns={"star_out": f"{side[0]}_star"})
        df[f"{side[0]}_star"] = (df[f"{side[0]}_star"].astype("boolean")
                                 .fillna(False).astype(bool))
        df[f"{side[0]}_postev"] = [
            pe.get((se, t, gd.date()), False)
            for se, t, gd in zip(df.season, df[side], df.game_date)]

    # taxonomy
    df["pick_home"] = df.p_us > 0.5
    df["same_side"] = (df.p_us - 0.5) * (df.p_mkt - 0.5) > 0
    df["p_us_side"] = np.where(df.pick_home, df.p_us, 1 - df.p_us)
    df["p_mkt_side"] = np.where(df.pick_home, df.p_mkt, 1 - df.p_mkt)
    df["edge"] = df.p_us_side - df.p_mkt_side
    df["hit"] = np.where(df.pick_home, df.y == 1, df.y == 0)
    df["sharper"] = df.same_side & (df.edge > 0)
    df["flatter"] = df.same_side & (df.edge < 0)

    # favorite orientation (market)
    fav_home = df.p_mkt >= 0.5
    df["fav_home"] = fav_home
    df["p_mkt_fav"] = np.where(fav_home, df.p_mkt, 1 - df.p_mkt)
    df["p_us_fav"] = np.where(fav_home, df.p_us, 1 - df.p_us)
    df["fav_team"] = np.where(fav_home, df.home, df.away)
    df["dog_team"] = np.where(fav_home, df.away, df.home)
    df["fav_star"] = np.where(fav_home, df.h_star, df.a_star).astype(bool)
    df["dog_star"] = np.where(fav_home, df.a_star, df.h_star).astype(bool)
    df["dead_h"] =(df.h_wpct_before < DEAD_WPCT) & (df.h_gp_before >= DEAD_GP)
    df["dead_a"] = (df.a_wpct_before < DEAD_WPCT) & (df.a_gp_before >= DEAD_GP)
    df["dead_fav"] = np.where(fav_home, df.dead_h, df.dead_a).astype(bool)
    df["dead_dog"] = np.where(fav_home, df.dead_a, df.dead_h).astype(bool)

    # features
    df["star_any"] = df.h_star | df.a_star
    df["postev_any"] = df.h_postev | df.a_postev
    df["b2b_any"] = df.h_b2b.astype(bool) | df.a_b2b.astype(bool)
    df["dead_any"] = df.dead_h | df.dead_a
    df["n_out_tot"] = df.n_out_home + df.n_out_away
    df["outs_hi"] = df.n_out_tot >= 4
    df["early_w6"] = np.minimum(df.h_gp_before, df.a_gp_before) < EARLY_SEASON_GP
    df["late"] = df.game_date.dt.month.isin([3, 4])
    df["mid"] = ~df.early_w6 & ~df.late
    df["heavy_fav"] = df.p_mkt_fav > 0.85
    df["mod_fav"] = (df.p_mkt_fav > 0.65) & (df.p_mkt_fav <= 0.85)
    df["close_game"] = df.p_mkt_fav <= 0.65
    # game_inactives coverage: star flags valid on 2023-24 + 2024-25 only
    df["star_cov"] = df.season.isin(["2023-24", "2024-25"])
    return df


FEATURES = ["star_any", "fav_star", "dog_star", "outs_hi", "postev_any",
            "b2b_any", "dead_any", "dead_fav", "dead_dog",
            "early_w6", "mid", "late", "fav_home", "pick_home",
            "close_game", "mod_fav", "heavy_fav"]
STAR_FEATS = {"star_any", "fav_star", "dog_star"}


def over_rep(name, A: pd.DataFrame, B: pd.DataFrame, labels=("A", "B")):
    print(f"\n{name}   [{labels[0]} n={len(A)} vs {labels[1]} n={len(B)}]")
    print(f"{'feature':<12}{'rate_' + labels[0]:>9}{'rate_' + labels[1]:>9}"
          f"{'diff':>8}{'CI_lo':>8}{'CI_hi':>8}{'p':>7}")
    for f in FEATURES:
        Af, Bf = A, B
        tag = ""
        if f in STAR_FEATS:                 # inactives coverage guard
            Af, Bf = A[A.star_cov], B[B.star_cov]
            tag = " (cov)"
            if min(len(Af), len(Bf)) < 30:
                print(f"{f:<12}  n/a — no game_inactives coverage in a group")
                continue
        obs, lo, hi, p = boot_diff(Af[f].astype(float), Bf[f].astype(float))
        star = " *" if p < 0.05 else ""
        print(f"{f:<12}{Af[f].mean():>9.3f}{Bf[f].mean():>9.3f}"
              f"{obs:>+8.3f}{lo:>+8.3f}{hi:>+8.3f}{p:>7.3f}{star}{tag}")


def main():
    df = build_frame().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    SH = df[df.sharper]
    SH_HIT, SH_MISS = SH[SH.hit], SH[~SH.hit]
    FL = df[df.flatter]
    FL_LAND = FL[FL.hit]          # flatter-and-favorite-lands bleed
    print("=" * 96)
    print("BET-DIAGNOSE — win/loss intuition vs market  "
          f"[{len(df)} games, 3 seasons]")
    print("=" * 96)
    print("taxonomy: sharper n=%d (hit %.1f%% -> hit %d / miss %d) | "
          "flatter n=%d (fav lands n=%d) | opposite n=%d"
          % (len(SH), 100 * SH.hit.mean(), len(SH_HIT), len(SH_MISS),
             len(FL), len(FL_LAND), int((~df.same_side).sum())))
    print("mean d (ll_us - ll_mkt, +=we bleed): sharper-hit %+.4f | "
          "sharper-miss %+.4f | flatter-lands %+.4f | flatter-upset %+.4f | "
          "opposite %+.4f"
          % (SH_HIT.d.mean(), SH_MISS.d.mean(), FL_LAND.d.mean(),
             FL[~FL.hit].d.mean(), df.d[~df.same_side].mean()))

    # ================= WHERE WE WIN =================
    print("\n" + "=" * 96)
    print("A. WHERE WE WIN")
    over_rep("B1. WHERE WE GO SHARPER: sharper (981) vs ALL games",
             SH, df, ("shrp", "all"))
    over_rep("B2. WHERE SHARPER IS RIGHT: sharper-HIT (663) vs sharper-MISS "
             "(318)", SH_HIT, SH_MISS, ("hit", "miss"))
    print("  NOTE: the fav-strength rows in B2 are partly MECHANICAL — "
          "heavy favorites land more often, so\n  hits over-represent them "
          "by construction; read them with the d columns in A3.")

    # hit rate + payoff of sharper games conditional on features
    print("\nA3. sharper-game hit rate and d by feature presence "
          "(star_any on covered seasons only):")
    print(f"{'feature':<12}{'n_on':>6}{'hit_on':>8}{'hit_off':>8}"
          f"{'d_on':>9}{'d_off':>9}")
    for f in ["star_any", "outs_hi", "postev_any", "early_w6", "mid", "late",
              "dead_any", "b2b_any"]:
        base = SH[SH.star_cov] if f in STAR_FEATS else SH
        on, off = base[base[f].astype(bool)], base[~base[f].astype(bool)]
        if len(on) < 15:
            continue
        print(f"{f:<12}{len(on):>6}{100*on.hit.mean():>7.1f}%"
              f"{100*off.hit.mean():>7.1f}%{on.d.mean():>+9.4f}"
              f"{off.d.mean():>+9.4f}")

    # 2025-26 mid-season rolling beat window
    print("\nA4. 2025-26 MID-SEASON ROLLING-BEAT WINDOW")
    s26 = df[df.season == "2025-26"].reset_index(drop=True)
    for w in (200, ROLL_W, 400):
        r = s26.d.rolling(w).mean()
        i = int(r.idxmin())
        print(f"  w={w}: min rolling mean d = {r.min():+.4f}  "
              f"games [{i - w + 1}..{i}]  "
              f"{s26.game_date[i - w + 1].date()} .. {s26.game_date[i].date()}")
    r = s26.d.rolling(ROLL_W).mean()
    i = int(r.idxmin())
    win = s26.iloc[i - ROLL_W + 1:i + 1]
    rest = s26.drop(win.index)
    print(f"  -> profiling w={ROLL_W} window: mean d {win.d.mean():+.4f} "
          f"(rest of 25-26: {rest.d.mean():+.4f}); sharper share "
          f"{win.sharper.mean():.3f} vs {rest.sharper.mean():.3f}; "
          f"sharper hit {win[win.sharper].hit.mean()*100:.1f}% vs "
          f"{rest[rest.sharper].hit.mean()*100:.1f}%")
    over_rep("A4b. window games vs rest of 2025-26", win, rest,
             ("win", "rest"))

    # team over-representation in sharper-hit (either side)
    print("\nA5. teams over-represented in sharper-HIT (either side, "
          "rate ratio vs all games, n_app>=25):")
    app_hit = pd.concat([SH_HIT.home, SH_HIT.away]).value_counts()
    app_all = pd.concat([df.home, df.away]).value_counts()
    rr = (app_hit / (2 * len(SH_HIT))) / (app_all / (2 * len(df)))
    rr = rr[app_hit >= 25].sort_values(ascending=False)
    for t, v in rr.head(8).items():
        print(f"    {t}: x{v:.2f} (n={int(app_hit[t])})")

    # ================= WHERE WE LOSE =================
    print("\n" + "=" * 96)
    print("B. WHERE WE LOSE — flatter-and-favorite-lands bleed "
          f"(n={len(FL_LAND)}, total {FL_LAND.d.sum():.1f} nats = "
          f"{FL_LAND.d.sum()/len(df):.5f} pooled)")
    FL_LAND = FL_LAND.copy()
    FL_LAND["short"] = -FL_LAND.edge          # prob shortfall on the fav side

    def bucket_table(title, groups):
        print(f"\n{title}")
        print(f"{'bucket':<16}{'n':>6}{'%n':>7}{'short':>8}{'d/game':>9}"
              f"{'nats':>8}{'%nats':>7}")
        tot = FL_LAND.d.sum()
        for lbl, m in groups:
            g = FL_LAND[m]
            if len(g) == 0:
                continue
            print(f"{lbl:<16}{len(g):>6}{100*len(g)/len(FL_LAND):>6.1f}%"
                  f"{g['short'].mean():>8.3f}{g.d.mean():>+9.4f}"
                  f"{g.d.sum():>8.1f}{100*g.d.sum()/tot:>6.1f}%")

    pf = FL_LAND.p_mkt_fav
    bucket_table("B1. by market favorite strength (p_mkt_fav):", [
        ("[.50,.60)", pf < 0.60), ("[.60,.70)", (pf >= 0.60) & (pf < 0.70)),
        ("[.70,.80)", (pf >= 0.70) & (pf < 0.80)),
        ("[.80,.90)", (pf >= 0.80) & (pf < 0.90)), ("[.90,1)", pf >= 0.90)])
    bucket_table("B2. by phase:", [
        ("early_w6", FL_LAND.early_w6), ("mid", FL_LAND.mid),
        ("late(Mar-Apr)", FL_LAND.late)])
    bucket_table("B3. by context flags (star-free):", [
        ("dead_dog", FL_LAND.dead_dog), ("b2b_any", FL_LAND.b2b_any),
        ("outs_hi", FL_LAND.outs_hi), ("postev_any", FL_LAND.postev_any),
        ("none-of-above", ~(FL_LAND.dead_dog | FL_LAND.b2b_any |
                            FL_LAND.outs_hi | FL_LAND.postev_any))])
    FLc = FL_LAND[FL_LAND.star_cov]
    print(f"\nB3b. star flags on covered seasons only "
          f"(n={len(FLc)}, {FLc.d.sum():.1f} nats):")
    print(f"{'bucket':<16}{'n':>6}{'%n':>7}{'short':>8}{'d/game':>9}"
          f"{'nats':>8}{'%nats':>7}")
    for lbl, m in (("dog_star", FLc.dog_star), ("fav_star", FLc.fav_star),
                   ("no_star", ~FLc.dog_star & ~FLc.fav_star)):
        g = FLc[m]
        print(f"{lbl:<16}{len(g):>6}{100*len(g)/len(FLc):>6.1f}%"
              f"{g['short'].mean():>8.3f}{g.d.mean():>+9.4f}"
              f"{g.d.sum():>8.1f}{100*g.d.sum()/FLc.d.sum():>6.1f}%")

    print("\nB4. top favorite teams by total bleed nats (n>=20):")
    g = FL_LAND.groupby("fav_team").agg(
        n=("d", "size"), nats=("d", "sum"), short=("short", "mean"))
    g = g[g.n >= 20].sort_values("nats", ascending=False)
    tot = FL_LAND.d.sum()
    for t, r in g.head(8).iterrows():
        print(f"    {t}: n={int(r.n)}  nats={r.nats:.1f} "
              f"({100*r.nats/tot:.1f}%)  short={r.short:.3f}")

    # uniformity test: is the shortfall (prob units) flat across fav strength?
    print("\nB5. UNIFORM vs CONCENTRATED: shortfall by fav-strength quintile "
          "(equal-n bins):")
    FL_LAND["q"] = pd.qcut(FL_LAND.p_mkt_fav, 5, labels=False)
    for qb, g2 in FL_LAND.groupby("q"):
        print(f"    q{int(qb)+1} p_mkt_fav[{g2.p_mkt_fav.min():.3f},"
              f"{g2.p_mkt_fav.max():.3f}]: short={g2['short'].mean():.3f}  "
              f"d/game={g2.d.mean():+.4f}  nats={g2.d.sum():.1f}")
    corr = np.corrcoef(FL_LAND.p_mkt_fav, FL_LAND["short"])[0, 1]
    print(f"    corr(p_mkt_fav, shortfall) = {corr:+.3f}")


if __name__ == "__main__":
    main()
