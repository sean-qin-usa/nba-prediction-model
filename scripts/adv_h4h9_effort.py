"""H4+H9 descriptive test (docs/ADVERSE_HYPOTHESES.md): effort mirror.
H4 CLINCHED-TEAM REST: teams that have locked a top-6 seed (lead over the
current 7th-place team > that team's games remaining) manage load late season
-> should UNDERPERFORM our predicted margin.
H9 URGENCY: must-win teams within 2 GB of the play-in cut (10th place) late
season -> should OVERPERFORM our predicted margin.

Margin residual = actual margin (team perspective) - predicted margin, with
m_us = 7.2 * logit(p_us) (production link) and m_mkt = spread margin.
Sizes with t-based and bootstrap CIs. Read-only DuckDB; printed report only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
RNG = np.random.default_rng(41)

EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
        "NYK", "ORL", "PHI", "TOR", "WAS"}
WEST = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC",
        "PHX", "POR", "SAC", "SAS", "UTA"}


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def boot_mean_ci(x, B=4000):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    return tuple(np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def main():
    con = duckdb.connect(DB, read_only=True)
    tg = con.execute("""SELECT season, game_id, game_date, team_abbrev, wl
        FROM nba_games WHERE game_id LIKE '002%'
          AND season IN ('2023-24','2024-25','2025-26')""").df()
    om = con.execute("""SELECT game_date, home, away, score_home, score_away,
        home_exp_margin FROM odds_market WHERE season_end >= 2024""").df()
    con.close()
    tg["game_date"] = pd.to_datetime(tg.game_date)
    om["game_date"] = pd.to_datetime(om.game_date).dt.date.astype(str)

    # per (season, team): sorted arrays of game dates and cumulative wins
    hist = {}
    for (season, team), gdf in tg.groupby(["season", "team_abbrev"]):
        gdf = gdf.sort_values("game_date")
        dates = gdf.game_date.values
        wins = np.cumsum((gdf.wl == "W").values)
        hist[(season, team)] = (dates, wins)

    def record_before(season, team, D):
        dates, wins = hist[(season, team)]
        i = int(np.searchsorted(dates, D))  # games strictly before D
        w = int(wins[i - 1]) if i > 0 else 0
        return w, i - w, i  # wins, losses, gp

    def conf_table(season, conf, D):
        rows = []
        for t in conf:
            w, l, gp = record_before(season, t, D)
            rows.append((t, w, l, gp, 82 - gp))
        rows.sort(key=lambda r: (-r[1], r[2]))  # by wins desc, losses asc
        return rows

    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df = df.merge(om, on=["game_date", "home", "away"], how="left")
    assert df.home_exp_margin.notna().all()
    df["margin_home"] = df.score_home - df.score_away
    df["m_us"] = 7.2 * logit(df.p_us)
    df["d"] = ll(df.p_us.values, df.y.values) - ll(df.p_mkt.values, df.y.values)

    flags = {"clinched": [], "mustwin": []}
    cache = {}
    for r in df.itertuples():
        season = r.season
        D = np.datetime64(r.game_date)
        for team, opp_, sgn in ((r.home, r.away, 1.0), (r.away, r.home, -1.0)):
            conf = EAST if team in EAST else WEST
            ck = (season, D, team in EAST)
            if ck not in cache:
                cache[ck] = conf_table(season, conf, D)
            tab = cache[ck]
            w, l, gp = record_before(season, team, D)
            rem = 82 - gp
            t7 = tab[6]      # 7th place (0-indexed 6)
            t10 = tab[9]     # 10th place
            clin = (w - (t7[1] + t7[4]) > 0) and team != t7[0]
            gb10 = ((w - t10[1]) - (l - t10[2])) / 2.0
            mw = (gp >= 65) and (abs(gb10) <= 2) and not clin and team != t10[0]
            # 10th-place team itself is in the race too
            if (gp >= 65) and team == t10[0] and not clin:
                mw = True
            row = [r.game_id, season, str(r.game_date), team, sgn, gp,
                   sgn * r.margin_home, sgn * r.m_us, sgn * r.home_exp_margin,
                   r.d, r.y if sgn > 0 else 1 - r.y,
                   r.p_us if sgn > 0 else 1 - r.p_us,
                   r.p_mkt if sgn > 0 else 1 - r.p_mkt]
            if clin:
                flags["clinched"].append(row)
            if mw:
                flags["mustwin"].append(row)

    cols = ["game_id", "season", "game_date", "team", "sgn", "gp", "margin",
            "m_us", "m_mkt", "d", "won", "p_us", "p_mkt"]
    for name, rows in flags.items():
        f = pd.DataFrame(rows, columns=cols)
        # drop games where both sides carry the same flag (residuals cancel)
        dup = f.game_id.duplicated(keep=False)
        f = f[~dup]
        f["resid_us"] = f.margin - f.m_us
        f["resid_mkt"] = f.margin - f.m_mkt
        print("=" * 72)
        print(f"{name.upper()}: n={len(f)} team-games "
              f"({int(dup.sum())} both-flagged rows dropped), "
              f"seasons {f.season.value_counts().sort_index().to_dict()}")
        for c in ("resid_us", "resid_mkt"):
            x = f[c].values
            se = x.std(ddof=1) / np.sqrt(len(x))
            lo, hi = boot_mean_ci(x)
            print(f"  {c:9s} mean {x.mean():+6.2f}  t-CI95 "
                  f"({x.mean() - 1.96 * se:+.2f},{x.mean() + 1.96 * se:+.2f})  "
                  f"boot ({lo:+.2f},{hi:+.2f})  sd {x.std(ddof=1):.1f}")
        print(f"  win rate {f.won.mean():.3f} vs p_us {f.p_us.mean():.3f} "
              f"vs p_mkt {f.p_mkt.mean():.3f}")
        lo, hi = boot_mean_ci(f.d.values)
        print(f"  d = L_us - L_mkt in these games: {f.d.mean():+.4f} "
              f"CI95({lo:+.4f},{hi:+.4f})   [all-games avg +0.0121]")
        # market minus us on the flagged team's margin: does market already
        # price the effect that we miss?
        dm = (f.m_mkt - f.m_us).values
        lo, hi = boot_mean_ci(dm)
        print(f"  m_mkt - m_us (does market shade these teams differently): "
              f"{dm.mean():+.2f} CI95({lo:+.2f},{hi:+.2f})")

    # baseline sanity: all late-season team-games residual ~ 0?
    late = df[pd.to_datetime(df.game_date).dt.month.isin([3, 4])]
    resid = (late.margin_home - late.m_us).values
    lo, hi = boot_mean_ci(resid)
    print("=" * 72)
    print(f"BASELINE Mar-Apr home-side residual vs us: n={len(late)} "
          f"mean {resid.mean():+.2f} CI95({lo:+.2f},{hi:+.2f})")


if __name__ == "__main__":
    main()
