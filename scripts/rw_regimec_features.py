"""Regime C (mid-season core, min team game-number 20-54) residual-mining feature layer.

Builds a per-game PIT feature matrix for the capstone universe (2023-24..2025-26)
covering 30+ market-free features: rest/travel/schedule, form, style/matchup clash,
h2h history, lineup continuity, availability quality (PIT injury feed + as-of DARKO),
ref crews (mine-only, no 25-26 coverage).

Rules honored: DuckDB read_only=True; PIT strict (all trailing stats exclude the
current game; as-of joins use date < game_date); market data only enters as
benchmark columns (m_mkt/p_mkt) for diagnosis, never as a feature.

Output: parquet in the session scratchpad (no new data files in the repo).
"""
import os
import sys
import unicodedata
import numpy as np
import pandas as pd
import duckdb

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CAP = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
OUTDIR = os.environ.get(
    "RW_OUT",
    "data/scratch",
)
SCALE = 7.2

ARENA = {  # lat, lon, UTC offset (standard)
    "ATL": (33.757, -84.396, -5), "BOS": (42.366, -71.062, -5), "BKN": (40.683, -73.975, -5),
    "CHA": (35.225, -80.839, -5), "CHI": (41.881, -87.674, -6), "CLE": (41.496, -81.688, -5),
    "DAL": (32.790, -96.810, -6), "DEN": (39.749, -105.008, -7), "DET": (42.341, -83.055, -5),
    "GSW": (37.768, -122.388, -8), "HOU": (29.751, -95.362, -6), "IND": (39.764, -86.156, -5),
    "LAC": (33.945, -118.267, -8), "LAL": (34.043, -118.267, -8), "MEM": (35.138, -90.051, -6),
    "MIA": (25.781, -80.187, -5), "MIL": (43.045, -87.917, -6), "MIN": (44.979, -93.276, -6),
    "NOP": (29.949, -90.082, -6), "NYK": (40.751, -73.994, -5), "OKC": (35.463, -97.515, -6),
    "ORL": (28.539, -81.384, -5), "PHI": (39.901, -75.172, -5), "PHX": (33.446, -112.071, -7),
    "POR": (45.532, -122.667, -8), "SAC": (38.580, -121.500, -8), "SAS": (29.427, -98.437, -6),
    "TOR": (43.643, -79.379, -5), "UTA": (40.768, -111.901, -7), "WAS": (38.898, -77.021, -5),
}
TEAM_FULL = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.split())


def team_game_log(con):
    g = con.execute("""
        with rs as (select season, game_id, game_date, team_abbrev as team, is_home, pts
                    from nba_games where game_id like '002%')
        select a.season, a.game_id, a.game_date, a.team, a.is_home, a.pts,
               b.pts as opp_pts, b.team as opp
        from rs a join rs b on a.game_id=b.game_id and a.team<>b.team
    """).fetchdf()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["season", "team", "game_date", "game_id"]).reset_index(drop=True)
    g["margin"] = g.pts - g.opp_pts
    g["win"] = (g.margin > 0).astype(float)
    grp = g.groupby(["season", "team"], sort=False)
    g["game_no"] = grp.cumcount() + 1
    g["prev_date"] = grp["game_date"].shift(1)
    g["days_rest"] = (g.game_date - g.prev_date).dt.days.clip(upper=10).fillna(10)
    g["b2b"] = (g.days_rest == 1).astype(float)
    # 3-in-4 / games in last 7 days (excluding today), via per-team date lists
    dates_map = {k: v["game_date"].values for k, v in g.groupby(["season", "team"])}
    n3in4, gl7 = np.zeros(len(g)), np.zeros(len(g))
    for (season, team), idx in g.groupby(["season", "team"]).groups.items():
        d = g.loc[idx, "game_date"].values
        for j, i in enumerate(idx):
            prior = d[:j]
            n3in4[g.index.get_loc(i)] = ((prior >= d[j] - np.timedelta64(3, "D")).sum() >= 2)
            gl7[g.index.get_loc(i)] = (prior >= d[j] - np.timedelta64(7, "D")).sum()
    g["is_3in4"], g["games_last7"] = n3in4, gl7
    # location / travel / tz
    g["loc"] = np.where(g.is_home, g.team, g.opp)
    g["lat"] = g["loc"].map(lambda t: ARENA[t][0])
    g["lon"] = g["loc"].map(lambda t: ARENA[t][1])
    g["tzo"] = g["loc"].map(lambda t: ARENA[t][2])
    g["home_tzo"] = g["team"].map(lambda t: ARENA[t][2])
    g["prev_lat"] = grp["lat"].shift(1)
    g["prev_lon"] = grp["lon"].shift(1)
    g["hop_km"] = haversine(g.prev_lat, g.prev_lon, g.lat, g.lon)
    g["hop_km"] = g.hop_km.fillna(0.0)
    g["tz_from_home"] = g.tzo - g.home_tzo
    g["prev_tzo"] = grp["tzo"].shift(1)
    g["tz_change"] = (g.tzo - g.prev_tzo).fillna(0.0)
    # travel_km last 3 days: sum of hops for games within 3 days + current hop
    trav3 = np.zeros(len(g))
    for (season, team), idx in g.groupby(["season", "team"]).groups.items():
        d = g.loc[idx, "game_date"].values
        h = g.loc[idx, "hop_km"].values
        for j in range(len(idx)):
            m = (d >= d[j] - np.timedelta64(3, "D")) & (d <= d[j])
            trav3[g.index.get_loc(idx[j])] = h[m].sum()
    g["travel3d_km"] = trav3
    # road trip length (consecutive away incl current) / home stand
    def _runlen(s):
        out = np.zeros(len(s), dtype=float)
        run = 0
        for j, v in enumerate(s):
            run = run + 1 if v else 0
            out[j] = run
        return out
    g["road_trip"] = grp["is_home"].transform(lambda s: _runlen(~s.values))
    g["home_stand"] = grp["is_home"].transform(lambda s: _runlen(s.values))
    # form (all shifted → entering the game)
    g["margin_prev"] = grp["margin"].shift(1)
    for w, name in [(10, "form10"), (5, "form5")]:
        g[name] = grp["margin"].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=3).mean())
    g["season_avg_margin"] = grp["margin"].transform(lambda s: s.shift(1).expanding(3).mean())
    g["momentum"] = g.form5 - g.season_avg_margin
    g["wpct"] = grp["win"].transform(lambda s: s.shift(1).expanding(1).mean())
    # streak entering
    def _streak(wins):
        out = np.zeros(len(wins))
        cur = 0
        for j in range(len(wins)):
            out[j] = cur
            if j < len(wins):
                if wins[j] == 1:
                    cur = cur + 1 if cur > 0 else 1
                else:
                    cur = cur - 1 if cur < 0 else -1
        return out
    g["streak"] = grp["win"].transform(lambda s: _streak(s.values))
    g["blowout15_15"] = g.groupby(["season", "team"], sort=False)["margin"].transform(
        lambda s: (s.abs() >= 15).shift(1).rolling(15, min_periods=5).mean())
    # opponent entering wpct on each game row → SOS of last 10
    opp_wp = g[["season", "game_id", "team", "wpct"]].rename(
        columns={"team": "opp", "wpct": "opp_wpct_entering"})
    g = g.merge(opp_wp, on=["season", "game_id", "opp"], how="left")
    g = g.sort_values(["season", "team", "game_date", "game_id"]).reset_index(drop=True)
    g["sos10"] = g.groupby(["season", "team"], sort=False)["opp_wpct_entering"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    return g


def box_style(con, g):
    b = con.execute("""
        select s.game_id, n.team_abbrev as team, sum(s.fga) fga, sum(s.fgm) fgm,
               sum(s.fg3a) fg3a, sum(s.fg3m) fg3m, sum(s.fta) fta, sum(s.ftm) ftm,
               sum(s.oreb) oreb, sum(s.dreb) dreb, sum(s.tov) tov, sum(s.ast) ast,
               sum(s.pf) pf, sum(s.shooting_fouls) sfl,
               sum(s.rima) rima, sum(s.thra) thra
        from player_game_stats s
        join (select distinct game_id, team_id, team_abbrev from nba_games
              where game_id like '002%') n
          on s.game_id=n.game_id and s.team_id=n.team_id
        group by 1,2
    """).fetchdf()
    other = b.rename(columns={c: "o_" + c for c in b.columns if c not in ("game_id", "team")})
    other = other.rename(columns={"team": "opp"})
    b = b.merge(other, on="game_id")
    b = b[b.team != b.opp]
    b["poss"] = b.fga - b.oreb + b.tov + 0.44 * b.fta
    b["ftr"] = b.fta / b.fga.clip(lower=1)
    b["p3rate"] = b.fg3a / b.fga.clip(lower=1)
    b["p3pct"] = b.fg3m / b.fg3a.clip(lower=1)
    b["orbp"] = b.oreb / (b.oreb + b.o_dreb).clip(lower=1)
    b["drbp"] = b.dreb / (b.dreb + b.o_oreb).clip(lower=1)
    b["astr"] = b.ast / b.fgm.clip(lower=1)
    b["tovr"] = b.tov / b.poss.clip(lower=1)
    b["rimrate"] = b.rima / b.fga.clip(lower=1)
    b["opp_ftr_alwd"] = b.o_fta / b.o_fga.clip(lower=1)   # defensive foul proneness
    b["opp_rim_alwd"] = b.o_rima / b.o_fga.clip(lower=1)  # rim protection (share allowed)
    b["opp_p3_alwd"] = b.o_fg3a / b.o_fga.clip(lower=1)
    cols = ["poss", "ftr", "p3rate", "orbp", "drbp", "astr", "tovr", "rimrate",
            "opp_ftr_alwd", "opp_rim_alwd", "opp_p3_alwd", "sfl"]
    m = g[["season", "game_id", "team", "game_date"]].merge(b[["game_id", "team", "p3pct"] + cols],
                                                            on=["game_id", "team"], how="left")
    m = m.sort_values(["season", "team", "game_date", "game_id"]).reset_index(drop=True)
    grp = m.groupby(["season", "team"], sort=False)
    out = m[["season", "game_id", "team"]].copy()
    for c in cols:
        out["td_" + c] = grp[c].transform(lambda s: s.shift(1).expanding(5).mean())
    out["td_p3pct_std"] = grp["p3pct"].transform(lambda s: s.shift(1).expanding(10).std())
    return out


def h2h(g):
    home = g[g.is_home][["season", "game_id", "game_date", "team", "opp", "margin"]].copy()
    home = home.rename(columns={"team": "home", "opp": "away"})
    home["pair"] = [frozenset((h, a)) for h, a in zip(home.home, home.away)]
    home = home.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    rows = []
    for (season, pair), sub in home.groupby(["season", "pair"]):
        sub = sub.sort_values(["game_date", "game_id"])
        prior_m, prior_d = [], []
        for _, r in sub.iterrows():
            if prior_m:
                # margins from current home team's perspective
                ms = [pm if ph == r.home else -pm for pm, ph in prior_m and prior_m]
                ms = [(pm if ph == r.home else -pm) for (pm, ph) in prior_m]
                rows.append((r.game_id, len(ms), float(np.mean(ms)),
                             (r.game_date - prior_d[-1]).days))
            else:
                rows.append((r.game_id, 0, np.nan, np.nan))
            prior_m.append((r.margin, r.home))
            prior_d.append(r.game_date)
    return pd.DataFrame(rows, columns=["game_id", "h2h_n", "h2h_mean_margin", "h2h_days_since"])


def officials(con, g):
    off = con.execute("select game_id, official_id from game_officials").fetchdf()
    homes = g[g.is_home][["game_id", "game_date", "margin"]].copy()
    homes["home_win"] = (homes.margin > 0).astype(float)
    off = off.merge(homes, on="game_id", how="inner").sort_values(["official_id", "game_date"])
    grp = off.groupby("official_id", sort=False)
    off["n_prior"] = grp.cumcount()
    off["cum_hw"] = grp["home_win"].transform(lambda s: s.shift(1).expanding(1).sum())
    prior = 0.54
    off["ref_hw_shrunk"] = (off.cum_hw.fillna(0) + 40 * prior) / (off.n_prior + 40)
    per_game = off.groupby("game_id")["ref_hw_shrunk"].mean().rename("ref_home_bias")
    return per_game.reset_index()


def outs_quality(con, g):
    rep = con.execute("""
        select report_date, game_date, matchup, team, player, status
        from injury_reports_pit where status='Out'
    """).fetchdf()
    rep["abbrev"] = rep.team.map(TEAM_FULL)
    rep = rep.dropna(subset=["abbrev"])
    rep["game_date"] = pd.to_datetime(rep.game_date)
    rep["report_date"] = pd.to_datetime(rep.report_date)
    rep = rep[rep.report_date <= rep.game_date]
    # keep latest report_date per (game_date, abbrev)
    rep = rep.sort_values("report_date").groupby(
        ["game_date", "abbrev", "player"], as_index=False).tail(1)
    players = con.execute("select player_id, full_name from nba_players").fetchdf()
    players["key"] = players.full_name.map(norm_name)
    pmap = dict(zip(players.key, players.player_id))
    def flip(nm):
        parts = str(nm).split(", ")
        return norm_name(parts[1] + " " + parts[0]) if len(parts) == 2 else norm_name(nm)
    rep["key"] = rep.player.map(flip)
    rep["player_id"] = rep.key.map(pmap)
    matched = rep.dropna(subset=["player_id"]).copy()
    print(f"outs: {len(rep)} out-rows, matched {len(matched)} ({len(matched)/max(len(rep),1):.1%})",
          file=sys.stderr)
    dk = con.execute("select player_id, date, dpm from darko_history").fetchdf()
    dk["date"] = pd.to_datetime(dk.date)
    dk = dk.sort_values("date")
    matched = matched.sort_values("game_date")
    matched["player_id"] = matched.player_id.astype("int64")
    asof = pd.merge_asof(matched, dk, by="player_id", left_on="game_date", right_on="date",
                         allow_exact_matches=False)  # dpm strictly before game date
    agg = asof.groupby(["game_date", "abbrev"]).agg(
        out_dpm_pos=("dpm", lambda s: np.nansum(np.clip(s, 0, None))),
        out_dpm_max=("dpm", "max"),
        n_out_feed=("dpm", "size")).reset_index()
    agg["star_out"] = (agg.out_dpm_max >= 2.0).astype(float)
    return agg


def continuity(con, g):
    st = con.execute("""
        select s.game_id, s.stint_idx, s.seconds, s.home_lineup, s.away_lineup
        from lineup_stints s where s.game_id like '002%'
    """).fetchdf()
    meta = g[g.is_home][["season", "game_id", "game_date", "team", "opp"]].rename(
        columns={"team": "home", "opp": "away"})
    st = st.merge(meta, on="game_id", how="inner")
    rows = []
    for side, lcol, tcol in (("H", "home_lineup", "home"), ("A", "away_lineup", "away")):
        sub = st[["season", "game_id", "game_date", tcol, lcol, "seconds", "stint_idx"]].copy()
        sub.columns = ["season", "game_id", "game_date", "team", "lineup", "seconds", "stint_idx"]
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    # per team-game: top unit seconds, total seconds, starting lineup
    per = long.groupby(["season", "team", "game_date", "game_id"]).apply(
        lambda d: pd.Series({
            "top_unit": d.groupby("lineup")["seconds"].sum().idxmax(),
            "top_unit_sec": d.groupby("lineup")["seconds"].sum().max(),
            "tot_sec": d.seconds.sum(),
            "starters": d.loc[d.stint_idx.idxmin(), "lineup"],
        }), include_groups=False).reset_index()
    per = per.sort_values(["season", "team", "game_date", "game_id"]).reset_index(drop=True)
    out = []
    for (season, team), sub in per.groupby(["season", "team"]):
        sub = sub.reset_index(drop=True)
        for j in range(len(sub)):
            prior = sub.iloc[max(0, j - 10):j]
            if len(prior) >= 5:
                # most common unit over prior 10 covered games, its minute share
                us = {}
                for _, r in prior.iterrows():
                    us[r.top_unit] = us.get(r.top_unit, 0) + r.top_unit_sec
                cont = max(us.values()) / prior.tot_sec.sum()
                churn = prior.starters.nunique()
            else:
                cont, churn = np.nan, np.nan
            out.append((season, team, sub.loc[j, "game_id"], cont, churn))
    return pd.DataFrame(out, columns=["season", "team", "game_id",
                                      "lineup_cont", "starter_churn"])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    con = duckdb.connect(DB, read_only=True)
    cap = pd.read_csv(CAP, dtype={"game_id": str})
    cap["game_date"] = pd.to_datetime(cap.game_date)
    eps = 1e-12
    cap["L_us"] = -(cap.y * np.log(cap.p_us + eps) + (1 - cap.y) * np.log(1 - cap.p_us + eps))
    cap["L_mkt"] = -(cap.y * np.log(cap.p_mkt + eps) + (1 - cap.y) * np.log(1 - cap.p_mkt + eps))
    cap["d_excess"] = cap.L_us - cap.L_mkt
    cap["m_us"] = SCALE * np.log(cap.p_us / (1 - cap.p_us))
    om = con.execute("""select game_date, home, away, home_exp_margin
                        from odds_market where season_end in (2024,2025,2026)""").fetchdf()
    om["game_date"] = pd.to_datetime(om.game_date)
    cap = cap.merge(om, on=["game_date", "home", "away"], how="left")
    cap = cap.rename(columns={"home_exp_margin": "m_mkt"})

    print("building team game log...", file=sys.stderr)
    g = team_game_log(con)
    print("box style...", file=sys.stderr)
    style = box_style(con, g)
    print("h2h...", file=sys.stderr)
    hh = h2h(g)
    print("officials...", file=sys.stderr)
    refs = officials(con, g)
    print("outs quality...", file=sys.stderr)
    outs = outs_quality(con, g)
    print("continuity...", file=sys.stderr)
    cont = continuity(con, g)

    tg = g.merge(style, on=["season", "game_id", "team"], how="left") \
          .merge(cont, on=["season", "game_id", "team"], how="left")
    tg = tg.merge(outs.rename(columns={"abbrev": "team"}),
                  on=["game_date", "team"], how="left")
    for c in ["out_dpm_pos", "star_out", "n_out_feed"]:
        tg[c] = tg[c].fillna(0.0)  # no report row = no listed outs (only when feed covers date)
    feed_last = pd.Timestamp("2025-12-21")
    tg["outs_feed_covered"] = (tg.game_date <= feed_last).astype(float)

    team_cols = ["game_no", "days_rest", "b2b", "is_3in4", "games_last7", "travel3d_km",
                 "tz_from_home", "tz_change", "road_trip", "home_stand", "form10", "form5",
                 "season_avg_margin", "momentum", "wpct", "streak", "blowout15_15", "sos10",
                 "td_poss", "td_ftr", "td_p3rate", "td_orbp", "td_drbp", "td_astr", "td_tovr",
                 "td_rimrate", "td_opp_ftr_alwd", "td_opp_rim_alwd", "td_opp_p3_alwd",
                 "td_sfl", "td_p3pct_std", "lineup_cont", "starter_churn",
                 "out_dpm_pos", "star_out", "n_out_feed", "outs_feed_covered"]
    H = tg[tg.is_home][["game_id"] + team_cols].rename(
        columns={c: c + "_H" for c in team_cols})
    A = tg[~tg.is_home][["game_id"] + team_cols].rename(
        columns={c: c + "_A" for c in team_cols})
    feat = cap.merge(H, on="game_id", how="inner").merge(A, on="game_id", how="inner")
    feat = feat.merge(hh, on="game_id", how="left").merge(refs, on="game_id", how="left")

    # actual margin
    am = g[g.is_home][["game_id", "margin"]].rename(columns={"margin": "actual_margin"})
    feat = feat.merge(am, on="game_id", how="left")
    feat["resid_us"] = feat.actual_margin - feat.m_us
    feat["dm"] = feat.m_mkt - feat.m_us

    # regime definition (matches mandate numbers: n=1567, +0.0078/gm)
    feat["gn_min"] = np.minimum(feat.game_no_H, feat.game_no_A)
    feat["regimeC"] = feat.gn_min.between(20, 54)

    out_path = os.path.join(OUTDIR, "regimec_features.pkl")
    feat.to_pickle(out_path)
    print(f"wrote {out_path}: {feat.shape}", file=sys.stderr)
    sub = feat[feat.regimeC]
    print(sub.groupby("season")["d_excess"].agg(["count", "mean", "sum"]), file=sys.stderr)


if __name__ == "__main__":
    main()
