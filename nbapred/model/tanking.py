"""April tank term (D73, gate-passed +0.00401 CI(+0.00112,+0.00672)).

production margin += k * (tank_home - tank_away), where tank(team) is the
team's tank composite, ACTIVE ONLY when the scored team's games-played
(before the game) >= 55 — exactly 0.0 otherwise, so games with both teams
gp<55 are bitwise-unchanged.

TANK COMPOSITE (equal-weight z-scores; every input strictly < game date;
construction ported verbatim from scripts/apr_program.py, the gate run):
  a. VET-MINUTES SHARE SHIFT: season-baseline share of team minutes to
     age>=27 players (darko_history age, asof) MINUS last-5 share.
  b. ROTATION EXPERIMENTATION: b1 = distinct proxy-starters (top-5 by
     seconds) in the last 10 games; b2 = season-first proxy-starts in the
     last 5 games.
  c. SHUTDOWN LISTINGS: distinct (player, game_date) injury-report 'Out'
     entries in the trailing 14 days with rest/management/maintenance
     reasons on season-mpg>=25 players; NaN (z=0) when the PIT report
     feed has no coverage in the trailing 7 days.
  d. STANDINGS INCENTIVE: play-in games-back, lottery-lock, seed-lock.
STANDARDIZATION: expanding z over ALL team-dates strictly before the
current date, pooled across teams/seasons (from the CORPUS FLOOR, see
below); z=0 until 300 prior obs; clip +-4; NaN raw -> z=0.

CORPUS FLOOR (D112; was a hardcoded `season >= '2022-23'` literal in four
queries until 2026-08-01 — hall-of-shame #8 in the D110 audit). The floor is
now DERIVED from the data by season_floor(): the earliest season such that
it and every later season have >= 99%% of their completed 002 games covered
by player_game_stats (components a and b are built from box scores, so a
season with holes would poison the pooled z and the k fit frame). On the
current corpus that resolves to 2021-22 (2020-21 has 780/1080 box scores,
2019-20 has none). Set TANK_SEASON_FLOOR=<season> to override — used by
same-run A/B controls and by the test suite to reproduce the pre-D112
2022-23-floor behaviour bitwise.
tank = mean(z_a, (z_b1+z_b2)/2, z_c, (z_d1+z_d2+z_d3)/3).

K ESTIMATION (production design — differs from the gate's estimator; see
DECISIONS.md D73-SHIPPED): fit walk-forward at fit_production time,
self-contained from the DB (live parity: predict_today gets the same k as
the backtest with no accumulated-state file), mirroring how
fit_schedule_layer estimates its terms:
  rows: ALL completed games strictly before `before` with a nonzero
        gp>=55-gated tank diff (corpus floor onward)
  OLS:  home_margin ~ [1, tank_diff, wpct_diff]  — season-to-date
        wpct-diff control de-confounds tank from team quality, exactly
        like the dead-team term's control in fit_schedule_layer
  k = n/(n+600) * beta_tank  (600-game shrink toward 0, = SCHED_SHRINK),
      0 until 20 active rows, clipped +-15.
Measured vs the gate arm (apr_capstone_pergame.csv): the shrink keeps
~70-75%% of the gate's MLE k (-1.6..-2.3 vs -2.5..-3.3); per-season
capstone deviation -0.0019/-0.0001/+0.0014 (23-24 BETTER: production has
2022-23 active history the gate's accumulated-hist design lacked; 25-26
slightly smaller effect from the shrink), pooled delta preserved.

LIVE PATH (D68 discipline): for games not yet in nba_games (tonight's
slate), pass virtual_games=[(season, team_id, game_date), ...] — virtual
rows are appended to the team-game table and flow through the SAME
component code paths, which only ever read strictly-prior data, so the
live number is by construction what the backtest will later compute.
predict_today primes the module cache with the slate BEFORE calling
fit_production (see get_tank_model).
"""
from __future__ import annotations

import datetime as dt
import math
import os
import warnings

import numpy as np
import pandas as pd

VET_AGE = 27.0
SHUTDOWN_MPG = 25.0
SHUTDOWN_WINDOW = 14          # days
Z_MIN_N = 300                 # prior obs before z activates
Z_CLIP = 4.0
GP_ACTIVE = 55                # term active only when scored team gp>=55
K_MIN_ACTIVE = 20             # k stays 0 until this many active rows
K_CLIP = 15.0
K_SHRINK = 600.0              # games of prior mass toward k=0 (= SCHED_SHRINK)

FLOOR_MIN_BOX_COVERAGE = 0.99   # share of 002 games needing a box score
FLOOR_FALLBACK = "2022-23"      # pre-D112 hardcoded literal (safety net only)

# D155 PINNED FLOOR. The derived floor below is a function of DATA COVERAGE, so
# it MOVES when a backfill lands — and it moved twice without anyone changing a
# line of model code:
#   D131  box-score completion pushed it 2021-22 -> 2020-21 six hours AFTER the
#         D122 certification, silently invalidating that table.
#   D153  the D152 historical backfill pushed it 2020-21 -> 2014-15, moving
#         35.28% of the 6,148 certified games (max|dp| 0.041, Feb-Mar).
# A baseline that drifts under us cannot anchor a gate: every registered
# comparison is denominated in the floor that was in force when it ran. So the
# resolved value is now PINNED to what the certified stack actually used, and
# the derived value is computed alongside for drift detection only.
# Changing PINNED_SEASON_FLOOR is a MODEL CHANGE and needs a gate + re-cert; it
# is not a bookkeeping edit. TANK_SEASON_FLOOR still overrides for same-run
# controls and pinned-fixture tests.
PINNED_SEASON_FLOOR = "2020-21"     # certified by D132/D153; see floor_audit()

EAST = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DET", "IND", "MIA",
        "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
WEST = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP",
        "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}


def season_floor(con) -> str:
    """Earliest season the tank composite can honestly be built on (D112).

    The composite's a/b components come from player_game_stats box scores, so
    a season with missing box scores would feed a biased pooled z and a thin
    k fit frame. Walk BACKWARD from the newest season and stop at the first
    one whose 002 box-score coverage drops below FLOOR_MIN_BOX_COVERAGE; the
    last season above the bar is the floor (a CONTIGUOUS run, so no holes can
    open up inside the corpus).

    TANK_SEASON_FLOOR overrides (same-run controls / pinned-fixture tests).
    """
    env = os.environ.get("TANK_SEASON_FLOOR")
    if env:
        return env
    if PINNED_SEASON_FLOOR:          # D155: stability beats adaptivity here
        return PINNED_SEASON_FLOOR
    return derived_season_floor(con)


def derived_season_floor(con) -> str:
    """The coverage-derived floor (D112's original rule), now used only by
    floor_audit(). Kept intact so drift is detectable rather than invisible."""
    rows = con.execute("""
        SELECT g.season,
               count(DISTINCT g.game_id)                       AS sched,
               count(DISTINCT CASE WHEN s.game_id IS NOT NULL
                                   THEN g.game_id END)         AS boxed
        FROM (SELECT DISTINCT game_id, season FROM nba_games
              WHERE game_id LIKE '002%' AND wl IS NOT NULL) g
        LEFT JOIN (SELECT DISTINCT game_id FROM player_game_stats
                   WHERE seconds > 0) s USING (game_id)
        GROUP BY 1 ORDER BY 1 DESC""").fetchall()
    floor = None
    for season, sched, boxed in rows:
        if sched and boxed / sched >= FLOOR_MIN_BOX_COVERAGE:
            floor = season
        else:
            break
    return floor or FLOOR_FALLBACK


def floor_audit(con) -> dict:
    """Report pinned vs derived so a corpus change is LOUD, not silent.

    Returns {pinned, derived, drifted}. `drifted=True` does NOT mean anything
    is broken — it means the data now supports a different floor than the one
    the certified stack was built on, and that moving to it is a model change
    requiring a gate and a re-certification (D155). Call this from the daily
    health check and from any capstone run.
    """
    derived = derived_season_floor(con)
    pinned = os.environ.get("TANK_SEASON_FLOOR") or PINNED_SEASON_FLOOR
    return {"pinned": pinned, "derived": derived, "drifted": derived != pinned}


def _to_date(d):
    if hasattr(d, "date") and not isinstance(d, dt.date):
        return d.date()
    if isinstance(d, dt.datetime):
        return d.date()
    return d


# --------------------------------------------------------------------------
# stat construction (verbatim port of apr_program.py; virtual-row support)
# --------------------------------------------------------------------------

def _team_games(con, floor: str) -> pd.DataFrame:
    tg = con.execute("""
        SELECT season, game_id, game_date, team_id, team_abbrev, pts, is_home
        FROM nba_games WHERE game_id LIKE '002%' AND wl IS NOT NULL
          AND season >= ?
        ORDER BY team_id, game_date, game_id""", [floor]).fetchdf()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg["gp_before"] = tg.groupby(["season", "team_id"]).cumcount()
    return tg


def _virtual_frame(con, tg: pd.DataFrame, virtual_games,
                   floor: str) -> pd.DataFrame | None:
    """Validate + normalize virtual (season, team_id, game_date) rows.
    Each must be strictly AFTER the team's last completed game (append-at-end
    semantics keep every trailing window of real rows untouched)."""
    if not virtual_games:
        return None
    ab = dict(con.execute("""SELECT DISTINCT team_id, team_abbrev
        FROM nba_games WHERE season >= ?""", [floor]).fetchall())
    lastd = tg.groupby("team_id")["game_date"].max().to_dict()
    rows, seen = [], set()
    for season, tid, d in virtual_games:
        tid = int(tid)
        d = pd.Timestamp(_to_date(d))
        if (tid, d) in seen:
            continue
        seen.add((tid, d))
        if tid in lastd and d <= lastd[tid]:
            warnings.warn(f"tanking: virtual game ({season},{tid},{d.date()}) "
                          "not after the team's last completed game — skipped")
            continue
        gp = int(((tg.team_id == tid) & (tg.season == season)).sum())
        rows.append((season, f"999V{d.strftime('%Y%m%d')}T{tid}", d, tid,
                     ab.get(tid, "UNK"), np.nan, None, gp))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["season", "game_id", "game_date",
                                       "team_id", "team_abbrev", "pts",
                                       "is_home", "gp_before"])


def _player_minutes(con, floor: str) -> pd.DataFrame:
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds/60.0 AS mins,
               g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
              WHERE season >= ?) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds > 0""", [floor]).fetchdf()
    pm["game_date"] = pd.to_datetime(pm["game_date"])
    return pm


def _attach_age(con, pm: pd.DataFrame, floor: str) -> pd.DataFrame:
    # DARKO ages from the June before the floor season's tip-off, so the
    # merge_asof has a backward match for every game date in the corpus.
    dk = con.execute("""
        SELECT player_id, date, age FROM darko_history
        WHERE date >= ? AND age IS NOT NULL
        ORDER BY date""", [f"{int(floor[:4])}-06-01"]).fetchdf()
    dk["date"] = pd.to_datetime(dk["date"])
    pm = pm.sort_values("game_date").reset_index(drop=True)
    pm = pd.merge_asof(pm, dk.rename(columns={"date": "game_date"}),
                       on="game_date", by="player_id", direction="backward")
    pm["is_vet"] = pm["age"] >= VET_AGE          # NaN age -> False (<27)
    return pm


def _comp_a_vetshift(pm: pd.DataFrame, virt: pd.DataFrame | None) -> pd.DataFrame:
    """Per team-game: a_raw = season-baseline vet-minute share (strictly
    before) minus last-5 share.  Positive = vets recently benched."""
    g = (pm.groupby(["season", "team_id", "game_id", "game_date"])
           .apply(lambda s: s.loc[s.is_vet, "mins"].sum() / max(s.mins.sum(), 1e-9))
           .rename("vet_share").reset_index()
           .sort_values(["season", "team_id", "game_date", "game_id"]))
    if virt is not None:
        vg = virt[["season", "team_id", "game_id", "game_date"]].copy()
        vg["vet_share"] = np.nan
        g = (pd.concat([g, vg], ignore_index=True)
               .sort_values(["season", "team_id", "game_date", "game_id"])
               .reset_index(drop=True))
    grp = g.groupby(["season", "team_id"], sort=False)
    prev = grp["vet_share"].shift(1)
    base = (prev.groupby([g.season, g.team_id]).expanding(min_periods=8)
            .mean().reset_index(level=[0, 1], drop=True))
    last5 = (prev.groupby([g.season, g.team_id]).rolling(5, min_periods=5)
             .mean().reset_index(level=[0, 1], drop=True))
    g["a_raw"] = base - last5
    return g[["season", "team_id", "game_id", "a_raw"]]


def _comp_b_rotation(pm: pd.DataFrame, virt: pd.DataFrame | None) -> pd.DataFrame:
    """b1 = distinct proxy-starters in last 10 games; b2 = season-first
    proxy-starts in last 5 games.  Proxy-starter = top-5 seconds."""
    pm = pm.sort_values(["game_id", "team_id", "mins", "player_id"],
                        ascending=[True, True, False, True])
    top5 = (pm.groupby(["season", "team_id", "game_date", "game_id"])
              .head(5)
              .groupby(["season", "team_id", "game_date", "game_id"])["player_id"]
              .apply(lambda s: frozenset(int(x) for x in s))
              .rename("starters").reset_index()
              .sort_values(["season", "team_id", "game_date", "game_id"]))
    if virt is not None:
        vg = virt[["season", "team_id", "game_date", "game_id"]].copy()
        vg["starters"] = [frozenset()] * len(vg)
        top5 = (pd.concat([top5, vg], ignore_index=True)
                  .sort_values(["season", "team_id", "game_date", "game_id"])
                  .reset_index(drop=True))
    rows = []
    for (season, tid), sub in top5.groupby(["season", "team_id"], sort=False):
        starters = sub["starters"].tolist()
        gids = sub["game_id"].tolist()
        first_idx: dict[int, int] = {}
        for i, st in enumerate(starters):
            for p in st:
                first_idx.setdefault(p, i)
        for i, gid in enumerate(gids):
            b1 = float(len(set().union(*starters[max(0, i - 10):i]))) \
                if i >= 10 else np.nan
            if i >= 5:
                b2 = float(sum(1 for j in range(i - 5, i)
                               for p in starters[j] if first_idx[p] == j))
            else:
                b2 = np.nan
            rows.append((season, tid, gid, b1, b2))
    return pd.DataFrame(rows, columns=["season", "team_id", "game_id",
                                       "b1_raw", "b2_raw"])


def _comp_c_shutdown(con, tg: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """Distinct (player, game_date) rest/management/maintenance 'Out'
    listings on mpg>=25 players in the trailing 14 days, report_date < d.
    NaN when the report feed has no coverage in the trailing 7 days."""
    from ..teams import abbrev_for       # D171: "LA Clippers" resolves here
    ent = con.execute("""
        SELECT DISTINCT i.report_date, i.game_date, i.team, p.player_id
        FROM injury_reports_pit i
        JOIN (SELECT player_id,
                     lower(first_name||' '||last_name) fn FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '||
                          trim(split_part(i.player,',',1)))
        WHERE i.status = 'Out'
          AND (lower(i.reason) LIKE '%rest%'
               OR lower(i.reason) LIKE '%management%'
               OR lower(i.reason) LIKE '%maintenance%')""").fetchdf()
    ent["report_date"] = pd.to_datetime(ent["report_date"])
    ent["game_date"] = pd.to_datetime(ent["game_date"])
    # D171: was `.map({full_name: abbrev})`, which silently dropped every
    # Clippers row because the PDFs say "LA Clippers"; the shutdown signal was
    # therefore structurally blank for 1 of 30 teams in every fit.
    ent["team_ab"] = ent["team"].map(abbrev_for)
    ent = ent.dropna(subset=["team_ab", "player_id"])
    by_team: dict[str, pd.DataFrame] = {
        ab: sub.sort_values("game_date") for ab, sub in ent.groupby("team_ab")}
    report_days = np.sort(pd.to_datetime(con.execute(
        "SELECT DISTINCT report_date FROM injury_reports_pit").fetchdf()
        ["report_date"]).values)

    # per (player, season): cumulative minutes history for mpg-before-d
    pmin: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (pid, season), sub in pm.groupby(["player_id", "season"], sort=False):
        sub = sub.sort_values("game_date")
        pmin[(int(pid), season)] = (sub.game_date.values,
                                    np.cumsum(sub.mins.values))

    def mpg_before(pid, season, d):
        h = pmin.get((int(pid), season))
        if h is None:
            return 0.0
        i = np.searchsorted(h[0], np.datetime64(d))
        return (h[1][i - 1] / i) if i > 0 else 0.0

    out = []
    for r in tg.itertuples():
        d = r.game_date
        i = np.searchsorted(report_days, np.datetime64(d))
        covered = i > 0 and (d - pd.Timestamp(report_days[i - 1])).days <= 7
        if not covered:
            out.append((r.season, r.team_id, r.game_id, np.nan))
            continue
        sub = by_team.get(r.team_abbrev)
        c = 0
        if sub is not None:
            w = sub[(sub.game_date >= d - pd.Timedelta(days=SHUTDOWN_WINDOW))
                    & (sub.game_date < d) & (sub.report_date < d)]
            seen = set()
            for e in w.itertuples():
                key = (int(e.player_id), e.game_date)
                if key in seen:
                    continue
                seen.add(key)
                if mpg_before(e.player_id, r.season, d) >= SHUTDOWN_MPG:
                    c += 1
        out.append((r.season, r.team_id, r.game_id, float(c)))
    return pd.DataFrame(out, columns=["season", "team_id", "game_id", "c_raw"])


def _comp_d_standings(tg: pd.DataFrame) -> pd.DataFrame:
    """Play-in games-back, lottery-lock, seed-locked per team-date.
    Emits rows for every completed-game date (verbatim apr_program) plus any
    virtual dates present in tg (state = W/L strictly before that date)."""
    out = []
    for season, sub in tg.groupby("season", sort=False):
        played = sub[sub["pts"].notna()]
        teams = played[["team_id", "team_abbrev"]].drop_duplicates()
        ab = dict(zip(teams.team_id, teams.team_abbrev))
        if not ab:
            continue
        res = played.copy()
        # win flag needs opponent pts
        opp = played[["game_id", "team_id", "pts"]].rename(
            columns={"team_id": "opp_id", "pts": "opp_pts"})
        res = res.merge(opp, on="game_id")
        res = res[res.team_id != res.opp_id]
        res["win"] = (res.pts > res.opp_pts).astype(int)
        res = res.sort_values(["game_date", "game_id"])
        dates = np.sort(res.game_date.unique())
        extra = np.sort(sub.loc[sub["pts"].isna(), "game_date"].unique())
        W = {t: 0 for t in ab}
        L = {t: 0 for t in ab}

        def wpct(w, l):
            return w / (w + l) if (w + l) else 0.5

        def emit(d, Wd, Ld):
            for conf in (EAST, WEST):
                ids = [t for t in ab if ab[t] in conf]
                if len(ids) < 10:
                    continue
                order = sorted(ids, key=lambda t: (-wpct(Wd[t], Ld[t]),
                                                   -Wd[t], ab[t]))
                rank = {t: i for i, t in enumerate(order)}
                t10 = order[9]
                for t in ids:
                    gp = Wd[t] + Ld[t]
                    rem = 82 - gp
                    gb10 = ((Wd[t10] - Wd[t]) + (Ld[t] - Ld[t10])) / 2.0
                    d1 = float(np.clip(max(gb10, 0.0), 0, 20))
                    d2 = float(max(gb10, 0.0) > rem)
                    i = rank[t]

                    def gap(u, v):   # GB distance between u (better) and v
                        return ((Wd[u] - Wd[v]) + (Ld[v] - Ld[u])) / 2.0
                    lock_below = i == len(order) - 1 or \
                        gap(t, order[i + 1]) > rem
                    lock_above = i == 0 or gap(order[i - 1], t) > rem
                    d3 = float(lock_below and lock_above)
                    out.append((season, t, d, d1, d2, d3))

        played_set = set(dates)
        for d in sorted(set(dates) | set(extra)):
            emit(d, W, L)                       # state BEFORE games on d
            if d in played_set:
                day = res[res.game_date == d]
                for r in day.itertuples():
                    if r.win:
                        W[r.team_id] += 1
                    else:
                        L[r.team_id] += 1
    st = pd.DataFrame(out, columns=["season", "team_id", "game_date",
                                    "d1_raw", "d2_raw", "d3_raw"])
    return st


def _expanding_z(df: pd.DataFrame, cols) -> pd.DataFrame:
    """PIT expanding z per metric: stats from rows with date < d, pooled
    across teams/seasons (corpus-floor burn-in included)."""
    df = df.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)
    date_groups = df.groupby("game_date", sort=True).indices
    date_keys = sorted(date_groups.keys())
    for c in cols:
        x = df[c].values.astype(float)
        z = np.zeros(len(df))
        n, s, ss = 0, 0.0, 0.0
        for key in date_keys:
            idx = np.asarray(date_groups[key])
            if n >= Z_MIN_N:
                mu = s / n
                sd = math.sqrt(max(ss / n - mu * mu, 1e-12))
                sd = max(sd, 1e-6)
                for i in idx:
                    if not math.isnan(x[i]):
                        z[i] = float(np.clip((x[i] - mu) / sd, -Z_CLIP, Z_CLIP))
            for i in idx:
                if not math.isnan(x[i]):
                    n += 1
                    s += x[i]
                    ss += x[i] * x[i]
        df["z_" + c[0:2].rstrip("_")] = z
    return df


def build_tank_stats(con, virtual_games=None, floor: str | None = None) -> pd.DataFrame:
    """Full per-team-game tank table (corpus floor onward). Every row's
    stats use only data strictly before its game_date, so one build serves
    every walk-forward `before` cutoff."""
    floor = floor or season_floor(con)
    tg = _team_games(con, floor)
    virt = _virtual_frame(con, tg, virtual_games, floor)
    pm = _player_minutes(con, floor)
    pm = _attach_age(con, pm, floor)
    a = _comp_a_vetshift(pm, virt)
    b = _comp_b_rotation(pm, virt)
    tga = tg if virt is None else (
        pd.concat([tg, virt], ignore_index=True)
          .sort_values(["team_id", "game_date", "game_id"])
          .reset_index(drop=True))
    try:
        c = _comp_c_shutdown(con, tga, pm)
    except Exception as e:                       # feed missing -> neutral z
        warnings.warn(f"tanking: shutdown component unavailable ({e}); z_c=0")
        c = tga[["season", "team_id", "game_id"]].copy()
        c["c_raw"] = np.nan
    d = _comp_d_standings(tga)
    df = tga.merge(a, on=["season", "team_id", "game_id"], how="left") \
            .merge(b, on=["season", "team_id", "game_id"], how="left") \
            .merge(c, on=["season", "team_id", "game_id"], how="left") \
            .merge(d, on=["season", "team_id", "game_date"], how="left")
    df = _expanding_z(df, ["a_raw", "b1_raw", "b2_raw", "c_raw",
                           "d1_raw", "d2_raw", "d3_raw"])
    df["z_b"] = (df["z_b1"] + df["z_b2"]) / 2.0
    df["z_d"] = (df["z_d1"] + df["z_d2"] + df["z_d3"]) / 3.0
    df["tank_score"] = (df["z_a"] + df["z_b"] + df["z_c"] + df["z_d"]) / 4.0
    return df


# --------------------------------------------------------------------------
# model object + k estimation
# --------------------------------------------------------------------------

class TankModel:
    """Prefit per-(team_id, game_date) tank map + walk-forward k estimator.
    Build once per process (heavy: ~2-3 min); fit_k(before) is cheap."""

    def __init__(self, con, virtual_games=None, floor: str | None = None):
        self.floor = floor or season_floor(con)
        self.df = build_tank_stats(con, virtual_games, floor=self.floor)
        self.map: dict[tuple[int, dt.date], tuple[float, int]] = {
            (int(r.team_id), _to_date(r.game_date)):
                (float(r.tank_score), int(r.gp_before))
            for r in self.df.itertuples()}
        self._build_active_frame(con)

    # -- lookups ----------------------------------------------------------
    def score(self, team_id: int, game_date) -> tuple[float, int]:
        return self.map.get((int(team_id), _to_date(game_date)), (0.0, 0))

    def active(self, team_id: int, game_date) -> float:
        """gp>=55-gated tank score; exactly 0.0 outside the window or for
        unknown (team, date)."""
        t, gp = self.score(team_id, game_date)
        return t if gp >= GP_ACTIVE else 0.0

    def diff(self, home_id: int, away_id: int, game_date) -> float:
        return self.active(home_id, game_date) - self.active(away_id, game_date)

    # -- k estimation ------------------------------------------------------
    def _build_active_frame(self, con):
        """Completed games with nonzero gated tank diff: (date, home margin,
        tank diff, season-to-date wpct diff), sorted by date."""
        g = con.execute("""
            WITH t AS (SELECT season, game_id, game_date, team_id, is_home, pts
                       FROM nba_games WHERE game_id LIKE '002%'
                       AND pts IS NOT NULL AND season >= ?)
            SELECT h.season, h.game_id, h.game_date, h.team_id ht,
                   a.team_id awt, h.pts hp, a.pts ap
            FROM t h JOIN t a USING (game_id)
            WHERE h.is_home AND NOT a.is_home
            ORDER BY h.game_date, h.game_id""", [self.floor]).fetchdf()
        g["game_date"] = pd.to_datetime(g["game_date"])
        # season-to-date wpct before each game, per team
        wins: dict[tuple[str, int], list[int]] = {}
        wp_rows = []
        for r in g.itertuples():
            kh, ka = (r.season, int(r.ht)), (r.season, int(r.awt))
            wh, wa = wins.setdefault(kh, [0, 0]), wins.setdefault(ka, [0, 0])
            wp_rows.append((wh[0] / wh[1] if wh[1] else 0.5,
                            wa[0] / wa[1] if wa[1] else 0.5))
            hw = r.hp > r.ap
            wh[0] += int(hw); wh[1] += 1
            wa[0] += int(not hw); wa[1] += 1
        dates, margins, tsds, wds = [], [], [], []
        for r, (wph, wpa) in zip(g.itertuples(), wp_rows):
            tsd = self.diff(int(r.ht), int(r.awt), r.game_date)
            if tsd == 0.0:
                continue
            dates.append(_to_date(r.game_date))
            margins.append(float(r.hp - r.ap))
            tsds.append(tsd)
            wds.append(wph - wpa)
        self._act_dates = np.array(dates, dtype="datetime64[D]")
        self._act_margin = np.array(margins, float)
        self._act_tsd = np.array(tsds, float)
        self._act_wd = np.array(wds, float)

    def fit_k(self, before=None) -> float:
        """Walk-forward k: OLS home_margin ~ [1, tank_diff, wpct_diff] on
        active rows strictly before `before`; n/(n+600) shrink toward 0."""
        if before is None:
            m = np.ones(len(self._act_dates), bool)
        else:
            m = self._act_dates < np.datetime64(_to_date(before))
        n = int(m.sum())
        if n < K_MIN_ACTIVE:
            return 0.0
        X = np.c_[np.ones(n), self._act_tsd[m], self._act_wd[m]]
        beta = np.linalg.lstsq(X, self._act_margin[m], rcond=None)[0]
        w = n / (n + K_SHRINK)
        return float(np.clip(w * beta[1], -K_CLIP, K_CLIP))


# --------------------------------------------------------------------------
# module cache (one heavy build per process / per DB state)
# --------------------------------------------------------------------------

_CACHE: dict = {}


def get_tank_model(con, virtual_games=None) -> TankModel:
    """Cached TankModel for the current DB state. Live use: call ONCE with
    virtual_games=[(season, team_id, today), ...] for tonight's slate BEFORE
    fit_production — the primed model is what fit_production then picks up."""
    sig = con.execute("""SELECT count(*), max(game_date) FROM nba_games
        WHERE game_id LIKE '002%'""").fetchone()
    floor = season_floor(con)
    key = (int(sig[0]), str(sig[1]), floor)
    if virtual_games is not None or key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = TankModel(con, virtual_games, floor=floor)
    return _CACHE[key]
