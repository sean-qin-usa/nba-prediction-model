"""Filter driver — backtest and daily shadow runs (M0 harness + M1 filter).

Causality contract: for every date d the driver (1) advances the filter to d
(predict_to — evolution + boundary shocks only), (2) emits predictions for
d's games, (3) only then feeds d's results. Backtests REPLAY the filter from
scratch (determinism over caching, V3_SPEC 5); the live daily job does the
same — a full replay is seconds of CPU.

Hyperparameters are refit MONTHLY (walk-forward: only obs strictly before the
refit date), warm-started at the previous optimum; after each refit the filter
replays from scratch under the new hypers up to the refit date and continues.

Shadow rule: predictions are LOGGED (v3_predictions, own version tag) and
never replace production output anywhere.
"""
from __future__ import annotations

import datetime as dt

from .hyper import TeamHyper, fit_team_hyper
from .link import p_home
from .margin import dlm_margin
from .team_dlm import TeamDLM
from .team_obs import build_team_obs, season_boundaries, team_keys

WARMUP_START = dt.date(2021, 10, 1)     # two full warm-up seasons before eval


def run_filter(con, start: dt.date, end: dt.date, hp: TeamHyper | None = None,
               warmup_start: dt.date = WARMUP_START, on_slate=None,
               hyperfit: bool = True, hyperfit_maxiter: int = 60,
               first_fit_maxiter: int = 150, verbose: bool = False):
    """Run the team DLM over [warmup_start, end]; `on_slate(date, dlm)` fires
    once per obs date in [start, end] AFTER predict_to and BEFORE that date's
    updates — the caller reads one-step-ahead predictions there.

    Returns (dlm, hyper_final, hyper_log) where hyper_log records each monthly
    refit (date, TeamHyper, loglik_per_obs).
    """
    obs = build_team_obs(con, warmup_start, end)
    if not obs:
        raise RuntimeError("no team observations in window")
    bounds = season_boundaries(con, warmup_start, end)
    teams = team_keys(obs)
    hyper = hp or TeamHyper()
    hyper_log = []

    def fresh(h):
        return TeamDLM(teams, h, season_boundaries=bounds, start=warmup_start)

    dlm = fresh(hyper)
    seen: list[tuple] = []
    fit_month = None                    # (year, month) of the last hyperfit
    i, n = 0, len(obs)
    while i < n:
        d = obs[i][0]
        if hyperfit and d >= start and (d.year, d.month) != fit_month and seen:
            it = first_fit_maxiter if fit_month is None else hyperfit_maxiter
            hyper = fit_team_hyper(seen, teams, hyper0=hyper, maxiter=it,
                                   seed_boundaries=bounds,
                                   loglik_from=d - dt.timedelta(days=730))
            dlm = fresh(hyper)
            ll = dlm.run(seen)
            hyper_log.append((d, hyper, ll / max(len(seen), 1)))
            fit_month = (d.year, d.month)
            if verbose:
                print(f"[hyperfit {d}] phi={hyper.phi:.4f} q={hyper.q:.4f} "
                      f"r_eff={hyper.r_eff:.1f} r_margin={hyper.r_margin:.1f} "
                      f"kappa={hyper.kappa:.3f} v_bound={hyper.v_bound:.2f} "
                      f"ll/obs={ll / max(len(seen), 1):.4f}", flush=True)
        dlm.predict_to(d)
        if on_slate is not None and start <= d <= end:
            on_slate(d, dlm)
        while i < n and obs[i][0] == d:
            dlm.update(obs[i])
            seen.append(obs[i])
            i += 1
    return dlm, hyper, hyper_log


def slate_games(con, date: dt.date) -> list[dict]:
    """Scheduled 002 games on `date` (finals or not) as
    {game_id, home_ab, away_ab, home_id, away_id}."""
    df = con.execute("""
        SELECT game_id, team_id, team_abbrev, matchup FROM nba_games
        WHERE game_id LIKE '002%' AND game_date = ?""", [date]).fetchdf()
    by = {}
    for r in df.itertuples():
        by.setdefault(r.game_id, []).append(r)
    out = []
    for gid, recs in by.items():
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if h is None or a is None:
            continue
        out.append(dict(game_id=gid, home_ab=h.team_abbrev, away_ab=a.team_abbrev,
                        home_id=int(h.team_id), away_id=int(a.team_id)))
    return out


def daily_shadow(date: dt.date, version: str = "m1.0-dlm",
                 hp: TeamHyper | None = None, write: bool = True):
    """M0 shadow harness (live daily job): replay the filter to `date`,
    predict the slate, persist v3_predictions + team-state snapshot + shocks —
    ALONGSIDE production, never replacing it.

    Opens its own read-only connection and CLOSES it before the write phase:
    DuckDB grants the write lock only when no other connection (any process,
    read-only included) holds the file — verified semantics, D1 discipline.
    """
    from ..db import connect
    from .schema import v3_writer
    from .shocks import detect_shocks, log_shocks

    con = connect(read_only=True)
    # obs STRICTLY before `date` (a retro shadow run must not see the slate's
    # own finals); boundaries THROUGH `date` (an opener-day run must shock).
    obs = [o for o in build_team_obs(con, WARMUP_START, date) if o[0] < date]
    if not obs:
        con.close()
        raise RuntimeError("no team observations before shadow date")
    bounds = season_boundaries(con, WARMUP_START, date)
    slate = slate_games(con, date)
    shocks = detect_shocks(con, date)
    team_ids = dict(con.execute(
        "SELECT DISTINCT team_abbrev, team_id FROM nba_games "
        "WHERE game_id LIKE '002%'").fetchall())
    con.close()

    teams = team_keys(obs)
    hyper = hp or fit_team_hyper(obs, teams, seed_boundaries=bounds,
                                 maxiter=150,
                                 loglik_from=date - dt.timedelta(days=730))
    dlm = TeamDLM(teams, hyper, season_boundaries=bounds, start=WARMUP_START)
    dlm.run(obs)
    dlm.predict_to(date)
    rows = []
    asof_ts = dt.datetime.combine(date, dt.time(0, 0))
    for g in slate:
        mu_n, sigma = dlm_margin(dlm, g["home_ab"], g["away_ab"],
                                 sigma_game=hyper.r_margin ** 0.5)
        mu = mu_n + float(dlm.x[1])          # + filter's own home-edge state
        rows.append((g["game_id"], asof_ts, "side", mu, sigma,
                     p_home(mu), version))
    if write and rows:
        with v3_writer() as w:
            w.executemany(
                "INSERT OR REPLACE INTO v3_predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows)
            snap = dlm.snapshot_rows(team_ids)
            w.execute('DELETE FROM player_states WHERE "asof" = ? AND dim IN '
                      "('team_off','team_def','league_mu','home_edge')", [date])
            w.executemany("INSERT INTO player_states VALUES (?, ?, ?, ?, ?, ?)",
                          [(date, e, t, dim, m, v) for (_, e, t, dim, m, v) in snap])
            log_shocks(w, shocks)
    return rows
