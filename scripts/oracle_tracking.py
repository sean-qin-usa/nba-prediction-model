"""PAID_ORACLES #7 — TRACKING-GEOMETRY oracle (Second Spectrum / Genius
license stand-in; committed "would-buy" per docs/PAID_ORACLES.md two-tier
policy).

Measures what full matchup ("who actually guarded whom") tracking data adds to
the WIN-PROB margin model. The tracking-derived axis production cannot see:
Four Factors sees RESULTS, not ASSIGNMENTS.

ORACLE CAVEAT (label): leagueseasonmatchups is a SEASON-LEVEL aggregate — the
per-defender D36 rating carries full-season hindsight within each season. This
is the BOUGHT-TIER CEILING for the margin channel; a real vendor feed would
supply the same axis daily (as-of), so live value <= this number.

PRE-REGISTERED CONSTRUCTION (one variant, no sweeps):
  * per-defender D36 on-ball rating (exact gate_creation_defense.py / D36
    construction): matchup rows with PARTIAL_POSS >= 10; expected pts =
    offensive player's season pts-per-partial-poss x PARTIAL_POSS;
    r = 100 * (expected - actual) / (partial_poss + 800)  [EB shrink, +800
    poss prior mass]; positive = good on-ball defender.
  * team ON-BALL aggregate as-of date d: minutes-weighted mean of r over the
    team's rotation — weights = player minutes over the team's last 10 games
    strictly before d, tonight's OUT set excluded, unrated players counted at
    r = 0 (league mean); team with no trailing games -> 0.0.
  * margin term: margin += k * (onball_home - onball_away). k fit walk-forward
    at every weekly refit (same cadence as the model): univariate OLS (with
    intercept, intercept discarded) of the CONTROL margin residual
    (actual_margin - control_margin) on the onball diff over ALL previously
    predicted games of the run (accumulated across seasons, chronological),
    shrunk toward 0 with 600-game prior mass: k = n/(n+600) * k_ols. k0 = 0.

CONTROL = same-run unmodified fit_production (includes the D62 carry). Loop is
a copy of scripts/prod_by_season.py with the default oracle-outs (bought-
availability tier), weekly refit. Reads nbapred.db read_only; never edits
nbapred/.

Outputs: data/oracle_tracking_pergame.csv, data/oracle_tracking_results.json.
Run:      python scripts/oracle_tracking.py            (full walk-forward)
Analyze:  python scripts/oracle_tracking.py --analyze  (bootstrap from CSV)
"""
import sys, warnings, json, glob
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import orjson
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.production import fit_production, SCALE, sigmoid
from nbapred.model.composition import CompositionModel
from nbapred.ingest.nba_stats import _frames

SEASONS = ("2023-24", "2024-25", "2025-26")
K_PRIOR_MASS = 600.0                      # games of shrink-to-0 prior mass
TRAIL_GAMES = 10                          # rotation window for minutes weights
CHAOS = {"PHI", "DAL", "BKN", "POR", "SAS"}   # D50 worst-decile teams
HEAVY_FAV = 0.35                          # |p_mkt - 0.5| threshold
PERGAME_CSV = "data/oracle_tracking_pergame.csv"
RESULTS_JSON = "data/oracle_tracking_results.json"


def load_matchup_def(season):
    """{defender_id: D36 on-ball rating} — exact D36/gate_creation_defense
    construction from the season-level leagueseasonmatchups cache (oracle)."""
    for f in glob.glob("data/raw/nba_api/matchups/*.json"):
        d = orjson.loads(open(f, "rb").read())
        if d["params"].get("season") == season:
            df = list(_frames(d["response"]).values())[0]
            df = df[df.PARTIAL_POSS >= 10]
            off = df.groupby("OFF_PLAYER_ID").agg(p=("PLAYER_PTS", "sum"),
                                                  q=("PARTIAL_POSS", "sum"))
            rate = (off.p / off.q).to_dict()
            df["exp"] = df.OFF_PLAYER_ID.map(rate) * df.PARTIAL_POSS
            g = df.groupby("DEF_PLAYER_ID").agg(a=("PLAYER_PTS", "sum"),
                                                e=("exp", "sum"),
                                                q=("PARTIAL_POSS", "sum"))
            g["r"] = 100 * (g.e - g.a) / (g.q + 800.0)
            return {int(k): float(v) for k, v in g["r"].items()}
    raise RuntimeError(f"no matchup cache for {season}")


def fit_k(acc):
    """Walk-forward k from accumulated (onball_diff, residual) pairs; OLS with
    intercept (discarded), shrunk toward 0 with K_PRIOR_MASS games."""
    n = len(acc)
    if n < 2:
        return 0.0
    a = np.array(acc)
    d, r = a[:, 0], a[:, 1]
    vd = d.var()
    if vd <= 0:
        return 0.0
    slope = ((d - d.mean()) * (r - r.mean())).mean() / vd
    return float(n / (n + K_PRIOR_MASS) * slope)


def season_run(season, acc):
    """Copy of the prod_by_season loop (default oracle-outs, weekly refit)
    with the pre-registered on-ball tracking term added next to the same-run
    control. `acc` is the cross-season (diff, resid) accumulator (mutated)."""
    con = connect(read_only=True)
    mdef = load_matchup_def(season)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    minmap = {}
    for r in pm.itertuples():
        minmap.setdefault((r.game_id, int(r.team_id)), {})[int(r.player_id)] = float(r.mins)
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, pts, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by = {}; order = []
    for x in meta.itertuples():
        if x.game_id not in by: order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    # this season's team-game sequence (for trailing-minutes rotation windows)
    tgames = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tgames.setdefault(x.team_id, []).append((d, x.game_id))
    for t in tgames: tgames[t].sort()

    def onball(tid, d, outs):
        past = [g for (dd, g) in tgames.get(tid, []) if dd < d][-TRAIL_GAMES:]
        if not past:
            return 0.0
        w = {}
        for gid2 in past:
            for pid, m in minmap.get((gid2, tid), {}).items():
                w[pid] = w.get(pid, 0.0) + m
        w = {p: m for p, m in w.items() if p not in outs}
        den = sum(w.values())
        if den <= 0:
            return 0.0
        return sum(m * mdef.get(p, 0.0) for p, m in w.items()) / den

    # b2b flags (PIT: prior days' games are known before tip)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)
    import datetime as _dt
    def b2b(tid, d): return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    y, pc, pv, pmk = [], [], [], []
    rows = []
    model = comp = None; last = None; k = fit_k(acc)
    for gid in order:
        recs = by[gid]
        if len(recs) != 2: continue
        mu = recs[0].matchup
        host = mu.split("@")[-1].strip() if "@" in mu else mu.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a: continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            k = fit_k(acc)                       # k refit at the same cadence
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None: continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        m_ctrl = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd,
                              b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        ob_h = onball(h.team_id, gd, outs[h.team_id])
        ob_a = onball(a.team_id, gd, outs[a.team_id])
        diff = ob_h - ob_a
        m_var = m_ctrl + k * diff
        y.append(int(h.wl == "W"))
        pc.append(float(sigmoid(m_ctrl / SCALE)))
        pv.append(float(sigmoid(m_var / SCALE)))
        pmk.append(pmv)
        acc.append((diff, float(h.pts - a.pts) - float(m_ctrl)))   # future refits only
        rows.append((season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                     y[-1], pc[-1], pv[-1], float(pmv), round(ob_h, 4), round(ob_a, 4),
                     round(k, 5), len(outs[h.team_id]), len(outs[a.team_id])))
    con.close()
    y = np.array(y)
    print(f"{season}: n={len(y)} ctrl={log_loss(y, pc):.4f} var={log_loss(y, pv):.4f} "
          f"mkt={log_loss(y, np.array(pmk)):.4f} k_final={k:+.5f} acc_n={len(acc)}",
          flush=True)
    return rows


# ---------------------------------------------------------------- analysis --
def _ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(delta, B=2000, seed=42):
    """Paired bootstrap mean + 95% CI of per-game logloss delta (ctrl - var;
    positive = tracking variant better)."""
    n = len(delta)
    if n == 0:
        return dict(n=0, mean=None, lo=None, hi=None)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, n, (B, n))].mean(axis=1)
    return dict(n=int(n), mean=round(float(delta.mean()), 6),
                lo=round(float(np.percentile(means, 2.5)), 6),
                hi=round(float(np.percentile(means, 97.5)), 6))


def analyze():
    import csv
    rows = list(csv.DictReader(open(PERGAME_CSV)))
    y = np.array([int(r["y"]) for r in rows])
    pc = np.array([float(r["p_ctrl"]) for r in rows])
    pv = np.array([float(r["p_var"]) for r in rows])
    pmk = np.array([float(r["p_mkt"]) for r in rows])
    seas = np.array([r["season"] for r in rows])
    chaos = np.array([r["home"] in CHAOS or r["away"] in CHAOS for r in rows])
    hfav = np.abs(pmk - 0.5) > HEAVY_FAV
    delta = _ll(y, pc) - _ll(y, pv)          # positive = variant better
    out = {"construction": "D36 minutes-weighted team on-ball aggregate, "
                           "walk-forward k, 600-game shrink (see docstring)",
           "oracle_caveat": "season-level matchup data = hindsight within "
                            "season; bought-tier ceiling",
           "seasons": {}, "logloss": {}}
    for s in SEASONS:
        m = seas == s
        out["seasons"][s] = boot(delta[m])
        out["logloss"][s] = {"n": int(m.sum()),
                             "ctrl": round(float(_ll(y[m], pc[m]).mean()), 4),
                             "var": round(float(_ll(y[m], pv[m]).mean()), 4),
                             "mkt": round(float(_ll(y[m], pmk[m]).mean()), 4)}
    out["pooled"] = boot(delta)
    out["chaos_teams"] = boot(delta[chaos])
    out["heavy_fav"] = boot(delta[hfav])
    out["chaos_and_heavy_fav"] = boot(delta[chaos & hfav])
    ks = [float(r["k"]) for r in rows]
    out["k_final_by_season"] = {s: round([float(r["k"]) for r in rows
                                          if r["season"] == s][-1], 5) for s in SEASONS}
    out["mean_abs_onball_diff"] = round(float(np.mean(
        [abs(float(r["ob_home"]) - float(r["ob_away"])) for r in rows])), 4)
    json.dump(out, open(RESULTS_JSON, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if "--analyze" not in sys.argv:
        import csv
        acc = []                                  # (diff, resid), cross-season
        allrows = []
        for s in SEASONS:
            allrows += season_run(s, acc)
        with open(PERGAME_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["season", "game_id", "game_date", "home", "away", "y",
                        "p_ctrl", "p_var", "p_mkt", "ob_home", "ob_away", "k",
                        "n_out_home", "n_out_away"])
            w.writerows(allrows)
    analyze()
