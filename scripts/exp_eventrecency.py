"""EXPERIMENT eventrecency (codex refinement of the recency rejection).

Hypothesis: GLOBAL recency weighting is dead (3-season re-gate), but recency may
pay ONLY after regime events, where the season-long four-factor ratings are
biased by the pre-event regime. Events (detected point-in-time from game data):
  (a) TRADE/ARRIVAL  — a >=25 min/g player appears for a team he had no games
      with earlier that season (and HAD games with another team => trade/claim);
  (b) STAR RETURN    — a >=30 min/g player returns after >=15 days out
      (same-season gap);
  (c) COACH CHANGE   — nbapred.market.windows.COACH_CHANGES registry: EMPTY at
      run time => skipped (per spec).

For a team within 15 games after its most recent event, its per-factor ridge
ratings (off AND def) are blended with a post-event-window estimate computed
from ONLY post-event games (opponent-adjusted with the season fit's opponent
ratings), weight ramping with completed games-since-event k:

    w(k) = k / (k + K0),   K0 = 12

K0 is theory-set, not grid-searched: optimal shrinkage weight for a k-game mean
vs a biased prior is k/(k + sigma_game^2/shift^2); team-game factor noise on the
rating scale is ~5 (efg*100) and a star-level regime shift is ~1.5 => K0 ~ 11;
rounded to 12 (blend is conservative also because the season fit already
contains the k post-event games).

Paired capstone rerun: EXACT copy of the scripts/prod_by_season.py loop (oracle
OUT-sets, weekly refits, sched layer, 0.5*FF + 0.5*comp), computing p_ctl
(season-long FF, control replication) and p_exp (event-blended FF) in the same
pass — everything except the FF term is byte-identical between the two.

Gates (paired bootstrap, 2000 resamples, 95% CI on per-game log-loss delta,
positive = improvement):
  1. PRIMARY: p_exp vs baseline data/capstone_pergame_sched.csv p_us, all games.
  2. ISOLATION: p_exp vs p_ctl (same code path, only the FF blend differs).
  3. FOCUSED: both of the above restricted to affected team-games (blend active,
     w>0 for either side).

READ-ONLY DB; no odds/market inputs to the model (market only joins/filters
games exactly as the baseline loop does). Never edits nbapred/.
"""
import sys, os, json, csv, warnings, datetime as dt
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.model.production import (SCALE, sigmoid, last_season_prior,
                                      fit_schedule_layer)
from nbapred.model.team_ratings import TeamRatings, game_rows
from nbapred.model.four_factors import FourFactors, FACTORS, factor_game_rows
from nbapred.model.composition import CompositionModel
from nbapred.market.windows import COACH_CHANGES

# ---- experiment constants (theory-set; see module docstring) ----------------
K0 = 12.0            # blend ramp w = k/(k+K0)
WINDOW_GAMES = 15    # team-games after an event that are "affected"
TRADE_MIN = 25.0     # avg min/g qualifying a trade arrival
STAR_MIN = 30.0      # avg min/g qualifying a star return
RETURN_DAYS = 15     # absence length qualifying a return
MIN_PRIOR_G = 5      # min games in the trailing average (stability)

SEASONS = ("2023-24", "2024-25", "2025-26")
BASELINE = Path(__file__).resolve().parent.parent / "data" / "capstone_pergame_sched.csv"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "exp_eventrecency_pergame.csv"
OUT_JSON = Path(__file__).resolve().parent.parent / "data" / "exp_eventrecency_summary.json"


# ---- event detection (PIT: qualification uses only games before the event;
#      an event is only USED for predictions dated strictly after it, and only
#      via post-event games already completed) --------------------------------

def player_logs(con, season):
    """Played player-games for the season: (player_id, team_id, date, mins)."""
    return con.execute("""
        SELECT s.player_id, s.team_id, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY g.game_date""", [season]).fetchall()


def detect_events(logs):
    """{team_id: sorted [event_date]}, plus detail list for reporting.
    NOTE player_game_stats has no season before 2023-24 => trailing averages are
    within-season only (early-season returns by players with <MIN_PRIOR_G games
    this season are undetectable — accepted limitation, noted in report)."""
    byp = {}
    for pid, tid, d, m in logs:
        d = d.date() if hasattr(d, "date") else d
        byp.setdefault(int(pid), []).append((d, int(tid), float(m)))
    events = {}          # (team, date) -> set(kinds)
    details = []
    for pid, gs in byp.items():
        gs.sort()
        seen_teams = set()
        for i, (d, tid, m) in enumerate(gs):
            prior = gs[max(0, i - 10):i]           # last up-to-10 games before
            avg = np.mean([x[2] for x in prior]) if prior else 0.0
            npr = len(prior)
            # (a) trade/arrival: first game with tid, had games w/ other teams
            if tid not in seen_teams and seen_teams and npr >= MIN_PRIOR_G \
                    and avg >= TRADE_MIN:
                events.setdefault((tid, d), set()).add("trade")
                details.append(dict(kind="trade", player_id=pid, team=tid,
                                    date=str(d), avg_min=round(float(avg), 1)))
            # (b) star return: same-season gap >= RETURN_DAYS
            if i > 0 and (d - gs[i - 1][0]).days >= RETURN_DAYS \
                    and npr >= MIN_PRIOR_G and avg >= STAR_MIN:
                events.setdefault((tid, d), set()).add("return")
                details.append(dict(kind="return", player_id=pid, team=tid,
                                    date=str(d), avg_min=round(float(avg), 1),
                                    days_out=(d - gs[i - 1][0]).days))
            seen_teams.add(tid)
    # (c) coach change — registry empty => skip (spec)
    assert not COACH_CHANGES, "registry populated; add coach events"
    ev_by_team = {}
    for (tid, d) in events:
        ev_by_team.setdefault(tid, []).append(d)
    for tid in ev_by_team:
        ev_by_team[tid].sort()
    return ev_by_team, details


# ---- event-blended four factors --------------------------------------------

class EventFourFactors(FourFactors):
    """Season-long FF fit + per-call override of a team's per-factor off/def
    ratings with post-event window estimates. margin_neutral() (inherited) is
    the control; margin_neutral_ev() applies overrides."""

    def _pred_f(self, f, tid, oid, is_home, ov):
        m = self.fms[f]
        off = ov.get(tid, {}).get(f, (None, None))[0]
        de = ov.get(oid, {}).get(f, (None, None))[1]
        if off is None:
            off = m.off.get(tid, 0.0)
        if de is None:
            de = m.deff.get(oid, 0.0)
        return m.mu + off - de + (m.home if is_home else 0.0)

    def _eortg_ev(self, tid, oid, is_home, ov):
        xf = np.array([self._pred_f(f, tid, oid, is_home, ov) for f in FACTORS])
        return float(xf @ self.W[:4] + self.W[4])

    def margin_neutral_ev(self, home_id, away_id, ov):
        return (self._eortg_ev(home_id, away_id, False, ov)
                - self._eortg_ev(away_id, home_id, False, ov))


class EventState:
    """Per-season event windows + full-season factor rows (PIT enforced by
    explicit date < gd filters at every use)."""

    def __init__(self, con, season):
        self.events, self.details = detect_events(player_logs(con, season))
        rows = factor_game_rows(con, season)     # full season; date-filtered on use
        self.off_rows, self.def_rows = {}, {}
        for r in rows:
            self.off_rows.setdefault(r["tid"], []).append(r)
            self.def_rows.setdefault(r["oid"], []).append(r)
        for d in (self.off_rows, self.def_rows):
            for t in d:
                d[t].sort(key=lambda r: r["date"])

    def overrides(self, tid, gd, ff):
        """{factor: (off_blend, def_blend)} for team tid predicting on gd, or
        {}. Returns (ov, w, k)."""
        evs = [e for e in self.events.get(tid, []) if e < gd]
        if not evs or not ff.ready:
            return {}, 0.0, 0
        e = evs[-1]                                   # most recent event
        post_o = [r for r in self.off_rows.get(tid, []) if e <= r["date"] < gd]
        post_d = [r for r in self.def_rows.get(tid, []) if e <= r["date"] < gd]
        k = len(post_o)
        # affected window: predicted game is game k+1 since event, need <= 15,
        # and >=1 completed post-event game to estimate from
        if not (1 <= k <= WINDOW_GAMES - 1):
            return {}, 0.0, k
        w = k / (k + K0)
        ov = {}
        for f in FACTORS:
            m = ff.fms[f]
            off_est = float(np.mean([
                100.0 * r[f] - m.mu + m.deff.get(r["oid"], 0.0)
                - (m.home if r["home"] else 0.0) for r in post_o]))
            if post_d:
                def_est = float(np.mean([
                    m.mu + m.off.get(r["tid"], 0.0)
                    + (m.home if r["home"] else 0.0) - 100.0 * r[f]
                    for r in post_d]))
                de = (1 - w) * m.deff.get(tid, 0.0) + w * def_est
            else:
                de = None
            ov[f] = ((1 - w) * m.off.get(tid, 0.0) + w * off_est, de)
        return {tid: ov}, w, k


# ---- production fit, copied from nbapred/model/production.py (READ-ONLY
#      source; only change: EventFourFactors + paired margin) -----------------

def fit_production_pair(con, season, before, w_comp=0.7):
    comp = CompositionModel(con, before=before)
    ff = EventFourFactors().fit(con, season, before=before)   # FF_LUCK off = baseline
    he, b_hb2b, b_ab2b, b_hdead, b_adead = fit_schedule_layer(con, before)
    tr = TeamRatings(ridge=25.0).fit(game_rows(con, before=before, season=season))
    prior = last_season_prior(con, season)
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    games_played = dict(con.execute("""
        SELECT team_id, count(*) FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL AND game_date < ? GROUP BY 1""", [season, before]).fetchall())

    class Predictor:
        ff_ready = ff.ready

        def ratings_margin(self, home_id, away_id):
            m = tr.pred_margin(home_id, away_id)
            gh = games_played.get(home_id, 0); ga = games_played.get(away_id, 0)
            wh = max(0.0, 1 - gh / 20.0); wa = max(0.0, 1 - ga / 20.0)
            ph = prior.get(id2ab.get(home_id, ""), 0.0)
            pa = prior.get(id2ab.get(away_id, ""), 0.0)
            return m + wh * ph - wa * pa

        def p_pair(self, home_id, away_id, out_home, out_away, gd,
                   b2b_home, b2b_away, ev: EventState):
            sched = (he + (b_hb2b if b2b_home else 0.0)
                     + (b_ab2b if b2b_away else 0.0))
            cm = comp.margin(home_id, away_id, out_home, out_away, gd,
                             home_edge=0.0)
            if ff.ready:
                fm_ctl = ff.margin_neutral(home_id, away_id)
                ovh, wh_, kh = ev.overrides(home_id, gd, ff)
                ova, wa_, ka = ev.overrides(away_id, gd, ff)
                ov = {**ovh, **ova}
                fm_exp = ff.margin_neutral_ev(home_id, away_id, ov) if ov else fm_ctl
                m_ctl = 0.5 * fm_ctl + 0.5 * cm + sched
                m_exp = 0.5 * fm_exp + 0.5 * cm + sched
                return (float(sigmoid(m_ctl / SCALE)), float(sigmoid(m_exp / SCALE)),
                        wh_, wa_, kh, ka)
            rm = self.ratings_margin(home_id, away_id) - tr.home
            m = w_comp * cm + (1 - w_comp) * rm + sched
            p = float(sigmoid(m / SCALE))
            return p, p, 0.0, 0.0, 0, 0

    return Predictor()


# ---- capstone loop: copied from scripts/prod_by_season.py (default env path =
#      oracle OUT-sets, weekly refit), paired p_ctl/p_exp ---------------------

def season_run(season):
    con = connect(read_only=True)
    ev = EventState(con, season)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by = {}; order = []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    model = comp = None; last = None
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production_pair(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        p_ctl, p_exp, w_h, w_a, k_h, k_a = model.p_pair(
            h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd,
            b2b(h.team_id, gd), b2b(a.team_id, gd), ev)
        rows.append(dict(season=season, game_id=gid, game_date=str(gd)[:10],
                         home=h.team_abbrev, away=a.team_abbrev,
                         y=int(h.wl == "W"), p_ctl=p_ctl, p_exp=p_exp,
                         p_mkt=float(pmv), w_home=round(w_h, 4),
                         w_away=round(w_a, 4), k_home=k_h, k_away=k_a))
    con.close()
    n_ev = sum(len(v) for v in ev.events.values())
    print(f"[{season}] games={len(rows)} events={n_ev} "
          f"(trade={sum(1 for d in ev.details if d['kind']=='trade')}, "
          f"return={sum(1 for d in ev.details if d['kind']=='return')}) "
          f"affected={sum(1 for r in rows if r['w_home'] > 0 or r['w_away'] > 0)}",
          flush=True)
    return rows, ev.details


# ---- gates ------------------------------------------------------------------

def pg_ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_gate(delta, n_boot=2000, seed=7):
    """delta per game (positive = experiment improves). Mean + 95% CI."""
    delta = np.asarray(delta, float)
    if len(delta) == 0:
        return dict(n=0, mean=None, lo=None, hi=None, verdict="NS")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    verdict = ("PASS" if lo > 0 else "FAIL" if hi < 0 else "NS")
    return dict(n=int(len(delta)), mean=float(delta.mean()),
                lo=float(lo), hi=float(hi), verdict=verdict)


def main():
    import pandas as pd
    all_details = {}
    if "--analyze-only" in sys.argv and OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV, dtype={"game_id": str})
    else:
        all_rows = []
        for s in SEASONS:
            rows, details = season_run(s)
            all_rows += rows
            all_details[s] = details
        with open(OUT_CSV, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wtr.writeheader(); wtr.writerows(all_rows)
        df = pd.DataFrame(all_rows)
    df["game_id"] = df.game_id.astype(str).str.zfill(10)
    base = pd.read_csv(BASELINE, dtype={"game_id": str})[["game_id", "y", "p_us"]] \
        .rename(columns={"y": "y_b", "p_us": "p_base"})
    base["game_id"] = base.game_id.str.zfill(10)
    df = df.merge(base, on="game_id", how="inner")
    assert (df.y == df.y_b).all(), "y mismatch vs baseline join"

    df["ll_base"] = pg_ll(df.y, df.p_base)
    df["ll_ctl"] = pg_ll(df.y, df.p_ctl)
    df["ll_exp"] = pg_ll(df.y, df.p_exp)
    df["aff"] = (df.w_home > 0) | (df.w_away > 0)

    res = dict(config=dict(K0=K0, WINDOW_GAMES=WINDOW_GAMES, TRADE_MIN=TRADE_MIN,
                           STAR_MIN=STAR_MIN, RETURN_DAYS=RETURN_DAYS,
                           MIN_PRIOR_G=MIN_PRIOR_G, coach_changes="skipped-empty"),
               seasons={},
               events={s: len(v) for s, v in all_details.items()} or "see run log")
    for s, g in df.groupby("season"):
        res["seasons"][s] = dict(
            n=int(len(g)),
            ll_base=round(float(g.ll_base.mean()), 4),
            ll_ctl=round(float(g.ll_ctl.mean()), 4),
            ll_exp=round(float(g.ll_exp.mean()), 4),
            n_aff=int(g.aff.sum()),
            gate_vs_base=boot_gate(g.ll_base - g.ll_exp),
            gate_vs_ctl=boot_gate(g.ll_ctl - g.ll_exp),
            aff_gate_vs_base=boot_gate((g.ll_base - g.ll_exp)[g.aff]),
            aff_gate_vs_ctl=boot_gate((g.ll_ctl - g.ll_exp)[g.aff]))
    res["pooled"] = dict(
        n=int(len(df)),
        gate_vs_base=boot_gate(df.ll_base - df.ll_exp),
        gate_vs_ctl=boot_gate(df.ll_ctl - df.ll_exp),
        aff_gate_vs_base=boot_gate((df.ll_base - df.ll_exp)[df.aff]),
        aff_gate_vs_ctl=boot_gate((df.ll_ctl - df.ll_exp)[df.aff]),
        n_aff=int(df.aff.sum()),
        ctl_vs_base_maxdiff=float((df.p_ctl - df.p_base).abs().max()))
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
