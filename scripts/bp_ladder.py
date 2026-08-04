#!/usr/bin/env python3
"""BP-LADDER — PART A of the BIGPLAYER capstone: the JOINT information ceiling.

Pre-registered in data/bigplayer_prereg.md
(sha256 fbcea42eafa263892c4c9b1874eb4ed4d21d2d7059311ba909541982dc89642c).

ONE walk-forward pass per season computes EVERY tier, so all tiers share the
identical weekly fit_production / CompositionModel fits and differ ONLY in the
information consumed at prediction time.  Anything else confounds the ladder
with refit jitter.

BUYABLE STACK (all rungs pregame-legitimate; see the prereg for the per-tier
table/column justification):
  T0  availability-BLIND floor           outs = {} both teams
  T1  + 5PM official injury report       injury_reports_pit, status='Out',
                                         report_date = game_date, edition 05PM
  T2  + official inactive list (T-30)    game_inactives  (UNION with T1)
  T3  + purchased minutes projections    SIMULATED: clip(real + N(0,sigma),0,48),
                                         sigma = MAE/0.79788, MAE 4.0 primary
                                         (3.0 / 5.0 sensitivity)
  T4  + tracking from PRIOR GAMES        D36/D72 on-ball rating built from the
                                         PREVIOUS season's leagueseasonmatchups
                                         aggregate (inert on 2023-24)
  T5  + best talent as of the prior day  DARKO(date<=gd-1) + EPM(asof<=gd-1),
                                         50/50 in z-space, rescaled to DARKO

CLAIRVOYANT ARMS (PART C — unattainable, reported separately, NEVER mixed in):
  C1  T2 + perfect availability          who actually appeared tonight.  This is
                                         ALSO the D132 certified default
                                         construction => the control-hash arm.
  C2  C1 + perfect minutes               realised minutes tonight
  C3  C2 + perfect talent                60-day CENTRED DARKO (D97)
  C4  T3 + SAME-season tracking          D72's own construction

Read-only DB.  nbapred/ untouched.  No production default changed.
Out: data/bp_ladder_pergame.csv, data/bp_ladder_diag.json
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)                                            # BEFORE numpy

# --- pin the certified environment explicitly (D132 defaults) --------------
os.environ["TANK_SEASON_FLOOR"] = "2020-21"        # PINNED per the brief
for _v in ("LATE_STATE", "ORACLE_MINUTES", "INACTIVE_OUTS", "REPORT_OUTS",
           "COVID_GUARD", "FF_LUCK", "OCT_BRIDGE", "OCT_BRIDGE_TRAIL",
           "TANK_TERM"):
    os.environ.pop(_v, None)                        # unset = certified default

import numpy as np                                                # noqa: E402
import orjson                                                     # noqa: E402

from nbapred.db import connect                                    # noqa: E402
from nbapred.eval.metrics import log_loss                         # noqa: E402
from nbapred.ingest.nba_stats import _frames                      # noqa: E402
from nbapred.model.composition import (ROSTER_DAYS,               # noqa: E402
                                       CompositionModel)
from nbapred.model.october_bridge import rotation_empty           # noqa: E402
from nbapred.model.production import (SCALE, fit_production,      # noqa: E402
                                      sigmoid)

SEASONS = ("2023-24", "2024-25", "2025-26")
PREV = {"2023-24": "2022-23", "2024-25": "2023-24", "2025-26": "2024-25"}
MAES = (3.0, 4.0, 5.0)
PRIMARY_MAE = 4.0
SIG_OF_MAE = {m: m / 0.7978845608 for m in MAES}   # MAE of N(0,s) = s*sqrt(2/pi)
SEED = 20260803
K_PRIOR_MASS = 600.0        # D72 verbatim
TRAIL_GAMES = 10            # D72 verbatim
HALF_WINDOW = 30            # D97 verbatim (60-day centred)
OUT_CSV = REPO / "data" / "bp_ladder_pergame.csv"
OUT_JSON = REPO / "data" / "bp_ladder_diag.json"


# ----------------------------------------------------------- information ----
def report_out_map(con):
    """{(game_date, team_abbrev): {player_id}} from the official 5PM report.
    VERBATIM from scripts/prod_by_season.py::report_out_map."""
    from nba_api.stats.static import teams as _t
    name2ab = {t["full_name"]: t["abbreviation"] for t in _t.get_teams()}
    rows = con.execute("""
        SELECT i.game_date, i.team, p.player_id FROM injury_reports_pit i
        JOIN (SELECT player_id, lower(first_name||' '||last_name) fn FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '||trim(split_part(i.player,',',1)))
        WHERE i.status = 'Out' AND i.report_date = i.game_date
    """).fetchall()
    out = {}
    for gd, team, pid in rows:
        ab = name2ab.get(team)
        if ab:
            out.setdefault((str(gd)[:10], ab), set()).add(int(pid))
    return out


def report_dates(con):
    """The set of game_dates on which the buyable 5PM report actually exists."""
    return {str(r[0])[:10] for r in con.execute(
        "SELECT DISTINCT game_date FROM injury_reports_pit").fetchall()}


def load_matchup_def(season):
    """{defender_id: D36 on-ball rating} — VERBATIM from D72/oracle_tracking.
    Returns None when the season has no cache (=> the tier is inert)."""
    for f in glob.glob(str(REPO / "data/raw/nba_api/matchups/*.json")):
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
    return None


def fit_k(acc):
    """Walk-forward k, VERBATIM from D72/oracle_tracking.fit_k."""
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


class DailyTalent:
    """As-of-PRIOR-DAY DARKO and EPM, advanced by a pointer over date-sorted
    rows so the whole season costs one scan.  z-ensembled 50/50 and rescaled to
    the DARKO scale (so the strength sum keeps its units)."""

    def __init__(self, con, lo, hi):
        self.d_rows = con.execute(
            "SELECT date, player_id, dpm FROM darko_history "
            "WHERE date >= ? AND date <= ? ORDER BY date", [lo, hi]).fetchall()
        self.e_rows = con.execute(
            "SELECT asof_date, player_id, tot_epm FROM epm_history_daily "
            "WHERE asof_date >= ? AND asof_date <= ? ORDER BY asof_date",
            [lo, hi]).fetchall()
        self.di = self.ei = 0
        self.darko, self.epm = {}, {}
        self._cache_date = None
        self._cache = None
        # seed with everything strictly before the window (latest per player)
        for tbl, tgt, col in (("darko_history", self.darko, "dpm"),
                              ("epm_history_daily", self.epm, "tot_epm")):
            dcol = "date" if tbl == "darko_history" else "asof_date"
            for pid, v in con.execute(
                    f"SELECT player_id, {col} FROM (SELECT player_id, {col}, "
                    f"row_number() OVER (PARTITION BY player_id ORDER BY {dcol} "
                    f"DESC) rn FROM {tbl} WHERE {dcol} < ?) WHERE rn = 1",
                    [lo]).fetchall():
                if v is not None:
                    tgt[int(pid)] = float(v)

    def _advance(self, cut):
        while self.di < len(self.d_rows) and self.d_rows[self.di][0] <= cut:
            _, pid, v = self.d_rows[self.di]
            if v is not None:
                self.darko[int(pid)] = float(v)
            self.di += 1
        while self.ei < len(self.e_rows) and self.e_rows[self.ei][0] <= cut:
            _, pid, v = self.e_rows[self.ei]
            if v is not None:
                self.epm[int(pid)] = float(v)
            self.ei += 1

    def ensemble(self, game_date):
        """{player_id: talent on the DARKO scale} as of game_date - 1 day."""
        cut = game_date - dt.timedelta(days=1)
        if self._cache_date == cut:
            return self._cache
        self._advance(cut)
        d, e = self.darko, self.epm
        dv = np.fromiter(d.values(), float)
        if len(dv) < 10:
            self._cache_date, self._cache = cut, dict(d)
            return self._cache
        dm, ds = dv.mean(), dv.std() or 1.0
        common = [p for p in e if p in d]
        if len(common) < 50:
            self._cache_date, self._cache = cut, dict(d)
            return self._cache
        ev = np.array([e[p] for p in common], float)
        em, es = ev.mean(), ev.std() or 1.0
        out = {}
        for p, v in d.items():
            zd = (v - dm) / ds
            if p in e:
                z = 0.5 * zd + 0.5 * ((e[p] - em) / es)
            else:
                z = zd
            out[p] = dm + ds * z
        self._cache_date, self._cache = cut, out
        return out


# ------------------------------------------------------------- the season ---
def season_run(season, acc_prior, acc_same, diag):
    t_start = time.time()
    con = connect(read_only=True)
    rout = report_out_map(con)
    rdates = report_dates(con)
    inact = {}
    for g, p in con.execute(
            "SELECT game_id, player_id FROM game_inactives").fetchall():
        inact.setdefault(g, set()).add(int(p))

    pm = con.execute("""SELECT s.game_id, s.team_id, s.player_id,
        s.seconds/60.0 AS mins FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season FROM nba_games) g USING (game_id)
        WHERE g.season=? AND s.game_id LIKE '002%'""", [season]).fetchdf()
    played, minmap_t, minmap = {}, {}, {}
    for r in pm.itertuples():
        gid, tid, pid, mn = r.game_id, int(r.team_id), int(r.player_id), float(r.mins)
        minmap[(gid, pid)] = mn
        minmap_t.setdefault((gid, tid), {})[pid] = mn
        if mn > 0:
            played.setdefault((gid, tid), set()).add(pid)

    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, pts,
        game_date FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market "
        "WHERE season_end=?", [int(season[:4]) + 1]).fetchall()}

    mdef_prior = load_matchup_def(PREV[season])      # T4  (legitimate)
    mdef_same = load_matchup_def(season)             # C4  (clairvoyant)

    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)

    tgames, tdates = {}, {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tgames.setdefault(x.team_id, []).append((d, x.game_id))
        tdates.setdefault(x.team_id, set()).add(d)
    for t in tgames:
        tgames[t].sort()

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    def onball(mdef, tid, d, outs):
        """D72 verbatim: minutes-weighted mean on-ball rating over the team's
        last 10 games STRICTLY BEFORE d, tonight's OUT set excluded."""
        if mdef is None:
            return 0.0
        past = [g for (dd, g) in tgames.get(tid, []) if dd < d][-TRAIL_GAMES:]
        if not past:
            return 0.0
        w = {}
        for gid2 in past:
            for pid, m in minmap_t.get((gid2, tid), {}).items():
                w[pid] = w.get(pid, 0.0) + m
        w = {p: m for p, m in w.items() if p not in outs}
        den = sum(w.values())
        if den <= 0:
            return 0.0
        return sum(m * mdef.get(p, 0.0) for p, m in w.items()) / den

    dates = [x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
             for x in meta.itertuples()]
    talent = DailyTalent(con, min(dates) - dt.timedelta(days=400), max(dates))

    rng = np.random.default_rng(SEED)
    zcache: dict = {}

    def zdraw(gid, pid):
        k = (gid, pid)
        if k not in zcache:
            zcache[k] = float(rng.standard_normal())
        return zcache[k]

    model = comp = None
    last = None
    k_prior = fit_k(acc_prior)
    k_same = fit_k(acc_same)
    or_talent_cache: dict = {}
    rows = []
    n_skip_nomkt = n_skip_nodate = n_bridge = 0

    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        mu = recs[0].matchup
        host = mu.split("@")[-1].strip() if "@" in mu else mu.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            k_prior = fit_k(acc_prior)
            k_same = fit_k(acc_same)
            or_talent_cache = dict(con.execute(
                "SELECT player_id, avg(dpm) FROM darko_history "
                "WHERE date >= ? AND date <= ? GROUP BY 1",
                [gd - dt.timedelta(days=HALF_WINDOW),
                 gd + dt.timedelta(days=HALF_WINDOW)]).fetchall())
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            n_skip_nomkt += 1
            continue
        if str(gd)[:10] not in rdates:          # LADDER FRAME restriction
            n_skip_nodate += 1
            continue

        hid, aid = h.team_id, a.team_id
        rot = {t: {p for p, d0 in comp.players.items()
                   if d0["team_id"] == t and (gd - d0["last_played"]).days <= ROSTER_DAYS}
               for t in (hid, aid)}

        # ---- OUT sets, one per tier ---------------------------------------
        o0 = {hid: set(), aid: set()}
        o1 = {t: (rout.get((str(gd)[:10], ab), set()) & rot[t])
              for t, ab in ((hid, h.team_abbrev), (aid, a.team_abbrev))}
        ina = inact.get(gid, set())
        o2 = {t: (o1[t] | {p for p in ina if p in rot[t]}) for t in (hid, aid)}
        oc = {t: (rot[t] - played.get((gid, t), set())) for t in (hid, aid)}

        bb = dict(b2b_home=b2b(hid, gd), b2b_away=b2b(aid, gd))
        m0 = model.margin(hid, aid, o0[hid], o0[aid], gd, **bb)
        m1 = model.margin(hid, aid, o1[hid], o1[aid], gd, **bb)
        m2 = model.margin(hid, aid, o2[hid], o2[aid], gd, **bb)
        mC1 = model.margin(hid, aid, oc[hid], oc[aid], gd, **bb)

        bridge_game = rotation_empty(comp, hid, aid, gd)
        if bridge_game:
            n_bridge += 1

        # ---- the composition legs, recomputed by hand (swap precedent) -----
        def cm(outs, minutes_fn, tal):
            s = 0.0
            for t, sign in ((hid, 1), (aid, -1)):
                for pid in rot[t]:
                    if pid in outs[t]:
                        continue
                    s += sign * tal(pid) * minutes_fn(pid) / 48.0
            return s

        pit_tal = {p: v["talent"] for p, v in comp.players.items()}
        trail = {p: v["trail_min"] for p, v in comp.players.items()}
        f_pit = lambda p: pit_tal.get(p, 0.0)                       # noqa: E731
        f_trail = lambda p: trail.get(p, 0.0)                       # noqa: E731

        cm2 = cm(o2, f_trail, f_pit)          # the leg actually inside m2
        cmC1 = cm(oc, f_trail, f_pit)         # the leg actually inside mC1

        # ---- T3: purchased minutes projections (SIMULATED) ----------------
        m3 = {}
        cm3 = {}
        for mae in MAES:
            sg = SIG_OF_MAE[mae]

            def f_proj(p, _sg=sg):
                real = minmap.get((gid, p), 0.0)
                return float(min(48.0, max(0.0, real + _sg * zdraw(gid, p))))

            c = cm(o2, f_proj, f_pit)
            cm3[mae] = c
            m3[mae] = m2 if bridge_game else (m2 - 0.5 * cm2 + 0.5 * c)

        # ---- T4: tracking from PRIOR games --------------------------------
        obp_h = onball(mdef_prior, hid, gd, o2[hid])
        obp_a = onball(mdef_prior, aid, gd, o2[aid])
        dprior = obp_h - obp_a
        m4 = {mae: m3[mae] + k_prior * dprior for mae in MAES}

        # ---- T5: best talent as of the PRIOR DAY --------------------------
        ens = talent.ensemble(gd)
        f_ens = lambda p: ens.get(p, pit_tal.get(p, 0.0))           # noqa: E731
        m5 = {}
        for mae in MAES:
            sg = SIG_OF_MAE[mae]

            def f_proj(p, _sg=sg):
                real = minmap.get((gid, p), 0.0)
                return float(min(48.0, max(0.0, real + _sg * zdraw(gid, p))))

            c5 = cm(o2, f_proj, f_ens)
            m5[mae] = m4[mae] if bridge_game else (m4[mae] - 0.5 * cm3[mae]
                                                   + 0.5 * c5)

        # ---- PART C: the clairvoyant arms ---------------------------------
        f_real = lambda p: minmap.get((gid, p), 0.0)                # noqa: E731
        cmC2 = cm(oc, f_real, f_pit)
        mC2 = mC1 if bridge_game else (mC1 - 0.5 * cmC1 + 0.5 * cmC2)
        f_or = lambda p: or_talent_cache.get(p, pit_tal.get(p, 0.0))  # noqa: E731
        cmC3 = cm(oc, f_real, f_or)
        mC3 = mC2 if bridge_game else (mC2 - 0.5 * cmC2 + 0.5 * cmC3)
        obs_h = onball(mdef_same, hid, gd, o2[hid])
        obs_a = onball(mdef_same, aid, gd, o2[aid])
        dsame = obs_h - obs_a
        mC4 = m3[PRIMARY_MAE] + k_same * dsame

        y = int(h.wl == "W")
        resid3 = float(h.pts - a.pts) - float(m3[PRIMARY_MAE])
        if mdef_prior is not None:          # never accumulate inert zero rows
            acc_prior.append((dprior, resid3))
        acc_same.append((dsame, resid3))

        def sg_(m):
            return float(sigmoid(m / SCALE))

        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=y, p_mkt=float(pmv),
            p_T0=sg_(m0), p_T1=sg_(m1), p_T2=sg_(m2),
            p_T3_m3=sg_(m3[3.0]), p_T3=sg_(m3[4.0]), p_T3_m5=sg_(m3[5.0]),
            p_T4_m3=sg_(m4[3.0]), p_T4=sg_(m4[4.0]), p_T4_m5=sg_(m4[5.0]),
            p_T5_m3=sg_(m5[3.0]), p_T5=sg_(m5[4.0]), p_T5_m5=sg_(m5[5.0]),
            p_C1=sg_(mC1), p_C2=sg_(mC2), p_C3=sg_(mC3), p_C4=sg_(mC4),
            n_out_T1_h=len(o1[hid]), n_out_T1_a=len(o1[aid]),
            n_out_T2_h=len(o2[hid]), n_out_T2_a=len(o2[aid]),
            n_out_C1_h=len(oc[hid]), n_out_C1_a=len(oc[aid]),
            n_rot_h=len(rot[hid]), n_rot_a=len(rot[aid]),
            bridge=int(bridge_game), k_prior=round(k_prior, 6),
            k_same=round(k_same, 6), d_prior=round(dprior, 5),
            d_same=round(dsame, 5)))
    con.close()

    y = np.array([r["y"] for r in rows])
    d = dict(season=season, n=len(rows), secs=round(time.time() - t_start, 1),
             skip_nomkt=n_skip_nomkt, skip_nodate=n_skip_nodate,
             bridge=n_bridge, k_prior=round(k_prior, 6), k_same=round(k_same, 6),
             prior_cache=PREV[season] if mdef_prior else None,
             ll={c: round(float(log_loss(y, np.array([r[c] for r in rows]))), 5)
                 for c in ("p_mkt", "p_T0", "p_T1", "p_T2", "p_T3", "p_T4",
                           "p_T5", "p_C1", "p_C2", "p_C3", "p_C4")},
             mean_outs={c: round(float(np.mean([r[f"n_out_{c}_h"] + r[f"n_out_{c}_a"]
                                                for r in rows])), 3)
                        for c in ("T1", "T2", "C1")})
    diag.append(d)
    print(json.dumps(d), flush=True)
    return rows


def main():
    diag = []
    acc_prior, acc_same = [], []
    allrows = []
    for s in SEASONS:
        allrows += season_run(s, acc_prior, acc_same, diag)
    import csv
    cols = list(allrows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    json.dump(diag, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_CSV} ({len(allrows)} rows)")


if __name__ == "__main__":
    main()
