"""EXPERIMENT garbage: garbage-time-filtered four-factor rate estimation.

Standalone experiment script — does NOT touch nbapred/ model files or
scripts/prod_by_season.py (logic copied where needed, per experiment rules).

Hypothesis (top consensus pick from two external reviews): garbage-time play
(blowout 4th quarters: benches, no defense, fouling games) contaminates the
per-100-possession four-factor rates the FF ridge consumes; excluding it should
sharpen team-strength estimates.

Garbage-time flag (Cleaning-the-Glass v1; starters-on-floor refinement ignored):
  4th quarter AND |margin| >= 25 with 12-9 min left,
                  |margin| >= 20 with 9-6 min left,
                  |margin| >= 10 with under 6 min left.
Margin is read from the action's running score (post-action).

Aggregate construction: filtered team-game stat = player-box-score sum
(player_game_stats, identical to production factor_game_rows) MINUS the
player-attributed garbage-time counts parsed from cached playbyplayv3 JSON.
Subtracting deltas (rather than rebuilding whole games from PBP) keeps the
non-garbage portion bit-identical to production inputs, so the treatment is
isolated to the garbage-time exclusion. Player-attributed only (personId>0):
team rebounds/turnovers are excluded from player box sums, so they must be
excluded from the subtraction too.

Subcommands:
  build     parse PBP -> data/ff_rows_nogarbage.parquet (cached, reruns cheap)
  validate  parser sanity: full-game PBP counts vs player_game_stats box sums
  run       walk-forward capstone, control (production FF) + treatment
            (garbage-filtered FF) sharing comp/sched/ratings per refit
            -> data/exp_garbage_pergame.csv
  gate      paired bootstrap (2000, seed 42) vs data/capstone_pergame_sched.csv
"""
import re
import sys
import time
import warnings
import datetime as _dt
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import orjson

from nbapred.db import connect
from nbapred.features.cache_index import game_index
from nbapred.model.four_factors import FACTORS, FourFactors, factor_game_rows
from nbapred.model.team_ratings import TeamRatings, game_rows

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "ff_rows_nogarbage.parquet"
RUN_CSV = ROOT / "data" / "exp_garbage_pergame.csv"
BASELINE_CSV = ROOT / "data" / "capstone_pergame_sched.csv"
SEASONS = ("2023-24", "2024-25", "2025-26")
STATS = ["fgm", "fga", "thrm", "thra", "tov", "oreb", "dreb", "fta", "pts"]

REB_RE = re.compile(r"REBOUND \(Off:(\d+) Def:(\d+)\)")
CLK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")


# ---------------------------------------------------------------- PBP parsing

def clock_seconds(c):
    m = CLK_RE.match(c or "")
    return 60 * int(m.group(1)) + float(m.group(2)) if m else None


def garbage_thr(tl):
    if tl > 540:          # 12-9 min left in the 4th
        return 25
    if tl > 360:          # 9-6 min left
        return 20
    return 10             # under 6 min


def parse_game(path):
    """One pass over a playbyplayv3 file.
    Returns (full, garb): {team_id: {stat: n}} for the whole game and for
    garbage-time actions only. Player-attributed events only (mirrors
    player_game_stats box sums, which exclude team rebounds/turnovers and
    2025-26 team-charged heaves)."""
    raw = orjson.loads(open(path, "rb").read())
    acts = raw["response"]["game"]["actions"]
    full, garb = {}, {}

    def bump(d, tid, stat, k):
        d.setdefault(tid, dict.fromkeys(STATS, 0))[stat] += k

    reb_prev = {}          # personId -> (cum off, cum def) from descriptions
    last_miss_team = None  # fallback rebound classifier
    sh = sa = 0
    for a in acts:
        s1 = a.get("scoreHome") or ""
        s2 = a.get("scoreAway") or ""
        if s1 != "":
            sh = int(s1)
        if s2 != "":
            sa = int(s2)
        at = a.get("actionType")
        tid = a.get("teamId") or 0
        pid = a.get("personId") or 0
        desc = a.get("description") or ""
        tl = clock_seconds(a.get("clock"))
        garbage = (a.get("period") == 4 and tl is not None
                   and abs(sh - sa) >= garbage_thr(tl))
        events = []
        if at == "Made Shot":
            sv = a.get("shotValue") or 2
            events += [("fgm", 1), ("fga", 1), ("pts", sv)]
            if sv == 3:
                events += [("thrm", 1), ("thra", 1)]
        elif at == "Missed Shot":
            events.append(("fga", 1))
            if (a.get("shotValue") or 0) == 3:
                events.append(("thra", 1))
            last_miss_team = tid
        elif at == "Free Throw":
            events.append(("fta", 1))
            if desc.startswith("MISS"):
                last_miss_team = tid
            else:
                events.append(("pts", 1))
        elif at == "Turnover":
            if pid > 0:  # personId==0 -> team turnover, not in player box
                events.append(("tov", 1))
        elif at == "Rebound":
            if pid > 0:  # team rebounds excluded (not in player box sums)
                kind = None
                m = REB_RE.search(desc)
                if m:
                    oc, dc = int(m.group(1)), int(m.group(2))
                    po, pd = reb_prev.get(pid, (0, 0))
                    if oc == po + 1 and dc == pd:
                        kind = "oreb"
                    elif dc == pd + 1 and oc == po:
                        kind = "dreb"
                    reb_prev[pid] = (oc, dc)
                if kind is None:  # counter desync / no counters in desc
                    kind = "oreb" if tid == last_miss_team else "dreb"
                events.append((kind, 1))
        elif at == "Heave":
            # 2025-26: end-of-period heave charged to the TEAM, absent from
            # player box sums -> excluded here too. Only track for rebounds.
            last_miss_team = tid
        if tid:
            for stat, k in events:
                bump(full, tid, stat, k)
                if garbage:
                    bump(garb, tid, stat, k)
    return full, garb


BOX_SQL = """SELECT g.season, s.game_id, s.team_id, g.game_date, g.matchup, g.team_abbrev,
    sum(s.fgm) fgm, sum(s.fga) fga, sum(s.thrm) thrm, sum(s.thra) thra, sum(s.tov) tov,
    sum(s.oreb) oreb, sum(s.dreb) dreb, sum(s.fta) fta, sum(s.pts) pts
    FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
    WHERE g.season IN ('2023-24','2024-25','2025-26') AND s.game_id LIKE '002%'
    GROUP BY 1,2,3,4,5,6"""


def cmd_build():
    t0 = time.time()
    gi = game_index("playbyplayv3")
    con = connect(read_only=True)
    box = con.execute(BOX_SQL).fetchdf()
    con.close()
    gids = sorted(box.game_id.unique())
    missing = [g for g in gids if g not in gi]
    print(f"team-games {len(box)}, games {len(gids)}, PBP missing {len(missing)}")

    from concurrent.futures import ThreadPoolExecutor
    gstats = {}
    def work(gid):
        _, garb = parse_game(gi[gid])
        return gid, garb
    with ThreadPoolExecutor(max_workers=8) as ex:
        for gid, garb in ex.map(work, [g for g in gids if g in gi]):
            for tid, d in garb.items():
                gstats[(gid, int(tid))] = d
    for st in STATS:
        box["g_" + st] = [gstats.get((r.game_id, int(r.team_id)), {}).get(st, 0)
                          for r in box.itertuples()]
        box["ng_" + st] = (box[st] - box["g_" + st]).clip(lower=0)
    mem = duckdb.connect()
    mem.register("box", box)
    mem.execute(f"COPY (SELECT * FROM box) TO '{PARQUET}' (FORMAT PARQUET)")
    # diagnostics
    gposs = box.g_fga + 0.44 * box.g_fta - box.g_oreb + box.g_tov
    poss = box.fga + 0.44 * box.fta - box.oreb + box.tov
    print(f"wrote {PARQUET} in {time.time()-t0:.0f}s")
    for s in SEASONS:
        m = box.season == s
        touched = (gposs[m] > 0).mean()
        print(f"  {s}: {touched:.1%} of team-games have garbage poss removed; "
              f"mean removed (when >0) {gposs[m][gposs[m]>0].mean():.1f} poss "
              f"({(gposs[m].sum()/poss[m].sum()):.2%} of all possessions)")


def cmd_validate(n_sample=150, seed=7):
    """Full-game PBP counts vs player box sums — parser correctness check."""
    gi = game_index("playbyplayv3")
    con = connect(read_only=True)
    box = con.execute(BOX_SQL).fetchdf()
    con.close()
    rng = np.random.default_rng(seed)
    gids = sorted(set(box.game_id) & set(gi))
    sample = list(rng.choice(gids, size=min(n_sample, len(gids)), replace=False))
    bx = {(r.game_id, int(r.team_id)): r for r in box.itertuples()}
    diffs = {st: [] for st in STATS}
    worst = []
    for gid in sample:
        full, _ = parse_game(gi[gid])
        for tid, d in full.items():
            b = bx.get((gid, int(tid)))
            if b is None:
                continue
            for st in STATS:
                dv = d[st] - int(getattr(b, st))
                diffs[st].append(dv)
                if abs(dv) > 2:
                    worst.append((gid, tid, st, d[st], int(getattr(b, st))))
    print(f"validate on {len(sample)} games ({len(diffs['fga'])} team-games): PBP - box")
    for st in STATS:
        a = np.array(diffs[st])
        print(f"  {st:5s} exact {np.mean(a==0):6.1%}  mean|d| {np.abs(a).mean():.3f}  max|d| {np.abs(a).max()}")
    for w in worst[:10]:
        print("  worst:", w)


# ------------------------------------------------------- model (copied logic)

def fit_ff_rows(rows, ridge=25.0):
    """FourFactors.fit body (no recency, no luck adjust — production config),
    taking pre-built rows so the treatment can swap in filtered aggregates."""
    ff = FourFactors(ridge=ridge)
    if len(rows) < 200:
        return ff
    ff.fms = {f: TeamRatings(ridge=ridge, team_home_ridge=None).fit(
        [(x["tid"], x["oid"], x["home"], x[f] * 100) for x in rows])
        for f in FACTORS}
    X = np.array([[ff.fms[f].pred_ortg(x["tid"], x["oid"], x["home"])
                   for f in FACTORS] for x in rows])
    y = np.array([x["ortg"] for x in rows])
    A = np.c_[X, np.ones(len(X))]
    ff.W = np.linalg.lstsq(A, y, rcond=None)[0]
    return ff


def load_ng_rows():
    """factor_game_rows equivalent from the filtered parquet, per season."""
    mem = duckdb.connect()
    df = mem.execute("SELECT * FROM read_parquet(?)", [str(PARQUET)]).fetchdf()
    out = {s: [] for s in SEASONS}
    for s in SEASONS:
        sub = df[df.season == s]
        by = {}
        for r in sub.itertuples():
            by.setdefault(r.game_id, []).append(r)
        for gid, recs in by.items():
            if len(recs) != 2:
                continue
            m = recs[0].matchup
            host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
            h = next((x for x in recs if x.team_abbrev == host), None)
            a = next((x for x in recs if x.team_abbrev != host), None)
            if not h or not a:
                continue
            for t, o, hm in ((h, a, True), (a, h, False)):
                fgm, fga, thrm, thra, tov, oreb, dreb, fta, pts = (
                    float(getattr(t, "ng_" + x)) for x in STATS)
                o_dreb = float(o.ng_dreb)
                poss = fga + 0.44 * fta - oreb + tov
                if poss < 50 or fga <= 0 or (oreb + o_dreb) <= 0:
                    continue
                out[s].append(dict(
                    tid=int(t.team_id), oid=int(o.team_id), home=hm,
                    date=(t.game_date.date() if hasattr(t.game_date, "date")
                          else t.game_date),
                    efg=(fgm + 0.5 * thrm) / fga, tovr=tov / poss,
                    orbr=oreb / (oreb + o_dreb), ftr=fta / fga,
                    thrm=thrm, thra=thra, pts=pts, fga=fga, poss=poss,
                    ortg=100 * pts / poss))
        out[s].sort(key=lambda x: x["date"])
    return out


def fit_production_pair(con, season, before, ng_rows_all, w_comp=0.7):
    """fit_production (nbapred/model/production.py) copied, building BOTH the
    control FF (production factor_game_rows) and the treatment FF (garbage-
    filtered rows) while sharing comp/sched/ratings/prior — exact pairing."""
    from nbapred.model.composition import CompositionModel
    from nbapred.model.production import (SCALE, fit_schedule_layer,
                                          last_season_prior, sigmoid)
    comp = CompositionModel(con, before=before)
    ff_ctl = fit_ff_rows(factor_game_rows(con, season, before))
    ff_ng = fit_ff_rows([x for x in ng_rows_all[season] if x["date"] < before])
    he, b_hb2b, b_ab2b, b_hdead, b_adead = fit_schedule_layer(con, before)
    tr = TeamRatings(ridge=25.0).fit(game_rows(con, before=before, season=season))
    prior = last_season_prior(con, season)
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    games_played = dict(con.execute("""
        SELECT team_id, count(*) FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL GROUP BY 1""", [season]).fetchall())

    class Predictor:
        def __init__(self, ff):
            self.ff = ff

        def ratings_margin(self, home_id, away_id):
            m = tr.pred_margin(home_id, away_id)
            gh = games_played.get(home_id, 0); ga = games_played.get(away_id, 0)
            wh = max(0.0, 1 - gh / 20.0); wa = max(0.0, 1 - ga / 20.0)
            ph = prior.get(id2ab.get(home_id, ""), 0.0)
            pa = prior.get(id2ab.get(away_id, ""), 0.0)
            return m + wh * ph - wa * pa

        def margin(self, home_id, away_id, out_home=None, out_away=None,
                   game_date=None, b2b_home=False, b2b_away=False):
            sched = (he + (b_hb2b if b2b_home else 0.0)
                     + (b_ab2b if b2b_away else 0.0))
            cm = comp.margin(home_id, away_id, out_home, out_away, game_date,
                             home_edge=0.0)
            if self.ff.ready:
                fm = self.ff.margin_neutral(home_id, away_id)
                return 0.5 * fm + 0.5 * cm + sched
            rm = self.ratings_margin(home_id, away_id) - tr.home
            return w_comp * cm + (1 - w_comp) * rm + sched

        def p_home(self, home_id, away_id, out_home=None, out_away=None,
                   game_date=None, b2b_home=False, b2b_away=False):
            return float(sigmoid(self.margin(home_id, away_id, out_home,
                                             out_away, game_date, b2b_home,
                                             b2b_away) / SCALE))

    return comp, Predictor(ff_ctl), Predictor(ff_ng)


def season_run(season, ng_rows_all):
    """scripts/prod_by_season.py season_run copied — default (oracle-outs)
    path only; both control and treatment predictions per game."""
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL ORDER BY game_date""",
        [season]).fetchdf()
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
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())
    rows = []
    comp = pc = pn = None; last = None
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
            comp, pc, pn = fit_production_pair(con, season, gd, ng_rows_all)
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
        args = (outs[h.team_id], outs[a.team_id], gd,
                b2b(h.team_id, gd), b2b(a.team_id, gd))
        rows.append((season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                     int(h.wl == "W"),
                     pc.p_home(h.team_id, a.team_id, *args),
                     pn.p_home(h.team_id, a.team_id, *args),
                     float(pmv)))
    con.close()
    return rows


def cmd_run():
    ng_rows_all = load_ng_rows()
    for s in SEASONS:
        print(f"  ng rows {s}: {len(ng_rows_all[s])}")
    import csv
    allrows = []
    for s in SEASONS:
        t0 = time.time()
        rows = season_run(s, ng_rows_all)
        allrows += rows
        y = np.array([r[5] for r in rows], float)
        pc = np.array([r[6] for r in rows]); pn = np.array([r[7] for r in rows])
        print(f"{s}: n={len(rows)} ctl {_ll(y, pc):.4f} ng {_ll(y, pn):.4f} "
              f"({time.time()-t0:.0f}s)")
    with open(RUN_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "game_date", "home", "away", "y",
                    "p_ctl", "p_ng", "p_mkt"])
        w.writerows(allrows)
    print("wrote", RUN_CSV)


# ------------------------------------------------------------------- the gate

def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_ci(d, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cmd_gate():
    import pandas as pd
    base = pd.read_csv(BASELINE_CSV)
    ours = pd.read_csv(RUN_CSV, dtype={"game_id": str})
    base["game_id"] = base["game_id"].astype(str).str.zfill(10)
    ours["game_id"] = ours["game_id"].astype(str).str.zfill(10)
    m = base.merge(ours, on=["season", "game_id"], suffixes=("", "_x"))
    assert len(m) == len(base) == len(ours), (len(base), len(ours), len(m))
    assert (m.y == m.y_x).all()
    rep = np.abs(m.p_ctl - m.p_us).max()
    print(f"n={len(m)}  replication check max|p_ctl - baseline p_us| = {rep:.2e}")
    res = {}
    for label, sub in [("POOLED", m)] + [(s, m[m.season == s]) for s in SEASONS]:
        y = sub.y.values
        d_base = _ll_vec(y, sub.p_us.values) - _ll_vec(y, sub.p_ng.values)
        d_ctl = _ll_vec(y, sub.p_ctl.values) - _ll_vec(y, sub.p_ng.values)
        lo, hi = boot_ci(d_base)
        lo2, hi2 = boot_ci(d_ctl)
        res[label] = dict(
            n=len(sub), ll_base=_ll(y, sub.p_us.values),
            ll_ctl=_ll(y, sub.p_ctl.values), ll_ng=_ll(y, sub.p_ng.values),
            ll_mkt=_ll(y, sub.p_mkt.values),
            delta=float(d_base.mean()), lo=lo, hi=hi,
            delta_ctl=float(d_ctl.mean()), lo2=lo2, hi2=hi2)
        r = res[label]
        print(f"{label:8s} n={r['n']:5d} base {r['ll_base']:.4f} ctl {r['ll_ctl']:.4f} "
              f"ng {r['ll_ng']:.4f} mkt {r['ll_mkt']:.4f} | "
              f"d(base-ng) {r['delta']:+.5f} CI({lo:+.5f},{hi:+.5f}) | "
              f"d(ctl-ng) {r['delta_ctl']:+.5f} CI({lo2:+.5f},{hi2:+.5f})")
    p = res["POOLED"]
    verdict = "PASS" if p["lo"] > 0 else ("FAIL" if p["hi"] < 0 else "NS")
    print(f"VERDICT vs shipped baseline (positive = improvement): {verdict}")
    return res, verdict


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gate"
    {"build": cmd_build, "validate": cmd_validate,
     "run": cmd_run, "gate": cmd_gate}[cmd]()
