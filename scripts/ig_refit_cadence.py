"""IG probe: weekly-refit (shipped backtest cadence) vs daily-refit (live cadence)
paired per-game log-loss on one season. Read-only. Mirrors prod_by_season's
default (oracle-outs) mode exactly; only the refit trigger differs."""
import sys, warnings, datetime as _dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.model.production import fit_production
from nbapred.model.composition import CompositionModel

SEASON = sys.argv[1] if len(sys.argv) > 1 else "2024-25"

con = connect(read_only=True)
pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins FROM player_game_stats
    WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
    WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL ORDER BY game_date""", [SEASON]).fetchdf()
mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
    "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
    [int(SEASON[:4]) + 1]).fetchall()}
by = {}; order = []
for x in meta.itertuples():
    if x.game_id not in by: order.append(x.game_id)
    by.setdefault(x.game_id, []).append(x)
tdates = {}
for x in meta.itertuples():
    d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
    tdates.setdefault(x.team_id, set()).add(d)
def b2b(tid, d): return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

y, pw, pd_, pmk = [], [], [], []
mw = cw = md = cd = None
last_w = last_d = None
for gid in order:
    recs = by[gid]
    if len(recs) != 2: continue
    m = recs[0].matchup
    host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
    h = next((x for x in recs if x.team_abbrev == host), None)
    a = next((x for x in recs if x.team_abbrev != host), None)
    if not h or not a: continue
    gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
    if last_w is None or (gd - last_w).days >= 7:
        mw = fit_production(con, SEASON, before=gd, w_comp=0.7)
        cw = CompositionModel(con, before=gd); last_w = gd
    if last_d is None or gd != last_d:
        md = fit_production(con, SEASON, before=gd, w_comp=0.7)
        cd = CompositionModel(con, before=gd); last_d = gd
    pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
    if pmv is None: continue
    def outs_for(comp):
        o = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            o[t] = {p for p, d0 in comp.players.items()
                    if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12 and p not in pl}
        return o
    ow, od = outs_for(cw), outs_for(cd)
    y.append(int(h.wl == "W"))
    pw.append(mw.p_home(h.team_id, a.team_id, ow[h.team_id], ow[a.team_id], gd,
                        b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd)))
    pd_.append(md.p_home(h.team_id, a.team_id, od[h.team_id], od[a.team_id], gd,
                         b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd)))
    pmk.append(pmv)
con.close()
y = np.array(y); pw = np.clip(np.array(pw), 1e-9, 1-1e-9); pd_ = np.clip(np.array(pd_), 1e-9, 1-1e-9)
lw = -(y*np.log(pw) + (1-y)*np.log(1-pw))
ld = -(y*np.log(pd_) + (1-y)*np.log(1-pd_))
d = lw - ld   # >0: daily better
rng = np.random.default_rng(0)
bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
print(f"{SEASON} n={len(y)} weekly={lw.mean():.4f} daily={ld.mean():.4f} "
      f"delta(weekly-daily)={d.mean():+.5f} CI95=({np.quantile(bs,.025):+.5f},{np.quantile(bs,.975):+.5f})")
