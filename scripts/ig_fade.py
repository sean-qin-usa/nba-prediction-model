"""IG probe A2: does fading the carry beat the 200-row hard stop?

Emulates prod_by_season.py EXACTLY for games up to Dec 15 (weekly refit, oracle
outs, b2b flags, market filter) and validates the emulation against
capstone_pergame_carry2.csv p_us. Then one pre-registered variant (no sweep):
carry pseudo-row weights multiplied by max(0, 1 - n_cur/400) at every fit
(smooth fade through the shipped hard-stop point) instead of the <200 cliff.
Scores both on identical games. Read-only.
"""
import sys, warnings, datetime as _dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect
from nbapred.model.production import (SCALE, sigmoid, fit_schedule_layer,
                                      continuity_map, CARRY_W0, CARRY_CONT_DEFAULT,
                                      _prev_season)
from nbapred.model.four_factors import factor_game_rows, FACTORS
from nbapred.model.team_ratings import TeamRatings
from nbapred.model.composition import CompositionModel

def fit_ff(rows, weights):
    fms = {f: TeamRatings(ridge=25.0, team_home_ridge=None).fit(
        [(x["tid"], x["oid"], x["home"], x[f] * 100) for x in rows], weights=weights)
        for f in FACTORS}
    X = np.array([[fms[f].pred_ortg(x["tid"], x["oid"], x["home"]) for f in FACTORS]
                  for x in rows])
    y = np.array([x["ortg"] for x in rows])
    A = np.c_[X, np.ones(len(X))]
    if weights is not None:
        sw = np.sqrt(np.asarray(weights))
        W = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    else:
        W = np.linalg.lstsq(A, y, rcond=None)[0]
    def margin_neutral(h, a):
        xh = np.array([fms[f].pred_ortg(h, a, False) for f in FACTORS])
        xa = np.array([fms[f].pred_ortg(a, h, False) for f in FACTORS])
        return float(xh @ W[:4] - xa @ W[:4])
    return margin_neutral

def ll(y, p):
    p = np.clip(np.asarray(p), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def season_run(con, season, until_md=(12, 15)):
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
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
    y0 = int(season[:4])
    stop = _dt.date(y0, until_md[0], until_md[1])
    prev_rows = factor_game_rows(con, _prev_season(season), before=None)
    out_rows = []
    last = None
    state = {}
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
        if gd > stop:
            break
        if last is None or (gd - last).days >= 7:
            cur = factor_game_rows(con, season, before=gd)
            n_cur = len(cur)
            cont = continuity_map(con, season, before=gd)
            cw = [CARRY_W0 * cont.get(x["tid"], CARRY_CONT_DEFAULT) for x in prev_rows]
            # shipped: hard stop
            if n_cur < 200:
                ship = fit_ff(list(prev_rows) + cur, np.array(cw + [1.0] * n_cur))
            else:
                ship = fit_ff(cur, None)
            # variant: linear fade of carry to zero at n_cur=400
            fade = max(0.0, 1 - n_cur / 400.0)
            if fade > 0:
                var = fit_ff(list(prev_rows) + cur,
                             np.array([w * fade for w in cw] + [1.0] * n_cur))
            else:
                var = fit_ff(cur, None)
            sched = fit_schedule_layer(con, gd)
            comp = CompositionModel(con, before=gd)
            state = dict(ship=ship, var=var, sched=sched, comp=comp)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        comp = state["comp"]
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12 and p not in pl}
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd, home_edge=0.0)
        he, bh, ba, _, _ = state["sched"]
        sc = he + (bh if b2b(h.team_id, gd) else 0.0) + (ba if b2b(a.team_id, gd) else 0.0)
        m_ship = 0.5 * state["ship"](h.team_id, a.team_id) + 0.5 * cm + sc
        m_var = 0.5 * state["var"](h.team_id, a.team_id) + 0.5 * cm + sc
        out_rows.append((season, gid, str(gd)[:10], int(h.wl == "W"),
                         float(sigmoid(m_ship / SCALE)), float(sigmoid(m_var / SCALE))))
    return out_rows

def main():
    con = connect(read_only=True)
    cap = pd.read_csv("/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_carry2.csv")
    cap["gid"] = cap.game_id.apply(lambda g: f"00{g}" if len(str(g)) == 8 else str(g))
    capm = {r.gid: r.p_us for r in cap.itertuples()}
    allrows = []
    for season in ("2023-24", "2024-25", "2025-26"):
        rows = season_run(con, season)
        df = pd.DataFrame(rows, columns=["season", "gid", "date", "y", "p_ship", "p_var"])
        df["p_csv"] = df.gid.map(capm)
        err = (df.p_ship - df.p_csv).abs()
        print(f"{season}: n={len(df)} emulation max|p_ship-p_csv|={err.max():.6f} "
              f"mean={err.mean():.6f}")
        allrows.append(df)
    df = pd.concat(allrows)
    d = ll(df.y, df.p_ship) - ll(df.y, df.p_var)   # >0 => fade better
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\nFADE vs HARD STOP (games through Dec 15, n={len(df)}):")
    print(f" pooled delta {d.mean():+.5f}/game CI95({lo:+.5f},{hi:+.5f})  (>0 = fade better)")
    for s, sub in df.assign(d=d).groupby("season"):
        print(f"  {s}: {sub.d.mean():+.5f} (n={len(sub)})")
    # zoom: handover fortnight (Nov 1-20)
    z = df[(df.date >= df.season.str[:4] + "-11-01") & (df.date <= df.season.str[:4] + "-11-20")]
    dz = ll(z.y, z.p_ship) - ll(z.y, z.p_var)
    print(f" handover window Nov 1-20: n={len(z)} delta {dz.mean():+.5f}")
    con.close()

if __name__ == "__main__":
    main()
