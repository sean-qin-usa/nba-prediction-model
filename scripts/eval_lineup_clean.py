"""Leak-free version of the lineup oracle: availability only (did the player
appear at all = would've been on the injury report as available), weighted by
TRAILING average minutes — no in-game info. This is what an October injury feed
actually gives us pre-game.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.team_ratings import TeamRatings, game_rows

SCALE=7.2
sig=lambda x:1/(1+np.exp(-np.asarray(x)))

def main():
    con=connect(read_only=True)
    games=con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY game_date""").fetchdf()
    cut=games.game_date.quantile(0.6)
    tr=TeamRatings(ridge=25).fit(game_rows(con,before=cut))
    darko={p:o+d for p,o,d in con.execute("SELECT nba_player_id,o_dpm,d_dpm FROM darko_dpm WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall()}
    pm=con.execute("""SELECT s.game_id, s.team_id, s.player_id, s.seconds/60.0 m, g.game_date
        FROM player_game_stats s JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    meta=con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' AND wl IS NOT NULL""").fetchdf()
    mkt={(str(r[0])[:10],r[1],r[2]):r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=2026").fetchall()}
    con.close()
    # trailing avg minutes per player (shifted, rolling 10)
    pm=pm.sort_values(["player_id","game_date"])
    pm["avg10"]=pm.groupby("player_id")["m"].transform(lambda s: s.shift(1).rolling(10,min_periods=3).mean())
    # availability: appeared at all this game (>0 min) -> was available pre-game
    pm["c"]=pm.player_id.map(lambda p: darko.get(p,0.0))*pm["avg10"].fillna(0)/48.0
    strength=pm[pm.m>0].groupby(["game_id","team_id"])["c"].sum().to_dict()
    by={}
    for x in meta.itertuples(): by.setdefault(x.game_id,[]).append(x)
    y,pl,pr,pmk=[],[],[],[]
    for gid,recs in by.items():
        if len(recs)!=2 or recs[0].game_date<=cut: continue
        m=recs[0].matchup; host=m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h=next((x for x in recs if x.team_abbrev==host),None); a=next((x for x in recs if x.team_abbrev!=host),None)
        if not h or not a: continue
        pmv=mkt.get((str(h.game_date)[:10],h.team_abbrev,a.team_abbrev))
        if pmv is None: continue
        sl=strength.get((gid,h.team_id),0)-strength.get((gid,a.team_id),0)
        y.append(int(h.wl=="W"))
        pl.append(float(sig((sl+3.0)/SCALE)))
        pr.append(float(sig(tr.pred_margin(h.team_id,a.team_id)/SCALE)))
        pmk.append(pmv)
    y=np.array(y)
    print(f"test games {len(y)}")
    print(f"  AVAILABILITY-ONLY lineup (trailing-min weights): {log_loss(y,pl):.4f}")
    print(f"  team ratings                                   : {log_loss(y,pr):.4f}")
    print(f"  market                                         : {log_loss(y,np.array(pmk)):.4f}")
    for w in (0.3,0.5,0.7):
        pb=np.clip(w*np.array(pl)+(1-w)*np.array(pr),1e-6,1-1e-6)
        print(f"  blend availability w={w}: {log_loss(y,pb):.4f}")

if __name__=="__main__": main()
