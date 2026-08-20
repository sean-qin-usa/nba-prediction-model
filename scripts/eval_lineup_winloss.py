"""Sean's ask: win/loss purely from PLAYER LINEUPS (oracle: who actually played,
minutes-weighted DARKO talent). If lineup-knowledge is the missing edge, this
should beat the team-rating model at least on roster-changed games.
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
    pm=con.execute("SELECT game_id, team_id, player_id, seconds/60.0 m FROM player_game_stats WHERE game_id LIKE '002%'").fetchdf()
    meta=con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' AND wl IS NOT NULL""").fetchdf()
    mkt={(str(r[0])[:10],r[1],r[2]):r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=2026").fetchall()}
    con.close()
    # lineup strength = sum over players who played: darko * min/48
    pm["c"]=pm.player_id.map(lambda p: darko.get(p,0.0))*pm.m/48.0
    strength=pm.groupby(["game_id","team_id"])["c"].sum().to_dict()
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
        pl.append(float(sig((sl+3.0)/SCALE)))                       # lineup-only oracle
        pr.append(float(sig(tr.pred_margin(h.team_id,a.team_id)/SCALE)))  # team ratings
        pmk.append(pmv)
    y=np.array(y)
    print(f"test games {len(y)}")
    print(f"  LINEUP-ORACLE (who played, DARKO-weighted): {log_loss(y,pl):.4f}")
    print(f"  team ratings (no lineup info)             : {log_loss(y,pr):.4f}")
    print(f"  market                                    : {log_loss(y,np.array(pmk)):.4f}")
    # blend lineup into ratings?
    for w in (0.2,0.4):
        pb=np.clip(w*np.array(pl)+(1-w)*np.array(pr),1e-6,1-1e-6)
        print(f"  blend lineup w={w}: {log_loss(y,pb):.4f}")

if __name__=="__main__": main()
