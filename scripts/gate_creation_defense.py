"""Creation-split x on-ball-defense interaction gate (D36 follow-up):
pull-up-heavy shooters facing elite ON-BALL defenses should lose efficiency;
C&S shooters shouldn't (their shots come off ball movement, not vs stopper).
Test: player-game 3PT% vs opponent's team-level ON-BALL defense (minute-weighted
matchup-def rating of the opposing roster), split by shooter's pull-up share.
Empirical interaction first; gate on prop CRPS only if the interaction is real.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, glob, orjson
from nbapred.db import connect
from nbapred.ingest.nba_stats import _frames

def load_range(range_name, season):
    out={}
    for f in glob.glob("data/raw/nba_api/ptshot/*.json"):
        d=orjson.loads(open(f,"rb").read())
        if d["params"].get("general_range_nullable")==range_name and d["params"].get("season")==season:
            df=list(_frames(d["response"]).values())[0]
            for r in df.itertuples():
                out[int(r.PLAYER_ID)]=dict(freq=float(r.FGA_FREQUENCY or 0))
    return out

def load_matchup_def(season):
    for f in glob.glob("data/raw/nba_api/matchups/*.json"):
        d=orjson.loads(open(f,"rb").read())
        if d["params"].get("season")==season:
            df=list(_frames(d["response"]).values())[0]
            df=df[df.PARTIAL_POSS>=10]
            off=df.groupby("OFF_PLAYER_ID").agg(p=("PLAYER_PTS","sum"),q=("PARTIAL_POSS","sum"))
            rate=(off.p/off.q).to_dict()
            df["exp"]=df.OFF_PLAYER_ID.map(rate)*df.PARTIAL_POSS
            g=df.groupby("DEF_PLAYER_ID").agg(a=("PLAYER_PTS","sum"),e=("exp","sum"),q=("PARTIAL_POSS","sum"))
            g["r"]=100*(g.e-g.a)/(g.q+800.0)
            return g["r"].to_dict()
    return {}

def main():
    con=connect(read_only=True)
    pu=load_range("Pullups","2024-25")
    mdef=load_matchup_def("2024-25")
    # per player-game 3PT in 2025-26, opponent on-ball defense = minute-weighted mean
    pg=con.execute("""SELECT s.player_id, s.team_id, s.game_id, g.game_date, g.matchup,
        g.team_abbrev, s.thrm, s.thra FROM player_game_stats s
        JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season='2025-26' AND s.game_id LIKE '002%' AND s.thra>=3""").fetchdf()
    mins=con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 m
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    con.close()
    # opponent team on-ball rating per game: minute-weighted mdef of opp players who played
    mins["md"]=mins.player_id.map(lambda p: mdef.get(p, np.nan))
    tm=mins.dropna(subset=["md"]).groupby(["game_id","team_id"]).apply(
        lambda x: np.average(x.md, weights=x.m)).to_dict()
    by={}
    for r in pg.itertuples(): by.setdefault(r.game_id,set()).add(r.team_id)
    rows=[]
    for r in pg.itertuples():
        teams=by.get(r.game_id,set())
        opp=[t for t in teams if t!=r.team_id]
        if not opp: continue
        od=tm.get((r.game_id,opp[0]))
        pshare=pu.get(int(r.player_id),{}).get("freq")
        if od is None or pshare is None: continue
        rows.append((pshare, od, r.thrm, r.thra))
    a=np.array(rows)
    print(f"player-games {len(a)}")
    hi_pu=a[:,0]>np.median(a[:,0])       # pull-up heavy shooters
    tough=a[:,1]>np.median(a[:,1])       # strong on-ball defense
    for lab,m1 in (("PULL-UP heavy",hi_pu),("C&S heavy   ",~hi_pu)):
        e=a[m1&tough]; s=a[m1&~tough]
        p_t=e[:,2].sum()/e[:,3].sum(); p_s=s[:,2].sum()/s[:,3].sum()
        print(f"  {lab}: 3P% vs TOUGH on-ball {p_t:.3f} (n={len(e)}) vs SOFT {p_s:.3f} -> {p_t-p_s:+.3f}")
    print("interaction real if pull-up delta << C&S delta")
    print("CRE_DEF_DONE",flush=True)

if __name__=="__main__": main()
