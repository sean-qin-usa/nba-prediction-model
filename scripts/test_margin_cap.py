"""Margin-of-victory diminishing returns (538-style): blowout margins carry
less info (garbage time, benches). Test: cap/compress game margins fed to the
SRS — soft-cap margin m -> sign(m)*(t + (|m|-t)*alpha) beyond threshold t.
Walk-forward, 3 recent seasons, vs raw-margin baseline.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from scripts.eval_coldstart import MarginSRS, season_end_ratings

SCALE=7.2
sig=lambda x:1/(1+np.exp(-np.asarray(x)))

def softcap(m,t=15.0,alpha=0.3):
    a=abs(m)
    return np.sign(m)*(a if a<=t else t+(a-t)*alpha)

def run(df, cap=False, regress=0.75, ridge=30.0, refit=10):
    prev={}
    Y,P=[],[]
    for se,grp in df.groupby("season_end"):
        grp=grp.sort_values("game_date").reset_index(drop=True)
        prior={t:regress*v for t,v in prev.items()}
        hist,srs,since=[],None,10**9
        for i,g in grp.iterrows():
            if since>=refit:
                srs=MarginSRS(ridge,prior=prior).fit(hist); since=0
            since+=1
            if se>=2024:  # score only recent 3 seasons
                Y.append(int(g.home_win))
                m=srs.margin(g.home,g.away) if srs and srs.r else prior.get(g.home,0)-prior.get(g.away,0)+3
                P.append(float(sig(m/SCALE)))
            mm=g.score_home-g.score_away
            hist.append((g.home,g.away, softcap(mm) if cap else mm))
        prev=season_end_ratings(hist,ridge)
    return log_loss(np.array(Y),P), len(Y)

def main():
    con=connect(read_only=True)
    df=con.execute("""SELECT season_end, game_date, home, away, score_home, score_away, home_win
        FROM odds_market WHERE season_end>=2021 ORDER BY season_end, game_date""").fetchdf()
    con.close()
    raw,n=run(df,cap=False)
    capd,_=run(df,cap=True)
    print(f"scored games (2023-24..2025-26): {n}")
    print(f"  raw margins    : {raw:.4f}")
    print(f"  soft-cap(15,.3): {capd:.4f}   delta {raw-capd:+.4f}")

if __name__=="__main__": main()
