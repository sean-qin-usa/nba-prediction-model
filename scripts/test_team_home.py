"""Team-specific home advantage (Denver altitude etc.), ridge-shrunk toward the
league home edge. Classic feature in published models. Walk-forward 3 seasons.
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

class HomeSRS(MarginSRS):
    def __init__(self, ridge=30.0, prior=None, home_ridge=200.0):
        super().__init__(ridge, prior); self.home_ridge=home_ridge; self.h={}
    def fit(self, rows):
        teams=sorted({t for h,a,_ in rows for t in (h,a)})
        idx={t:i for i,t in enumerate(teams)}; T=len(teams)
        if T<5 or len(rows)<40:
            self.r=dict(self.prior); return self
        # params: [r_0..r_{T-1}, mu_home, dh_0..dh_{T-1}] (dh = team home deviation)
        X=np.zeros((len(rows),2*T+1)); y=np.zeros(len(rows))
        for k,(h,a,m) in enumerate(rows):
            X[k,idx[h]]+=1; X[k,idx[a]]-=1; X[k,T]=1; X[k,T+1+idx[h]]=1; y[k]=m
        p=np.array([self.prior.get(t,0.0) for t in teams]+[3.0]+[0.0]*T)
        yr=y-X@p
        P=np.full(2*T+1,self.ridge); P[T]=0.0; P[T+1:]=self.home_ridge
        beta=p+np.linalg.solve(X.T@X+np.diag(P),X.T@yr)
        rr=beta[:T]-beta[:T].mean()
        self.r={t:float(rr[idx[t]]) for t in teams}
        self.home=float(beta[T]); self.h={t:float(beta[T+1+idx[t]]) for t in teams}
        return self
    def margin(self,h,a):
        return self.r.get(h,self.prior.get(h,0.0))-self.r.get(a,self.prior.get(a,0.0))+self.home+self.h.get(h,0.0)

def run(df, cls, regress=0.75, ridge=30.0, refit=10, **kw):
    prev={}; Y,P=[],[]
    for se,grp in df.groupby("season_end"):
        grp=grp.sort_values("game_date").reset_index(drop=True)
        prior={t:regress*v for t,v in prev.items()}
        hist,srs,since=[],None,10**9
        for i,g in grp.iterrows():
            if since>=refit:
                srs=cls(ridge,prior=prior,**kw).fit(hist) if kw else cls(ridge,prior=prior).fit(hist)
                since=0
            since+=1
            if se>=2024:
                Y.append(int(g.home_win))
                m=srs.margin(g.home,g.away) if srs and srs.r else prior.get(g.home,0)-prior.get(g.away,0)+3
                P.append(float(sig(m/SCALE)))
            hist.append((g.home,g.away,g.score_home-g.score_away))
        prev=season_end_ratings(hist,ridge)
    return log_loss(np.array(Y),P), len(Y)

def main():
    con=connect(read_only=True)
    df=con.execute("""SELECT season_end, game_date, home, away, score_home, score_away, home_win
        FROM odds_market WHERE season_end>=2021 ORDER BY season_end, game_date""").fetchdf()
    con.close()
    base,n=run(df,MarginSRS)
    for hr in (400.0,200.0,100.0):
        th,_=run(df,HomeSRS,home_ridge=hr)
        print(f"team-home ridge={hr:.0f}: {th:.4f}  (base {base:.4f}, delta {base-th:+.4f}, n={n})")

if __name__=="__main__": main()

def bootstrap():
    con=connect(read_only=True)
    df=con.execute("""SELECT season_end, game_date, home, away, score_home, score_away, home_win
        FROM odds_market WHERE season_end>=2021 ORDER BY season_end, game_date""").fetchdf()
    con.close()
    # per-game losses for base and team-home(200)
    def losses(cls,**kw):
        prev={}; L=[]
        for se,grp in df.groupby("season_end"):
            grp=grp.sort_values("game_date").reset_index(drop=True)
            prior={t:0.75*v for t,v in prev.items()}
            hist,srs,since=[],None,10**9
            for i,g in grp.iterrows():
                if since>=10:
                    srs=cls(30.0,prior=prior,**kw).fit(hist) if kw else cls(30.0,prior=prior).fit(hist)
                    since=0
                since+=1
                if se>=2024:
                    m=srs.margin(g.home,g.away) if srs and srs.r else 3.0
                    p=np.clip(float(sig(m/SCALE)),1e-9,1-1e-9)
                    yv=int(g.home_win)
                    L.append(-(yv*np.log(p)+(1-yv)*np.log(1-p)))
                hist.append((g.home,g.away,g.score_home-g.score_away))
            prev=season_end_ratings(hist,30.0)
        return np.array(L)
    lb=losses(MarginSRS); lh=losses(HomeSRS,home_ridge=200.0)
    d=lb-lh; rng=np.random.default_rng(0)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(3000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"n={len(d)} team-home improvement {d.mean():+.5f} CI ({lo:+.5f},{hi:+.5f}) -> {'KEEP' if lo>0 else 'NS'}")

bootstrap()
