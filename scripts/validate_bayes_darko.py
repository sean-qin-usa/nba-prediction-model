"""Same predictive test, but the Bayes prior z comes from DARKO o_dpm instead of
the per-dimension 2K attribute. Head-to-head: which prior source helps more?"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from scripts.validate_bayes_updating import agg, binom_ll, DIMS
from nbapred.model.shooting import fit_dimension

def main():
    con=connect(read_only=True)
    dates=con.execute("""SELECT DISTINCT game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY 1""").fetchdf().game_date
    cut=dates.iloc[int(len(dates)*0.6)]
    tr,te=agg(con,hi=cut),agg(con,lo=cut)
    darko=dict(con.execute("SELECT nba_player_id, o_dpm FROM darko_dpm "
        "WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall())
    con.close()
    for dim,(mk,at,_) in DIMS.items():
        both=tr.join(te,how="inner",lsuffix="_tr",rsuffix="_te")
        both=both[(both[f"{at}_tr"]>=5)&(both[f"{at}_te"]>=5)]
        n_tr=both[f"{at}_tr"].to_numpy(float); m_tr=both[f"{mk}_tr"].to_numpy(float)
        n_te=both[f"{at}_te"].to_numpy(float); m_te=both[f"{mk}_te"].to_numpy(float)
        z=np.array([float(darko.get(int(p)) or np.nan) for p in both.index])
        z=np.nan_to_num((z-np.nanmean(z))/(np.nanstd(z) or 1))
        res=fit_dimension(z,n_tr.astype(int),m_tr.astype(int),num_warmup=300,num_samples=500)
        p=res["p_mean"]
        low=n_tr<np.median(n_tr)
        def sc(mask=None):
            mask=np.ones(len(p),bool) if mask is None else mask
            return binom_ll(m_te[mask],n_te[mask],p[mask]).sum()/n_te[mask].sum()
        print(f"{dim}: BAYES(DARKO o_dpm prior): {sc():.4f} | low-sample {sc(low):.4f}  (beta {res['beta_mean']:+.3f})")

if __name__=="__main__": main()
