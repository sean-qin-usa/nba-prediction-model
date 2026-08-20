"""THE Bayesian-updating test (Sean, 2026-07-28): does the posterior actually
CAPTURE SKILL — i.e., predict a player's FUTURE performance better than his raw
past rates do?

Design (temporal split, no leakage): aggregate each player's sufficient stats
over the FIRST 60% of 2025-26; estimate three ways; score each on the LAST 40%:
  1. RAW MLE          — past makes/attempts, no shrinkage (what a naive model does)
  2. EB SHRINKAGE     — league-mean shrinkage (skill_priors.py machinery)
  3. BAYES POSTERIOR  — the v1 hierarchical fit (2K prior + hierarchy) on the
                        train half only
Scoring: per-player binomial log-likelihood of TEST makes given predicted rate,
attempt-weighted; plus the low-sample subset (< median attempts) where shrinkage
must matter most. If Bayes(3) >= EB(2) > MLE(1), updating works as designed.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect

DIMS = {"thr": ("thrm", "thra", "Three-Point Shot"),
        "ft": ("ftm", "fta", "Free Throw"),
        "rim": ("rimm", "rima", "Close Shot")}


def agg(con, lo=None, hi=None):
    cond = []
    params = []
    if lo is not None:
        cond.append("g.game_date >= ?"); params.append(lo)
    if hi is not None:
        cond.append("g.game_date < ?"); params.append(hi)
    where = (" AND " + " AND ".join(cond)) if cond else ""
    return con.execute(f"""SELECT s.player_id,
        sum(s.thrm) thrm, sum(s.thra) thra, sum(s.ftm) ftm, sum(s.fta) fta,
        sum(s.rimm) rimm, sum(s.rima) rima
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season='2025-26' AND s.game_id LIKE '002%'{where} GROUP BY 1""",
        params).fetchdf().set_index("player_id")


def binom_ll(m, n, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return m * np.log(p) + (n - m) * np.log(1 - p)


def main():
    con = connect(read_only=True)
    dates = con.execute("""SELECT DISTINCT game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY 1""").fetchdf().game_date
    cut = dates.iloc[int(len(dates) * 0.6)]
    tr, te = agg(con, hi=cut), agg(con, lo=cut)

    # Bayes posterior on train half: reuse v1 fit machinery, shooting dims only
    import json
    xw = con.execute("SELECT nba_player_id, name_2k FROM player_xwalk WHERE name_2k IS NOT NULL").fetchdf()
    r2k = con.execute("SELECT player_name, attributes FROM ratings_2k "
                      "WHERE scrape_date=(SELECT max(scrape_date) FROM ratings_2k)").fetchdf()
    name2attr = {r.player_name: json.loads(r.attributes) for r in r2k.itertuples()}
    pid2attr = {int(r.nba_player_id): name2attr.get(r.name_2k) for r in xw.itertuples()}
    con.close()

    from nbapred.model.shooting import fit_dimension
    print(f"train players {len(tr)}  test players {len(te)}  cut {cut}")
    for dim, (mk, at, a2k) in DIMS.items():
        both = tr.join(te, how="inner", lsuffix="_tr", rsuffix="_te")
        both = both[(both[f"{at}_tr"] >= 5) & (both[f"{at}_te"] >= 5)]
        n_tr = both[f"{at}_tr"].to_numpy(float); m_tr = both[f"{mk}_tr"].to_numpy(float)
        n_te = both[f"{at}_te"].to_numpy(float); m_te = both[f"{mk}_te"].to_numpy(float)

        # 1 raw MLE
        p_mle = m_tr / n_tr
        # 2 EB league shrinkage (method of moments)
        from nbapred.features.skill_priors import _beta_mom
        a, b = _beta_mom(m_tr, n_tr)
        p_eb = (m_tr + a) / (n_tr + a + b)
        # 3 Bayes hierarchical + 2K prior (train half only)
        z = np.array([float((pid2attr.get(int(p)) or {}).get(a2k) or np.nan) for p in both.index])
        z = np.nan_to_num((z - np.nanmean(z)) / (np.nanstd(z) or 1))
        res = fit_dimension(z, n_tr.astype(int), m_tr.astype(int),
                            num_warmup=300, num_samples=500)
        p_bayes = res["p_mean"]

        def score(p, mask=None):
            mask = np.ones(len(p), bool) if mask is None else mask
            return binom_ll(m_te[mask], n_te[mask], p[mask]).sum() / n_te[mask].sum()

        low = n_tr < np.median(n_tr)
        print(f"\n{dim}: test LL/attempt (higher=better) | all -> low-sample players")
        print(f"  raw MLE : {score(p_mle):.4f} | {score(p_mle, low):.4f}")
        print(f"  EB      : {score(p_eb):.4f} | {score(p_eb, low):.4f}")
        print(f"  BAYES   : {score(p_bayes):.4f} | {score(p_bayes, low):.4f}")


if __name__ == "__main__":
    main()
