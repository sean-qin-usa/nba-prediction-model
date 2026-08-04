#!/usr/bin/env python3
"""IG probe (read-only): was the pace-props rejection an artifact of the stale
LEAGUE_PACE=99.5 divisor?

apply_pace multiplies every volume rate by game_pace/99.5, but measured team-game
pace is ~100.8-101.8 across all four seasons -> the 'pace' arm of the ablation
inflated EVERY player's volume by ~+1.5-2% uniformly. Re-run three arms paired:
  base      — no pace adjustment (shipped)
  shipped   — apply_pace as-is (divide by 99.5)
  corrected — mult = game_pace / own_team_pace (rates were earned at own pace;
              level bias cancels by construction)
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from nbapred.db import connect
from nbapred.engine.props import (apply_pace, player_rates_from_stats,
                                  simulate_player, team_pace)


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def apply_pace_corrected(rates, opp_pace, own_pace):
    game_pace = 0.5 * (own_pace + opp_pace)
    mult = game_pace / own_pace
    out = dict(rates)
    for k in ("rate_rim", "rate_mid", "rate_thr", "fta_per_min", "reb_per_min", "ast_per_min"):
        out[k] = rates[k] * mult
    return out


def main(sims=2000, max_eval=600):
    con = connect(read_only=True)
    pg = con.execute("""SELECT s.player_id, s.team_id, g.game_date, g.matchup, g.team_abbrev, s.pts
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '00225%' AND s.seconds>=720 ORDER BY g.game_date""").fetchdf()
    stride = max(1, len(pg) // (max_eval * 2))
    pg = pg.iloc[::stride]
    ab2id = {r[1]: r[0] for r in con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE game_id LIKE '002%'").fetchall()}
    pace_cache, mults_shipped, mults_corr = {}, [], []
    arms = {k: [] for k in ("base", "shipped", "corrected")}
    n = 0
    for r in pg.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if not rates or rates["n_games"] < 8 or rates["proj_min"] < 20:
            continue
        opp = None
        for tok in r.matchup.replace("vs.", "@").split("@"):
            tok = tok.strip()
            if tok and tok != r.team_abbrev:
                opp = tok
        oid = ab2id.get(opp)
        if oid is None:
            continue
        for key, tid in ((("o", oid, str(r.game_date)), oid),
                         (("s", r.team_id, str(r.game_date)), r.team_id)):
            if key not in pace_cache:
                pace_cache[key] = team_pace(con, int(tid), before=r.game_date)
        po = pace_cache[("o", oid, str(r.game_date))]
        ps = pace_cache[("s", r.team_id, str(r.game_date))]
        y = float(r.pts)
        arms["base"].append(crps(simulate_player(rates, sims, seed=n)["points"], y))
        adj = apply_pace(rates, po, ps)
        arms["shipped"].append(crps(simulate_player(adj, sims, seed=n)["points"], y))
        adj2 = apply_pace_corrected(rates, po, ps)
        arms["corrected"].append(crps(simulate_player(adj2, sims, seed=n)["points"], y))
        mults_shipped.append(0.5 * (po + ps) / 99.5)
        mults_corr.append(0.5 * (po + ps) / ps)
        n += 1
    con.close()
    print(f"player-games: {n}")
    print(f"pace mult shipped: mean {np.mean(mults_shipped):.4f} sd {np.std(mults_shipped):.4f}"
          f"  (pure level bias = mean-1)")
    print(f"pace mult corrected: mean {np.mean(mults_corr):.4f} sd {np.std(mults_corr):.4f}")
    for k in arms:
        print(f"  CRPS {k:9s}: {np.mean(arms[k]):.4f}")
    rng = np.random.default_rng(0)
    for a, b in (("base", "shipped"), ("base", "corrected")):
        d = np.array(arms[a]) - np.array(arms[b])
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  CRPS {a}-{b}: {d.mean():+.4f}  CI ({lo:+.4f},{hi:+.4f})"
              f"  ({'B better' if d.mean() > 0 else 'A better'})")


if __name__ == "__main__":
    main()
