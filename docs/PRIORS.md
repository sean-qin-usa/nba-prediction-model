# Where player-skill priors come from (2K vs stats vs DARKO)

Sean's question (2026-07-26): *2K ratings lag ~a full season — can we get
skill priors from actual stats instead?* Yes. The plan below makes live stats
the primary prior and demotes 2K to the one job it's actually best at.

## The lag problem is real
NBA 2K's base ratings are set preseason from last season's film + scouting, and
patched only intermittently during the year. For an established player mid-2026,
2K encodes ~2025. That's stale for a forecast. Three free, current alternatives:

| Source | What it gives | Lag | Join key |
|---|---|---|---|
| **nba_api** (own PBP/box) | empirical per-dimension rates (3PT%, rim FG%, defended-FG%, STL/BLK/DREB rate, FT%…) | days | player_id |
| **DARKO DPM** | Bayesian talent estimate: O/D DPM, box vs on-off (RAPM-like), age-aware | **daily** | player_id |
| **2K ratings** | scouted attributes incl. traits with little/no NBA sample | ~1 season | name (crosswalk) |

DARKO and nba_api both key on the NBA `player_id` — **no name matching**, unlike
2K (which needs the crosswalk).

## Decision: stats-primary, 2K for thin data only

The handoff's prior is `θ_{i,k} ~ N(α_k + β_k·r_{i,k}, σ_k²)` with `(α_k, β_k)`
learned league-wide — i.e. the model already learns *how much to trust each
rating dimension*. We extend `r` from "2K only" to a small set of regressors per
dimension and let the βs sort out trust:

1. **Established players → live stats drive the prior center.** For each latent
   skill, regress onto the matching empirical rate from our own PBP/box data
   (shrunk by sample (`II.2` luck regression), age-curved per `II.1`). DARKO's
   O/D DPM enters as the two-way-impact regressor — and does double duty as a
   ready-made target/anchor for the stint-margin (RAPM) likelihood in `II.2`.
2. **2K becomes the low-sample prior, not the default.** For rookies and
   players with thin NBA samples, the stats estimate is noise; here 2K's
   scouting (and the v1.5 college-translation model) carries real information
   the box score can't yet. So 2K's weight should *rise* as NBA sample falls —
   the opposite of using it as everyone's center.
3. **Let β expose disagreement.** Keeping 2K in the regressor set (not deleting
   it) lets us read off, per dimension, how much 2K adds *over* current stats.
   The handoff's own sanity check (FT should show the highest 2K-β) still holds;
   now we also learn where 2K is merely echoing stale stats and can be dropped.

Net: 2K stops being the trusted center for stars (where it lags) and is kept
where it's genuinely additive (no-sample players). This is strictly more
current than the original design and costs nothing.

## Status
- DARKO ingester live (`scripts/load_darko.py` → `darko_dpm`, daily snapshots).
- Per-dimension empirical rates: derived in the PBP→sufficient-stats layer (next
  build step); this doc is the contract that layer targets.
- 2K scrape already stored; crosswalk already ~99.6%.
