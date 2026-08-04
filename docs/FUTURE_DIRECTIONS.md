# Future directions — gated on data we don't have yet

Things worth doing that are BLOCKED on data availability. Recorded so they're
not forgotten; meanwhile we pursue paths that need no new data (see bottom).

## Blocked on LIVE / IN-SEASON market data (available ~October 2026)
The live betting path is BUILT (odds logger, injury-report poller) but has NO
data now — it's the offseason (July), there are no NBA games, so no odds, no
injury reports, no lines to capture. Every edge measurement needs this:

- **Prop-line comparison (the actual H-B edge test)** — our prop sim is
  internally calibrated (PIT 0.54) but has never faced a real prop line. Needs
  The Odds API player-prop markets (in-season; credit-priced). UNTIL THEN we
  cannot measure prop edge, only internal calibration.
- **Opening lines + line movement (H-A / CLV)** — the timing edge (reprice
  faster than the book on lineup news) needs timestamped OPENING lines and the
  moves to close. Odds logger captures these once games exist.
- **SGP / parlay quotes** — to price our joint-distribution edge against the
  book's correlation assumption. Not in any free feed; Odds API SGP is limited.
- **Historical player-prop odds: NOT free anywhere** (Kaggle searched 2026-07-28:
  nothing). Paid: The Odds API historical props (since 2023). Free FORWARD plan
  from October: log Odds API free-tier props (budget-limited) + PrizePicks /
  Underdog keyless projection endpoints (current lines, no history — logging
  them daily builds our own archive; they're softer books = better edge targets).
- **defended_fg tracking ingest incomplete** (only 2024-25 'Overall', 568 rows;
  other categories/seasons didn't land) — rerun nbapred/ingest/tracking.py load
  before using individual-matchup features.
- **Injury reports** — the official PDF feed publishes only on game days;
  poller armed, nothing to capture until October.

Action: when the season starts, run the logger from day one (opens are the
scarcest, non-reconstructible asset). Free tier (500 credits/mo) budget-paces
main lines only; props need a paid tier decision (docs/PAID_OPTIONS.md).

## Blocked on TRACKING data (partially available, ingest running)
- **Individual matchup ('who guards the shooter')** — defended-FG% ingest
  launched (nbapred/ingest/tracking.py). This is the one place def-RAPM can beat
  raw team allowance in props. Speed / pace-adaptation skills need SportVU
  speed/distance (available via nba_api, not yet pulled).

## Blocked on HISTORICAL play-by-play beyond 2024-26
- Multi-season RAPM / skill fits need nba_api PBP for older seasons (backfillable
  — just ingestion time). Would reduce RAPM noise (which killed injury-pricing).

---

## Paths NOT blocked — pursue these now (no new data needed)
- **Minutes projection** — DIAGNOSED as the #1 prop lever: minutes error = 0.60
  CRPS = 11.2% of prop error, 30-60x the scoring-rate improvement. This is why
  Kalman didn't help props (it fixed the rate center, not the minutes
  bottleneck). Buildable entirely from player_game_stats. HIGHEST PRIORITY.
- **Trade / roster-continuity** — chemistry disruption season ratings miss;
  derivable from roster co-appearance history.
- **Rest-on-props** — players score less on zero rest; testable now.
- **Empirical minutes distribution** — real minutes are skewed (blowouts/foul
  trouble/DNP), not Normal; sampling the empirical distribution may fix the tails.
