# OCTOBER CAPTURE SPEC — what must be on disk before opening night

Frozen 2026-08-08, before any 2026-27 price exists. Everything below is a
schema and a decision rule, not a model: the point is that the choices are made
while no outcome can influence them.

**The one irrecoverable thing.** An opening price not captured on the night is
not recoverable the next day. Every other item here can be backfilled or
re-derived; the open cannot. That is the whole reason this document exists.

---

## 1. WHAT THE LOGGER ALREADY DOES (D228)

The cadence is tip-relative, not clock-relative, because `wlm_chart` measured
76% / 80% / 91% of open-to-close movement complete at T-4h / T-2h / T-1h and
~75% already gone by the 5PM ET report. A report-anchored snapshot pair would
sit at a different completion fraction for every game.

    ladder    OPEN (10:00 ET or T-4h, whichever is earlier)
              T-4h -> T-2h -> T-1h -> CLOSE (T-15m)
    cost      1 credit/poll (spreads only; h2h+totals ride along once a day)
    budget    simulated 28 dense nights from 500 credits: 483 spent, 28/28
              captured; degrades to OPEN+CLOSE and stops
    props     rationed, 1 market (player_points), nightly event cap

Records land in `data/raw/odds/YYYY-MM-DD.jsonl`, one appended fsync'd line per
poll, stamped with `target_kind` so the ladder position survives to load time.
`scripts/load_odds.py` flattens into `odds_quotes`, which already carries
`book_last_update` — so **quote age is derivable prospectively at no extra
cost**, and quote age was one of the three features the conditional-trust idea
needs and the historical archive cannot supply.

## 2. WHAT MUST BE ADDED — INJURY-STATUS TRANSITIONS

The archive stores injury reports as SNAPSHOTS keyed by report date. What the
market responds to is the TRANSITION, and a transition is only visible if both
sides of it are on disk with times attached.

Required table (`injury_events`), written by the existing report poller:

| column | meaning |
|---|---|
| `observed_ts` | when WE saw it (our clock) |
| `report_edition` | the NBA's own edition stamp, e.g. `05PM` |
| `game_date`, `team`, `player_id` | the subject |
| `status_from`, `status_to` | the transition; `status_from` NULL on first sight |
| `reason` | the report's own reason string, unparsed |
| `seq` | monotone per (player, game_date), so ordering never depends on ties |

**A transition row is written only when `status_to != status_from`.** Snapshots
that repeat the prior status are not events and must not inflate the event
count — otherwise a player listed Questionable for six straight editions looks
like six pieces of news.

## 3. THE PAIRING RULE, DECIDED NOW

For each injury event, the market response is

    delta_spread = consensus(t + w) - consensus(t - w)

with **w = 30 minutes** and `consensus` the median home-margin across books
quoting inside the window. Fixed in advance because w is exactly the kind of
parameter that can be tuned until an effect appears.

Events are DISCARDED, not shrunk, when:
- fewer than 2 books quote on either side of the window;
- another event for the same game falls inside the same window (the response is
  not attributable);
- the window crosses tip.

Discards are COUNTED AND REPORTED. A silent discard rule is how a filtered
sample starts looking like a clean one.

## 4. SEPARATING NEWS FROM STALE-BOOK CATCH-UP

The known trap: a book that had not moved yet will "respond" to an event that
already happened, and that is not information arriving, it is a slow book
converging. Quote age is what distinguishes them, and it is now recorded.

Pre-registered split: a book's move counts as RESPONSE only if its
`book_last_update` precedes the event and its next quote follows it. A book
whose quote was already stale before the event is classified CATCH-UP and
reported separately. **Both series are kept; neither is dropped.**

## 5. WHAT WILL BE ESTIMATED, AND WHAT WOULD FALSIFY IT

    delta_spread ~ player_impact * delta_P(out)

fitted with hierarchical shrinkage by player toward positional and
minutes-band means, since most players will have very few events in one season.

Pre-registered predictions:
- **P1** The coefficient is positive and larger for higher-minute players.
- **P2** Player impacts estimated this way correlate positively with the same
  players' `talent x trailing minutes` from the composition leg. If they do not,
  either the market is pricing something the model cannot see, or the event
  pairing is broken — and the correlation is what tells the two apart.
- **P3** CATCH-UP moves are smaller than RESPONSE moves. If they are the same
  size, the split is not doing its job and the whole channel is suspect.
- **P4** One season yields too few events per player to beat the composition
  leg outright. The first-season deliverable is a MEASUREMENT, not a model.

## 6. GO / NO-GO ON OPENING NIGHT

Run `scripts/logger_canary.py`. It checks OUTPUT, not process, because
`systemctl is-active` stays green through an exhausted key, a revoked key, a
wrong sport key, and a season that started while the poller idled.

NO-GO if any of: no capture in 48h; credits at zero; zero events listed on a
game day; no `open` snapshot for a game date that has games.

## 7. WHAT IS DELIBERATELY NOT IN SCOPE

- No live betting. The 2026-27 season is a measurement season.
- No re-selection of the sides configuration on 2026-27 outcomes.
- The offset layer and the D232 absence term are FROZEN as of this document;
  `data/cert_manifest.json` records the code and coefficient hashes that define
  "frozen", and `cert_manifest.py --check` is the test.
