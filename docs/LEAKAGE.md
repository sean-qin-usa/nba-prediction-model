# Data-leakage audit

Cardinal rule: a feature predicting a game may use ONLY information that existed
strictly before tip-off. Audit of the current pipeline, worst risks first.

## HIGH — "current snapshot" ratings (the trap Sean named)

- **DARKO (`darko_dpm`)** is TODAY's talent estimate — it already contains the
  rest of every current-season game and beyond. Joining it as a *prior* for any
  past game leaks the future into the backtest. We store one snapshot (today).
- **2K ratings (`ratings_2k`)** are the CURRENT edition (2K27 ≈ 2025 form).
  Same look-ahead, plus an era mismatch (below).

  **Guard shipped:** `nbapred/pit.py` (`darko_asof`, `ratings_2k_asof`) returns
  only the latest snapshot dated *strictly before* the cutoff. Verified: a 2020
  as-of query returns 0 rows (can't leak), a future date returns today's 530.
  **Consequence:** today's DARKO/2K are usable ONLY going forward (live), NOT as
  backtest priors. Backtesting the skill model on past seasons needs historical
  snapshots we don't yet have — until then, backtests must fall back to
  stats-derived priors built from trailing game data only.

## HIGH — aggregating game truths into pregame features

`player_game_stats` / `lineup_stints` are per-game *truths*; the table is fine,
but a skill feature built from them must use a TRAILING window (games before the
target), never a full-season or whole-corpus mean. Full-season means are the
classic silent leak. **Guard shipped:** `pit.trailing_player_stats(player, before)`
filters to `game_date < before`. The fitting layer must route through it.

## MEDIUM — season / era pooling and rule regimes

- Do NOT pool raw ratings or fits across NBA rule-regime breaks (hand-check,
  defensive-three-seconds, take-foul, rest rules). Fit era/regime-specific
  intercepts; freeze calibration across breaks (handoff I.6).
- Walk-forward is **by season only** (III.3); the most recent season(s) are
  never inspected during development.

## VERIFIED CLEAN

- **Elo baseline (`walkforward.py`, `market_accuracy.py`)**: each game is scored
  with `p_home()` BEFORE `update()` runs — the outcome never informs its own
  prediction. First season is burn-in. Between-season regression uses only past
  ratings. No leak.
- **`schedule_features`**: days_rest / travel / games_last_7 read only `past`
  games (the history list is appended to *after* the row is built). `rest_adv`
  uses the opponent's pregame rest — also known before tip. No leak.
- **Odds**: the model is market-blind (I.6); we never feed a line into a
  forecast. H-A uses the OPEN as the divergence point and measures movement to
  the CLOSE — close is an evaluation target, never an input.
- **PIT storage**: every table carries `ingest_ts`; source-event times
  (`snapshot_ts`, `book_last_update`, `scrape_date`, `snapshot_date`) are stored
  separately so as-of joins are possible.

## Deeper pass — subtler forward-looking risks (2nd audit)

Code paths re-verified by inspection, plus risks that are latent in code not yet
written (flagged now so the fit layer can't introduce them):

- **`nba_players.is_active` is a CURRENT flag — survivorship trap.** Filtering
  any historical analysis by `is_active` silently drops players who've since
  retired/left, biasing backtests toward survivors. Rule: never filter a
  past-season sample by today's active flag; use who was rostered *then*.
  (The DARKO "488/530 active" join is a coverage stat, not a filter — fine.)
- **Elo hyperparameters (k, home_edge, regress) are hand-set, untuned.** No leak
  today. But if we ever tune them, tune on a train split only — tuning on the
  full 2007-2023 span would fit the test set. Same for any baseline knob.
- **Elo baseline pools across rule-regime breaks** (continuous 2007-2023). Not
  leakage, but a validity caveat: the 0.617 is a rough baseline, not a
  regime-respecting fit. The real model must honor the I.6 guardrail.
- **`eval_baselines.py` positive control blends the CLOSING line on purpose** —
  it's an illustration that the ablation gate fires, NOT a legitimate forecast
  feature (violates market-blind + uses post-open info). Don't mistake it for a
  model input.
- **Trailing-feature construction (not built yet).** EWMA minutes, season-to-date
  usage, form — every such feature must route through `pit.trailing_player_stats`
  with `before = target game date`. The season game log is re-fetched as it
  grows (immutable=False), so a careless "player's season stats" join would pull
  in games *after* an old target game. Guard exists; the caller must use it.
- **Injury reports (not wired yet):** use the report published BEFORE tip
  (`report_ts < commence_time`), never the final/consolidated report. Schema
  already stores `report_ts`.

Not leakage (checked, to preempt the question): the **market-accuracy** number
scores the market's *pre-game* closing probability against the *post-game*
outcome — that's a legitimate forecast→truth score, not look-ahead. SBR covers
every game each season, so no survivorship in the odds sample.

## Backtest checklist (run before any backtest claim)

1. Ratings priors joined as-of (`pit.py`), never the current snapshot.
2. All player/team features from trailing windows only.
3. Walk-forward split; holdout season untouched in dev.
4. No pooling across a rule-regime break.
5. Market never an input to the forecast being scored.

---

# The 2K era-comparability question (Sean)

*"2K stat accuracy in 1997 vs 2025 is probably less accurate / inflated — does
it balance out?"* **No, not automatically — and it must be handled explicitly.**

Two separate problems:

1. **Scale drift / inflation.** Rating distributions have shifted across editions
   (more 90+ overalls now; attribute definitions changed; category granularity
   grew). A raw `θ ~ α + β·r_2k` pooled across eras conflates era scale with
   skill. **Fix:** standardize each 2K edition to zero-mean/unit-variance
   *within edition* before using it as a regressor. Then β acts on a
   comparable z-score, and the edition intercept absorbs inflation.
2. **Accuracy drift.** Older 2K is a noisier signal of true skill (less data
   behind the ratings, cruder scouting). **Fix:** let β (the learned trust in
   2K) vary by era/edition. The hierarchical model *should* assign older editions
   a smaller β automatically — but only if you let β be era-specific. Forcing a
   single global β lets 1990s noise contaminate modern estimates. This is the
   same β-learning that (per the handoff) should make FT the highest-trust
   dimension; here it also makes recent editions higher-trust than old ones.

So it does not balance out on its own; the design's defenses are
(a) within-edition standardization, (b) era-varying β, (c) the regime guardrail.
Practically it's moot for now — we hold only 2K27, and on-court event data
(nba_api) is the more comparable cross-era signal anyway. If we ever scrape
historical 2K editions for backtesting, apply (a)+(b) before fitting.

## Information policy (Sean, 2026-07-30) — pregame-public vs leakage vs buyable
LEGITIMATE (close forms after these; use freely as model inputs):
- 5PM official injury report (+ trajectory across days)  - confirmed starters (T-30 submission; boxscore START_POSITION)
- official active/inactive lists (pregame)               - referee crew assignments (~9am ET)
- schedule/B2B/travel; standings; coach identity         - all published season/career stats as-of date
LEAKAGE — NEVER inputs (in-game or hindsight; "insider" if known pregame):
- who plays >0 minutes (DNP-CD decided in-game)          - actual minutes played
- actual game stats/outcome of tonight                   - post-tip news
Oracle runs may use these ONLY as labeled ceiling measurements (PAID_ORACLES.md),
never in shipped predictions. ORACLE = must correspond to a BUYABLE product;
"perfect prediction of X" with no vendor behind it is not a valid oracle row.

## INCIDENT LOG

### 2026-08-03 — the certified backtest consumed an availability ORACLE (D158)

**What happened.** `scripts/prod_by_season.py`, the script that produces the
certified capstone artifact `data/capstone_pergame.csv`, built its availability
OUT-sets from *tonight's box score*:

    pl = played.get((gid, t), set())          # player_game_stats, seconds > 0
    outs[t] = {p for p, d0 in comp.players.items()
               if d0["team_id"] == t and ... and p not in pl}

That is "who plays >0 minutes" — the first item on the LEAKAGE list above. It
was the branch taken when `INACTIVE_OUTS` and `REPORT_OUTS` were both unset,
i.e. **exactly the D122 and D132 certification commands**. Every certified
headline from D122 through D156 that cites `capstone_pergame.csv` is therefore
an availability-oracle number.

**Blast radius: BACKTEST ONLY. The live path was never affected.** The October
entrypoint builds OUT sets from `injury_reports_pit` (`nbapred/engine/slate.py`
:54-63, :77-79) and `scripts/predict_today.py:48,58-60` consumes those. So the
defect made the *published expectation too good*; it never made a live
prediction wrong. The same defect exists in `scripts/history_eval.py:208-211`
(the D153 15-year OOS run) — flagged, not yet re-run.

**Cost.** Pooled log loss +0.00386, normalized gap 11.13% -> **14.95%**
(+3.82pp). The leak is monotone in the OUT count (zero OUTs -> zero effect;
6+ OUTs -> +0.00652) and the oracle carried 4.05 OUT players per game against
the honest 1.88.

**Fix (D158).** The honest construction — official 5PM injury report UNION
official pregame inactives, empty where no feed covers the season — is now the
DEFAULT. The oracle survives only behind `ORACLE_PLAYED_OUTS=1`, prints a
NOT-CERTIFIABLE banner, and is mechanically redirected away from the certified
CSV path so it cannot re-occupy it.

**Rules this incident adds.**
1. An oracle construction must never be reachable as a DEFAULT. Leaky paths are
   opt-in, loudly named, and self-labelling in the run header.
2. Any script that writes a certifiable artifact must PRINT the information
   tier that produced it. "Which construction made this number" must never
   again be recoverable only by reading the source.
3. A backtest whose availability feed does not cover a season falls back to
   EMPTY out-sets (honest, weaker), never to a hindsight set.
