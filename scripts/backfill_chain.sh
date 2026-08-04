#!/usr/bin/env bash
# Sequential season backfill chain (D152). ONE worker: seasons run one after
# another so stats.nba.com never sees more than the two backfill streams we
# budgeted. Each season logs to data/logs/bf_<season>.log.
#
#   setsid nohup scripts/backfill_chain.sh 2>&1 &
#
# Args: season labels, in the order to pull.
#   "S:full" every game type, GameRotation attempted (best effort)
#   "S:all"  every game type, no GameRotation
#   "S"      regular season only, no GameRotation  <- the archive default
# GameRotation is off by default because a game the endpoint does not have
# costs a MEASURED 31 s (the server hangs, then returns an empty body) — 1,300
# archive games would be ~11 h of pure timeout for ~40% yield. Only 002 feeds
# four_factors / continuity_map / era_measure, hence --only-regular.
set -u
cd /hdd/steveqin/sean_dev/nba_model || exit 1
for spec in "$@"; do
  season="${spec%%:*}"
  mode="${spec#*:}"
  args=(--season "$season")
  case "$mode" in
    full) ;;                                   # all types + rotations
    all)  args+=(--no-rotations) ;;            # all types, no rotations
    *)    args+=(--only-regular --no-rotations) ;;
  esac
  echo "=== $(date -Is) START $season (mode=${mode})" >> data/logs/backfill_chain.log
  timeout 21600 python3 scripts/backfill_history.py pull "${args[@]}" \
      >> "data/logs/bf_${season}.log" 2>&1
  echo "=== $(date -Is) END   $season rc=$?" >> data/logs/backfill_chain.log
done
echo "=== $(date -Is) CHAIN COMPLETE" >> data/logs/backfill_chain.log
