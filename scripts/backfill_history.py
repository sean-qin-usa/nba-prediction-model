#!/usr/bin/env python3
"""Historical NBA backfill driver (D152).

Three modes, all reusing nbapred.ingest.nba_stats' throttled+cached client
(>=0.65s + jitter, 4-try exponential backoff, raw JSON cached before parse):

  probe   -- how far back does each source actually go, and at what cost.
             Per season: LeagueGameFinder count, then one sample 002 game
             through boxscoretraditionalv3 / playbyplayv3 / gamerotation,
             recording whether the fields our features need are present.
  pull    -- land one season: game log -> nba_games, then per-game artifacts.
             Network happens OUTSIDE the DB lock (standing rule); the only
             write window is load_season_games' DELETE+executemany.
  verify  -- per-season reconciliation: 002 game count vs expectation,
             player_game_stats rows/games, zone-feature completeness.

Usage:
  python scripts/backfill_history.py probe --from 1996-97 --to 2018-19
  python scripts/backfill_history.py pull --season 2018-19
  python scripts/backfill_history.py verify [--season S]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred import threads
threads.pin(1)  # noqa: E402  -- before pandas/numpy

import hashlib  # noqa: E402

from nbapred.config import RAW_NBA  # noqa: E402
from nbapred.db import connect  # noqa: E402
from nbapred.ingest import nba_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill")

# regular-season game counts we expect, by season (82-game era + known shorts)
EXPECTED = {
    "2011-12": 990,    # lockout, 66 games
    # 2012-13: game 0021201214 (BOS @ IND, 2013-04-16) was CANCELLED after the
    # Boston Marathon bombing and never made up. Verified in-data: exactly BOS
    # and IND have 81 GP, every other team 82, and 1214 is the only gap in the
    # sequence. 1229 is CORRECT, not a backfill hole.
    "2012-13": 1229,
    "2019-20": 1059,   # COVID stop + 88 bubble seeding games
    "2020-21": 1080,   # COVID, 72 games
    "1998-99": 725,    # lockout, 50 games
}


def seasons_between(a: str, b: str) -> list[str]:
    ya, yb = int(a[:4]), int(b[:4])
    step = 1 if yb >= ya else -1
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(ya, yb + step, step)]


def expected_002(season: str) -> int:
    if season in EXPECTED:
        return EXPECTED[season]
    y = int(season[:4])
    if y >= 2004:
        return 1230
    if y >= 1995:
        return 1189
    return 1107


def cached(bucket: str, gid: str) -> bool:
    key = hashlib.sha1(json.dumps({"game_id": gid}, sort_keys=True).encode()).hexdigest()[:16]
    return (RAW_NBA / bucket / f"{key}.json").exists()


# ---------------------------------------------------------------- probe ----

def probe_season(season: str) -> dict:
    out = {"season": season}
    t0 = time.monotonic()
    try:
        df = nba_stats.pull_season_games(season)
    except Exception as e:  # noqa: BLE001
        out["gamefinder"] = f"FAIL {type(e).__name__}: {e}"
        return out
    out["teamgame_rows"] = len(df)
    gids = sorted(set(df["GAME_ID"]))
    out["games_all"] = len(gids)
    reg = sorted(g for g in gids if g.startswith("002"))
    out["games_002"] = len(reg)
    out["pre_001"] = sum(1 for g in gids if g.startswith("001"))
    out["post_004"] = sum(1 for g in gids if g.startswith("004"))
    out["gf_secs"] = round(time.monotonic() - t0, 1)
    if not reg:
        return out
    # sample a game from the middle of the season (avoids opening-night oddities)
    gid = reg[len(reg) // 2]
    out["sample_game"] = gid

    # boxscore
    try:
        raw = nba_stats.pull_boxscore(gid)
        bg = raw["boxScoreTraditional"]
        ps = bg["homeTeam"]["players"]
        st = ps[0]["statistics"] if ps else {}
        out["box"] = "ok"
        out["box_players"] = len(ps) + len(bg["awayTeam"]["players"])
        for k, lbl in [("minutes", "min"), ("plusMinusPoints", "pm"), ("steals", "stl"),
                       ("blocks", "blk"), ("turnovers", "tov"),
                       ("threePointersAttempted", "fg3a")]:
            out[f"box_{lbl}"] = st.get(k, "MISSING")
    except Exception as e:  # noqa: BLE001
        out["box"] = f"FAIL {type(e).__name__}: {str(e)[:80]}"

    # play-by-play + the zone fields possessions.py needs
    try:
        pbp = nba_stats.pull_play_by_play(gid)
        out["pbp"] = "ok"
        out["pbp_actions"] = len(pbp)
        cols = set(pbp.columns)
        out["pbp_has_shotValue"] = "shotValue" in cols
        out["pbp_has_shotDistance"] = "shotDistance" in cols
        out["pbp_has_subType"] = "subType" in cols
        out["pbp_has_personId"] = "personId" in cols
        if len(pbp):
            shots = pbp[pbp.get("actionType", "").isin(["Made Shot", "Missed Shot"])] \
                if "actionType" in cols else pbp.iloc[0:0]
            out["pbp_shots"] = len(shots)
            if len(shots):
                nn = shots["shotDistance"].notna().mean() if "shotDistance" in cols else 0.0
                out["pbp_shotdist_nonnull"] = round(float(nn), 3)
                nz = (shots["personId"] > 0).mean() if "personId" in cols else 0.0
                out["pbp_personid_nonzero"] = round(float(nz), 3)
    except Exception as e:  # noqa: BLE001
        out["pbp"] = f"FAIL {type(e).__name__}: {str(e)[:80]}"

    # rotations (lineup_stints source)
    try:
        rot = nba_stats.pull_rotations(gid)
        n = sum(len(v) for v in rot.values())
        out["rot"] = "ok" if n else "empty"
        out["rot_rows"] = n
    except Exception as e:  # noqa: BLE001
        out["rot"] = f"FAIL {type(e).__name__}: {str(e)[:80]}"
    return out


def cmd_probe(args) -> None:
    seasons = seasons_between(args.to, getattr(args, "from"))  # newest first
    res = []
    outp = Path(args.out)
    for s in seasons:
        r = probe_season(s)
        res.append(r)
        log.info("%s", json.dumps(r))
        outp.write_text(json.dumps(res, indent=1, default=str))
    log.info("wrote %s", outp)


# ----------------------------------------------------------------- pull ----

def yielding_connect(tag: str = "", max_wait_min: float = 240.0):
    """Write connection that YIELDS to whoever already holds the lock.

    The box runs concurrent gate scripts and the nba_model cron pullers against
    the same DuckDB file; a backfill must never be the reason one of them
    fails. duckdb.connect raises IOException while another process holds the
    write lock, so poll politely (60s, per the standing rule) instead of
    hammering, and log the holder so contention is diagnosable.
    """
    import re

    import duckdb
    t0 = time.monotonic()
    while True:
        try:
            return connect(retry_s=0)
        except duckdb.IOException as e:
            if (time.monotonic() - t0) / 60 > max_wait_min:
                raise
            m = re.search(r"PID (\d+)", str(e))
            log.info("%s write lock held%s; yielding 60s (waited %.0f min)",
                     tag, f" by PID {m.group(1)}" if m else "",
                     (time.monotonic() - t0) / 60)
            time.sleep(60)


def cmd_pull(args) -> None:
    season = args.season
    # 1. NETWORK, no lock
    df = nba_stats.pull_season_games(season)
    log.info("%s: gamefinder %d team-game rows", season, len(df))
    # 2. per-game artifacts FIRST: raw files only, NO lock held at all.
    #    (Ordered before the nba_games write on purpose — the DB write can wait
    #    for a busy writer for as long as it likes without stalling the pull.)
    gids = sorted(set(df["GAME_ID"]))
    if args.only_regular:
        gids = [g for g in gids if g.startswith("002")]

    # COMPLETENESS = boxscore + PBP only. GameRotation is deliberately NOT a
    # completeness condition here: the endpoint returns an empty body for most
    # pre-2019 games and scattered later ones (measured, see notes), so making
    # it required would re-attempt every structurally-absent game on every run.
    # It is still attempted once per new game below, best-effort.
    def complete(g: str) -> bool:
        return cached("playbyplayv3", g) and cached("boxscoretraditionalv3", g)

    todo = [g for g in gids if not complete(g)]
    if args.max_games:
        todo = todo[: args.max_games]
    log.info("%s: %d/%d games need artifacts", season, len(todo), len(gids))
    fails = []
    n_rot = n_rot_miss = 0
    t0 = time.monotonic()
    for i, gid in enumerate(todo):
        try:
            nba_stats.pull_play_by_play(gid)
            nba_stats.pull_boxscore(gid)
            if gid.startswith(("002", "004")) and not args.no_rotations \
                    and not cached("gamerotation", gid):
                try:
                    nba_stats.pull_rotations(gid, attempts=1)   # best effort
                    n_rot += 1
                except Exception:  # noqa: BLE001
                    n_rot_miss += 1
        except Exception as e:  # noqa: BLE001
            log.warning("game %s failed: %s", gid, str(e)[:120])
            fails.append(gid)
        if (i + 1) % 50 == 0:
            el = time.monotonic() - t0
            log.info("%s progress %d/%d  %.1f s  (%.2f s/game, eta %.0f min)",
                     season, i + 1, len(todo), el, el / (i + 1),
                     (len(todo) - i - 1) * el / (i + 1) / 60)
    log.info("%s DONE artifacts in %.0f min; %d failures %s; rotations %d ok / "
             "%d absent", season, (time.monotonic() - t0) / 60, len(fails),
             fails[:20], n_rot, n_rot_miss)

    # 3. SHORT WRITE WINDOW, last, and only when the lock is actually free.
    con = yielding_connect(season)
    n = nba_stats.load_season_games(con, season, df=df)
    con.close()
    log.info("%s: nba_games <- %d rows (lock released)", season, n)


# ------------------------------------------------------------------ pgs ----
# Land player_game_stats for one season FROM THE RAW CACHE ONLY (D160).
#
# D152's chains completed every artifact pull down to 1996-97 (bf_*.log: "DONE
# artifacts ... 0 failures") but `build_features.py` ran mid-pull on 2026-08-02
# ~09:40 and only saw a partial cache, so 2008-09 stopped at 386 games and
# 2013-14 at 27 — and load_corpus' incremental `have` set then marked exactly
# those game_ids done. Finishing the job is therefore a LOCAL PARSE, no
# network. This subcommand does it ONE SEASON AT A TIME with the zone-dead
# refusal D152 asked for applied BEFORE anything is written.

ZONE_DEAD_MAX_SHARE = 0.02   # share of games allowed to have fga>0 but zero
                             # zone attempts before the season is REFUSED
ZONE_RATIO_MIN = 0.98        # SUM(zone attempts)/SUM(fga); healthy seasons
                             # measure exactly 1.000 (every shot maps to a zone)


def season_code(season: str) -> str:
    return f"{int(season[:4]) % 100:02d}"


def season_of(gid: str) -> str:
    y = int(gid[3:5])
    yy = 1900 + y if y >= 46 else 2000 + y
    return f"{yy}-{(yy + 1) % 100:02d}"


def cmd_pgs(args) -> None:
    """Parse cached box+PBP into player_game_stats for ONE season.

    NETWORK: none. LOCK: taken only once the batch is built, via
    yielding_connect, and released immediately (register + INSERT..SELECT).
    """
    import datetime as dt

    from nbapred.features import possessions
    from nbapred.features.cache_index import game_index

    season = args.season
    if int(season[:4]) < 1996:
        raise SystemExit(
            f"REFUSED: {season} is below the playbyplayv3 hard floor (1996-97). "
            "D152 measured HTTP 200 with ZERO actions for every season <= "
            "1995-96, which would land every zone count 0 and silently degrade "
            "eFG to fgm/fga. Pre-1996-97 is refused, not deferred.")

    boxes = game_index("boxscoretraditionalv3")
    pbps = game_index("playbyplayv3")
    pre = "002" + season_code(season)
    gids = sorted(g for g in boxes if g.startswith(pre))
    have_pbp = [g for g in gids if g in pbps]
    log.info("%s: cache has %d box / %d with PBP", season, len(gids), len(have_pbp))

    con = connect(read_only=True)
    have = {r[0] for r in con.execute(
        "SELECT DISTINCT game_id FROM player_game_stats WHERE substr(game_id,1,5)=?",
        [pre]).fetchall()}
    con.close()
    todo = have_pbp if args.force else [g for g in have_pbp if g not in have]
    log.info("%s: %d already in player_game_stats, %d to parse%s", season,
             len(have), len(todo), " (FORCE re-parse)" if args.force else "")
    if not todo:
        log.info("%s: nothing to do", season)
        return

    # ---- PARSE (no lock, no network) --------------------------------------
    now = dt.datetime.now(dt.timezone.utc)
    rows, per_game, bad = [], {}, []
    t0 = time.monotonic()
    for i, gid in enumerate(todo):
        try:
            players = possessions.parse_box(boxes[gid])
            possessions.apply_pbp_zones(pbps[gid], players)
        except Exception as e:  # noqa: BLE001
            log.warning("%s parse failed: %s", gid, str(e)[:120])
            bad.append(gid)
            continue
        za = fga = 0
        for pid, c in players.items():
            rows.append([gid, pid, c["team_id"], c["seconds"],
                         c["fga"], c["fgm"], c["fg3a"], c["fg3m"], c["fta"], c["ftm"],
                         c["ast"], c["tov"], c["oreb"], c["dreb"], c["stl"], c["blk"],
                         c["pf"], c["pts"], c["plus_minus"],
                         c["rima"], c["rimm"], c["mida"], c["midm"], c["thra"], c["thrm"],
                         c["shooting_fouls"], now])
            za += c["rima"] + c["mida"] + c["thra"]
            fga += c["fga"]
        per_game[gid] = (za, fga)
        if (i + 1) % 400 == 0:
            log.info("%s parsed %d/%d (%.0f s)", season, i + 1, len(todo),
                     time.monotonic() - t0)

    n_games = len(per_game)
    tot_za = sum(v[0] for v in per_game.values())
    tot_fga = sum(v[1] for v in per_game.values())
    dead = [g for g, (za, fga) in per_game.items() if fga > 0 and za == 0]
    ratio = tot_za / tot_fga if tot_fga else 0.0
    log.info("%s PARSED %d games / %d rows in %.0f s; zone att %d vs fga %d "
             "(ratio %.4f); zone-dead games %d; parse failures %d",
             season, n_games, len(rows), time.monotonic() - t0, tot_za, tot_fga,
             ratio, len(dead), len(bad))

    # ---- ZONE-DEAD GATE, BEFORE ANY WRITE ---------------------------------
    if not rows:
        raise SystemExit(f"REFUSED: {season} parsed 0 rows")
    if tot_fga > 0 and tot_za == 0:
        raise SystemExit(
            f"REFUSED: {season} is ZONE-DEAD — {tot_fga} field-goal attempts "
            f"and ZERO zone attempts. PBP landed empty; nothing written.")
    if len(dead) > ZONE_DEAD_MAX_SHARE * max(n_games, 1):
        raise SystemExit(
            f"REFUSED: {season} has {len(dead)}/{n_games} zone-dead games "
            f"(> {ZONE_DEAD_MAX_SHARE:.0%}); nothing written. First 10: {dead[:10]}")
    if ratio < ZONE_RATIO_MIN:
        raise SystemExit(
            f"REFUSED: {season} zone/fga ratio {ratio:.4f} < {ZONE_RATIO_MIN}; "
            "PBP is partially empty. Nothing written.")
    if args.dry_run:
        log.info("%s DRY RUN — gate PASSED, %d rows NOT written", season, len(rows))
        return

    # ---- WRITE (short lock windows, yielded) ------------------------------
    B = 60000
    for j in range(0, len(rows), B):
        chunk = rows[j:j + B]
        tw = time.monotonic()
        possessions._write_rows(lambda: yielding_connect(f"pgs {season}"), chunk)
        log.info("%s wrote %d rows in %.2f s (lock released)", season,
                 len(chunk), time.monotonic() - tw)
    log.info("%s DONE: %d games / %d rows landed", season, n_games, len(rows))


# --------------------------------------------------------------- verify ----

def cmd_verify(args) -> None:
    con = connect(read_only=True)
    q = """
    SELECT g.season,
           COUNT(DISTINCT g.game_id)                                   AS games_002,
           COUNT(DISTINCT p.game_id)                                   AS pgs_games,
           COALESCE(SUM(p.n), 0)                                       AS pgs_rows
    FROM (SELECT DISTINCT season, game_id FROM nba_games
          WHERE substr(game_id,1,3)='002') g
    LEFT JOIN (SELECT game_id, COUNT(*) n FROM player_game_stats GROUP BY 1) p
           ON p.game_id = g.game_id
    GROUP BY 1 ORDER BY 1
    """
    rows = con.execute(q).fetchall()
    print(f"{'season':9} {'002':>6} {'exp':>6} {'d':>5} {'pgsG':>6} {'miss':>5} {'pgsRows':>9}")
    for s, g002, pg, prow in rows:
        if args.season and s != args.season:
            continue
        exp = expected_002(s)
        print(f"{s:9} {g002:6d} {exp:6d} {g002-exp:5d} {pg:6d} {g002-pg:5d} {int(prow):9d}")
    # zone completeness: games whose PBP zone attempts are all zero
    z = con.execute("""
        SELECT g.season, COUNT(*) FROM (
          SELECT game_id, SUM(rima+mida+thra) za, SUM(fga) fga
          FROM player_game_stats GROUP BY 1) t
        JOIN (SELECT DISTINCT season, game_id FROM nba_games) g USING (game_id)
        WHERE t.za = 0 AND t.fga > 0 GROUP BY 1 ORDER BY 1""").fetchall()
    print("\nzone-dead games (fga>0 but no PBP zones):", z or "none")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--from", default="1996-97")
    p.add_argument("--to", default="2018-19")
    p.add_argument("--out", default="data/source_depth_probe.json")
    p.set_defaults(func=cmd_probe)
    p = sub.add_parser("pull")
    p.add_argument("--season", required=True)
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--only-regular", action="store_true")
    p.add_argument("--no-rotations", action="store_true",
                   help="skip GameRotation entirely (it does not exist for "
                        "pre-2018-19 games; saves one call per game)")
    p.set_defaults(func=cmd_pull)
    p = sub.add_parser("pgs", help="parse cached box+PBP into player_game_stats "
                                   "for one season (NO network)")
    p.add_argument("--season", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="parse and run the zone-dead gate, write nothing")
    p.add_argument("--force", action="store_true",
                   help="re-parse games already in player_game_stats "
                        "(INSERT OR REPLACE); use after a parser fix")
    p.set_defaults(func=cmd_pgs)
    p = sub.add_parser("verify")
    p.add_argument("--season", default=None)
    p.set_defaults(func=cmd_verify)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
