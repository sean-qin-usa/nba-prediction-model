#!/usr/bin/env python3
"""D233 — MULTI-STATE AVAILABILITY: does a player who PLAYS after being listed
Questionable play his normal minutes?

THE GAP IN THE SHIPPED LEG.  D201/D202 replaced the hard out-set with a single
probability per player, and the composition leg spends it as

    weight = 1 - P(out)

applied to `talent x trailing_minutes`.  That treats availability as binary with
a probability attached: a player is either fully himself or absent.  It has no
way to express the obvious third state — he plays, but on a minutes restriction.

If players returning from doubt systematically log FEWER minutes than their
trailing norm, then `1 - P(out)` over-credits them, and it does so precisely on
the games where availability is in question, which are the games that move the
line.

WHAT THIS MEASURES.  For every player-game where the player ACTUALLY PLAYED,
conditional on the last status published strictly BEFORE game day:

    attenuation = actual_minutes / trailing_minutes

If attenuation is ~1.0 for every status, the binary-with-probability form is
adequate and there is nothing here.  If it falls with status severity, the leg
needs a minutes multiplier and this quantifies it.

BET-TIME CLEAN.  Status is the last one published before game day (the D199
as-of-open rule); trailing minutes are strictly prior games.  Nothing here reads
tonight's box score except the outcome being measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

from nbapred.db import connect                                    # noqa: E402
from nbapred import teams as T                                    # noqa: E402

HALF_LIFE = 8.0
MIN_HIST = 10


def load():
    con = connect(read_only=True)
    stats_df = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds,
               g.game_date, g.season, g.team_abbrev
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, team_id, season, game_date, team_abbrev
              FROM nba_games WHERE game_id LIKE '002%') g
          ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE s.game_id LIKE '002%'
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    rep = con.execute("""
        SELECT i.report_date, i.game_date, i.team, i.status, p.player_id
        FROM injury_reports_pit i
        JOIN (SELECT player_id, lower(first_name||' '||last_name) fn
              FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '
                          ||trim(split_part(i.player,',',1)))
    """).fetchdf()
    con.close()
    return stats_df, rep


def asof_status(rep: pd.DataFrame) -> pd.DataFrame:
    """Last status published STRICTLY BEFORE game day, per (player, game_date).

    Same carry-forward rule as the production out-set (D199): most game days have
    no advance report of their own, so what a bettor knows at the open is the
    most recent published status, not a same-day one.
    """
    rep = rep.copy()
    rep["report_date"] = pd.to_datetime(rep["report_date"])
    rep["game_date"] = pd.to_datetime(rep["game_date"])
    rep = rep[rep["report_date"] < rep["game_date"]]
    rep = rep.sort_values(["player_id", "game_date", "report_date"])
    return (rep.groupby(["player_id", "game_date"], as_index=False)
               .last()[["player_id", "game_date", "status"]])


def main():
    st, rep = load()
    st["game_date"] = pd.to_datetime(st["game_date"])
    st["mins"] = st["seconds"] / 60.0
    a = 1 - 0.5 ** (1 / HALF_LIFE)
    g = st.groupby("player_id", sort=False)
    # trailing minutes over games the player PLAYED, strictly prior
    played = st[st["mins"] > 0].copy()
    pg = played.groupby("player_id", sort=False)
    played["tr_mins"] = pg["mins"].transform(
        lambda s: s.ewm(alpha=a, adjust=False).mean().shift(1))
    played["n_hist"] = pg.cumcount()
    played = played[(played["n_hist"] >= MIN_HIST) & (played["tr_mins"] > 8.0)]

    s = asof_status(rep)
    f = played.merge(s, on=["player_id", "game_date"], how="left")
    f["status"] = f["status"].fillna("(none)")
    f["atten"] = f["mins"] / f["tr_mins"]
    f = f[f["season"] >= "2018-19"]
    print(f"frame {len(f):,} player-games that were PLAYED, "
          f"{f.player_id.nunique():,} players, {f.season.min()}..{f.season.max()}")

    print("\n=== MINUTES ATTENUATION BY LAST STATUS BEFORE GAME DAY ===")
    print("    (players who PLAYED; 1.00 = full trailing minutes)")
    print(f"\n{'status':14} {'n':>8} {'atten':>8} {'sd':>7} {'95% CI':>18} "
          f"{'mins':>7} {'trail':>7}")
    order = ["(none)", "Available", "Probable", "Questionable", "Doubtful", "Out"]
    rows = []
    for k in order:
        sub = f[f["status"] == k]
        if len(sub) < 50:
            continue
        m = sub["atten"].mean()
        sd = sub["atten"].std()
        se = sd / np.sqrt(len(sub))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        rows.append(dict(status=k, n=int(len(sub)), atten=float(m),
                         sd=float(sd), lo=float(lo), hi=float(hi),
                         mins=float(sub["mins"].mean()),
                         trail=float(sub["tr_mins"].mean())))
        print(f"{k:14} {len(sub):8,} {m:8.4f} {sd:7.3f} "
              f"[{lo:6.4f},{hi:6.4f}] {sub['mins'].mean():7.2f} "
              f"{sub['tr_mins'].mean():7.2f}")

    base = next(r for r in rows if r["status"] == "(none)")
    print(f"\n  reference = '(none)' at {base['atten']:.4f}")
    for r in rows:
        if r["status"] == "(none)":
            continue
        d = r["atten"] - base["atten"]
        print(f"  {r['status']:14} {d:+.4f}  ->  a player who plays after this "
              f"status logs {100*d:+.1f}% of his usual minutes vs an unlisted one")

    # season stability of the Questionable effect — the one that matters most
    print("\n=== IS THE QUESTIONABLE EFFECT STABLE ACROSS SEASONS? ===")
    per = []
    for ssn, sub in f.groupby("season"):
        q = sub[sub["status"] == "Questionable"]
        n0 = sub[sub["status"] == "(none)"]
        if len(q) < 100 or len(n0) < 100:
            continue
        d = q["atten"].mean() - n0["atten"].mean()
        per.append(dict(season=ssn, n_q=int(len(q)), d=float(d)))
        print(f"  {ssn}  n_Q={len(q):5,}  delta {d:+.4f}")
    dd = np.array([p["d"] for p in per])
    k = len(dd)
    se = dd.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    print(f"\n  season-clustered mean {dd.mean():+.4f} "
          f"95% CI [{dd.mean()-tc*se:+.4f}, {dd.mean()+tc*se:+.4f}] "
          f"same sign {int((dd<0).sum() if dd.mean()<0 else (dd>0).sum())}/{k}")
    json.dump({"by_status": rows, "per_season_questionable": per},
              open(ROOT / "data" / "d233_multistate.json", "w"), default=float)
    print("\nwrote data/d233_multistate.json")


if __name__ == "__main__":
    main()
