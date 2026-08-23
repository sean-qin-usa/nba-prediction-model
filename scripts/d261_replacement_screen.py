#!/usr/bin/env python3
"""D261 — SCREEN BEFORE BUILD: is a replacement-quality signal ALIGNED with the
model's error?

D260 established the criterion that should have been applied to the last three
input-side projects before any of them was built:

    dRMSE  ~  (var(shift) - 2 * cov(shift, error)) / (2 * RMSE)

A candidate input change is worth nothing unless `cov(shift, error) > var/2`,
and is actively HARMFUL below that. Both sides are computable from a cheap
version of the signal, so the screen costs a fraction of the build.

THE CANDIDATE. `composition.py` drops absent players (`if pid in out: continue`)
and never reallocates their minutes; every remaining player is carried at his
unchanged trailing rate. D232 prices the leftover with a FLAT per-absence charge,
`margin += beta * (E[outs_home] - E[outs_away])`, beta = -0.8284 blind. Flat is
the suspicious part: a star's absence and a tenth man's absence are charged
identically, when the real cost should scale with the QUALITY GAP between who is
missing and who absorbs the minutes.

    gap_team = sum over absent rotation players of
                   (talent_absent - talent_replacement) * trail_min / 48
    signal   = gap_home - gap_away

`talent_replacement` is the minutes-weighted talent of the available players
BELOW the rotation core -- the people who actually inherit minutes. This is a
deliberately crude version: the screen asks whether the mechanism points the
right way at all, not whether this particular parameterisation is optimal.

THE TEST MUST BE INCREMENTAL. D232's flat term is already shipped, so the signal
is residualised against the outs differential before alignment is measured.
Otherwise the screen would rediscover D232 and call it new.

ABSENCE IS REALISED NON-APPEARANCE, which buys all 19 seasons instead of the 7
the injury report covers. That is legitimate here because absence is known
before tip and is only used to CONDITION, never as a forecast input -- but a
shipped version would have to read the report, so PIT coverage would drop to
2019-20+. Noted rather than glossed.
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

ROSTER_DAYS = 12
MIN_ROT = 12.0        # a rotation player, matching the shipped floor
CORE_N = 8            # top-8 by trailing minutes = the core; below = absorbers


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def main():
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    pg = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, player_id, team_id, seconds
        FROM player_game_stats
        WHERE CAST(game_id AS VARCHAR) LIKE '002%'""").df()
    dk = con.execute("SELECT player_id, date, dpm FROM darko_history "
                     "WHERE dpm IS NOT NULL").df()
    con.close()
    pg["gid"] = pg.gid.str.zfill(10)
    pg["mins"] = pg.seconds / 60.0
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f.game_date)
    d = pg.merge(f[["game_id", "season", "game_date", "home", "away"]],
                 left_on="gid", right_on="game_id", how="inner")

    # trailing minutes over games PLAYED, PIT
    played = d[d.mins >= MIN_ROT].sort_values(["player_id", "game_date"])
    played["trail"] = (played.groupby(["player_id", "season"]).mins
                       .transform(lambda s: s.shift(1)
                                  .rolling(10, min_periods=3).mean()))
    played["last_played"] = played.groupby(["player_id", "season"]).game_date.shift(1)
    tr = played[["gid", "player_id", "team_id", "season", "game_date",
                 "trail", "last_played", "mins"]].dropna(subset=["trail"])

    # ---- ROSTER, carried forward per team (the composition.py rule) -------
    # The first pass built the rotation only from games a player PLAYED, so an
    # absent player had no row and `played_now` was always True: the signal came
    # out identically zero. The roster must be maintained INDEPENDENTLY of
    # tonight's box score -- a player is on it if he appeared for this team
    # within ROSTER_DAYS, whether or not he plays tonight.
    dk["date"] = pd.to_datetime(dk["date"]).astype("datetime64[ns]")
    tr["game_date"] = tr.game_date.astype("datetime64[ns]")
    tr = pd.merge_asof(tr.sort_values("game_date"), dk.sort_values("date"),
                       left_on="game_date", right_on="date", by="player_id",
                       direction="backward", allow_exact_matches=False)
    tr = tr.dropna(subset=["dpm"])
    print(f"{len(tr):,} player-game rows with PIT talent and trailing minutes")

    appeared = {}
    for g, p in zip(pg.gid, pg.player_id):
        appeared.setdefault(g, set()).add(p)

    rows = []
    for (tid, season), tg in tr.groupby(["team_id", "season"], sort=False):
        tg = tg.sort_values("game_date")
        roster = {}                      # pid -> (last_date, trail, dpm)
        for gid, gg in tg.groupby("gid", sort=False):
            gdate = gg.game_date.iloc[0]
            # roster as of tonight, from appearances STRICTLY BEFORE
            live = {p: v for p, v in roster.items()
                    if (gdate - v[0]).days <= ROSTER_DAYS}
            if len(live) >= 5:
                here = appeared.get(gid, set())
                av = [(p, v) for p, v in live.items() if p in here]
                ab = [(p, v) for p, v in live.items() if p not in here]
                if len(av) >= 5:
                    av.sort(key=lambda kv: -kv[1][1])
                    absorb = av[CORE_N:] or av[-3:]
                    w = np.array([v[1] for _, v in absorb], float)
                    tl = np.array([v[2] for _, v in absorb], float)
                    rep = float(np.average(tl, weights=w)) if w.sum() > 0 \
                        else float(tl.mean())
                    gap = float(sum((v[2] - rep) * v[1] / 48.0 for _, v in ab))
                    rows.append(dict(gid=gid, team_id=tid, season=season,
                                     gap=gap, n_out=len(ab),
                                     min_out=float(sum(v[1] for _, v in ab))))
            # now record tonight's appearances for future games
            for r in gg.itertuples():
                if r.player_id in appeared.get(gid, set()):
                    roster[r.player_id] = (gdate, r.trail, r.dpm)

    t = pd.DataFrame(rows)
    two = t.groupby("gid").filter(lambda x: len(x) == 2)
    print(f"{two.gid.nunique():,} games with both teams")

    # orient to home/away using the frame
    tm = pg.groupby(["gid", "team_id"]).size().reset_index()[["gid", "team_id"]]
    order = (two.sort_values(["gid", "team_id"]).groupby("gid")
             .agg(g1=("gap", "first"), g2=("gap", "last"),
                  n1=("n_out", "first"), n2=("n_out", "last"),
                  season=("season", "first")))
    # sign is arbitrary per game; calibrate against the model error by |corr|
    fr = f.dropna(subset=["margin_actual", "m_us", "m_us_blind"]).set_index("game_id")
    j = order.join(fr[["margin_actual", "m_us", "m_us_blind", "open_margin"]],
                   how="inner")
    j["sig"] = j.g1 - j.g2
    j["outs"] = j.n1 - j.n2
    j["err_blind"] = j.margin_actual - j.m_us_blind
    j["err_ship"] = j.margin_actual - j.m_us
    print(f"{len(j):,} games joined to a model error\n")

    print("=" * 74)
    print("THE D260 SCREEN — is the replacement-gap signal ALIGNED?")
    print("=" * 74)
    out = {}
    for ecol, lab, rmse_ref in (("err_blind", "MARKET-BLIND margin", None),
                                ("err_ship", "SHIPPED offset margin", None)):
        e = j[ecol].to_numpy(float)
        # residualise the signal against D232's flat outs term -- the increment
        X = np.column_stack([np.ones(len(j)), j.outs.to_numpy(float)])
        b = np.linalg.lstsq(X, j.sig.to_numpy(float), rcond=None)[0]
        s_res = j.sig.to_numpy(float) - X @ b
        R = float(np.sqrt(np.mean(e ** 2)))
        var = float(np.var(s_res))
        cov = float(np.mean((s_res - s_res.mean()) * (e - e.mean())))
        drmse = (var - 2 * cov) / (2 * R)
        print(f"\n  --- {lab} (RMSE {R:.3f}) ---")
        print(f"    var(signal)          {var:.5f}")
        print(f"    cov(signal, error)   {cov:.5f}")
        print(f"    threshold var/2      {var/2:.5f}   "
              f"{'PASSES' if cov > var/2 else 'FAILS'}")
        print(f"    implied dRMSE        {drmse:+.5f} pts  "
              f"({'improvement' if drmse < 0 else 'HARM'})")
        print(f"    implied corr          {cov/(np.sqrt(var)*R):+.4f}")
        # season-clustered slope of error on signal, the direct check
        per = []
        for s, gg in j.groupby("season"):
            if len(gg) < 200:
                continue
            xx = gg.sig - np.polyval(np.polyfit(gg.outs, gg.sig, 1), gg.outs)
            per.append(float(np.polyfit(xx, gg[ecol], 1)[0]))
        v = np.array(per); se = v.std(ddof=1) / np.sqrt(len(v))
        tc = stats.t.ppf(.975, len(v) - 1)
        print(f"    slope of error on signal: {v.mean():+.4f} "
              f"CI [{v.mean()-tc*se:+.4f}, {v.mean()+tc*se:+.4f}] "
              f"k={len(v)}  {'SIG' if v.mean()-tc*se>0 or v.mean()+tc*se<0 else 'ns'}")
        out[lab] = dict(var=var, cov=cov, drmse=drmse,
                        slope=float(v.mean()),
                        ci=[float(v.mean()-tc*se), float(v.mean()+tc*se)])

    print("\n" + "=" * 74)
    print("  A positive slope means the model UNDER-charges for a high-quality")
    print("  absence, which is what a replacement allocator would fix. cov must")
    print("  clear var/2 for the fix to be worth anything at the outcome.")
    json.dump(out, open(ROOT / "data" / "d261_screen.json", "w"), default=float)
    print("\nwrote data/d261_screen.json")


if __name__ == "__main__":
    main()
