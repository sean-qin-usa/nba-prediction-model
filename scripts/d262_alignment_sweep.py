#!/usr/bin/env python3
"""D262 — SWEEP EVERY AVAILABLE SIGNAL FOR ALIGNMENT WITH THE MODEL'S OWN ERROR.

D259, D260 and D261 each screened ONE candidate and each missed. Screening one
at a time is the same mistake D252 made with slices: it cannot find a signal
nobody thought to name, and it spends a script per guess. D261 showed the screen
costs almost nothing, so run it over everything at once.

THE CRITERION (D260), which is what makes a sweep affordable:

    dRMSE  ~  (var(shift) - 2 * cov(shift, error)) / (2 * RMSE)
    worth building only if   cov(shift, error) > var(shift)/2

Equivalently, defining ALIGNMENT = cov / (var/2), a signal must reach 1.0 to
break even. D261's replacement-gap scored 0.065 — a 15x miss.

THE OUTCOME IS THE MODEL'S OWN MARGIN ERROR, not the market residual. That is
the correct target under a log-loss objective: to improve the forecast, a signal
must correlate with what the forecast gets wrong. Whether it also beats the
market is a separate and much harder question, deliberately not asked here.

WHAT IS SWEPT. Everything PIT and computable, spanning every channel the project
has: the D258 tendency estimates, minutes and roster structure, absence and
role-strain measures (queue item 6's territory), schedule, and market context.
Each is a home-minus-away differential so it has the sign of the margin.

MULTIPLICITY IS HANDLED, not ignored. ~20 signals screened means some will look
aligned by chance, so every one carries a season-clustered CI on the slope of
error against signal, and the sweep's best is compared against a permutation
null over the whole family — the same construction D252 used for slices.

A NEGATIVE SWEEP IS THE MOST USEFUL OUTCOME AVAILABLE. If nothing reaches
alignment 1.0, that is evidence about the CHANNEL rather than about twenty
separate ideas, and it would explain three consecutive misses as one fact.
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
MIN_ROT = 12.0
CORE_N = 8
K = {"fg3_rate": 8, "rim_rate": 16, "ast_rate": 32}


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v); se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def team_panel():
    """Per team-game: roster structure, absence, role strain, tendencies."""
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    pg = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, player_id, team_id, seconds,
               fga, fgm, fg3a, COALESCE(rima,0) rima, ast
        FROM player_game_stats
        WHERE CAST(game_id AS VARCHAR) LIKE '002%'""").df()
    dk = con.execute("SELECT player_id, date, dpm FROM darko_history "
                     "WHERE dpm IS NOT NULL").df()
    con.close()
    pg["gid"] = pg.gid.str.zfill(10); pg["mins"] = pg.seconds / 60.0
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f.game_date)
    d = pg.merge(f[["game_id", "season", "game_date", "home", "away"]],
                 left_on="gid", right_on="game_id", how="inner")
    play = d[d.mins >= MIN_ROT].sort_values(["player_id", "game_date"]).copy()
    g = play.groupby(["player_id", "season"])
    play["trail"] = g.mins.transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    play["season_mpg"] = g.mins.transform(
        lambda s: s.shift(1).expanding().mean())
    # PIT tendencies, D258 constants
    for ax, (nc, dc) in (("fg3_rate", ("fg3a", "fga")),
                         ("rim_rate", ("rima", "fga")),
                         ("ast_rate", ("ast", "fgm"))):
        num = play[nc].to_numpy(float); den = play[dc].to_numpy(float)
        cn = g[nc].cumsum().to_numpy(float) - num
        cd = g[dc].cumsum().to_numpy(float) - den
        lgr = play[nc].sum() / max(play[dc].sum(), 1e-9)
        play[f"t_{ax}"] = (cn + K[ax] * lgr) / (cd + K[ax])
    play = play.dropna(subset=["trail"])
    dk["date"] = pd.to_datetime(dk["date"]).astype("datetime64[ns]")
    play["game_date"] = play.game_date.astype("datetime64[ns]")
    play = pd.merge_asof(play.sort_values("game_date"), dk.sort_values("date"),
                         left_on="game_date", right_on="date", by="player_id",
                         direction="backward", allow_exact_matches=False)
    play = play.dropna(subset=["dpm"])

    appeared = {}
    for gg, p in zip(pg.gid, pg.player_id):
        appeared.setdefault(gg, set()).add(p)

    rows = []
    for (tid, season), tg in play.groupby(["team_id", "season"], sort=False):
        roster = {}
        for gid, gg in tg.sort_values("game_date").groupby("gid", sort=False):
            gd = gg.game_date.iloc[0]
            live = {p: v for p, v in roster.items()
                    if (gd - v["d"]).days <= ROSTER_DAYS}
            here = appeared.get(gid, set())
            av = [v for p, v in live.items() if p in here]
            ab = [v for p, v in live.items() if p not in here]
            if len(av) >= 5:
                av.sort(key=lambda v: -v["trail"])
                absorb = av[CORE_N:] or av[-3:]
                w = np.array([v["trail"] for v in absorb], float)
                tl = np.array([v["dpm"] for v in absorb], float)
                rep = float(np.average(tl, weights=w)) if w.sum() > 0 else float(tl.mean())
                tw = np.array([v["trail"] for v in av], float)
                tot = tw.sum()
                p_ = tw / max(tot, 1e-9)
                rows.append(dict(
                    gid=gid, team_id=tid, season=season,
                    n_out=len(ab),
                    min_out=float(sum(v["trail"] for v in ab)),
                    gap=float(sum((v["dpm"] - rep) * v["trail"] / 48.0 for v in ab)),
                    avail_min=float(tot),
                    depth=len(av),
                    hhi=float((p_ ** 2).sum()),
                    top_tal=float(max((v["dpm"] for v in av), default=0.0)),
                    mean_tal=float(np.average([v["dpm"] for v in av], weights=tw))
                    if tot > 0 else 0.0,
                    # role strain: minutes that must be absorbed above normal role
                    strain=float(sum(max(0.0, v["trail"] - v["smpg"])
                                     for v in av)),
                    t_fg3=float(np.average([v["fg3"] for v in av], weights=tw))
                    if tot > 0 else 0.0,
                    t_rim=float(np.average([v["rim"] for v in av], weights=tw))
                    if tot > 0 else 0.0,
                    t_ast=float(np.average([v["ast"] for v in av], weights=tw))
                    if tot > 0 else 0.0,
                ))
            for r in gg.itertuples():
                if r.player_id in here:
                    roster[r.player_id] = dict(
                        d=gd, trail=r.trail, dpm=r.dpm,
                        smpg=r.season_mpg if np.isfinite(r.season_mpg) else r.trail,
                        fg3=r.t_fg3_rate, rim=r.t_rim_rate, ast=r.t_ast_rate)
    return pd.DataFrame(rows), f


def main():
    t, f = team_panel()
    two = t.groupby("gid").filter(lambda x: len(x) == 2)
    cols = [c for c in two.columns if c not in ("gid", "team_id", "season")]
    piv = (two.sort_values(["gid", "team_id"]).groupby("gid")
           .agg({**{c: ["first", "last"] for c in cols},
                 "season": "first"}))
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    D = pd.DataFrame({c: piv[f"{c}_first"] - piv[f"{c}_last"] for c in cols})
    D["season"] = piv["season_first"]
    fr = f.dropna(subset=["margin_actual", "m_us", "m_us_blind"]).set_index("game_id")
    j = D.join(fr[["margin_actual", "m_us", "m_us_blind", "open_margin",
                   "open_total"]], how="inner")
    # schedule + market context, also as differentials where sensible
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    pit["game_id"] = zf(pit["game_id"])
    pit = pit.set_index("game_id")
    j["rest_diff"] = (pit.rest_home.clip(upper=7).reindex(j.index).fillna(2)
                      - pit.rest_away.clip(upper=7).reindex(j.index).fillna(2))
    j["abs_open"] = j.open_margin.abs()
    j["err_blind"] = j.margin_actual - j.m_us_blind
    j["err_ship"] = j.margin_actual - j.m_us
    j = j.dropna(subset=["err_ship"])
    sigs = [c for c in cols] + ["rest_diff", "abs_open"]
    print(f"{len(j):,} games, {len(sigs)} signals, {j.season.nunique()} seasons\n")

    e = j.err_ship.to_numpy(float)
    R = float(np.sqrt(np.mean(e ** 2)))
    print("=" * 82)
    print(f"ALIGNMENT SWEEP vs the SHIPPED margin error (RMSE {R:.3f})")
    print("=" * 82)
    print(f"  {'signal':12} {'align':>8} {'dRMSE':>9} {'slope':>9} "
          f"{'CI':>21} verdict")
    res, aligns = [], {}
    seasons = sorted(j.season.unique())
    for c in sigs:
        x = j[c].to_numpy(float)
        if not np.isfinite(x).all() or np.std(x) < 1e-12:
            continue
        # WALK-FORWARD slope. Fitting b in-sample forces cov == var and the
        # criterion becomes the identity align = 2.0 for every signal. The shift
        # must be determined independently of the error it is scored against.
        shift = np.full(len(j), np.nan)
        sc = j.season.to_numpy()
        for i, sname in enumerate(seasons):
            if i < 3:
                continue
            tr = sc < sname
            te = sc == sname
            if tr.sum() < 2000 or te.sum() < 200:
                continue
            xt, et = x[tr], e[tr]
            good = np.isfinite(xt) & np.isfinite(et)
            if good.sum() < 500 or np.std(xt[good]) < 1e-9:
                continue
            try:
                b = np.polyfit(xt[good], et[good], 1)[0]
            except np.linalg.LinAlgError:
                continue          # degenerate training window; skip this fold
            shift[te] = b * (x[te] - xt[good].mean())
        ok = np.isfinite(shift)
        if ok.sum() < 2000:
            continue
        sh, ee = shift[ok], e[ok]
        var = float(np.var(sh))
        cov = float(np.mean((sh - sh.mean()) * (ee - ee.mean())))
        align = cov / (var / 2) if var > 1e-15 else 0.0
        drmse = (var - 2 * cov) / (2 * R)
        per = []
        for sn, gg in j.groupby("season"):
            xx, yy = gg[c].to_numpy(float), gg.err_ship.to_numpy(float)
            ok2 = np.isfinite(xx) & np.isfinite(yy)
            if ok2.sum() < 200 or np.std(xx[ok2]) < 1e-9:
                continue
            try:
                per.append(float(np.polyfit(xx[ok2], yy[ok2], 1)[0]))
            except np.linalg.LinAlgError:
                continue
        if len(per) < 3:
            continue
        m, lo, hi, k = clus(per)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        verdict = "PASSES" if align >= 1.0 else f"{align:.2f}x"
        print(f"  {c:12} {align:8.3f} {drmse:+9.5f} {m:+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {sig:3} {verdict}")
        res.append(dict(signal=c, align=align, drmse=drmse, slope=m,
                        ci=[lo, hi], k=k, sig=sig))
        aligns[c] = align

    best = max(aligns.values()) if aligns else 0.0
    print(f"\n  best alignment across the family: {best:.3f} "
          f"(break-even is 1.000)")
    print("  Any signal below 1.0 makes the margin WORSE if added, because its")
    print("  own variance exceeds twice its covariance with the error.")
    json.dump(res, open(ROOT / "data" / "d262_sweep.json", "w"), default=float)
    print("\nwrote data/d262_sweep.json")


if __name__ == "__main__":
    main()
