#!/usr/bin/env python3
"""D239 — IS THE SIGNAL RANKABLE? The reviewer's fundamental diagnostic, plus
continuous regime interactions on the CLV endpoint. Diagnostic; nothing ships.

THE TWO REGRESSIONS (their spec, verbatim):

    signed opener residual   sign(d) * (Y - O)  =  a + b|d| + e
    signed CLV               sign(d) * (C - O)  =  a + b|d| + e

with d = m_offset - m_open. The first asks whether larger claimed edges produce
larger realised advantages against the opener; the second asks the same against
the CLOSE, which is the low-variance endpoint (D147: open->close movement is
17.1% predictable, and the market-blind model is what predicts it, so the CLV
slope here is that finding restated per game).

THE INTERACTION MODEL replaces D238's quartile buckets, which the reviewer is
right to reject — arbitrary boundaries, information thrown away:

    signed_CLV = a + b|d| + g*z + delta*(|d| x z)

delta is the trust modifier: does a claimed edge become MORE reliable in state
z? The |d| main effect is the control the reviewer demands, so an interaction
cannot "win" merely by selecting larger edges. THREE states, pre-listed as the
whole family (Bonferroni 0.0167 applies):

    T_roster   roster-transition factor: 1 - returning-minutes share, the
               MECHANISM behind D238's early-season hint, replacing the
               calendar label as the reviewer proposes
    mkt_ll     trailing market log loss (the "wonky market" state, again)
    tot_eo     total expected absences in the game (information-state block)

All states strictly prior. Roster share is season-to-date, shifted one game.
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

FROM = "2019-20"
WIN = 200
MIN_PRIOR_MIN = 300 * 60      # seconds: prior-season minutes to count as returning


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def next_season(s):
    y = int(s[:4]) + 1
    return f"{y}-{(y + 1) % 100:02d}"


def clustered(vals):
    v = np.asarray(vals, float)
    k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def roster_transition() -> pd.DataFrame:
    """Per game: 1 - mean(returning-minutes share of the two teams), strictly
    prior (season-to-date, shifted one team-game)."""
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds, g.season, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season, game_date FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id)
        WHERE s.seconds > 0""").fetchdf()
    home = con.execute("""
        SELECT game_id, team_id, is_home FROM nba_games
        WHERE game_id LIKE '002%'""").fetchdf()
    con.close()
    pm = (pg.groupby(["season", "team_id", "player_id"], as_index=False)
            .seconds.sum())
    pm["season"] = pm["season"].map(next_season)     # membership applies NEXT season
    pm["is_ret"] = pm["seconds"] >= MIN_PRIOR_MIN
    pg = pg.merge(pm[["season", "team_id", "player_id", "is_ret"]],
                  on=["season", "team_id", "player_id"], how="left")
    pg["is_ret"] = pg["is_ret"].fillna(False)
    pg["ret_sec"] = pg["seconds"].where(pg["is_ret"], 0.0)
    tg = (pg.groupby(["game_id", "team_id", "season", "game_date"], as_index=False)
            .agg(tot=("seconds", "sum"), ret=("ret_sec", "sum"))
            .sort_values(["season", "team_id", "game_date"]))
    g = tg.groupby(["season", "team_id"])
    tg["cum_tot"] = g["tot"].cumsum() - tg["tot"]     # strictly prior
    tg["cum_ret"] = g["ret"].cumsum() - tg["ret"]
    tg["share"] = np.where(tg["cum_tot"] > 0, tg["cum_ret"] / tg["cum_tot"], np.nan)
    tg = tg.merge(home, on=["game_id", "team_id"])
    piv = tg.pivot_table(index="game_id", columns="is_home", values="share")
    piv.columns = ["share_away", "share_home"]
    out = piv.reset_index()
    out["game_id"] = zf(out["game_id"])
    out["T_roster"] = 1.0 - 0.5 * (out["share_home"] + out["share_away"])
    return out[["game_id", "T_roster"]]


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f["season"] >= "2017-18"].copy()
    f["game_date"] = pd.to_datetime(f["game_date"])
    f["game_id"] = zf(f["game_id"])
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual", "m_us"])
    f = f.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    f["d"] = f["m_us"] - f["open_margin"]
    f = f[f["d"].abs() > 1e-9].copy()
    f["absd"] = f["d"].abs()
    sgn = np.sign(f["d"])
    f["sres"] = sgn * (f["margin_actual"] - f["open_margin"])
    f["sclv"] = sgn * (f["close_margin"] - f["open_margin"])

    # trailing market LL state (as D237/D238)
    p = 1 / (1 + np.exp(-f["open_margin"] / 6.96))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    y = (f["margin_actual"] > 0).astype(float)
    f["llo"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    per = f.groupby("game_date").agg(ll=("llo", "mean"), n=("llo", "size")).reset_index()
    buf, rows = [], []
    for r in per.sort_values("game_date").itertuples():
        rows.append((r.game_date, np.mean(buf[-WIN:]) if len(buf) >= 50 else np.nan))
        buf += [r.ll] * int(r.n)
    f = f.merge(pd.DataFrame(rows, columns=["game_date", "mkt_ll"]), on="game_date")

    cap = pd.read_csv(ROOT / "data" / "capstone_2019_26.csv")
    cap["game_id"] = zf(cap["game_id"])
    f = f.merge(cap[["game_id", "eo_home", "eo_away"]], on="game_id", how="left")
    f["tot_eo"] = f["eo_home"] + f["eo_away"]
    f = f.merge(roster_transition(), on="game_id", how="left")

    g = f[f["season"] >= FROM].copy()
    print(f"frame {len(g):,} games {g.season.min()}..{g.season.max()}")
    print(f"coverage: mkt_ll {g.mkt_ll.notna().mean():.1%}  "
          f"tot_eo {g.tot_eo.notna().mean():.1%}  "
          f"T_roster {g.T_roster.notna().mean():.1%}")

    # ============ A. THE RANKABILITY REGRESSIONS ==========================
    print("\n" + "=" * 70)
    print("A. RANKABILITY — signed advantage vs |d|  (the fundamental test)")
    print("=" * 70)
    out = {}
    for col, name, unit in (("sres", "signed OPENER residual sign(d)*(Y-O)", "pts"),
                            ("sclv", "signed CLV sign(d)*(C-O)", "pts")):
        gg = g.dropna(subset=[col])
        b, a = np.polyfit(gg["absd"], gg[col], 1)
        per_season = []
        for s, sub in gg.groupby("season"):
            per_season.append(np.polyfit(sub["absd"], sub[col], 1)[0])
        m, lo, hi, t, K = clustered(per_season)
        print(f"\n  {name}")
        print(f"    mean {gg[col].mean():+.4f} {unit}   sd {gg[col].std():.3f}")
        print(f"    pooled:  {a:+.4f} {b:+.4f}*|d|")
        print(f"    season-clustered slope {m:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  t {t:+.2f}  K={K}  "
              f"{'SIGNIFICANT' if lo > 0 else 'ns'}")
        print(f"    per-season slopes: " +
              " ".join(f"{v:+.3f}" for v in per_season))
        q = pd.qcut(gg["absd"], 5, labels=False, duplicates="drop")
        prof = [(float(gg[q == k]["absd"].median()),
                 float(gg[q == k][col].mean())) for k in sorted(q.unique())]
        print(f"    by |d| quintile: " +
              "  ".join(f"|d|={a_:.2f}:{v:+.3f}" for a_, v in prof))
        out[col] = dict(slope=m, ci=[lo, hi], t=t, mean=float(gg[col].mean()),
                        per_season=[float(v) for v in per_season], profile=prof)

    both = out["sres"]["ci"][0] > 0 and out["sclv"]["ci"][0] > 0
    print(f"\n  VERDICT (reviewer's rubric): "
          f"{'BOTH slopes positive with CIs excluding zero -> the signal is RANKABLE' if both else 'see individual rows'}")

    # ============ B. CONTINUOUS TRUST INTERACTIONS ========================
    print("\n" + "=" * 70)
    print("B. TRUST INTERACTIONS on signed CLV — delta = does state z make a")
    print("   claimed edge more reliable?  (family of 3; Bonferroni 0.0167)")
    print("=" * 70)
    res = {}
    for zcol in ("T_roster", "mkt_ll", "tot_eo"):
        gg = g.dropna(subset=["sclv", zcol]).copy()
        mu, sd = gg[zcol].mean(), gg[zcol].std(ddof=1)
        gg["z"] = (gg[zcol] - mu) / sd
        deltas, gammas = [], []
        for s, sub in gg.groupby("season"):
            X = np.column_stack([np.ones(len(sub)), sub["absd"], sub["z"],
                                 sub["absd"] * sub["z"]])
            beta = np.linalg.lstsq(X, sub["sclv"].to_numpy(float), rcond=None)[0]
            gammas.append(beta[2]); deltas.append(beta[3])
        dm, dlo, dhi, dt, K = clustered(deltas)
        gm, glo, ghi, gt, _ = clustered(gammas)
        sig = dlo > 0 or dhi < 0
        print(f"\n  z = {zcol}  (n={len(gg):,}, standardised)")
        print(f"    gamma (level)        {gm:+.4f}  CI [{glo:+.4f}, {ghi:+.4f}]")
        print(f"    delta (|d| x z TRUST) {dm:+.4f}  CI [{dlo:+.4f}, {dhi:+.4f}]"
              f"  t {dt:+.2f}  {'SIG at 0.05' if sig else 'ns'}")
        res[zcol] = dict(delta=dm, ci=[dlo, dhi], t=dt, gamma=gm)
    n_sig = sum(1 for r in res.values() if r["ci"][0] > 0 or r["ci"][1] < 0)
    print(f"\n  family: {n_sig}/3 nominally significant; Bonferroni bar is "
          f"t ~ +/-3.2 at K-1=6 dof")

    json.dump({"rankability": out, "interactions": res},
              open(ROOT / "data" / "d239_rankability.json", "w"), default=float)
    print("\nwrote data/d239_rankability.json")


if __name__ == "__main__":
    main()
