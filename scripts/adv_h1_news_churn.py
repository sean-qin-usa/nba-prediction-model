"""H1 test (docs/ADVERSE_HYPOTHESES.md): news-timing selection.
If our divergences from the market are stale-info artifacts, opposite-side
games (p_us vs p_mkt on different sides) should sit on HIGH-news days:
more injury-report churn (status changes between the D-1 5PM report and the
game-day D 5PM report for the two teams) and more unresolved (Questionable)
statuses at 5PM game day.

Universe: capstone_pergame_tank games with 5PM reports available for both D
and D-1 (reports run 2023-10-24 .. 2025-12-21).
Comparisons: opposite-side vs same-side (raw + season/|p_mkt-0.5|-matched
pairs), same for the market-confident-we-not region; churn vs per-game
d = L_us - L_mkt.

Read-only DuckDB. Output: printed report only.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import duckdb

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
RNG = np.random.default_rng(11)

ABB2FULL = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets",
    "IND": "Indiana Pacers", "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"}


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_dmean_ci(a, b, B=4000):
    """CI for mean(a) - mean(b), independent bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = RNG.integers(0, len(a), (B, len(a)))
    ib = RNG.integers(0, len(b), (B, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def boot_paired_ci(d, B=4000):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def mannwhitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        return np.nan


def main():
    con = duckdb.connect(DB, read_only=True)
    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["gd"] = pd.to_datetime(df.game_date).dt.date
    rep = con.execute("""SELECT report_date, game_date, team, player, status
                         FROM injury_reports_pit""").df()
    con.close()
    rep["report_date"] = pd.to_datetime(rep.report_date).dt.date
    rep["game_date"] = pd.to_datetime(rep.game_date).dt.date
    report_days = set(rep.report_date.unique())

    # index: (report_date, game_date, team_full) -> {player: status}
    key = rep.groupby(["report_date", "game_date", "team"])
    smap = {k: dict(zip(v.player, v.status)) for k, v in key}

    rows = []
    for r in df.itertuples():
        D = r.gd
        Dm1 = D - dt.timedelta(days=1)
        if D not in report_days or Dm1 not in report_days:
            continue
        churn_any = churn_strict = quest = qd = listed = 0
        for ab in (r.home, r.away):
            tf = ABB2FULL[ab]
            cur = smap.get((D, D, tf), {})
            prv = smap.get((Dm1, D, tf), {})
            for p in set(cur) | set(prv):
                s1, s0 = cur.get(p), prv.get(p)
                if s1 != s0:
                    churn_any += 1
                    if s1 is not None and s0 is not None:
                        churn_strict += 1
            quest += sum(1 for s in cur.values() if s == "Questionable")
            qd += sum(1 for s in cur.values()
                      if s in ("Questionable", "Doubtful"))
            listed += len(cur)
        rows.append([r.game_id, r.season, D, r.home, r.away, r.y, r.p_us,
                     r.p_mkt, churn_any, churn_strict, quest, qd, listed])
    g = pd.DataFrame(rows, columns=[
        "game_id", "season", "gd", "home", "away", "y", "p_us", "p_mkt",
        "churn_any", "churn_strict", "quest", "qd", "listed"])
    g["d"] = ll(g.p_us.values, g.y.values) - ll(g.p_mkt.values, g.y.values)
    g["opp"] = (g.p_us - 0.5) * (g.p_mkt - 0.5) < 0
    g["mkconf"] = (np.abs(g.p_mkt - 0.5) > 0.35) & (np.abs(g.p_us - 0.5) <= 0.35)
    g["ext"] = np.abs(g.p_mkt - 0.5)
    print(f"games with D and D-1 reports: {len(g)} / {len(df)} "
          f"(opp {int(g.opp.sum())}, mkconf {int(g.mkconf.sum())})")
    print(f"churn_any mean {g.churn_any.mean():.2f}  "
          f"churn_strict {g.churn_strict.mean():.2f}  quest {g.quest.mean():.2f}")

    for region in ("opp", "mkconf"):
        sel = g[g[region]]
        oth = g[~g[region]]
        print("\n" + "=" * 72)
        print(f"REGION {region}: n={len(sel)} vs rest n={len(oth)}")
        for c in ("churn_any", "churn_strict", "quest", "qd", "listed"):
            lo, hi = boot_dmean_ci(sel[c], oth[c])
            pv = mannwhitney_p(sel[c], oth[c])
            print(f"  {c:13s} {sel[c].mean():6.2f} vs {oth[c].mean():6.2f} "
                  f"diff {sel[c].mean() - oth[c].mean():+5.2f} "
                  f"CI95({lo:+.2f},{hi:+.2f})  MW p={pv:.3f}")
        # matched pairs: same season, nearest |p_mkt-0.5|, no replacement
        used = set()
        pairs = []
        for r in sel.itertuples():
            cand = oth[(oth.season == r.season) & (~oth.index.isin(used))]
            if not len(cand):
                continue
            j = (cand.ext - r.ext).abs().idxmin()
            used.add(j)
            pairs.append((r.Index, j))
        A = sel.loc[[i for i, _ in pairs]]
        Bm = oth.loc[[j for _, j in pairs]]
        print(f"  matched pairs n={len(pairs)} "
              f"(|p_mkt-.5| bal: {A.ext.mean():.3f} vs {Bm.ext.mean():.3f})")
        for c in ("churn_any", "churn_strict", "quest", "qd"):
            dd = A[c].values - Bm[c].values
            lo, hi = boot_paired_ci(dd)
            print(f"  {c:13s} paired diff {dd.mean():+5.2f} "
                  f"CI95({lo:+.2f},{hi:+.2f})")

    # robustness of the matched result: (a) stratified (ext-decile x season)
    # estimator, order-free; (b) matching-order permutations
    print("\n" + "=" * 72)
    print("MATCHING ROBUSTNESS (opp region)")
    g["extbin"] = pd.qcut(g.ext, 10, labels=False, duplicates="drop")
    for c in ("churn_any", "churn_strict", "quest", "qd"):
        diffs, ws = [], []
        for (s, b), sub in g.groupby(["season", "extbin"]):
            a = sub[sub.opp][c]
            o = sub[~sub.opp][c]
            if len(a) >= 3 and len(o) >= 3:
                diffs.append(a.mean() - o.mean())
                ws.append(len(a))
        est = np.average(diffs, weights=ws)
        # bootstrap over games within strata
        bs = []
        for _ in range(1000):
            dd, ww = [], []
            for (s, b), sub in g.groupby(["season", "extbin"]):
                a = sub[sub.opp][c].values
                o = sub[~sub.opp][c].values
                if len(a) >= 3 and len(o) >= 3:
                    dd.append(a[RNG.integers(0, len(a), len(a))].mean()
                              - o[RNG.integers(0, len(o), len(o))].mean())
                    ww.append(len(a))
            bs.append(np.average(dd, weights=ww))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {c:13s} stratified diff {est:+.2f} CI95({lo:+.2f},{hi:+.2f})")

    # does churn/unresolved predict our underperformance d?
    print("\n" + "=" * 72)
    print("CHURN vs per-game d = L_us - L_mkt (all covered games)")
    from scipy.stats import spearmanr
    for c in ("churn_any", "churn_strict", "quest", "qd"):
        rho, pv = spearmanr(g[c], g.d)
        print(f"  spearman({c}, d) = {rho:+.4f} (p={pv:.3f})")
    for c in ("churn_any", "quest"):
        t = g[c].rank(pct=True)
        for name, m in [("low", t <= 1 / 3), ("mid", (t > 1 / 3) & (t <= 2 / 3)),
                        ("high", t > 2 / 3)]:
            lo, hi = boot_paired_ci(g.d[m].values)
            print(f"  {c} {name:4s} tertile: n={int(m.sum())} "
                  f"d={g.d[m].mean():+.4f} CI95({lo:+.4f},{hi:+.4f})")

    # decisive for the actionable rule: WITHIN opposite-side games, do
    # high-churn/unresolved ones drive the loss?
    print("\nWITHIN opposite-side games: d by news split")
    sel = g[g.opp]
    for c in ("churn_any", "churn_strict", "quest", "qd"):
        med = sel[c].median()
        hi_m = sel[c] > med
        lo_m = ~hi_m
        dlo, dhi = sel.d[lo_m], sel.d[hi_m]
        ci = boot_dmean_ci(dhi, dlo)
        wr_hi = np.where(sel.p_us[hi_m] > 0.5, sel.y[hi_m],
                         1 - sel.y[hi_m]).mean()
        wr_lo = np.where(sel.p_us[lo_m] > 0.5, sel.y[lo_m],
                         1 - sel.y[lo_m]).mean()
        print(f"  {c:13s} (med {med:.1f}): d_high={dhi.mean():+.4f} (n={hi_m.sum()},"
              f" our-side win {wr_hi:.3f}) vs d_low={dlo.mean():+.4f} "
              f"(n={lo_m.sum()}, win {wr_lo:.3f}); "
              f"diff {dhi.mean() - dlo.mean():+.4f} CI95({ci[0]:+.4f},{ci[1]:+.4f})")


if __name__ == "__main__":
    main()
