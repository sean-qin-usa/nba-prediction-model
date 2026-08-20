"""EXP rolescale — Sean's role-scalability stat.

Hypothesis: players differ in how their scoring EFFICIENCY (TS%) responds to an
exogenous role expansion (a rotation teammate absent). D33 showed attempts lift
~+20% in star-out games; D34 showed uniform-efficiency points-lift FAILS (mean
efficiency drops). This experiment estimates a per-player role-slope
(delta TS% in exogenous-expansion games), EB-shrinks it, and gates it on
held-out star-out games by out-of-sample MSE on pts/36 deviation.

ENDOGENEITY HANDLING (explicit):
  - Treatment is NEVER realized minutes/rank/rotation-size (hot shooting earns
    minutes). Treatment = team had >=1 fresh (<=12d) absent rotation player
    (trailing-5-game avg >= 20 min), detected from teammates' appearance gaps
    — exogenous to tonight's shooting by the player under analysis.
  - Role baseline = trailing-5-game avg minutes (pregame, shift(1)).
  - We also compute the ENDOGENOUS version (realized minutes > trail5 + 6) to
    demonstrate the inflation the exogenous design avoids.

Walk-forward: slopes + EB prior + attempts-lift L are re-fit at each month
boundary from data strictly before that month. No same-day or future info.
Read-only DB. Standalone file — copies logic, edits nothing in nbapred/.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from nbapred.db import connect

RNG_SEED = 7
N_BOOT = 2000
FRESH_DAYS = 12
ROT_MIN = 20.0     # rotation-player trailing avg minutes
STAR_MIN = 28.0    # star trailing avg minutes (gate condition)
MIN_SECONDS = 600  # analysis player-games
N1_MIN, N0_MIN = 3, 10  # min expansion / normal games for a raw slope
GATE_SEASONS = ["2023-24", "2024-25", "2025-26"]  # matches capstone baseline


def load_appearances(con):
    df = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.season, g.game_date,
               s.seconds/60.0 AS mins, s.pts, s.fga, s.fta
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY s.player_id, g.game_date""").fetchdf()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["tsa"] = df["fga"] + 0.44 * df["fta"]
    g = df.groupby("player_id", sort=False)
    # planned-role baseline: trailing 5-game avg minutes, PREGAME (shift 1)
    df["trail5_min"] = g["mins"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    # rotation status as of a given (last) appearance: trailing 5 INCLUDING that game
    df["trail5_incl"] = g["mins"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    # trailing rate baselines over last 15 games (pregame, shift 1, min 8)
    for c in ("pts", "tsa", "mins"):
        df[f"sum_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(15, min_periods=8).sum())
    df["pts36_base"] = df["sum_pts"] / df["sum_mins"] * 36.0
    df["tsa36_base"] = df["sum_tsa"] / df["sum_mins"] * 36.0
    df["ts_base"] = df["sum_pts"] / (2.0 * df["sum_tsa"])
    return df


def detect_absences(df):
    """From appearance gaps: for each team-game, count fresh (<=FRESH_DAYS)
    absent players whose trailing-5 (incl. last appearance) avg minutes was
    >= ROT_MIN (rotation) / >= STAR_MIN (star). A player traded away stops
    counting at their first appearance for another team."""
    team_games = df[["team_id", "game_id", "game_date"]].drop_duplicates()
    tg_dates, tg_ids = {}, {}
    for t, sub in team_games.groupby("team_id"):
        sub = sub.sort_values("game_date")
        tg_dates[t] = sub["game_date"].values
        tg_ids[t] = sub["game_id"].values
    played = df.groupby(["team_id", "game_id"])["player_id"].apply(set).to_dict()

    rot_out, star_out = {}, {}
    ap = df[["player_id", "team_id", "game_date", "trail5_incl"]].sort_values(
        ["player_id", "game_date"])
    for pid, sub in ap.groupby("player_id", sort=False):
        dates = sub["game_date"].values
        teams = sub["team_id"].values
        t5 = sub["trail5_incl"].values
        n = len(dates)
        for i in range(n):
            if not (t5[i] >= ROT_MIN):
                continue
            d0 = dates[i]
            hi = d0 + np.timedelta64(FRESH_DAYS, "D")
            if i + 1 < n:
                hi = min(hi, dates[i + 1] - np.timedelta64(1, "D"))
            if hi <= d0:
                continue
            T = teams[i]
            gd = tg_dates[T]
            lo_ix = np.searchsorted(gd, d0, side="right")
            hi_ix = np.searchsorted(gd, hi, side="right")
            for j in range(lo_ix, hi_ix):
                key = (T, tg_ids[T][j])
                if pid in played.get(key, ()):  # safety (shouldn't happen)
                    continue
                rot_out[key] = rot_out.get(key, 0) + 1
                if t5[i] >= STAR_MIN:
                    star_out[key] = star_out.get(key, 0) + 1
    return rot_out, star_out


def dl_eb(y, se2):
    """DerSimonian-Laird method-of-moments EB with normal prior."""
    w = 1.0 / se2
    mu_f = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - mu_f) ** 2)
    k = len(y)
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / denom) if denom > 0 else 0.0
    w_r = 1.0 / (se2 + tau2)
    mu = np.sum(w_r * y) / np.sum(w_r)
    shrunk = mu + (tau2 / (tau2 + se2)) * (y - mu)
    return shrunk, mu, tau2


def player_slopes(train, treat_col="expand"):
    """Within-player contrast of game TS% (treat vs not), pooled-variance SE,
    then DL empirical-Bayes shrinkage. Returns (per-player df, mu, tau2)."""
    agg = train.groupby(["player_id", treat_col])["ts_game"].agg(
        n="count", m="mean", v="var").reset_index()
    piv = agg.pivot(index="player_id", columns=treat_col)
    try:
        n1 = piv[("n", True)]; n0 = piv[("n", False)]
        m1 = piv[("m", True)]; m0 = piv[("m", False)]
        v1 = piv[("v", True)]; v0 = piv[("v", False)]
    except KeyError:
        return None, np.nan, np.nan
    ok = (n1 >= N1_MIN) & (n0 >= N0_MIN)
    n1, n0, m1, m0 = n1[ok], n0[ok], m1[ok], m0[ok]
    v1, v0 = v1[ok].fillna(0.0), v0[ok].fillna(0.0)
    if ok.sum() < 20:
        return None, np.nan, np.nan
    s2 = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
    se2 = (s2 * (1.0 / n1 + 1.0 / n0)).clip(lower=1e-6)
    raw = (m1 - m0).astype(float)
    shrunk, mu, tau2 = dl_eb(raw.values, se2.values)
    out = pd.DataFrame({"player_id": raw.index, "raw": raw.values,
                        "se2": se2.values, "shrunk": shrunk})
    return out, mu, tau2


def clustered_boot(err_a, err_b, players, n_boot=N_BOOT, seed=RNG_SEED):
    """delta = MSE_a - MSE_b (positive => b better). Players are the clusters."""
    d = err_a - err_b
    uniq = np.unique(players)
    idx_by_p = {p: np.where(players == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by_p[p] for p in pick])
        stats[b] = d[ix].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    con = connect(read_only=True)
    df = load_appearances(con)
    print(f"appearances: {len(df)}  players: {df.player_id.nunique()}  "
          f"seasons: {sorted(df.season.unique())}")

    rot_out, star_out = detect_absences(df)
    keys = pd.MultiIndex.from_frame(df[["team_id", "game_id"]])
    df["n_rot_out"] = [rot_out.get(k, 0) for k in keys]
    df["n_star_out"] = [star_out.get(k, 0) for k in keys]
    df["expand"] = df["n_rot_out"] >= 1
    df["starout"] = df["n_star_out"] >= 1

    # analysis set
    an = df[(df.mins * 60 >= MIN_SECONDS) & (df.tsa > 0)].copy()
    an["ts_game"] = an["pts"] / (2.0 * an["tsa"])
    an["pts36"] = an["pts"] / an["mins"] * 36.0
    an["tsa36"] = an["tsa"] / an["mins"] * 36.0
    has_base = an["pts36_base"].notna() & an["tsa36_base"].notna() & (an["sum_tsa"] > 0)
    print(f"analysis player-games (>= {MIN_SECONDS}s, tsa>0): {len(an)}  "
          f"expansion share: {an.expand.mean():.3f}  star-out share: {an.starout.mean():.3f}")

    # ---------- first stage: does exogenous expansion actually expand roles? ----------
    fs = an[has_base & an.trail5_min.notna()]
    for lab, m in [("expansion", fs.expand), ("star-out", fs.starout)]:
        a, b = fs[m], fs[~m]
        print(f"first-stage [{lab}]  n={len(a)}  d_mins(vs trail5) {a.mins.sub(a.trail5_min).mean():+.2f} "
              f"vs {b.mins.sub(b.trail5_min).mean():+.2f}   tsa36/base {(a.tsa36/a.tsa36_base).mean():.3f} "
              f"vs {(b.tsa36/b.tsa36_base).mean():.3f}   dTS {(a.ts_game-a.ts_base).mean():+.4f} "
              f"vs {(b.ts_game-b.ts_base).mean():+.4f}")

    # ---------- endogeneity demonstration (full data, descriptive only) ----------
    endo = an[an.trail5_min.notna()].copy()
    endo["endo_expand"] = endo["mins"] > endo["trail5_min"] + 6.0
    tab_x, mu_x, tau2_x = player_slopes(an, "expand")
    tab_e, mu_e, tau2_e = player_slopes(endo, "endo_expand")
    print(f"\nENDOGENEITY CHECK (full-data, descriptive):")
    print(f"  exogenous (teammate-absent) slope prior: mu={mu_x:+.4f} tau={np.sqrt(tau2_x):.4f} k={len(tab_x)}")
    print(f"  endogenous (realized mins>trail5+6):     mu={mu_e:+.4f} tau={np.sqrt(tau2_e):.4f} k={len(tab_e)}")

    names = con.execute("SELECT player_id, full_name FROM nba_players").fetchdf()
    full = tab_x.merge(names, on="player_id", how="left").sort_values("shrunk")
    print("\nbottom-8 role-scalers (shrunk dTS in expansion):")
    print(full.head(8)[["full_name", "raw", "shrunk"]].to_string(index=False))
    print("top-8 role-scalers:")
    print(full.tail(8)[["full_name", "raw", "shrunk"]].to_string(index=False))

    # starter split (descriptive): starters from lineup_stints stint 0
    st = con.execute("""
        SELECT game_id, home_lineup AS lu FROM lineup_stints WHERE stint_idx=0
        UNION ALL SELECT game_id, away_lineup FROM lineup_stints WHERE stint_idx=0""").fetchdf()
    starters = set()
    for r in st.itertuples():
        for p in str(r.lu).split(","):
            if p.strip().isdigit():
                starters.add((r.game_id, int(p)))
    an["started"] = [(g, p) in starters for g, p in zip(an.game_id, an.player_id)]
    frac_start = an.groupby("player_id")["started"].mean()
    full2 = tab_x.set_index("player_id").join(frac_start.rename("frac_start"))
    for lab, m in [("mostly-starters", full2.frac_start >= 0.5),
                   ("mostly-bench", full2.frac_start < 0.5)]:
        print(f"  {lab}: k={m.sum()}  mean shrunk slope {full2.loc[m,'shrunk'].mean():+.4f}")
    con.close()

    # ---------- walk-forward validation gate ----------
    test_all = an[has_base & an.starout & an.season.isin(GATE_SEASONS)].copy()
    test_all["month"] = test_all["game_date"].values.astype("datetime64[M]")
    rows = []
    for m in sorted(test_all["month"].unique()):
        m_start = pd.Timestamp(m)
        train = an[an.game_date < m_start]
        tab, mu, tau2 = player_slopes(train, "expand")
        if tab is None:
            continue
        # uniform attempts lift fitted on training star-out games (PIT)
        tr = train[train.starout & train.pts36_base.notna() & (train.tsa36_base >= 5)]
        L = float(np.clip(tr["tsa36"] / tr["tsa36_base"], 0.3, 3.0).mean()) if len(tr) >= 50 else 1.0
        sub = test_all[test_all.month == m].merge(
            tab[["player_id", "shrunk"]], on="player_id", how="left")
        sub["L"], sub["mu"] = L, mu
        rows.append(sub)
    ev = pd.concat(rows, ignore_index=True)
    cov = ev["shrunk"].notna().mean()
    ev = ev[ev["shrunk"].notna()].copy()
    print(f"\nGATE eval star-out player-games (walk-forward monthly): n={len(ev)}  "
          f"slope coverage {cov:.2%}  players {ev.player_id.nunique()}")

    dev = (ev.pts36 - ev.pts36_base).values
    p0 = (ev.pts36_base * (ev.L - 1)).values                                   # D33-style uniform efficiency
    p1 = p0 + (2 * ev.tsa36_base * ev.L * ev.mu).values                        # + mean efficiency shift
    p2 = p0 + (2 * ev.tsa36_base * ev.L * ev.shrunk).values                    # + player role-slope
    e0, e1, e2 = (dev - p0) ** 2, (dev - p1) ** 2, (dev - p2) ** 2
    pl = ev.player_id.values

    dA, loA, hiA = clustered_boot(e0, e2, pl)          # GATE A (the spec): slope vs uniform
    dB, loB, hiB = clustered_boot(e1, e2, pl, seed=11)  # GATE B: heterogeneity only
    dC, loC, hiC = clustered_boot(e0, e1, pl, seed=23)  # attribution: mean shift alone
    print(f"MSE uniform(P0)={e0.mean():.3f}  +mean-shift(P1)={e1.mean():.3f}  +role-slope(P2)={e2.mean():.3f}")
    print(f"GATE A (spec) P0-P2: {dA:+.4f} CI({loA:+.4f},{hiA:+.4f})")
    print(f"GATE B (hetero) P1-P2: {dB:+.4f} CI({loB:+.4f},{hiB:+.4f})")
    print(f"attribution (mean) P0-P1: {dC:+.4f} CI({loC:+.4f},{hiC:+.4f})")

    print("\nper-season:")
    per = {}
    for s in GATE_SEASONS:
        m = (ev.season == s).values
        if m.sum() < 30:
            continue
        ds, los, his = clustered_boot(e0[m], e2[m], pl[m], seed=5)
        dBs, loBs, hiBs = clustered_boot(e1[m], e2[m], pl[m], seed=6)
        per[s] = dict(n=int(m.sum()), mse0=float(e0[m].mean()), mse1=float(e1[m].mean()),
                      mse2=float(e2[m].mean()), dA=ds, dA_lo=los, dA_hi=his,
                      dB=dBs, dB_lo=loBs, dB_hi=hiBs)
        print(f"  {s}: n={m.sum()}  MSE0 {e0[m].mean():.3f} MSE2 {e2[m].mean():.3f}  "
              f"A {ds:+.4f} CI({los:+.4f},{his:+.4f})  B {dBs:+.4f} CI({loBs:+.4f},{hiBs:+.4f})")

    # ---------- diagnostics: why? ----------
    res0 = dev - p0
    print(f"\ndiagnostics: mean dev {dev.mean():+.3f}  mean P0 {p0.mean():+.3f}  "
          f"mean resid(P0) {res0.mean():+.3f}  sd(dev) {dev.std():.2f}  mean L {ev.L.mean():.3f}")
    # out-of-sample validity of the shrunk slope at the TS level
    dts = (ev.ts_game - ev.ts_base).values
    r_game = np.corrcoef(ev.shrunk.values, dts)[0, 1]
    per_p = ev.groupby("player_id").agg(sh=("shrunk", "mean"), d=("ts_game", "mean"),
                                        b=("ts_base", "mean"), n=("shrunk", "size"))
    per_p = per_p[per_p.n >= 5]
    r_player = np.corrcoef(per_p.sh, per_p.d - per_p.b)[0, 1]
    print(f"corr(shrunk slope, OOS dTS): per-game {r_game:+.3f}  per-player(n>=5) "
          f"{r_player:+.3f} (k={len(per_p)})")
    # split-half reliability of the raw slope (is there ANY stable signal?)
    sh = an.sort_values(["player_id", "game_date"]).copy()
    sh["half"] = sh.groupby(["player_id", "expand"]).cumcount() % 2
    halves = []
    for h in (0, 1):
        t, _, _ = player_slopes(sh[sh.half == h], "expand")
        halves.append(t.set_index("player_id")["raw"].rename(f"h{h}"))
    both = pd.concat(halves, axis=1).dropna()
    r_split = np.corrcoef(both.h0, both.h1)[0, 1]
    print(f"split-half reliability of raw slope (odd/even games, k={len(both)}): r={r_split:+.3f}")

    verdict = "PASS" if loA > 0 else ("FAIL" if hiA < 0 else "NS")
    print(f"\nVERDICT (Gate A, spec): {verdict}")
    print("EXP_DONE", flush=True)
    return dict(dA=dA, loA=loA, hiA=hiA, dB=dB, loB=loB, hiB=hiB,
                dC=dC, loC=loC, hiC=hiC, per=per, verdict=verdict,
                n=len(ev), cov=cov, mu_x=mu_x, tau_x=float(np.sqrt(tau2_x)),
                mu_e=mu_e, k=len(tab_x))


if __name__ == "__main__":
    main()
