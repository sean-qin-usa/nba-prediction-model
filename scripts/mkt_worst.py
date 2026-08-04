"""MKT-WORST: the market's 150 biggest misses (highest L_mkt), diagnosed.

Sean's inversion of the usual autopsy: instead of asking where WE were wrong,
rank games by MARKET log loss (confident favorite loses), enrich each from the
DB (phase, star-outs, b2b, tank, returns, rest, form, series game), and test
against a q-matched control of confident favorites that LANDED whether any
PIT-visible factor over-represents in market blowups (= exploitable) or the
blowups are pure priced variance.

Also quantifies OUR aggregate performance on those games and decomposes it:
a flatter model mechanically wins on upsets and loses on landed favorites, so
our blowup edge is compared against a zero-information "flattened market"
p_flat = sigmoid(a + b*logit(p_mkt)) with (a,b) fit by OLS of logit(p_us) on
logit(p_mkt) over all games. Real signal = our edge minus the flat edge.

Read-only DB. Writes ~/nba_review_bundle/market_worst.csv + market_worst_summary.md.

Conventions:
  star-out (inactives): game_inactives row w/ trailing-5 mean minutes >= 25
    (before tonight). Coverage: 2023-24 + 2024-25 only (25-26 empty).
  star-absent (ex-post, all seasons): player whose last appearance was for
    this team within 10 days, trailing-5 mean min (incl last app) >= 25, and
    who does not appear in tonight's box score.
  returning: appears tonight, gap since previous appearance >= 10 days,
    trailing-5 mean minutes >= 25.
  tsd: home-away tank-score diff (per bet_sim2/tanking.py); tsd_fav is
    favorite-signed. |tsd|>0.5 = "big tank info" per D73 convention.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "nba_review_bundle"
N_TOP = 150
Q_CONF = 0.75           # "market-confident" threshold for pooled tests
STAR_MIN = 25.0         # trailing-min threshold for star flags
RET_GAP = 10            # days absent to count as a "return" / recent-absence window
TSD_BIG = 0.5
SEED = 73
N_BOOT = 4000

rng = np.random.default_rng(SEED)


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- load CSV
df = pd.read_csv(ROOT / "data" / "capstone_pergame_tank.csv", dtype={"game_id": str})
df["game_id"] = df.game_id.str.zfill(10)
df["game_date"] = pd.to_datetime(df.game_date)
df["L_mkt"] = logloss(df.p_mkt, df.y)
df["L_us"] = logloss(df.p_us, df.y)
df["dLL"] = df.L_mkt - df.L_us          # >0 = we beat the market on this game
df["fav_home"] = df.p_mkt >= 0.5
df["q_mkt"] = np.where(df.fav_home, df.p_mkt, 1 - df.p_mkt)
df["y_fav"] = np.where(df.fav_home, df.y, 1 - df.y).astype(int)
df["p_us_fav"] = np.where(df.fav_home, df.p_us, 1 - df.p_us)
df["tsd_fav"] = np.where(df.fav_home, df.tsd, -df.tsd)
df["n_out_fav"] = np.where(df.fav_home, df.n_out_home, df.n_out_away)
df["n_out_dog"] = np.where(df.fav_home, df.n_out_away, df.n_out_home)

# ---------------------------------------------------------------- DB enrich
con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)

tg = con.execute("""
    SELECT season, game_id, game_date, team_id, team_abbrev, is_home, wl, pts
    FROM nba_games WHERE game_id LIKE '002%'
""").fetchdf()
tg["game_date"] = pd.to_datetime(tg.game_date)
tg = tg.sort_values(["team_abbrev", "game_date"]).reset_index(drop=True)
tg["win"] = (tg.wl == "W").astype(float)
g_team = tg.groupby(["season", "team_abbrev"], sort=False)
tg["gp_before"] = g_team.cumcount()
tg["form5"] = (g_team.win.transform(
    lambda s: s.rolling(5, min_periods=1).mean().shift(1)))
# schedule spot features derived from the schedule itself (schedule_features
# table only covers 2025-26, so compute for all seasons uniformly)
tg["prev_gdate"] = g_team.game_date.shift(1)
tg["gap"] = (tg.game_date - tg.prev_gdate).dt.days
tg["is_b2b"] = tg.gap == 1
tg["days_rest"] = (tg.gap - 1).astype("float64")        # NaN at season opener
tg["is_3in4"] = (tg.game_date - g_team.game_date.shift(2)).dt.days <= 3

# margins per game
home = tg[tg.is_home][["game_id", "team_abbrev", "pts", "game_date", "season"]]
away = tg[~tg.is_home][["game_id", "team_abbrev", "pts"]]
gm = home.merge(away, on="game_id", suffixes=("_h", "_a"))
gm["home_margin"] = gm.pts_h - gm.pts_a
# season series meeting number
gm["pair"] = np.where(gm.team_abbrev_h < gm.team_abbrev_a,
                      gm.team_abbrev_h + "_" + gm.team_abbrev_a,
                      gm.team_abbrev_a + "_" + gm.team_abbrev_h)
gm = gm.sort_values("game_date")
gm["series_gm"] = gm.groupby(["season", "pair"]).cumcount() + 1

pm = con.execute("""
    SELECT p.game_id, p.player_id, p.team_id, p.seconds/60.0 AS minutes,
           g.game_date
    FROM player_game_stats p
    JOIN (SELECT DISTINCT game_id, game_date FROM nba_games
          WHERE game_id LIKE '002%') g USING (game_id)
    WHERE p.seconds > 0
""").fetchdf()
pm["game_date"] = pd.to_datetime(pm.game_date)
pm = pm.sort_values(["player_id", "game_date"]).reset_index(drop=True)
gp_p = pm.groupby("player_id", sort=False)
pm["trail5_excl"] = gp_p.minutes.transform(
    lambda s: s.rolling(5, min_periods=1).mean().shift(1))          # before tonight
pm["trail5_incl"] = gp_p.minutes.transform(
    lambda s: s.rolling(5, min_periods=1).mean())                    # incl tonight
pm["prev_date"] = gp_p.game_date.shift(1)
pm["next_date"] = gp_p.game_date.shift(-1)
pm["gap_days"] = (pm.game_date - pm.prev_date).dt.days

# returning stars: appear tonight after >=10d absence, star-level trailing min
ret = (pm[(pm.gap_days >= RET_GAP) & (pm.trail5_excl >= STAR_MIN)]
       .groupby(["game_id", "team_id"]).size().rename("n_ret").reset_index())

# star-absent (ex-post): star intervals joined to team schedule (in-mem duckdb)
iv = pm[(pm.trail5_incl >= STAR_MIN)][
    ["player_id", "team_id", "game_date", "next_date"]].copy()
iv["next_date"] = iv.next_date.fillna(pd.Timestamp("2099-01-01"))
iv = iv[(iv.next_date - iv.game_date).dt.days >= 2]     # else no game missable
tgm = tg[["game_id", "team_id", "game_date"]].copy()
mem = duckdb.connect()
mem.register("iv", iv)
mem.register("tgm", tgm)
absent = mem.execute("""
    SELECT t.game_id, t.team_id, COUNT(*) AS n_absent
    FROM tgm t JOIN iv v
      ON t.team_id = v.team_id
     AND t.game_date > v.game_date AND t.game_date < v.next_date
     AND date_diff('day', v.game_date, t.game_date) <= ?
    GROUP BY 1, 2
""", [RET_GAP]).fetchdf()

# star-out via game_inactives (coverage 2023-24 + 2024-25)
gi = con.execute("SELECT game_id, player_id, team_id FROM game_inactives").fetchdf()
gi = gi.merge(gm[["game_id", "game_date"]], on="game_id", how="inner")
gi = gi.sort_values(["player_id", "game_date"]).reset_index(drop=True)
last_app = pm[["player_id", "game_date", "trail5_incl"]].rename(
    columns={"trail5_incl": "last_trail"}).sort_values(["game_date", "player_id"])
gi = pd.merge_asof(gi.sort_values(["game_date", "player_id"]), last_app,
                   on="game_date", by="player_id", allow_exact_matches=False)
inact = (gi[gi.last_trail >= STAR_MIN]
         .groupby(["game_id", "team_id"]).size().rename("n_inact").reset_index())

# ------------------------------------------------- assemble per-side features
abbrev2id = tg.drop_duplicates(["season", "team_abbrev"])[
    ["season", "team_abbrev", "team_id"]]


def side_feats(base, team_col, tag):
    """Attach team-level features for the team in `team_col` of `base`."""
    b = base.merge(
        tg[["season", "game_id", "team_abbrev", "gp_before", "form5", "team_id",
            "is_b2b", "is_3in4", "days_rest"]],
        left_on=["season", "game_id", team_col],
        right_on=["season", "game_id", "team_abbrev"], how="left")
    for src, myname in [(ret, "n_ret"), (absent, "n_absent"), (inact, "n_inact")]:
        b = b.merge(src, on=["game_id", "team_id"], how="left")
        b[myname] = b[myname].fillna(0).astype(int)
    keep = {"gp_before": f"{tag}_gp", "form5": f"{tag}_form5",
            "is_b2b": f"{tag}_b2b", "is_3in4": f"{tag}_3in4",
            "days_rest": f"{tag}_rest", "n_ret": f"{tag}_ret",
            "n_absent": f"{tag}_absent", "n_inact": f"{tag}_inact"}
    return b.rename(columns=keep)[list(base.columns) + list(keep.values())]


df["fav"] = np.where(df.fav_home, df.home, df.away)
df["dog"] = np.where(df.fav_home, df.away, df.home)
enr = side_feats(df, "fav", "fav")
enr = side_feats(enr, "dog", "dog")
enr = enr.merge(gm[["game_id", "home_margin", "series_gm"]], on="game_id", how="left")
enr["fav_margin"] = np.where(enr.fav_home, enr.home_margin, -enr.home_margin)
enr["fav_b2b"] = enr.fav_b2b.fillna(False).astype(bool)
enr["dog_b2b"] = enr.dog_b2b.fillna(False).astype(bool)
enr["fav_3in4"] = enr.fav_3in4.fillna(False).astype(bool)
enr["dog_3in4"] = enr.dog_3in4.fillna(False).astype(bool)
enr["fav_rest"] = enr.fav_rest.astype("float64")
enr["dog_rest"] = enr.dog_rest.astype("float64")
enr["rest_diff"] = enr.fav_rest - enr.dog_rest
enr["late"] = enr[["fav_gp", "dog_gp"]].min(axis=1) >= 55
enr["early"] = enr[["fav_gp", "dog_gp"]].min(axis=1) < 10
enr["phase"] = np.where(enr.late, "late", np.where(enr.early, "early", "mid"))
enr["star_cov"] = enr.season.isin(["2023-24", "2024-25"])

# ---------------------------------------------------------------- cases/controls
enr = enr.sort_values("L_mkt", ascending=False).reset_index(drop=True)
blow = enr.head(N_TOP).copy()
blow["grp"] = "blowup"
print(f"[cases] top {N_TOP} by L_mkt: L_mkt {blow.L_mkt.min():.3f}.."
      f"{blow.L_mkt.max():.3f}; q_mkt {blow.q_mkt.min():.3f}.."
      f"{blow.q_mkt.max():.3f} (mean {blow.q_mkt.mean():.3f}); "
      f"all favorite losses: {(blow.y_fav == 0).all()}")

pool = enr[(enr.y_fav == 1) & ~enr.game_id.isin(blow.game_id)].copy()
used, ctrl_rows = set(), []
for _, r in blow.sort_values("q_mkt", ascending=False).iterrows():
    cand = pool[(pool.season == r.season) & ~pool.game_id.isin(used)]
    pick = cand.iloc[(cand.q_mkt - r.q_mkt).abs().argsort().iloc[0]]
    used.add(pick.game_id)
    ctrl_rows.append((r.game_id, pick.game_id))
pair_map = pd.DataFrame(ctrl_rows, columns=["blow_gid", "ctrl_gid"])
ctrl = (pair_map.merge(enr, left_on="ctrl_gid", right_on="game_id")
        .set_index("blow_gid"))
blow_i = blow.set_index("game_id").loc[pair_map.blow_gid]
print(f"[match] q_mkt blowups {blow_i.q_mkt.mean():.4f} vs controls "
      f"{ctrl.q_mkt.mean():.4f} (mean |dq| {np.abs(blow_i.q_mkt.values - ctrl.q_mkt.values).mean():.4f})")

# ---------------------------------------------------------------- factor table
FACTORS = [
    ("fav_absent>0",  lambda d: d.fav_absent > 0,  "all"),
    ("dog_absent>0",  lambda d: d.dog_absent > 0,  "all"),
    ("fav_inact>0",   lambda d: d.fav_inact > 0,   "cov"),
    ("dog_inact>0",   lambda d: d.dog_inact > 0,   "cov"),
    ("fav_ret>0",     lambda d: d.fav_ret > 0,     "all"),
    ("dog_ret>0",     lambda d: d.dog_ret > 0,     "all"),
    ("fav_b2b",       lambda d: d.fav_b2b,          "all"),
    ("dog_b2b",       lambda d: d.dog_b2b,          "all"),
    ("fav_3in4",      lambda d: d.fav_3in4,         "all"),
    ("rest_diff<0",   lambda d: d.rest_diff < 0,    "all"),
    ("fav_road",      lambda d: ~d.fav_home,        "all"),
    ("tank_big",      lambda d: d.tsd.abs() > TSD_BIG, "all"),
    ("tsd_fav<-0.5",  lambda d: d.tsd_fav < -TSD_BIG, "all"),  # DOG is the tankier side
    ("fav_cold(<=.4)", lambda d: d.fav_form5 <= 0.4, "all"),
    ("fav_hot(>=.8)", lambda d: d.fav_form5 >= 0.8, "all"),
    ("dog_hot(>=.6)", lambda d: d.dog_form5 >= 0.6, "all"),
    ("late(gp>=55)",  lambda d: d.late,              "all"),
    ("early(gp<10)",  lambda d: d.early,             "all"),
    ("series_gm>=3",  lambda d: d.series_gm >= 3,    "all"),
    ("n_out_fav>=3",  lambda d: d.n_out_fav >= 3,    "all"),
    ("n_out_dog>=3",  lambda d: d.n_out_dog >= 3,    "all"),
]

rows = []
for name, fn, cov in FACTORS:
    mask = blow_i.star_cov.values if cov == "cov" else np.ones(len(blow_i), bool)
    b = fn(blow_i).values[mask].astype(float)
    c = fn(ctrl).values[mask].astype(float)
    d = b - c
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append((name, len(d), b.mean(), c.mean(), d.mean(), lo, hi,
                 "SIG" if lo > 0 or hi < 0 else ""))
pat = pd.DataFrame(rows, columns=["factor", "n_pairs", "blowup_rate",
                                  "ctrl_rate", "diff", "ci_lo", "ci_hi", "sig"])
print("\n=== PAIRED FACTOR RATES: blowups vs q-matched landed favorites ===")
print(pat.to_string(index=False,
                    float_format=lambda x: f"{x:.3f}"))

# ------------------------------------------- pooled logistic on confident set
conf = enr[enr.q_mkt >= Q_CONF].copy()
conf["fav_lost"] = 1 - conf.y_fav
print(f"\n[conf] q_mkt>={Q_CONF}: n={len(conf)}, upset rate "
      f"{conf.fav_lost.mean():.4f} vs implied {1 - conf.q_mkt.mean():.4f}")
# calibration by bucket
conf["qb"] = pd.cut(conf.q_mkt, [0.75, 0.80, 0.85, 0.90, 1.0])
cal = conf.groupby("qb", observed=True).agg(
    n=("y_fav", "size"), mean_q=("q_mkt", "mean"), emp_fav_win=("y_fav", "mean"))
print("\n=== MARKET CALIBRATION (confident games) ===")
print(cal.to_string(float_format=lambda x: f"{x:.4f}"))


def fit_logit(X, y, ridge=1e-6):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(60):
        p = sigmoid(X @ w)
        Wd = p * (1 - p)
        H = X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = sigmoid(X @ w)
    Wd = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1]))
    return w, np.sqrt(np.diag(cov))


preds = ["lq", "fav_absent_b", "dog_absent_b", "fav_ret_b", "dog_ret_b",
         "fav_b2b_f", "dog_b2b_f", "rest_diff_z", "fav_form5_c",
         "tsd_fav", "late_f", "early_f", "road_f"]
conf["lq"] = logit(conf.q_mkt)
conf["fav_absent_b"] = (conf.fav_absent > 0).astype(float)
conf["dog_absent_b"] = (conf.dog_absent > 0).astype(float)
conf["fav_ret_b"] = (conf.fav_ret > 0).astype(float)
conf["dog_ret_b"] = (conf.dog_ret > 0).astype(float)
conf["fav_b2b_f"] = conf.fav_b2b.astype(float)
conf["dog_b2b_f"] = conf.dog_b2b.astype(float)
conf["rest_diff_z"] = (conf.rest_diff - conf.rest_diff.mean()) / conf.rest_diff.std()
conf["fav_form5_c"] = conf.fav_form5.fillna(conf.fav_form5.mean()) - conf.fav_form5.mean()
conf["late_f"] = conf.late.astype(float)
conf["early_f"] = conf.early.astype(float)
conf["road_f"] = (~conf.fav_home).astype(float)
cc = conf.dropna(subset=["rest_diff_z"])
w, se = fit_logit(cc[preds].values.astype(float), cc.fav_lost.values.astype(float))
lg = pd.DataFrame({"term": ["intercept"] + preds, "coef": w, "se": se,
                   "z": w / se})
print(f"\n=== POOLED LOGISTIC: fav_lost ~ factors | q  (n={len(cc)}) ===")
print(lg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ------------------------------------------------ net-signal decomposition
a_b = np.polyfit(logit(df.p_mkt), logit(df.p_us), 1)   # slope, intercept
slope, icept = a_b[0], a_b[1]
enr["p_flat"] = sigmoid(icept + slope * logit(enr.p_mkt))
enr["L_flat"] = logloss(enr.p_flat, enr.y)
print(f"\n[flat] logit(p_us) = {icept:+.4f} + {slope:.4f} * logit(p_mkt)  "
      f"(slope<1 = we are mechanically flatter)")


def decomp(d, label):
    e_us = (d.L_mkt - d.L_us).sum()
    e_fl = (d.L_mkt - d.L_flat).sum()
    print(f"  {label:34s} n={len(d):5d}  edge_us {e_us:+8.3f} "
          f"({(d.L_mkt - d.L_us).mean():+.4f}/gm)  mech_flat {e_fl:+8.3f}  "
          f"real {e_us - e_fl:+8.3f}")
    return e_us, e_fl


print("\n=== NET-SIGNAL DECOMPOSITION (edge vs market, nats; real = us - flat) ===")
bl = enr[enr.game_id.isin(blow.game_id)]
ct = enr[enr.game_id.isin(pair_map.ctrl_gid)]
cf = enr[enr.q_mkt >= Q_CONF]
decomp(bl, "150 market blowups")
decomp(ct, "150 matched landed favorites")
decomp(cf[cf.y_fav == 0], f"all confident upsets (q>={Q_CONF})")
decomp(cf[cf.y_fav == 1], f"all confident landed (q>={Q_CONF})")
e_us_c, e_fl_c = decomp(cf, "ALL confident games (net)")
decomp(enr, "ALL games")

# --------------------------------------------------------------- outputs
def md_table(d: pd.DataFrame, fmt=".3f") -> str:
    d = d.reset_index(drop=True)
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else format(v, fmt)
        return str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines)


OUT.mkdir(exist_ok=True)
cols = ["season", "game_id", "game_date", "fav", "dog", "fav_home", "q_mkt",
        "fav_margin", "L_mkt", "p_us_fav", "L_us", "dLL", "phase", "fav_gp",
        "dog_gp", "fav_absent", "dog_absent", "fav_inact", "dog_inact",
        "fav_ret", "dog_ret", "fav_b2b", "dog_b2b", "fav_3in4", "dog_3in4",
        "fav_rest", "dog_rest", "rest_diff", "tsd_fav", "fav_form5",
        "dog_form5", "series_gm", "n_out_fav", "n_out_dog", "star_cov"]
out_csv = blow[cols].copy()
for c in ["q_mkt", "L_mkt", "p_us_fav", "L_us", "dLL", "tsd_fav",
          "fav_form5", "dog_form5"]:
    out_csv[c] = out_csv[c].round(4)
out_csv["game_date"] = out_csv.game_date.dt.date
out_csv.to_csv(OUT / "market_worst.csv", index=False)
print(f"\n[write] {OUT / 'market_worst.csv'} ({len(out_csv)} rows)")

with open(OUT / "market_worst_summary.md", "w") as f:
    f.write("# Market's 150 biggest misses — enrichment + pattern findings\n\n")
    f.write("Source: capstone_pergame_tank.csv (shipped D73 model, 2023-24..2025-26, "
            f"n={len(df)}). Cases = top {N_TOP} games by MARKET log loss (all are "
            "favorite losses). Controls = per-case same-season nearest-q_mkt "
            "favorite WINS, matched without replacement "
            f"(mean q: cases {blow_i.q_mkt.mean():.4f} vs controls {ctrl.q_mkt.mean():.4f}).\n\n"
            "Column dictionary for market_worst.csv: q_mkt = market favorite prob "
            "(de-vig close, spread-converted); fav_margin = favorite's actual margin "
            "(negative = upset); L_mkt/L_us = per-game log loss of market / our model; "
            "dLL = L_mkt - L_us (>0 = we beat the close on that game); p_us_fav = our "
            "prob on the favorite; phase early = either team <10 gp, late = both >=55 "
            "(tank window); fav/dog_absent = ex-post star absences (trailing-5 min "
            ">=25, last played for this team within 10d, not in tonight's box); "
            "fav/dog_inact = official game_inactives star-outs (valid only where "
            "star_cov=True, i.e. 2023-24/2024-25); fav/dog_ret = star returning "
            "tonight after >=10d absence; tsd_fav = favorite-signed tank-score diff "
            "(negative = the DOG is the tankier/deader team... note sign: tsd_fav<-0.5 "
            "means favorite faces a big-tank dog); fav_form5 = favorite's win rate "
            "last 5; series_gm = meeting number in season series; n_out_fav/dog = "
            "report-based OUT counts from the shipped pipeline.\n\n")
    f.write("## Paired factor rates (blowups vs matched landed favorites)\n\n")
    f.write(md_table(pat) + "\n\n")
    f.write("## Market calibration on confident games (q>=%.2f, n=%d)\n\n"
            % (Q_CONF, len(conf)))
    f.write(md_table(cal.reset_index().assign(qb=lambda d: d.qb.astype(str)), ".4f") + "\n\n")
    f.write("## Pooled logistic: fav_lost ~ q + factors (confident games)\n\n")
    f.write(md_table(lg) + "\n\n")
    f.write("## Our performance & net-signal decomposition\n\n")
    f.write("Flat baseline: logit(p_flat) = %+.4f + %.4f*logit(p_mkt) (OLS of our "
            "logits on market logits over all games) — the zero-information "
            "'shrunk market'. Real signal = our edge minus flat edge.\n\n"
            % (icept, slope))
    f.write("| set | n | edge_us (nats) | edge_us/gm | mech_flat | real |\n"
            "|---|---|---|---|---|---|\n")
    for d, label in [(bl, "150 blowups"), (ct, "150 matched landed"),
                     (cf[cf.y_fav == 0], "all confident upsets"),
                     (cf[cf.y_fav == 1], "all confident landed"),
                     (cf, "ALL confident (net)"), (enr, "ALL games")]:
        e_us = (d.L_mkt - d.L_us).sum()
        e_fl = (d.L_mkt - d.L_flat).sum()
        f.write(f"| {label} | {len(d)} | {e_us:+.3f} | "
                f"{(d.L_mkt - d.L_us).mean():+.4f} | {e_fl:+.3f} | "
                f"{e_us - e_fl:+.3f} |\n")
    f.write("\nNotes: game_inactives is empty for 2025-26, so *_inact rows are "
            "computed on 2023-24+2024-25 pairs only (star_cov). *_absent is the "
            "all-season ex-post proxy. Blowup rows are sorted by L_mkt desc.\n")
print(f"[write] {OUT / 'market_worst_summary.md'}")
