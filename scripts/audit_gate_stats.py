"""Audit: statistical methodology of the gate battery (read-only, no prod edits).

1. Baseline-drift check: does data/capstone_pergame.csv reproduce the D46
   registered capstone (0.5999/0.5962/0.5918)? Does the sched artifact?
2. D46 gate under resampling schemes: iid game bootstrap (the shipped method)
   vs cluster bootstraps (date, ISO-week, home-team, season) on the paired
   per-game log-loss delta legacy->sched. If clustered CIs are materially
   wider, every marginal PASS in the register is over-confident.
3. Serial dependence of the paired delta (lag-1..5 autocorrelation in date
   order) — the ablate.py docstring promises a BLOCK bootstrap; code is iid.
4. Multiplicity: enumerate the CI-gated comparisons in DECISIONS/ledger/journal,
   compute expected false PASSes at the one-sided 2.5% keep rule, P(>=1), and
   which shipped passes survive a Bonferroni family correction.
"""
import numpy as np, pandas as pd

EPS = 1e-12

def ll_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def load(path):
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

base_dir = "/hdd/steveqin/sean_dev/nba_model/data/"
cur   = load(base_dir + "capstone_pergame.csv")
sched = load(base_dir + "capstone_pergame_sched.csv")
leg   = load(base_dir + "capstone_pergame_legacy.csv")

print("=" * 70)
print("1) BASELINE DRIFT")
for name, df in (("current(capstone_pergame)", cur), ("sched(D46 artifact)", sched),
                 ("legacy(pre-D46)", leg)):
    per = df.groupby("season").apply(
        lambda g: ll_vec(g.y.values, g.p_us.values).mean())
    print(f"  {name:28s} " + "  ".join(f"{s}:{v:.4f}" for s, v in per.items()))
mkt = cur.groupby("season").apply(lambda g: ll_vec(g.y.values, g.p_mkt.values).mean())
print("  market close                 " + "  ".join(f"{s}:{v:.4f}" for s, v in mkt.items()))
merged_cs = cur.merge(sched, on="game_id", suffixes=("_c", "_s"))
print(f"  max |p_us current - sched| on same game: "
      f"{np.abs(merged_cs.p_us_c - merged_cs.p_us_s).max():.4f}  "
      f"mean {np.abs(merged_cs.p_us_c - merged_cs.p_us_s).mean():.5f}")

print("=" * 70)
print("2) D46 PAIRED GATE UNDER RESAMPLING SCHEMES (legacy -> sched)")
m = leg.merge(sched[["game_id", "p_us", "y"]], on="game_id", suffixes=("_b", "_n"))
assert (m.y_b == m.y_n).all()
y = m.y_b.values.astype(float)
d = ll_vec(y, m.p_us_b.values) - ll_vec(y, m.p_us_n.values)   # + = sched better
dates = m.game_date.values
order = np.argsort(dates)
d_ord = d[order]
n = len(d)
rng = np.random.default_rng(12345)
print(f"  n={n}  mean delta {d.mean():+.5f}  (registered D46: +0.00539)")

def iid_ci(x, B=4000):
    idx = rng.integers(0, len(x), (B, len(x)))
    bs = x[idx].mean(axis=1)
    return bs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), bs.std()

def cluster_ci(x, labels, B=4000):
    labs = pd.unique(labels)
    groups = {l: x[labels == l] for l in labs}
    sums = np.array([groups[l].sum() for l in labs])
    cnts = np.array([len(groups[l]) for l in labs])
    K = len(labs)
    bs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, K, K)
        bs[b] = sums[pick].sum() / cnts[pick].sum()
    return bs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), bs.std()

mu, lo, hi, se = iid_ci(d)
print(f"  iid game bootstrap      : CI({lo:+.5f},{hi:+.5f}) SE {se:.5f}  "
      f"{'PASS' if lo > 0 else 'FAIL'}")
week = m.game_date.dt.strftime("%G-%V").values
season_lab = m.season.values
home_lab = m.home.values
date_lab = m.game_date.dt.strftime("%F").values
for lab_name, lab in (("date", date_lab), ("iso-week", week),
                      ("home-team", home_lab), ("season", season_lab)):
    mu, lo, hi, se = cluster_ci(d, lab)
    infl = se / iid_ci(d)[3] if lab_name == "date" else None
    mu0, lo0, hi0, se0 = iid_ci(d)
    print(f"  cluster by {lab_name:10s}: CI({lo:+.5f},{hi:+.5f}) SE {se:.5f}  "
          f"SE-inflation x{se / se0:4.2f}  {'PASS' if lo > 0 else 'FAIL'}")

print("  lag autocorrelation of paired delta (date order):",
      "  ".join(f"lag{k}:{np.corrcoef(d_ord[:-k], d_ord[k:])[0,1]:+.3f}"
               for k in (1, 2, 3, 5)))
# same-date intraclass correlation
df_icc = pd.DataFrame({"d": d, "date": date_lab})
gm = df_icc.groupby("date")["d"]
between = gm.mean().var()
within = gm.apply(lambda s: s.var(ddof=1) if len(s) > 1 else np.nan).mean()
print(f"  same-date ICC (rough): between-date var of means {between:.6f} vs "
      f"mean within-date var {within:.6f}")

print("=" * 70)
print("3) MULTIPLICITY ACROSS THE GATE FAMILY")
# Enumerated CI-gated comparisons (register D1-D51 + ledger + journal 07/26-30).
# One entry per bootstrap-CI (or t-test) keep/kill comparison actually run.
gates = [
    # game-model level
    "cold-start prior (D16)", "team-home ridge (D20)", "four-factors (D21)",
    "8-factor extension (D21)", "GBM challenger (D22)", "fitted-blend-weights (D22)",
    "recency 21d (orig)", "recency 60d retest (D18)", "recency-FF 3season regate",
    "fitted-blend regate", "comp-heavy 60/40 regate", "sched layer (D46)",
    "dead-team term (D47)", "SCALE recal (D48)", "FF_LUCK blunt",
    "ffluck2 defense-only", "ffluck2 team-trailing", "eventrecency pooled",
    "eventrecency isolation", "garbage-time FF", "homesens roster-agg",
    "sigma~pace", "rolescale gate A", "rolescale gate B", "rolescale gate C",
    "market-blend weight (D14)", "totals blend weight",
    # props / player level
    "rest props", "B2B props", "overdispersion", "empirical-minutes",
    "EWMA-hl5 minutes", "Kalman props", "blowout throttle",
    "opp-pace (orig)", "opp-pace EB retest", "teammate-out (orig)",
    "teammate-out minutes-only", "defender-aware 1", "defender-aware 2",
    "defender-aware 3", "defender-aware 4", "def-RAPM aggregate",
    "age-adjust rates (D28)", "injury adj 1", "injury adj 2", "injury adj 3",
    "D33 softmax attempts", "D34 naive points CRPS", "D35 skill-curve CRPS",
    "D38 conditional C&S tilt", "D45 x_minutes", "D45 x_minutes blend",
    "x_minutes vs shipped projector", "creation-split stability",
    "playtype prior naive (D41)", "playtype prior FE (D41)",
]
K = len(gates)
alpha = 0.025  # keep rule: 95% two-sided CI lower bound > 0  == one-sided 2.5%
print(f"  enumerated CI-gated comparisons K = {K} (undercount: excludes "
      f"informal split/threshold variants inside each script)")
print(f"  expected false PASSes under global null = K*alpha = {K*alpha:.2f}")
print(f"  P(>=1 false PASS) = 1-(1-alpha)^K = {1-(1-alpha)**K:.1%}")
for K2 in (50, 80):
    print(f"    at K={K2}: expect {K2*alpha:.2f} chance PASSes, "
          f"P(>=1)={1-(1-alpha)**K2:.1%}")
# which shipped PASSes survive Bonferroni (one-sided alpha/K)?
# approx one-sided p from registered CIs: z = mean / SE, SE = (hi-lo)/3.92
from scipy.stats import norm
passes = {
    "four-factors (D21)":      (0.0178, 0.008, 0.027),
    "team-home (D20)":         (0.0020, 0.0004, 0.0036),
    "sched layer (D46)":       (0.00539, 0.0024, 0.0085),
    "D33 softmax attempts":    (0.044, 0.033, 0.055),
    "cold-start (D16 proxy)":  (0.025, 0.010, 0.040),   # early-season delta, CI approx from journal
}
a_bonf = alpha * 2 / K   # two-sided family alpha 0.05 spread over K
print(f"  Bonferroni per-test alpha (two-sided 0.05/K) = {0.05/K:.5f}")
for name, (mu_, lo_, hi_) in passes.items():
    se_ = (hi_ - lo_) / 3.92
    z = mu_ / se_
    p1 = 1 - norm.cdf(z)
    print(f"    {name:26s} z={z:4.2f}  one-sided p={p1:.2e}  "
          f"{'survives' if p1 < 0.05/K/2 else 'FAILS family correction'}")
print("AUDIT_GATE_STATS_DONE")
