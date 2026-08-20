"""TASK C support: power arithmetic for the shipping gate in the small-effect era.

Read-only analysis (CSV artifacts only; no DuckDB, no prod imports needed).
Sections:
  A) Observed per-game paired loss-delta sd for real shipped/tested changes
     (legacy->sched = D46; csfix->carry = D62/D63 ship-confirm; carry->carry2
     drift check), including the CHANGED-GAME FOOTPRINT of each change and the
     footprint-subset gate. Bootstrap (2000x, fixed seed) vs normal-approx CI
     cross-check to justify the normal power formulas.
  B) Power curves / MDE at 80% power for the paired gate: n in {1230, 3690},
     sd in {0.05, 0.10, 0.15} + the observed sds, effects 0.0005..0.005.
  C) NS-portfolio arithmetic (BRIEF4 ask 1/2): per-member power if the point
     estimates are the true effects; expected number of PASSes; sign-test and
     selection-corrected Stouffer combination for the family.
  D) Past-ship audit (D46/D55/D62) under the proposed GATE_POLICY_V2 tiers +
     BH-FDR across the enumerated K=57 gate family (audit_gate_stats.py list).

Outputs stdout only. Numbers cited in docs/GATE_POLICY_V2.md.
"""
import numpy as np, pandas as pd
from scipy.stats import norm

EPS = 1e-12
B = 2000
SEED = 20260730
DATA = "/hdd/steveqin/sean_dev/nba_model/data/"
Z975 = norm.ppf(0.975)          # 1.960
Z80 = norm.ppf(0.80)            # 0.842


def ll_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load(name):
    df = pd.read_csv(DATA + name)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def boot_ci(x, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    bs = x[idx].mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def pair_stats(base, var, label):
    m = base.merge(var[["game_id", "p_us", "y"]], on="game_id",
                   suffixes=("_b", "_v"))
    assert (m.y_b == m.y_v).all(), label
    y = m.y_b.values.astype(float)
    d = ll_vec(y, m.p_us_b.values) - ll_vec(y, m.p_us_v.values)  # + = variant better
    changed = np.abs(m.p_us_b.values - m.p_us_v.values) > 1e-9
    n, f = len(d), changed.mean()
    mu, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    lo, hi = boot_ci(d)
    z = mu / se if se > 0 else np.nan
    print(f"  {label}")
    print(f"    n={n}  mean {mu:+.5f}  per-game sd {sd:.4f}  SE {se:.5f}  "
          f"z={z:+.2f}  p1={1-norm.cdf(z):.4f}")
    print(f"    boot CI ({lo:+.5f},{hi:+.5f})  normal CI "
          f"({mu-Z975*se:+.5f},{mu+Z975*se:+.5f})  "
          f"footprint(changed games) {changed.sum()} = {f:.1%}")
    if 0 < changed.sum() < n:
        dc = d[changed]
        muc, sdc = dc.mean(), dc.std(ddof=1)
        sec = sdc / np.sqrt(len(dc))
        loc_, hic = boot_ci(dc, seed=SEED + 1)
        zc = muc / sec
        print(f"    FOOTPRINT gate: n={len(dc)}  mean {muc:+.5f}  sd {sdc:.4f}  "
              f"z={zc:+.2f}  boot CI ({loc_:+.5f},{hic:+.5f})  "
              f"{'PASS' if loc_ > 0 else 'NS'}")
    return d, sd


print("=" * 78)
print("A) OBSERVED PER-GAME PAIRED LOSS-DELTA SD (real artifacts)")
files = ["capstone_pergame_legacy.csv", "capstone_pergame_sched.csv",
         "capstone_pergame_csfix.csv", "capstone_pergame_carry.csv",
         "capstone_pergame_carry2.csv", "capstone_pergame.csv",
         "capstone_pergame_dead.csv"]
dfs = {f: load(f) for f in files}
def season_ll(df, col):
    return {s: ll_vec(g.y.values, g[col].values).mean()
            for s, g in df.groupby("season")}

for f, df in dfs.items():
    per = season_ll(df, "p_us")
    print(f"  {f:34s} " + "  ".join(f"{s}:{v:.4f}" for s, v in per.items()))
per = season_ll(dfs["capstone_pergame_carry.csv"], "p_mkt")
print(f"  {'market close':34s} " + "  ".join(f"{s}:{v:.4f}" for s, v in per.items()))
print()
d46, sd46 = pair_stats(dfs["capstone_pergame_legacy.csv"],
                       dfs["capstone_pergame_sched.csv"],
                       "D46 sched layer (legacy -> sched), GLOBAL footprint")
d62, sd62 = pair_stats(dfs["capstone_pergame_csfix.csv"],
                       dfs["capstone_pergame_carry.csv"],
                       "D62/D63 carry ship-confirm (csfix -> carry), LOCAL footprint")
_ = pair_stats(dfs["capstone_pergame_carry.csv"],
               dfs["capstone_pergame_carry2.csv"],
               "carry -> carry2 (2022-23 refill drift check)")
_ = pair_stats(dfs["capstone_pergame_sched.csv"],
               dfs["capstone_pergame_dead.csv"],
               "dead-team term (D47 artifact vs sched, its era baseline)")

print("=" * 78)
print("B) POWER CURVES for the paired gate (normal approx, validated above)")
print("   pass rule = 95% CI lower bound > 0  ==  one-sided alpha 2.5%")
print(f"   MDE80 = (z.975+z.80)*sd/sqrt(n) = {Z975+Z80:.3f}*sd/sqrt(n)")
sds = [0.05, 0.10, 0.15, round(sd46, 3), round(sd62, 3)]
print(f"\n   MDE at 80% power:")
print(f"   {'sd':>8s} | {'n=1230':>10s} | {'n=3690':>10s}")
for s in sds:
    print(f"   {s:8.3f} | {(Z975+Z80)*s/np.sqrt(1230):10.5f} | "
          f"{(Z975+Z80)*s/np.sqrt(3690):10.5f}")
print("\n   Power to PASS at true effect delta (n=3690):")
deltas = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
hdr = "   " + f"{'sd':>8s} |" + "".join(f" {d:>7.4f}" for d in deltas)
print(hdr)
for s in sds:
    row = [norm.cdf(d * np.sqrt(3690) / s - Z975) for d in deltas]
    print(f"   {s:8.3f} |" + "".join(f" {p:6.1%} " for p in row))
print("\n   Power at n=1230 (single season):")
print(hdr)
for s in sds:
    row = [norm.cdf(d * np.sqrt(1230) / s - Z975) for d in deltas]
    print(f"   {s:8.3f} |" + "".join(f" {p:6.1%} " for p in row))

print("=" * 78)
print("C) NS-PORTFOLIO (BRIEF4): if the point estimates were TRUE, how many")
print("   would the single-test gate be expected to pass?")
# (delta, SE, source).  SE exact where a CI was registered; approx marked ~.
port = [
    ("3P-luck defense-only",   0.000413, 0.000423, "ffluck2_results.json (exact)"),
    ("event-recency isolation", 0.00138, 0.001117, "journal 36cffb (exact)"),
    ("comp-heavy 60/40",        0.00100, 0.000510, "~ hairline-pass D48-era regate"),
    ("dead-team term",          0.00038, 0.001280, "recomputed in A vs sched (exact)"),
    ("continuity prior",        0.00020, 0.000400, "~ es_continuity"),
    ("carry ship-confirm",      0.00083, 0.000508, "D63 addendum (exact)"),
]
zs = []
exp_pass = 0.0
for name, mu, se, src in port:
    z = mu / se
    pw = norm.cdf(z - Z975)   # power if true effect == point estimate
    zs.append(z)
    exp_pass += pw
    print(f"   {name:24s} d={mu:+.5f} SE={se:.5f} z={z:4.2f} "
          f"power-if-true={pw:5.1%}  [{src}]")
zs = np.array(zs)
k = len(zs)
print(f"   -> expected PASSes if ALL {k} are true at point estimates: "
      f"{exp_pass:.2f} of {k}  (i.e. expect to MISS ~{k-exp_pass:.1f})")
print(f"   sign test: P(all {k} positive | all null) = {0.5**k:.4f}")
stouffer = zs.sum() / np.sqrt(k)
p_naive = 1 - norm.cdf(stouffer)
# selection correction: members were chosen BECAUSE positive; under the null,
# a sign-selected z is half-normal: mean sqrt(2/pi)=.798, sd sqrt(1-2/pi)=.603
mu0 = np.sqrt(2 / np.pi)
s0 = np.sqrt(1 - 2 / np.pi)
z_sel = (zs.sum() - k * mu0) / (s0 * np.sqrt(k))
print(f"   Stouffer combined z = {stouffer:.2f} (naive p={p_naive:.4f}) — "
      f"BIASED by sign selection")
print(f"   selection-corrected (half-normal null): z = {z_sel:.2f}, "
      f"one-sided p = {1-norm.cdf(z_sel):.3f}")

print("=" * 78)
print("D) PAST-SHIP AUDIT under GATE_POLICY_V2 tiers")
ships = {
    # name: (delta, lo95, hi95, note)
    "D46 sched layer": (0.00539, 0.0024, 0.0085, "feature; global footprint"),
    "D55 coldstart fix": (0.00099, 0.00013, 0.00185, "BUG FIX (D54 audit)"),
    "D62 carry (authorizing gate)": (0.00097, 0.000085, 0.001816,
                                     "feature; pre-reg local footprint"),
    "D62 footprint gp[0,5)": (0.0154, 0.0038, 0.0269, "pre-registered basis"),
    "D62 early (gp<20 either)": (0.00387, 0.00048, 0.00724, "pre-reg window"),
    "D63 ship-confirm": (0.00083, -0.00018, 0.00181, "port confirm, NS"),
}
K = 57  # enumerated family, audit_gate_stats.py
for name, (mu, lo, hi, note) in ships.items():
    se = (hi - lo) / (2 * Z975)
    z = mu / se
    p1 = 1 - norm.cdf(z)
    tier = ("T1 solo-ship" if mu >= 0.002 else
            "T2 portfolio" if mu >= 0.0005 else "T3 reject")
    print(f"   {name:30s} d={mu:+.5f} z={z:5.2f} p1={p1:.2e} "
          f"pooled-tier={tier}  [{note}]")
# BH-FDR across the family: approximate the family's p-value set with the
# registered PASSes (from audit_gate_stats) + the ships above + ~49 nulls.
named = {
    "four-factors D21": 0.0178 / ((0.027 - 0.008) / (2 * Z975)),
    "D33 softmax": 0.044 / ((0.055 - 0.033) / (2 * Z975)),
    "cold-start D16": 0.025 / ((0.040 - 0.010) / (2 * Z975)),
    "team-home D20": 0.0020 / ((0.0036 - 0.0004) / (2 * Z975)),
    "sched D46": 0.00539 / ((0.0085 - 0.0024) / (2 * Z975)),
    "csfix D55": 0.00099 / ((0.00185 - 0.00013) / (2 * Z975)),
    "carry D62": 0.00097 / ((0.001816 - 0.000085) / (2 * Z975)),
}
ps = sorted((1 - norm.cdf(z), n) for n, z in named.items())
print(f"\n   BH-FDR q=0.10, family K={K} (worst case: all other "
      f"{K-len(ps)} tests are null with uniform p ~ not smaller):")
for i, (p, n) in enumerate(ps, start=1):
    thr = i / K * 0.10
    print(f"     rank{i} {n:18s}: p={p:.2e} vs BH thr {thr:.4f} "
          f"-> {'PASS' if p <= thr else 'fail'}")
print("BA_GATEPOWER_DONE")
