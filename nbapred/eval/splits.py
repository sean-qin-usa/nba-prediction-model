"""MULTI-SPLIT EVALUATION HARNESS + ERA STRATIFICATION (D139).

Replaces the single dev(2023-24..2025-26)/holdout(2021-22..2022-23) partition
that every gate from D46 through D136 used.  That partition is CONFOUNDED: its
boundary is 2023-10-24, which is also the start of the Player Participation
Policy, the In-Season Tournament and the new CBA apron regime.  "Passed dev,
failed holdout" was therefore ambiguous between OVERFITTING and a genuine ERA
EFFECT, and the register has one documented instance of each (D111 overfit;
D70 nonstationarity).  This module makes the two distinguishable.

WHAT IT OPERATES ON
-------------------
Any per-unit paired artifact: rows carrying a season, a control loss and a
treatment loss (or y / p_ctrl / p_treat for a binary endpoint).  A "unit" is a
game for the win-probability endpoint and a player-game for props.  Positive
delta ALWAYS means "treatment better" (loss_ctrl - loss_treat), matching every
gate in DECISIONS.md.

THE FOUR SPLITS
---------------
(a) LEAVE-ONE-SEASON-OUT — delete-one-season jackknife of the pooled estimate.
    ** THIS IS A STABILITY DIAGNOSTIC, NOT k INDEPENDENT CONFIRMATIONS. **
    Any two LOSO folds on a 5-season corpus share 3 of their 5 seasons
    (pairwise overlap 3/4 of the data), so their estimates are correlated at
    roughly 0.75 and the k folds carry about ONE corpus worth of information,
    not k.  `loso()` returns `independent_folds = 1` and an explicit
    `interpretation` string so this cannot be quietly reported as k proofs.
(b) ROLLING-ORIGIN / expanding-window walk-forward — train on seasons <= k,
    test on k+1.  THE ONLY SPLIT THAT RESPECTS CAUSALITY and the only one that
    mirrors live use.  Note that our production stack is ALREADY walk-forward
    at the parameter level (weekly refits on a trailing window), so for a
    walk-forward artifact the per-season fold estimate IS the rolling-origin
    estimate; what rolling-origin adds is the requirement to read the folds in
    chronological order and to look at the CUMULATIVE curve for drift.  What is
    NOT walk-forward in this repo is HYPOTHESIS selection (D111), which no
    re-scoring of a fixed artifact can undo.
(c) LEGACY dev/holdout — kept verbatim for continuity of every existing
    citation.  Reported alongside, never alone.
(d) BLOCK BOOTSTRAP — resamples contiguous calendar blocks (default 7 days)
    instead of independent units, so temporal correlation (a team's hot streak,
    a shared week of schedule, a league-wide rules point of emphasis) is not
    treated as n independent observations.  Compare its CI to the iid paired
    bootstrap: the ratio is the temporal design effect.

ERA DECOMPOSITION
-----------------
`era_decomposition()` splits the per-unit variance of the delta into
BETWEEN-ERA and WITHIN-ERA components and runs a DerSimonian-Laird random
effects test on the era means (Q, I-squared, tau).  tau > 0 with I-squared
above ~50% means the effect size itself moves with the era: the pooled point
estimate is then a weighted average of different things and MUST NOT be quoted
as a single number.  That is the quantitative form of the D70 verdict.

READ-ONLY: this module touches no database and no production path.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# ERA TABLE — the measured boundaries.  Backed by scripts/era_measure.py and
# documented with per-era signatures in docs/ERAS.md.  Cite the era code, not
# the season list, so future corpus growth does not silently change a gate's
# meaning.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Era:
    code: str
    name: str
    lo: str            # inclusive ISO date
    hi: str            # inclusive ISO date
    scorable: bool     # can the production stack score it at all?
    note: str = ""


ERAS: tuple[Era, ...] = (
    Era("E-3", "pre-lockout CBA", "1996-11-01", "2011-06-30", True,
        "D152 backfill. Only 2010-11 has landed with a full prior season, so "
        "only 2010-11 is scorable here. LOWEST 3PA share (.222) and fewest "
        "players used (10.22) measured anywhere; home margin +3.167."),
    Era("E-2", "post-lockout, pre-3PT-boom", "2011-12-25", "2014-06-30", True,
        "2011-12 (lockout, 990 games, densest schedule we hold — separate "
        "stratum, never pooled), 2012-13 (1,229 games; BOS@IND cancelled), "
        "2013-14 (box data still in flight at D153). 3PA share .226/.243 vs "
        ".384 in 2019-20; pace trough 93.89; home margin +2.82 / +3.23."),
    Era("E-1", "3PT ramp", "2014-10-28", "2019-06-30", True,
        "2015-16..2018-19 scorable at D153 (2014-15's carry is starved by the "
        "in-flight 2013-14). 3PA share ramps .285 -> .359; pace recovers to "
        "102.5; HIGHEST travel measured anywhere (868-904 km/team-game)."),
    Era("E0", "pre-COVID normal", "2019-10-22", "2020-03-11", False,
        "2019-20 pre-shutdown. 971 games, normal crowds, home margin +2.174. "
        "NOT SCORABLE: fit_production raises 'no carry rows from 2018-19' "
        "(D131) — there is no 2018-19 player_game_stats to carry from."),
    Era("E0H", "COVID hiatus", "2020-03-12", "2020-07-29", False,
        "Zero games played."),
    Era("E1", "Orlando bubble", "2020-07-30", "2020-10-11", False,
        "88 seeding games at ONE site: TRUE travel is 0 for every team-game. "
        "nbapred.model.travel USED to assign 1,505.5 km/team-game from the "
        "nominal host arena; FIXED at D140 (neutral-venue chaining + hiatus "
        "reset), so E1 travel/tz/elevation are now 0 as they should be. "
        "Still NOT SCORABLE: no player_game_stats rows exist for these 88 "
        "games at all, and E1's schedule density is unlike any other era."),
    Era("E2", "no-crowd compressed", "2020-12-22", "2021-05-16", False,
        "72-game season, limited/zero crowds, densest schedule in the corpus. "
        "Home margin +0.944 = -1.014 pts vs the 2021-26 baseline (D131). "
        "NOT IN THE EVAL CORPUS (D131 section 6: +17.6% n buys only 7.8% "
        "tighter MDE and imports a measured regime break)."),
    Era("E3", "re-entry + Omicron", "2021-10-19", "2022-04-10", True,
        "Crowds return; Omicron wave spikes absence in Dec 2021. "
        "Sub-stratum E3-OMI below."),
    Era("E4", "post-COVID baseline", "2022-10-18", "2023-04-09", True,
        "The season the owner's taxonomy called 'the only clean normal one'. "
        "MEASURED CORRECTION: it is the corpus's HOME-ADVANTAGE OUTLIER "
        "(home WR .5805 vs .5423-.5520 in all four others, ~2.4 sigma). "
        "Clean of COVID, not clean of era effects."),
    Era("E5", "PPP + In-Season Tournament + new CBA", "2023-10-24", "2024-04-14",
        True,
        "Player Participation Policy, first NBA Cup (game_id prefix 006 first "
        "appears in 2023-24 — measured, not asserted), CBA apron regime "
        "begins. THIS IS THE LEGACY DEV/HOLDOUT BOUNDARY."),
    Era("E6", "apron maturing", "2024-10-22", "2026-04-30", True,
        "2024-25 + 2025-26. Second-apron hard constraints bind; measured "
        "core-DNP rate is the corpus maximum (.2502 / .2435)."),
)

SUB_ERAS: tuple[Era, ...] = (
    Era("E3-OMI", "Omicron wave (acute)", "2021-12-13", "2022-01-02", True,
        "Boundary set BY THE DATA, not by the calendar: ISO weeks 2021-W50..W52 "
        "carry core-player DNP 0.3033 against a 0.1332 pre-wave baseline "
        "(2021-W42..W49) — a 2.28x jump. The same calendar weeks a year later "
        "are flat (0.1696 vs 0.1614). Elevated but sub-acute through 2022-W05 "
        "(0.2098)."),
)

# Seasons the production stack can score at all (D131).
SCORABLE_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

# The legacy partition, preserved verbatim for citation continuity.
LEGACY_DEV = ("2023-24", "2024-25", "2025-26")
LEGACY_HOLDOUT = ("2021-22", "2022-23")

SEASON_ERA = {"2021-22": "E3", "2022-23": "E4", "2023-24": "E5",
              "2024-25": "E6", "2025-26": "E6",
              "2020-21": "E2", "2019-20": "E0",
              # D152/D153 historical seasons.  Codes and boundaries come from
              # docs/ERAS.md §1/§7; adding them here does NOT widen the eval
              # corpus (SCORABLE_SEASONS below is unchanged) — it lets the era
              # decomposition run on a historical artifact.
              "2010-11": "E-3", "2011-12": "E-2", "2012-13": "E-2",
              "2013-14": "E-2", "2014-15": "E-1", "2015-16": "E-1",
              "2016-17": "E-1", "2017-18": "E-1", "2018-19": "E-1"}

# Chronological order for reporting (pooled/era tables and charts).
ERA_ORDER = ["E-3", "E-2", "E-1", "E0", "E1", "E2", "E3", "E4", "E5", "E6"]


def era_of(season: str, date=None) -> str:
    """Era code for a unit.  Date is optional and only refines 2019-20
    (pre-shutdown vs bubble)."""
    if date is not None and season == "2019-20":
        d = str(date)[:10]
        for e in ERAS:
            if e.lo <= d <= e.hi:
                return e.code
    return SEASON_ERA.get(season, "UNK")


def in_sub_era(code: str, date) -> bool:
    for e in SUB_ERAS:
        if e.code == code:
            return e.lo <= str(date)[:10] <= e.hi
    return False


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------

def _ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


@dataclass
class Panel:
    """Per-unit paired deltas with season / date / cluster labels.

    d[i] = loss_ctrl[i] - loss_treat[i]; POSITIVE = treatment better.
    """
    season: np.ndarray
    d: np.ndarray
    date: np.ndarray | None = None
    cluster: np.ndarray | None = None
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.season = np.asarray(self.season).astype(str)
        self.d = np.asarray(self.d, float)
        if self.date is not None:
            self.date = np.asarray([str(x)[:10] for x in self.date])
        if self.cluster is not None:
            self.cluster = np.asarray(self.cluster)
        if len(self.season) != len(self.d):
            raise ValueError("season and d length mismatch")

    # ---- constructors ----------------------------------------------------
    @classmethod
    def from_logloss(cls, season, y, p_ctrl, p_treat, date=None, cluster=None,
                     label="", meta=None):
        return cls(season, _ll(y, p_ctrl) - _ll(y, p_treat), date, cluster,
                   label, meta or {})

    @classmethod
    def from_losses(cls, season, loss_ctrl, loss_treat, date=None, cluster=None,
                    label="", meta=None):
        return cls(season, np.asarray(loss_ctrl, float) - np.asarray(loss_treat, float),
                   date, cluster, label, meta or {})

    # ---- views -----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.d)

    @property
    def seasons(self) -> list[str]:
        return sorted(set(self.season.tolist()))

    @property
    def era(self) -> np.ndarray:
        dates = self.date if self.date is not None else [None] * self.n
        return np.array([era_of(s, dd) for s, dd in zip(self.season, dates)])

    def subset(self, mask) -> "Panel":
        mask = np.asarray(mask, bool)
        return Panel(self.season[mask], self.d[mask],
                     None if self.date is None else self.date[mask],
                     None if self.cluster is None else self.cluster[mask],
                     self.label, dict(self.meta))

    def by_seasons(self, seasons) -> "Panel":
        return self.subset(np.isin(self.season, list(seasons)))


# --------------------------------------------------------------------------
# Bootstraps
# --------------------------------------------------------------------------

def paired_bootstrap(d, B: int = 2000, seed: int = 20260801, cluster=None,
                     alpha: float = 0.05) -> dict:
    """Percentile paired bootstrap of mean(d).  If `cluster` is given, whole
    clusters are resampled (the D133 by-player convention)."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return dict(n=0, est=float("nan"), lo=float("nan"), hi=float("nan"),
                    se=float("nan"), p_wrongside=float("nan"), sig=False)
    rng = np.random.default_rng(seed)
    if cluster is None:
        idx = rng.integers(0, n, size=(B, n))
        boots = d[idx].mean(axis=1)
    else:
        cl = np.asarray(cluster)
        keys, inv = np.unique(cl, return_inverse=True)
        groups = [np.flatnonzero(inv == i) for i in range(len(keys))]
        sums = np.array([d[g].sum() for g in groups])
        cnts = np.array([len(g) for g in groups], float)
        pick = rng.integers(0, len(groups), size=(B, len(groups)))
        boots = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    est = float(d.mean())
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_wrong = float(min((boots <= 0).mean(), (boots >= 0).mean()))
    return dict(n=int(n), est=est, lo=float(lo), hi=float(hi),
                se=float(boots.std(ddof=1)), p_wrongside=p_wrong,
                sig=bool(lo > 0 or hi < 0))


def block_bootstrap(d, date, B: int = 2000, seed: int = 20260801,
                    block_days: int = 7, alpha: float = 0.05) -> dict:
    """Non-overlapping calendar-block bootstrap.

    Units are bucketed into contiguous `block_days`-long calendar blocks and
    WHOLE BLOCKS are resampled with replacement.  This is the temporally
    honest bootstrap: within a week, games share schedule structure, roster
    news and league-wide officiating emphasis, so the iid paired bootstrap
    understates the SE whenever the treatment effect is correlated in time.
    """
    d = np.asarray(d, float)
    if date is None:
        raise ValueError("block_bootstrap needs dates")
    days = np.array([(dt.date.fromisoformat(str(x)[:10]) - dt.date(2000, 1, 1)).days
                     for x in date])
    blk = days // int(block_days)
    keys, inv = np.unique(blk, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(keys))]
    sums = np.array([d[g].sum() for g in groups])
    cnts = np.array([len(g) for g in groups], float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), size=(B, len(groups)))
    boots = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(n=int(len(d)), n_blocks=int(len(groups)), block_days=int(block_days),
                est=float(d.mean()), lo=float(lo), hi=float(hi),
                se=float(boots.std(ddof=1)),
                p_wrongside=float(min((boots <= 0).mean(), (boots >= 0).mean())),
                sig=bool(lo > 0 or hi < 0))


def mde80(d) -> float:
    """GATE_POLICY_V2 section 5.5 power floor, unchanged: 2.802*sd/sqrt(n)."""
    d = np.asarray(d, float)
    return float(2.802 * d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 else float("nan")


# --------------------------------------------------------------------------
# CLUSTERING — the anti-conservatism correction (D139 addition)
#
# Every SIDES gate in this repo (of_transfer_ablation, apr_program/D73,
# es2_hardstop/D62, tv_gate/D136, pg_urgency2/D130, ba_gatepower itself) used
#   rng.integers(0, len(delta), size=(B, len(delta)))
# i.e. an i.i.d. resample of individual GAME deltas.  Per-game deltas inside a
# season are NOT independent: the feature's coefficients (schedule-layer betas,
# tank k, carry weights, the props ramp table) are estimated from shared data
# and move every game in the season together.  An i.i.d. resample therefore
# UNDERSTATES the SE and the gate is ANTI-CONSERVATIVE.  Same failure mode as
# the sister football project's retracted "we beat the market" claim
# (/hdd/steveqin/sean_dev/football_exercise, COMPARISON.md section 3:
# 0.9540 vs 0.9572 looked like a win; clustered by season it was -0.0031
# CI[-0.0069,+0.0008] ns).
# --------------------------------------------------------------------------

def icc_oneway(d, group) -> dict:
    """One-way random-effects intra-class correlation of the per-unit delta
    within `group` (normally the season), plus the implied design effect.

    ICC = (MSB - MSW) / (MSB + (n0-1) MSW);  DEFF = 1 + (n0-1) ICC.
    A design effect of D means the honest SE is sqrt(D) x the i.i.d. SE.
    """
    d = np.asarray(d, float)
    g = np.asarray(group)
    keys = sorted(set(g.tolist()))
    K, N = len(keys), len(d)
    if K < 2 or N <= K:
        return dict(K=K, N=N, icc=float("nan"), deff=float("nan"),
                    se_inflation=float("nan"))
    gm = d.mean()
    ns = np.array([int((g == k).sum()) for k in keys], float)
    means = np.array([d[g == k].mean() for k in keys])
    msb = float((ns * (means - gm) ** 2).sum() / (K - 1))
    msw = float(sum(((d[g == k] - d[g == k].mean()) ** 2).sum() for k in keys) / (N - K))
    n0 = (N - (ns ** 2).sum() / N) / (K - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else 0.0
    deff = 1.0 + (n0 - 1) * icc
    return dict(K=K, N=N, n0=float(n0), MSB=msb, MSW=msw,
                icc=float(icc), deff=float(max(deff, 0.0)),
                se_inflation=float(math.sqrt(max(deff, 1e-12))))


def cluster_mean_t_interval(d, group, alpha: float = 0.05) -> dict:
    """The small-K honest interval: treat each cluster MEAN as one observation
    and use a t interval with K-1 dof.

    With only 5 scorable seasons this is deliberately brutal (t_{.975,4} =
    2.776).  It is the right answer to "how sure are we that this generalises
    to a NEW season", which is exactly the question a gate is asking.  Report
    it next to the i.i.d. CI; do not choose one and hide the other.
    """
    d = np.asarray(d, float)
    g = np.asarray(group)
    keys = sorted(set(g.tolist()))
    ns = np.array([int((g == k).sum()) for k in keys], float)
    means = np.array([d[g == k].mean() for k in keys])
    K = len(keys)
    if K < 2:
        return dict(K=K, est=float(d.mean()), lo=float("nan"), hi=float("nan"),
                    sig=False)
    est = float((ns * means).sum() / ns.sum())
    se = float(means.std(ddof=1) / math.sqrt(K))
    tcrit = _t_crit(K - 1, alpha)
    lo, hi = float(means.mean() - tcrit * se), float(means.mean() + tcrit * se)
    return dict(K=K, dof=K - 1, t_crit=tcrit, est=est,
                unweighted_mean=float(means.mean()), se=se,
                lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0),
                cluster_means={k: float(m) for k, m in zip(keys, means)})


_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
            20: 2.086, 30: 2.042, 60: 2.000}


def _t_crit(dof: int, alpha: float = 0.05) -> float:
    if alpha != 0.05:
        raise ValueError("only two-sided 95% is tabulated")
    if dof in _T_TABLE:
        return _T_TABLE[dof]
    keys = sorted(_T_TABLE)
    for k in keys:
        if dof < k:
            return _T_TABLE[k]
    return 1.96


def clustering_report(panel: Panel, B: int = 2000, seed: int = 20260801) -> dict:
    """i.i.d. vs season-clustered vs week-block inference, side by side."""
    iid = paired_bootstrap(panel.d, B, seed)
    seas = paired_bootstrap(panel.d, B, seed, cluster=panel.season)
    out = dict(
        iid=iid, season_cluster_boot=seas,
        season_mean_t=cluster_mean_t_interval(panel.d, panel.season),
        icc_season=icc_oneway(panel.d, panel.season),
        design_effect_season=float(seas["se"] / iid["se"]) if iid["se"] else float("nan"),
    )
    if panel.date is not None:
        mon = np.array([str(x)[:7] for x in panel.date])
        wk = block_bootstrap(panel.d, panel.date, B, seed, 7)
        out["month_cluster_boot"] = paired_bootstrap(panel.d, B, seed, cluster=mon)
        out["week_block_boot"] = wk
        out["icc_month"] = icc_oneway(panel.d, mon)
        out["design_effect_week"] = float(wk["se"] / iid["se"]) if iid["se"] else float("nan")
    out["flips"] = bool(iid["sig"] and not seas["sig"])
    out["flips_on_t"] = bool(iid["sig"] and not out["season_mean_t"]["sig"])
    return out


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def per_season(panel: Panel, B: int = 2000, seed: int = 20260801) -> list[dict]:
    out = []
    for s in panel.seasons:
        p = panel.by_seasons([s])
        r = paired_bootstrap(p.d, B, seed, p.cluster)
        r.update(season=s, era=SEASON_ERA.get(s, "UNK"), mde80=mde80(p.d))
        out.append(r)
    return out


def loso(panel: Panel, B: int = 2000, seed: int = 20260801) -> dict:
    """LEAVE-ONE-SEASON-OUT.  Two readouts per fold:

      test_on   = estimate ON the left-out season (what people usually mean)
      jackknife = estimate on the REMAINING seasons (the influence diagnostic:
                  how much does the pooled verdict depend on this one season?)

    The folds are NOT independent — see `interpretation`.
    """
    ss = panel.seasons
    k = len(ss)
    folds = []
    for s in ss:
        held = panel.by_seasons([s])
        rest = panel.by_seasons([x for x in ss if x != s])
        t = paired_bootstrap(held.d, B, seed, held.cluster)
        j = paired_bootstrap(rest.d, B, seed, rest.cluster)
        folds.append(dict(left_out=s, era=SEASON_ERA.get(s, "UNK"),
                          test_on=t, jackknife=j,
                          influence=float(j["est"] - panel.d.mean())))
    ests = [f["test_on"]["est"] for f in folds]
    overlap = (k - 2) / (k - 1) if k > 1 else 0.0
    return dict(
        k=k, folds=folds,
        mean_of_folds=float(np.mean(ests)), sd_of_folds=float(np.std(ests, ddof=1)) if k > 1 else float("nan"),
        min_fold=float(np.min(ests)), max_fold=float(np.max(ests)),
        sign_consistency=f"{sum(1 for e in ests if e > 0)}/{k}",
        jackknife_range=[float(min(f["jackknife"]["est"] for f in folds)),
                         float(max(f["jackknife"]["est"] for f in folds))],
        independent_folds=1,
        pairwise_data_overlap=float(overlap),
        interpretation=(
            f"STABILITY DIAGNOSTIC ONLY. Any two of these {k} jackknife folds "
            f"share {overlap:.0%} of their data, so they are ~1 corpus of "
            f"information, NOT {k} independent confirmations. Report the SPREAD "
            f"and the INFLUENCE of each season; never count folds as replications."),
    )


def rolling_origin(panel: Panel, B: int = 2000, seed: int = 20260801,
                   min_train: int = 1) -> dict:
    """EXPANDING-WINDOW WALK-FORWARD: train on seasons <= k, test on k+1.

    The only split here that respects causality.  For an artifact produced by a
    walk-forward stack (every production number in this repo), the fold
    estimate on season k+1 is already free of future information, so this
    function reads the SAME per-season deltas as `per_season` — but in
    chronological order, with the cumulative curve attached, and with the first
    `min_train` seasons excluded from the headline because their parameters
    burned in inside the corpus.
    """
    ss = panel.seasons
    folds = []
    for i in range(min_train, len(ss)):
        train, test = ss[:i], ss[i]
        te = panel.by_seasons([test])
        cum = panel.by_seasons(ss[min_train:i + 1])
        r = paired_bootstrap(te.d, B, seed, te.cluster)
        c = paired_bootstrap(cum.d, B, seed, cum.cluster)
        folds.append(dict(train=list(train), test=test,
                          era=SEASON_ERA.get(test, "UNK"),
                          n_train_seasons=len(train), fold=r, cumulative=c,
                          mde80=mde80(te.d)))
    ests = [f["fold"]["est"] for f in folds]
    trend = float("nan")
    if len(ests) > 2:
        x = np.arange(len(ests), dtype=float)
        trend = float(np.polyfit(x, np.array(ests), 1)[0])
    return dict(n_folds=len(folds), folds=folds,
                mean_of_folds=float(np.mean(ests)) if ests else float("nan"),
                sd_of_folds=float(np.std(ests, ddof=1)) if len(ests) > 1 else float("nan"),
                sign_consistency=f"{sum(1 for e in ests if e > 0)}/{len(ests)}",
                drift_per_season=trend,
                final_cumulative=folds[-1]["cumulative"] if folds else None,
                interpretation=("Causal / live-mirroring. These folds ARE "
                                "independent in their test data. Sign consistency "
                                "here is the closest thing the corpus offers to "
                                "replication."))


def legacy_split(panel: Panel, B: int = 2000, seed: int = 20260801,
                 dev=LEGACY_DEV, holdout=LEGACY_HOLDOUT) -> dict:
    """The pre-D139 single partition, kept verbatim for citation continuity."""
    dv, ho = panel.by_seasons(dev), panel.by_seasons(holdout)
    out = dict(dev_seasons=list(dev), holdout_seasons=list(holdout),
               dev=paired_bootstrap(dv.d, B, seed, dv.cluster),
               holdout=paired_bootstrap(ho.d, B, seed, ho.cluster),
               pooled=paired_bootstrap(panel.d, B, seed, panel.cluster))
    out["transfer_ratio"] = (out["holdout"]["est"] / out["dev"]["est"]
                             if out["dev"]["est"] else float("nan"))
    out["confound_warning"] = (
        "The dev/holdout boundary (2023-10-24) IS the E5 era boundary: Player "
        "Participation Policy + In-Season Tournament + new CBA all start on it. "
        "A dev/holdout disagreement is therefore NOT evidence of overfitting "
        "until the era decomposition has been read.")
    return out


# --------------------------------------------------------------------------
# Era decomposition
# --------------------------------------------------------------------------

def era_decomposition(panel: Panel, B: int = 2000, seed: int = 20260801) -> dict:
    """Between-era vs within-era variance + DerSimonian-Laird heterogeneity.

    tau2 > 0 and I2 above ~50% => the effect SIZE moves with the era and the
    pooled point estimate is an average over different things.
    """
    era = panel.era
    codes = sorted(set(era.tolist()))
    per = []
    for c in codes:
        m = era == c
        p = panel.subset(m)
        r = paired_bootstrap(p.d, B, seed, p.cluster)
        r.update(era=c, n=int(m.sum()),
                 seasons=sorted(set(panel.season[m].tolist())))
        per.append(r)

    # variance decomposition on the raw per-unit deltas
    grand = float(panel.d.mean())
    n = panel.n
    between = sum(int((era == c).sum()) * (panel.d[era == c].mean() - grand) ** 2
                  for c in codes) / n
    within = sum(float(((panel.d[era == c] - panel.d[era == c].mean()) ** 2).sum())
                 for c in codes) / n
    total = float(((panel.d - grand) ** 2).mean())

    # DerSimonian-Laird on the era means
    theta = np.array([p["est"] for p in per])
    sev = np.array([p["se"] for p in per])
    ok = np.isfinite(theta) & np.isfinite(sev) & (sev > 0)
    theta, sev = theta[ok], sev[ok]
    if len(theta) > 1:
        w = 1.0 / sev ** 2
        fixed = float((w * theta).sum() / w.sum())
        Q = float((w * (theta - fixed) ** 2).sum())
        dof = len(theta) - 1
        C = float(w.sum() - (w ** 2).sum() / w.sum())
        tau2 = max(0.0, (Q - dof) / C) if C > 0 else 0.0
        I2 = max(0.0, (Q - dof) / Q) if Q > 0 else 0.0
        p_Q = _chi2_sf(Q, dof)
    else:
        fixed, Q, dof, tau2, I2, p_Q = float("nan"), float("nan"), 0, 0.0, 0.0, float("nan")

    # DECISION RULE: the Q test, at p=0.10.  I2 is reported as the MAGNITUDE of
    # heterogeneity but is not the decision, because with only 3-4 era groups
    # I2 exceeds 50% about 11% of the time under a true null (E[Q]=dof, and
    # I2>0.5 <=> Q>2*dof) — using it as the rule would flag one in nine honest
    # era-stable effects.  Conversely a non-significant Q at dof<=3 is WEAK
    # evidence of stability: the test is underpowered, so `era_stable=True`
    # means "not shown to move with the era", never "shown not to".
    era_stable = bool(not np.isfinite(p_Q) or p_Q > 0.10)
    high_i2 = bool(np.isfinite(I2) and I2 >= 0.5)
    if era_stable and high_i2:
        verdict = ("ERA-STABLE (WEAK): Q is not significant, but I2 is at or "
                   f"above 50% and the test has only {dof} dof — treat "
                   "stability as unproven and publish the per-era estimates "
                   "anyway.")
    elif era_stable:
        verdict = ("ERA-STABLE: era means are consistent with one common effect "
                   "(Q not significant). NOTE: with %d dof this test is "
                   "underpowered; it cannot prove stability." % dof)
    else:
        verdict = ("ERA-UNSTABLE: the era means differ by more than sampling "
                   "noise. The pooled estimate averages different regimes and "
                   "must not be quoted alone (D70 class).")
    return dict(per_era=per, grand_mean=grand,
                var_total=total, var_between_era=float(between),
                var_within_era=float(within),
                between_share=float(between / total) if total > 0 else float("nan"),
                Q=Q, dof=dof, p_heterogeneity=p_Q, I2=I2, tau=math.sqrt(tau2),
                fixed_effect=fixed,
                era_stable=era_stable, high_I2=high_i2,
                heterogeneity_underpowered=bool(dof <= 3),
                verdict=verdict)


def _chi2_sf(x: float, k: int) -> float:
    """Upper tail of chi-square with k dof (Wilson-Hilferty; adequate here)."""
    if k <= 0 or not np.isfinite(x):
        return float("nan")
    if x <= 0:
        return 1.0
    z = ((x / k) ** (1.0 / 3.0) - (1 - 2.0 / (9 * k))) / math.sqrt(2.0 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# --------------------------------------------------------------------------
# Full report
# --------------------------------------------------------------------------

def full_report(panel: Panel, B: int = 2000, seed: int = 20260801,
                block_days: int = 7) -> dict:
    rep = dict(
        label=panel.label, n=panel.n, seasons=panel.seasons,
        pooled=paired_bootstrap(panel.d, B, seed, panel.cluster),
        pooled_mde80=mde80(panel.d),
        per_season=per_season(panel, B, seed),
        rolling_origin=rolling_origin(panel, B, seed),
        loso=loso(panel, B, seed),
        legacy=legacy_split(panel, B, seed),
        era=era_decomposition(panel, B, seed),
        clustering=clustering_report(panel, B, seed),
        meta=panel.meta,
    )
    if panel.date is not None:
        rep["block_bootstrap"] = block_bootstrap(panel.d, panel.date, B, seed,
                                                 block_days)
        bb, ib = rep["block_bootstrap"]["se"], rep["pooled"]["se"]
        rep["temporal_design_effect"] = float(bb / ib) if ib else float("nan")
    rep["verdict"] = adjudicate(rep)
    return rep


def adjudicate(rep: dict) -> dict:
    """The V3 pass/fail reading when folds disagree.  Mechanical, so it cannot
    be argued after the fact.  This does NOT replace the V2 tier rules — it is
    the split-rigor overlay applied on top of them."""
    ro = rep["rolling_origin"]
    lo = rep["loso"]
    era = rep["era"]
    pooled = rep["pooled"]
    ro_signs = [f["fold"]["est"] > 0 for f in ro["folds"]]
    n_ro = len(ro_signs)
    ro_pos = sum(ro_signs)
    flags = []
    if not pooled["sig"]:
        flags.append("pooled CI straddles zero")
    if n_ro and ro_pos < n_ro:
        flags.append(f"rolling-origin sign flips in {n_ro - ro_pos}/{n_ro} folds")
    if not era["era_stable"]:
        flags.append(f"era-unstable (I2={era['I2']:.0%}, p_Q={era['p_heterogeneity']:.3f})")
    elif era.get("high_I2"):
        flags.append(f"era heterogeneity I2={era['I2']:.0%} with Q not significant "
                     f"(p={era['p_heterogeneity']:.3f}) — stability UNPROVEN at "
                     f"{era['dof']} dof; publish per-era estimates")
    if rep.get("temporal_design_effect", 1.0) > 1.25:
        flags.append(f"temporal design effect {rep['temporal_design_effect']:.2f} "
                     "— iid bootstrap understates the SE")
    jr = lo["jackknife_range"]
    if pooled["est"] and (jr[0] * jr[1] <= 0):
        flags.append("a single season flips the sign of the pooled estimate")
    if pooled["sig"] and rep.get("block_bootstrap") and not rep["block_bootstrap"]["sig"]:
        flags.append("significance does not survive the block bootstrap")
    cl = rep.get("clustering")
    if cl:
        if cl["flips"]:
            flags.append("SIG under i.i.d. resampling, NOT SIG under a "
                         "season-clustered bootstrap (anti-conservative gate)")
        if cl["flips_on_t"]:
            flags.append("SIG under i.i.d. resampling, NOT SIG under the "
                         "season-mean t interval (K-1 dof)")
        de = cl.get("design_effect_season")
        if de and np.isfinite(de) and de > 1.25:
            flags.append(f"season design effect {de:.2f} — the i.i.d. CI is "
                         f"{de:.2f}x too narrow")

    if not flags:
        tier = "MULTI-SPLIT PASS"
    elif (pooled["sig"] and n_ro and ro_pos == n_ro and era["era_stable"]):
        tier = "MULTI-SPLIT PASS (with notes)"
    elif pooled["sig"] and era["era_stable"]:
        tier = "PASS-WEAK: pooled significant and era-stable, folds noisy"
    elif pooled["sig"] and not era["era_stable"]:
        tier = ("ERA-CONDITIONAL: significant in aggregate but the effect size "
                "moves with the era — ship only with the era stated, and never "
                "extrapolate to a NEW era")
    else:
        tier = "NO-PASS under multi-split"
    return dict(tier=tier, flags=flags,
                rolling_origin_sign=f"{ro_pos}/{n_ro}",
                era_stable=era["era_stable"],
                pooled_sig=pooled["sig"])


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def format_report(rep: dict) -> str:
    L = []
    a = L.append
    a(f"### {rep['label']}   n={rep['n']}  seasons={','.join(rep['seasons'])}")
    p = rep["pooled"]
    a(f"POOLED      {p['est']:+.5f} CI({p['lo']:+.5f},{p['hi']:+.5f}) "
      f"{'SIG' if p['sig'] else 'ns '}  MDE80={rep['pooled_mde80']:.5f}  "
      f"p_wrongside={p['p_wrongside']:.3f}")
    if "block_bootstrap" in rep:
        b = rep["block_bootstrap"]
        a(f"BLOCK-BOOT  {b['est']:+.5f} CI({b['lo']:+.5f},{b['hi']:+.5f}) "
          f"{'SIG' if b['sig'] else 'ns '}  ({b['n_blocks']} x {b['block_days']}d blocks; "
          f"design effect {rep['temporal_design_effect']:.2f})")
    a("")
    a("ROLLING-ORIGIN (causal; train<=k, test k+1)")
    for f in rep["rolling_origin"]["folds"]:
        r = f["fold"]
        a(f"  train {len(f['train'])}s -> test {f['test']} [{f['era']}] "
          f"{r['est']:+.5f} CI({r['lo']:+.5f},{r['hi']:+.5f}) "
          f"{'SIG' if r['sig'] else 'ns '}  n={r['n']}  MDE80={f['mde80']:.5f}")
    ro = rep["rolling_origin"]
    a(f"  sign {ro['sign_consistency']}  mean {ro['mean_of_folds']:+.5f}  "
      f"sd {ro['sd_of_folds']:.5f}  drift/season {ro['drift_per_season']:+.5f}")
    a("")
    a("LEAVE-ONE-SEASON-OUT (STABILITY DIAGNOSTIC — NOT k CONFIRMATIONS)")
    for f in rep["loso"]["folds"]:
        t, j = f["test_on"], f["jackknife"]
        a(f"  -{f['left_out']} [{f['era']}]  test_on {t['est']:+.5f} "
          f"CI({t['lo']:+.5f},{t['hi']:+.5f}) {'SIG' if t['sig'] else 'ns '} | "
          f"rest {j['est']:+.5f} {'SIG' if j['sig'] else 'ns '} "
          f"(influence {f['influence']:+.5f})")
    a(f"  fold spread {rep['loso']['min_fold']:+.5f}..{rep['loso']['max_fold']:+.5f}  "
      f"sign {rep['loso']['sign_consistency']}  independent folds="
      f"{rep['loso']['independent_folds']} (pairwise overlap "
      f"{rep['loso']['pairwise_data_overlap']:.0%})")
    a("")
    lg = rep["legacy"]
    a(f"LEGACY dev/holdout   dev {lg['dev']['est']:+.5f} "
      f"CI({lg['dev']['lo']:+.5f},{lg['dev']['hi']:+.5f}) "
      f"{'SIG' if lg['dev']['sig'] else 'ns '} | holdout {lg['holdout']['est']:+.5f} "
      f"CI({lg['holdout']['lo']:+.5f},{lg['holdout']['hi']:+.5f}) "
      f"{'SIG' if lg['holdout']['sig'] else 'ns '} | transfer "
      f"{lg['transfer_ratio']:.2f}")
    a("")
    e = rep["era"]
    a("ERA DECOMPOSITION")
    for x in e["per_era"]:
        a(f"  {x['era']:5s} n={x['n']:5d} {x['est']:+.5f} "
          f"CI({x['lo']:+.5f},{x['hi']:+.5f}) {'SIG' if x['sig'] else 'ns '} "
          f"{','.join(x['seasons'])}")
    a(f"  between-era share of variance {e['between_share']:.4%}  "
      f"Q={e['Q']:.2f} dof={e['dof']} p={e['p_heterogeneity']:.3f} "
      f"I2={e['I2']:.0%} tau={e['tau']:.5f}")
    a(f"  {e['verdict']}")
    if "clustering" in rep:
        c = rep["clustering"]
        a("")
        a("CLUSTERED INFERENCE (the anti-conservatism check)")
        for k, lab in (("iid", "i.i.d. game-level (legacy)"),
                       ("week_block_boot", "week blocks"),
                       ("month_cluster_boot", "month clusters"),
                       ("season_cluster_boot", "SEASON clusters")):
            if k in c:
                x = c[k]
                a(f"  {lab:28s} {x['est']:+.5f} CI({x['lo']:+.5f},{x['hi']:+.5f}) "
                  f"{'SIG' if x['sig'] else 'ns '} se={x['se']:.5f}")
        t = c["season_mean_t"]
        a(f"  {'season-mean t (dof=' + str(t['dof']) + ')':28s} {t['est']:+.5f} "
          f"CI({t['lo']:+.5f},{t['hi']:+.5f}) {'SIG' if t['sig'] else 'ns '} "
          f"t_crit={t['t_crit']}")
        ic = c["icc_season"]
        a(f"  intra-season ICC {ic['icc']:+.5f}  design effect {ic['deff']:.2f} "
          f"(SE x{ic['se_inflation']:.2f})  |  bootstrap design effect "
          f"{c['design_effect_season']:.2f}")
    a("")
    v = rep["verdict"]
    a(f"VERDICT: {v['tier']}")
    for f in v["flags"]:
        a(f"    flag: {f}")
    return "\n".join(L)
