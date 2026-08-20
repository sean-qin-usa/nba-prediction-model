"""D139 multi-split harness: the properties the POLICY depends on.

These are not "does numpy work" tests. Each one pins a claim that
docs/GATE_POLICY_V2.md (V3 sections 8-11) makes about the harness, so the
policy cannot silently drift away from the code that implements it.
"""
import numpy as np
import pytest

from nbapred.eval import splits as S


def _panel(effect_by_season, n_per=1230, sd=0.09, seed=7, dates=True):
    rng = np.random.default_rng(seed)
    seasons, d, dt_ = [], [], []
    for i, (s, eff) in enumerate(sorted(effect_by_season.items())):
        seasons += [s] * n_per
        d.append(rng.normal(eff, sd, n_per))
        if dates:
            base = np.datetime64(f"{s[:4]}-10-20")
            dt_ += [str(base + np.timedelta64(int(j * 160 / n_per), "D"))
                    for j in range(n_per)]
    return S.Panel(np.array(seasons), np.concatenate(d),
                   date=np.array(dt_) if dates else None, label="synthetic")


# ---------------------------------------------------------------- era table

def test_era_table_covers_every_scorable_season():
    for s in S.SCORABLE_SEASONS:
        assert S.era_of(s) in {"E3", "E4", "E5", "E6"}
    # the legacy split boundary IS an era boundary — the whole point of D139
    assert S.era_of(S.LEGACY_HOLDOUT[-1]) != S.era_of(S.LEGACY_DEV[0])


def test_unscorable_eras_are_flagged_unscorable():
    by = {e.code: e for e in S.ERAS}
    for code in ("E0", "E1", "E2"):
        assert by[code].scorable is False
    for code in ("E3", "E4", "E5", "E6"):
        assert by[code].scorable is True


def test_bubble_dates_resolve_to_E1():
    assert S.era_of("2019-20", "2020-08-05") == "E1"
    assert S.era_of("2019-20", "2020-01-05") == "E0"


# ------------------------------------------------------------------- LOSO

def test_loso_declares_itself_a_stability_diagnostic():
    """POLICY: LOSO must never be presentable as k independent confirmations."""
    rep = S.loso(_panel({s: 0.002 for s in S.SCORABLE_SEASONS}), B=200)
    assert rep["independent_folds"] == 1
    assert rep["k"] == 5
    assert rep["pairwise_data_overlap"] == pytest.approx(0.75)
    assert "NOT" in rep["interpretation"]


def test_loso_influence_sums_to_zero_ish():
    p = _panel({s: 0.002 for s in S.SCORABLE_SEASONS})
    rep = S.loso(p, B=200)
    infl = [f["influence"] for f in rep["folds"]]
    assert abs(sum(infl)) < 5e-4


# --------------------------------------------------------- rolling origin

def test_rolling_origin_is_causal_and_drops_the_first_season():
    p = _panel({s: 0.002 for s in S.SCORABLE_SEASONS})
    rep = S.rolling_origin(p, B=200)
    assert rep["n_folds"] == 4
    for f in rep["folds"]:
        assert f["test"] not in f["train"]
        assert all(t < f["test"] for t in f["train"])


def test_rolling_origin_detects_a_trend():
    p = _panel(dict(zip(S.SCORABLE_SEASONS, [0.008, 0.006, 0.004, 0.002, 0.000])))
    rep = S.rolling_origin(p, B=200)
    assert rep["drift_per_season"] < 0


# -------------------------------------------------------- era decomposition

def test_era_decomposition_flags_a_real_era_effect():
    """A term worth +0.02 only in E6 must come back ERA-UNSTABLE."""
    p = _panel(dict(zip(S.SCORABLE_SEASONS, [0.0, 0.0, 0.0, 0.02, 0.02])))
    e = S.era_decomposition(p, B=400)
    assert e["era_stable"] is False
    assert e["I2"] > 0.5


def test_era_decomposition_is_calibrated_on_homogeneous_effects():
    """CALIBRATION, not a single draw: under a true common effect the
    heterogeneity flag must fire only occasionally (nominal 10%). A single
    seed is a coin flip at 3 dof and would make this test a lottery."""
    stable = sum(S.era_decomposition(
        _panel({s: 0.004 for s in S.SCORABLE_SEASONS}, seed=k), B=300)["era_stable"]
        for k in range(12))
    assert stable >= 9, f"only {stable}/12 homogeneous panels read as era-stable"


# ------------------------------------------------------------- clustering

def test_icc_detects_a_season_level_shift():
    """A per-season common shift is exactly the dependence the i.i.d.
    bootstrap misses; the ICC must see it and the design effect must exceed 1."""
    p = _panel(dict(zip(S.SCORABLE_SEASONS,
                        [0.010, -0.006, 0.008, -0.004, 0.006])), sd=0.09)
    ic = S.icc_oneway(p.d, p.season)
    assert ic["icc"] > 0
    assert ic["deff"] > 1.5
    assert ic["se_inflation"] > 1.2


def test_iid_and_season_cluster_agree_when_there_is_no_season_effect():
    p = _panel({s: 0.002 for s in S.SCORABLE_SEASONS}, sd=0.09, seed=11)
    iid = S.paired_bootstrap(p.d, B=800, seed=1)
    cl = S.paired_bootstrap(p.d, B=800, seed=1, cluster=p.season)
    assert 0.6 < cl["se"] / iid["se"] < 1.7


def test_cluster_mean_t_uses_K_minus_1_dof():
    p = _panel({s: 0.002 for s in S.SCORABLE_SEASONS})
    t = S.cluster_mean_t_interval(p.d, p.season)
    assert t["dof"] == 4
    assert t["t_crit"] == pytest.approx(2.776)


# --------------------------------------------------------- block bootstrap

def test_block_bootstrap_runs_and_reports_blocks():
    p = _panel({s: 0.002 for s in S.SCORABLE_SEASONS})
    b = S.block_bootstrap(p.d, p.date, B=300, block_days=7)
    assert b["n_blocks"] > 20
    assert b["lo"] < b["est"] < b["hi"]


# ------------------------------------------------------------ adjudication

def test_adjudicate_passes_a_clean_stable_effect():
    """Same calibration logic: a clean, large, homogeneous effect must be
    adjudicated a PASS in the large majority of draws."""
    tiers = [S.full_report(_panel({s: 0.006 for s in S.SCORABLE_SEASONS}, seed=k),
                           B=300)["verdict"]["tier"] for k in range(10)]
    n_pass = sum(t.startswith("MULTI-SPLIT PASS") for t in tiers)
    assert n_pass >= 7, f"only {n_pass}/10 passed: {tiers}"


def test_adjudicate_refuses_a_null():
    rep = S.full_report(_panel({s: 0.0 for s in S.SCORABLE_SEASONS}), B=400)
    assert rep["verdict"]["tier"] == "NO-PASS under multi-split"


def test_adjudicate_calls_an_era_specific_effect_era_conditional():
    rep = S.full_report(
        _panel(dict(zip(S.SCORABLE_SEASONS, [0.0, 0.0, 0.0, 0.02, 0.02]))), B=400)
    assert "ERA-CONDITIONAL" in rep["verdict"]["tier"]


def test_report_formats_without_error():
    rep = S.full_report(_panel({s: 0.003 for s in S.SCORABLE_SEASONS}), B=200)
    txt = S.format_report(rep)
    for must in ("ROLLING-ORIGIN", "LEAVE-ONE-SEASON-OUT", "ERA DECOMPOSITION",
                 "CLUSTERED INFERENCE", "STABILITY DIAGNOSTIC"):
        assert must in txt


# --------------------------------------------------- constructor equivalence

def test_from_logloss_matches_manual_log_loss():
    y = np.array([1, 0, 1, 0])
    pc = np.array([0.6, 0.4, 0.7, 0.3])
    pt = np.array([0.7, 0.3, 0.8, 0.2])
    p = S.Panel.from_logloss(["2023-24"] * 4, y, pc, pt)
    manual = (-(y * np.log(pc) + (1 - y) * np.log(1 - pc))
              + (y * np.log(pt) + (1 - y) * np.log(1 - pt)))
    assert np.allclose(p.d, manual)
