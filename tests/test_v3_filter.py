"""v3 M0/M1 tests: filter recovery, season-shock response, persistence,
gate smoke. Pure simulation + temp DuckDB — never touches data/nba.duckdb."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.v3.hyper import TeamHyper
from nbapred.v3.state_bank import DIMS, StateBank
from nbapred.v3.team_dlm import TeamDLM

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
D0 = dt.date(2023, 10, 24)


def _sim_margin_obs(nets, n_games, rng, start=D0, noise=12.0, home_edge=2.3):
    """Random round-robin margin observations from FIXED team nets."""
    obs = []
    teams = list(nets)
    for k in range(n_games):
        d = start + dt.timedelta(days=k // 6)
        h, a = rng.choice(teams, size=2, replace=False)
        m = home_edge + nets[h] - nets[a] + rng.normal(0, noise)
        obs.append((d, "margin", h, a, float(m), None))
    return obs


def test_constant_state_recovery():
    """A constant true state must be recovered: posterior nets correlate with
    truth and beat the zero-prior RMSE by a wide margin."""
    rng = np.random.default_rng(7)
    nets = {t: v for t, v in zip(TEAMS, [8.0, 4.0, 1.0, -1.0, -4.0, -8.0])}
    obs = _sim_margin_obs(nets, 900, rng)
    h = TeamHyper(phi=0.999, q=0.005, r_margin=144.0)
    dlm = TeamDLM(TEAMS, h, start=D0)
    dlm.run(obs)
    est = np.array([dlm.net(t) for t in TEAMS])
    tru = np.array([nets[t] for t in TEAMS])
    assert np.corrcoef(est, tru)[0, 1] > 0.98
    rmse = float(np.sqrt(np.mean((est - tru) ** 2)))
    prior_rmse = float(np.sqrt(np.mean(tru ** 2)))
    assert rmse < 0.35 * prior_rmse, (rmse, prior_rmse)
    # home-edge state converges near truth
    assert abs(float(dlm.x[1]) - 2.3) < 1.5


def test_variance_contracts_with_data():
    rng = np.random.default_rng(1)
    nets = {t: 0.0 for t in TEAMS}
    obs = _sim_margin_obs(nets, 600, rng)
    dlm = TeamDLM(TEAMS, TeamHyper(), start=D0)
    v0 = dlm.margin_neutral_var("AAA", "BBB")
    dlm.run(obs)
    assert dlm.margin_neutral_var("AAA", "BBB") < 0.5 * v0


def test_season_shock_response():
    """A level break at the season boundary: the shocked filter must adapt
    faster than a continuity-1 filter (lower one-step MSE right after)."""
    rng = np.random.default_rng(11)
    nets1 = {t: v for t, v in zip(TEAMS, [8.0, 4.0, 1.0, -1.0, -4.0, -8.0])}
    nets2 = {t: -v for t, v in zip(TEAMS, [8.0, 4.0, 1.0, -1.0, -4.0, -8.0])}
    obs1 = _sim_margin_obs(nets1, 500, rng, start=D0)
    boundary = D0 + dt.timedelta(days=200)              # ~offseason gap after
    obs2 = _sim_margin_obs(nets2, 240, rng, start=boundary)
    shocked = TeamDLM(TEAMS, TeamHyper(kappa=0.5, v_bound=16.0),
                      season_boundaries=[boundary], start=D0)
    rigid = TeamDLM(TEAMS, TeamHyper(kappa=1.0, v_bound=0.0),
                    season_boundaries=[boundary], start=D0)
    for f in (shocked, rigid):
        f.run(obs1)
    def post_mse(f, k=90):
        errs = []
        for ob in obs2[:k]:
            f.predict_to(ob[0])
            errs.append((f.margin_neutral(ob[2], ob[3])
                         - (nets2[ob[2]] - nets2[ob[3]])) ** 2)
            f.update(ob)
        return float(np.mean(errs))
    mse_s, mse_r = post_mse(shocked), post_mse(rigid)
    assert mse_s < mse_r, (mse_s, mse_r)
    # and the shock inflates uncertainty at the boundary
    f2 = TeamDLM(TEAMS, TeamHyper(kappa=0.5, v_bound=16.0),
                 season_boundaries=[boundary], start=D0)
    f2.run(obs1)
    v_before = f2.margin_neutral_var("AAA", "BBB")
    f2.predict_to(boundary)
    assert f2.margin_neutral_var("AAA", "BBB") > v_before + 30.0


def test_eff_obs_identifies_off_def():
    """Efficiency-pair observations split net into offense and defense."""
    rng = np.random.default_rng(3)
    off = {t: v for t, v in zip(TEAMS, [6.0, 3.0, 1.0, -1.0, -3.0, -6.0])}
    dfn = {t: v for t, v in zip(TEAMS, [-4.0, 2.0, -2.0, 4.0, 1.0, -1.0])}
    obs = []
    for k in range(1200):
        d = D0 + dt.timedelta(days=k // 6)
        h, a = rng.choice(TEAMS, size=2, replace=False)
        oh = 112.0 + 2.0 + off[h] - dfn[a] + rng.normal(0, 10.0)
        oa = 112.0 + off[a] - dfn[h] + rng.normal(0, 10.0)
        obs.append((d, "eff", h, a, float(oh), float(oa)))
    dlm = TeamDLM(TEAMS, TeamHyper(phi=0.999, q=0.005, r_eff=100.0), start=D0)
    dlm.run(obs)
    eo = np.array([dlm.x[dlm._io(t)] for t in TEAMS])
    ed = np.array([dlm.x[dlm._id(t)] for t in TEAMS])
    assert np.corrcoef(eo, [off[t] for t in TEAMS])[0, 1] > 0.9
    assert np.corrcoef(ed, [dfn[t] for t in TEAMS])[0, 1] > 0.9
    assert abs(float(dlm.x[0]) - 112.0) < 2.0


def test_loglik_prefers_true_process_noise():
    """Marginal likelihood is a working hyper-selection signal: the true q
    beats a badly wrong q on simulated drifting states."""
    rng = np.random.default_rng(5)
    teams = TEAMS
    nets = {t: 0.0 for t in teams}
    obs = []
    for k in range(1500):
        d = D0 + dt.timedelta(days=k // 6)
        if k % 6 == 0:
            for t in teams:                       # daily random walk, sd .3
                nets[t] += rng.normal(0, 0.3)
        h, a = rng.choice(teams, size=2, replace=False)
        obs.append((d, "margin", h, a,
                    float(2.3 + nets[h] - nets[a] + rng.normal(0, 12.0)), None))
    def ll(q):
        f = TeamDLM(teams, TeamHyper(phi=0.9999, q=q, r_margin=144.0), start=D0)
        return f.run(obs)
    assert ll(0.09) > ll(0.0001)
    assert ll(0.09) > ll(5.0)


def test_batch_equals_sequential():
    """The per-date block update must equal sequential scalar updates
    (same state, same covariance, same total loglik)."""
    rng = np.random.default_rng(9)
    nets = {t: v for t, v in zip(TEAMS, [5.0, 2.0, 0.0, -1.0, -2.0, -4.0])}
    obs = _sim_margin_obs(nets, 300, rng)
    # mix in eff obs on the same dates
    for k in range(0, 300, 3):
        d, _, h, a, m, _ = obs[k]
        obs[k] = (d, "eff", h, a, 112.0 + m / 2, 112.0 - m / 2)
    obs.sort(key=lambda o: o[0])
    f_seq = TeamDLM(TEAMS, TeamHyper(), start=D0)
    ll_seq = 0.0
    for ob in obs:
        f_seq.predict_to(ob[0])
        ll_seq += f_seq.update(ob)
    f_bat = TeamDLM(TEAMS, TeamHyper(), start=D0)
    ll_bat = f_bat.run(obs)
    assert np.allclose(f_seq.x, f_bat.x, atol=1e-8)
    assert np.allclose(f_seq.P, f_bat.P, atol=1e-8)
    assert abs(ll_seq - ll_bat) < 1e-6, (ll_seq, ll_bat)


def test_state_bank_roundtrip(tmp_path):
    import duckdb

    from nbapred.v3.schema import ensure_v3_tables
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    ensure_v3_tables(con)
    bank = StateBank(dt.date(2025, 1, 5))
    rng = np.random.default_rng(0)
    for pid in (101, 202):
        bank.add_player(pid, anchor=rng.normal(size=len(DIMS)),
                        prior_var=np.abs(rng.normal(1, 0.1, len(DIMS))))
    bank.pace[1610612744] = (99.5, 2.0)
    bank.snapshot(con)
    b2 = StateBank.load(con, dt.date(2025, 1, 5))
    for pid in (101, 202):
        assert np.allclose(b2.theta[pid], bank.theta[pid])
        assert np.allclose(b2.P[pid], bank.P[pid])
    assert b2.pace[1610612744] == (99.5, 2.0)
    # idempotent re-snapshot (PK safety)
    bank.snapshot(con)
    n = con.execute("SELECT count(*) FROM player_states").fetchone()[0]
    assert n == 2 * len(DIMS) + 1
    con.close()


def test_bank_evolution_toward_anchor():
    from nbapred.v3.hyper import HyperParams
    bank = StateBank(dt.date(2025, 1, 1))
    anchor = np.zeros(len(DIMS))
    bank.add_player(1, anchor=anchor, prior_var=np.full(len(DIMS), 0.5))
    bank.theta[1] = np.full(len(DIMS), 1.0)
    hp = HyperParams()
    bank.predict_to(dt.date(2025, 1, 31), hp)
    th, P = bank.theta[1], bank.P[1]
    assert np.all(th < 1.0) and np.all(th > 0.0)      # shrunk toward anchor
    assert np.all(P > 0.5)                            # variance grew
    # shock inflation multiplies Q accumulation
    bank2 = StateBank(dt.date(2025, 1, 1))
    bank2.add_player(1, anchor=anchor, prior_var=np.full(len(DIMS), 0.5))
    bank2.predict_to(dt.date(2025, 1, 31), hp, shocks={1: 10.0})
    assert np.all(bank2.P[1] > P)


def test_gate_smoke_zero_delta():
    from nbapred.eval.ablate import paired_bootstrap_delta
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.55).astype(float)
    p = np.clip(rng.normal(0.55, 0.1, 500), 0.01, 0.99)
    r = paired_bootstrap_delta(y, p, p, n_boot=200)
    assert abs(r["delta_logloss"]) < 1e-12 and r["keep"] is False
