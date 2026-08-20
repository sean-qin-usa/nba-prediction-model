"""ES experiment: cold-start prior FADE SHAPE (Sean: "why 20 games? extend/remove/wave?")

PRE-REGISTERED DESIGN (7 configs, fixed before any scoring; Bonferroni /7):
  Shipped production (control): prior weight wh = max(0, 1 - gp/20) — linear,
  hard zero at 20 games, applied ONLY inside the ratings fallback (coef
  (1-w_comp)=0.3 on the margin), and the whole ratings+prior channel vanishes
  the moment ff.ready flips (~2 weeks in, 200 factor rows) — a discontinuity.

  Variants change BOTH the fade shape AND extend the prior into the FF-ready
  era per the directive ("blended_margin += wh*prior_adj even after ff.ready"):
    pre-FF  (fallback): margin = 0.7*cm + 0.3*(tr - tr.home + wh*ph - wa*pa) + sched
                        (identical structure to shipped; only wh shape differs)
    post-FF (extension): margin = 0.5*fm + 0.5*cm + sched + (wh*ph - wa*pa)
                        (control adds nothing here — shipped behavior)
  Configs:
    C0 control   : linear-20, fallback-only (exact shipped replication via
                   fit_production().p_home — not a reimplementation)
    C1 exp7_ext  : wh = exp(-gp/7)    + extension
    C2 exp15_ext : wh = exp(-gp/15)   + extension
    C3 exp30_ext : wh = exp(-gp/30)   + extension
    C4 prec5_ext : wh = 5/(5+gp)      + extension  (Bayesian shrinkage, k pseudo-games)
    C5 prec10_ext: wh = 10/(10+gp)    + extension
    C6 prec20_ext: wh = 20/(20+gp)    + extension
  TRUNCATION (pre-registered): all variant fades are hard-floored to wh=0 at
  gp>=30 so that every config is provably IDENTICAL to control once both teams
  have 30 frozen games played — this bounds the experiment to the early-season
  target window and makes window-only scoring exact. (Without it the slow
  fades leak a last-season prior into March, a different experiment.)
  NOTE: shape and extension are confounded by design — the 7-config budget
  cannot isolate them; pre-FF the shapes barely differ (gp<=~7: linear-20 in
  [0.65,1], exp15 in [0.63,1]), so the extension carries the signal.

  gp = games_played FROZEN at the weekly refit date (exactly what shipped
  wh sees; D54 fix semantics). ph/pa = last-season regressed prior (abbrev).

SCORING (single shared-component pass — all 7 configs reuse the same fitted
  comp/ff/tr/sched/prior extracted from the shipped predictor's closures, so
  marginal config cost is zero and NO production code is touched):
  * GATE window ("w30"): games with min(gh, ga) < 30 (frozen gp) — the only
    games where any config can differ from control. Verified empirically:
    max |margin_cfg - margin_ctrl| over all out-of-window games must be 0.
  * early-season subset ("w20"): min(gh, ga) < 20 (frozen) ~ Oct/Nov target.
  * Paired bootstrap (game-level, 2000 draws, seed 42) on delta =
    loss(control) - loss(variant), positive = variant better. Best config by
    pooled w30 delta; gate at Bonferroni alpha 0.05/7 (99.2857% CI).

REPLICATION CONTROL: control p comes from the shipped fit_production Predictor
  (weekly refit, oracle OUT sets, market-filter — byte-for-byte the
  prod_by_season.py loop) and is cross-checked against
  data/capstone_pergame_csfix.csv; the component reconstruction of the control
  margin is asserted against model.margin() to <1e-9 for every game.

Usage: python scripts/es_fadeshape.py
Outputs: data/es_fadeshape_pergame.csv, data/es_fadeshape_summary.json
"""
import sys, json, warnings, datetime as _dt
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from nbapred.db import connect
from nbapred.model.production import fit_production, SCALE, sigmoid

SEASONS = ("2023-24", "2024-25", "2025-26")
TRUNC = 30          # hard floor: variant wh = 0 at gp >= TRUNC (pre-registered)
BOOT, SEED = 2000, 42
N_CONFIGS = 7       # Bonferroni family size (incl. control; 6 comparisons but /7 per directive)

CONFIGS = {         # name -> wh(gp) BEFORE truncation (control handled by shipped code)
    "exp7_ext":   lambda gp: float(np.exp(-gp / 7.0)),
    "exp15_ext":  lambda gp: float(np.exp(-gp / 15.0)),
    "exp30_ext":  lambda gp: float(np.exp(-gp / 30.0)),
    "prec5_ext":  lambda gp: 5.0 / (5.0 + gp),
    "prec10_ext": lambda gp: 10.0 / (10.0 + gp),
    "prec20_ext": lambda gp: 20.0 / (20.0 + gp),
}


def wh_var(fn, gp):
    return 0.0 if gp >= TRUNC else fn(gp)


def wh_ctrl(gp):
    return max(0.0, 1.0 - gp / 20.0)


def closure(bound_method):
    fn = bound_method.__func__
    return dict(zip(fn.__code__.co_freevars,
                    [c.cell_contents for c in fn.__closure__]))


def season_run(season):
    """Replicates prod_by_season.season_run (default oracle-outs mode) exactly,
    additionally scoring the 6 variant configs from shared components."""
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    model = None
    last = None
    fv_m = fv_r = None
    max_recon_err = 0.0
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m_ = recs[0].matchup
        host = m_.split("@")[-1].strip() if "@" in m_ else m_.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            fv_m = closure(model.margin)
            fv_r = closure(model.ratings_margin)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        comp, ff, tr = fv_m["comp"], fv_m["ff"], fv_m["tr"]
        he, b_hb2b, b_ab2b = fv_m["he"], fv_m["b_hb2b"], fv_m["b_ab2b"]
        gp_map, prior, id2ab = fv_r["games_played"], fv_r["prior"], fv_r["id2ab"]
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        # shipped control probability — the actual production code path
        p_ctrl = model.p_home(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                              gd, b2b_home=bh, b2b_away=ba)
        # shared components -> base margin (prior term excluded)
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, home_edge=0.0)
        sched = he + (b_hb2b if bh else 0.0) + (b_ab2b if ba else 0.0)
        gh, ga = gp_map.get(h.team_id, 0), gp_map.get(a.team_id, 0)
        ph = prior.get(id2ab.get(h.team_id, ""), 0.0)
        pa = prior.get(id2ab.get(a.team_id, ""), 0.0)
        if ff.ready:
            base = 0.5 * ff.margin_neutral(h.team_id, a.team_id) + 0.5 * cm + sched
            coef_fb, ext = 0.0, 1.0            # extension coef for variants
            m_ctrl_rec = base                  # control adds nothing post-FF
        else:
            rm0 = tr.pred_margin(h.team_id, a.team_id) - tr.home
            base = 0.7 * cm + 0.3 * rm0 + sched
            coef_fb, ext = 0.3, 0.0
            m_ctrl_rec = base + 0.3 * (wh_ctrl(gh) * ph - wh_ctrl(ga) * pa)
        m_ctrl = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                              gd, b2b_home=bh, b2b_away=ba)
        max_recon_err = max(max_recon_err, abs(m_ctrl_rec - m_ctrl))
        row = dict(season=season, game_id=gid, game_date=str(gd)[:10],
                   home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
                   gh=gh, ga=ga, ff_ready=int(ff.ready), p_mkt=float(pmv),
                   p_ctrl=float(p_ctrl))
        for name, fn in CONFIGS.items():
            pw = wh_var(fn, gh) * ph - wh_var(fn, ga) * pa
            m_v = base + (coef_fb * pw if not ff.ready else ext * pw)
            row["p_" + name] = float(sigmoid(m_v / SCALE))
        rows.append(row)
    con.close()
    return rows, max_recon_err


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_boot(delta, n_boot=BOOT, seed=SEED, alphas=(0.05, 0.05 / N_CONFIGS)):
    rng = np.random.default_rng(seed)
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = delta[idx].mean(axis=1)
    out = {"mean": float(delta.mean())}
    for a in alphas:
        lo, hi = np.percentile(means, [100 * a / 2, 100 * (1 - a / 2)])
        out["ci%g" % round((1 - a) * 100, 4)] = [float(lo), float(hi)]
    return out


def main():
    all_rows, errs = [], {}
    for s in SEASONS:
        r, e = season_run(s)
        all_rows.extend(r)
        errs[s] = e
        print(f"{s}: {len(r)} games, control-margin reconstruction max err {e:.2e}",
              flush=True)
    assert max(errs.values()) < 1e-9, f"control reconstruction failed: {errs}"

    import csv as _csv
    root = Path(__file__).resolve().parent.parent
    cols = list(all_rows[0].keys())
    with open(root / "data/es_fadeshape_pergame.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    gmin = np.array([min(r["gh"], r["ga"]) for r in all_rows])
    season = np.array([r["season"] for r in all_rows])
    l_ctrl = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    l_mkt = ll_vec(y, [r["p_mkt"] for r in all_rows])
    l_cfg = {c: ll_vec(y, [r["p_" + c] for r in all_rows]) for c in CONFIGS}

    # verification: identical outside the w30 window
    outw = gmin >= TRUNC
    max_out = max(float(np.max(np.abs(l_cfg[c][outw] - l_ctrl[outw]))) for c in CONFIGS)
    n_out_diff = int(sum(int((np.abs(l_cfg[c][outw] - l_ctrl[outw]) > 1e-12).sum())
                         for c in CONFIGS))
    print(f"out-of-window (min gp >= {TRUNC}): n={int(outw.sum())} games x 6 configs, "
          f"max |loss diff| = {max_out:.2e}, n differing = {n_out_diff}")

    # baseline csfix replication check on control
    import pandas as pd
    base = pd.read_csv(root / "data/capstone_pergame_csfix.csv", dtype={"game_id": str})
    bmap = {str(r.game_id): float(r.p_us) for r in base.itertuples()}
    diffs = [abs(r["p_ctrl"] - bmap[r["game_id"]]) for r in all_rows if r["game_id"] in bmap]
    rep = dict(n_joined=len(diffs), n_ours=len(all_rows), n_baseline=len(base),
               max_abs_dp=float(max(diffs)), mean_abs_dp=float(np.mean(diffs)))
    print("csfix replication:", rep)

    summary = dict(design="es_fadeshape 7-config pre-registered", trunc=TRUNC,
                   recon_err=errs, out_of_window_check=dict(
                       n_games=int(outw.sum()), max_loss_diff=max_out,
                       n_differing=n_out_diff),
                   csfix_replication=rep, windows={})
    for wname, mask in (("w30_gate", gmin < TRUNC), ("w20_early", gmin < 20),
                        ("full", np.ones(len(y), bool))):
        wsum = dict(n=int(mask.sum()),
                    ll_control=float(l_ctrl[mask].mean()),
                    ll_market=float(l_mkt[mask].mean()),
                    per_season={}, configs={})
        for c in CONFIGS:
            d = (l_ctrl - l_cfg[c])[mask]      # + = variant better
            wsum["configs"][c] = dict(ll=float(l_cfg[c][mask].mean()),
                                      **paired_boot(d))
        for s in SEASONS:
            sm = mask & (season == s)
            ps = dict(n=int(sm.sum()), ll_control=float(l_ctrl[sm].mean()),
                      ll_market=float(l_mkt[sm].mean()),
                      configs={c: dict(ll=float(l_cfg[c][sm].mean()),
                                       **paired_boot((l_ctrl - l_cfg[c])[sm]))
                               for c in CONFIGS})
            wsum["per_season"][s] = ps
        summary["windows"][wname] = wsum

    gate = summary["windows"]["w30_gate"]["configs"]
    best = max(gate, key=lambda c: gate[c]["mean"])
    bkey = [k for k in gate[best] if k.startswith("ci") and k != "ci95"][0]
    lo_b, hi_b = gate[best][bkey]
    verdict = ("PASS" if lo_b > 0 else "FAIL" if hi_b < 0 else "NS")
    summary["best"] = dict(config=best, window="w30_gate", verdict=verdict,
                           bonferroni_ci_key=bkey, **gate[best])
    json.dump(summary, open(root / "data/es_fadeshape_summary.json", "w"), indent=1)
    print(json.dumps(summary["best"], indent=1))
    for wname in ("w30_gate", "w20_early", "full"):
        w_ = summary["windows"][wname]
        print(f"\n== {wname} (n={w_['n']}) ctrl {w_['ll_control']:.5f} mkt {w_['ll_market']:.5f}")
        for c in CONFIGS:
            g = w_["configs"][c]
            print(f"  {c:12s} ll {g['ll']:.5f} d {g['mean']:+.5f} "
                  f"ci95 [{g['ci95'][0]:+.5f},{g['ci95'][1]:+.5f}]")


if __name__ == "__main__":
    main()
