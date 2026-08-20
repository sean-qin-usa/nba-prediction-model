"""PRE-REGISTERED GATE: defence-conditioned possession-level margin component.

Pre-registration: data/poss_v2_prereg.md (written BEFORE this run; 3 specs,
one config each, no sweeps). Context: D113 reversed the D29/D31 sufficiency
proof at the possession-likelihood level; this is the game-level gate of the
defence-conditioned possession fit as a team-margin input.

Specs (fit = Poisson IRLS ridge on possessions_v2, verbatim D113 likelihood
y=min(points,4), eta clip [-8,4], intercept free, ridge 3200 fixed ex ante;
window = 002 possessions in [cutoff-730d, cutoff); weekly walk-forward refits
at the exact prod_by_season cadence):
  S1  [1 | +off_player | -def_player]   (primary; D113's winning design)
  S2  [1 | +off_player]                 (ablation control: offence-only)
  S3  [1 | +off_player | -def_team]     (team-level defence conditioning)

Aggregation: comp-rotation weights (trail_min/48, roster<=12d, oracle OUTs —
the SAME CompositionModel instance and OUT sets as the comp leg), then
  pm = 100*(exp(mu + off_h - def_a) - exp(mu + off_a - def_h))    [S1]
  pm = 100*(exp(mu + off_h) - exp(mu + off_a))                    [S2]
  pm = 100*(exp(mu + off_h - dteam_a) - exp(mu + off_a - dteam_h))[S3]
Entry (fixed, no fitted weight): m_exp = m_ctl + 0.25*(pm - cm_used), the
within-comp-slot 50/50 swap; cm_used is the bridge-aware comp leg read from
the fit_production closure (pg_eventrecency pattern; nbapred/ untouched).

Control: same-run certified D122 stack. REQUIRED env:
  env -u LATE_STATE -u TANK_TERM -u ORACLE_MINUTES -u INACTIVE_OUTS \
      -u REPORT_OUTS -u TANK_SEASON_FLOOR -u OCT_BRIDGE OCT_BRIDGE_TRAIL= \
      python3 scripts/pg_possdef.py [--holdout] [--analyze-only]
Cross-check REQUIRED: per-game |p_ctl - capstone p_us| < 1e-9 every season.

Gate: paired bootstrap 2000x seed 7 on ll_ctl - ll_exp (positive = better),
pooled DEV (2023-24..2025-26). HOLDOUT (2021-22..2022-23) only via --holdout,
only if a dev gate passed CI (cross-corpus rule). Read-only DB (60s retry).
Artifacts: data/pg_possdef_pergame.csv, data/pg_possdef_summary.json.
"""
import sys, json, csv, time, warnings, datetime as dt
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg as sp_cg

RIDGE = 3200.0          # D113 validation-selected possession ridge, FIXED ex ante
WINDOW_DAYS = 730
SWAP_W = 0.25           # within-comp-slot 50/50 swap of the 0.5 comp weight
POSS_PER_SIDE = 100.0
MIN_POSS = 1000         # below this the component is inert (pm = 0)
EC_THRESH = 2.0         # pre-registered effect-concentration |pm - cm| >= 2.0
DEV_SEASONS = ("2023-24", "2024-25", "2025-26")
HOLDOUT_SEASONS = ("2021-22", "2022-23")
OUT_CSV = REPO / "data" / "pg_possdef_pergame.csv"
OUT_JSON = REPO / "data" / "pg_possdef_summary.json"
CAPSTONE = REPO / "data" / "capstone_pergame.csv"


def connect_retry(read_only=True, attempts=120, wait_s=60):
    """Read-only connect; on writer-lock failure wait 60s and retry (task rule)."""
    from nbapred.db import connect as _c
    last = None
    for _ in range(attempts):
        try:
            return _c(read_only=read_only)
        except Exception as e:
            last = e
            print(f"DB connect failed ({e}); retry in {wait_s}s", flush=True)
            time.sleep(wait_s)
    raise last


from nbapred.model.composition import CompositionModel          # noqa: E402
from nbapred.model.production import SCALE, fit_production, sigmoid  # noqa: E402


def _closure_var(bound_method, name):
    fn = bound_method.__func__
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents


# ------------------------------------------------------------ possession fits
def irls_poisson(X, y, ridge, iters=25, tol=1e-8):
    """Verbatim cg_v2_sufficiency.py IRLS (D113)."""
    p = X.shape[1]
    b = np.zeros(p)
    pen = np.full(p, ridge); pen[0] = 0.0
    prev = None
    for _ in range(iters):
        eta = X @ b
        np.clip(eta, -8, 4, out=eta)
        mu = np.exp(eta)
        g = X.T @ (mu - y) + pen * b
        H = (X.T @ sparse.diags(mu) @ X).tocsc() + sparse.diags(pen)
        step, _ = sp_cg(H, -g, rtol=1e-7, maxiter=500)
        b = b + step
        ll = float((y * (X @ b) - np.exp(np.clip(X @ b, -8, 4))).mean())
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll
    return b


class PossFits:
    """The three walk-forward possession fits at one cutoff.
    Exposes off/def maps keyed by nba player id (S1, S2) / team id (S3)."""

    def __init__(self, con, cutoff: dt.date):
        t0 = time.time()
        df = con.execute("""
            SELECT p.def_team, p.off_lineup, p.def_lineup, p.points
            FROM possessions_v2 p
            JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g
              ON g.game_id = p.game_id
            WHERE p.game_id LIKE '002%' AND g.game_date < ? AND g.game_date >= ?
        """, [cutoff, cutoff - dt.timedelta(days=WINDOW_DAYS)]).fetchdf()
        pids, tids = {}, {}
        O, D, T, Y = [], [], [], []
        for r in df.itertuples():
            try:
                o = [int(x) for x in r.off_lineup.split(",")]
                d = [int(x) for x in r.def_lineup.split(",")]
            except Exception:
                continue
            if len(o) != 5 or len(d) != 5:
                continue
            for p in o + d:
                pids.setdefault(p, len(pids))
            tids.setdefault(int(r.def_team), len(tids))
            O.append([pids[p] for p in o])
            D.append([pids[p] for p in d])
            T.append(tids[int(r.def_team)])
            Y.append(min(r.points, 4))
        self.n = len(Y)
        self.ok = self.n >= MIN_POSS
        self.n_players, self.n_teams = len(pids), len(tids)
        if not self.ok:
            self.s1_off = self.s1_def = self.s2_off = self.s3_off = {}
            self.s3_dteam = {}
            self.mu1 = self.mu2 = self.mu3 = 0.0
            print(f"    poss-fit @{cutoff}: only {self.n} possessions -> inert",
                  flush=True)
            return
        O = np.array(O); D = np.array(D); T = np.array(T)
        Y = np.array(Y, float)
        P, K, n = len(pids), len(tids), len(Y)
        rows11 = np.repeat(np.arange(n), 11)
        # S1: [1 | +off | -def_player]
        c1 = np.concatenate([np.zeros((n, 1), int), 1 + O, 1 + P + D], axis=1).ravel()
        v1 = np.concatenate([np.ones((n, 1)), np.ones((n, 5)), -np.ones((n, 5))],
                            axis=1).ravel()
        X1 = sparse.csr_matrix((v1, (rows11, c1)), shape=(n, 1 + 2 * P))
        b1 = irls_poisson(X1, Y, RIDGE)
        # S2: [1 | +off]
        rows6 = np.repeat(np.arange(n), 6)
        c2 = np.concatenate([np.zeros((n, 1), int), 1 + O], axis=1).ravel()
        v2 = np.concatenate([np.ones((n, 1)), np.ones((n, 5))], axis=1).ravel()
        X2 = sparse.csr_matrix((v2, (rows6, c2)), shape=(n, 1 + P))
        b2 = irls_poisson(X2, Y, RIDGE)
        # S3: [1 | +off | -def_team]
        rows7 = np.repeat(np.arange(n), 7)
        c3 = np.concatenate([np.zeros((n, 1), int), 1 + O,
                             (1 + P + T)[:, None]], axis=1).ravel()
        v3 = np.concatenate([np.ones((n, 1)), np.ones((n, 5)),
                             -np.ones((n, 1))], axis=1).ravel()
        X3 = sparse.csr_matrix((v3, (rows7, c3)), shape=(n, 1 + P + K))
        b3 = irls_poisson(X3, Y, RIDGE)
        inv_p = {v: k for k, v in pids.items()}
        inv_t = {v: k for k, v in tids.items()}
        self.mu1, self.mu2, self.mu3 = float(b1[0]), float(b2[0]), float(b3[0])
        self.s1_off = {inv_p[i]: float(b1[1 + i]) for i in range(P)}
        self.s1_def = {inv_p[i]: float(b1[1 + P + i]) for i in range(P)}
        self.s2_off = {inv_p[i]: float(b2[1 + i]) for i in range(P)}
        self.s3_off = {inv_p[i]: float(b3[1 + i]) for i in range(P)}
        self.s3_dteam = {inv_t[i]: float(b3[1 + P + i]) for i in range(K)}
        print(f"    poss-fit @{cutoff}: {n} poss, {P} players, {K} def-teams "
              f"({time.time()-t0:.0f}s)  mu1 {self.mu1:+.4f} "
              f"off-sd {np.std(list(self.s1_off.values())):.5f} "
              f"def-sd {np.std(list(self.s1_def.values())):.5f}", flush=True)

    def _strength(self, comp, team_id, out, gd, table):
        s = 0.0
        for pid, p in comp.players.items():
            if p["team_id"] != team_id or pid in out:
                continue
            if (gd - p["last_played"]).days > 12:       # ROSTER_DAYS, comp rule
                continue
            s += table.get(pid, 0.0) * p["trail_min"] / 48.0
        return s

    def margins(self, comp, h, a, out_h, out_a, gd):
        """(pm1, pm2, pm3): neutral possession margins, points per 100."""
        if not self.ok:
            return 0.0, 0.0, 0.0
        o1h = self._strength(comp, h, out_h, gd, self.s1_off)
        o1a = self._strength(comp, a, out_a, gd, self.s1_off)
        d1h = self._strength(comp, h, out_h, gd, self.s1_def)
        d1a = self._strength(comp, a, out_a, gd, self.s1_def)
        pm1 = POSS_PER_SIDE * (np.exp(self.mu1 + o1h - d1a)
                               - np.exp(self.mu1 + o1a - d1h))
        o2h = self._strength(comp, h, out_h, gd, self.s2_off)
        o2a = self._strength(comp, a, out_a, gd, self.s2_off)
        pm2 = POSS_PER_SIDE * (np.exp(self.mu2 + o2h) - np.exp(self.mu2 + o2a))
        o3h = self._strength(comp, h, out_h, gd, self.s3_off)
        o3a = self._strength(comp, a, out_a, gd, self.s3_off)
        dth = self.s3_dteam.get(h, 0.0)
        dta = self.s3_dteam.get(a, 0.0)
        pm3 = POSS_PER_SIDE * (np.exp(self.mu3 + o3h - dta)
                               - np.exp(self.mu3 + o3a - dth))
        return float(pm1), float(pm2), float(pm3)


# ------------------------------- capstone loop: prod_by_season.py VERBATIM ---
def season_run(season):
    t0 = time.time()
    con = connect_retry(read_only=True)
    pm_df = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm_df.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by = {}; order = []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    model = comp = pf = None
    comp_cl = bridge_cl = rot_empty = None
    last = None
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            comp_cl = _closure_var(model.margin, "comp")
            bridge_cl = _closure_var(model.margin, "bridge")
            rot_empty = _closure_var(model.margin, "_rot_empty")
            pf = PossFits(con, gd)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        m_ctl = model.margin(h.team_id, a.team_id, outs[h.team_id],
                             outs[a.team_id], gd,
                             b2b_home=b2b(h.team_id, gd),
                             b2b_away=b2b(a.team_id, gd))
        # cm_used: the comp leg the Predictor actually used (bridge-aware)
        cm_used = comp_cl.margin(h.team_id, a.team_id, outs[h.team_id],
                                 outs[a.team_id], gd, home_edge=0.0)
        if bridge_cl is not None and rot_empty(comp_cl, h.team_id, a.team_id, gd):
            cm_used = bridge_cl.margin(h.team_id, a.team_id,
                                       outs[h.team_id], outs[a.team_id])
        pm1, pm2, pm3 = pf.margins(comp, h.team_id, a.team_id,
                                   outs[h.team_id], outs[a.team_id], gd)
        r = dict(season=season, game_id=gid, game_date=str(gd)[:10],
                 home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
                 p_ctl=float(sigmoid(m_ctl / SCALE)), p_mkt=float(pmv),
                 cm_used=round(float(cm_used), 4),
                 pm1=round(pm1, 4), pm2=round(pm2, 4), pm3=round(pm3, 4))
        for k, pm in (("p_s1", pm1), ("p_s2", pm2), ("p_s3", pm3)):
            r[k] = float(sigmoid((m_ctl + SWAP_W * (pm - cm_used)) / SCALE))
        rows.append(r)
    con.close()
    print(f"[{season}] games={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return rows


# --------------------------------------------------------------------- gates
def pg_ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_gate(delta, n_boot=2000, seed=7):
    delta = np.asarray(delta, float)
    if len(delta) == 0:
        return dict(n=0, mean=None, lo=None, hi=None, verdict="NS")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    verdict = ("PASS" if lo > 0 else "FAIL" if hi < 0 else "NS")
    return dict(n=int(len(delta)), mean=float(delta.mean()),
                lo=float(lo), hi=float(hi), verdict=verdict,
                p_wrongside=float((means <= 0).mean()),
                sd=float(delta.std()),
                mde80=float(2.802 * delta.std() / max(np.sqrt(len(delta)), 1)))


def analyze(df):
    import pandas as pd
    cap = pd.read_csv(CAPSTONE, dtype={"game_id": str})[["season", "game_id", "p_us"]]
    cap["game_id"] = cap.game_id.str.zfill(10)
    mg = df.merge(cap, on=["season", "game_id"], how="inner")
    max_dp = float((mg.p_ctl - mg.p_us).abs().max()) if len(mg) else None
    repl = dict(baseline="capstone_pergame.csv", n_matched=int(len(mg)),
                n_ours=int(len(df)), max_abs_diff=max_dp)
    if max_dp is None or max_dp > 1e-9:
        raise RuntimeError(f"CONTROL REPLICATION FAILED: max|dp|={max_dp} "
                           "(required < 1e-9); gate aborted, fix env/harness")
    df["ll_ctl"] = pg_ll(df.y, df.p_ctl)
    df["ll_mkt"] = pg_ll(df.y, df.p_mkt)
    for s in ("s1", "s2", "s3"):
        df[f"ll_{s}"] = pg_ll(df.y, df[f"p_{s}"])
    res = dict(prereg="data/poss_v2_prereg.md", replication=repl,
               config=dict(ridge=RIDGE, window_days=WINDOW_DAYS, swap_w=SWAP_W,
                           poss_per_side=POSS_PER_SIDE, min_poss=MIN_POSS,
                           ec_thresh=EC_THRESH, boot="2000x seed 7"),
               component_spread=dict(
                   sd_cm=float(df.cm_used.std()), sd_pm1=float(df.pm1.std()),
                   sd_pm2=float(df.pm2.std()), sd_pm3=float(df.pm3.std()),
                   corr_pm1_cm=float(np.corrcoef(df.pm1, df.cm_used)[0, 1]),
                   corr_pm1_pm2=float(np.corrcoef(df.pm1, df.pm2)[0, 1]),
                   corr_pm1_pm3=float(np.corrcoef(df.pm1, df.pm3)[0, 1])),
               seasons={}, pooled={})
    for s in ("s1", "s2", "s3"):
        d = df.ll_ctl - df[f"ll_{s}"]
        res["pooled"][f"{s}_vs_ctl"] = boot_gate(d)
        w = (df[f"pm{s[1]}"] - df.cm_used).abs() >= EC_THRESH
        res["pooled"][f"{s}_vs_ctl_ecwindow"] = boot_gate(d[w])
    res["pooled"]["s1_vs_s2"] = boot_gate(df.ll_s2 - df.ll_s1)
    for season, g in df.groupby("season"):
        e = dict(n=int(len(g)), ll_ctl=round(float(g.ll_ctl.mean()), 5),
                 ll_mkt=round(float(g.ll_mkt.mean()), 5))
        for s in ("s1", "s2", "s3"):
            e[f"ll_{s}"] = round(float(g[f"ll_{s}"].mean()), 5)
            e[f"{s}_vs_ctl"] = boot_gate(g.ll_ctl - g[f"ll_{s}"])
        res["seasons"][season] = e
    return res


def main():
    import pandas as pd
    holdout = "--holdout" in sys.argv
    seasons = HOLDOUT_SEASONS if holdout else DEV_SEASONS
    csv_path = (REPO / "data" / "pg_possdef_holdout_pergame.csv") if holdout else OUT_CSV
    json_path = (REPO / "data" / "pg_possdef_holdout_summary.json") if holdout else OUT_JSON
    if "--analyze-only" in sys.argv and csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"game_id": str})
    else:
        for v in ("LATE_STATE", "TANK_TERM", "ORACLE_MINUTES", "INACTIVE_OUTS",
                  "REPORT_OUTS", "TANK_SEASON_FLOOR", "OCT_BRIDGE"):
            import os
            assert os.environ.get(v) is None, f"{v} must be unset (D122 env)"
        import os
        assert os.environ.get("OCT_BRIDGE_TRAIL") == "", \
            "OCT_BRIDGE_TRAIL= (empty) required to reproduce D122 certification"
        all_rows = []
        for s in seasons:
            all_rows += season_run(s)
        with open(csv_path, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wtr.writeheader(); wtr.writerows(all_rows)
        df = pd.DataFrame(all_rows)
    df["game_id"] = df.game_id.astype(str).str.zfill(10)
    res = analyze(df)
    res["corpus"] = "holdout 2021-22..2022-23" if holdout else "dev 2023-24..2025-26"
    json.dump(res, open(json_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
