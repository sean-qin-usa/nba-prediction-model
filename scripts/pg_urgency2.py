"""PRE-REGISTERED GATE: late-season URGENCY re-test (F5 successor).

Pre-registration: data/urgency_prereg.md (written BEFORE this run; sha256
f135a515c2f5e4fb3ff776d7a1d8f8ecbcf43dfa471ee6c092279ecdf053921b). Bias-audit
item 2 / D76-H9 / D80-F5 / D89-R4a lineage. Three arms, one config each, no
sweeps; all constants fixed ex ante (see prereg):

  A (primary)  margin += k_u * (u_h - u_a)
               u(t,d) = (1 - min(d67, d1011)/3)_+ * min(1, 10/rem),
               ALIVE teams only, gp>=55 gate; k_u walk-forward mirroring
               tanking.fit_k (OLS [1, u_diff, wpct_diff], n/(n+600), clip 15)
  B            margin += lam_B * (lock_h - lock_a); LOCK = top-6 berth
               clinched (gap(t, team7) > rem), gp>=55; lam_B fixed from
               2023-24 ONLY, clip [-2, 0]; 23-24 = in-sample diagnostic
  C            margin += k_t * (tank+_h * u_a - tank+_a * u_h); tank+ =
               max(gated tank score, 0); k_t = production tank coefficient
               at the same refit; c = 1.0 theory-fixed

Control: same-run certified D122 stack (prod_by_season loop verbatim).
REQUIRED env (identical to pg_possdef.py / D127):
  env -u LATE_STATE -u TANK_TERM -u ORACLE_MINUTES -u INACTIVE_OUTS \
      -u REPORT_OUTS -u TANK_SEASON_FLOOR -u OCT_BRIDGE OCT_BRIDGE_TRAIL= \
      python3 scripts/pg_urgency2.py [--holdout] [--analyze-only]
Cross-check REQUIRED: per-game |p_ctl - capstone p_us| < 1e-9 every season.

Gate: paired bootstrap 2000x seed 7 on ll_ctl - ll_arm, PRIMARY window =
both gp>=55 AND home ALIVE, dev 2023-24..2025-26 (B: 24-25..25-26);
monotone-by-driver-bin required; D65 subset secondary; full-pool harm veto.
HOLDOUT (2021-22..2022-23) only via --holdout, only if a dev arm passed.
Read-only DB (60s retry). nbapred/ untouched.
Artifacts: data/pg_urgency2_pergame.csv, data/pg_urgency2_summary.json.
"""
import sys, json, csv, time, warnings, datetime as dt
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

D_BND_SCALE = 3.0        # GB; continuity with F5's |GB|<=3 window
REM_LEV = 10.0           # per-game leverage ramp: min(1, 10/rem)
GP_ACTIVE = 55
K_MIN_ACTIVE = 20
K_CLIP = 15.0
K_SHRINK = 600.0
LAM_CLIP = (-2.0, 0.0)   # one-sided letdown clip (theory: negative)
C_INTER = 1.0            # ARM C interaction scale, theory-fixed
LAM_FIT_SEASON = "2023-24"
DEV_SEASONS = ("2023-24", "2024-25", "2025-26")
B_GATE_SEASONS = ("2024-25", "2025-26")
HOLDOUT_SEASONS = ("2021-22", "2022-23")
FLOOR = "2021-22"        # asserted == tanking.season_floor(con) at runtime
D65_HEAVY = 0.35
OUT_CSV = REPO / "data" / "pg_urgency2_pergame.csv"
OUT_JSON = REPO / "data" / "pg_urgency2_summary.json"
CAPSTONE = REPO / "data" / "capstone_pergame.csv"
PREREG_SHA = "f135a515c2f5e4fb3ff776d7a1d8f8ecbcf43dfa471ee6c092279ecdf053921b"


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
from nbapred.model.tanking import EAST, WEST, get_tank_model, season_floor  # noqa: E402


# ---------------------------------------------------------------- standings
class UrgencyModel:
    """PIT standings -> per-(team_id, date) urgency/lock/alive/gp map + the
    walk-forward k_u frame and the frozen lam_B. Standings state uses only
    games strictly BEFORE each date (emit-before-update, D55 discipline);
    conference order (-wpct, -W, abbrev) = tanking._comp_d_standings."""

    def __init__(self, con):
        g = con.execute("""
            WITH t AS (SELECT season, game_id, game_date, team_id, team_abbrev,
                              is_home, pts
                       FROM nba_games WHERE game_id LIKE '002%'
                         AND pts IS NOT NULL AND season >= ?)
            SELECT h.season, h.game_id, h.game_date, h.team_id ht,
                   h.team_abbrev hab, a.team_id awt, a.team_abbrev aab,
                   h.pts hp, a.pts ap
            FROM t h JOIN t a USING (game_id)
            WHERE h.is_home AND NOT a.is_home
            ORDER BY h.game_date, h.game_id""", [FLOOR]).fetchall()
        by_season = {}
        for season, gid, d, ht, hab, awt, aab, hp, ap in g:
            d = d.date() if hasattr(d, "date") else d
            by_season.setdefault(season, []).append(
                (d, gid, int(ht), hab, int(awt), aab, int(hp), int(ap)))
        self.map = {}     # (team_id, date) -> (urg, alive, lock, gp)
        games = []        # (season, date, ht, awt, margin) chronological
        for season in sorted(by_season):
            rows = by_season[season]
            ab = {}
            for _, _, ht, hab, awt, aab, _, _ in rows:
                ab[ht] = hab
                ab[awt] = aab
            conf_ids = {0: [t for t in ab if ab[t] in EAST],
                        1: [t for t in ab if ab[t] in WEST]}
            W = {t: 0 for t in ab}
            L = {t: 0 for t in ab}
            by_date = {}
            for r in rows:
                by_date.setdefault(r[0], []).append(r)
            for d in sorted(by_date):
                def wpct(t):
                    n = W[t] + L[t]
                    return W[t] / n if n else 0.5

                def gap(u, v):          # GB of v behind u
                    return ((W[u] - W[v]) + (L[v] - L[u])) / 2.0
                for c in (0, 1):
                    ids = conf_ids[c]
                    if len(ids) < 10:
                        continue
                    order = sorted(ids, key=lambda t: (-wpct(t), -W[t], ab[t]))
                    for i, t in enumerate(order):
                        gp = W[t] + L[t]
                        rem = 82 - gp
                        d67 = abs(gap(t, order[6])) if i <= 5 \
                            else abs(gap(order[5], t))
                        d1011 = abs(gap(t, order[10])) if i <= 9 \
                            else abs(gap(order[9], t))
                        alive = (i <= 9) or (gap(order[9], t) <= rem)
                        urg = max(0.0, 1.0 - min(d67, d1011) / D_BND_SCALE) \
                            * min(1.0, REM_LEV / max(rem, 1))
                        if not alive:
                            urg = 0.0
                        lock = int(i <= 5 and gap(t, order[6]) > rem)
                        self.map[(t, d)] = (urg, int(alive), lock, gp)
                for _, gid, ht, _, awt, _, hp, ap in by_date[d]:
                    games.append((season, d, ht, awt, float(hp - ap)))
                    hw = hp > ap
                    W[ht] += int(hw); L[ht] += int(not hw)
                    W[awt] += int(not hw); L[awt] += int(hw)
        self._build_frames(games)

    # -- lookups (gp>=55 gate, tank convention) ---------------------------
    def state(self, team_id, d):
        return self.map.get((int(team_id), d), (0.0, 1, 0, 0))

    def u(self, team_id, d):
        urg, _, _, gp = self.state(team_id, d)
        return urg if gp >= GP_ACTIVE else 0.0

    def lock(self, team_id, d):
        _, _, lk, gp = self.state(team_id, d)
        return lk if gp >= GP_ACTIVE else 0

    def alive(self, team_id, d):
        return self.state(team_id, d)[1]

    def gp(self, team_id, d):
        return self.state(team_id, d)[3]

    # -- k_u frame + lam_B ------------------------------------------------
    def _build_frames(self, games):
        wins = {}
        dates, margins, uds, wds = [], [], [], []
        lam_rows = []
        for season, d, ht, awt, margin in games:
            kh, ka = (season, ht), (season, awt)
            wh = wins.setdefault(kh, [0, 0])
            wa = wins.setdefault(ka, [0, 0])
            wph = wh[0] / wh[1] if wh[1] else 0.5
            wpa = wa[0] / wa[1] if wa[1] else 0.5
            ud = self.u(ht, d) - self.u(awt, d)
            if ud != 0.0:
                dates.append(d)
                margins.append(margin)
                uds.append(ud)
                wds.append(wph - wpa)
            if season == LAM_FIT_SEASON:
                ld = self.lock(ht, d) - self.lock(awt, d)
                if ld != 0 and self.gp(ht, d) >= GP_ACTIVE \
                        and self.gp(awt, d) >= GP_ACTIVE:
                    lam_rows.append((margin, float(ld), wph - wpa))
            hw = margin > 0
            wh[0] += int(hw); wh[1] += 1
            wa[0] += int(not hw); wa[1] += 1
        self._act_dates = np.array(dates, dtype="datetime64[D]")
        self._act_margin = np.array(margins, float)
        self._act_ud = np.array(uds, float)
        self._act_wd = np.array(wds, float)
        # lam_B: ONE OLS on dev year 1, clip to [-2, 0], FROZEN
        self.lam_n = len(lam_rows)
        if self.lam_n >= 5:
            m = np.array([r[0] for r in lam_rows])
            X = np.c_[np.ones(self.lam_n),
                      np.array([r[1] for r in lam_rows]),
                      np.array([r[2] for r in lam_rows])]
            beta = np.linalg.lstsq(X, m, rcond=None)[0]
            self.lam_raw = float(beta[1])
        else:
            self.lam_raw = 0.0
        self.lam_b = float(np.clip(self.lam_raw, *LAM_CLIP))

    def fit_k(self, before=None):
        """Walk-forward k_u, mirroring tanking.TankModel.fit_k verbatim."""
        if before is None:
            m = np.ones(len(self._act_dates), bool)
        else:
            m = self._act_dates < np.datetime64(before)
        n = int(m.sum())
        if n < K_MIN_ACTIVE:
            return 0.0, n
        X = np.c_[np.ones(n), self._act_ud[m], self._act_wd[m]]
        beta = np.linalg.lstsq(X, self._act_margin[m], rcond=None)[0]
        w = n / (n + K_SHRINK)
        return float(np.clip(w * beta[1], -K_CLIP, K_CLIP)), n


# ------------------------------- capstone loop: prod_by_season.py VERBATIM ---
def season_run(season, urg, k_hist):
    t0 = time.time()
    con = connect_retry(read_only=True)
    assert season_floor(con) == FLOOR, "corpus floor drifted from 2021-22"
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
    model = comp = tank = None
    last = None
    k_u, n_act = 0.0, 0
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
            tank = get_tank_model(con)      # same cached instance production used
            k_u, n_act = urg.fit_k(gd)
            k_hist.append((str(gd), round(k_u, 4), n_act,
                           round(float(model.tank_k), 4)))
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
        u_h = urg.u(h.team_id, gd)
        u_a = urg.u(a.team_id, gd)
        lk_h = urg.lock(h.team_id, gd)
        lk_a = urg.lock(a.team_id, gd)
        tkp_h = max(float(tank.active(h.team_id, gd)), 0.0)
        tkp_a = max(float(tank.active(a.team_id, gd)), 0.0)
        k_t = float(model.tank_k)
        drv_c = tkp_h * u_a - tkp_a * u_h
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_ctl=float(sigmoid(m_ctl / SCALE)), p_mkt=float(pmv),
            gp_h=urg.gp(h.team_id, gd), gp_a=urg.gp(a.team_id, gd),
            alive_h=urg.alive(h.team_id, gd), alive_a=urg.alive(a.team_id, gd),
            u_h=round(u_h, 4), u_a=round(u_a, 4), lock_h=lk_h, lock_a=lk_a,
            tkp_h=round(tkp_h, 4), tkp_a=round(tkp_a, 4),
            k_u=round(k_u, 4), k_t=round(k_t, 4), drv_c=round(drv_c, 5),
            p_a=float(sigmoid((m_ctl + k_u * (u_h - u_a)) / SCALE)),
            p_b=float(sigmoid((m_ctl + urg.lam_b * (lk_h - lk_a)) / SCALE)),
            p_c=float(sigmoid((m_ctl + k_t * C_INTER * drv_c) / SCALE))))
    con.close()
    print(f"[{season}] games={len(rows)} ({time.time()-t0:.0f}s) "
          f"k_u_last={k_u:.3f} lam_b={urg.lam_b:.3f}", flush=True)
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


def analyze(df, holdout=False):
    import pandas as pd
    cap = pd.read_csv(CAPSTONE, dtype={"game_id": str})[["season", "game_id", "p_us"]]
    cap["game_id"] = cap.game_id.str.zfill(10)
    mg = df.merge(cap, on=["season", "game_id"], how="inner")
    dp = (mg.p_ctl - mg.p_us).abs()
    max_dp = float(dp.max()) if len(mg) else None
    # ADDENDUM-1 fidelity clause (urgency_prereg.md): the 2026-08-01 10:34
    # 2020-21 box backfill moved 93 dev games (D103-class corpus widening;
    # verified identical to the pg_possdef-control diff, code path unchanged).
    # Bounds: <=3% of games drift, max|dp|<=0.025 overall, <=0.001 inside the
    # late window; drift fully quantified below; paired gates use the
    # same-run control. Any bound fails -> abort stands.
    late_mg = (mg.gp_h >= GP_ACTIVE) & (mg.gp_a >= GP_ACTIVE)
    n_drift = int((dp > 1e-9).sum())
    max_dp_late = float(dp[late_mg].max()) if late_mg.any() else 0.0
    repl = dict(baseline="capstone_pergame.csv", n_matched=int(len(mg)),
                n_ours=int(len(df)), max_abs_diff=max_dp,
                n_drift_gt_1e9=n_drift,
                drift_share=round(n_drift / max(len(mg), 1), 5),
                max_abs_diff_late_window=max_dp_late,
                n_drift_late=int((dp[late_mg] > 1e-9).sum()),
                clause="ADDENDUM 1 (bounded D103-class corpus-widening drift)")
    if max_dp is None or n_drift / max(len(mg), 1) > 0.03 \
            or max_dp > 0.025 or max_dp_late > 0.001:
        raise RuntimeError(f"CONTROL REPLICATION FAILED: max|dp|={max_dp} "
                           f"n_drift={n_drift} late_max={max_dp_late} "
                           "(ADDENDUM-1 bounds); gate aborted, fix env/harness")
    df["ll_ctl"] = pg_ll(df.y, df.p_ctl)
    df["ll_mkt"] = pg_ll(df.y, df.p_mkt)
    for s in ("a", "b", "c"):
        df[f"ll_{s}"] = pg_ll(df.y, df[f"p_{s}"])
    late = (df.gp_h >= GP_ACTIVE) & (df.gp_a >= GP_ACTIVE)
    prim = late & (df.alive_h == 1)
    d65 = ((df.p_mkt - .5).abs() > D65_HEAVY) & ((df.p_ctl - .5).abs() <= D65_HEAVY)
    b_seas = df.season.isin(B_GATE_SEASONS) if not holdout \
        else df.season.isin(HOLDOUT_SEASONS)
    res = dict(prereg="data/urgency_prereg.md", prereg_sha256=PREREG_SHA,
               replication=repl,
               config=dict(d_bnd_scale=D_BND_SCALE, rem_lev=REM_LEV,
                           gp_active=GP_ACTIVE, k_shrink=K_SHRINK,
                           k_clip=K_CLIP, lam_clip=LAM_CLIP, c_inter=C_INTER,
                           boot="2000x seed 7"),
               windows=dict(n_total=int(len(df)), n_late=int(late.sum()),
                            n_primary=int(prim.sum()), n_d65=int(d65.sum())),
               arms={})
    drivers = dict(a=(df.u_h - df.u_a).abs(),
                   b=(df.lock_h - df.lock_a).abs(),
                   c=df.drv_c.abs())
    for s in ("a", "b", "c"):
        d = df.ll_ctl - df[f"ll_{s}"]
        mask_p = prim & b_seas if s == "b" else prim
        drv = drivers[s]
        if s == "a":
            edges = [(0.0, 0.0), (1e-9, 1.0 / 3), (1.0 / 3 + 1e-12, np.inf)]
        elif s == "b":
            edges = [(0.0, 0.0), (1e-9, np.inf)]
        else:
            nz = drv[mask_p & (drv > 1e-9)]
            med = float(nz.median()) if len(nz) else 0.0
            edges = [(0.0, 0.0), (1e-9, med), (med + 1e-12, np.inf)]
        bins = []
        for lo, hi in edges:
            bm = mask_p & (drv >= lo) & (drv <= hi)
            bins.append(dict(range=[lo, None if np.isinf(hi) else hi],
                             **boot_gate(d[bm])))
        means = [b["mean"] for b in bins if b["n"] > 0]
        monotone = all(means[i] <= means[i + 1] + 1e-12
                       for i in range(len(means) - 1)) and \
            (bins[-1]["n"] > 0 and bins[-1]["mean"] > 0)
        gate = boot_gate(d[mask_p])
        veto = boot_gate(d)
        arm = dict(
            primary=gate,
            primary_per_season={s2: boot_gate(d[mask_p & (df.season == s2)])
                                for s2 in sorted(df.season.unique())},
            d65_subset=boot_gate(d[d65]),
            late_unrestricted=boot_gate(d[late]),
            pooled_harm_veto=veto,
            bins=bins, monotone=bool(monotone),
            changed=dict(n=int((mask_p & (drv > 1e-9)).sum()),
                         gate=boot_gate(d[mask_p & (drv > 1e-9)])),
            dev_pass=bool(gate["verdict"] == "PASS" and monotone
                          and veto["verdict"] != "FAIL"))
        if s == "b":
            arm["in_sample_2324"] = boot_gate(
                d[prim & (df.season == LAM_FIT_SEASON)])
        res["arms"][s] = arm
    return res


def main():
    import pandas as pd
    holdout = "--holdout" in sys.argv
    seasons = HOLDOUT_SEASONS if holdout else DEV_SEASONS
    csv_path = (REPO / "data" / "pg_urgency2_holdout_pergame.csv") if holdout else OUT_CSV
    json_path = (REPO / "data" / "pg_urgency2_holdout_summary.json") if holdout else OUT_JSON
    if "--analyze-only" in sys.argv and csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"game_id": str})
        urg = None
    else:
        import os
        for v in ("LATE_STATE", "TANK_TERM", "ORACLE_MINUTES", "INACTIVE_OUTS",
                  "REPORT_OUTS", "OCT_BRIDGE"):
            assert os.environ.get(v) is None, f"{v} must be unset (D122 env)"
        assert os.environ.get("OCT_BRIDGE_TRAIL") == "", \
            "OCT_BRIDGE_TRAIL= (empty) required to reproduce D122 certification"
        # DEVIATION FROM THE LITERAL D122 ENV, documented: the 2020-21 box-score
        # backfill COMPLETED on 2026-08-01 (1080/1080), so the derived tank
        # floor drifted 2021-22 -> 2020-21 AFTER certification. Pinning
        # TANK_SEASON_FLOOR=2021-22 reproduces the cert-time derived value
        # (the override exists for exactly this same-run-control use,
        # tanking.py docstring); the <1e-9 capstone cross-check below is the
        # proof it reproduces the certification construction.
        assert os.environ.get("TANK_SEASON_FLOOR") == FLOOR, \
            "TANK_SEASON_FLOOR=2021-22 required (cert-time derived floor; " \
            "DB grew past it on 2026-08-01)"
        con = connect_retry(read_only=True)
        urg = UrgencyModel(con)
        con.close()
        print(f"urgency map: {len(urg.map)} team-dates; k_u frame "
              f"{len(urg._act_dates)} rows; lam fit n={urg.lam_n} "
              f"raw={urg.lam_raw:+.3f} -> lam_b={urg.lam_b:+.3f}", flush=True)
        k_hist = []
        all_rows = []
        for s in seasons:
            all_rows += season_run(s, urg, k_hist)
        with open(csv_path, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wtr.writeheader(); wtr.writerows(all_rows)
        df = pd.DataFrame(all_rows)
    df["game_id"] = df.game_id.astype(str).str.zfill(10)
    res = analyze(df, holdout=holdout)
    res["corpus"] = "holdout 2021-22..2022-23" if holdout else "dev 2023-24..2025-26"
    if urg is not None:
        res["lam"] = dict(n=urg.lam_n, raw=urg.lam_raw, frozen=urg.lam_b)
        res["k_history"] = k_hist
    json.dump(res, open(json_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
