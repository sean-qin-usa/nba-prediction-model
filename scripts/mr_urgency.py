"""BUILD B (pre-registered, Sean's team-motivations directive) — URGENCY
TERM: the positive-effort mirror of the D73 tank term.

Basis (docs/ADVERSE_HYPOTHESES.md H9; D76 / journal 92d7d6): must-win
play-in-race teams overperform our margins (resid +1.49, win rate .572 vs
our .536) and the market shades them +0.47 pts vs us CI(+0.03,+0.93) —
best-supported next margin feature, underpowered at the diagnostic's
definition (2 GB / gp>=65, n=229) -> pre-registered WIDER definition for
power.

TERM:  margin += k_u * (urgency_home - urgency_away)
  urgency(team, date) = 1 if the team is within 3 games of the play-in
  cutoff (10th place in conference, EITHER side of it, |GB_10| <= 3) with
  gp >= 60 (games played this season strictly before date), else 0.
  Standings PIT: W/L strictly before date; conference table sorted by
  (-wins, losses) exactly like the H9 diagnostic (adv_h4h9_effort.py).

K ESTIMATION — mirrors TankModel.fit_k verbatim (walk-forward, self-
contained, 2022-23 burn-in): active rows = completed games strictly before
`before` with nonzero urgency diff; OLS home_margin ~ [1, urgency_diff,
wpct_diff] (season-to-date wpct control de-confounds urgency from team
quality); k_u = n/(n+600) * beta_u, 0 until 20 active rows, clip +-15.
Refit at each weekly refit date. ONE config, no sweeps.

CONTROL = shipped production EXACTLY (fit_production incl. D62 carry + D73
tank), prod_by_season.py loop verbatim (weekly refit, oracle-outs path).
Replication check vs data/capstone_pergame_tank.csv (current headline).

GATE: paired bootstrap 2000x 95% CI on per-game logloss deltas (control -
variant; positive = variant better). Report pooled, per-season, late-season
(Mar-Apr), play-in-team games (either side urgent), active games
(urgency diff != 0). Read-only DB. New file scripts/mr_urgency.py;
nbapred/ untouched.
"""
import csv
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from nbapred.db import connect
from nbapred.model.composition import CompositionModel
from nbapred.model.production import SCALE, fit_production, sigmoid
from nbapred.model.tanking import EAST, WEST

OUT_DIR = REPO / "data"
SEASONS = ("2023-24", "2024-25", "2025-26")
GB_WINDOW = 3.0          # within 3 games of the play-in cutoff (either side)
GP_MIN = 60              # gp >= 60 (wider than the diagnostic's 2 GB / gp>=65)
K_MIN_ACTIVE = 20
K_CLIP = 15.0
K_SHRINK = 600.0


class UrgencyModel:
    """Per-(team_id, date) urgency flag + walk-forward k_u estimator.
    Standings/active-frame built once from nba_games (2022-23 burn-in)."""

    def __init__(self, con):
        tg = con.execute("""
            SELECT season, game_id, game_date, team_id, team_abbrev, wl
            FROM nba_games WHERE game_id LIKE '002%' AND wl IS NOT NULL
              AND season >= '2022-23'
            ORDER BY game_date, game_id""").fetchall()
        # group by season
        by_season = {}
        for season, gid, d, tid, ab, wl in tg:
            d = d.date() if hasattr(d, "date") else d
            by_season.setdefault(season, []).append((d, gid, int(tid), ab, wl))
        self.umap = {}           # (team_id, date) -> 0/1
        for season, rows in by_season.items():
            ab = {}
            for _, _, tid, a, _ in rows:
                ab[tid] = a
            conf_ids = {True: [t for t in ab if ab[t] in EAST],
                        False: [t for t in ab if ab[t] in WEST]}
            W = {t: 0 for t in ab}
            L = {t: 0 for t in ab}
            by_date = {}
            for d, gid, tid, a, wl in rows:
                by_date.setdefault(d, []).append((tid, wl))
            for d in sorted(by_date):
                # emit urgency for ALL teams with state strictly before d
                for is_east in (True, False):
                    ids = conf_ids[is_east]
                    if len(ids) < 10:
                        continue
                    order = sorted(ids, key=lambda t: (-W[t], L[t], ab[t]))
                    t10 = order[9]
                    for t in ids:
                        gp = W[t] + L[t]
                        gb10 = ((W[t] - W[t10]) + (L[t10] - L[t])) / 2.0
                        self.umap[(t, d)] = int(gp >= GP_MIN
                                                and abs(gb10) <= GB_WINDOW)
                for tid, wl in by_date[d]:
                    if wl == "W":
                        W[tid] += 1
                    else:
                        L[tid] += 1
        self._build_active_frame(con)

    def u(self, team_id: int, game_date) -> int:
        return self.umap.get((int(team_id), game_date), 0)

    def diff(self, home_id: int, away_id: int, game_date) -> int:
        return self.u(home_id, game_date) - self.u(away_id, game_date)

    def _build_active_frame(self, con):
        """Completed games with nonzero urgency diff: (date, home margin,
        u_diff, season-to-date wpct diff) — mirrors TankModel."""
        g = con.execute("""
            WITH t AS (SELECT season, game_id, game_date, team_id, is_home, pts
                       FROM nba_games WHERE game_id LIKE '002%'
                       AND pts IS NOT NULL AND season >= '2022-23')
            SELECT h.season, h.game_date, h.team_id ht, a.team_id awt,
                   h.pts hp, a.pts ap
            FROM t h JOIN t a USING (game_id)
            WHERE h.is_home AND NOT a.is_home
            ORDER BY h.game_date, h.game_id""").fetchall()
        wins = {}
        dates, margins, uds, wds = [], [], [], []
        for season, d, ht, awt, hp, ap in g:
            d = d.date() if hasattr(d, "date") else d
            kh, ka = (season, int(ht)), (season, int(awt))
            wh = wins.setdefault(kh, [0, 0])
            wa = wins.setdefault(ka, [0, 0])
            wph = wh[0] / wh[1] if wh[1] else 0.5
            wpa = wa[0] / wa[1] if wa[1] else 0.5
            ud = self.diff(int(ht), int(awt), d)
            if ud != 0:
                dates.append(d)
                margins.append(float(hp - ap))
                uds.append(float(ud))
                wds.append(wph - wpa)
            hw = hp > ap
            wh[0] += int(hw); wh[1] += 1
            wa[0] += int(not hw); wa[1] += 1
        self._act_dates = np.array(dates, dtype="datetime64[D]")
        self._act_margin = np.array(margins, float)
        self._act_ud = np.array(uds, float)
        self._act_wd = np.array(wds, float)

    def fit_k(self, before=None):
        """Walk-forward k_u: OLS home_margin ~ [1, u_diff, wpct_diff] on
        active rows strictly before `before`; n/(n+600) shrink toward 0.
        Returns (k_u, n_active)."""
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


def season_run(season, urg, k_hist):
    """prod_by_season.py loop VERBATIM (default oracle-outs path, weekly
    refit); control = fit_production margin, variant = + k_u * u_diff."""
    t0 = time.time()
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
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
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    model = comp = None
    last = None
    nrefit = 0
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
            nrefit += 1
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            k_u, n_act = urg.fit_k(gd)
            k_hist.append((str(gd), round(k_u, 4), n_act))
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
        mm = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                          gd, b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        uh = urg.u(h.team_id, gd)
        ua = urg.u(a.team_id, gd)
        ud = uh - ua
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), u_home=uh, u_away=ua, k_u=round(k_u, 4),
            p_ctrl=float(sigmoid(mm / SCALE)),
            p_var=float(sigmoid((mm + k_u * ud) / SCALE))))
    con.close()
    print(f"[{season}] n={len(rows)} refits={nrefit} ({time.time()-t0:.0f}s)",
          flush=True)
    return rows


def paired_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    con = connect(read_only=True)
    urg = UrgencyModel(con)
    con.close()

    k_hist = []
    all_rows = []
    for s in SEASONS:
        all_rows += season_run(s, urg, k_hist)

    with open(OUT_DIR / "mr_urgency_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    month = np.array([int(r["game_date"][5:7]) for r in all_rows])
    uh = np.array([r["u_home"] for r in all_rows])
    ua = np.array([r["u_away"] for r in all_rows])
    ud = uh - ua
    ll_c = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    ll_v = ll_vec(y, [r["p_var"] for r in all_rows])
    ll_m = ll_vec(y, [r["p_mkt"] for r in all_rows])
    d = ll_c - ll_v          # positive = variant better

    # ---- control replication check vs shipped capstone (tank headline) ----
    base = {}
    with open(OUT_DIR / "capstone_pergame_tank.csv") as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(baseline="capstone_pergame_tank.csv", n_matched=len(diffs),
                n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None)

    anyurg = (uh == 1) | (ua == 1)
    active = ud != 0
    late = (month == 3) | (month == 4)

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5),
                    ll_market=round(float(ll_m[mask].mean()), 5))

    ks = np.array([r["k_u"] for r in all_rows if r["u_home"] != r["u_away"]])
    res = dict(
        config=dict(gb_window=GB_WINDOW, gp_min=GP_MIN,
                    k_fit="OLS margin~[1,u_diff,wpct_diff], n/(n+600) shrink, "
                          "0 until 20 active rows, clip +-15, weekly refit",
                    definition="within 3 GB of 10th place (either side), "
                               "gp>=60 — wider than diagnostic (2 GB/gp>=65)",
                    gate="paired bootstrap 2000x 95% CI, variant vs control"),
        replication=repl,
        control_ll=dict(pooled=round(float(ll_c.mean()), 5),
                        market=round(float(ll_m.mean()), 5),
                        per_season={s: round(float(ll_c[seas == s].mean()), 4)
                                    for s in SEASONS}),
        variant_ll=dict(pooled=round(float(ll_v.mean()), 5),
                        per_season={s: round(float(ll_v[seas == s].mean()), 4)
                                    for s in SEASONS}),
        gate=dict(
            pooled=sub(np.ones(len(d), bool)),
            per_season={s: sub(seas == s) for s in SEASONS},
            late_season_mar_apr=sub(late),
            playin_team_games=sub(anyurg),
            active_udiff=sub(active)),
        diag=dict(
            n_any_urgent=int(anyurg.sum()),
            n_active=int(active.sum()),
            n_active_per_season={s: int((active & (seas == s)).sum())
                                 for s in SEASONS},
            k_u_on_active=dict(
                mean=round(float(ks.mean()), 4) if len(ks) else 0.0,
                min=round(float(ks.min()), 4) if len(ks) else 0.0,
                max=round(float(ks.max()), 4) if len(ks) else 0.0),
            k_history_tail=[k_hist[i] for i in range(len(k_hist))
                            if i % 4 == 0 or i >= len(k_hist) - 3]),
        )
    with open(OUT_DIR / "mr_urgency_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
