"""PRE-REGISTERED RE-GATE of F5 (BUILD B urgency term) at 4-season power.

D80 froze the urgency term as F5: pooled +0.00028 CI(-0.00025,+0.00079),
positive point in 3/3 seasons, walk-forward k_u monotone +0.34..+1.24 —
"positive-NS as predicted underpowered — FROZEN as F5 for re-gate when
2021-22 lands (with tank-term interaction check)".

2021-22 player_game_stats has landed. This is the ONE-SHOT power test:
scripts/mr_urgency.py construction EXACTLY — same term, same definition
(|GB_10| <= 3, gp >= 60), same k_u estimator (OLS margin ~ [1, u_diff,
wpct_diff], n/(n+600) shrink, 0 until 20 active rows, clip +-15, weekly
refit), same gate (paired bootstrap 2000x, seed 42, 95% CI) — with the
corpus expanded:

  * WALK-FORWARD k_u CORPUS: + 2021-22. nba_games has NO 2021-22 rows, so
    the 2021-22 standings/active-frame is built from odds_market
    season_end=2022 (scores + dates + home/away abbrevs; regular season =
    game_date <= 2022-04-10; 1228/1230 games, 0 score nulls). Identical
    urgency/GB/wpct arithmetic, abbrev-keyed. NO other definition change.
  * EVAL SEASONS: odds_market DOES cover season_end 2022, but 2021-22
    CANNOT be an eval season — the model corpus (nba_games meta, matchup/
    home flags, player_game_stats-to-date joins) has no 2021-22 rows, so
    fit_production cannot fit that season. Per the pre-registered fallback,
    2021-22 is k_u-training power only; eval = 2022-23..2025-26 (market
    coverage: se2023 matches 1230/1230; se2024-26 as before). 2022-23 has
    NO shipped capstone baseline — control is SAME-RUN (fit_production
    CURRENT = D62 carry + D73 tank), replication checked vs
    capstone_pergame_tank.csv on the seasons it covers (2023-24..2025-26).

D80 addendum honored: tank-term interaction check reported (per-row gated
tank diff recorded; corr(u_diff, tank_diff); active-game gate split by
tank-active). Diagnostics only — the term construction is untouched.

SHIP RULE: pooled 95% CI excludes 0 -> SHIP recommendation with port spec;
else stays FROZEN for 2026-27; report power gained (CI width shrinkage vs
data/mr_urgency_results.json).

Read-only DB. New file scripts/pg_urgency.py; nbapred/ untouched.
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


def connect_retry(read_only=True, attempts=120, wait_s=30):
    """Read-only connect with retry: another session's batch loaders take the
    write lock transiently; a mid-run crash would waste hours of walk-forward.
    Harness robustness only — no modeling change."""
    last = None
    for _ in range(attempts):
        try:
            return connect(read_only=read_only)
        except Exception as e:      # duckdb.IOException on lock conflict
            last = e
            time.sleep(wait_s)
    raise last


SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")   # eval (all with market)
PRIOR_SEASONS = ("2023-24", "2024-25", "2025-26")        # D80 run, for power cmp
GB_WINDOW = 3.0          # within 3 games of the play-in cutoff (either side)
GP_MIN = 60              # gp >= 60 (identical to mr_urgency.py)
K_MIN_ACTIVE = 20
K_CLIP = 15.0
K_SHRINK = 600.0
RS_2122_END = dt.date(2022, 4, 10)   # last day of the 2021-22 regular season


class UrgencyModel:
    """Per-(team_id, date) urgency flag + walk-forward k_u estimator.
    Standings/active-frame from nba_games (2022-23 onward, VERBATIM
    mr_urgency.py) + 2021-22 active rows from odds_market scores."""

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
        self._extend_frame_2122(con)

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

    def _extend_frame_2122(self, con):
        """CORPUS EXPANSION (the only change vs mr_urgency.py): 2021-22
        active rows from odds_market season_end=2022 (nba_games has no
        2021-22). Same standings arithmetic, keyed by abbrev; conference
        via EAST/WEST; sort tiebreak = abbrev (identical semantics to the
        id->abbrev tiebreak above). Regular season = date <= 2022-04-10."""
        rows = con.execute("""
            SELECT game_date, home, away, score_home, score_away
            FROM odds_market
            WHERE season_end = 2022 AND game_date <= DATE '2022-04-10'
              AND score_home IS NOT NULL AND score_away IS NOT NULL
            ORDER BY game_date""").fetchall()
        teams = {t for _, h, a, _, _ in rows for t in (h, a)}
        assert teams <= (EAST | WEST), f"unknown abbrevs: {teams - (EAST | WEST)}"
        conf = {True: [t for t in teams if t in EAST],
                False: [t for t in teams if t in WEST]}
        W = {t: 0 for t in teams}
        L = {t: 0 for t in teams}
        by_date = {}
        for d, h, a, sh, sa in rows:
            d = d.date() if hasattr(d, "date") else d
            by_date.setdefault(d, []).append((h, a, int(sh), int(sa)))
        dates, margins, uds, wds = [], [], [], []
        for d in sorted(by_date):
            u = {}
            for is_east in (True, False):
                ids = conf[is_east]
                if len(ids) < 10:
                    continue
                order = sorted(ids, key=lambda t: (-W[t], L[t], t))
                t10 = order[9]
                for t in ids:
                    gp = W[t] + L[t]
                    gb10 = ((W[t] - W[t10]) + (L[t10] - L[t])) / 2.0
                    u[t] = int(gp >= GP_MIN and abs(gb10) <= GB_WINDOW)
            for h, a, sh, sa in by_date[d]:
                gph, gpa = W[h] + L[h], W[a] + L[a]
                wph = W[h] / gph if gph else 0.5
                wpa = W[a] / gpa if gpa else 0.5
                ud = u.get(h, 0) - u.get(a, 0)
                if ud != 0:
                    dates.append(d)
                    margins.append(float(sh - sa))
                    uds.append(float(ud))
                    wds.append(wph - wpa)
            for h, a, sh, sa in by_date[d]:
                if sh > sa:
                    W[h] += 1; L[a] += 1
                else:
                    W[a] += 1; L[h] += 1
        self.n_2122_games = sum(len(v) for v in by_date.values())
        self.n_2122_active = len(dates)
        self._act_dates = np.concatenate(
            [np.array(dates, dtype="datetime64[D]"), self._act_dates])
        self._act_margin = np.concatenate([np.array(margins, float),
                                           self._act_margin])
        self._act_ud = np.concatenate([np.array(uds, float), self._act_ud])
        self._act_wd = np.concatenate([np.array(wds, float), self._act_wd])

    def fit_k(self, before=None):
        """Walk-forward k_u: OLS home_margin ~ [1, u_diff, wpct_diff] on
        active rows strictly before `before`; n/(n+600) shrink toward 0.
        Returns (k_u, n_active). VERBATIM mr_urgency.py."""
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
    refit); control = fit_production margin (CURRENT = carry + tank),
    variant = + k_u * u_diff. Only addition vs mr_urgency.py: per-row gated
    tank diff recorded for the D80 interaction check."""
    t0 = time.time()
    con = connect_retry(read_only=True)
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
        tsd = float(model.tank_diff(h.team_id, a.team_id, gd))
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), u_home=uh, u_away=ua, k_u=round(k_u, 4),
            tsd=round(tsd, 6),
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
    con = connect_retry(read_only=True)
    urg = UrgencyModel(con)
    con.close()
    print(f"2021-22 corpus: {urg.n_2122_games} games, "
          f"{urg.n_2122_active} active rows; total active frame "
          f"{len(urg._act_dates)}", flush=True)

    k_hist = []
    all_rows = []
    for s in SEASONS:
        all_rows += season_run(s, urg, k_hist)

    with open(OUT_DIR / "pg_urgency_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    month = np.array([int(r["game_date"][5:7]) for r in all_rows])
    uh = np.array([r["u_home"] for r in all_rows])
    ua = np.array([r["u_away"] for r in all_rows])
    ud = uh - ua
    tsd = np.array([r["tsd"] for r in all_rows])
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
                note="2022-23 has no shipped capstone -> same-run control only",
                max_abs_diff=float(max(diffs)) if diffs else None)

    anyurg = (uh == 1) | (ua == 1)
    active = ud != 0
    late = (month == 3) | (month == 4)
    in2326 = np.isin(seas, PRIOR_SEASONS)
    tank_on = tsd != 0.0

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5),
                    ll_market=round(float(ll_m[mask].mean()), 5))

    # ---- power comparison vs the D80 run --------------------------------
    prior_path = OUT_DIR / "mr_urgency_results.json"
    power = None
    if prior_path.exists():
        pr = json.load(open(prior_path))
        p_pool = pr["gate"]["pooled"]["delta"]
        p_act = pr["gate"]["active_udiff"]["delta"]
        new_pool = paired_ci(d)
        new_act = paired_ci(d[active])
        power = dict(
            prior_pooled=dict(n=pr["gate"]["pooled"]["n"], delta=p_pool,
                              ci_width=p_pool[2] - p_pool[1]),
            new_pooled=dict(n=int(len(d)), delta=list(new_pool),
                            ci_width=new_pool[2] - new_pool[1]),
            new_pooled_2326=dict(n=int(in2326.sum()),
                                 delta=list(paired_ci(d[in2326]))),
            prior_active=dict(n=pr["gate"]["active_udiff"]["n"], delta=p_act,
                              ci_width=p_act[2] - p_act[1]),
            new_active=dict(n=int(active.sum()), delta=list(new_act),
                            ci_width=new_act[2] - new_act[1]))

    ks = np.array([r["k_u"] for r in all_rows if r["u_home"] != r["u_away"]])
    res = dict(
        config=dict(gb_window=GB_WINDOW, gp_min=GP_MIN,
                    k_fit="OLS margin~[1,u_diff,wpct_diff], n/(n+600) shrink, "
                          "0 until 20 active rows, clip +-15, weekly refit",
                    definition="within 3 GB of 10th place (either side), "
                               "gp>=60 — IDENTICAL to mr_urgency.py (D80/F5)",
                    corpus="k_u active frame: 2021-22 (odds_market scores, "
                           "regular season <= 2022-04-10) + 2022-23.. "
                           "(nba_games); eval seasons 2022-23..2025-26",
                    eval_note="odds_market covers season_end 2022 but "
                              "nba_games/player joins have no 2021-22 rows "
                              "-> 2021-22 is k_u power only (pre-registered "
                              "fallback); all 4 eval seasons market-covered",
                    gate="paired bootstrap 2000x 95% CI, variant vs control "
                         "(same-run, fit_production current = carry+tank)"),
        corpus_2122=dict(games=urg.n_2122_games, active_rows=urg.n_2122_active),
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
            pooled_2326=sub(in2326),
            per_season={s: sub(seas == s) for s in SEASONS},
            late_season_mar_apr=sub(late),
            playin_team_games=sub(anyurg),
            active_udiff=sub(active)),
        tank_interaction=dict(
            note="D80 freeze addendum: diagnostics only, construction "
                 "untouched",
            corr_ud_tsd_all=round(float(np.corrcoef(ud, tsd)[0, 1]), 4),
            corr_ud_tsd_either=round(float(np.corrcoef(
                ud[active | tank_on], tsd[active | tank_on])[0, 1]), 4)
            if int((active | tank_on).sum()) > 2 else None,
            n_active_and_tank=int((active & tank_on).sum()),
            active_tank_on=sub(active & tank_on),
            active_tank_off=sub(active & ~tank_on)),
        power_vs_prior=power,
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
    with open(OUT_DIR / "pg_urgency_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
