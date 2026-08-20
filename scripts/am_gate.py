"""AM GATE — the three pre-registered arms.

Pre-registration: data/adaptive_memory_prereg.md
sha256 a3aae24b3193c2cf04c5e438b4c3c0b7ea51211ac5100c474f3d3dec64ec3bc9

  ARM A  F1 late-gated FORM term, re-gated on the CURRENT certified stack.
         fit_form_k / FORM_GP / FORM_N / FORM_SHRINK / FORM_WINDOW_D are
         IMPORTED VERBATIM from scripts/ba_windowed.py (the D71 gate script),
         exactly as D102 RT2 did.  The licence for this third look is that the
         baseline MATERIALLY CHANGED: RT2's control contained the D90
         late-state layer, which D112/D118 REVERTED (LATE_STATE defaults "0").

  ARM B  PHASE-VARYING memory correction, h(gp) = 34 / 13 / 8 on
         gp[20,41)/[41,55)/[55,inf).  The owner's hypothesis, literal.

  ARM C  CONSTANT memory correction, h = 21 everywhere on gp>=20.  Identical
         code path to ARM B with one substituted h function.

Control: data/capstone_pergame_d132.csv (D132 certified, md5
dc256d0b85a072c0083f074083194283).  Margins recovered exactly by
m = SCALE*logit(p_us).  Writes data/am_pergame.csv + data/am_gate_results.json.
Read-only DB.  Edits nothing under nbapred/.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from ba_windowed import (FORM_GP, FORM_N, FORM_SHRINK, FORM_WINDOW_D,  # noqa: E402
                         fit_form_k)                                   # VERBATIM D71
from ds_corpus import arm_connection  # noqa: E402
from nbapred.model.production import SCALE, sigmoid  # noqa: E402

CTRL = ROOT / "data" / "capstone_pergame_d132.csv"
LIVE = ROOT / "data" / "capstone_pergame.csv"
SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

GAP_GP = 20               # both-teams gate for the memory-correction arms
GAP_SHRINK = FORM_SHRINK  # 600, mirrored from fit_form_k
GAP_WINDOW_D = FORM_WINDOW_D  # 730, mirrored from fit_form_k


def h_phase(gp: int) -> float:
    """ARM B — MONOTONE DECREASING memory (more aggressive later)."""
    if gp < 41:
        return 34.0
    if gp < 55:
        return 13.0
    return 8.0


def h_const(gp: int) -> float:
    """ARM C — CONSTANT memory at the measured optimum."""
    return 21.0


def wmean(ms, h) -> float:
    n = len(ms)
    if n == 0:
        return 0.0
    age = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / h)
    return float(np.dot(w, ms) / w.sum())


def gap(ms, hfun) -> float:
    """Recency INNOVATION: decayed mean minus the uniform mean the production
    stack already carries.  Orthogonal to level by construction."""
    if len(ms) < 1:
        return 0.0
    return wmean(ms, hfun(len(ms))) - float(np.mean(ms))


def host_of(matchup: str) -> str:
    """The production home-team rule, verbatim from prod_by_season.py."""
    return (matchup.split("@")[-1].strip() if "@" in matchup
            else matchup.split("vs.")[0].strip())


def fit_gap_k(con, before, hfun):
    """k for the memory-correction term — a STRUCTURAL MIRROR of
    ba_windowed.fit_form_k: same 730d trailing window, same
    X = [1, regressor_diff, wpct_diff] with the wpct control fit-only, same
    n/(n+600) shrink toward 0.  Only the regressor and the row gate differ:
    the regressor is the recency innovation `gap` and rows require BOTH teams
    at gp >= 20.  Returns (k, n, beta_raw)."""
    rows = con.execute("""
        SELECT season, game_id, game_date, team_id, is_home, pts, matchup
        FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
        AND game_date < ? ORDER BY game_date, game_id""", [before]).fetchall()
    byg = {}
    for season, gid, d, tid, ish, pts, mu in rows:
        d = d.date() if hasattr(d, "date") else d
        g = byg.setdefault(gid, dict(season=season, date=d, rows=[]))
        g["rows"].append((int(tid), str(mu), float(pts), bool(ish)))
    glist = []
    for gid, g in byg.items():
        if len(g["rows"]) != 2:
            continue
        (t0, mu0, p0, ih0), (t1, mu1, p1, ih1) = g["rows"]
        glist.append(g)
        g["h"], g["a"] = ((t0, p0), (t1, p1)) if ih0 else ((t1, p1), (t0, p0))
    glist.sort(key=lambda g: (g["date"], g["season"]))
    lo = before - _dt.timedelta(days=GAP_WINDOW_D)
    marg = {}
    X, y = [], []
    for g in glist:
        s = g["season"]
        ht, hp = g["h"]
        at, ap = g["a"]
        mh = marg.get((s, ht), [])
        ma = marg.get((s, at), [])
        if (g["date"] >= lo and len(mh) >= GAP_GP and len(ma) >= GAP_GP):
            gd = gap(mh, hfun) - gap(ma, hfun)
            wd = (sum(m > 0 for m in mh) / len(mh)
                  - sum(m > 0 for m in ma) / len(ma))
            X.append([1.0, gd, wd])
            y.append(hp - ap)
        marg.setdefault((s, ht), []).append(hp - ap)
        marg.setdefault((s, at), []).append(ap - hp)
    n = len(X)
    if n == 0:
        return 0.0, 0, 0.0
    beta = np.linalg.lstsq(np.array(X), np.array(y, float), rcond=None)[0]
    w = n / (n + GAP_SHRINK)
    return float(w * beta[1]), n, float(beta[1])


def build_state(con):
    """Per (season, team) chronological signed margins for the SCORED seasons,
    with home/away resolved by the PRODUCTION matchup rule (so the 10
    neutral-site games RT2 dropped are handled the way production handles
    them).  Returns a snapshot per game_id taken BEFORE that game is added."""
    rows = con.execute("""
        SELECT season, game_id, game_date, team_id, team_abbrev, is_home, pts,
               matchup
        FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
          AND season IN ('2021-22','2022-23','2023-24','2024-25','2025-26')
        ORDER BY game_date, game_id""").fetchall()
    byg = {}
    for season, gid, d, tid, ab, ish, pts, mu in rows:
        d = d.date() if hasattr(d, "date") else d
        byg.setdefault(gid, dict(season=season, date=d, gid=gid,
                                 rows=[]))["rows"].append(
            (int(tid), str(ab), float(pts), str(mu)))
    games = []
    for gid, g in byg.items():
        if len(g["rows"]) != 2:
            continue
        host = host_of(g["rows"][0][3])
        r0, r1 = g["rows"]
        h, a = (r0, r1) if r0[1] == host else (r1, r0)
        g["home"], g["away"] = h, a
        games.append(g)
    games.sort(key=lambda g: (g["date"], g["gid"]))
    hist, snap = {}, {}
    for g in games:
        s = g["season"]
        ht, _, hp, _ = g["home"]
        at, _, ap, _ = g["away"]
        mh = hist.setdefault((s, ht), [])
        ma = hist.setdefault((s, at), [])
        snap[g["gid"]] = (list(mh), list(ma), g["home"][1], g["away"][1])
        mh.append(hp - ap)
        ma.append(ap - hp)
    return snap


def main():
    ctrl = list(csv.DictReader(open(CTRL)))
    con = arm_connection(None)                 # FULL corpus, read-only views
    snap = build_state(con)

    # weekly refit cadence, identical to prod_by_season.py
    refit = {}
    for s in SEASONS:
        ds = sorted({_dt.date.fromisoformat(r["game_date"])
                     for r in ctrl if r["season"] == s})
        last = None
        for d in ds:
            if last is None or (d - last).days >= 7:
                last = d
            refit[(s, d)] = last
    rdates = sorted(set(refit.values()))
    print(f"refit dates: {len(rdates)}", flush=True)

    kf, kgB, kgC = {}, {}, {}
    klog = []
    for i, rd in enumerate(rdates):
        kf[rd] = fit_form_k(con, rd)
        kgB[rd] = fit_gap_k(con, rd, h_phase)
        kgC[rd] = fit_gap_k(con, rd, h_const)
        klog.append({"date": str(rd), "k_form": round(kf[rd][0], 5),
                     "n_form": kf[rd][1], "k_gapB": round(kgB[rd][0], 5),
                     "n_gapB": kgB[rd][1], "k_gapC": round(kgC[rd][0], 5),
                     "n_gapC": kgC[rd][1]})
        if i % 10 == 0:
            print(f"  [{i+1}/{len(rdates)}] {rd} kform={kf[rd][0]:+.5f} "
                  f"kB={kgB[rd][0]:+.5f} kC={kgC[rd][0]:+.5f}", flush=True)

    out_rows = []
    hostmismatch = 0
    for r in ctrl:
        gid = r["game_id"]
        if gid not in snap:
            raise SystemExit(f"control row {gid} absent from DB state")
        mh, ma, hab, aab = snap[gid]
        if hab != r["home"] or aab != r["away"]:
            hostmismatch += 1
        s, d = r["season"], _dt.date.fromisoformat(r["game_date"])
        rd = refit[(s, d)]
        p_ctl = float(r["p_us"])
        m_ctl = SCALE * float(np.log(p_ctl / (1 - p_ctl)))
        gph, gpa = len(mh), len(ma)

        # ---- ARM A : D71 late-gated form (verbatim construction) ----------
        tA = 0.0
        if gph >= FORM_GP or gpa >= FORM_GP:
            f5h = float(np.mean(mh[-FORM_N:])) if gph >= FORM_N else 0.0
            f5a = float(np.mean(ma[-FORM_N:])) if gpa >= FORM_N else 0.0
            tA = kf[rd][0] * (f5h - f5a)
        # ---- ARMS B / C : memory correction --------------------------------
        tB = tC = 0.0
        if gph >= GAP_GP and gpa >= GAP_GP:
            tB = kgB[rd][0] * (gap(mh, h_phase) - gap(ma, h_phase))
            tC = kgC[rd][0] * (gap(mh, h_const) - gap(ma, h_const))
        out_rows.append(dict(
            season=s, game_id=gid, game_date=r["game_date"], y=int(r["y"]),
            p_mkt=float(r["p_mkt"]), p_ctl=p_ctl, gp_home=gph, gp_away=gpa,
            term_A=tA, term_B=tB, term_C=tC,
            p_A=float(sigmoid((m_ctl + tA) / SCALE)),
            p_B=float(sigmoid((m_ctl + tB) / SCALE)),
            p_C=float(sigmoid((m_ctl + tC) / SCALE)),
            neutral=int(hab != r["home"] and aab != r["away"])))
    print(f"host-rule mismatches vs certified CSV: {hostmismatch}")

    with open(ROOT / "data" / "am_pergame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ------- control fidelity (D134) --------------------------------------
    live = {r["game_id"]: float(r["p_us"]) for r in csv.DictReader(open(LIVE))}
    dps = [abs(r["p_ctl"] - live[r["game_id"]]) for r in out_rows
           if r["game_id"] in live]
    fid = {"n_matched": len(dps), "max_abs_dp": max(dps) if dps else None,
           "games_moved": int(sum(x > 1e-12 for x in dps))}

    def ll(p, y):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = {"prereg_sha256": "a3aae24b3193c2cf04c5e438b4c3c0b7ea51211ac5100c474f3"
                            "d3dec64ec3bc9",
           "control": str(CTRL.name), "control_fidelity_vs_live": fid,
           "n": len(out_rows), "k_trajectory": klog,
           "constants": {"FORM_GP": FORM_GP, "FORM_N": FORM_N,
                         "FORM_SHRINK": FORM_SHRINK,
                         "FORM_WINDOW_D": FORM_WINDOW_D, "GAP_GP": GAP_GP,
                         "h_phase": "34 / 13 / 8 on gp[20,41)/[41,55)/[55,)",
                         "h_const": 21.0}}
    for arm in "ABC":
        d = np.array([ll(r["p_ctl"], r["y"]) - ll(r[f"p_{arm}"], r["y"])
                      for r in out_rows])
        touched = np.array([r[f"term_{arm}"] != 0.0 for r in out_rows])
        off = [abs(r[f"p_{arm}"] - r["p_ctl"]) for r in out_rows
               if r[f"term_{arm}"] == 0.0]
        res[f"arm_{arm}"] = {
            "n_touched": int(touched.sum()),
            "mean_abs_term": float(np.mean([abs(r[f"term_{arm}"])
                                            for r in out_rows
                                            if r[f"term_{arm}"] != 0.0])),
            "max_abs_dp": float(max(abs(r[f"p_{arm}"] - r["p_ctl"])
                                    for r in out_rows)),
            "zero_outside_window_max_abs_dp": float(max(off) if off else 0.0),
            "pooled_mean_delta": float(d.mean())}
    (ROOT / "data" / "am_gate_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "k_trajectory"},
                     indent=1))


if __name__ == "__main__":
    main()
