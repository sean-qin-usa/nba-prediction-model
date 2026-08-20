"""D85 step 4 — TALENT-ENSEMBLE GATE (pre-registered, ONE config, zero
fitted params; docs/EXTERNAL_MODELS.md + DECISIONS.md registration).

CONFIG (registered verbatim, no sweep):
  * At each weekly refit date d: for each leg s in {DARKO, EPM, BPM} take the
    PIT map as-of d (DARKO: darko_history date < d; EPM: latest
    identity-bearing Wayback capture asof < d — the ?date= endpoint masks
    identity (lock finding), so captures are the free PIT path; BPM: in-house
    bpm_asof(d), 365d rolling, published low-minutes regression, validated
    corr 0.985/0.994 vs basketball-reference).
  * Common set = players present in ALL available legs; z-score each leg
    cross-sectionally on the common set; ens_z(p) = mean of z over the legs
    that carry p; talent_ens(p) = mu_darko + ens_z(p) * sd_darko (maps back
    to DPM scale, preserving composition.py's scale). Equal weights 1/3
    FIXED A PRIORI. Players in no leg: PIT DARKO fallback (= comp talent,
    counted). EPM missing entirely (refits before a season's first capture):
    ens = mean of the two available legs — the registered missing-source
    rule applied league-wide.
  * Wiring: variant margin = m_ctrl - 0.5*cm_pit + 0.5*cm_ens — the exact
    ORACLE_MINUTES / ao_talent_oracle swap precedent at the
    composition-talent level; nbapred/ is NOT edited.

CONTROL = same-run unmodified fit_production — which as of 2026-07-31 ~03:17
INCLUDES the D90 late-state layer shipped by a parallel session; the paired
delta is control-vs-variant within ONE process (module snapshot at import),
so the gate is internally valid. data/capstone_pergame_tank.csv PREDATES D90
(mtime Jul 30 19:24): the replication check vs it is reported as LINEAGE
DIAGNOSTIC ONLY (expected max|diff| ~0.29 concentrated where latestate
fires), not the ~1e-14 bitwise assertion of the ao_talent_oracle era.

GATE (per execution directive 2026-07-31): PRIMARY = pooled 3-season paired
bootstrap 2000x seed 20260730 on per-game logloss deltas; SECONDARY =
early-season gp<20 window (D62 localized clause; the doc's draft had this
window primary — both reported) and mid-distribution |p_mkt-0.5| <= 0.35
(D77 region — does ensemble talent move toss-ups?). Calibration veto per
COMPLEXITY.md. Family register +1 at pre-registration.

Read-only DB. Outputs: data/ens_talent_gate_pergame.csv,
data/ens_talent_gate_results.json.
Run:     python scripts/ens_talent_gate.py
Analyze: python scripts/ens_talent_gate.py --analyze
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
from nbapred.features.bpm import bpm_asof
from nbapred.model.composition import ROSTER_DAYS, CompositionModel
from nbapred.model.production import (SCALE, _prev_season, continuity_map,
                                      fit_production, sigmoid)

OUT_DIR = REPO / "data"
WB_DIR = REPO / "data/raw/ext_epm/wayback"
SEASONS = ("2023-24", "2024-25", "2025-26")
MID_BAND = 0.35                       # |p_mkt - 0.5| <= 0.35 (D77 region)
SEED = 20260730                       # registered bootstrap seed
PERGAME_CSV = OUT_DIR / "ens_talent_gate_pergame.csv"
RESULTS_JSON = OUT_DIR / "ens_talent_gate_results.json"
CAPSTONE = OUT_DIR / "capstone_pergame_tank.csv"
LEGS = ("darko", "epm", "bpm")


def epm_captures():
    caps = []
    for f in sorted(WB_DIR.glob("parsed_*.json")):
        meta = json.loads(f.read_text())
        caps.append((meta["asof"], f))
    caps.sort()
    return caps


def epm_asof(caps, d):
    best = None
    for asof, f in caps:
        if dt.date.fromisoformat(asof) < d:
            best = (asof, f)
        else:
            break
    if best is None:
        return None, None
    meta = json.loads(best[1].read_text())
    return ({r["player_id"]: float(r["tot"]) for r in meta["rows"]},
            (d - dt.date.fromisoformat(best[0])).days)


def ensemble_map(con, caps, gd, diag):
    """{pid: ensemble talent in DPM scale} per the registered config."""
    darko = CompositionModel._darko_asof(con, gd)
    epm, age = epm_asof(caps, gd)
    bpm = {p: v["bpm"] for p, v in bpm_asof(con, gd).items()}
    legs = dict(darko=darko, bpm=bpm)
    if epm:
        legs["epm"] = epm
    common = set.intersection(*(set(m) for m in legs.values()))
    stats = {}
    for s, m in legs.items():
        v = np.array([m[p] for p in common], float)
        stats[s] = (float(v.mean()), float(v.std()) or 1.0)
    mu_d, sd_d = stats["darko"]
    allp = set().union(*(set(m) for m in legs.values()))
    out = {}
    for p in allp:
        zs = [(legs[s][p] - stats[s][0]) / stats[s][1]
              for s in legs if p in legs[s]]
        out[p] = mu_d + float(np.mean(zs)) * sd_d
    diag.setdefault("refits", []).append(dict(
        date=str(gd), n_common=len(common), n_all=len(allp),
        epm_age_days=age, legs=sorted(legs)))
    return out


def check_ff_ready(con, season, before):
    from nbapred.model.four_factors import factor_game_rows
    cur = factor_game_rows(con, season, before=before)
    if cur:
        return True
    if continuity_map(con, season, before=before) is None:
        return False
    return bool(factor_game_rows(con, _prev_season(season), before=None))


def season_run(season, caps, diag):
    """ao_talent_oracle.season_run VERBATIM except the talent swap source."""
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
    gp_live = {}
    model = comp = None
    etal = {}
    last = None
    nrefit = 0
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
            if last is None:
                ready = check_ff_ready(con, season, gd)
                diag.setdefault("ff_ready_week1", {})[season] = bool(ready)
                if not ready:
                    raise RuntimeError(
                        f"ff not ready at {gd} — 0.5 blend swap invalid")
            nrefit += 1
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            emap = ensemble_map(con, caps, gd, diag)
            nfall = 0
            etal = {}
            for pid, p in comp.players.items():
                if pid in emap:
                    etal[pid] = float(emap[pid])
                else:
                    etal[pid] = p["talent"]
                    nfall += 1
            diag.setdefault("fallbacks", []).append(
                (str(gd), nfall, len(comp.players)))
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        gph = gp_live.get(h.team_id, 0)
        gpa = gp_live.get(a.team_id, 0)
        gp_live[h.team_id] = gph + 1
        gp_live[a.team_id] = gpa + 1
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
        cm_pit = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                             gd, home_edge=0.0)

        def cm_with(talent):
            s = 0.0
            for pid, p in comp.players.items():
                tid = p["team_id"]
                if tid == h.team_id and pid not in outs[h.team_id] \
                        and (gd - p["last_played"]).days <= ROSTER_DAYS:
                    s += talent[pid] * p["trail_min"] / 48.0
                elif tid == a.team_id and pid not in outs[a.team_id] \
                        and (gd - p["last_played"]).days <= ROSTER_DAYS:
                    s -= talent[pid] * p["trail_min"] / 48.0
            return s

        cm_check = cm_with({pid: p["talent"] for pid, p in comp.players.items()})
        if abs(cm_check - cm_pit) > 1e-9:
            raise RuntimeError(f"comp replication broke: {cm_check} vs {cm_pit}")
        cm_ens = cm_with(etal)
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), gp_home=gph, gp_away=gpa,
            cm_pit=round(cm_pit, 4), cm_ens=round(cm_ens, 4),
            p_ctrl=float(sigmoid(mm / SCALE)),
            p_var=float(sigmoid((mm - 0.5 * cm_pit + 0.5 * cm_ens) / SCALE))))
    con.close()
    print(f"[{season}] n={len(rows)} refits={nrefit} ({time.time()-t0:.0f}s)",
          flush=True)
    return rows


def paired_ci(d, B=2000, seed=SEED):
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


def analyze(all_rows, diag):
    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    pmk = np.array([r["p_mkt"] for r in all_rows])
    gpmin = np.array([min(r["gp_home"], r["gp_away"]) for r in all_rows])
    mid = np.abs(pmk - 0.5) <= MID_BAND
    ll_c = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    ll_v = ll_vec(y, [r["p_var"] for r in all_rows])
    ll_m = ll_vec(y, pmk)
    d = ll_c - ll_v                        # positive = ensemble better
    dvm = ll_m - ll_v

    base = {}
    with open(CAPSTONE) as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    late = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
            for r in all_rows if (r["season"], r["game_id"]) in base
            and r["game_date"][5:7] in ("02", "03", "04")]
    early = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base
             and r["game_date"][5:7] not in ("02", "03", "04")]
    repl = dict(baseline=CAPSTONE.name, n_matched=len(diffs),
                n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None,
                note="capstone CSV predates the D90 latestate ship "
                     "(2026-07-31); lineage diagnostic only — paired gate "
                     "uses SAME-RUN control",
                max_abs_diff_feb_apr=float(max(late)) if late else None,
                max_abs_diff_oct_jan=float(max(early)) if early else None)

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    delta_vs_mkt=paired_ci(dvm[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5),
                    ll_market=round(float(ll_m[mask].mean()), 5))

    # calibration veto inputs: slope proxy via decile calibration gap
    cal = {}
    for tag, p in (("ctrl", [r["p_ctrl"] for r in all_rows]),
                   ("var", [r["p_var"] for r in all_rows])):
        p = np.asarray(p)
        bins = np.clip((p * 10).astype(int), 0, 9)
        gap = [abs(float(y[bins == b].mean() - p[bins == b].mean()))
               for b in range(10) if (bins == b).sum() >= 30]
        cal[tag] = round(float(np.mean(gap)), 5)

    shift = [abs(r["cm_ens"] - r["cm_pit"]) for r in all_rows]
    ages = [r.get("epm_age_days") for r in diag.get("refits", [])
            if r.get("epm_age_days") is not None]
    res = dict(
        config=dict(legs=list(LEGS), weights="equal 1/3 fixed a priori",
                    zscore="cross-sectional on common set, back to DPM via "
                           "darko mu/sd", mid_band=MID_BAND, seed=SEED,
                    swap="m - 0.5*cm_pit + 0.5*cm_ens (ORACLE_MINUTES "
                         "precedent)",
                    epm_source="identity-bearing Wayback captures (lock "
                               "finding: ?date= endpoint masks identity)"),
        replication=repl,
        gate=dict(pooled_PRIMARY=sub(np.ones(len(d), bool)),
                  gp_lt20_SECONDARY=sub(gpmin < 20),
                  mid_distribution_SECONDARY=sub(mid),
                  per_season={s: sub(seas == s) for s in SEASONS}),
        calibration_decile_gap=cal,
        diagnostics=dict(
            mid_per_season={s: sub(mid & (seas == s)) for s in SEASONS},
            gp_ge20=sub(gpmin >= 20),
            mean_abs_cm_shift=round(float(np.mean(shift)), 4),
            epm_capture_age_days=dict(
                mean=round(float(np.mean(ages)), 1) if ages else None,
                median=float(np.median(ages)) if ages else None,
                max=float(np.max(ages)) if ages else None),
            n_refits=len(diag.get("refits", [])),
            fallback_share_last_refit=(
                None if not diag.get("fallbacks") else
                round(diag["fallbacks"][-1][1] / diag["fallbacks"][-1][2], 4)),
            ff_ready_week1=diag.get("ff_ready_week1")))
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


def main():
    if "--analyze" in sys.argv:
        with open(PERGAME_CSV) as f:
            all_rows = []
            for r in csv.DictReader(f):
                r.update(y=int(r["y"]), p_mkt=float(r["p_mkt"]),
                         p_ctrl=float(r["p_ctrl"]), p_var=float(r["p_var"]),
                         cm_pit=float(r["cm_pit"]), cm_ens=float(r["cm_ens"]),
                         gp_home=int(r["gp_home"]), gp_away=int(r["gp_away"]))
                all_rows.append(r)
        diag = {}
        dj = RESULTS_JSON.with_suffix(".diag.json")
        if dj.exists():
            diag = json.loads(dj.read_text())
        analyze(all_rows, diag)
        return
    caps = epm_captures()
    print(f"EPM captures: {len(caps)}"
          + (f" ({caps[0][0]}..{caps[-1][0]})" if caps else ""))
    diag = {}
    all_rows = []
    for s in SEASONS:
        all_rows += season_run(s, caps, diag)
    with open(PERGAME_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)
    RESULTS_JSON.with_suffix(".diag.json").write_text(
        json.dumps(diag, default=str))
    analyze(all_rows, diag)


if __name__ == "__main__":
    main()
