#!/usr/bin/env python3
"""PCA / LOW-RANK COMPRESSION OF THE REJECTED PILE — the scoring run.

Pre-registration: data/pca_prereg.md
sha256 0c3720ba99b668ff769b147d23aa45fa507eecd7b038a5e35bfdff44e1b63938
(frozen BEFORE this file produced a single log-loss number).

Control = data/capstone_pergame.csv, the D132 certified artifact.  Every arm is
a purely ADDITIVE schedule-layer change, so
    p_arm = sigmoid((SCALE*logit(p_us) + dm) / SCALE)
and the D134 control hash is max|dp| = 0.0 on 6148/6148 games trivially; the
compensating identity checks are prereg T1 (PCR at r=k == raw ALL15) and
`ca_verify.py`'s 8.3e-14.

  TANK_SEASON_FLOOR=2020-21 python scripts/pc_ladder.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nbapred.threads as _t  # noqa: E402

_t.pin(1)
import datetime as dt  # noqa: E402

import numpy as np  # noqa: E402

from ca_bank import TERM_NAMES, Layer, load_bank  # noqa: E402
from ca_ladder import refit_dates  # noqa: E402
from nbapred.model.production import SCALE  # noqa: E402
from pc_layer import (MAX_PC, PC_SLOT0, PILE13, EBLayer, Rotation,  # noqa: E402
                      install, widen)

CERT = REPO / "data" / "capstone_pergame.csv"
SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
NBOOT = 2000
SEED = 20260803
PREREG = "0c3720ba99b668ff769b147d23aa45fa507eecd7b038a5e35bfdff44e1b63938"

R_SWEEP = list(range(0, 14))            # PILE13 -> r components (+ dead2)
RB_SWEEP = [5, 10, 20, 30, 35, 40, 43]  # JOINT43 -> r components (+ dead2)
# The OTHER one-parameter shrinkage family, for the like-for-like comparison
# the directive asks for: L2 on the STANDARDISED carried block.  lambda=0 must
# reproduce raw ALL15 exactly.  Descriptive curve, not a gated arm.
L_SWEEP = [0.0, 12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0,
           3200.0, 6400.0, 12800.0]
DEAD = (0, 1)


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(bank, seasons, verbose=True):
    """Returns (order, dates, dm dict, diag)."""
    L = Layer(bank)                    # arms A/B and the raw comparators
    E = EBLayer(bank, eb=True)         # arm C
    T = EBLayer(bank, eb=True)         # |t| ordering of the components
    R = {lam: EBLayer(bank, ridge=lam) for lam in L_SWEEP}   # ridge family
    RD = {lam: EBLayer(bank, ridge=lam, ridge_dead=True) for lam in L_SWEEP}

    seas = bank["season"].astype(str)
    sel = np.where(np.isin(seas, list(seasons)))[0]
    order = sel[np.argsort(bank["date"][sel], kind="stable")]
    dates = bank["date"][order].astype("datetime64[D]").astype(dt.date)

    names = (["raw:ALL15", "raw:ALL15+TEAMHOME", "raw:DENSE13",
              "A:PCA90", "B:PCA90+TH", "C:EB_ALL15", "C2:EB_ALL15+TEAMHOME"]
             + [f"pca:r{r}" for r in R_SWEEP]
             + [f"pcat:r{r}" for r in R_SWEEP]
             + [f"pcaTH:r{r}" for r in RB_SWEEP]
             + [f"ridge:L{lam:g}" for lam in L_SWEEP]
             + [f"ridgeD:L{lam:g}" for lam in L_SWEEP])
    dm = {k: np.zeros(len(order)) for k in names}
    dm["_shipped"] = np.zeros(len(order))
    diag = {"r_rule": [], "eb": [], "rotation": [], "identity": []}

    rd_all = []
    for s in seasons:
        m = seas[order] == s
        if not m.any():
            continue
        for d in refit_dates(sorted(set(dates[m]))):
            rd_all.append((s, d))
    gov = np.empty(len(order), dtype=object)
    for s in seasons:
        m = seas[order] == s
        if not m.any():
            continue
        rds = refit_dates(sorted(set(dates[m])))
        j = 0
        for i in np.where(m)[0]:
            while j + 1 < len(rds) and rds[j + 1] <= dates[i]:
                j += 1
            gov[i] = rds[j]
    if verbose:
        print(f"[pca] {len(order)} games, {len(rd_all)} refit dates, "
              f"{len(names)} arms", flush=True)

    t0 = time.time()
    for n_, (s, d) in enumerate(rd_all):
        rows = np.where(gov == d)[0]
        rows = rows[seas[order][rows] == s]
        if len(rows) == 0:
            continue
        b_int = int(np.datetime64(d, "D").astype("int64"))
        fit_rows = L._rows(d, b_int - L.window)
        need = np.unique(np.concatenate([fit_rows, order[rows]]))

        b5, _, _, _, _, _ = L.fit(d)                        # shipped baseline
        for i in rows:
            dm["_shipped"][i] = L.sched_value(order[i], b5, {}, {}, [])

        def emit(name, layer, cols, teamhome=False, apply_dead=DEAD):
            bb5, ex, thd, _, _, nb = layer.fit(d, cols=cols, teamhome=teamhome)
            for i in rows:
                dm[name][i] = layer.sched_value(
                    order[i], bb5, ex, thd, nb, apply_dead=apply_dead) \
                    - dm["_shipped"][i]
            return ex, thd

        # ---- raw comparators (reproduce D154's ladder in-run) -------------
        emit("raw:ALL15", L, PILE13)
        emit("raw:ALL15+TEAMHOME", L, PILE13, teamhome=True)
        emit("raw:DENSE13", L, PILE13, apply_dead=())
        emit("C:EB_ALL15", E, PILE13)
        emit("C2:EB_ALL15+TEAMHOME", E, PILE13, teamhome=True)

        # ---- ARM A / the variance-ordered sweep --------------------------
        rot = Rotation(0.90).fit(bank, fit_rows, d)
        diag["r_rule"].append(dict(date=str(d), block="PILE13", p=rot.p,
                                   r90=rot.r_rule, n=int(len(fit_rows)),
                                   ev=[round(float(x), 4) for x in rot.ev]))
        for r in R_SWEEP:
            slots = install(bank, rot, need, r) if r else []
            emit(f"pca:r{r}", L, slots)
            if r == rot.r_rule:
                dm["A:PCA90"][rows] = dm[f"pca:r{r}"][rows]

        # ---- RIDGE family on the STANDARDISED pile (V = identity) --------
        rz = Rotation(0.90)
        rz.mu, rz.sd, rz.ev = rot.mu, rot.sd, np.ones(rot.p)
        rz.V, rz.cols, rz.teamhome, rz.teams, rz.p = (np.eye(rot.p), rot.cols,
                                                      False, [], rot.p)
        slots_z = install(bank, rz, need, rot.p)
        for lam in L_SWEEP:
            emit(f"ridge:L{lam:g}", R[lam], slots_z)
            emit(f"ridgeD:L{lam:g}", RD[lam], slots_z)

        # ---- |t|-ordered sweep (descriptive, supervised ordering) --------
        slots_all = install(bank, rot, need, rot.p)
        T.eb_diag.clear()
        T.fit(d, cols=slots_all)
        tt = T.eb_diag[-1]["t_extras"]
        ordt = sorted(range(rot.p), key=lambda j: -abs(tt[PC_SLOT0 + j]))
        Vt = rot.V[:, ordt]
        rt = Rotation(0.90)
        rt.mu, rt.sd, rt.V, rt.ev = rot.mu, rot.sd, Vt, rot.ev[ordt]
        rt.cols, rt.teamhome, rt.teams, rt.p = rot.cols, False, [], rot.p
        for r in R_SWEEP:
            slots = install(bank, rt, need, r) if r else []
            emit(f"pcat:r{r}", L, slots)

        # ---- ARM B / the team-home sweep ---------------------------------
        rotB = Rotation(0.90, teamhome=True).fit(bank, fit_rows, d)
        diag["r_rule"].append(dict(date=str(d), block="JOINT43", p=rotB.p,
                                   r90=rotB.r_rule, n=int(len(fit_rows))))
        for r in sorted(set(RB_SWEEP + [rotB.r_rule])):
            slots = install(bank, rotB, need, min(r, MAX_PC))
            nm = f"pcaTH:r{r}"
            if nm in dm:
                emit(nm, L, slots)
            if r == rotB.r_rule:
                bb5, ex, thd, _, _, nb = L.fit(d, cols=slots)
                for i in rows:
                    dm["B:PCA90+TH"][i] = L.sched_value(
                        order[i], bb5, ex, thd, nb, apply_dead=DEAD) \
                        - dm["_shipped"][i]
        if verbose and (n_ + 1) % 20 == 0:
            print(f"   refit {n_+1}/{len(rd_all)}  {time.time()-t0:.0f}s",
                  flush=True)

    diag["eb"] = E.eb_diag
    diag["ridge_edf"] = {f"ridge:L{lam:g}": float(np.mean(R[lam].edf))
                         for lam in L_SWEEP if R[lam].edf}
    diag["ridge_edf"].update({f"ridgeD:L{lam:g}": float(np.mean(RD[lam].edf))
                              for lam in L_SWEEP if RD[lam].edf})
    if verbose:
        print(f"[pca] done in {time.time()-t0:.0f}s", flush=True)
    return order, dates, dm, diag, names


if __name__ == "__main__":
    os.environ.setdefault("TANK_SEASON_FLOOR", "2020-21")
    bank0 = load_bank()
    bank = widen(bank0)
    assert bank["X"].shape[1] == 15 + MAX_PC

    order, dates, dm, diag, names = run(bank, SEASONS)

    # ---- prereg T1: PCR at r = k must reproduce raw ALL15 EXACTLY --------
    t1 = float(np.abs(dm["pca:r13"] - dm["raw:ALL15"]).max())
    print(f"\n[T1 rotation-invariance identity] max|dm_PCA(r=13) - dm_ALL15| "
          f"= {t1:.3e}   {'OK' if t1 < 1e-9 else 'FAIL'}")
    assert t1 < 1e-9, "ROTATION INVARIANCE VIOLATED — harness is wrong"
    t1z = float(np.abs(dm["ridge:L0"] - dm["raw:ALL15"]).max())
    print(f"[T1z ridge lambda=0 identity]     max|dm_ridge(0) - dm_ALL15|    "
          f"= {t1z:.3e}   {'OK' if t1z < 1e-9 else 'FAIL'}")
    assert t1z < 1e-9, "RIDGE(0) != OLS — harness is wrong"
    dth = np.abs(dm["pcaTH:r43"] - dm["raw:ALL15+TEAMHOME"])
    t1b = float(dth.max())
    n_off = int((dth > 1e-8).sum())
    print(f"[T1b joint block]                 max|dm_PCA(r=43) - dm_ALL15+TH| "
          f"= {t1b:.3e} on {n_off}/{len(dth)} games")
    print("    (NOT an identity: the ALL15+TEAMHOME design is RANK-DEFICIENT — "
          "alt_home_km is an exact linear function of the home-team dummies, so "
          "the fitted function is determined only on the fit frame's span and "
          "the two parametrisations disagree exactly at NEUTRAL-SITE games, "
          "where that collinearity is broken.)")

    # ---- reproduction of D154's own rows, where they exist ---------------
    rep = {}
    rz = REPO / "data" / "carryall_ladder_rows.npz"
    if rz.exists():
        z = np.load(rz, allow_pickle=True)
        gid_ca = z["gid"].astype(str)
        pos = {g: i for i, g in enumerate(gid_ca)}
        gids_all = bank["gid"][order].astype(str)
        for mine, theirs in (("raw:ALL15", "dm_pile:ALL15"),
                             ("raw:ALL15+TEAMHOME", "dm_pile:ALL15+TEAMHOME"),
                             ("raw:DENSE13", "dm_pile:DENSE13")):
            if theirs not in z.files:
                continue
            m = np.array([g in pos for g in gids_all])
            a = dm[mine][m]
            b = z[theirs][[pos[g] for g in gids_all[m]]]
            rep[mine] = float(np.abs(a - b).max())
        print("[D154 reproduction] max|dm_mine - dm_D154|: "
              + ", ".join(f"{k} {v:.3e}" for k, v in rep.items()))

    cert = {r["game_id"]: (float(r["p_us"]), float(r["y"]), r["season"],
                           float(r["p_mkt"]))
            for r in csv.DictReader(open(CERT))}
    gids = bank["gid"][order].astype(str)
    keep = np.array([g in cert for g in gids])
    idx = np.where(keep)[0]
    print(f"[pca] matched {keep.sum()}/{len(gids)} games against {CERT.name}")

    p0 = np.array([cert[g][0] for g in gids[idx]])
    y = np.array([cert[g][1] for g in gids[idx]])
    seas = np.array([cert[g][2] for g in gids[idx]])
    pmk = np.array([cert[g][3] for g in gids[idx]])
    m_base = SCALE * np.log(p0 / (1 - p0))
    dts = np.array([dates[i] for i in idx])
    loss0 = ll(y, p0)

    from nbapred.eval.splits import Panel, full_report

    NCOL = {}
    r90 = int(np.median([x["r90"] for x in diag["r_rule"]
                         if x["block"] == "PILE13"]))
    r90B = int(np.median([x["r90"] for x in diag["r_rule"]
                          if x["block"] == "JOINT43"]))
    for nm in names:
        if nm.startswith("pca:r") or nm.startswith("pcat:r"):
            NCOL[nm] = int(nm.split("r")[-1]) + 2
        elif nm.startswith("pcaTH:r"):
            NCOL[nm] = int(nm.split("r")[-1]) + 2
        elif nm.startswith("ridge:L") or nm.startswith("ridgeD:L"):
            NCOL[nm] = 15
    NCOL.update({"raw:ALL15": 15, "raw:ALL15+TEAMHOME": 45, "raw:DENSE13": 13,
                 "A:PCA90": r90 + 2, "B:PCA90+TH": r90B + 2,
                 "C:EB_ALL15": 15, "C2:EB_ALL15+TEAMHOME": 45})

    out = {"prereg_sha256": PREREG, "n": int(len(y)), "seasons": list(SEASONS),
           "control_ll": float(loss0.mean()), "mkt_ll": float(ll(y, pmk).mean()),
           "E_pq": float((p0 * (1 - p0)).mean()),
           "tank_floor_env": os.environ.get("TANK_SEASON_FLOOR"),
           "T1_rotation_identity": t1, "T1z_ridge0_identity": t1z,
           "T1b_joint": t1b, "T1b_games_off": n_off,
           "d154_reproduction": rep,
           "r90_PILE13": r90, "r90_JOINT43": r90B,
           "arms": {}, "paired": {}}

    losses = {}
    print(f"\n{'arm':24s} {'m':>4s} {'rms(dm)':>8s} {'delta':>10s} "
          f"{'season-cluster CI':>24s} {'MDE80':>8s} {'noise':>9s} verdict")
    for nm in names:
        d = dm[nm][idx]
        p1 = 1.0 / (1.0 + np.exp(-(m_base + d) / SCALE))
        losses[nm] = ll(y, p1)
        pan = Panel.from_logloss(seas, y, p0, p1, date=dts, label=nm)
        rep_ = full_report(pan, B=NBOOT, seed=SEED)
        cl = rep_["clustering"]["season_cluster_boot"]
        st = dict(ncol=NCOL[nm], rms_dm=float(np.sqrt((d ** 2).mean())),
                  max_abs_dm=float(np.abs(d).max()),
                  delta=rep_["pooled"]["est"], iid_lo=rep_["pooled"]["lo"],
                  iid_hi=rep_["pooled"]["hi"], iid_se=rep_["pooled"]["se"],
                  cl_lo=cl["lo"], cl_hi=cl["hi"], cl_se=cl["se"],
                  t_lo=rep_["clustering"]["season_mean_t"]["lo"],
                  t_hi=rep_["clustering"]["season_mean_t"]["hi"],
                  t_stat=rep_["clustering"]["season_mean_t"].get("t"),
                  icc=rep_["clustering"]["icc_season"],
                  deff=rep_["clustering"].get("design_effect_season"),
                  mde80=rep_["pooled_mde80"],
                  per_season={x["season"]: x["est"] for x in rep_["per_season"]},
                  ro=[f["fold"]["est"] for f in rep_["rolling_origin"]["folds"]],
                  ro_pos=int(sum(f["fold"]["est"] > 0
                                 for f in rep_["rolling_origin"]["folds"])),
                  ro_n=len(rep_["rolling_origin"]["folds"]),
                  loso_test=[f["test_on"]["est"] for f in rep_["loso"]["folds"]],
                  jack_range=rep_["loso"]["jackknife_range"],
                  legacy={k: rep_["legacy"][k]["est"] for k in rep_["legacy"]
                          if isinstance(rep_["legacy"][k], dict)
                          and "est" in rep_["legacy"][k]},
                  block_boot=(rep_.get("block_bootstrap") or {}).get("est"),
                  block_lo=(rep_.get("block_bootstrap") or {}).get("lo"),
                  block_hi=(rep_.get("block_bootstrap") or {}).get("hi"),
                  temporal_deff=rep_.get("temporal_design_effect"),
                  era={e["era"]: e["est"] for e in rep_["era"]["by_era"]}
                  if "by_era" in rep_["era"] else {},
                  I2=rep_["era"]["I2"], era_stable=rep_["era"]["era_stable"],
                  flags=rep_["verdict"]["flags"])
        out["arms"][nm] = st
        sig = "SIG" if (cl["lo"] > 0 or cl["hi"] < 0) else "ns"
        print(f"{nm:24s} {NCOL[nm]:4d} {st['rms_dm']:8.4f} {st['delta']:+10.5f} "
              f"({cl['lo']:+10.5f},{cl['hi']:+10.5f}) {st['mde80']:8.5f} "
              f"{-1e-4*NCOL[nm]:+9.5f} {sig}")

    # ---- the PAIRED question: does compression beat the raw pile? --------
    PAIRS = [("A:PCA90", "raw:ALL15"), ("B:PCA90+TH", "raw:ALL15+TEAMHOME"),
             ("C:EB_ALL15", "raw:ALL15"),
             ("C2:EB_ALL15+TEAMHOME", "raw:ALL15+TEAMHOME"),
             (f"pca:r{r90}", "raw:ALL15"), ("pca:r9", "raw:ALL15"),
             ("pca:r12", "raw:ALL15"), ("C:EB_ALL15", "A:PCA90"),
             ("ridge:L200", "raw:ALL15"), ("ridge:L800", "raw:ALL15"),
             ("ridge:L200", "A:PCA90"), ("pca:r3", "raw:ALL15"),
             ("ridgeD:L200", "raw:ALL15"), ("ridgeD:L800", "raw:ALL15"),
             ("ridgeD:L1600", "raw:ALL15"), ("raw:DENSE13", "raw:ALL15")]
    print(f"\n{'paired (arm - comparator)':44s} {'delta':>10s} "
          f"{'season-cluster CI':>24s} {'MDE80':>8s} verdict")
    for a, c in PAIRS:
        pan = Panel.from_losses(seas, losses[c], losses[a], date=dts,
                                label=f"{a}-{c}")
        rp = full_report(pan, B=NBOOT, seed=SEED)
        cl = rp["clustering"]["season_cluster_boot"]
        out["paired"][f"{a} vs {c}"] = dict(
            delta=rp["pooled"]["est"], cl_lo=cl["lo"], cl_hi=cl["hi"],
            mde80=rp["pooled_mde80"],
            t_lo=rp["clustering"]["season_mean_t"]["lo"],
            t_hi=rp["clustering"]["season_mean_t"]["hi"],
            ro_pos=int(sum(f["fold"]["est"] > 0
                           for f in rp["rolling_origin"]["folds"])),
            ro_n=len(rp["rolling_origin"]["folds"]),
            per_season={x["season"]: x["est"] for x in rp["per_season"]},
            I2=rp["era"]["I2"], flags=rp["verdict"]["flags"])
        sig = "SIG" if (cl["lo"] > 0 or cl["hi"] < 0) else "ns"
        print(f"{a+' - '+c:44s} {rp['pooled']['est']:+10.5f} "
              f"({cl['lo']:+10.5f},{cl['hi']:+10.5f}) "
              f"{rp['pooled_mde80']:8.5f} {sig}")

    out["ridge_edf"] = diag.get("ridge_edf", {})
    out["diag_r_rule"] = diag["r_rule"][:4]
    out["eb_diag_sample"] = diag["eb"][:3]
    if diag["eb"]:
        keys = sorted(diag["eb"][-1]["w_eb"].keys())
        out["eb_mean_w"] = {TERM_NAMES[int(k)]:
                            round(float(np.mean([e["w_eb"][k] for e in diag["eb"]
                                                 if k in e["w_eb"]])), 4)
                            for k in keys}
        out["eb_mean_absT"] = {TERM_NAMES[int(k)]:
                               round(float(np.mean([abs(e["t_extras"][k])
                                                    for e in diag["eb"]
                                                    if k in e["t_extras"]])), 3)
                               for k in keys}
        out["eb_mean_w_dead"] = [round(float(np.mean([e["w_eb_dead"][j]
                                                      for e in diag["eb"]])), 4)
                                 for j in (0, 1)]
        out["eb_mean_absT_dead"] = [round(float(np.mean([abs(e["t_dead"][j])
                                                         for e in diag["eb"]])), 3)
                                    for j in (0, 1)]
        out["w_global_mean"] = round(float(np.mean([e["w_global"]
                                                    for e in diag["eb"]])), 5)

    json.dump(out, open(REPO / "data" / "pca_ladder.json", "w"), indent=1)
    np.savez_compressed(REPO / "data" / "pca_ladder_rows.npz",
                        gid=gids[idx], seas=seas, y=y, p0=p0, pmk=pmk,
                        date=np.array([str(x) for x in dts]),
                        **{("dm_" + k): dm[k][idx] for k in names})
    print("\nwrote data/pca_ladder.json + pca_ladder_rows.npz")
