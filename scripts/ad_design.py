#!/usr/bin/env python3
"""AVAILABILITY-DEPTH DESIGN DIAGNOSTICS — run and DISCLOSED before the
pre-registration; the ENDPOINT (points CRPS) is never touched here.

Three questions, all at the MINUTES level (the props bottleneck, D12/D133):

  A. NATIONAL TV, CONDITIONAL ON PLAYING. The mechanism measurement showed a
     -0.022 within-player-season core-DNP suppression. But the props eval
     universe conditions on `seconds>=720`, i.e. on the player having PLAYED —
     so a DNP-hazard term can only reach points CRPS if it ALSO shifts minutes
     (or rates) conditional on playing. Measure that shift.

  B. GAMEROTATION ROLE STATE. `player_rates_from_stats` projects minutes from
     an EWMA(hl=10) of PLAYED minutes and is structurally blind to role. The
     rotation cache gives a starter flag (`first_in<=0`), stint count and
     time-weighted on-floor usage that the box score does not carry. Does a
     trailing role state predict the residual (proj_min - realized) that D133's
     gp-ramp leaves behind?

  C. GARBAGE-TIME / LEVERAGE. Per-stint PT_DIFF + the running margin from
     `lineup_stints` give each player's share of minutes played in a decided
     game. Does trailing garbage share explain residual minutes?

Writes data/ad_design.json. DB read_only=True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine.props import minutes_ramp

HL = 10.0


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def cluster_boot_mean(d, g, B=2000, seed=20260801):
    uniq, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (B, len(uniq)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(m.std(ddof=1))


def main():
    out = {}
    con = connect(read_only=True)

    # ------------------------------------------------------------- corpus
    df = con.execute("""
        SELECT s.player_id, s.team_id, s.game_id, g.season, g.game_date,
               s.seconds/60.0 AS mins, s.pts, s.fga, s.fta, s.tov
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g
          USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    df["game_id"] = df["game_id"].astype(str)
    df["ord"] = df["game_date"].astype("datetime64[ns]").values.astype(
        "datetime64[D]").astype(int)

    rot = pd.read_csv(ROOT / "data" / "ad_rotation_pg.csv.gz",
                      dtype={"game_id": str})
    rot = rot[["game_id", "team_id", "player_id", "n_stints", "rot_sec",
               "first_in", "usg_w", "ptdiff_sum", "is_starter"]]
    tv = pd.read_csv(ROOT / "data" / "ad_natl_tv.csv", dtype={"game_id": str})

    d = df.merge(rot, on=["game_id", "team_id", "player_id"], how="left")
    d = d.merge(tv[["game_id", "is_natl_tv"]], on="game_id", how="left")
    d["has_rot"] = d["n_stints"].notna().astype(int)
    print(f"corpus {len(d)} played rows; rotation coverage "
          f"{d.has_rot.mean():.4f}; natl-tv coverage {d.is_natl_tv.notna().mean():.4f}")
    out["corpus"] = dict(n=len(d), rot_cov=float(d.has_rot.mean()),
                         tv_cov=float(d.is_natl_tv.notna().mean()))

    # ------------------------------- production-identical trailing features
    recs = []
    for pid, sub in d.groupby("player_id", sort=False):
        sub = sub.sort_values("ord")
        mins = sub["mins"].to_numpy(float)
        seas = sub["season"].to_numpy(object)
        star = sub["is_starter"].to_numpy(float)      # NaN where uncovered
        nst = sub["n_stints"].to_numpy(float)
        usg = sub["usg_w"].to_numpy(float)
        pdf = sub["ptdiff_sum"].to_numpy(float)
        gid = sub["game_id"].to_numpy(object)
        tvf = sub["is_natl_tv"].to_numpy(float)
        ordv = sub["ord"].to_numpy()
        for i in range(len(sub)):
            if i < 8:                                  # n_games>=8 (props gate)
                continue
            w = 0.5 ** (np.arange(i)[::-1] / HL)
            proj_raw = wmean(mins[:i], w)
            gp = int((seas[:i] == seas[i]).sum())
            proj = max(proj_raw - minutes_ramp(gp), 0.0)   # SHIPPED production
            if proj < 20:
                continue
            # trailing rotation role state (STRICTLY prior games only)
            hs = star[:i]
            m5 = ~np.isnan(hs)
            cov = int(m5.sum())
            last5 = hs[m5][-5:] if cov else np.array([])
            last10 = hs[m5][-10:] if cov else np.array([])
            nst5 = nst[:i][~np.isnan(nst[:i])][-5:]
            usg5 = usg[:i][~np.isnan(usg[:i])][-5:]
            recs.append(dict(
                player_id=int(pid), game_id=gid[i], season=seas[i],
                ord=int(ordv[i]), gp=gp, proj=proj, y=mins[i],
                resid=proj - mins[i],
                is_natl_tv=tvf[i],
                rot_cov=cov,
                sr5=float(last5.mean()) if len(last5) else np.nan,
                sr10=float(last10.mean()) if len(last10) else np.nan,
                sr_last=float(hs[m5][-1]) if cov else np.nan,
                nst5=float(nst5.mean()) if len(nst5) else np.nan,
                usg5=float(usg5.mean()) if len(usg5) else np.nan,
                y_starter=star[i], y_nst=nst[i], y_usg=usg[i], y_ptdiff=pdf[i],
            ))
    r = pd.DataFrame(recs)
    print(f"eval-universe rows {len(r)}  players {r.player_id.nunique()}")
    out["universe"] = dict(n=len(r), players=int(r.player_id.nunique()))

    # ------------------------------------------------ A. NATIONAL TV MINUTES
    a = r[r.is_natl_tv.notna()].copy()
    a["ps"] = a.player_id.astype(str) + "_" + a.season
    print(f"\n=== A. NATIONAL TV | PLAYED (n={len(a)}) ===")
    blk = {"n": int(len(a))}
    for col, lab in (("y", "realized minutes"), ("resid", "proj-realized resid")):
        gm = a.groupby("is_natl_tv")[col].mean()
        # within player-season
        dm = a[col] - a.groupby("ps")[col].transform("mean")
        xd = a.is_natl_tv - a.groupby("ps")["is_natl_tv"].transform("mean")
        beta = float(np.dot(xd, dm) / np.dot(xd, xd))
        # cluster bootstrap by player on the FE beta
        uniq, inv = np.unique(a.player_id.to_numpy(), return_inverse=True)
        idxby = [np.where(inv == i)[0] for i in range(len(uniq))]
        rng = np.random.default_rng(20260801)
        bs = np.empty(1000)
        xdv, dmv = xd.to_numpy(), dm.to_numpy()
        for b in range(1000):
            pick = rng.integers(0, len(uniq), len(uniq))
            sel = np.concatenate([idxby[i] for i in pick])
            den = np.dot(xdv[sel], xdv[sel])
            bs[b] = np.dot(xdv[sel], dmv[sel]) / den if den > 0 else np.nan
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        blk[col] = dict(local=float(gm.get(0.0, np.nan)), natl=float(gm.get(1.0, np.nan)),
                        fe_beta=beta, lo=float(lo), hi=float(hi),
                        sig="SIG" if (lo > 0 or hi < 0) else "ns")
        print(f"  {lab:22s} local {blk[col]['local']:.4f}  natl {blk[col]['natl']:.4f}"
              f"  FE beta {beta:+.4f} CI[{lo:+.4f},{hi:+.4f}] {blk[col]['sig']}")
    # rate channel: points per minute
    out["A_natl_tv_minutes"] = blk

    # ------------------------------------------- B. ROTATION ROLE STATE
    b = r[r.rot_cov >= 3].copy()
    print(f"\n=== B. ROTATION ROLE STATE (rows with >=3 covered prior games: "
          f"{len(b)} = {len(b)/len(r):.3f} of universe) ===")
    blkB = {"n": int(len(b)), "share": float(len(b) / len(r))}
    print(f"  mean resid overall {b.resid.mean():+.4f}")
    for cut, lab in ((0.2, "sr5<=0.2 (bench)"), (0.8, "sr5>=0.8 (starter)")):
        pass
    bins = [(-0.01, 0.001, "sr5=0 pure bench"), (0.001, 0.399, "sr5 (0,0.4)"),
            (0.399, 0.799, "sr5 [0.4,0.8)"), (0.799, 1.01, "sr5 >=0.8")]
    tab = []
    for lo_, hi_, lab in bins:
        m = (b.sr5 > lo_) & (b.sr5 <= hi_)
        if m.sum() < 50:
            continue
        pt, cl, ch, se = cluster_boot_mean(b.resid[m].to_numpy(), b.player_id[m].to_numpy())
        tab.append(dict(bucket=lab, n=int(m.sum()), resid=pt, lo=cl, hi=ch))
        print(f"  {lab:22s} n={m.sum():6d} resid {pt:+.4f} CI[{cl:+.4f},{ch:+.4f}]")
    blkB["by_sr5"] = tab
    # ROLE CHANGE: last game's starter status vs the 5-game rate
    b["delta_role"] = b.sr_last - b.sr5
    tabc = []
    for lo_, hi_, lab in ((-1.01, -0.35, "demoted (d<=-0.35)"),
                          (-0.35, 0.35, "stable"),
                          (0.35, 1.01, "promoted (d>=0.35)")):
        m = (b.delta_role > lo_) & (b.delta_role <= hi_)
        if m.sum() < 50:
            continue
        pt, cl, ch, se = cluster_boot_mean(b.resid[m].to_numpy(), b.player_id[m].to_numpy())
        tabc.append(dict(bucket=lab, n=int(m.sum()), resid=pt, lo=cl, hi=ch))
        print(f"  {lab:22s} n={m.sum():6d} resid {pt:+.4f} CI[{cl:+.4f},{ch:+.4f}]")
    blkB["by_role_change"] = tabc
    # incremental regression: resid ~ sr5 + nst5 + gp-dummies
    m = b.sr5.notna() & b.nst5.notna()
    bb = b[m]
    X0 = np.column_stack([np.ones(len(bb))])
    X1 = np.column_stack([bb.sr5, bb.nst5, bb.sr_last - bb.sr5, np.ones(len(bb))])
    y = bb.resid.to_numpy()
    r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    coef = np.linalg.lstsq(X1, y, rcond=None)[0]
    blkB["reg"] = dict(rmse0=float(np.sqrt((r0**2).mean())),
                       rmse1=float(np.sqrt((r1**2).mean())),
                       coef_sr5=float(coef[0]), coef_nst5=float(coef[1]),
                       coef_drole=float(coef[2]), intercept=float(coef[3]),
                       n=int(len(bb)))
    print(f"  regression n={len(bb)}: rmse {blkB['reg']['rmse0']:.4f} -> "
          f"{blkB['reg']['rmse1']:.4f}  (sr5 {coef[0]:+.3f}, nst5 {coef[1]:+.3f}, "
          f"drole {coef[2]:+.3f})")
    out["B_rotation_role"] = blkB

    # ------------------------------------------- C. GARBAGE TIME / LEVERAGE
    print("\n=== C. GARBAGE-TIME SHARE ===")
    st = pd.read_csv(ROOT / "data" / "ad_rotation.csv.gz", dtype={"game_id": str})
    ls = con.execute("""
        SELECT game_id, stint_idx, t_start, t_end, margin, home_team_id
        FROM lineup_stints ORDER BY game_id, stint_idx
    """).fetchdf()
    ls["game_id"] = ls["game_id"].astype(str)
    ls["cum"] = ls.groupby("game_id")["margin"].cumsum()      # running home margin
    # garbage = |running home margin| >= 20 AND t >= 2160 s (start of Q4)
    ls["gt"] = ((ls["cum"].abs() >= 20) & (ls["t_end"] >= 2160)).astype(float)
    gseg = {gid: (g["t_start"].to_numpy(), g["t_end"].to_numpy(), g["gt"].to_numpy())
            for gid, g in ls.groupby("game_id")}
    st = st[st.game_id.isin(gseg)].copy()
    gsec = np.zeros(len(st))
    ts = (st.in_t.to_numpy() / 10.0)
    te = (st.out_t.to_numpy() / 10.0)
    gids = st.game_id.to_numpy()
    for i in range(len(st)):
        seg = gseg.get(gids[i])
        if seg is None:
            continue
        a0, a1, g = seg
        ov = np.clip(np.minimum(te[i], a1) - np.maximum(ts[i], a0), 0, None)
        gsec[i] = float(np.dot(ov, g))
    st["gsec"] = gsec
    gpg = st.groupby(["game_id", "team_id", "player_id"]).agg(
        gsec=("gsec", "sum"), tsec=("stint_sec", "sum")).reset_index()
    gpg["gshare"] = np.where(gpg.tsec > 0, gpg.gsec / gpg.tsec, 0.0)
    print(f"  player-games with garbage overlap computed: {len(gpg)}; "
          f"mean garbage share {gpg.gshare.mean():.4f}, "
          f"share with >20% garbage {float((gpg.gshare>0.2).mean()):.4f}")
    out["C_garbage"] = dict(n=int(len(gpg)), mean_share=float(gpg.gshare.mean()),
                            p_gt20=float((gpg.gshare > 0.2).mean()))
    gpg.to_csv(ROOT / "data" / "ad_garbage_pg.csv.gz", index=False, compression="gzip")

    con.close()
    r.to_csv(ROOT / "data" / "ad_design_rows.csv.gz", index=False, compression="gzip")
    (ROOT / "data" / "ad_design.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nAD_DESIGN_DONE")


if __name__ == "__main__":
    main()
