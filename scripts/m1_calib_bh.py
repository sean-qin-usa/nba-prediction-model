"""M1 — (a) CALIBRATION VETO BATTERY (COMPLEXITY.md gate 2) and
        (b) BENJAMINI-HOCHBERG q=0.10 over the running gate family.

READ-ONLY except data/m1_calib_bh.json + data/logs/m1_calib_bh.log.
Scores the on-disk D138 artifact data/nsport_joint_pergame.csv (p_ctl vs
p_defonly), the same 6,148 games D138 scored.
"""
import sys
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.eval.metrics import log_loss, brier, ece, reliability_table
from nbapred.eval import splits as S

ART = ROOT / "data" / "nsport_joint_pergame.csv"
FAM = ROOT / "data" / "bh_family.csv"
OUT = ROOT / "data" / "m1_calib_bh.json"

SEED = 20260801
B = 2000


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def logistic_link(y, p, eps=1e-12):
    """Recalibration link test: fit y ~ sigmoid(a + b*logit(p)).
    Perfect calibration is (a, b) = (0, 1).  Newton-Raphson, 60 iters."""
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    X = np.c_[np.ones(len(lp)), lp]
    beta = np.array([0.0, 1.0])
    for _ in range(60):
        mu = 1.0 / (1.0 + np.exp(-X @ beta))
        W = mu * (1 - mu) + 1e-12
        g = X.T @ (y - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = 1.0 / (1.0 + np.exp(-X @ beta))
    W = mu * (1 - mu) + 1e-12
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    se = np.sqrt(np.diag(cov))
    return dict(a=float(beta[0]), b=float(beta[1]),
                se_a=float(se[0]), se_b=float(se[1]),
                z_a=float(beta[0] / se[0]), z_b=float((beta[1] - 1.0) / se[1]))


def hosmer_lemeshow(y, p, g=10):
    """HL chi-square on g equal-count deciles of p; dof = g-2."""
    order = np.argsort(p)
    chunks = np.array_split(order, g)
    stat = 0.0
    for c in chunks:
        o1, e1 = float(y[c].sum()), float(p[c].sum())
        n = len(c)
        e0 = n - e1
        if e1 > 0:
            stat += (o1 - e1) ** 2 / e1
        if e0 > 0:
            stat += ((n - o1) - e0) ** 2 / e0
    dof = g - 2
    return dict(chi2=float(stat), dof=dof, p=float(_chi2_sf(stat, dof)))


def _chi2_sf(x, k):
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        m = k // 2
        t = math.exp(-x / 2)
        s = t
        for i in range(1, m):
            t *= (x / 2) / i
            s += t
        return min(1.0, s)
    z = math.sqrt(x)
    s = 2 * _norm_sf(z)
    t = math.sqrt(2 / math.pi) * z * math.exp(-x / 2)
    for i in range(1, (k - 1) // 2 + 1):
        s += t
        t *= x / (2 * i + 1)
    return min(1.0, max(0.0, s))


def calib_block(y, p, label):
    return dict(label=label, n=int(len(y)),
                log_loss=log_loss(y, p), brier=brier(y, p),
                ece10=ece(y, p, 10), ece20=ece(y, p, 20),
                mean_p=float(np.mean(p)), base_rate=float(np.mean(y)),
                bias=float(np.mean(p) - np.mean(y)),
                link=logistic_link(y, p), hl=hosmer_lemeshow(y, p),
                reliability=reliability_table(y, p, 10))


def main():
    df = pd.read_csv(ART, dtype={"game_id": str})
    y = df["y"].values.astype(float)
    pc = df["p_ctl"].values.astype(float)
    pt = df["p_defonly"].values.astype(float)
    seas = df["season"].values.astype(str)

    res = {}

    # ------------------------------------------------------------------
    # (a) CALIBRATION VETO BATTERY
    # ------------------------------------------------------------------
    print("=" * 74)
    print("(a) CALIBRATION VETO BATTERY (COMPLEXITY.md gate 2)")
    print("=" * 74)
    ctl = calib_block(y, pc, "control (D132)")
    trt = calib_block(y, pt, "M1 defonly")
    res["pooled"] = dict(control=ctl, m1=trt)
    for blk in (ctl, trt):
        print(f"{blk['label']:16s} LL {blk['log_loss']:.5f}  Brier {blk['brier']:.5f}  "
              f"ECE10 {blk['ece10']:.5f}  ECE20 {blk['ece20']:.5f}  "
              f"bias {blk['bias']:+.5f}")
        L = blk["link"]
        print(f"{'':16s} link a={L['a']:+.4f}(z={L['z_a']:+.2f}) "
              f"b={L['b']:.4f}(z_b-1={L['z_b']:+.2f})  "
              f"HL chi2={blk['hl']['chi2']:.2f} dof={blk['hl']['dof']} "
              f"p={blk['hl']['p']:.3f}")
    d_ece10 = trt["ece10"] - ctl["ece10"]
    d_ece20 = trt["ece20"] - ctl["ece20"]
    d_brier = trt["brier"] - ctl["brier"]
    print(f"\nDELTA (M1 - control): dECE10 {d_ece10:+.6f}  dECE20 {d_ece20:+.6f}  "
          f"dBrier {d_brier:+.7f}   (negative = M1 better)")

    # bootstrap the ECE delta (iid and season-clustered)
    rng = np.random.default_rng(SEED)
    n = len(y)
    keys = sorted(set(seas.tolist()))
    idx_by = {k: np.where(seas == k)[0] for k in keys}
    de10, de20 = [], []
    de10c = []
    for _ in range(B):
        ii = rng.integers(0, n, n)
        de10.append(ece(y[ii], pt[ii], 10) - ece(y[ii], pc[ii], 10))
        de20.append(ece(y[ii], pt[ii], 20) - ece(y[ii], pc[ii], 20))
        pick = rng.integers(0, len(keys), len(keys))
        jj = np.concatenate([idx_by[keys[k]] for k in pick])
        de10c.append(ece(y[jj], pt[jj], 10) - ece(y[jj], pc[jj], 10))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    res["delta"] = dict(
        d_ece10=d_ece10, d_ece10_ci_iid=ci(de10),
        d_ece10_ci_seasoncluster=ci(de10c),
        d_ece20=d_ece20, d_ece20_ci_iid=ci(de20), d_brier=d_brier)
    print(f"  dECE10 CI iid            ({ci(de10)[0]:+.6f},{ci(de10)[1]:+.6f})")
    print(f"  dECE10 CI season-cluster ({ci(de10c)[0]:+.6f},{ci(de10c)[1]:+.6f})")
    print(f"  dECE20 CI iid            ({ci(de20)[0]:+.6f},{ci(de20)[1]:+.6f})")

    # per-season
    print("\nPER SEASON (ECE10 / link b / bias):")
    ps = {}
    for s in keys:
        m = seas == s
        c, t = calib_block(y[m], pc[m], "ctl"), calib_block(y[m], pt[m], "m1")
        ps[s] = dict(control=c, m1=t)
        print(f"  {s}  ctl ECE {c['ece10']:.5f} b {c['link']['b']:.3f} "
              f"bias {c['bias']:+.5f}   |   M1 ECE {t['ece10']:.5f} "
              f"b {t['link']['b']:.3f} bias {t['bias']:+.5f}   "
              f"dECE {t['ece10']-c['ece10']:+.6f}")
    res["per_season"] = ps
    worse = sum(1 for s in keys
                if ps[s]["m1"]["ece10"] > ps[s]["control"]["ece10"])
    print(f"  seasons where M1's ECE10 is WORSE: {worse}/{len(keys)}")

    # reliability table, pooled
    print("\nRELIABILITY (10 bins, pooled): conf -> emp (n)")
    for (cf, em, cnt), (cf2, em2, cnt2) in zip(ctl["reliability"], trt["reliability"]):
        print(f"  ctl {cf:.3f}->{em:.3f} (n={cnt:4d})    "
              f"M1 {cf2:.3f}->{em2:.3f} (n={cnt2:4d})")

    veto_fail = (ci(de10)[0] > 0) or (ci(de20)[0] > 0)
    res["veto"] = dict(fails=bool(veto_fail),
                       reason=("M1's ECE is significantly WORSE" if veto_fail
                               else "no significant calibration degradation"))
    print(f"\nCALIBRATION VETO: {'FAIL' if veto_fail else 'PASS'} — "
          f"{res['veto']['reason']}")

    # ------------------------------------------------------------------
    # (b) BENJAMINI-HOCHBERG q = 0.10
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("(b) BENJAMINI-HOCHBERG q = 0.10 OVER THE RUNNING GATE FAMILY")
    print("=" * 74)
    fam = pd.read_csv(FAM)
    bt = json.load(open(ROOT / "data" / "m1_v3_battery.json"))
    cl = bt["m1"]["clustering"]
    p_iid = float(cl["iid"]["p_wrongside"])
    t = cl["season_mean_t"]
    tstat = t["unweighted_mean"] / t["se"]
    # one-sided p from the K-1 dof cluster-mean t (the SS9.3 conservative bound)
    p_t = _t_sf(tstat, t["dof"])
    print(f"M1 p candidates: iid bootstrap p_wrongside = {p_iid:.4f};  "
          f"season-cluster bootstrap p_wrongside = "
          f"{cl['season_cluster_boot']['p_wrongside']:.4f} (K=5, not credible); "
          f"cluster-mean t = {tstat:.3f} at {t['dof']} dof -> one-sided "
          f"p = {p_t:.4f}")

    bh = {}
    for tag, pm1 in (("iid_0.0155", p_iid), ("clusterT_conservative", p_t)):
        names = fam["name"].tolist() + ["M1 3P-luck defense-only SOLO (this gate, 5-season)"]
        ps_ = fam["p_onesided"].tolist() + [pm1]
        ps_ = [1.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
               for x in ps_]
        K = len(ps_)
        order = np.argsort(ps_)
        sp = np.array(ps_)[order]
        thr = (np.arange(1, K + 1) * 0.10) / K
        ok = np.where(sp <= thr)[0]
        imax = int(ok[-1]) + 1 if len(ok) else 0
        rank_m1 = int(np.where(order == K - 1)[0][0]) + 1
        rejected = imax
        passes = rank_m1 <= imax
        bh[tag] = dict(K=K, p_m1=pm1, rank_m1=rank_m1,
                       bh_threshold_at_rank=float(thr[rank_m1 - 1]),
                       max_rejected_rank=imax, n_rejected=rejected,
                       m1_survives=bool(passes),
                       largest_p_rejected=float(sp[imax - 1]) if imax else None)
        print(f"\n  [{tag}] K={K} (106 family + this gate), p_M1={pm1:.4f}")
        print(f"    M1 rank {rank_m1}/{K}, BH threshold at that rank = "
              f"{thr[rank_m1-1]:.5f}")
        print(f"    BH step-up rejects the {imax} smallest p "
              f"(largest rejected p = {sp[imax-1]:.5f})" if imax else
              "    BH rejects NOTHING")
        print(f"    ==> M1 {'SURVIVES' if passes else 'FAILS'} BH q=0.10")
    res["bh"] = bh
    res["bh_note"] = ("28 of the 106 enumerated family members have no published "
                      "CI/z/p; they are entered at p=1.0 — conservative for them, "
                      "and they still count in the K denominator, which is the "
                      "correct conservative handling.")

    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("\nwrote", OUT)


def _t_sf(t, dof):
    """One-sided upper-tail P(T_dof > t), exact for integer dof."""
    if dof <= 0:
        return float("nan")
    x = dof / (dof + t * t)
    ib = _betainc(dof / 2.0, 0.5, x)
    p = 0.5 * ib
    return p if t > 0 else 1.0 - p


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x) / a
    return 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * _cf(b, a, 1 - x) / b


def _cf(a, b, x, itmax=300, eps=1e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


if __name__ == "__main__":
    main()
