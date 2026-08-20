#!/usr/bin/env python3
"""D238 — WHEN DO WE WIN? Signal strength, CLV decomposition, bet-time
conditioners, and a PCA regime state — one diagnostic pass, nothing shipped.

Owner: "think of multiple other ways to condition betting on the right game.
do we simply not have good enough signal? very interested in regime gates —
maybe pca. when to determine when market is truly profitable? analyze when we
win and dont."

DISCIPLINE. Every split below is a subset selector, and the register's
manufacturing-capacity result says random subsets of this same book buy +2.54
ROI points from nothing. So: (a) every conditioning variable examined is listed
IN the family; (b) one permutation null is run over the WHOLE family, so the
observed best split is judged against the best split noise can produce across
the same number of looks; (c) nothing here ships, and anything interesting is a
candidate for a prospective 2026-27 pre-registration, not a re-run on these
seasons.

FOUR PARTS.
  A  SIGNAL — the dose-response of P(model side covers) in |stated edge|, on
     ALL games. This is the "do we have signal" question in its powered form.
  B  THE BOOK — when do the 888 bets win: edge size, CLV decomposition
     (diagnostic — CLV is not knowable at bet time), and every bet-time
     conditioner, under the family null.
  C  PCA REGIME — the owner's ask. Eight strictly-prior market-state variables,
     rotated on a pre-2019 lead-in, applied forward, tested against the
     all-games edge endpoint.
  D  POWER — what sample any of this needs, on the ROI endpoint vs the CLV
     endpoint, i.e. when we would actually KNOW the market is profitable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402
from scipy.optimize import minimize_scalar                        # noqa: E402

WIN = 200
FROM = "2019-20"
BE = 100 / 190 * 100        # 52.63... no — break-even at -110 is 110/210
BE = 110 / 210 * 100        # 52.381%


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def clustered_slope(d, xcol, ycol):
    per = []
    for s, sub in d.groupby("season"):
        if len(sub) >= 40 and sub[xcol].std() > 0:
            per.append(np.polyfit(sub[xcol], sub[ycol], 1)[0])
    per = np.array(per)
    k = len(per)
    se = per.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return per.mean(), per.mean() - tc * se, per.mean() + tc * se, k


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f["season"] >= "2017-18"].copy()
    f["game_date"] = pd.to_datetime(f["game_date"])
    f["game_id"] = zf(f["game_id"])
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual", "m_us"])
    f = f.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    f["edge_pts"] = f["m_us"] - f["open_margin"]
    f["side"] = np.sign(f["edge_pts"])                       # +1 home, -1 away
    f["cover_sign"] = np.sign(f["margin_actual"] - f["open_margin"])
    f["covered"] = (f["cover_sign"] == f["side"]).astype(float)
    f["push"] = (f["cover_sign"] == 0)
    f["y"] = (f["margin_actual"] > 0).astype(float)

    g = f[f["season"] >= FROM].copy()
    print(f"frame {len(g):,} games {g.season.min()}..{g.season.max()}, "
          f"pushes {int(g.push.sum())}")

    # ================= A. SIGNAL: dose-response on ALL games ==============
    print("\n" + "=" * 70)
    print("A. THE SIGNAL — P(model side covers) vs |stated edge|, ALL games")
    print("=" * 70)
    gg = g[~g.push]
    q = pd.qcut(gg["edge_pts"].abs(), 10, labels=False, duplicates="drop")
    a_rows = []
    for k in sorted(q.unique()):
        sub = gg[q == k]
        cr = 100 * sub.covered.mean()
        se = 100 * sub.covered.std(ddof=1) / np.sqrt(len(sub))
        a_rows.append(dict(decile=int(k), n=len(sub),
                           edge=float(sub.edge_pts.abs().median()),
                           cover=float(cr), se=float(se)))
        flag = " <- above break-even" if cr - 1.0 * se > BE else ""
        print(f"  D{int(k)+1:2}  n={len(sub):4}  |edge|={sub.edge_pts.abs().median():5.2f} pts"
              f"  cover {cr:5.2f}% +/-{se:4.2f}{flag}")
    m, lo, hi, K = clustered_slope(gg.assign(a=gg.edge_pts.abs()), "a", "covered")
    print(f"  break-even at -110: {BE:.2f}%")
    print(f"  season-clustered slope of P(cover) on |edge|: {100*m:+.3f} pp/pt "
          f"95% CI [{100*lo:+.3f}, {100*hi:+.3f}]  K={K}  "
          f"{'SIGNIFICANT' if lo > 0 else 'ns'}")

    # ================= B. THE BOOK: when do we win ========================
    print("\n" + "=" * 70)
    print("B. THE 888-BET BOOK — when do the bets win (observed 1-book tier)")
    print("=" * 70)
    pb = json.load(open(ROOT / "data" / "wf_perbet_OFFSET.json"))["k=1 raw"]
    b = pd.DataFrame([x for x in pb if x["season"] >= FROM])
    b["game_id"] = zf(b["gid"])
    d = b.merge(f[["game_id", "edge_pts", "side", "open_margin", "close_margin",
                   "margin_actual", "covered", "push", "game_date"]],
                on="game_id", how="left", validate="one_to_one")
    ok = d.edge_pts.notna().mean()
    assert ok > 0.99, f"join {ok:.1%}"
    cap = pd.read_csv(ROOT / "data" / "capstone_2019_26.csv")
    cap["game_id"] = zf(cap["game_id"])
    d = d.merge(cap[["game_id", "eo_home", "eo_away"]], on="game_id", how="left")
    d["won"] = d.ev > 0.001
    d["days_in"] = (d.game_date
                    - d.groupby("season")["game_date"].transform("min")).dt.days
    d["out_diff_our_side"] = d.side * (d.eo_home - d.eo_away)
    graded = d[~d.push]
    print(f"bets {len(d)}, graded {len(graded)}, cover "
          f"{100*graded.won.mean():.2f}%, ROI {100*d.ev.mean():+.2f}%")
    print(f"median |edge| in the book {d.edge_pts.abs().median():.2f} pts "
          f"(slate median {gg.edge_pts.abs().median():.2f})")

    # CLV decomposition — DIAGNOSTIC (close is not knowable at bet time)
    d["clv_pts"] = d.side * (d.close_margin - d.open_margin)
    print(f"\n--- CLV decomposition (diagnostic: not knowable at bet time) ---")
    print(f"  line moved TOWARD our side on {100*(d.clv_pts>0).mean():.1f}% of "
          f"bets, away on {100*(d.clv_pts<0).mean():.1f}%, "
          f"flat {100*(d.clv_pts==0).mean():.1f}%  (mean {d.clv_pts.mean():+.3f} pts)")
    for lab, m_ in (("CLV > 0 (market later agreed) ", d.clv_pts > 0),
                    ("CLV = 0                       ", d.clv_pts == 0),
                    ("CLV < 0 (market later opposed)", d.clv_pts < 0)):
        sub = d[m_ & ~d.push]
        if len(sub):
            print(f"  {lab}  n={len(sub):4}  cover {100*sub.won.mean():5.2f}%  "
                  f"ROI {100*d[m_].ev.mean():+7.2f}%")
    per = d.groupby("season").clv_pts.mean()
    print(f"  per-season mean CLV: " +
          "  ".join(f"{s[-5:]} {v:+.2f}" for s, v in per.items()))

    # bet-time conditioners, all in ONE family
    print(f"\n--- bet-time conditioners (the FAMILY; one null covers all) ---")
    fam = {}
    q5 = pd.qcut(d.edge_pts.abs(), 5, labels=False, duplicates="drop")
    fam["edge_quintile"] = q5
    fam["side_home"] = (d.side > 0).astype(int)
    fam["on_favorite"] = (d.side == np.sign(d.open_margin)).astype(int)
    fam["big_line"] = (d.open_margin.abs()
                       > d.open_margin.abs().median()).astype(int)
    fam["outs_our_side"] = np.sign(d.out_diff_our_side.fillna(0)
                                   .round(1)).astype(int)
    fam["season_third"] = pd.cut(d.days_in, [-1, 60, 120, 999],
                                 labels=False)
    obs_spread = {}
    for name, lab in fam.items():
        rows = []
        for v in sorted(pd.Series(lab).dropna().unique()):
            sub = d[lab == v]
            gsub = sub[~sub.push]
            rows.append((v, len(sub), 100 * gsub.won.mean() if len(gsub) else np.nan,
                         100 * sub.ev.mean()))
        spread = max(r[3] for r in rows) - min(r[3] for r in rows)
        obs_spread[name] = spread
        cells = "  ".join(f"[{r[0]}] n={r[1]} {r[2]:.1f}%/{r[3]:+.1f}%"
                          for r in rows)
        print(f"  {name:14} {cells}   spread {spread:.1f}")
    # family permutation null
    rng = np.random.default_rng(238)
    null = []
    ev = d.ev.to_numpy(float)
    labs = {k: pd.Series(v).to_numpy() for k, v in fam.items()}
    for _ in range(2000):
        sh = ev.copy(); rng.shuffle(sh)
        mx = 0.0
        for lab in labs.values():
            gs = [100 * sh[lab == v].mean()
                  for v in np.unique(lab[~pd.isna(lab)])]
            mx = max(mx, max(gs) - min(gs))
        null.append(mx)
    omax = max(obs_spread.values())
    p = float(np.mean(np.array(null) >= omax))
    print(f"  FAMILY NULL: observed max spread {omax:.1f} ROI pts; null median "
          f"{np.median(null):.1f}, 95th {np.percentile(null,95):.1f}; p = {p:.3f}")

    # ================= C. PCA REGIME ======================================
    print("\n" + "=" * 70)
    print("C. PCA REGIME STATE — 8 strictly-prior variables, lead-in rotation")
    print("=" * 70)
    # per-date market state from strictly prior games
    f2 = f.copy()
    p696 = 1 / (1 + np.exp(-f2.open_margin / 6.96))
    f2["ll_open_fix"] = nll(np.clip(p696, 1e-9, 1 - 1e-9), f2.y)
    f2["move"] = (f2.close_margin - f2.open_margin).abs()
    f2["fav_cov"] = (f2.cover_sign == np.sign(f2.open_margin)).astype(float)
    f2["home_cov"] = (f2.cover_sign > 0).astype(float)
    per_date = f2.groupby("game_date").agg(
        ll=("ll_open_fix", "mean"), ae=("move", "size"),
        move=("move", "mean"), fav=("fav_cov", "mean"),
        home=("home_cov", "mean"), absline=("open_margin",
                                            lambda s: s.abs().mean()),
        tot=("open_total", "mean"),
        err=("margin_actual", "size")).reset_index()
    ae2 = f2.assign(e=(f2.margin_actual - f2.open_margin).abs()) \
            .groupby("game_date")["e"].mean().rename("aerr").reset_index()
    per_date = per_date.merge(ae2, on="game_date").sort_values("game_date")
    VARS = ["ll", "aerr", "move", "fav", "home", "absline", "tot"]
    bufs = {v: [] for v in VARS}
    rows = []
    for r in per_date.itertuples():
        rows.append([getattr(r, "game_date")] +
                    [np.mean(bufs[v][-WIN:]) if len(bufs[v]) >= 50 else np.nan
                     for v in VARS])
        for v in VARS:
            bufs[v].append(getattr(r, v))
    st = pd.DataFrame(rows, columns=["game_date"] + [f"tr_{v}" for v in VARS])
    dd = f.merge(st, on="game_date").dropna(subset=[f"tr_{v}" for v in VARS])
    dd["days_in"] = (dd.game_date -
                     dd.groupby("season")["game_date"].transform("min")).dt.days
    COLS = [f"tr_{v}" for v in VARS] + ["days_in"]
    lead = dd[dd.season < FROM]
    test = dd[dd.season >= FROM].copy()
    mu, sd = lead[COLS].mean(), lead[COLS].std(ddof=1)
    Z = ((lead[COLS] - mu) / sd).to_numpy(float)
    C = np.cov(Z.T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    print("  eigenvalues:", np.round(w, 2).tolist())
    print("  variance explained by PC1-3: "
          f"{100*w[:3].sum()/w.sum():.0f}%")
    for i in range(3):
        tops = sorted(zip(COLS, V[:, i]), key=lambda t: -abs(t[1]))[:4]
        print(f"  PC{i+1} loadings: " +
              ", ".join(f"{c} {v:+.2f}" for c, v in tops))
    Zt = ((test[COLS] - mu) / sd).to_numpy(float)
    for i in range(3):
        test[f"pc{i+1}"] = Zt @ V[:, i]

    # edge endpoint per game (walk-forward scales, as D237b)
    seasons = sorted(test.season.unique())
    allf = dd.copy()
    for col, out in (("open_margin", "llo"), ("m_us", "llf")):
        allf[out] = np.nan
        ss = sorted(allf.season.unique())
        for i, s in enumerate(ss):
            te = (allf.season == s).to_numpy()
            tr = allf.season.isin(ss[:i]).to_numpy() if i else te
            sc = fit_scale(allf.loc[tr, col].to_numpy(float),
                           allf.loc[tr, "y"].to_numpy(float))
            allf.loc[te, out] = nll(
                1 / (1 + np.exp(-allf.loc[te, col].to_numpy(float) / sc)),
                allf.loc[te, "y"].to_numpy(float))
    test = test.merge(allf[["game_id", "llo", "llf"]], on="game_id")
    test["edge_ll"] = test.llo - test.llf
    for i in range(3):
        c = f"pc{i+1}"
        qq = pd.qcut(test[c], 4, labels=False, duplicates="drop")
        bs = [test[qq == k].edge_ll.mean() for k in sorted(qq.unique())]
        m, lo, hi, K = clustered_slope(test, c, "edge_ll")
        print(f"  {c}: edge by quartile "
              + " ".join(f"{100*v:+.2f}" for v in bs)
              + f" (x1e-2)  slope {m:+.5f} CI [{lo:+.5f},{hi:+.5f}] "
              f"{'SIG' if lo > 0 or hi < 0 else 'ns'}")

    # ================= D. POWER ==========================================
    print("\n" + "=" * 70)
    print("D. POWER — when would we KNOW")
    print("=" * 70)
    sd_ev = d.ev.std(ddof=1)
    n_yr = len(d) / d.season.nunique()
    print(f"  per-bet return sd {sd_ev:.3f}, ~{n_yr:.0f} bets/season at 1 book")
    for eff in (0.05, 0.10):
        n = 2 * (2.8 * sd_ev / eff) ** 2
        print(f"  to detect a subgroup {100*eff:.0f} ROI pts better than the "
              f"rest: n~{n:,.0f}/group = {2*n/n_yr:.0f} seasons of betting")
    n_roi = (2.8 * sd_ev / 0.052) ** 2
    print(f"  to distinguish the pooled +5.2% ROI from zero: n~{n_roi:,.0f} "
          f"bets = {n_roi/n_yr:.0f} seasons")
    sd_clv = d.clv_pts.std(ddof=1)
    mean_clv = d.clv_pts.mean()
    n_clv = (2.8 * sd_clv / max(mean_clv, 1e-9)) ** 2
    print(f"  CLV endpoint: mean {mean_clv:+.3f} pts, sd {sd_clv:.2f} -> "
          f"n~{n_clv:,.0f} bets = {n_clv/n_yr:.1f} seasons to confirm CLV > 0")

    json.dump({"A_dose": a_rows, "B_family_p": p, "B_obs_spread": obs_spread,
               "C_eigen": w.tolist(), "clv_mean": float(mean_clv)},
              open(ROOT / "data" / "d238_autopsy.json", "w"), default=float)
    print("\nwrote data/d238_autopsy.json")


if __name__ == "__main__":
    main()
