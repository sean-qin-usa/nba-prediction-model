#!/usr/bin/env python3
"""D240 — PARTICIPATION-V2. Prereg sha256 30c8dbd2...

THE DEFECT. `d200_participation.py:124` sets the label to

    y_out = (status_today == "Out")

so the shipped model predicts whether the NBA will publish an administrative
"Out" designation. The composition leg then spends `1 - p_out` as an EXPECTED
PARTICIPATION weight. Those are different questions, and the gap is whatever
kind of non-appearance never earns a same-day Out label: G-League assignments
(28,881 reason rows), coach's DNPs, late scratches.

This realigns the label to ACTUAL NON-APPEARANCE and adds the roster-state
features the administrative target had no reason to carry — chiefly the parsed
reason category and days since the player last appeared.

Everything is as-of-open: the last status published STRICTLY BEFORE game day
(the D199 carry-forward rule), trailing minutes over prior games only, and
appearance history strictly prior. Walk-forward: fitted on seasons before the
one being scored.

Part 1 is the label-alignment AUDIT (T1) and stands on its own regardless of
whether the model wins. Part 2 is the head-to-head on the participation label.
The downstream full-stack gate is D240b.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

from d200_participation import logistic_fit                       # noqa: E402
from nbapred.db import connect                                    # noqa: E402
from nbapred import teams as T                                    # noqa: E402

FROM = "2019-20"
ROT_DAYS = 21
ROT_MIN = 8.0
STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]


def reason_cat(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "none"
    t = s.lower()
    if "g league" in t or "g-league" in t:
        return "gleague"
    if "protocol" in t:
        return "protocol"
    if "not with team" in t or "personal" in t or "rest" in t:
        return "rest_personal"
    if "surgery" in t:
        return "surgery"
    if "illness" in t and "injury/illness" not in t:
        return "illness"
    if any(k in t for k in ("ankle", "knee", "hamstring", "calf", "foot",
                            "achilles", "hip", "groin", "quad", "toe", "leg")):
        return "lower_body"
    return "other_injury"


def load():
    con = connect(read_only=True)
    box = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds,
               g.season, g.game_date, g.team_abbrev
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, team_id, season, game_date, team_abbrev
              FROM nba_games WHERE game_id LIKE '002%') g
          USING (game_id, team_id)""").fetchdf()
    sched = con.execute("""
        SELECT DISTINCT game_id, team_id, season, game_date, team_abbrev
        FROM nba_games WHERE game_id LIKE '002%'""").fetchdf()
    rep = con.execute("""
        SELECT i.report_date, i.game_date, i.team, i.status, i.reason,
               p.player_id
        FROM injury_reports_pit i
        JOIN (SELECT player_id, lower(first_name||' '||last_name) fn
              FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '
                          ||trim(split_part(i.player,',',1)))""").fetchdf()
    con.close()
    for d in (box, sched):
        d["game_date"] = pd.to_datetime(d["game_date"])
    rep["report_date"] = pd.to_datetime(rep["report_date"])
    rep["game_date"] = pd.to_datetime(rep["game_date"])
    return box, sched, rep


def build_frame():
    box, sched, rep = load()
    box["mins"] = box["seconds"] / 60.0
    played = box[box["mins"] > 0].copy().sort_values(["player_id", "game_date"])

    # --- rotation universe: appeared for THIS team within ROT_DAYS with
    #     trailing minutes >= ROT_MIN, evaluated strictly before game_date
    pl = played[["player_id", "team_id", "game_date", "mins"]].copy()
    pl = pl.sort_values(["player_id", "team_id", "game_date"])
    g = pl.groupby(["player_id", "team_id"])
    pl["tr_min"] = g["mins"].transform(
        lambda s: s.ewm(halflife=8, adjust=False).mean().shift(1))
    pl["prev_date"] = g["game_date"].shift(1)

    # Restrict to the scored window plus one lead-in season, then sweep each
    # team ONCE keeping a running last-appearance map. The first version
    # filtered the whole team history inside the per-game loop and rebuilt the
    # universe across 20 seasons before discarding most of it.
    LEAD = "2018-19"
    sched = sched[sched["season"] >= LEAD].sort_values(["team_id", "game_date"])
    pl = pl[pl["game_date"] >= sched["game_date"].min() - pd.Timedelta(days=400)]
    cand = []
    for tid, tg in sched.groupby("team_id"):
        hist = pl[pl["team_id"] == tid].sort_values("game_date")
        hi, n = 0, len(hist)
        h_pid = hist["player_id"].to_numpy()
        h_dt = hist["game_date"].to_numpy()
        h_tr = hist["tr_min"].fillna(hist["mins"]).to_numpy()
        last = {}                       # player_id -> (date, trailing minutes)
        for r in tg.itertuples():
            gd64 = np.datetime64(r.game_date)
            while hi < n and h_dt[hi] < gd64:
                last[h_pid[hi]] = (h_dt[hi], h_tr[hi])
                hi += 1
            if not last:
                continue
            pids, trs, dsa = [], [], []
            for pid, (dt_, tr_) in last.items():
                gap = (gd64 - dt_) / np.timedelta64(1, "D")
                if gap <= ROT_DAYS and tr_ >= ROT_MIN:
                    pids.append(pid); trs.append(tr_); dsa.append(gap)
            if not pids:
                continue
            cand.append(pd.DataFrame({
                "game_id": r.game_id, "team_id": tid, "season": r.season,
                "game_date": r.game_date, "team_abbrev": r.team_abbrev,
                "player_id": pids, "tr_min": trs, "days_since_app": dsa}))
    f = pd.concat(cand, ignore_index=True)

    # --- label: did the candidate APPEAR in this game?
    app = box[box["mins"] > 0][["game_id", "player_id"]].assign(app=1)
    f = f.merge(app, on=["game_id", "player_id"], how="left")
    f["y_absent"] = (f["app"].isna()).astype(float)
    f = f.drop(columns="app")

    # --- as-of-open status: last report STRICTLY BEFORE game day
    # CARRY-FORWARD, not a same-date join. THE FIRST VERSION OF THIS SCRIPT
    # KEPT ONLY `report_date < game_date` ROWS, which is D199's original defect
    # reproduced: 74.1% of the archive is same-day (report_date == game_date)
    # and only 891 of 4,752 game dates carry ANY advance row, so 81% of
    # candidates were assigned status 'none' BY CONSTRUCTION. What a bettor
    # knows at the open is the last edition published strictly before game day,
    # whatever game it was filed for -- exactly what `report_out_map` does.
    import bisect
    rep2 = rep.dropna(subset=["player_id"]).copy()
    rep2["rd"] = rep2["report_date"].dt.strftime("%Y-%m-%d")
    by_rd = {}
    for rd, pid, st, rs in zip(rep2["rd"], rep2["player_id"],
                               rep2["status"], rep2["reason"]):
        by_rd.setdefault(rd, {})[int(pid)] = (st, rs)
    rep_dates = sorted(by_rd)
    gd_str = f["game_date"].dt.strftime("%Y-%m-%d").to_numpy()
    pids = f["player_id"].to_numpy()
    st_col, rs_col, age = [], [], []
    cache = {}
    for gd, pid in zip(gd_str, pids):
        j = cache.get(gd)
        if j is None:
            j = bisect.bisect_left(rep_dates, gd) - 1
            cache[gd] = j
        if j < 0:
            st_col.append(None); rs_col.append(None); age.append(np.nan); continue
        ed = by_rd[rep_dates[j]]
        hit = ed.get(int(pid))
        st_col.append(hit[0] if hit else None)
        rs_col.append(hit[1] if hit else None)
        age.append((pd.Timestamp(gd) - pd.Timestamp(rep_dates[j])).days)
    f["status"] = st_col
    f["reason"] = rs_col
    f["report_date"] = f["game_date"] - pd.to_timedelta(age, unit="D")
    f["edition"] = [rep_dates[cache[g]] if cache[g] >= 0 else None
                    for g in gd_str]

    # --- status history over the EDITION SEQUENCE (carry-forward aware).
    # Runs are counted across consecutive editions in which the player appears,
    # so "three straight Out reports" means three straight editions, not three
    # calendar days.
    r2 = rep2[["player_id", "rd", "status"]].drop_duplicates(
        ["player_id", "rd"], keep="last").sort_values(["player_id", "rd"])
    r2["is_out"] = (r2["status"] == "Out").astype(int)
    r2["is_qd"] = r2["status"].isin(["Questionable", "Doubtful"]).astype(int)
    gg = r2.groupby("player_id")
    for c in ("is_out", "is_qd"):
        run = gg[c].transform(
            lambda s_: s_.groupby((s_ != s_.shift()).cumsum()).cumcount() + 1)
        r2[f"run_{c}"] = np.where(r2[c] == 1, run, 0)
    r2["prev_status"] = gg["status"].shift(1).fillna("none")
    r2 = r2.rename(columns={"rd": "edition"})
    r2["player_id"] = r2["player_id"].astype("int64")
    f["player_id"] = f["player_id"].astype("int64")
    f = f.merge(r2[["player_id", "edition", "run_is_out", "run_is_qd",
                    "prev_status"]], on=["player_id", "edition"], how="left")
    for c in ("run_is_out", "run_is_qd"):
        f[c] = f[c].fillna(0)
    f["prev_status"] = f["prev_status"].fillna("none")
    f["status"] = f["status"].fillna("none")
    f["rcat"] = f["reason"].map(reason_cat)
    f["days_since_rep"] = (f["game_date"] - f["report_date"]).dt.days.fillna(99)
    sev = {"none": 0, "Available": 1, "Probable": 2, "Questionable": 3,
           "Doubtful": 4, "Out": 5}
    f["trans"] = (f["status"].map(sev).fillna(0)
                  - f["prev_status"].map(sev).fillna(0))

    # played the team's previous game?
    prev = played[["game_id", "player_id", "team_id", "game_date"]].copy()
    tg = sched[["team_id", "game_date"]].drop_duplicates().sort_values(
        ["team_id", "game_date"])
    tg["prev_team_game"] = tg.groupby("team_id")["game_date"].shift(1)
    f = f.merge(tg, on=["team_id", "game_date"], how="left")
    pset = set(zip(prev["player_id"], prev["team_id"], prev["game_date"]))
    f["played_prev"] = [
        1.0 if (p, t, d) in pset else 0.0
        for p, t, d in zip(f["player_id"], f["team_id"], f["prev_team_game"])]
    return f[f["season"] >= FROM].reset_index(drop=True)


def design(d):
    X = [pd.get_dummies(d["status"]).reindex(
            columns=STATUSES + ["none"], fill_value=0).to_numpy(float),
         pd.get_dummies(d["rcat"]).reindex(
            columns=["gleague", "protocol", "rest_personal", "surgery",
                     "illness", "lower_body", "other_injury", "none"],
            fill_value=0).to_numpy(float),
         np.column_stack([
             d["run_is_out"].clip(0, 15), d["run_is_qd"].clip(0, 15),
             d["trans"], np.minimum(d["days_since_rep"], 30),
             np.minimum(d["days_since_app"], 30), d["played_prev"],
             np.log1p(d["tr_min"])])]
    return np.hstack(X)


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def main():
    f = build_frame()
    print(f"universe {len(f):,} candidate player-games, "
          f"{f.player_id.nunique():,} players, {f.season.min()}..{f.season.max()}")
    print(f"absence rate {f.y_absent.mean():.4f}")

    # ---------- T1: LABEL-ALIGNMENT AUDIT ------------------------------
    print("\n" + "=" * 70)
    print("T1. LABEL ALIGNMENT — administrative 'Out' vs actual non-appearance")
    print("=" * 70)
    ab = f[f.y_absent == 1]
    lab_out = (ab["status"] == "Out").mean()
    print(f"  true non-appearances: {len(ab):,}")
    print(f"    carried a prior-report 'Out'      : {100*lab_out:.1f}%")
    print(f"    carried NO 'Out' label            : {100*(1-lab_out):.1f}%  <- T1")
    print("  their last-known status:")
    for k, v in ab["status"].value_counts().head(6).items():
        print(f"    {k:14} {v:6,}  ({100*v/len(ab):4.1f}%)")
    print("  reason category among unlabelled absences:")
    for k, v in ab[ab.status != "Out"]["rcat"].value_counts().head(6).items():
        print(f"    {k:14} {v:6,}")
    conv = f[f["status"] == "Out"]["y_absent"].mean()
    print(f"  conversely, P(absent | prior-report Out) = {conv:.3f}")

    # ---------- Part 2: head-to-head on the participation label ---------
    print("\n" + "=" * 70)
    print("PART 2. v2 vs the shipped p_out artifact, on ACTUAL non-appearance")
    print("=" * 70)
    inc = pd.read_csv(ROOT / "data" / "p_out.csv.gz")
    inc["game_date"] = pd.to_datetime(inc["game_date"])
    f = f.merge(inc, on=["player_id", "game_date"], how="left")
    f["p_out"] = f["p_out"].fillna(0.0)     # its implicit call for unlisted players
    seasons = sorted(f.season.unique())
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        w = logistic_fit(design(tr), tr["y_absent"].to_numpy(float), lam=5.0)
        Xte = np.column_stack([np.ones(len(te)), design(te)])
        p2 = 1 / (1 + np.exp(-np.clip(Xte @ w, -30, 30)))
        y = te["y_absent"].to_numpy(float)
        p1 = np.clip(te["p_out"].to_numpy(float), 1e-6, 1 - 1e-6)
        p2c = np.clip(p2, 1e-6, 1 - 1e-6)
        ll1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1)).mean()
        ll2 = -(y * np.log(p2c) + (1 - y) * np.log(1 - p2c)).mean()
        rows.append(dict(season=s, n=len(te), ll_inc=ll1, ll_v2=ll2,
                         d_ll=ll2 - ll1,
                         br_inc=float(((p1 - y) ** 2).mean()),
                         br_v2=float(((p2 - y) ** 2).mean()),
                         mean_inc=float(p1.mean()), mean_v2=float(p2.mean()),
                         base=float(y.mean())))
        rows[-1]["d_br"] = rows[-1]["br_v2"] - rows[-1]["br_inc"]
    r = pd.DataFrame(rows)
    print(r[["season", "n", "ll_inc", "ll_v2", "d_ll", "br_inc", "br_v2",
             "d_br"]].to_string(index=False,
                                float_format=lambda v: f"{v:9.5f}"))
    for col, nm in (("d_ll", "log loss"), ("d_br", "Brier")):
        m, lo, hi, t, k = clus(r[col])
        print(f"  {nm:9} season-clustered {m:+.5f}  CI [{lo:+.5f}, {hi:+.5f}] "
              f" t {t:+.2f}  better {int((r[col]<0).sum())}/{k}  "
              f"{'SIG' if hi < 0 else 'ns'}")
    print(f"  mean predicted: incumbent {r.mean_inc.mean():.4f}  "
          f"v2 {r.mean_v2.mean():.4f}  actual {r.base.mean():.4f}")

    # coefficient read (last fold)
    names = (STATUSES + ["st_none"] +
             ["r_gleague", "r_protocol", "r_rest", "r_surgery", "r_illness",
              "r_lower", "r_otherinj", "r_none"] +
             ["run_out", "run_qd", "trans", "d_since_rep", "d_since_app",
              "played_prev", "log_trmin"])
    print("\n  strongest v2 coefficients (final fold):")
    for n_, v in sorted(zip(names, w[1:]), key=lambda z: -abs(z[1]))[:8]:
        print(f"    {n_:14} {v:+.3f}")

    # ---------- emit the v2 artifact -----------------------------------
    out = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        w2 = logistic_fit(design(tr), tr["y_absent"].to_numpy(float), lam=5.0)
        p = 1 / (1 + np.exp(-np.clip(
            np.column_stack([np.ones(len(te)), design(te)]) @ w2, -30, 30)))
        out.append(pd.DataFrame({"game_date": te["game_date"].dt.strftime("%Y-%m-%d"),
                                 "player_id": te["player_id"], "p_out": p}))
    art = pd.concat(out, ignore_index=True).drop_duplicates(
        ["game_date", "player_id"])
    art.to_csv(ROOT / "data" / "p_out_v2.csv.gz", index=False, compression="gzip")
    print(f"\nwrote data/p_out_v2.csv.gz  ({len(art):,} rows)")
    json.dump({"t1_unlabelled_share": float(1 - lab_out),
               "p_absent_given_out": float(conv), "rows": rows},
              open(ROOT / "data" / "d240_participation.json", "w"), default=float)


if __name__ == "__main__":
    main()
