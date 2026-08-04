#!/usr/bin/env python3
"""D94 — statistical re-identification of the masked EPM daily endpoint grid.

AUTHORIZED: Sean, 2026-07-31 (explicit directive to re-identify the pulled
endpoint grid). Scope: data/raw/ext_epm/{date}.json, 638 request dates
2023-10-01..2026-04-30 + 2026-07-31 (values complete, identity masked beyond
the daily top-5: player_id=4 / "Locked Player" / team NBA).

MEASURED FACTS THIS BUILD RESTS ON (verified in-session, 2026-07-31):
 * The mask ALSO genericizes demographics — every locked row carries the same
   bio (age 25, 77in, 205lb, G, rookie 2018), so the originally-planned
   cumulative-stat fingerprint join (gp/min/FGA) is impossible: the payload
   has NO cumulative box stats. What is NOT masked: ~26 continuous per-player
   model fields (off/def/tot EPM + p_* projected rates). Those are the
   identity carriers.
 * Same-player vectors move slowly day to day: nearest-neighbor link distance
   (z-RMS over the 26 fields) median 0.01-0.02 adjacent days with 2nd/1st
   margin median 30-60x; across offseason boundaries margins 7-22x; named
   top-5 rows link 5/5 correct on every probe.
 * The 2026-07-31 request returns the CURRENT table (game_dt 2026-06-13) with
   values byte-identical to the fully-NAMED live /epm page capture
   (live_page_2026-07-31.html, 602 players) -> perfect terminal anchor.
 * Era-B Wayback captures (Dec-2024+) match grid values EXACTLY per player
   (median |dTot| 0.0000) -> high-precision named anchors and held-out
   validation. Era-A captures (2023-10..2024-11): the endpoint serves
   REWRITTEN values (D86: mean|dTot| 0.3-1.0, sorted-rank corr 0.993-0.995),
   so era-A supports only trajectory-level matching / consistency bounds.
 * Grid files with equal game_dt are identical tables (off-day requests);
   the chain runs over DISTINCT game_dt slates.

PIPELINE
 1. Decode all grid dates, dedupe to distinct game_dt.
 2. Backward chaining with an open-head pool: walking game_dt DESC, each
    date's rows are linked one-to-one to open track heads by mutual nearest
    neighbor with distance gates (gap-dependent) and ambiguity margins;
    unlinked rows start new tracks. Handles season rolls (Oct universe
    resets) and mid-season debuts/absences.
 3. PHASE 1 (validation arm): tracks are named ONLY via the live-page
    exact-value join at game_dt 2026-06-13. Top-5 named grid rows and all
    112 Wayback captures are HELD OUT ->
      (i)  top-5 check: % of named grid rows whose track name agrees;
      (ii) era-B Wayback check: per capture, wb rows paired to grid rows by
           exact value; % of pairs where our track name == wayback name;
      (iii) era-A consistency bound: cross-sectional corr + gross-mismatch
           share of our named values vs archived values (rewrite-limited).
 4. PHASE 2 (final table): add top-5 names (server truth), then era-B
    Wayback anchors for still-unnamed tracks (unanimous votes), then era-A
    trajectory assignment (offset-tolerant RMSE with margin gates).
    Confidence tiers: A = named-top5 / chain-live (accuracy measured in
    phase 1), B = wayback-B anchor, C = wayback-A trajectory, else unnamed.
 5. --load: writes DuckDB table epm_history_daily (single-writer discipline,
    retry-safe, same convention as epm_history: asof_date = last game date
    included; PIT consumers filter asof_date < d).

Outputs: data/raw/ext_epm/reid/reid_rows.csv.gz,
         data/logs/epm_reid_validation.json
Run:  python scripts/epm_reid.py          # build + validate
      python scripts/epm_reid.py --load   # load epm_history_daily
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RAW_EPM = REPO / "data/raw/ext_epm"
WB_DIR = RAW_EPM / "wayback"
LIVE_HTML = RAW_EPM / "live_page_2026-07-31.html"
OUT_DIR = RAW_EPM / "reid"
OUT_CSV = OUT_DIR / "reid_rows.csv.gz"
VAL_JSON = REPO / "data/logs/epm_reid_validation.json"
GRID_MIN_REQUEST = "2023-10-01"          # probe files (2003/2022) excluded

# identity-carrying continuous fields (present in every era of the grid)
VEC = ("off", "def", "tot", "p_pct_start", "p_t_poss_48", "p_mp_48", "p_usg",
       "p_pts_100", "p_tspct", "p_efg", "p_fga_rim_100", "p_fga_mid_100",
       "p_fg2a_100", "p_fg3a_100", "p_fta_100", "p_fgpct_rim", "p_fgpct_mid",
       "p_fg2pct", "p_fg3pct", "p_ftpct", "p_ast_100", "p_tov_100",
       "p_orb_100", "p_drb_100", "p_stl_100", "p_blk_100")

# link gates (theory: measured link-distance distributions above; adjacent
# p95 0.07-0.12, offseason p95 0.26-0.28 -> gates sit far into the tail)
GATE_SHORT, GATE_MID, GATE_LONG = 0.45, 0.90, 1.40     # by head gap (days)
GAP_MID, GAP_LONG, GAP_MAX = 5, 45, 500
MARGIN_REL, MARGIN_ABS = 1.35, 0.010     # accept iff d2 >= 1.35*d1 + 0.010

# era-B wayback value pairing. Two tiers: exact float agreement (the
# endpoint reproduces post-boundary values bit-near-exactly) and a loose
# tier; a CAPTURE only counts as value-consistent when >=50% of its rows
# pair — era-B *format* precedes the model-version boundary (Nov-Dec 2024,
# D86), so early era-B captures hold rewritten values and must not anchor.
WB_EXACT, WB_EXACT_MARGIN = 0.002, 0.010
WB_B_TOL, WB_B_MARGIN = 0.010, 0.030
WB_LOOSE, WB_LOOSE_ABS = 0.060, 0.012    # margin-dominant loose tier: the
# endpoint re-runs history at ~0.02-0.05 |dsum| even post-boundary, so a
# pair is kept when the runner-up is at least twice as far + 0.012
WB_CAP_MIN_COV = 0.50
# era-A trajectory assignment (margin applies AFTER the hard DB debut
# filter, which removes most false candidates)
WA_MIN_DATES, WA_SCORE, WA_MARGIN = 4, 0.35, 0.08

LIVE_ROW = re.compile(
    r'\{season:\d+,game_dt:"[\d-]+",player_id:(\d+),player_name:"([^"]*)",'
    r'team_id:(\d+),team_alias:"([^"]*)",age:\d+,inches:"\d*",weight:\d*,'
    r'rookie_year:\d*,position:"[^"]*",'
    r'off:(-?[\d.]+|null),def:(-?[\d.]+|null),tot:(-?[\d.]+|null)')


def decode_full(path: Path) -> list[dict]:
    """SvelteKit devalue decode, ALL fields, rows with non-null tot."""
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        for node in chunk.get("nodes") or []:
            if not (node and node.get("type") == "data"):
                continue
            data = node["data"]
            root = data[0]
            if not (isinstance(root, dict) and "stats" in root):
                continue
            rows = []
            for si in data[root["stats"]]:
                ref = data[si]
                r = {k: data[v] for k, v in ref.items()}
                if isinstance(r.get("tot"), (int, float)):
                    rows.append(r)
            return rows
    return []


def load_grid():
    """Distinct-game_dt slates: [(game_dt, request_date, rows)] ASC."""
    per_dt: dict[str, tuple[str, list[dict]]] = {}
    files = sorted(p for p in RAW_EPM.glob("2*.json")
                   if p.stem >= GRID_MIN_REQUEST)
    for p in files:
        rows = decode_full(p)
        if not rows:
            print(f"  WARN empty decode: {p.name}")
            continue
        gdt = rows[0]["game_dt"]
        if gdt not in per_dt:                    # keep earliest request
            per_dt[gdt] = (p.stem, rows)
    out = [(g, req, rows) for g, (req, rows) in sorted(per_dt.items())]
    print(f"grid: {len(files)} request files -> {len(out)} distinct "
          f"game_dt slates ({out[0][0]} .. {out[-1][0]})")
    return out


def vec_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r[k]) if isinstance(r.get(k), (int, float))
                      else np.nan for k in VEC] for r in rows])


def parse_live() -> list[dict]:
    html = LIVE_HTML.read_text(errors="replace")
    out = []
    for m in LIVE_ROW.finditer(html):
        if m.group(7) == "null":
            continue
        out.append(dict(player_id=int(m.group(1)), player_name=m.group(2),
                        team_id=int(m.group(3)), team_alias=m.group(4),
                        off=float(m.group(5)), def_=float(m.group(6)),
                        tot=float(m.group(7))))
    return out


# ---------------------------------------------------------------- chaining
class Tracks:
    def __init__(self):
        self.rows: list[list[tuple[int, int]]] = []   # track -> [(di, ri)]
        self.maxd: list[float] = []                   # worst link distance

    def new(self, di, ri) -> int:
        self.rows.append([(di, ri)])
        self.maxd.append(0.0)
        return len(self.rows) - 1

    def extend(self, tid, di, ri, d):
        self.rows[tid].append((di, ri))
        self.maxd[tid] = max(self.maxd[tid], float(d))


def build_tracks(slates):
    """Backward chain over distinct game_dt slates. Returns (tracks,
    row_track[di][ri] -> tid, link diagnostics)."""
    mats = [vec_matrix(rows) for _, _, rows in slates]
    allm = np.vstack(mats)
    mu = np.nanmean(allm, axis=0)
    sd = np.nanstd(allm, axis=0)
    sd[sd <= 0] = 1.0
    zs = []
    for m in mats:
        z = (m - mu) / sd
        z[np.isnan(z)] = 0.0
        zs.append(z)
    dates = [dt.date.fromisoformat(g) for g, _, _ in slates]
    tr = Tracks()
    row_track = [np.full(len(m), -1, int) for m in mats]
    head_vec: list[np.ndarray] = []      # per open head
    head_tid: list[int] = []
    head_date: list[dt.date] = []
    diag = dict(links=0, new_tracks=0, ambiguous=0, gap_hist={})
    for di in range(len(slates) - 1, -1, -1):
        Z = zs[di]
        n = len(Z)
        if head_vec:
            keep = [j for j in range(len(head_vec))
                    if (head_date[j] - dates[di]).days <= GAP_MAX]
            head_vec = [head_vec[j] for j in keep]
            head_tid = [head_tid[j] for j in keep]
            head_date = [head_date[j] for j in keep]
        if head_vec:
            H = np.vstack(head_vec)
            gaps = np.array([(hd - dates[di]).days for hd in head_date])
            gate = np.where(gaps <= GAP_MID, GATE_SHORT,
                            np.where(gaps <= GAP_LONG, GATE_MID, GATE_LONG))
            D = np.sqrt(((Z[:, None, :] - H[None, :, :]) ** 2).mean(axis=2))
            # mutual NN with margin, one pass (vectorized)
            jstar = D.argmin(axis=1)
            dsort = np.sort(D, axis=1)
            d1 = dsort[:, 0]
            d2 = dsort[:, 1] if D.shape[1] > 1 else np.full(n, np.inf)
            istar = D.argmin(axis=0)
            csort = np.sort(D, axis=0)
            c2 = csort[1, :] if D.shape[0] > 1 else np.full(D.shape[1], np.inf)
            used_j = set()
            for i in range(n):
                j = jstar[i]
                ok = (istar[j] == i and d1[i] <= gate[j]
                      and d2[i] >= MARGIN_REL * d1[i] + MARGIN_ABS
                      and c2[j] >= MARGIN_REL * d1[i] + MARGIN_ABS)
                if ok and j not in used_j:
                    used_j.add(j)
                    tid = head_tid[j]
                    tr.extend(tid, di, i, d1[i])
                    row_track[di][i] = tid
                    head_vec[j] = Z[i]
                    head_date[j] = dates[di]
                    head_tid[j] = tid
                    diag["links"] += 1
                    g = int(gaps[j])
                    diag["gap_hist"][g] = diag["gap_hist"].get(g, 0) + 1
                elif istar[j] == i and d1[i] <= gate[j]:
                    diag["ambiguous"] += 1
        for i in range(n):
            if row_track[di][i] < 0:
                tid = tr.new(di, i)
                row_track[di][i] = tid
                head_vec.append(Z[i])
                head_tid.append(tid)
                head_date.append(dates[di])
                diag["new_tracks"] += 1
    return tr, row_track, diag, zs


# stitch gates: same theory as the chain gates but applied to track
# endpoints (mean of up to 3 boundary rows -> less noise), candidates
# restricted to time-ordered fragment pairs
ST_GATE_MID, ST_GATE_LONG, ST_GAP_MID, ST_GAP_MAX = 0.90, 1.40, 60, 240
ST_MARGIN_REL, ST_MARGIN_ABS = 1.20, 0.008


def stitch_tracks(tr, slates, row_track, zs):
    """One endpoint-stitch pass: re-attempt the links the per-date chain
    refused, now at TRACK level (fragment A ending right before fragment B
    starts, boundary-mean vectors, mutual NN + margin). Values only — no
    name information used, so held-out validations stay clean."""
    dates = [dt.date.fromisoformat(g) for g, _, _ in slates]
    ntr = len(tr.rows)

    def bmean(t, tail):
        rr = tr.rows[t][:3] if tail else tr.rows[t][-3:]
        return np.mean([zs[di][ri] for di, ri in rr], axis=0)

    ends = [(t, tr.rows[t][0][0], bmean(t, True)) for t in range(ntr)]
    starts = [(t, tr.rows[t][-1][0], bmean(t, False)) for t in range(ntr)]
    E = np.vstack([v for _, _, v in ends])
    S = np.vstack([v for _, _, v in starts])
    e_day = np.array([dates[d].toordinal() for _, d, _ in ends])
    s_day = np.array([dates[d].toordinal() for _, d, _ in starts])
    nf = E.shape[1]
    D2 = ((S ** 2).sum(1)[:, None] + (E ** 2).sum(1)[None, :]
          - 2.0 * S @ E.T)
    D = np.sqrt(np.maximum(D2, 0.0) / nf)
    gap = s_day[:, None] - e_day[None, :]
    D[(gap <= 0) | (gap > ST_GAP_MAX)] = np.inf
    gate = np.where(gap <= ST_GAP_MID, ST_GATE_MID, ST_GATE_LONG)
    parent = list(range(ntr))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_st = 0
    jb = D.argmin(axis=1)
    ds = np.sort(D, axis=1)
    ib = D.argmin(axis=0)
    cs = np.sort(D, axis=0)
    for b in range(ntr):
        a = jb[b]
        if not np.isfinite(D[b, a]):
            continue
        d1 = D[b, a]
        d2 = ds[b, 1] if ntr > 1 else np.inf
        c2 = cs[1, a] if ntr > 1 else np.inf
        if (ib[a] == b and d1 <= gate[b, a]
                and d2 >= ST_MARGIN_REL * d1 + ST_MARGIN_ABS
                and c2 >= ST_MARGIN_REL * d1 + ST_MARGIN_ABS):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                n_st += 1
    # rebuild merged tracks with new sequential ids
    groups: dict[int, list] = {}
    for t in range(ntr):
        groups.setdefault(find(t), []).append(t)
    tr2 = Tracks()
    remap = {}
    for root, members in groups.items():
        rows = sorted((x for m in members for x in tr.rows[m]))
        nid = len(tr2.rows)
        tr2.rows.append(rows[::-1])          # keep DESC order convention
        tr2.maxd.append(max(tr.maxd[m] for m in members))
        for m in members:
            remap[m] = nid
    for di in range(len(row_track)):
        row_track[di] = np.array([remap[int(x)] for x in row_track[di]])
    return tr2, row_track, n_st


# ------------------------------------------------------------------ naming
def wayback_captures():
    caps = []
    for f in sorted(WB_DIR.glob("parsed_*.json")):
        m = json.loads(f.read_text())
        caps.append(m)
    return caps


def db_first_games() -> dict:
    """(player_id, season_start_year) -> first regular-season game date,
    from OUR box scores (read-only)."""
    from nbapred.db import connect
    con = connect(read_only=True)
    q = con.execute("""
        SELECT s.player_id, g.season, min(g.game_date)
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season, game_date FROM nba_games) g
          USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds > 0
        GROUP BY 1, 2""").fetchall()
    con.close()
    out = {}
    for pid, season, d0 in q:
        d0 = d0.date() if hasattr(d0, "date") else d0
        out[(int(pid), int(season[:4]))] = d0
    return out


def pair_capture(cap, slates, slate_idx_by_dt):
    """Era-B value pairing: wb row -> (di, ri) by |doff|+|dtot| with margin.
    Returns list of (wb_row, di, ri) for unambiguous pairs + coverage."""
    asof = cap["asof"]
    di = slate_idx_by_dt.get(asof)
    if di is None:                        # nearest slate <= asof
        cand = [i for i, (g, _, _) in enumerate(slates) if g <= asof]
        if not cand:
            return None, None
        di = cand[-1]
        if (dt.date.fromisoformat(asof)
                - dt.date.fromisoformat(slates[di][0])).days > 3:
            return None, None
    rows = slates[di][2]
    gt = np.array([r["tot"] for r in rows], float)
    go = np.array([r["off"] for r in rows], float)
    gd = np.array([r["def"] for r in rows], float)
    pairs = []
    n_amb = 0
    for r in cap["rows"]:
        d = (np.abs(gt - r["tot"]) + np.abs(go - r["off"])
             + np.abs(gd - r["def_"]))
        j = int(d.argmin())
        s = np.partition(d, 1)[:2]
        ok = ((s[0] <= WB_EXACT and s[1] >= WB_EXACT_MARGIN)
              or (s[0] <= WB_B_TOL and s[1] >= s[0] + WB_B_MARGIN)
              or (s[0] <= WB_LOOSE and s[1] >= 2.0 * s[0] + WB_LOOSE_ABS))
        if ok:
            pairs.append((r, di, j))
        else:
            n_amb += 1
    cov = len(pairs) / max(len(cap["rows"]), 1)
    return pairs, dict(di=di, n_wb=len(cap["rows"]), n_paired=len(pairs),
                       n_unpaired=n_amb, coverage=cov,
                       value_consistent=cov >= WB_CAP_MIN_COV)


def main_build():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slates = load_grid()
    slate_idx_by_dt = {g: i for i, (g, _, _) in enumerate(slates)}
    tr, row_track, chain_diag, zs = build_tracks(slates)
    ntracks = len(tr.rows)
    sizes = np.array([len(x) for x in tr.rows])
    print(f"tracks: {ntracks} (rows {sizes.sum()}, median len "
          f"{int(np.median(sizes))}, links {chain_diag['links']}, "
          f"ambiguous-unlinked {chain_diag['ambiguous']}) "
          f"[{time.time()-t0:.0f}s]")
    tr, row_track, n_st = stitch_tracks(tr, slates, row_track, zs)
    ntracks = len(tr.rows)
    sizes = np.array([len(x) for x in tr.rows])
    print(f"stitch pass: {n_st} merges -> {ntracks} tracks "
          f"(median len {int(np.median(sizes))})")

    # ---- phase 1: live-page anchor only -------------------------------
    live = parse_live()
    anchor_di = slate_idx_by_dt.get("2026-06-13")
    if anchor_di is None:
        raise RuntimeError("anchor slate 2026-06-13 missing")
    arows = slates[anchor_di][2]
    key = {(round(r["off"], 5), round(r["def"], 5), round(r["tot"], 5)): i
           for i, r in enumerate(arows)}
    live_name = {}
    pid_name = {}
    n_anchor = 0
    for lr in live:
        k = (round(lr["off"], 5), round(lr["def_"], 5), round(lr["tot"], 5))
        i = key.get(k)
        if i is None:
            continue
        tid = row_track[anchor_di][i]
        if tid in live_name and live_name[tid] != lr["player_id"]:
            raise RuntimeError("live anchor conflict")
        live_name[tid] = lr["player_id"]
        pid_name[lr["player_id"]] = lr["player_name"]
        n_anchor += 1
    print(f"phase-1 anchors: {n_anchor}/{len(live)} live players "
          f"exact-matched to the 2026-06-13 slate; {len(live_name)} tracks")

    # ---- validation (i): top-5 named grid rows (held out) --------------
    v1 = dict(correct=0, wrong=0, uncovered=0, per_season={})
    wrong_examples = []
    for di, (gdt, _, rows) in enumerate(slates):
        if di == anchor_di:
            continue
        seas = gdt[:4]
        for ri, r in enumerate(rows):
            pid = r.get("player_id") or 0
            if pid <= 100:
                continue
            pid_name.setdefault(pid, r.get("player_name"))
            tid = row_track[di][ri]
            got = live_name.get(tid)
            ps = v1["per_season"].setdefault(seas, [0, 0, 0])
            if got is None:
                v1["uncovered"] += 1
                ps[2] += 1
            elif got == pid:
                v1["correct"] += 1
                ps[0] += 1
            else:
                v1["wrong"] += 1
                ps[1] += 1
                if len(wrong_examples) < 20:
                    wrong_examples.append((gdt, pid, got))
    ntop = v1["correct"] + v1["wrong"] + v1["uncovered"]
    acc_cov = v1["correct"] / max(v1["correct"] + v1["wrong"], 1)
    print(f"validation(i) top-5 held-out: n={ntop} correct={v1['correct']} "
          f"wrong={v1['wrong']} uncovered={v1['uncovered']} "
          f"-> acc|covered={acc_cov:.4f}")

    # ---- validation (ii): wayback captures (held out) ------------------
    # value-consistent era-B captures (post model-version boundary) give the
    # exact name check; era-A captures AND era-B-format captures that fail
    # value consistency (format flipped ~Nov 2024, model boundary Dec 2024 —
    # D86) go to the rewritten-values consistency bound.
    caps = wayback_captures()
    vB = dict(n_caps=0, n_rewritten_caps=0, n_pairs=0, correct=0, wrong=0,
              uncovered=0, pairing_cov=[])
    vA = dict(n_caps=0, corr=[], ceiling=[], gross=[], n_checked=0,
              true_drift=[], true_gross=[], by_era={})
    wb_wrong = []
    consistent_caps = []
    for cap in caps:
        pairs = meta = None
        if cap["era"] == "B":
            pairs, meta = pair_capture(cap, slates, slate_idx_by_dt)
        if meta is not None and meta["value_consistent"]:
            consistent_caps.append(cap["ts"])
            vB["n_caps"] += 1
            vB["pairing_cov"].append(meta["coverage"])
            for wbr, di, ri in pairs:
                tid = row_track[di][ri]
                got = live_name.get(tid)
                vB["n_pairs"] += 1
                if got is None:
                    vB["uncovered"] += 1
                elif got == wbr["player_id"]:
                    vB["correct"] += 1
                else:
                    vB["wrong"] += 1
                    if len(wb_wrong) < 20:
                        wb_wrong.append((cap["asof"], wbr["player_id"], got))
                pid_name.setdefault(wbr["player_id"], wbr["player_name"])
        else:                     # rewritten values: consistency bound only
            if cap["era"] == "B":
                vB["n_rewritten_caps"] += 1
            asof = cap["asof"]
            cand = [i for i, (g, _, _) in enumerate(slates) if g <= asof]
            if not cand:
                continue
            di = cand[-1]
            byname, top5 = {}, {}
            for ri, r in enumerate(slates[di][2]):
                pid = r.get("player_id") or 0
                if pid > 100:
                    top5[pid] = r["tot"]     # server truth on this slate
                    continue
                got = live_name.get(row_track[di][ri])
                if got is not None:
                    byname[got] = r["tot"]
            xs, ys = [], []
            for r in cap["rows"]:
                pid_name.setdefault(r["player_id"], r["player_name"])
                if r["player_id"] in top5:
                    d = abs(top5[r["player_id"]] - r["tot"])
                    vA["true_drift"].append(d)      # known-correct yardstick
                    vA["true_gross"].append(float(d > 1.0))
                if r["player_id"] in byname:
                    xs.append(byname[r["player_id"]])
                    ys.append(r["tot"])
            if len(xs) < 50:
                continue
            xs, ys = np.array(xs), np.array(ys)
            vA["n_caps"] += 1
            vA["n_checked"] += len(xs)
            corr = float(np.corrcoef(xs, ys)[0, 1])
            ceil = float(np.corrcoef(np.sort(xs), np.sort(ys))[0, 1])
            vA["corr"].append(corr)
            vA["ceiling"].append(ceil)               # same-set rank ceiling
            vA["gross"].append(float(np.mean(np.abs(xs - ys) > 1.0)))
            era_key = "A" if cap["era"] == "A" else "B-rewritten"
            e = vA["by_era"].setdefault(era_key, dict(corr=[], ceiling=[],
                                                      med_drift=[]))
            e["corr"].append(corr)
            e["ceiling"].append(ceil)
            e["med_drift"].append(float(np.median(np.abs(xs - ys))))
    accB = vB["correct"] / max(vB["correct"] + vB["wrong"], 1)
    print(f"validation(ii) era-B value-consistent held-out: "
          f"caps={vB['n_caps']} (rewritten-B excluded: "
          f"{vB['n_rewritten_caps']}) pairs={vB['n_pairs']} "
          f"correct={vB['correct']} wrong={vB['wrong']} "
          f"uncovered={vB['uncovered']} -> acc|covered={accB:.4f}")
    if vA["corr"]:
        print(f"validation(iii) rewritten-era bound: caps={vA['n_caps']} "
              f"pairs={vA['n_checked']} corr med="
              f"{float(np.median(vA['corr'])):.4f} (same-set sorted ceiling "
              f"med={float(np.median(vA['ceiling'])):.4f}) gross med="
              f"{float(np.median(vA['gross'])):.4f} | KNOWN-CORRECT top-5 "
              f"drift: n={len(vA['true_drift'])} med="
              f"{float(np.median(vA['true_drift'])):.3f} p80="
              f"{float(np.percentile(vA['true_drift'],80)):.3f} "
              f"share>1.0={float(np.mean(vA['true_gross'])):.3f}")

    # ---- phase 2: final naming ----------------------------------------
    track_pid = dict(live_name)                       # tier A (chain-live)
    track_method = {t: "chain-live" for t in track_pid}
    # (a) top-5 server truth: name tracks containing named rows
    conflicts = 0
    for di, (gdt, _, rows) in enumerate(slates):
        for ri, r in enumerate(rows):
            pid = r.get("player_id") or 0
            if pid <= 100:
                continue
            tid = row_track[di][ri]
            if tid not in track_pid:
                track_pid[tid] = pid
                track_method[tid] = "top5-chain"
            elif track_pid[tid] != pid:
                conflicts += 1                       # chain break: row keeps
    # (b) era-B wayback anchors for unnamed tracks (unanimous votes from
    # VALUE-CONSISTENT captures only; date-overlap guard vs existing claims)
    def track_range(t):
        gs = [slates[di][0] for di, _ in tr.rows[t]]
        return min(gs), max(gs)

    pid_ranges: dict[int, list] = {}
    for t, p in track_pid.items():
        pid_ranges.setdefault(p, []).append(track_range(t))
    votes: dict[int, set] = {}
    for cap in caps:
        if cap["era"] != "B" or cap["ts"] not in consistent_caps:
            continue
        pairs, _ = pair_capture(cap, slates, slate_idx_by_dt)
        if not pairs:
            continue
        for wbr, di, ri in pairs:
            tid = row_track[di][ri]
            if tid not in track_pid:
                votes.setdefault(tid, set()).add(wbr["player_id"])
    nB = 0
    for tid, vs in sorted(votes.items(),
                          key=lambda kv: -len(tr.rows[kv[0]])):
        if len(vs) != 1:
            continue
        pid = next(iter(vs))
        g0, g1 = track_range(tid)
        if any(not (g1 < a or g0 > b) for a, b in pid_ranges.get(pid, [])):
            continue                       # overlaps an existing claim
        track_pid[tid] = pid
        track_method[tid] = "wayback-B"
        pid_ranges.setdefault(pid, []).append((g0, g1))
        nB += 1
    # (c) era-A trajectory assignment for remaining unnamed tracks:
    # 3-component drift-tolerant series match (per-component offset removed:
    # the rewrite is a roughly-constant per-player level shift) + a HARD
    # debut filter from OUR player_game_stats (a row enters the universe at
    # the player's first regular-season game of that season).
    db_first = db_first_games()
    season_start = {}
    for (p, y), d0 in db_first.items():
        if y not in season_start or d0 < season_start[y]:
            season_start[y] = d0
    era_a = [c for c in caps if c["era"] == "A"]
    wb_series: dict[int, dict[str, tuple]] = {}
    for cap in era_a:
        for r in cap["rows"]:
            wb_series.setdefault(r["player_id"], {})[cap["asof"]] = (
                r["off"], r["def_"], r["tot"])
    unnamed = [t for t in range(ntracks) if t not in track_pid
               and len(tr.rows[t]) >= WA_MIN_DATES]
    track_val = {}
    for t in unnamed:
        m = {}
        for di, ri in tr.rows[t]:
            g = slates[di][0]
            if g <= "2024-12-01":
                r = slates[di][2][ri]
                m[g] = (r["off"], r["def"], r["tot"])
        if len(m) >= WA_MIN_DATES:
            track_val[t] = m
    gdts = sorted({g for m in track_val.values() for g in m})
    nA = 0
    if track_val and wb_series:
        asof2g = {}
        for cap in era_a:
            cand = [g for g in gdts if g <= cap["asof"]]
            if cand:
                asof2g[cap["asof"]] = cand[-1]

        def season_of(g):
            y, mth = int(g[:4]), int(g[5:7])
            return y if mth >= 7 else y - 1

        def debut_ok(t, p):
            """Roster-based universe (measured: injured vets appear from
            their team's opener) -> the valid direction is NO-LATE-ENTRY:
            the track must not START after the pid's first game (+4d); the
            pid must have played every season the track spans."""
            by_season = {}
            for di, _ in tr.rows[t]:
                g = slates[di][0]
                y = season_of(g)
                if y in season_start and g >= str(season_start[y]):
                    by_season.setdefault(y, []).append(g)
            if not by_season:
                return False
            for y in by_season:
                if (p, y) not in db_first:
                    return False
            y0 = min(by_season)
            t0 = dt.date.fromisoformat(min(by_season[y0]))
            return (t0 - db_first[(p, y0)]).days <= 4
        scores = []
        diag_sc, diag_mg = [], []
        for t, m in track_val.items():
            best = []
            for p, series in wb_series.items():
                xs, ys = [], []
                for asof, v in series.items():
                    g = asof2g.get(asof)
                    if g in m:
                        xs.append(m[g])
                        ys.append(v)
                if len(xs) < WA_MIN_DATES:
                    continue
                xs, ys = np.array(xs), np.array(ys)
                sc = 0.0
                for c in range(3):
                    off = float(np.median(ys[:, c] - xs[:, c]))
                    sc += float(np.mean(np.abs(ys[:, c] - xs[:, c] - off))) \
                        + 0.10 * abs(off)
                sc /= 3.0
                best.append((sc, p, len(xs)))
            best.sort()
            if not best:
                continue
            diag_sc.append(best[0][0])
            diag_mg.append(best[1][0] - best[0][0] if len(best) > 1
                           else np.inf)
            filt = [b for b in best if debut_ok(t, b[1])]
            if filt and filt[0][0] <= WA_SCORE and (
                    len(filt) == 1 or filt[1][0] - filt[0][0] >= WA_MARGIN):
                scores.append((filt[0][0], t, filt[0][1], filt[0][2]))
        if diag_sc:
            q = np.percentile(diag_sc, [10, 25, 50, 75, 90])
            fin = [x for x in diag_mg if np.isfinite(x)]
            qm = np.percentile(fin, [10, 25, 50, 75]) if fin else []
            print(f"  era-A trajectory diag: {len(diag_sc)} tracks scored, "
                  f"best-score q10/25/50/75/90 = "
                  f"{[round(float(x),3) for x in q]}, margin q10/25/50/75 = "
                  f"{[round(float(x),3) for x in qm]}, pass-gates "
                  f"{len(scores)}")
        used = set()
        for sc, t, p, npts in sorted(scores):
            if p in used:
                continue
            g0 = min(slates[di][0] for di, _ in tr.rows[t])
            g1 = max(slates[di][0] for di, _ in tr.rows[t])
            if any(not (g1 < a or g0 > b)
                   for a, b in pid_ranges.get(p, [])):
                continue
            track_pid[t] = p
            track_method[t] = "wayback-A"
            pid_ranges.setdefault(p, []).append((g0, g1))
            used.add(p)
            nA += 1
    print(f"phase-2: +top5-chain "
          f"{sum(1 for m in track_method.values() if m == 'top5-chain')} "
          f"tracks, +wayback-B {nB}, +wayback-A {nA}, "
          f"top5-conflicts {conflicts}")

    # ---- emit rows -----------------------------------------------------
    conf = {"named-top5": "A", "chain-live": "A", "top5-chain": "A",
            "wayback-B": "B", "wayback-A": "C"}
    n_out = n_unnamed = 0
    rot_total = rot_named = 0            # p_mp_48 >= 12 (rotation proxy)
    first_named: dict[tuple, tuple] = {}   # (pid, season_y) -> (gdt, method)
    dupes = 0
    with gzip.open(OUT_CSV, "wt") as f:
        f.write("game_dt,request_date,player_id,player_name,team_id,"
                "team_alias,off,def,tot,method,confidence,track_id,"
                "track_maxd\n")
        for di, (gdt, req, rows) in enumerate(slates):
            seen = set()
            for ri, r in enumerate(rows):
                tid = row_track[di][ri]
                pid = r.get("player_id") or 0
                is_rot = isinstance(r.get("p_mp_48"), (int, float)) \
                    and r["p_mp_48"] >= 12.0
                rot_total += is_rot
                if pid > 100:
                    meth = "named-top5"
                    team_id, team_alias = r["team_id"], r["team_alias"]
                    name = r["player_name"]
                else:
                    pid = track_pid.get(tid, 0)
                    meth = track_method.get(tid, "")
                    team_id, team_alias = 0, ""
                    name = pid_name.get(pid, "")
                if not pid:
                    n_unnamed += 1
                    continue
                if pid in seen:
                    dupes += 1
                    continue
                seen.add(pid)
                rot_named += is_rot
                y = int(gdt[:4]) if int(gdt[5:7]) >= 7 else int(gdt[:4]) - 1
                k = (pid, y)
                if k not in first_named or gdt < first_named[k][0]:
                    first_named[k] = (gdt, meth)
                nm = (name or "").replace(",", " ")
                f.write(f"{gdt},{req},{pid},{nm},{team_id},{team_alias},"
                        f"{r['off']},{r['def']},{r['tot']},{meth},"
                        f"{conf.get(meth,'')},{tid},"
                        f"{tr.maxd[tid]:.4f}\n")
                n_out += 1
    total_rows = int(sizes.sum())
    print(f"rows out: {n_out}/{total_rows} named "
          f"({n_out/total_rows:.3f}; rotation p_mp_48>=12: "
          f"{rot_named}/{rot_total} = {rot_named/max(rot_total,1):.3f}), "
          f"unnamed {n_unnamed}, dupes-dropped {dupes} -> {OUT_CSV}")

    # ---- validation (iv): DB no-late-entry check (independent of D&T) --
    # The endpoint universe is ROSTER-based (measured: injured veterans
    # appear from their team's opener, e.g. LeBron 2025-10-21 vs first game
    # 2025-11-18), so entry BEFORE the first game is legitimate. The valid
    # error signal is the other direction: a player cannot have PLAYED
    # while his row was absent -> first named date must be <= first game
    # (+4d). Late debuts (>25d after season start) are the sharp subset.
    vd = dict(n=0, ok=0, late_entry=0, no_db_season=0,
              late=dict(n=0, ok=0), by_method={}, flag_examples=[])
    for (pid, y), (gdt, meth) in sorted(first_named.items()):
        # only seasons whose start the grid covers (2022-23 appears solely
        # as the June-2023 carryover slate -> entry undefined there)
        if y not in (2023, 2024, 2025) or y not in season_start:
            continue
        vd["n"] += 1
        bm = vd["by_method"].setdefault(meth, [0, 0, 0])
        if (pid, y) not in db_first:
            vd["no_db_season"] += 1     # rostered-never-played is possible
            bm[2] += 1
            continue
        d0 = db_first[(pid, y)]
        late = (d0 - season_start[y]).days > 25
        ok = (dt.date.fromisoformat(gdt) - d0).days <= 4
        vd["ok"] += ok
        vd["late_entry"] += (not ok)
        bm[0 if ok else 1] += 1
        if late:
            vd["late"]["n"] += 1
            vd["late"]["ok"] += ok
        if not ok and len(vd["flag_examples"]) < 15:
            vd["flag_examples"].append((pid, y, gdt, str(d0), meth))
    print(f"validation(iv) DB no-late-entry: n={vd['n']} ok={vd['ok']} "
          f"({vd['ok']/max(vd['n'],1):.4f}) late-entry-flags="
          f"{vd['late_entry']} rostered-no-game={vd['no_db_season']} | "
          f"late-debut subset {vd['late']['ok']}/{vd['late']['n']} = "
          f"{vd['late']['ok']/max(vd['late']['n'],1):.4f}")

    res = dict(
        built=dt.datetime.now().isoformat(timespec="seconds"),
        config=dict(vec_fields=list(VEC), gates=[GATE_SHORT, GATE_MID,
                    GATE_LONG], gaps=[GAP_MID, GAP_LONG, GAP_MAX],
                    margin=[MARGIN_REL, MARGIN_ABS],
                    wb_b=[WB_B_TOL, WB_B_MARGIN],
                    wb_a=[WA_MIN_DATES, WA_SCORE, WA_MARGIN]),
        grid=dict(slates=len(slates), rows=total_rows,
                  span=[slates[0][0], slates[-1][0]]),
        chain=dict(tracks=ntracks, links=chain_diag["links"],
                   ambiguous_unlinked=chain_diag["ambiguous"],
                   stitch_merges=n_st,
                   median_track_len=int(np.median(sizes))),
        phase1=dict(anchors=n_anchor, live_players=len(live)),
        validation_top5=dict(
            n=ntop, correct=v1["correct"], wrong=v1["wrong"],
            uncovered=v1["uncovered"],
            accuracy_given_covered=round(acc_cov, 5),
            per_season=v1["per_season"], wrong_examples=wrong_examples,
            note="held out of phase-1; anchor slate excluded"),
        validation_wayback_B=dict(
            n_caps_value_consistent=vB["n_caps"],
            n_caps_rewritten_format_B=vB["n_rewritten_caps"],
            n_pairs=vB["n_pairs"],
            correct=vB["correct"], wrong=vB["wrong"],
            uncovered=vB["uncovered"],
            accuracy_given_covered=round(accB, 5),
            mean_pairing_coverage=round(float(np.mean(vB["pairing_cov"])), 4)
            if vB["pairing_cov"] else None,
            wrong_examples=wb_wrong,
            note="fully held out of phase-1 naming; era-B-FORMAT captures "
                 "predating the model-version boundary fail the >=50% "
                 "value-pairing gate and are scored in the bound below"),
        validation_rewritten_bound=dict(
            n_caps=vA["n_caps"], n_pairs=vA["n_checked"],
            corr_median=round(float(np.median(vA["corr"])), 4)
            if vA["corr"] else None,
            corr_min=round(float(np.min(vA["corr"])), 4)
            if vA["corr"] else None,
            sorted_ceiling_median=round(float(np.median(vA["ceiling"])), 4)
            if vA["ceiling"] else None,
            gross_mismatch_median=round(float(np.median(vA["gross"])), 4)
            if vA["gross"] else None,
            known_correct_top5_drift=dict(
                n=len(vA["true_drift"]),
                median=round(float(np.median(vA["true_drift"])), 3),
                p80=round(float(np.percentile(vA["true_drift"], 80)), 3),
                share_gt_1=round(float(np.mean(vA["true_gross"])), 3))
            if vA["true_drift"] else None,
            note="rewritten-era captures (era A + pre-boundary era-B "
                 "format): endpoint serves REWRITTEN values (D86), so only "
                 "an identity-corr consistency bound is possible; the "
                 "known-correct top-5 drift is the yardstick for how much "
                 "drift correct pairs show"),
        validation_db_no_late_entry=dict(
            n=vd["n"], ok=vd["ok"], late_entry_flags=vd["late_entry"],
            rostered_no_game=vd["no_db_season"],
            pass_share=round(vd["ok"] / max(vd["n"], 1), 4),
            late_debut_subset=dict(
                n=vd["late"]["n"], ok=vd["late"]["ok"],
                pass_share=round(vd["late"]["ok"]
                                 / max(vd["late"]["n"], 1), 4)),
            by_method={k: dict(ok=v[0], late_entry=v[1], no_db=v[2])
                       for k, v in vd["by_method"].items()},
            flag_examples=vd["flag_examples"],
            note="independent of D&T values; the universe is ROSTER-based "
                 "(injured vets appear at their team's opener), so the "
                 "one-directional check is: a player cannot have PLAYED a "
                 "002 game while his row was absent (first named date <= "
                 "first game +4d vs player_game_stats)"),
        rewritten_bound_by_era={
            k: dict(n_caps=len(v["corr"]),
                    corr_median=round(float(np.median(v["corr"])), 4),
                    ceiling_median=round(float(np.median(v["ceiling"])), 4),
                    med_drift_median=round(
                        float(np.median(v["med_drift"])), 4))
            for k, v in vA["by_era"].items()},
        phase2=dict(top5_chain=sum(1 for m in track_method.values()
                                   if m == "top5-chain"),
                    wayback_B=nB, wayback_A=nA, top5_conflicts=conflicts),
        output=dict(rows_named=n_out, rows_unnamed=n_unnamed,
                    dupes_dropped=dupes,
                    named_share=round(n_out / total_rows, 4),
                    rotation_rows=rot_total, rotation_named=rot_named,
                    rotation_named_share=round(
                        rot_named / max(rot_total, 1), 4)))
    VAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    VAL_JSON.write_text(json.dumps(res, indent=1))
    print(f"validation -> {VAL_JSON}  [{time.time()-t0:.0f}s total]")


# ------------------------------------------------------------------- load
WRITERS = re.compile(r"^\s*\d+\s+python[0-9.]*\s+\S*(build_features|"
                     r"backfill_\w+|load_\w+|build_player_stats)\.py")


def writers_running() -> list[str]:
    me = str(Path(__file__).name)
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                         text=True).stdout
    return [x.strip()[:120] for x in out.splitlines()
            if WRITERS.search(x) and me not in x]


DDL = """
CREATE TABLE IF NOT EXISTS epm_history_daily (
    asof_date    DATE NOT NULL,
    request_date DATE NOT NULL,
    player_id    BIGINT NOT NULL,
    player_name  VARCHAR,
    team_id      BIGINT,
    team_alias   VARCHAR,
    off_epm      DOUBLE,
    def_epm      DOUBLE,
    tot_epm      DOUBLE,
    method       VARCHAR,
    confidence   VARCHAR,
    track_id     INTEGER,
    ingest_ts    TIMESTAMPTZ
);
"""


def main_load():
    import duckdb
    import pandas as pd

    from nbapred.config import DB_PATH
    hits = writers_running()
    if hits:
        print("WRITER ACTIVE — load deferred:")
        for h in hits:
            print("  ", h)
        sys.exit(2)
    df = pd.read_csv(OUT_CSV)
    df["ingest_ts"] = dt.datetime.now(dt.timezone.utc)
    con = None
    for attempt in range(6):
        try:
            con = duckdb.connect(str(DB_PATH))
            break
        except duckdb.IOException as e:
            print(f"  lock busy ({e}); retry {attempt + 1}/6 in 10s")
            time.sleep(10)
    if con is None:
        print("DB LOCKED — load deferred (raw artifacts intact; rerun)")
        sys.exit(2)
    con.execute(DDL)
    con.register("reid_stage", df)
    con.execute("BEGIN")
    con.execute("DELETE FROM epm_history_daily")
    con.execute("""INSERT INTO epm_history_daily
        SELECT game_dt, request_date, player_id,
               NULLIF(player_name,''), NULLIF(team_id,0)::BIGINT,
               NULLIF(team_alias,''), off, "def", tot, method, confidence,
               track_id, ingest_ts FROM reid_stage""")
    con.execute("COMMIT")
    chk = con.execute("""
        SELECT count(*), count(DISTINCT asof_date),
               count(DISTINCT player_id), min(asof_date), max(asof_date)
        FROM epm_history_daily""").fetchone()
    con.close()
    print(f"epm_history_daily loaded: rows={chk[0]} asof_dates={chk[1]} "
          f"players={chk[2]} span {chk[3]}..{chk[4]}")


if __name__ == "__main__":
    if "--load" in sys.argv:
        main_load()
    else:
        main_build()
