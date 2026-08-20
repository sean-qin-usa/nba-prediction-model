#!/usr/bin/env python3
"""D172 TASK 1 — the FULL team-name/abbreviation join audit.

Third instance of the same bug class (D119 name-map "scrape failure",
D161 938 games lost to era abbreviations, D171 2,514 Clippers rows).
This enumerates EVERY surface in the repo where a team is matched by NAME or
ABBREVIATION rather than by team_id, measures the match rate in BOTH
directions, and reports every unresolved value with its row count.

READ-ONLY on data/nba.duckdb.
"""
from __future__ import annotations

import sys                                                        # noqa: E402
from pathlib import Path                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import json                                                       # noqa: E402

import duckdb                                                     # noqa: E402
import pandas as pd                                               # noqa: E402

DB = str(ROOT / "data" / "nba.duckdb")

from nbapred import teams as T                                    # noqa: E402


def ro(attempts=10, wait=60.0):
    import time
    for i in range(attempts):
        try:
            return duckdb.connect(DB, read_only=True)
        except Exception as e:                                    # noqa: BLE001
            if ("lock" not in str(e).lower() and "held" not in str(e).lower()) \
               or i == attempts - 1:
                raise
            time.sleep(wait)


con = ro()
FINDINGS = []          # one row per (surface, unresolved value)
SURFACES = []          # one row per surface


def add_surface(name, side_a, side_b, a_vals, b_vals, a_counts=None,
                b_counts=None, note=""):
    """a_vals/b_vals: sets. counts: dict value->rows (side A only needed)."""
    a, b = set(a_vals), set(b_vals)
    a_only, b_only = sorted(a - b), sorted(b - a)
    ca = a_counts or {}
    cb = b_counts or {}
    rows_a = sum(ca.values()) if ca else 0
    rows_lost = sum(ca.get(v, 0) for v in a_only)
    SURFACES.append(dict(
        surface=name, side_a=side_a, side_b=side_b,
        n_a=len(a), n_b=len(b),
        match_a_to_b=(len(a) - len(a_only)) / len(a) if a else float("nan"),
        match_b_to_a=(len(b) - len(b_only)) / len(b) if b else float("nan"),
        unresolved_a=len(a_only), unresolved_b=len(b_only),
        rows_a=rows_a, rows_unmatched=rows_lost, note=note))
    for v in a_only:
        FINDINGS.append(dict(surface=name, direction="A->B", value=v,
                             rows=ca.get(v, -1)))
    for v in b_only:
        FINDINGS.append(dict(surface=name, direction="B->A", value=v,
                             rows=cb.get(v, -1)))


def vc(sql):
    """value -> row count dict from a 2-col (value, n) query."""
    return {r[0]: int(r[1]) for r in con.execute(sql).fetchall()
            if r[0] is not None}


print("=" * 100)
print("D172 §1  TEAM-NAME / ABBREVIATION JOIN AUDIT — EVERY SURFACE")
print("=" * 100)

# ---------------------------------------------------------------- vocabularies
import nba_api.stats.static.teams as _t                           # noqa: E402
API = _t.get_teams()
MODERN_AB = {t["abbreviation"] for t in API}
MODERN_NAME = {t["full_name"] for t in API}
ID2AB = {int(t["id"]): t["abbreviation"] for t in API}

# The FRANCHISE crosswalk as it exists in the repo (3 copies: k19_model.py,
# k19_t2.py, ats19_score.py — all identical).
FRANCHISE = {"SEA": "OKC", "NJN": "BKN", "NOH": "NOP", "NOK": "NOP",
             "VAN": "MEM", "CHH": "CHA"}

print(f"\nnba_api modern universe: {len(MODERN_AB)} abbrevs, "
      f"{len(MODERN_NAME)} full names")
print(f"repo FRANCHISE crosswalk (3 duplicate copies): {FRANCHISE}")

# ============================================================== SURFACE GROUP 1
# nba_games.team_abbrev — the spine everything joins to.
print("\n" + "-" * 100)
print("A. nba_games.team_abbrev — the spine.  What codes exist, by season?")
print("-" * 100)
ng = con.execute("""
    SELECT season, team_abbrev, COUNT(*) n, COUNT(DISTINCT team_id) nid
      FROM nba_games GROUP BY 1,2 ORDER BY 1,2""").fetchdf()
all_ng_ab = vc("SELECT team_abbrev, COUNT(*) FROM nba_games GROUP BY 1")
print(f"nba_games: {len(all_ng_ab)} distinct abbrevs over "
      f"{sum(all_ng_ab.values()):,} team-game rows, "
      f"{ng.season.nunique()} seasons "
      f"({ng.season.min()}..{ng.season.max()})")
era_codes = sorted(set(all_ng_ab) - MODERN_AB)
print(f"NON-MODERN codes present: {era_codes}")
for c in era_codes:
    ss = ng[ng.team_abbrev == c]
    print(f"   {c}: {all_ng_ab[c]:>5,} rows, seasons "
          f"{ss.season.min()}..{ss.season.max()}, "
          f"crosswalk={'YES -> ' + FRANCHISE[c] if c in FRANCHISE else '*** MISSING ***'}")

# team_id vs team_abbrev consistency: does the id always agree with the code?
mism = con.execute("""
    SELECT team_id, team_abbrev, COUNT(*) n FROM nba_games
     GROUP BY 1,2 ORDER BY 1,2""").fetchdf()
mism["ab_from_id"] = mism.team_id.map(ID2AB)
bad = mism[mism.ab_from_id.notna() & (mism.ab_from_id != mism.team_abbrev)]
print(f"\nteam_id -> abbrev disagreements (the ERA codes, by construction): "
      f"{len(bad)} (team_id, abbrev) pairs, {int(bad.n.sum()):,} rows")
unk_id = mism[mism.ab_from_id.isna()]
print(f"team_ids NOT in nba_api's 30: {len(unk_id)} "
      f"({int(unk_id.n.sum()) if len(unk_id) else 0:,} rows)")

# ============================================================== SURFACE 2..4
# market joins: nba_games (fx-mapped) vs odds tables
print("\n" + "-" * 100)
print("B. MARKET JOINS — nba_games.team_abbrev  vs  odds_* home/away")
print("-" * 100)

ng_raw = set(all_ng_ab)
ng_fx = {FRANCHISE.get(a, a) for a in ng_raw}
ng_fx_counts = {}
for a, n in all_ng_ab.items():
    k = FRANCHISE.get(a, a)
    ng_fx_counts[k] = ng_fx_counts.get(k, 0) + n

for tbl, cols in (("odds_market", ("home", "away")),
                  ("odds_open", ("home", "away")),
                  ("odds_hist_sbr", ("home", "visitor"))):
    n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    if n == 0:
        print(f"\n{tbl}: EMPTY, skipped")
        continue
    cnt = {}
    for c in cols:
        for k, v in vc(f"SELECT {c}, COUNT(*) FROM {tbl} GROUP BY 1").items():
            cnt[k] = cnt.get(k, 0) + v
    print(f"\n{tbl}: {n:,} game rows, {len(cnt)} distinct codes")
    add_surface(f"{tbl}.{'/'.join(cols)}", tbl, "nba_games.team_abbrev(fx)",
                cnt, ng_fx, cnt, ng_fx_counts,
                note="market join key is (game_date, home, away)")
    # RAW direction — what happens WITHOUT the crosswalk (D161's bug)
    add_surface(f"{tbl}.{'/'.join(cols)} [NO fx crosswalk]", tbl,
                "nba_games.team_abbrev(RAW)", cnt, ng_raw, cnt, all_ng_ab,
                note="D161 regression probe: unmatched here = games silently lost")

# how many nba_games rows carry a code the odds tables can never see
for tbl in ("odds_market", "odds_open"):
    if con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] == 0:
        continue
    codes = set()
    for c in ("home", "away"):
        codes |= {r[0] for r in con.execute(
            f"SELECT DISTINCT {c} FROM {tbl}").fetchall() if r[0]}
    lost_raw = {a: n for a, n in all_ng_ab.items() if a not in codes}
    print(f"\n  nba_games rows whose RAW abbrev is absent from {tbl}: "
          f"{sum(lost_raw.values()):,} "
          f"({ {k: v for k, v in sorted(lost_raw.items())} })")
    lost_fx = {a: n for a, n in ng_fx_counts.items() if a not in codes}
    print(f"  ... after the FRANCHISE crosswalk: {sum(lost_fx.values()):,} "
          f"{sorted(lost_fx)}")

# ============================================================== SURFACE 5
print("\n" + "-" * 100)
print("C. INJURY REPORTS — injury_reports_pit.team (PDF names) vs teams.py")
print("-" * 100)
inj = vc("SELECT team, COUNT(*) FROM injury_reports_pit GROUP BY 1")
ok, bad_names = T.resolve_map(inj)
print(f"injury_reports_pit: {sum(inj.values()):,} rows, {len(inj)} distinct "
      f"'team' strings")
print(f"  resolved by nbapred/teams.py: {len(ok)}")
print(f"  UNRESOLVED: {len(bad_names)} -> "
      f"{ {b: inj[b] for b in bad_names} }")
# legacy behaviour (the D171 bug), for the record
legacy = {t["full_name"]: t["abbreviation"] for t in API}
legacy_bad = {k: v for k, v in inj.items() if k not in legacy}
print(f"  would have been dropped by the PRE-D171 inline map: "
      f"{sum(legacy_bad.values()):,} rows over {len(legacy_bad)} strings")
add_surface("injury_reports_pit.team", "injury PDF names",
            "nbapred/teams.py", set(inj), set(inj) - set(bad_names),
            inj, {}, note="routed through teams.py at D171")
# reverse: which of the 30 franchises never appears?
seen_ab = {T.abbrev_for(k) for k in ok}
print(f"  REVERSE — franchises with ZERO injury rows: "
      f"{sorted(MODERN_AB - seen_ab)}")

# ============================================================== SURFACE 6
print("\n" + "-" * 100)
print("D. schedule_features.team / .opponent  vs  nba_games.team_abbrev")
print("-" * 100)
sf = {}
for c in ("team", "opponent"):
    for k, v in vc(f"SELECT {c}, COUNT(*) FROM schedule_features GROUP BY 1").items():
        sf[k] = sf.get(k, 0) + v
sfs = con.execute("SELECT MIN(season), MAX(season), COUNT(*) "
                  "FROM schedule_features").fetchone()
print(f"schedule_features: {sfs[2]:,} rows, seasons {sfs[0]}..{sfs[1]}, "
      f"{len(sf)} distinct codes")
add_surface("schedule_features.team/opponent", "schedule_features",
            "nba_games.team_abbrev(RAW)", sf, ng_raw, sf, all_ng_ab)

# ============================================================== SURFACE 7
print("\n" + "-" * 100)
print("E. epm_history / epm_history_daily .team_alias  vs modern abbrevs")
print("-" * 100)
for tbl in ("epm_history", "epm_history_daily"):
    al = vc(f"SELECT team_alias, COUNT(*) FROM {tbl} GROUP BY 1")
    print(f"{tbl}: {sum(al.values()):,} rows, {len(al)} distinct aliases")
    add_surface(f"{tbl}.team_alias", tbl, "nba_api abbreviations",
                al, MODERN_AB, al, {})
    # is team_alias consistent with team_id in the same row?
    mm = con.execute(f"""
        SELECT team_id, team_alias, COUNT(*) n FROM {tbl}
         WHERE team_id IS NOT NULL GROUP BY 1,2""").fetchdf()
    if len(mm):
        mm["ab"] = mm.team_id.map(ID2AB)
        d = mm[mm.ab.notna() & (mm.ab != mm.team_alias)]
        print(f"   alias != abbrev(team_id): {len(d)} pairs, "
              f"{int(d.n.sum()) if len(d) else 0:,} rows"
              + (f"  e.g. {d.head(6).to_dict('records')}" if len(d) else ""))

# ============================================================== SURFACE 8
print("\n" + "-" * 100)
print("F. ratings_2k.team_slug  vs modern abbrevs")
print("-" * 100)
slug = vc("SELECT team_slug, COUNT(*) FROM ratings_2k GROUP BY 1")
print(f"ratings_2k: {sum(slug.values()):,} rows, {len(slug)} distinct slugs")
print(f"  sample: {sorted(slug)[:12]}")
add_surface("ratings_2k.team_slug", "ratings_2k", "nba_api abbreviations",
            slug, MODERN_AB, slug, {})

# ============================================================== SURFACE 9
print("\n" + "-" * 100)
print("G. data/arenas.csv  vs  nba_games.team_abbrev (RAW, era-aware)")
print("-" * 100)
ar = pd.read_csv(ROOT / "data" / "arenas.csv")
arset = set(ar.team.astype(str))
print(f"arenas.csv: {len(arset)} teams")
add_surface("data/arenas.csv:team", "arenas.csv",
            "nba_games.team_abbrev(RAW)", arset, ng_raw, {}, all_ng_ab)

# ============================================================== SURFACE 10
print("\n" + "-" * 100)
print("H. game_officials  — is there a team join at all?")
print("-" * 100)
go = con.execute("""SELECT COUNT(*) n, COUNT(DISTINCT game_id) g,
                           MIN(game_id), MAX(game_id) FROM game_officials""").fetchone()
gj = con.execute("""SELECT COUNT(DISTINCT o.game_id) FROM game_officials o
                     WHERE EXISTS (SELECT 1 FROM nba_games g
                                    WHERE g.game_id = o.game_id)""").fetchone()[0]
print(f"game_officials: {go[0]:,} rows over {go[1]:,} games; "
      f"{gj:,} of those game_ids exist in nba_games "
      f"({gj / go[1]:.2%}) — join is by GAME_ID, no team string. NOT A RISK.")

# ============================================================== SURFACE 11
print("\n" + "-" * 100)
print("I. FILE-LEVEL FEEDS — TeamRankings / ESPN / ActionNetwork / Kaggle")
print("-" * 100)
RAW = ROOT / "data" / "raw"
TR_TEAMS = {"BK": "BKN", "GS": "GSW", "NO": "NOP", "NY": "NYK",
            "PHO": "PHX", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}

# I.1 TeamRankings spread_movement.jsonl
trp = RAW / "teamrankings" / "spread_movement.jsonl"
if trp.exists():
    recs = [json.loads(x) for x in open(trp) if x.strip()]
    tr = pd.DataFrame(recs)
    tr_names = {}
    for c in ("fav_team", "dog_team", "home_team", "away_team", "team1", "team2"):
        if c in tr.columns:
            for k, v in tr[c].dropna().astype(str).value_counts().items():
                tr_names[k] = tr_names.get(k, 0) + int(v)
    mapped = {TR_TEAMS.get(k.upper(), k.upper()) for k in tr_names}
    cm = {}
    for k, v in tr_names.items():
        kk = TR_TEAMS.get(k.upper(), k.upper())
        cm[kk] = cm.get(kk, 0) + v
    print(f"teamrankings jsonl: {len(recs):,} records, {len(tr_names)} raw "
          f"team strings -> {len(mapped)} after TR_TEAMS")
    add_surface("raw/teamrankings/spread_movement.jsonl", "TeamRankings(mapped)",
                "nba_api abbreviations", mapped, MODERN_AB, cm, {},
                note="TR_TEAMS map is duplicated in build_odds_open.py + bo_lineshop.py")
else:
    print("teamrankings jsonl: ABSENT")

# I.2 ESPN + Action Network per-season csvs
extd = RAW / "sbr_ext"
if extd.exists():
    for pat, lbl, cols in ((("espn_nba_open_close_*.csv"), "ESPN",
                            ("home_abbr", "away_abbr")),
                           (("an_nba_open_close_*.csv"), "ActionNetwork",
                            ("home_abbr", "away_abbr"))):
        fs = sorted(extd.glob(pat))
        if not fs:
            print(f"{lbl}: no files")
            continue
        cnt = {}
        cols_seen = None
        for f in fs:
            d = pd.read_csv(f)
            cols_seen = list(d.columns)
            for c in cols:
                if c in d.columns:
                    for k, v in d[c].dropna().astype(str).value_counts().items():
                        cnt[k] = cnt.get(k, 0) + int(v)
            # also collect full names if present
        if not cnt:
            print(f"{lbl}: no abbr columns; cols={cols_seen}")
            continue
        ABBR = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
                "UTAH": "UTA", "WSH": "WAS"}
        m = {}
        for k, v in cnt.items():
            kk = ABBR.get(k.upper(), k.upper())
            m[kk] = m.get(kk, 0) + v
        print(f"{lbl}: {len(fs)} files, {sum(cnt.values()):,} team-slots, "
              f"{len(cnt)} raw codes -> {len(m)} mapped")
        add_surface(f"raw/sbr_ext {lbl} home_abbr/away_abbr", lbl,
                    "nba_api abbreviations", set(m), MODERN_AB, m, {},
                    note="ABBR map in build_nba_open_close.py (6 keys) vs "
                         "TR_TEAMS (8 keys) — different maps, same job")
        # RAW (unmapped) direction, the D119 shape
        add_surface(f"raw/sbr_ext {lbl} [NO ABBR map]", lbl,
                    "nba_api abbreviations", set(cnt), MODERN_AB, cnt, {})

# I.3 Kaggle odds source
kag = RAW / "kaggle"
if kag.exists():
    for f in sorted(kag.rglob("*.csv")):
        try:
            d = pd.read_csv(f, nrows=200000)
        except Exception:                                          # noqa: BLE001
            continue
        tc = [c for c in d.columns
              if c.lower() in ("home", "away", "team_home", "team_away",
                               "home_team", "away_team", "team", "team_abbreviation")]
        if not tc:
            continue
        cnt = {}
        for c in tc:
            for k, v in d[c].dropna().astype(str).value_counts().items():
                cnt[k] = cnt.get(k, 0) + int(v)
        if not cnt or len(cnt) > 200:
            continue
        TM = {"gs": "GSW", "no": "NOP", "ny": "NYK", "sa": "SAS",
              "phx": "PHX", "utah": "UTA", "wsh": "WAS"}
        m = {}
        for k, v in cnt.items():
            kk = TM.get(k.lower(), k.upper())
            m[kk] = m.get(kk, 0) + v
        print(f"kaggle {f.relative_to(kag)}: cols={tc}, {len(cnt)} raw codes")
        add_surface(f"raw/kaggle/{f.relative_to(kag)} {'+'.join(tc)}",
                    "kaggle", "nba_api abbreviations", set(m), MODERN_AB, m, {},
                    note="TEAM_MAP in nbapred/ingest/kaggle_odds.py")

# I.4 softbook / props feeds
for sub, lbl in (("softbook", "softbook"), ("props", "props"),
                 ("odds", "odds_logger")):
    p = RAW / sub
    if not p.exists():
        continue
    fs = list(p.rglob("*.json")) + list(p.rglob("*.jsonl")) + \
         list(p.rglob("*.csv"))
    print(f"raw/{sub}: {len(fs)} files")

con.close()

# ================================================================== OUTPUT
print("\n" + "=" * 100)
print("SURFACE SUMMARY")
print("=" * 100)
S = pd.DataFrame(SURFACES)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(S[["surface", "n_a", "n_b", "match_a_to_b", "match_b_to_a",
         "unresolved_a", "unresolved_b", "rows_unmatched"]].to_string(index=False))

print("\n" + "=" * 100)
print("EVERY UNRESOLVED VALUE, EVERY SURFACE")
print("=" * 100)
F = pd.DataFrame(FINDINGS)
if len(F):
    print(F.sort_values(["surface", "direction", "value"]).to_string(index=False))
else:
    print("(none)")

out = ROOT / "data"
S.to_csv(out / "d172_surfaces.csv", index=False)
F.to_csv(out / "d172_unresolved.csv", index=False)
print(f"\nWROTE {out/'d172_surfaces.csv'}  {out/'d172_unresolved.csv'}")
