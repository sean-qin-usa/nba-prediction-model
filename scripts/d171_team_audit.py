"""D171 TASK 1 — audit the PDF-spelling <-> nba_api-spelling mapping that
`report_out_map()` relies on, in BOTH directions, for all 30 franchises.

Read-only. Never writes the DB.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nbapred.threads; nbapred.threads.pin(1)
from nbapred.db import connect
from nba_api.stats.static import teams as _t

con = connect(read_only=True, retry_s=60.0)

# --- side A: every distinct team string the injury PDFs actually produce -----
pdf = con.execute("""
    SELECT team, count(*) n_rows,
           sum(CASE WHEN status='Out' THEN 1 ELSE 0 END) n_out,
           sum(CASE WHEN status='Out' AND report_date=game_date THEN 1 ELSE 0 END) n_out_sameday,
           min(game_date) d0, max(game_date) d1
    FROM injury_reports_pit GROUP BY team ORDER BY n_rows DESC
""").fetchall()

# --- side B: nba_api's full_name universe, which is what the map keys on -----
api = _t.get_teams()
name2ab = {t["full_name"]: t["abbreviation"] for t in api}
ab2name = {t["abbreviation"]: t["full_name"] for t in api}

# --- also: the abbrevs nba_games actually uses (the join target) -------------
game_ab = {r[0] for r in con.execute(
    "SELECT DISTINCT team_abbrev FROM nba_games WHERE game_id LIKE '002%'").fetchall()}

print("="*100)
print("A. DISTINCT `team` STRINGS IN injury_reports_pit  (n=%d)" % len(pdf))
print("="*100)
print(f"{'pdf team string':<34}{'rows':>8}{'OUT':>8}{'OUTsd':>8}  {'->abbrev':<10}{'MATCH?':<8}")
matched_rows = matched_out = dropped_rows = dropped_out = dropped_out_sd = 0
unmatched = []
for team, n_rows, n_out, n_out_sd, d0, d1 in pdf:
    ab = name2ab.get(team)
    ok = ab is not None
    if ok:
        matched_rows += n_rows; matched_out += n_out
    else:
        dropped_rows += n_rows; dropped_out += n_out; dropped_out_sd += (n_out_sd or 0)
        unmatched.append((team, n_rows, n_out, n_out_sd, str(d0), str(d1)))
    print(f"{team[:33]:<34}{n_rows:>8}{n_out:>8}{n_out_sd or 0:>8}  {str(ab or '-'):<10}{'ok' if ok else 'DROPPED':<8}")

print()
print("="*100)
print("B. REVERSE DIRECTION — every nba_api full_name, is it ever produced by a PDF?")
print("="*100)
pdf_names = {r[0] for r in pdf}
print(f"{'nba_api full_name':<28}{'ab':<5}{'appears in PDFs?':<20}{'pdf spelling used'}")
rev_missing = []
for t in sorted(api, key=lambda x: x["full_name"]):
    fn, ab = t["full_name"], t["abbreviation"]
    hit = fn in pdf_names
    # what DOES the pdf call this franchise? try to find it by nickname
    nick = t["nickname"]
    cand = sorted(x for x in pdf_names if x.endswith(nick))
    if not hit:
        rev_missing.append((fn, ab, cand))
    print(f"{fn:<28}{ab:<5}{('YES' if hit else 'NO  <== MISMATCH'):<20}{','.join(cand) if cand else '(none)'}")

print()
print("="*100)
print("C. SUMMARY")
print("="*100)
print("distinct pdf team strings          :", len(pdf))
print("pdf strings that map               :", len(pdf) - len(unmatched))
print("pdf strings that SILENTLY DROP     :", len(unmatched))
for u in unmatched:
    print("   DROP  %-30s rows=%-7d OUT=%-6d OUT@sameday=%-6d  %s..%s" % u)
print("rows kept / dropped                : %d / %d" % (matched_rows, dropped_rows))
print("OUT rows kept / dropped            : %d / %d" % (matched_out, dropped_out))
print("OUT@report_date=game_date dropped  : %d   <-- these are the T1/T2 losses" % dropped_out_sd)
print("nba_api names never seen in PDFs   :", len(rev_missing))
for fn, ab, cand in rev_missing:
    print("   REV   %-28s %-5s pdf uses: %s" % (fn, ab, cand or "(nothing)"))
print("nba_games abbrevs not in nba_api   :", sorted(game_ab - set(ab2name)))
print("nba_api abbrevs not in nba_games   :", sorted(set(ab2name) - game_ab))

json.dump({"unmatched": unmatched, "rev_missing": [(a,b,c) for a,b,c in rev_missing],
           "dropped_rows": dropped_rows, "dropped_out": dropped_out,
           "dropped_out_sameday": dropped_out_sd,
           "matched_rows": matched_rows, "matched_out": matched_out,
           "n_pdf_strings": len(pdf),
           "game_ab_extra": sorted(game_ab - set(ab2name)),
           "api_ab_extra": sorted(set(ab2name) - game_ab)},
          open("data/d171_team_audit.json","w"), indent=1)
con.close()
