"""D171 TASK 1 verification — (a) Clippers OUT rows now resolve,
(b) no other franchise has a silent mismatch, (c) the fix is confined to the
Clippers (no other team's out-set changes)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbapred.threads; nbapred.threads.pin(1)
from nbapred.db import connect
from nbapred.teams import abbrev_for, resolve_map
from nba_api.stats.static import teams as _t

con = connect(read_only=True, retry_s=60.0)
pdf = con.execute("""SELECT team, count(*) n, sum(CASE WHEN status='Out' THEN 1 ELSE 0 END) o,
   sum(CASE WHEN status='Out' AND report_date=game_date THEN 1 ELSE 0 END) osd
   FROM injury_reports_pit GROUP BY team""").fetchall()

print("="*88); print("VERIFY (a)/(b): every distinct PDF team string, AFTER the fix"); print("="*88)
bad = []
for team, n, o, osd in sorted(pdf, key=lambda r: -r[1]):
    ab = abbrev_for(team)
    if ab is None:
        bad.append((team, n, o, osd))
print("distinct pdf strings          :", len(pdf))
print("resolve after fix             :", len(pdf) - len(bad))
print("still unresolved              :", len(bad))
for t in bad: print("   %-32s rows=%-6d OUT=%-5d OUT@sameday=%-5d" % t)

# reverse direction, all 30
api = _t.get_teams()
pdf_names = {r[0] for r in pdf}
seen_ab = {abbrev_for(x) for x in pdf_names} - {None}
missing = [t for t in api if t["abbreviation"] not in seen_ab]
print("nba_api franchises with NO resolving PDF spelling:", len(missing),
      [t["abbreviation"] for t in missing])

# the specific claim
lac = [r for r in pdf if abbrev_for(r[0]) == "LAC"]
print("\nCLIPPERS: pdf spelling(s)=%s -> %s   rows=%d OUT=%d OUT@sameday=%d"
      % ([r[0] for r in lac], "LAC", sum(r[1] for r in lac),
         sum(r[2] for r in lac), sum(r[3] for r in lac)))

# (c) confinement: rebuild the out-map both ways and diff
from prod_by_season import report_out_map
new_map, rcov = report_out_map(con)
from nba_api.stats.static import teams as _t2
old_n2a = {t["full_name"]: t["abbreviation"] for t in _t2.get_teams()}
rows = con.execute("""
    SELECT i.game_date, i.team, p.player_id FROM injury_reports_pit i
    JOIN (SELECT player_id, lower(first_name||' '||last_name) fn FROM nba_players) p
      ON p.fn = lower(trim(split_part(i.player,',',2))||' '||trim(split_part(i.player,',',1)))
    WHERE i.status = 'Out' AND i.report_date = i.game_date""").fetchall()
old_map = {}
for gd, team, pid in rows:
    ab = old_n2a.get(team)
    if ab: old_map.setdefault((str(gd)[:10], ab), set()).add(int(pid))

keys = set(new_map) | set(old_map)
delta_by_ab = {}
for k in keys:
    d = len(new_map.get(k, set()) ^ old_map.get(k, set()))
    if d: delta_by_ab[k[1]] = delta_by_ab.get(k[1], 0) + d
print("\nVERIFY (c) CONFINEMENT — team-date cells whose OUT set changed, by team:")
for ab, d in sorted(delta_by_ab.items(), key=lambda kv: -kv[1]):
    print("   %-5s player-slots changed: %d" % (ab, d))
print("   teams affected:", sorted(delta_by_ab) or "NONE")
print("   new (date,team) cells:", len(set(new_map) - set(old_map)),
      " old cells lost:", len(set(old_map) - set(new_map)))
print("   total OUT player-slots: old=%d new=%d  (+%d)"
      % (sum(len(v) for v in old_map.values()), sum(len(v) for v in new_map.values()),
         sum(len(v) for v in new_map.values()) - sum(len(v) for v in old_map.values())))
con.close()
