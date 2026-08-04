"""IG probe E2: composition.py talent fallback `darko.get(pid, 0.0)`.

DPM=0.0 is LEAGUE AVERAGE, not replacement level — players missing from
darko_history as-of the cutoff (rookies pre-first-update, two-way callups) are
silently priced as average NBA players weighted by their trailing minutes.
Measure the minutes mass and count at cutoffs. Read-only.
"""
import sys, warnings, datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect
from nbapred.model.composition import CompositionModel

def main():
    con = connect(read_only=True)
    for season, y0 in (("2023-24", 2023), ("2024-25", 2024), ("2025-26", 2025)):
        for cut in (dt.date(y0, 11, 5), dt.date(y0 + 1, 2, 5)):
            comp = CompositionModel(con, before=cut)
            # active roster per production filter
            act = {p: d for p, d in comp.players.items()
                   if (cut - d["last_played"]).days <= 12}
            hist = {r[0] for r in con.execute(
                "SELECT DISTINCT player_id FROM darko_history WHERE date < ?", [cut]).fetchall()}
            miss = {p: d for p, d in act.items() if p not in hist}
            m_mass = sum(d["trail_min"] for d in miss.values())
            tot_mass = sum(d["trail_min"] for d in act.values())
            zero_t = sum(1 for d in act.values() if d["talent"] == 0.0)
            print(f" {season} @ {cut}: active={len(act)}  no-DARKO-history={len(miss)} "
                  f"({100*len(miss)/max(len(act),1):.1f}%)  their min-mass {m_mass:.0f} "
                  f"({100*m_mass/tot_mass:.1f}% of total)  talent==0.0 exactly: {zero_t}")
            big = sorted(miss.items(), key=lambda kv: -kv[1]["trail_min"])[:3]
            names = dict(con.execute("SELECT player_id, full_name FROM nba_players").fetchall())
            for p, d in big:
                print(f"    biggest missing: {names.get(p, p)} trail_min {d['trail_min']:.1f}")
    con.close()

if __name__ == "__main__":
    main()
