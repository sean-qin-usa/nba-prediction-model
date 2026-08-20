"""Parse captured Underdog lines -> (player_id, stat, line) rows, ready for the
day-one H-B comparison when NBA lines appear. Title format: '<Name> <Stat> O/U'.
Name matching reuses the crosswalk normalizer (accents/suffixes handled)."""
from __future__ import annotations
import json, re
from ..ids import norm_name

STAT_MAP = {  # underdog display stat -> our sim output key
    "points": "points", "pts": "points",
    "rebounds": "rebounds", "reb": "rebounds",
    "assists": "assists", "ast": "assists",
    "3-pointers made": "threes", "threes made": "threes", "3pm": "threes",
    "pts + reb + ast": "pra", "pts+reb+ast": "pra",
}

def build_name_index(con):
    rows = con.execute("SELECT player_id, full_name FROM nba_players WHERE is_active").fetchall()
    return {norm_name(n): int(p) for p, n in rows}

def parse_lines(jsonl_path: str, name_index: dict):
    """Yield dicts for NBA-parseable lines in one capture file."""
    out = []
    for raw in open(jsonl_path):
        rec = json.loads(raw)
        if rec.get("kind") != "underdog":
            continue
        for l in rec["data"].get("lines", []):
            title = (l.get("title") or "")
            stat = (l.get("appearance_stat") or "").lower().strip()
            key = STAT_MAP.get(stat)
            if not key:
                continue
            # name = title minus trailing stat phrase / 'O/U'
            nm = re.sub(r"\s+O/U\s*$", "", title, flags=re.I)
            nm = nm.replace(l.get("appearance_stat") or "", "").strip()
            pid = name_index.get(norm_name(nm))
            if pid is None:
                # try first 2-3 tokens (titles sometimes append qualifiers)
                toks = nm.split()
                for k in (3, 2):
                    pid = name_index.get(norm_name(" ".join(toks[:k])))
                    if pid:
                        break
            if pid:
                out.append(dict(ts=rec["ts"], player_id=pid, stat=key,
                                line=l.get("stat_value"), options=l.get("options")))
    return out
