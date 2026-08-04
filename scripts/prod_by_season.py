"""Full-season walk-forward of the CURRENT production model (composition w=0.7 +
ratings + cold-start prior, weekly refit) — the numbers for the by-year chart
and for certification. DARKO-snapshot caveat applies (retro talent); flagged on
the chart.

AVAILABILITY CONSTRUCTION (D158; this is the thing that was wrong until
2026-08-03).  The DEFAULT out-set tier is now **T2 HONEST**: the official 5PM
injury report (`injury_reports_pit`, status='Out') UNION the official pregame
inactive list (`game_inactives`), each intersected with the 12-day roster
window — both PREGAME-PUBLIC sources per docs/LEAKAGE.md:127.  Where a feed
does not cover a season the out-set falls back to EMPTY (availability-blind,
honest-but-weaker) — NEVER to the played set.

Until D158 the default was the PLAYED-SET ORACLE (tonight's box score), which
docs/LEAKAGE.md:131 lists as forbidden.  It is still reachable, but only behind
ORACLE_PLAYED_OUTS=1, and only as a LABELLED CEILING measurement per
LEAKAGE.md:133 — it must never be the shipped/certified number again.

Env switches (availability tier, highest precedence first):
  ORACLE_PLAYED_OUTS=1  C1 CLAIRVOYANT — roster window minus who actually
                        played tonight.  LEAKAGE.  Ceiling measurement only.
  REPORT_OUTS=1         T1 — 5PM official injury report only.
  INACTIVE_OUTS=1       T2 — accepted alias of the default (no-op).
  (unset)               T2 HONEST — report UNION inactives.  <== production
Tier names follow D156 (data/bigplayer_notes.md, prereg sha256 fbcea42e...).
"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nbapred.threads; nbapred.threads.pin(1)   # shared box: 1 BLAS thread
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.production import fit_production
from nbapred.model.composition import CompositionModel, ROSTER_DAYS

import os
ORACLE_MINUTES = os.environ.get("ORACLE_MINUTES") == "1"  # PAID_ORACLES #2: perfect minutes
ORACLE_PLAYED_OUTS = os.environ.get("ORACLE_PLAYED_OUTS") == "1"  # LEAKAGE — ceiling only
INACTIVE_OUTS = os.environ.get("INACTIVE_OUTS") == "1"  # alias of the default T2
REPORT_OUTS = os.environ.get("REPORT_OUTS") == "1"      # T1: 5PM report only

if ORACLE_PLAYED_OUTS:
    AVAIL_TIER = "C1-ORACLE-PLAYED"
    AVAIL_DESC = ("CLAIRVOYANT played-set OUT sets (tonight's box score) — "
                  "LEAKAGE per docs/LEAKAGE.md:131, CEILING MEASUREMENT ONLY, "
                  "NOT CERTIFIABLE")
elif REPORT_OUTS:
    AVAIL_TIER = "T1-REPORT"
    AVAIL_DESC = "5PM official injury report only (pregame-public)"
else:
    AVAIL_TIER = "T2-HONEST"
    AVAIL_DESC = ("5PM official injury report UNION official pregame inactives, "
                  "roster window %dd; EMPTY where a feed does not cover the "
                  "season (never the played set)" % ROSTER_DAYS)


def report_out_map(con):
    """{(game_date, team_abbrev): {player_id}} from the official 5PM report,
    plus the set of game_dates the report actually covers (so a team with
    nobody Out is distinguishable from a season with no feed)."""
    from nba_api.stats.static import teams as _t
    name2ab = {t["full_name"]: t["abbreviation"] for t in _t.get_teams()}
    rows = con.execute("""
        SELECT i.game_date, i.team, p.player_id FROM injury_reports_pit i
        JOIN (SELECT player_id, lower(first_name||' '||last_name) fn FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '||trim(split_part(i.player,',',1)))
        WHERE i.status = 'Out' AND i.report_date = i.game_date
    """).fetchall()
    out = {}
    for gd, team, pid in rows:
        ab = name2ab.get(team)
        if ab:
            out.setdefault((str(gd)[:10], ab), set()).add(int(pid))
    covered = {str(d)[:10] for (d,) in con.execute(
        "SELECT DISTINCT game_date FROM injury_reports_pit "
        "WHERE report_date = game_date").fetchall()}
    return out, covered


def season_run(season):
    con=connect(read_only=True)
    # Both pregame-public feeds are ALWAYS loaded now: the honest tier is the
    # default, so the oracle branch is the special case, not this one.
    rout, rcov = report_out_map(con)
    inact = {}
    for g,p in con.execute("SELECT game_id, player_id FROM game_inactives").fetchall():
        inact.setdefault(g,set()).add(int(p))
    pm=con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    minmap={(r.game_id,int(r.player_id)):float(r.mins) for r in pm.itertuples()} if ORACLE_MINUTES else {}
    played={(g,t):set(grp.player_id) for (g,t),grp in pm.groupby(["game_id","team_id"])}
    meta=con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL ORDER BY game_date""",[season]).fetchdf()
    mkt={(str(r[0])[:10],r[1],r[2]):r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4])+1]).fetchall()}
    by={}; order=[]
    for x in meta.itertuples():
        if x.game_id not in by: order.append(x.game_id)
        by.setdefault(x.game_id,[]).append(x)
    # b2b flags (PIT: prior days' games are known before tip)
    tdates={}
    for x in meta.itertuples():
        d=x.game_date.date() if hasattr(x.game_date,"date") else x.game_date
        tdates.setdefault(x.team_id,set()).add(d)
    import datetime as _dt
    def b2b(tid,d): return (d-_dt.timedelta(days=1)) in tdates.get(tid,set())
    # dead-team flags (PIT: standings from games already played this season)
    from nbapred.model.production import DEAD_WPCT, DEAD_GP
    hist={}
    for x in meta.itertuples():
        d=x.game_date.date() if hasattr(x.game_date,"date") else x.game_date
        hist.setdefault(x.team_id,[]).append((d,x.wl=="W"))
    for t in hist: hist[t].sort()
    def dead(tid,d):
        past=[w for (dd,w) in hist.get(tid,[]) if dd<d]
        return len(past)>=DEAD_GP and (sum(past)/len(past))<DEAD_WPCT
    y,pp,pmk=[],[],[]
    cov={"report":0,"inactives":0,"either":0,"neither":0}  # feed coverage, scored games
    n_out=[]  # mean OUT players per team-game, the tier's own sanity check
    rows=[]   # per-game dump for residual autopsy
    model=comp=None; last=None
    for gid in order:
        recs=by[gid]
        if len(recs)!=2: continue
        m=recs[0].matchup; host=m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h=next((x for x in recs if x.team_abbrev==host),None); a=next((x for x in recs if x.team_abbrev!=host),None)
        if not h or not a: continue
        gd=h.game_date.date() if hasattr(h.game_date,"date") else h.game_date
        if last is None or (gd-last).days>=7:
            model=fit_production(con,season,before=gd,w_comp=0.7)
            comp=CompositionModel(con,before=gd); last=gd
        pmv=mkt.get((str(gd)[:10],h.team_abbrev,a.team_abbrev))
        if pmv is None: continue
        # availability OUT sets — D156 tier definitions, D158 honest default
        ds=str(gd)[:10]
        rot={t:{p for p,d0 in comp.players.items()
                if d0["team_id"]==t and (gd-d0["last_played"]).days<=ROSTER_DAYS}
             for t in (h.team_id,a.team_id)}
        has_r=ds in rcov; has_i=gid in inact
        cov["report"]+=int(has_r); cov["inactives"]+=int(has_i)
        cov["either"]+=int(has_r or has_i); cov["neither"]+=int(not (has_r or has_i))
        outs={}
        for t,ab_ in ((h.team_id,h.team_abbrev),(a.team_id,a.team_abbrev)):
            if ORACLE_PLAYED_OUTS:                      # C1 — LEAKAGE, ceiling only
                outs[t]=rot[t]-played.get((gid,t),set())
            else:
                o=rout.get((ds,ab_),set()) & rot[t]     # T1: 5PM report
                if not REPORT_OUTS:                     # T2: UNION official inactives
                    o=o | (inact.get(gid,set()) & rot[t])
                outs[t]=o                               # empty where no feed covers
        y.append(int(h.wl=="W"))
        # dead-team flags NOT passed: term failed the paired gate (ns, D47) —
        # infra kept for the post-injury-feed reconstruction
        if ORACLE_MINUTES:
            from nbapred.model.production import SCALE, sigmoid
            cm0=comp.margin(h.team_id,a.team_id,outs[h.team_id],outs[a.team_id],gd,home_edge=0.0)
            cm=0.0
            for t,sign in ((h.team_id,1),(a.team_id,-1)):
                for pid in played.get((gid,t),set()):
                    pi=comp.players.get(int(pid))
                    if pi: cm+=sign*pi["talent"]*minmap.get((gid,int(pid)),0.0)/48.0
            m=model.margin(h.team_id,a.team_id,outs[h.team_id],outs[a.team_id],gd,
                           b2b_home=b2b(h.team_id,gd),b2b_away=b2b(a.team_id,gd))
            pp.append(float(sigmoid((m-0.5*cm0+0.5*cm)/SCALE)))
        else:
            pp.append(model.p_home(h.team_id,a.team_id,outs[h.team_id],outs[a.team_id],gd,
                               b2b_home=b2b(h.team_id,gd),b2b_away=b2b(a.team_id,gd)))
        pmk.append(pmv)
        # D73 audit columns: gated tank diff + walk-forward k (term itself is
        # applied inside model.margin; nothing extra passed by this caller)
        tsd=model.tank_diff(h.team_id,a.team_id,gd)
        rows.append((season,gid,str(gd)[:10],h.team_abbrev,a.team_abbrev,
                     y[-1],float(pp[-1]),float(pmv),
                     len(outs[h.team_id]),len(outs[a.team_id]),
                     round(float(tsd),6),round(float(model.tank_k),4)))
        n_out.append(len(outs[h.team_id])); n_out.append(len(outs[a.team_id]))
    con.close()
    y=np.array(y)
    n=len(y)
    # per-season feed reality: which honest tier was ACTUALLY available here
    # NB: the coverage counters describe the PREGAME-PUBLIC feeds; on an oracle
    # run they are reported for reference but the active tier is the oracle.
    tier=(AVAIL_TIER if ORACLE_PLAYED_OUTS else
          "no-feed (EMPTY outs, availability-blind)" if cov["either"]==0 else
          ("full "+AVAIL_TIER if cov["either"]==n else "partial "+AVAIL_TIER))
    print(f"  {season}: n={n} report={cov['report']}/{n} inactives="
          f"{cov['inactives']}/{n} neither={cov['neither']}/{n} "
          f"mean_outs/team={np.mean(n_out) if n_out else 0:.2f}  -> {tier}",
          flush=True)
    return {"season":season,"n":n,"prod":round(log_loss(y,pp),4),
            "mkt":round(log_loss(y,np.array(pmk)),4),
            "coverage":cov,"tier":tier,
            "mean_outs_per_team":round(float(np.mean(n_out)) if n_out else 0.0,3),
            "rows":rows}

if __name__=="__main__":
    print("="*78)
    print(f"AVAILABILITY TIER: {AVAIL_TIER}")
    print(f"  {AVAIL_DESC}")
    if ORACLE_PLAYED_OUTS:
        print("  *** ORACLE RUN — THIS OUTPUT IS A CEILING MEASUREMENT AND MUST")
        print("  *** NOT BE CERTIFIED OR PUBLISHED AS A MODEL RESULT (D158).")
    print(f"  ORACLE_MINUTES={'1' if ORACLE_MINUTES else '0'}  "
          f"TANK_SEASON_FLOOR={os.environ.get('TANK_SEASON_FLOOR','(code default)')}")
    CERT="data/capstone_pergame.csv"
    dest=os.environ.get("CAPSTONE_OUT",CERT)
    if ORACLE_PLAYED_OUTS and dest==CERT:
        # D158 guardrail: the certified artifact is the ONE path a leaky
        # construction must never be able to occupy. Redirect, loudly.
        dest="data/capstone_pergame_oracle_ceiling.csv"
        print(f"  *** ORACLE RUN redirected away from {CERT} -> {dest}")
    print("="*78, flush=True)
    out=[season_run(s) for s in ("2021-22","2022-23","2023-24","2024-25","2025-26")]
    import csv
    with open(dest,"w",newline="") as f:
        wtr=csv.writer(f)
        wtr.writerow(["season","game_id","game_date","home","away","y",
                      "p_us","p_mkt","n_out_home","n_out_away","tsd","k"])
        for o in out:
            wtr.writerows(o.pop("rows"))
    for o in out: o["avail_tier"]=AVAIL_TIER
    print(f"[{AVAIL_TIER}] -> {dest}")
    print(out)
    json.dump(out, open(os.environ.get("PROD_JSON_OUT","data/prod_by_season.json"),"w"))
