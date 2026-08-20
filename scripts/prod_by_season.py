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
# D199: AS-OF-OPEN availability. The default T2 tier uses the SAME-DAY 5PM report
# UNION official pregame inactives. Both post-date the OPENING line, so any study
# that PRICES AT THE OPEN while using them is giving the model information the
# bettor did not have. Measured exposure: 19.3% of the minutes-weighted out-set
# is new on the day. This switch rebuilds the out-set from the most recent report
# STRICTLY BEFORE game day and drops the inactives union entirely.
OPEN_TIME_OUTS = os.environ.get("OPEN_TIME_OUTS") == "1"
# D201: SOFT availability. Instead of a hard OUT set, pass the composition leg
# {player_id: P(out tonight)} from data/p_out.csv.gz (walk-forward, PIT). A
# player last listed Questionable is out ~28.9% of the time; the hard rule scores
# him 0.0 and is wrong in BOTH directions (D200). Implies OPEN_TIME_OUTS.
# D226: PROMOTED TO PRODUCTION DEFAULT after gate D202 (season-clustered delta
# -0.002265, 95% CI [-0.0041,-0.0004] excluding zero, better 5/5 seasons,
# calibration veto passed). Set SOFT_AVAIL=0 to fall back to the hard out-set.
SOFT_AVAIL = os.environ.get("SOFT_AVAIL", "1") == "1"
COMPONENT_OUT = os.environ.get("COMPONENT_OUT", "")  # D230 channel dump


def _eout(o):
    """EXPECTED absences: sum of P(out) for a soft dict, |set| for a hard set."""
    return float(sum(o.values())) if isinstance(o, dict) else float(len(o))
if SOFT_AVAIL:
    OPEN_TIME_OUTS = True

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
    nobody Out is distinguishable from a season with no feed).

    D171 FIX — team names resolve through `nbapred.teams.abbrev_for`, which
    knows the PDFs spell the Clippers "LA Clippers" while nba_api's full_name
    is "Los Angeles Clippers".  Until this fix the inline
    `{full_name: abbreviation}` lookup returned None for that one string and
    the row was DISCARDED IN SILENCE: all 2,514 Clippers report rows (1,919
    same-day 'Out') had never entered a T1 or T2 out-set, in the certified
    seasons too (D170 §6).  Anything still unresolvable is now REPORTED with
    its row count rather than dropped quietly — this is the third instance of
    this bug class in the repo (D119, D161)."""
    from nbapred.teams import abbrev_for
    if OPEN_TIME_OUTS:
        # AS-OF-OPEN, BY CARRY-FORWARD. The archive holds same-day reports
        # (report_date = game_date, 93,110 rows) AND previous-evening reports
        # (report_date = game_date - 1, 32,594 rows). Using only the advance
        # rows is WRONG — most game-days have none, and outs/team collapses to
        # 0.3 against a true ~1.2. What a bettor actually knows at the open is
        # the LAST PUBLISHED STATUS carried forward, so: for each game date D,
        # take the out-set from the most recent report_date strictly < D.
        raw = con.execute("""
            SELECT i.report_date, i.team, p.player_id
            FROM injury_reports_pit i
            JOIN (SELECT player_id, lower(first_name||' '||last_name) fn
                  FROM nba_players) p
              ON p.fn = lower(trim(split_part(i.player,',',2))||' '
                              ||trim(split_part(i.player,',',1)))
            WHERE i.status = 'Out'
        """).fetchall()
        by_rd = {}
        for rd, team, pid in raw:
            by_rd.setdefault(str(rd)[:10], []).append((team, int(pid)))
        rep_dates = sorted(by_rd)
        game_dates = sorted({str(d)[:10] for (d,) in con.execute(
            "SELECT DISTINCT game_date FROM nba_games "
            "WHERE game_id LIKE '002%'").fetchall()})
        import bisect
        rows = []
        for gd in game_dates:
            j = bisect.bisect_left(rep_dates, gd) - 1   # latest report < gd
            if j < 0:
                continue
            for team, pid in by_rd[rep_dates[j]]:
                rows.append((gd, team, pid))
    else:
        rows = con.execute("""
            SELECT i.game_date, i.team, p.player_id FROM injury_reports_pit i
            JOIN (SELECT player_id, lower(first_name||' '||last_name) fn
                  FROM nba_players) p
              ON p.fn = lower(trim(split_part(i.player,',',2))||' '
                              ||trim(split_part(i.player,',',1)))
            WHERE i.status = 'Out' AND i.report_date = i.game_date
        """).fetchall()
    out = {}
    unresolved = {}
    for gd, team, pid in rows:
        ab = abbrev_for(team)
        if ab:
            out.setdefault((str(gd)[:10], ab), set()).add(int(pid))
        else:
            unresolved[team] = unresolved.get(team, 0) + 1
    if unresolved:
        top = sorted(unresolved.items(), key=lambda kv: -kv[1])[:5]
        print("  report_out_map: %d row(s) on %d unresolved team string(s) "
              "DROPPED: %s" % (sum(unresolved.values()), len(unresolved),
                               ", ".join("%r x%d" % t for t in top)), flush=True)
    covered = ({g for (g, _t, _p) in rows} if OPEN_TIME_OUTS else
               {str(d)[:10] for (d,) in con.execute(
                   "SELECT DISTINCT game_date FROM injury_reports_pit "
                   "WHERE report_date = game_date").fetchall()})
    return out, covered


def _load_pout():
    """{game_date: {player_id: p_out}} from the D201 walk-forward artifact."""
    import csv, gzip
    # D240: POUT_ARTIFACT swaps in an alternative availability artifact so a
    # challenger can be scored through the FULL stack. Default is unchanged.
    name = os.environ.get("POUT_ARTIFACT", "p_out.csv.gz")
    f = REPO / "data" / name if "REPO" in globals() else \
        Path(__file__).resolve().parent.parent / "data" / name
    m = {}
    if not f.exists():
        raise SystemExit(f"SOFT_AVAIL=1 but {f} is missing; run "
                         f"scripts/d201_pout_artifact.py first")
    with gzip.open(f, "rt") as fh:
        for row in csv.DictReader(fh):
            m.setdefault(row["game_date"], {})[int(row["player_id"])] = \
                float(row["p_out"])
    return m


_pout = _load_pout() if SOFT_AVAIL else {}


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
    crows=[]  # D230: per-game CHANNEL dump (gated by COMPONENT_OUT)
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
        if SOFT_AVAIL:
            pm = _pout.get(ds, {})
            for t in (h.team_id, a.team_id):
                # every rostered player carries a probability; absent from the
                # artifact means "never on a report" => available
                outs[t] = {pid: pm[pid] for pid in rot[t] if pid in pm}
        else:
         for t,ab_ in ((h.team_id,h.team_abbrev),(a.team_id,a.team_abbrev)):
            if ORACLE_PLAYED_OUTS:                      # C1 — LEAKAGE, ceiling only
                outs[t]=rot[t]-played.get((gid,t),set())
            else:
                o=rout.get((ds,ab_),set()) & rot[t]     # T1: 5PM report
                if not REPORT_OUTS and not OPEN_TIME_OUTS:  # T2: UNION official inactives
                    # OPEN_TIME_OUTS never unions inactives: those are released
                    # ~30 min pre-tip and cannot inform a bet at the open.
                    o=o | (inact.get(gid,set()) & rot[t])
                outs[t]=o                               # empty where no feed covers
        # n_out is a HEADCOUNT: under SOFT_AVAIL `outs[t]` is a dict over every
        # rostered player carrying ANY out-probability, so len() counts players
        # under doubt, NOT the expected number absent (D232 — the previous
        # comment here claimed otherwise and was wrong: len != sum). Both are
        # emitted now; `eo_*` is the expected count, sum of P(out).
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
        if COMPONENT_OUT:
            # D230: the five channels the margin is the SUM of. Gated, so the
            # certified path takes no extra model call by default; when it is
            # on, the sum is asserted against margin() on EVERY game, which is
            # what makes the dump trustworthy as a decomposition rather than a
            # parallel re-derivation that might drift.
            _kw=dict(b2b_home=b2b(h.team_id,gd),b2b_away=b2b(a.team_id,gd))
            cc=model.margin_components(h.team_id,a.team_id,outs[h.team_id],
                                       outs[a.team_id],gd,**_kw)
            mt=model.margin(h.team_id,a.team_id,outs[h.team_id],outs[a.team_id],
                            gd,**_kw)
            assert abs(sum(cc.values())-mt)<1e-12,(gid,cc,mt)
            crows.append((season,gid,str(gd)[:10],h.team_abbrev,a.team_abbrev,
                          y[-1],cc["ff"],cc["comp"],cc["sched"],cc["tank"],
                          cc["late"],mt))
        rows.append((season,gid,str(gd)[:10],h.team_abbrev,a.team_abbrev,
                     y[-1],float(pp[-1]),float(pmv),
                     len(outs[h.team_id]),len(outs[a.team_id]),
                     round(_eout(outs[h.team_id]),6),
                     round(_eout(outs[a.team_id]),6),
                     round(float(tsd),6),round(float(model.tank_k),4)))
        for _t in (h.team_id, a.team_id):
            _o = outs[_t]
            n_out.append(sum(_o.values()) if isinstance(_o, dict) else len(_o))
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
            "rows":rows,"crows":crows}

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
    # D230: the certified span is the default and is unchanged. PROD_SEASONS
    # widens it (the component work needs 2019-20 onward, the first fully
    # injury-report-covered season) without forking this script.
    CERT_SEASONS=("2021-22","2022-23","2023-24","2024-25","2025-26")
    seasons=tuple(os.environ.get("PROD_SEASONS","").split(",")) \
        if os.environ.get("PROD_SEASONS") else CERT_SEASONS
    out=[season_run(s) for s in seasons]
    import csv
    with open(dest,"w",newline="") as f:
        wtr=csv.writer(f)
        wtr.writerow(["season","game_id","game_date","home","away","y",
                      "p_us","p_mkt","n_out_home","n_out_away",
                      "eo_home","eo_away","tsd","k"])
        for o in out:
            wtr.writerows(o.pop("rows"))
    if COMPONENT_OUT:
        with open(COMPONENT_OUT,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["season","game_id","game_date","home","away","y",
                        "m_ff","m_comp","m_sched","m_tank","m_late","m_total"])
            for o in out: w.writerows(o["crows"])
        print(f"[D230] channel dump -> {COMPONENT_OUT}")
    for o in out: o.pop("crows",None)
    for o in out: o["avail_tier"]=AVAIL_TIER
    print(f"[{AVAIL_TIER}] -> {dest}")
    print(out)
    json.dump(out, open(os.environ.get("PROD_JSON_OUT","data/prod_by_season.json"),"w"))
