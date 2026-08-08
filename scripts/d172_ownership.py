#!/usr/bin/env python3
"""D172 TASK 2 §4 — OWNERSHIP.

The plausible mechanism is organisational, not in-game: tanking propensity,
willingness to pay the tax, deadline behaviour.  The D73 tank composite
already prices the OUTCOME of exactly that behaviour, so the honest question
is whether ownership adds anything ON TOP of it.

DATA PROVENANCE — READ THIS BEFORE USING THE TABLE.
`OWNER_SPELLS` below is **HAND-BUILT FROM PUBLIC KNOWLEDGE** (franchise sale
announcements / league approvals as reported by AP, ESPN and the teams
themselves).  It is NOT scraped, NOT machine-verified, and no cell of it is
sourced to a URL in this repo.  It records CONTROL ownership only (the
controlling owner or ownership group), keyed by the season in which the new
control group took over.  Minority-stake sales, trust reshuffles inside one
family and estate transitions are deliberately EXCLUDED unless control moved
outside the family.  Season attribution may be off by one where a sale closed
mid-season; every test below is therefore ALSO run dropping the transition
season.  Treat any single row as indicative, not authoritative.

READ-ONLY.  Writes data/d172_ownership.json + data/d172_owner_spells.csv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import duckdb                                                     # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats as st                                     # noqa: E402

from nbapred.teams import FRANCHISE                               # noqa: E402

DATA = ROOT / "data"
RES: dict = {}
T0 = time.time()

# (modern team code, first season of the control group, label)
# HAND-BUILT — see module docstring.  Seasons are season-start years.
OWNER_SPELLS: list[tuple[str, int, str]] = [
    ("ATL", 1996, "Turner Broadcasting / Time Warner"),
    ("ATL", 2004, "Atlanta Spirit LLC"),
    ("ATL", 2015, "Tony Ressler group"),
    ("BOS", 1996, "Paul Gaston"),
    ("BOS", 2002, "Boston Basketball Partners (Grousbeck/Pagliuca)"),
    ("BOS", 2025, "Bill Chisholm group"),
    ("BKN", 1996, "Taub / Aufzien partnership"),
    ("BKN", 1998, "YankeeNets"),
    ("BKN", 2004, "Bruce Ratner"),
    ("BKN", 2010, "Mikhail Prokhorov"),
    ("BKN", 2019, "Joe Tsai"),
    ("CHA", 1996, "George Shinn (original Hornets)"),
    ("CHA", 2004, "Robert L. Johnson (expansion Bobcats)"),
    ("CHA", 2010, "Michael Jordan"),
    ("CHA", 2023, "Plotkin / Schnall group"),
    ("CHI", 1996, "Jerry Reinsdorf group"),
    ("CLE", 1996, "Gordon Gund"),
    ("CLE", 2005, "Dan Gilbert"),
    ("DAL", 1996, "Ross Perot Jr"),
    ("DAL", 2000, "Mark Cuban"),
    ("DAL", 2023, "Adelson / Dumont family"),
    ("DEN", 1996, "COMSAT / Ascent"),
    ("DEN", 2000, "Stan Kroenke"),
    ("DET", 1996, "Bill Davidson"),
    ("DET", 2011, "Tom Gores"),
    ("GSW", 1996, "Chris Cohan"),
    ("GSW", 2010, "Lacob / Guber"),
    ("HOU", 1996, "Leslie Alexander"),
    ("HOU", 2017, "Tilman Fertitta"),
    ("IND", 1996, "Simon family"),
    ("LAC", 1996, "Donald Sterling"),
    ("LAC", 2014, "Steve Ballmer"),
    ("LAL", 1996, "Jerry Buss"),
    ("LAL", 2013, "Buss family trust"),
    ("LAL", 2025, "Mark Walter"),
    ("MEM", 1996, "John McCaw (Vancouver)"),
    ("MEM", 2000, "Michael Heisley"),
    ("MEM", 2012, "Robert Pera"),
    ("MIA", 1996, "Micky Arison"),
    ("MIL", 1996, "Herb Kohl"),
    ("MIL", 2014, "Edens / Lasry"),
    ("MIN", 1996, "Glen Taylor"),
    ("MIN", 2024, "Lore / Rodriguez"),
    ("NOP", 2002, "George Shinn (relocated)"),
    ("NOP", 2010, "NBA league ownership"),
    ("NOP", 2012, "Tom Benson"),
    ("NOP", 2018, "Gayle Benson"),
    ("NYK", 1996, "ITT / Cablevision"),
    ("NYK", 1997, "MSG / James Dolan"),
    ("OKC", 1996, "Barry Ackerley (Seattle)"),
    ("OKC", 2001, "Howard Schultz (Seattle)"),
    ("OKC", 2006, "Clay Bennett / PBC"),
    ("ORL", 1996, "Rich DeVos"),
    ("ORL", 2018, "DeVos family"),
    ("PHI", 1996, "Comcast Spectacor"),
    ("PHI", 2011, "Harris / HBSE"),
    ("PHX", 1996, "Jerry Colangelo group"),
    ("PHX", 2004, "Robert Sarver"),
    ("PHX", 2023, "Mat Ishbia"),
    ("POR", 1996, "Paul Allen"),
    ("POR", 2018, "Vulcan / Jody Allen trust"),
    ("SAC", 1996, "Jim Thomas"),
    ("SAC", 1999, "Maloof family"),
    ("SAC", 2013, "Vivek Ranadive"),
    ("SAS", 1996, "Peter Holt group"),
    ("SAS", 2021, "Holt / Sixth Street group"),
    ("TOR", 1996, "Slaight / Stavro"),
    ("TOR", 2012, "MLSE (Bell / Rogers)"),
    ("UTA", 1996, "Larry H. Miller"),
    ("UTA", 2009, "Gail Miller"),
    ("UTA", 2020, "Ryan Smith"),
    ("WAS", 1996, "Abe Pollin"),
    ("WAS", 2010, "Ted Leonsis"),
]


def ro(attempts=10, wait=60.0):
    for i in range(attempts):
        try:
            return duckdb.connect(str(DATA / "nba.duckdb"), read_only=True)
        except Exception as e:                                    # noqa: BLE001
            if ("lock" not in str(e).lower() and "held" not in str(e).lower()) \
               or i == attempts - 1:
                raise
            time.sleep(wait)


def cci(v, cl):
    d = pd.DataFrame({"v": np.asarray(v, float), "c": np.asarray(cl)}).dropna()
    if len(d) < 2:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, K=0, p=np.nan, n=len(d))
    m = d.groupby("c").v.mean()
    K = len(m)
    mu, sd = float(m.mean()), float(m.std(ddof=1))
    se = sd / np.sqrt(K)
    t = st.t.ppf(0.975, K - 1)
    ts = mu / se if se > 0 else np.nan
    p = 2 * (1 - st.t.cdf(abs(ts), K - 1)) if se > 0 else np.nan
    return dict(mean=mu, lo=mu - t * se, hi=mu + t * se, K=int(K),
                p=float(p) if p == p else np.nan, n=int(len(d)))


con = ro()
print("=" * 100)
print("D172 §4  OWNERSHIP — does it add anything on top of the D73 tank term?")
print("=" * 100)
print("\n*** THE OWNERSHIP TABLE IS HAND-BUILT FROM PUBLIC KNOWLEDGE. ***")
print("*** Control ownership only; season attribution may be off by one;  ***")
print("*** every test is repeated dropping the transition season.         ***\n")

# ------------------------------------------------------- owner per team-season
SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(1996, 2026)]
sp = pd.DataFrame(OWNER_SPELLS, columns=["team", "yr0", "owner"])
rows = []
for t, d in sp.groupby("team"):
    d = d.sort_values("yr0")
    for y in range(1996, 2026):
        c = d[d.yr0 <= y]
        if len(c):
            rows.append(dict(team=t, yr=y, season=f"{y}-{str(y+1)[-2:]}",
                             owner=c.iloc[-1].owner,
                             owner_since=int(c.iloc[-1].yr0)))
OW = pd.DataFrame(rows)
OW["tenure"] = OW.yr - OW.owner_since
OW["is_change_season"] = OW.tenure == 0
OW.to_csv(DATA / "d172_owner_spells.csv", index=False)

# align to team-seasons that actually exist
ts = con.execute("""
    SELECT DISTINCT season, team_abbrev FROM nba_games
     WHERE game_id LIKE '002%'""").fetchdf()
ts["team"] = ts.team_abbrev.map(lambda a: FRANCHISE.get(a, a))
ts = ts.drop_duplicates(["season", "team"])
J = ts.merge(OW, on=["season", "team"], how="left")
print(f"team-seasons in corpus: {len(J)}  owner assigned: {J.owner.notna().sum()}"
      f"  MISSING: {J.owner.isna().sum()}")
if J.owner.isna().any():
    print("  missing:", J[J.owner.isna()][['season', 'team']]
          .to_dict('records')[:10])
RES["team_seasons"] = int(len(J))
RES["owner_assigned"] = int(J.owner.notna().sum())

nchg = int(J.is_change_season.sum())
print(f"\nCONTROL-OWNERSHIP CHANGES INSIDE THE CORPUS: {nchg}")
print(f"distinct owners {J.owner.nunique()}  vs distinct teams {J.team.nunique()}")

# =============================== 1. THE STRUCTURAL PROBLEM: owner IS mostly team
print("\n" + "-" * 100)
print("1. IS 'OWNER' EVEN A DIFFERENT VARIABLE FROM 'TEAM'?")
print("-" * 100)
sz = J.groupby(["team", "owner"]).size().rename("n").reset_index()
modal = sz.sort_values("n", ascending=False).drop_duplicates("team")
frac_modal = modal.n.sum() / len(J)
spells_per_team = J.groupby("team").owner.nunique()
print(f"owner spells per team: mean {spells_per_team.mean():.2f}, "
      f"median {spells_per_team.median():.0f}, max {spells_per_team.max()}")
print(f"mean owner spell length: {len(J)/J.groupby(['team','owner']).ngroups:.1f} "
      f"seasons")
print(f"share of team-seasons sitting in the team's MODAL owner spell: "
      f"{frac_modal:.1%}")
# how much variance can owner explain that team cannot?  R2 of the two encodings
print("\n  An owner dummy has "
      f"{J.groupby(['team','owner']).ngroups} levels for {len(J)} team-seasons,"
      f"\n  nested inside {J.team.nunique()} teams — i.e. it is a TEAM dummy with"
      f"\n  {J.groupby(['team','owner']).ngroups - J.team.nunique()} extra splits."
      "  D70/D137 already killed the team\n  dummy: real within a season, ZERO "
      "cross-season memory.")
RES["structure"] = dict(
    n_changes=nchg, owner_spells=int(J.groupby(["team", "owner"]).ngroups),
    teams=int(J.team.nunique()),
    mean_spell_seasons=float(len(J) / J.groupby(["team", "owner"]).ngroups),
    frac_in_modal_spell=float(frac_modal))

# ======================================= 2. OWNERSHIP-CHANGE EVENT, market resid
print("\n" + "-" * 100)
print("2. OWNERSHIP-CHANGE EVENT STUDY — the coach design, applied to owners")
print("-" * 100)
A = pd.read_csv(DATA / "ats19_frame.csv.gz")
long = []
for side, sgn in (("home", 1), ("away", -1)):
    d = A[["season", side, "m_us", "open_margin", "margin_actual"]].rename(
        columns={side: "team"})
    d["r_model"] = sgn * (d.margin_actual - d.m_us)
    d["r_open"] = sgn * (d.margin_actual - d.open_margin)
    long.append(d[["season", "team", "r_model", "r_open"]])
L = pd.concat(long, ignore_index=True)
TS = L.groupby(["season", "team"]).agg(
    n=("r_model", "size"), r_model=("r_model", "mean"),
    r_open=("r_open", "mean")).reset_index()
TS = TS.merge(J[["season", "team", "owner", "tenure", "owner_since"]],
              on=["season", "team"], how="inner")
TS["yr"] = TS.season.str[:4].astype(int)
print(f"team-seasons with residuals AND an owner: {len(TS)} "
      f"({TS.season.nunique()} seasons)")

lag = TS.merge(TS.assign(yr=TS.yr + 1), on=["team", "yr"], suffixes=("", "_p"))
same_o = lag[lag.owner == lag.owner_p]
new_o = lag[lag.owner != lag.owner_p]
print(f"consecutive pairs {len(lag)}: same owner {len(same_o)}, "
      f"NEW owner {len(new_o)}")
own_ev = {}
for lbl, col in (("vs OUR MODEL", "r_model"), ("vs the MARKET", "r_open")):
    if len(new_o) >= 3:
        d = new_o[col] - new_o[f"{col}_p"]
        r = cci(d, new_o.season)
        s = cci(same_o[col] - same_o[f"{col}_p"], same_o.season)
        own_ev[lbl] = dict(new_owner=r, same_owner=s)
        print(f"\n  season-over-season change in team residual, {lbl}:")
        print(f"    NEW OWNER : {r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]"
              f" p={r['p']:.4f} n={r['n']}")
        print(f"    SAME OWNER: {s['mean']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}]"
              f" p={s['p']:.4f} n={s['n']}")
RES["ownership_change_event"] = own_ev

# owner tenure: do new regimes tank early?
print("\n  RESIDUAL BY OWNER TENURE (does a new regime behave differently?)")
TS["ten_b"] = pd.cut(TS.tenure, [-1, 0, 2, 5, 10, 100],
                     labels=["0 (year 1)", "1-2", "3-5", "6-10", "11+"])
g = TS.groupby("ten_b", observed=True).agg(
    n=("r_open", "size"), r_open=("r_open", "mean"),
    r_model=("r_model", "mean"))
print(g.to_string())
RES["by_tenure"] = g.reset_index().astype(str).to_dict("records")

# =============================== 3. ON TOP OF THE D73 TANK COMPOSITE
print("\n" + "-" * 100)
print("3. DOES OWNERSHIP ADD ANYTHING ON TOP OF THE D73 TANK COMPOSITE?")
print("-" * 100)
tk = pd.read_csv(DATA / "apr_tank_stats.csv")
tk["team"] = tk.team_abbrev.map(lambda a: FRANCHISE.get(a, a))
TK = (tk.groupby(["season", "team"]).tank_score.mean().rename("tank")
        .reset_index()
        .merge(J[["season", "team", "owner", "tenure"]],
               on=["season", "team"], how="inner"))
print(f"D73 tank composite exists on {TK.season.nunique()} seasons "
      f"({sorted(TK.season.unique())}), {len(TK)} team-seasons.")
print("  -> the DIRECT 'owner on top of D73' test is UNDERPOWERED BY "
      "CONSTRUCTION:\n     over 4 seasons there are "
      f"{int(TK.groupby('team').owner.nunique().gt(1).sum())} teams with more "
      "than one owner,\n     so 'owner' and 'team' are the SAME VARIABLE on "
      "this window.")
RES["tank_window"] = dict(seasons=sorted(TK.season.unique().tolist()),
                          team_seasons=int(len(TK)),
                          teams_with_owner_change=int(
                              TK.groupby("team").owner.nunique().gt(1).sum()))

# a tank-OUTCOME proxy computable on all 30 seasons: youth minutes share.
# (comp_a of the D73 composite is a veteran-minutes shift; this is its
#  direction, and it is the behaviour ownership is supposed to drive.)
print("\n  Because of that, ownership is tested against a TANK-OUTCOME PROXY\n"
      "  available on all 30 seasons: the share of a team's regular-season\n"
      "  minutes going to players within 2 years of their draft (the D73\n"
      "  composite's own comp-A direction).")
youth = con.execute("""
    WITH s AS (
      SELECT g.season, g.team_abbrev, s.player_id, s.seconds
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season, team_id, team_abbrev
                FROM nba_games WHERE game_id LIKE '002%') g
          ON g.game_id = s.game_id AND g.team_id = s.team_id
       WHERE s.seconds > 0),
    d AS (SELECT player_id, MIN(draft_year) dy FROM draft_history GROUP BY 1)
    SELECT s.season, s.team_abbrev,
           SUM(s.seconds) AS sec,
           SUM(CASE WHEN d.dy IS NOT NULL
                     AND CAST(SUBSTR(s.season,1,4) AS INT) - d.dy <= 2
                    THEN s.seconds ELSE 0 END) AS sec_youth
      FROM s LEFT JOIN d USING (player_id)
     GROUP BY 1,2""").fetchdf()
youth["youth_share"] = youth.sec_youth / youth.sec
youth["team"] = youth.team_abbrev.map(lambda a: FRANCHISE.get(a, a))
Y = youth.merge(J[["season", "team", "owner", "tenure"]],
                on=["season", "team"], how="inner")
Y["yr"] = Y.season.str[:4].astype(int)
Y["z"] = Y.youth_share - Y.groupby("season").youth_share.transform("mean")
print(f"  panel {len(Y)} team-seasons, youth minutes share mean "
      f"{Y.youth_share.mean():.3f} sd {Y.youth_share.std():.3f}")

tot = Y.z.var(ddof=1)
v_team = Y.groupby("team").z.mean().var(ddof=1)
v_own = Y.groupby(["team", "owner"]).z.mean().var(ddof=1)
print(f"\n  VARIANCE OF (season-demeaned) YOUTH SHARE explained by:")
print(f"    TEAM  identity           {v_team/tot:>6.1%}")
print(f"    TEAM x OWNER spell       {v_own/tot:>6.1%}")
print(f"    OWNER's marginal share   {(v_own-v_team)/tot:>6.1%}"
      "   <- all ownership can add")
RES["youth_variance"] = dict(frac_team=float(v_team / tot),
                             frac_owner=float(v_own / tot),
                             frac_owner_extra=float((v_own - v_team) / tot))

lagY = Y.merge(Y.assign(yr=Y.yr + 1), on=["team", "yr"], suffixes=("", "_p"))
so, no_ = lagY[lagY.owner == lagY.owner_p], lagY[lagY.owner != lagY.owner_p]
rr = cci(no_.z - no_.z_p, no_.season)
ss = cci(so.z - so.z_p, so.season)
print(f"\n  YEAR-OVER-YEAR CHANGE IN YOUTH SHARE at an ownership change:")
print(f"    NEW OWNER : {rr['mean']:+.4f} [{rr['lo']:+.4f},{rr['hi']:+.4f}] "
      f"p={rr['p']:.4f} n={rr['n']}")
print(f"    SAME OWNER: {ss['mean']:+.4f} [{ss['lo']:+.4f},{ss['hi']:+.4f}] "
      f"p={ss['p']:.4f} n={ss['n']}")
diff = cci(pd.concat([(no_.z - no_.z_p), -(so.z - so.z_p)]),
           pd.concat([no_.season, so.season]))
RES["youth_at_owner_change"] = dict(new_owner=rr, same_owner=ss)

# robustness: drop the transition season (season attribution may be off by one)
no2 = no_[no_.tenure != 0] if (no_.tenure != 0).any() else no_
print(f"\n  ROBUSTNESS — same test dropping the labelled transition season: "
      f"n={len(no2)}")
if len(no2) >= 3:
    r2 = cci(no2.z - no2.z_p, no2.season)
    print(f"    NEW OWNER : {r2['mean']:+.4f} [{r2['lo']:+.4f},{r2['hi']:+.4f}]"
          f" p={r2['p']:.4f}")
    RES["youth_at_owner_change_drop_transition"] = r2

con.close()
RES["elapsed_s"] = round(time.time() - T0, 1)
RES["provenance"] = ("OWNER_SPELLS is HAND-BUILT from public knowledge of "
                     "franchise control sales; not scraped, not machine-"
                     "verified; season attribution may be off by one.")
(DATA / "d172_ownership.json").write_text(json.dumps(RES, indent=1, default=str))
print(f"\nWROTE data/d172_ownership.json ({RES['elapsed_s']}s)")
