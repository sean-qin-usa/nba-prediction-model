import pytest
import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbapred.ids import norm_name


def _load_odds_module():
    spec = importlib.util.spec_from_file_location("load_odds", ROOT / "scripts/load_odds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_norm_name():
    assert norm_name("Luka Dončić") == "luka doncic"
    assert norm_name("Jaren Jackson Jr.") == "jaren jackson"
    assert norm_name("De'Aaron Fox") == "de aaron fox"
    assert norm_name("Alperen Şengün") == "alperen sengun"
    assert norm_name("P.J. Washington") == "p j washington"


def test_flatten_odds_record():
    lo = _load_odds_module()
    rec = {
        "snapshot_ts": "2026-07-26T12:00:00+00:00",
        "data": [{
            "id": "e1", "commence_time": "2026-10-21T23:30:00Z",
            "home_team": "H", "away_team": "A",
            "bookmakers": [{"key": "dk", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "H", "price": 1.5}, {"name": "A", "price": 2.6}]},
                {"key": "player_points", "outcomes": [
                    {"name": "Over", "description": "X Y", "price": 1.9, "point": 20.5}]},
            ]}],
        }],
    }
    rows = lo.flatten(rec, "f.jsonl", dt.datetime.now(dt.timezone.utc))
    assert len(rows) == 3
    prop = [r for r in rows if r[9] == "player_points"][0]
    assert prop[11] == "X Y" and prop[13] == 20.5


def test_parse_player_archived():
    from nbapred.ingest.ratings_2k import parse_player, parse_team
    day_dirs = sorted((ROOT / "data/raw/ratings_2k").glob("*/"))
    if not day_dirs:
        # A DATA PRECONDITION, NOT A DEFECT: this exercises the parser against
        # archived scrapes, which are not committed, so a fresh clone has
        # nothing to parse. Asserting here reported a missing archive as a
        # broken parser (D229).
        pytest.skip("no archived 2K HTML in this checkout")
    team_files = [f for d in day_dirs for f in d.glob("*.html")]
    parsed_any = False
    for f in team_files[:50]:
        p = parse_player(f.read_text())
        if p and p.get("attributes"):
            assert p["overall"] and len(p["attributes"]) > 20
            parsed_any = True
            break
    assert parsed_any


def test_sbr_resolve_spread_total():
    from nbapred.ingest.sbr_hist import _num
    assert _num("pk") == 0.0
    assert _num("229.5") == 229.5
    assert _num("½".replace("½", "3.5")) == 3.5
    assert _num("nan") is None
    assert _num("") is None


def test_sbr_year_inference_survives_the_bubble():
    """SBR dates are MM/DD with no year. The old rule ("month >= 9 -> season
    start year") silently dated the 2019-20 bubble a full year early: 109 games
    played Aug-Oct 2020 landed in 2019. Year must advance on the calendar wrap
    only, so a season can legitimately run Oct 2019 -> Oct 2020."""
    import pandas as pd

    from nbapred.ingest.sbr_hist import parse_season

    def row(date, vh, team, final, open_, close):
        return {"Date": date, "Rot": 0, "VH": vh, "Team": team, "1st": 0,
                "2nd": 0, "3rd": 0, "4th": 0, "Final": final, "Open": open_,
                "Close": close, "ML": -110, "2H": 0}

    # Oct 2019 tip-off -> Mar 2020 -> the Jul/Sep/Oct 2020 bubble restart.
    df = pd.DataFrame([
        row("1022", "V", "Boston", 100, 220, 220), row("1022", "H", "Toronto", 101, 3, 3),
        row("311", "V", "Utah", 100, 220, 220), row("311", "H", "Denver", 101, 3, 3),
        row("730", "V", "Miami", 100, 220, 220), row("730", "H", "Milwaukee", 101, 3, 3),
        row("1011", "V", "Miami", 100, 220, 220), row("1011", "H", "LALakers", 101, 3, 3),
    ])
    got = [g for g in parse_season(df, "2019-20").game_date]
    assert got == [dt.date(2019, 10, 22), dt.date(2020, 3, 11),
                   dt.date(2020, 7, 30), dt.date(2020, 10, 11)], got


def test_darko_schema_cols():
    from nbapred.ingest.darko import _COLS
    # all target columns present in the rename map
    targets = set(_COLS.values())
    assert {"nba_player_id", "dpm", "o_dpm", "d_dpm", "onoff_o_dpm"} <= targets


def test_shot_zone_classification():
    from nbapred.features.possessions import _zone
    assert _zone(27, 3, "Jump Shot") == "thr"
    assert _zone(1, 2, "Driving Layup Shot") == "rim"
    assert _zone(18, 2, "Pullup Jump shot") == "mid"
    assert _zone(2, 2, "Dunk") == "rim"


def test_stint_elapsed_time():
    from nbapred.features.stints import _elapsed
    assert _elapsed(1, "PT12M00.00S") == 0.0
    assert _elapsed(1, "PT11M00.00S") == 60.0
    assert _elapsed(2, "PT12M00.00S") == 720.0
    assert _elapsed(5, "PT05M00.00S") == 2880.0   # OT starts at 4*720


def test_minutes_to_sec():
    from nbapred.features.possessions import _mins_to_sec
    assert _mins_to_sec("12:18") == 738
    assert _mins_to_sec("0:00") == 0
    assert _mins_to_sec("") == 0


def test_haversine_and_host():
    from nbapred.features.schedule import _haversine, _parse_host
    assert _parse_host("PHI @ NYK") == "NYK"
    assert _parse_host("BOS vs. LAL") == "BOS"
    d = _haversine("LAL", "BOS")   # ~4170 km
    assert 3900 < d < 4400
    assert _haversine("LAL", "LAL") == 0.0


def test_metrics():
    from nbapred.eval.metrics import log_loss, brier, ece
    import numpy as np
    y = np.array([1, 0, 1, 0])
    assert abs(log_loss(y, [0.5]*4) - 0.6931) < 1e-3
    assert brier(y, [0.5]*4) == 0.25
    # perfect calibration -> low ECE
    assert ece([1,1,0,0], [0.9,0.9,0.1,0.1], bins=5) < 0.2


def test_ablation_gate_rejects_noise():
    from nbapred.eval.ablate import paired_bootstrap_delta
    import numpy as np
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.full(500, 0.5)
    p_noise = np.clip(p + rng.normal(0, 0.01, 500), 0.01, 0.99)
    r = paired_bootstrap_delta(y, p, p_noise)
    assert r["keep"] is False   # pure noise must not pass the gate


def test_elo_probabilities():
    from nbapred.eval.walkforward import Elo
    e = Elo()
    p = e.p_home("A", "B")   # equal ratings + home edge -> >0.5
    assert 0.5 < p < 0.75


def test_market_devig():
    from scripts.market_accuracy import american_to_prob, devig_home
    assert abs(american_to_prob(-200) - 0.6667) < 1e-3
    assert abs(american_to_prob(+150) - 0.4) < 1e-3
    p = devig_home(-200, +170)   # home favored, de-vigged, sums to 1 w/ away
    assert 0.5 < p < 0.75


def test_pit_asof_blocks_future(tmp_path=None):
    import datetime as dt
    from nbapred.db import connect
    from nbapred.pit import darko_asof
    con = connect(read_only=True)
    past = darko_asof(con, dt.date(2020, 1, 1))
    con.close()
    assert len(past) == 0   # cannot leak a future snapshot into a past game


def test_eb_shrinkage_behavior():
    import numpy as np, pandas as pd
    from nbapred.features.skill_priors import estimate
    # 60 shooters at ~35% on 100 att each + one 0/2 player -> 0/2 must pull to ~league
    rng = np.random.default_rng(1)
    rows = []
    for pid in range(60):
        att = 100
        mk = rng.binomial(att, 0.35)
        rows.append(dict(player_id=pid, seconds=1800, fg3m=mk, fg3a=att,
                         rimm=0, rima=0, midm=0, mida=0, ftm=0, fta=0,
                         ast=0, tov=0, oreb=0, dreb=0, stl=0, blk=0, pf=0))
    rows.append(dict(player_id=999, seconds=300, fg3m=0, fg3a=2, rimm=0, rima=0,
                     midm=0, mida=0, ftm=0, fta=0, ast=0, tov=0, oreb=0, dreb=0,
                     stl=0, blk=0, pf=0))
    est = estimate(pd.DataFrame(rows), min_minutes=0)
    thin = est[est.player_id == 999]["fg3_pct"].iloc[0]
    league = est[est.player_id < 60]["fg3_pct"].mean()
    # a 0/2 player must be shrunk to within a hair of the league mean, not 0.0
    assert abs(thin - league) < 0.05, (thin, league)
    assert 0.28 < thin < 0.42


def test_engine_produces_basketball():
    from nbapred.engine.possession import LEAGUE, simulate_matchup
    res = simulate_matchup(LEAGUE, LEAGUE, n=800, seed=3)
    assert 95 < res["home_pts_mean"] < 130
    assert 95 < res["away_pts_mean"] < 130
    assert 0.45 < res["p_home_win"] < 0.65      # home edge, not extreme
    assert res["total_sd"] > 5                    # real variance, not degenerate


def test_compose_bridge():
    import warnings; warnings.filterwarnings("ignore")
    from nbapred.db import connect
    from nbapred.engine.compose import matchup_rates, team_defense_shift
    con = connect(read_only=True)
    tid = con.execute("SELECT team_id FROM player_game_stats WHERE game_id LIKE '002%' "
                      "GROUP BY team_id ORDER BY count(*) DESC LIMIT 1").fetchone()
    if not tid:
        con.close(); return  # no data yet
    t = tid[0]
    opp = con.execute("SELECT team_id FROM player_game_stats WHERE team_id<>? AND game_id LIKE '002%' "
                      "GROUP BY team_id LIMIT 1", [t]).fetchone()[0]
    r = matchup_rates(con, t, opp)
    shift = team_defense_shift(con, t)
    con.close()
    # shares sum to 1, FG% in (0,1), shrunk defense shift bounded
    assert abs(sum(r["zone_share"].values()) - 1.0) < 1e-6
    assert all(0 < v < 1 for v in r["zone_fg"].values())
    assert -0.6 < shift < 0.6   # shrinkage keeps it sane on small samples
