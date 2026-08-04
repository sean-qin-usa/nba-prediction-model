"""CLV (closing line value) logger — the primary KPI (handoff I.5).

Workflow: at decision time, record a bet CANDIDATE (timestamp, game, side, the
price we could get NOW, our model prob, which window rule fired). After close,
fill in the closing price. CLV = implied prob at close minus implied prob at
bet — positive = we beat the close. Skill shows in CLV long before ROI does.

Candidates are recorded even when we don't bet (paper mode) — the H-A test is
"does the line move our way in flagged windows", which needs no money.
"""
from __future__ import annotations

import datetime as dt

SCHEMA = """
CREATE TABLE IF NOT EXISTS clv_log (
    candidate_ts   TIMESTAMPTZ NOT NULL,   -- when we would have bet (PIT)
    game_date      DATE, home VARCHAR, away VARCHAR,
    side           VARCHAR NOT NULL,       -- 'home'/'away'
    rule           VARCHAR,                -- flagged-window rule (W1..W4) or 'unflagged'
    model_p        DOUBLE,                 -- our prob for the side at candidate_ts
    price_decimal  DOUBLE,                 -- best available decimal odds at candidate_ts
    implied_p      DOUBLE,                 -- de-vigged implied prob at candidate_ts
    close_price    DOUBLE,                 -- filled post-close
    close_implied  DOUBLE,
    clv            DOUBLE,                 -- close_implied - implied_p (>0 = beat close)
    outcome        INTEGER,               -- 1 side won / 0 lost (filled post-game)
    stake_units    DOUBLE DEFAULT 0.0,     -- 0 = paper
    PRIMARY KEY (candidate_ts, home, away, side)
);
"""


def record_candidate(con, *, game_date, home, away, side, rule, model_p,
                     price_decimal, implied_p, stake_units=0.0):
    con.execute(SCHEMA)
    con.execute("""INSERT OR REPLACE INTO clv_log VALUES
        (?,?,?,?,?,?,?,?,?, NULL, NULL, NULL, NULL, ?)""",
        [dt.datetime.now(dt.timezone.utc), game_date, home, away, side, rule,
         model_p, price_decimal, implied_p, stake_units])


def fill_close(con, *, game_date, home, away, side, close_price, close_implied,
               outcome):
    con.execute("""UPDATE clv_log SET close_price=?, close_implied=?,
        clv = ? - implied_p, outcome=?
        WHERE game_date=? AND home=? AND away=? AND side=?""",
        [close_price, close_implied, close_implied, outcome,
         game_date, home, away, side])


def clv_report(con) -> dict:
    """The H-A scorecard: CLV by window rule. Positive mean CLV in flagged
    windows with n>=300 = the pre-registered success criterion."""
    rows = con.execute("""SELECT coalesce("rule",'unflagged') AS r, count(*) n,
        avg(clv) mean_clv, avg(outcome) win_rate
        FROM clv_log WHERE clv IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    return {r[0]: {"n": r[1], "mean_clv": r[2], "win_rate": r[3]} for r in rows}
