#!/usr/bin/env python3
"""TS-STRATEGY — the trading strategy, designed around our LOSSES.

=============================================================================
INFORMATION DISCIPLINE (read this first)
=============================================================================
The MODEL is market-blind: `p_us` is produced by nbapred/ without ever
seeing a price.  That property is preserved here and is not negotiable.

The STRATEGY is not market-blind and must not be.  A strategy that cannot
see the price it is transacting against cannot compute an edge; comparing
our probability to the price we are about to trade is the definition of
the job, and it is NOT lookahead.  What would be lookahead is using any
price that is only knowable AFTER the decision.

Our historical price series is the CLOSE (`odds_market.p_home_spread`, a
de-vigged closing-spread probability).  So the only honest formulation is a
BET-AT-CLOSE strategy: it transacts at the close and uses THAT price as its
decision input.  Nothing in this file reads a price from the future.

That is deliberately the hardest possible test.  The close is the sharpest
price of the day, so an edge that survives at the close is a LOWER BOUND on
the edge available earlier.  scripts/ts_openers.py measures exactly how much
we give up by being forced to this formulation:
    * openers exist in odds_hist_sbr only, and that table dies mid-2022-23 —
      there is NO opener for any OOS season, so bet-at-open is UNTESTABLE
      out-of-sample in this repo;
    * on the covered window (n=1,892) the open->close sharpening is
      +0.0091 nats/game, our mean edge is -0.0079 vs the open against
      -0.0192 vs the close, and betting our side at the open earns
      +0.0112 probability of CLV (t=+6.7) while the favourite-drift control
      earns -0.0036 (t=-2.1).  The CLV is ours, not free.
So: everything below is the pessimistic case, by construction.

=============================================================================
DESIGN INPUTS — the loss forensics this strategy is built around
=============================================================================
(1) Our net deficit vs the market is a THIN UNIFORM BLEED across every
    confidence bucket (+4..+10 nats each) PLUS a FAT TAIL: the worst 1% of
    games carry 68% of the net deficit.
      => There is no confidence bucket that is "safe".  The only structural
         escape is to REMOVE THE TAIL.  Hence Layer A3 (tail veto) is the
         centrepiece of this strategy, not an afterthought.
(2) We are well calibrated at high confidence (85%+: predict 89.3%, hit
    88.3%) and useless at low confidence (50-55%: predict 52.4%, hit 52.9%,
    market 56.8%).
      => Layer A1: only trade games where |p_us - 0.5| is large.
(3) When our confidence EXCEEDS the market's we lose 0.0150/gm; when it is
    BELOW the market's we lose only 0.0064/gm.
      => This is the uncomfortable one.  The textbook bet -- back the side
         where our probability beats the price -- lands in the FIRST regime
         whenever we back the shared favourite, i.e. exactly where we are
         worst.  The second regime is reached by backing the shared DOG
         (we think the game is closer than the market prices it).
         Both directions are therefore PRE-REGISTERED ARMS (FAV / DOG) and
         both are reported.  The forensics predict DOG > FAV; D77/D82
         predict the opposite ("fade favourites contraindicated",
         "pro-shaded-side structure").  The data arbitrates, and the
         multiplicity is paid for in the family-wise noise test.
(4) Prior sims: the late-season window was OOS-positive at vig in 3
    consecutive sims; T20.D03-10 base (n=63) was -0.65% at vig / +3.83%
    fair.  The late window is carried as a pre-registered overlay.

=============================================================================
STRATEGY SPEC
=============================================================================
LAYER A — ELIGIBILITY (which games we will CONSIDER; no outcome is touched)
  A1 CONFIDENCE TIER   |p_us - 0.5| >= 0.20                         [input 2]
  A2 SHARED SIDE       (p_us-0.5)*(p_mkt-0.5) > 0 — we and the market agree
                       who is favoured.  Opposite-side games are known net
                       negative (D78) and are never traded.
  A3 TAIL VETO (optional arm; the fat-tail control)  [input 1, D101/D84]
       veto if ANY of
         EARLY  min(gp_home, gp_away) < 20      (D84-A early regime, 25.1%)
         CHAOS  max(m5_abs_h, m5_abs_a) >= 18.0 (top quintile of trailing-5
                mean |margin| — the "chaos team" proxy, 20.0%)
         FRESH  max over teams of #inactives who averaged >=24 min over
                their last 10 appearances AND played in that team's
                PREVIOUS game >= 2  (a multi-player unmodelled availability
                shock — the "post-event window" proxy, 6.9%)
       All three thresholds are fixed a priori from COVARIATE quantiles
       only; no outcome, price or PnL was consulted to choose them.
  A4 LATE WINDOW (optional arm)  max(gp_home, gp_away) >= 55        [input 4]

LAYER B — EDGE TEST vs the transacted price, INCLUDING VIG
  Side is set by the arm: FAV = the shared favourite, DOG = the shared dog.
  Offered price: proportional overround V on the de-vigged close,
      q_side = p_mkt_side * V,  dec = max(1/q_side, 1.01),  V = 1.045.
  TEST      p_us_side > q_side      (strictly positive EV at the price we
            actually transact — equivalently Kelly f* > 0).  Note this is
            side-asymmetric BY CONSTRUCTION and correctly so: the vig
            hurdle is p_mkt_side*(V-1), which is 3.4pts of probability on a
            0.75 favourite and 1.1pts on a 0.25 dog.
  CAP       p_us_side - p_mkt_side <= 0.10   (adverse-selection cap: large
            divergence means the market knows something structural —
            D13/H10, and directly implied by input (3)).

LAYER C — SIZING
  Fractional Kelly on the VIGGED odds:  f* = (p_us_side*dec - 1)/(dec - 1)
  stake_frac = min(KELLY_FRAC * f*, PER_BET_CAP)   KELLY_FRAC=0.25 (quarter
               Kelly is the standard haircut for parameter uncertainty),
               PER_BET_CAP = 0.02 of bankroll — the HARD per-bet cap.
  DAY_CAP    total staked on one calendar date <= 0.06 of bankroll, scaled
             pro rata if exceeded (same-day bets are the correlated-loss
             channel; this is the second tail control).
  Bankroll COMPOUNDS; bets settle end of day.
  FLAT sizing (1u, non-compounding) is reported alongside for comparability
  with the D75/D78 sims.

LAYER D — BANKROLL SIM with drawdown tracking (equity curve, max DD as a
  fraction of running peak, per-season decomposition).

=============================================================================
EVALUATION (deliberately reversed vs the earlier sims)
=============================================================================
  IS  = 2022-23 + 2023-24   (the LESS-developed seasons)
  OOS = 2024-25 + 2025-26
  and the REVERSE direction is reported too.  The earlier sims selected on
  2023-24/2024-25 and validated on 2025-26; putting the development
  seasons in the HOLDOUT is the harder and more honest arrangement.
  2021-22 is reported per-season where available (no game_inactives before
  2022-23, so the FRESH veto leg cannot be built there — it is excluded
  from the headline windows and shown as a degraded-feature footnote).

  Reported: ROI at vig, ROI fair, hit rate, n bets, max drawdown, bankroll
  curve, per-season; bootstrap CIs on ROI; and noise-compatibility
  P(observed ROI >= x | true breakeven) both per-rule and FAMILY-WISE.

RULES HONORED: DuckDB read_only=True; new file scripts/ts_strategy.py only;
nbapred/ untouched; helpers imported (not edited) from ba_intersection.py.

Run:  python scripts/ts_strategy.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ba_intersection import team_schedule                        # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
PERGAME = os.path.join(ROOT, "data", "ds_rt3_evrec5_pergame.csv")
FRAME_CACHE = os.path.join(ROOT, "data", "ts_frame.csv")
OUT_JSON = os.path.join(ROOT, "data", "ts_strategy.json")
OUT_BETS = os.path.join(ROOT, "data", "ts_strategy_bets.csv")

# ---- pricing -------------------------------------------------------------
OVERROUND = 1.045        # proportional; the D75/D78 convention
MIN_DEC = 1.01

# ---- eligibility (all fixed a priori) ------------------------------------
CONF_MIN = 0.20          # |p_us - 0.5|
DIV_CAP = 0.10           # adverse-selection cap on gross edge
EARLY_GP = 20            # D84-A early regime
CHAOS_M5 = 18.0          # top quintile of max trailing-5 mean |margin|
FRESH_MIN = 2            # multi-player fresh rotation absence
LATE_GP = 55             # D73/D75 late-season window

# ---- sizing --------------------------------------------------------------
KELLY_FRAC = 0.25
PER_BET_CAP = 0.02
DAY_CAP = 0.06
BANKROLL0 = 100.0

# ---- eval ----------------------------------------------------------------
WIN_A = ("2022-23", "2023-24")
WIN_B = ("2024-25", "2025-26")
ALL_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
N_BOOT = 4000
N_NULL = 20000
SEED = 20260801
ROT_MIN = 24.0           # minutes bar for "rotation player"
ROT_TRAIL = 10
ROT_MIN_GP = 5


# ==========================================================================
# frame
# ==========================================================================
def _fresh_outs(con) -> pd.DataFrame:
    """Per (game_id, team): count of inactives who (a) averaged >= ROT_MIN
    minutes over their last ROT_TRAIL 12+-minute appearances strictly before
    the game, and (b) PLAYED in that team's immediately previous game.
    (b) is what makes it a *fresh* shock rather than a long-term absence the
    model has already absorbed."""
    ina = con.execute("""
        SELECT i.game_id, i.player_id, g.team_abbrev AS team, g.game_date
        FROM game_inactives i
        JOIN nba_games g ON g.game_id = i.game_id AND g.team_id = i.team_id
        WHERE i.game_id LIKE '002%'
    """).fetchdf()
    tg = con.execute("""
        SELECT season, game_id, game_date, team_abbrev AS team
        FROM nba_games WHERE game_id LIKE '002%'
    """).fetchdf()
    app = con.execute("""
        SELECT game_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds >= 720
    """).fetchdf()
    pmin = con.execute("""
        SELECT s.player_id, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()

    tg["game_date"] = pd.to_datetime(tg.game_date)
    tg = tg.sort_values(["season", "team", "game_date", "game_id"])
    tg["prev_gid"] = tg.groupby(["season", "team"]).game_id.shift(1)
    prev = dict(zip(zip(tg.game_id.astype(str), tg.team), tg.prev_gid))
    appset = set(zip(app.game_id.astype(str), app.player_id.astype(int)))

    hist: dict[int, tuple] = {}
    for pid, sub in pmin.groupby("player_id", sort=False):
        hist[int(pid)] = (sub.game_date.values.astype("datetime64[D]"),
                          sub.mins.values)

    gd = pd.to_datetime(ina.game_date).values.astype("datetime64[D]")
    fresh = np.zeros(len(ina), bool)
    for k, (gid, pid, team, d) in enumerate(
            zip(ina.game_id.astype(str), ina.player_id.astype(int),
                ina.team, gd)):
        h = hist.get(pid)
        if h is None:
            continue
        i = np.searchsorted(h[0], d)          # strictly-before games (PIT)
        if i < ROT_MIN_GP:
            continue
        if h[1][max(0, i - ROT_TRAIL):i].mean() < ROT_MIN:
            continue
        pg = prev.get((gid, team))
        if pg is not None and (str(pg), pid) in appset:
            fresh[k] = True
    ina["fresh"] = fresh
    out = (ina.groupby(["game_id", "team"])
           .agg(n_inact=("player_id", "size"), n_fresh=("fresh", "sum"))
           .reset_index())
    out["game_id"] = out.game_id.astype(str).str.zfill(10)
    return out


def build_frame(rebuild: bool = False) -> pd.DataFrame:
    if os.path.exists(FRAME_CACHE) and not rebuild:
        return pd.read_csv(FRAME_CACHE, dtype={"game_id": str}, parse_dates=["game_date"])

    df = pd.read_csv(PERGAME, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df.rename(columns={"p_ctl": "p_us"})[
        ["season", "game_id", "game_date", "home", "away", "y", "p_us",
         "p_mkt"]].copy()

    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
        fo = _fresh_outs(con)
    finally:
        con.close()
    ts["game_id"] = ts.game_id.astype(str).str.zfill(10)

    cols = ["gp_before", "m5_abs", "m5_signed", "wpct_before", "rest"]
    for side in ("home", "away"):
        s = side[0]
        m = ts.rename(columns={"team": side})[["game_id", side] + cols]
        df = df.merge(m, on=["game_id", side], how="left")
        df = df.rename(columns={c: f"{c}_{s}" for c in cols})
        j = fo.rename(columns={"team": side})
        df = df.merge(j, on=["game_id", side], how="left")
        df = df.rename(columns={"n_inact": f"n_inact_{s}",
                                "n_fresh": f"n_fresh_{s}"})

    assert df.gp_before_h.notna().all() and df.gp_before_a.notna().all()
    # m5_abs is NaN for a team's first <3 games -> those are EARLY-vetoed
    # anyway; fill with 0 so the CHAOS test is well defined and never fires
    # on a missing value.
    for c in ("m5_abs_h", "m5_abs_a"):
        df[c] = df[c].fillna(0.0)
    for c in ("n_fresh_h", "n_fresh_a", "n_inact_h", "n_inact_a"):
        df[c] = df[c].fillna(0.0)

    df["gp_min"] = df[["gp_before_h", "gp_before_a"]].min(axis=1)
    df["gp_max"] = df[["gp_before_h", "gp_before_a"]].max(axis=1)
    df["chaos"] = df[["m5_abs_h", "m5_abs_a"]].max(axis=1)
    df["fresh"] = df[["n_fresh_h", "n_fresh_a"]].max(axis=1)
    df["has_inact"] = df.season != "2021-22"     # game_inactives starts 22-23

    # --- market/model geometry (all knowable at the close, before tip) ---
    df["conf_us"] = (df.p_us - 0.5).abs()
    df["conf_mkt"] = (df.p_mkt - 0.5).abs()
    df["fav_home"] = df.p_mkt > 0.5
    df["same_side"] = (df.p_us - 0.5) * (df.p_mkt - 0.5) > 0
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    df.to_csv(FRAME_CACHE, index=False)
    return df


# ==========================================================================
# strategy object
# ==========================================================================
@dataclass
class StrategyConfig:
    name: str
    direction: str = "FAV"            # FAV | DOG
    conf_min: float = CONF_MIN
    div_cap: float = DIV_CAP
    tail_veto: bool = True
    late_only: bool = False
    overround: float = OVERROUND
    kelly_frac: float = KELLY_FRAC
    per_bet_cap: float = PER_BET_CAP
    day_cap: float = DAY_CAP


@dataclass
class Strategy:
    """Full strategy object: eligibility -> edge test -> sizing -> bankroll."""
    cfg: StrategyConfig
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------- Layer A ------------------------------------------
    def eligibility(self, df: pd.DataFrame) -> pd.Series:
        c = self.cfg
        m = (df.conf_us >= c.conf_min) & df.same_side          # A1, A2
        if c.tail_veto:                                        # A3
            veto = ((df.gp_min < EARLY_GP)
                    | (df.chaos >= CHAOS_M5)
                    | (df.fresh >= FRESH_MIN))
            m &= ~veto
        if c.late_only:                                        # A4
            m &= df.gp_max >= LATE_GP
        return m

    # ---------------- Layer B ------------------------------------------
    def price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Side selection + the price we transact against + the edge test."""
        c = self.cfg
        # FAV = shared favourite; DOG = shared dog.  A2 guarantees we and the
        # market agree who the favourite is, so 'side' is unambiguous.
        bet_home = df.fav_home if c.direction == "FAV" else ~df.fav_home
        p_us_side = np.where(bet_home, df.p_us, 1 - df.p_us)
        p_mkt_side = np.where(bet_home, df.p_mkt, 1 - df.p_mkt)
        q = p_mkt_side * c.overround
        dec = np.maximum(1.0 / q, MIN_DEC)
        out = pd.DataFrame({
            "bet_home": bet_home.values if hasattr(bet_home, "values")
            else bet_home,
            "p_us_side": p_us_side, "p_mkt_side": p_mkt_side,
            "q": q, "dec": dec, "dec_fair": 1.0 / p_mkt_side,
            "e_gross": p_us_side - p_mkt_side,
            "e_net": p_us_side - q,
        }, index=df.index)
        out["hit"] = np.where(out.bet_home, df.y == 1, df.y == 0)
        return out

    def edge_test(self, df: pd.DataFrame, px: pd.DataFrame) -> pd.Series:
        return (px.e_net > 0) & (px.e_gross <= self.cfg.div_cap)

    # ---------------- bet set ------------------------------------------
    def bets(self, df: pd.DataFrame) -> pd.DataFrame:
        px = self.price(df)
        m = self.eligibility(df) & self.edge_test(df, px)
        b = pd.concat([df.loc[m, ["season", "game_id", "game_date", "home",
                                  "away", "y", "p_us", "p_mkt", "gp_min",
                                  "gp_max", "chaos", "fresh"]],
                       px.loc[m]], axis=1)
        return b.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # ---------------- Layer C + D ---------------------------------------
    def simulate(self, b: pd.DataFrame, sizing: str = "kelly",
                 bankroll0: float = BANKROLL0) -> dict:
        """Bankroll simulation with drawdown tracking.  Bets settle end of
        day; same-day bets share the DAY_CAP exposure budget."""
        c = self.cfg
        if len(b) == 0:
            return dict(n=0, n_wins=0, staked=0.0, pnl=0.0, roi=np.nan,
                        roi_fair=np.nan, hit=np.nan, sharpe=np.nan,
                        final=bankroll0, ret=0.0, maxdd=0.0, maxdd_units=0.0,
                        curve=np.array([bankroll0]), dates=[],
                        stakes=np.array([]), pnls=np.array([]),
                        pnls_fair=np.array([]), decs=np.array([]))
        bank = bankroll0
        curve, dates = [bankroll0], []
        stakes = np.zeros(len(b)); pnls = np.zeros(len(b))
        pnls_fair = np.zeros(len(b))
        for day, idx in b.groupby("game_date", sort=True).groups.items():
            idx = np.asarray(idx)
            sub = b.loc[idx]
            if sizing == "flat":
                st = np.ones(len(idx))                 # 1 unit, no compounding
            else:
                f = (sub.p_us_side * sub.dec - 1) / (sub.dec - 1)
                frac = np.minimum(c.kelly_frac * f.values, c.per_bet_cap)
                frac = np.maximum(frac, 0.0)
                tot = frac.sum()
                if tot > c.day_cap and tot > 0:        # DAY exposure cap
                    frac *= c.day_cap / tot
                st = frac * bank
            win = sub.hit.values
            pl = np.where(win, st * (sub.dec.values - 1), -st)
            plf = np.where(win, st * (sub.dec_fair.values - 1), -st)
            stakes[idx] = st; pnls[idx] = pl; pnls_fair[idx] = plf
            if sizing != "flat":
                bank += pl.sum()
            else:
                bank = bankroll0 + pnls.sum()
            curve.append(bank); dates.append(day)
        curve = np.asarray(curve)
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / np.where(peak > 0, peak, 1.0)
        staked = float(stakes.sum())
        return dict(
            n=len(b), n_wins=int(b.hit.sum()), staked=staked,
            pnl=float(pnls.sum()),
            roi=float(pnls.sum() / staked) if staked > 0 else np.nan,
            roi_fair=float(pnls_fair.sum() / staked) if staked > 0 else np.nan,
            hit=float(b.hit.mean()),
            sharpe=float(np.mean(pnls / np.where(stakes > 0, stakes, 1))
                         / np.std(pnls / np.where(stakes > 0, stakes, 1),
                                  ddof=1)) if len(b) > 1 else np.nan,
            final=float(curve[-1]), ret=float(curve[-1] / bankroll0 - 1),
            maxdd=float(dd.max()), maxdd_units=float((peak - curve).max()),
            curve=curve, dates=dates, stakes=stakes, pnls=pnls,
            pnls_fair=pnls_fair, decs=b.dec.values)


# ==========================================================================
# pre-registered config family
# ==========================================================================
def config_family() -> list[StrategyConfig]:
    fam = []
    for d in ("FAV", "DOG"):
        for veto in (False, True):
            for late in (False, True):
                nm = (f"{d}{'.veto' if veto else '.base'}"
                      f"{'.late' if late else ''}")
                fam.append(StrategyConfig(name=nm, direction=d,
                                          tail_veto=veto, late_only=late))
    return fam


# ==========================================================================
# statistics
# ==========================================================================
def boot_roi(stakes, pnls, n=N_BOOT, seed=SEED):
    """Bet-level bootstrap percentile CI on ROI = sum(pnl)/sum(stake)."""
    if len(stakes) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stakes), size=(n, len(stakes)))
    s = stakes[idx].sum(axis=1)
    p = pnls[idx].sum(axis=1)
    r = np.where(s > 0, p / np.where(s > 0, s, 1), np.nan)
    return (float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5)))


def noise_compat(stakes, decs, roi_obs, n=N_NULL, seed=SEED):
    """P(ROI >= observed | TRUE BREAKEVEN at the transacted price).
    Null: each bet wins with probability exactly 1/dec, i.e. the offered
    price is fair and our edge is identically zero.  This is the earlier
    analysis's question, asked per-rule."""
    if len(stakes) == 0 or not np.isfinite(roi_obs):
        return np.nan
    rng = np.random.default_rng(seed)
    q = 1.0 / decs
    w = rng.random((n, len(q))) < q
    pnl = np.where(w, stakes * (decs - 1), -stakes).sum(axis=1)
    return float((pnl / stakes.sum() >= roi_obs).mean())


def family_noise(df_eval, betsets, roi_key, n=N_NULL, seed=SEED):
    """FAMILY-WISE: P(max over the pre-registered family of fair-ROI >=
    the best observed | the de-vigged market price IS the truth).
    Simulated at the GAME level so that rules sharing games stay correlated
    exactly as they do in reality."""
    rng = np.random.default_rng(seed + 1)
    gids = df_eval.game_id.values
    pos = {g: i for i, g in enumerate(gids)}
    pm = df_eval.p_mkt.values
    Y = rng.random((n, len(pm))) < pm                      # y=1 -> home wins
    best_obs = -np.inf
    packs = []
    for nm, b in betsets.items():
        if len(b) == 0:
            continue
        j = np.array([pos[g] for g in b.game_id.values])
        packs.append((np.asarray(b.bet_home, bool), j,
                      b.dec_fair.values, np.ones(len(b))))
        best_obs = max(best_obs, roi_key.get(nm, -np.inf))
    if not packs or not np.isfinite(best_obs):
        return np.nan, np.nan
    mx = np.full(n, -np.inf)
    for bh, j, dfair, st in packs:
        win = np.where(bh[None, :], Y[:, j], ~Y[:, j])
        pnl = np.where(win, st * (dfair - 1), -st).sum(axis=1)
        mx = np.maximum(mx, pnl / st.sum())
    return float((mx >= best_obs).mean()), float(best_obs)


def loglosses(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ==========================================================================
# reporting
# ==========================================================================
HDR = (f"{'config':<18}{'sizing':<7}{'n':>5}{'hit%':>7}{'ROI%':>8}"
       f"{'ROIfair%':>10}{'CI95 ROI%':>20}{'maxDD%':>8}{'final':>9}"
       f"{'P(noise)':>10}")


def row(name, sizing, r, ci, pnull):
    ci_s = (f"[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]"
            if np.isfinite(ci[0]) else "n/a")
    return (f"{name:<18}{sizing:<7}{r['n']:>5}"
            f"{(r['hit']*100 if r['n'] else 0):>7.1f}"
            f"{(r['roi']*100 if r['n'] else 0):>8.2f}"
            f"{(r['roi_fair']*100 if r['n'] else 0):>10.2f}"
            f"{ci_s:>20}{r['maxdd']*100:>8.1f}{r['final']:>9.1f}"
            f"{pnull:>10.3f}")


def tail_diagnostic(df: pd.DataFrame) -> dict:
    """The HIGH-POWER structural test.  Betting ROI on n~100 bets is nearly
    powerless; log-loss on n~5000 games is not.  Question: does the
    eligibility + tail veto actually produce a subset where our net deficit
    vs the market is gone?  If it does not, no sizing rule can rescue it."""
    print("\n" + "=" * 96)
    print("STRUCTURAL DIAGNOSTIC — does the filter remove the deficit? "
          "(log-loss, high power)")
    print("=" * 96)
    d = df[df.season.isin(ALL_SEASONS)].copy()
    d["L_us"] = loglosses(d.p_us, d.y)
    d["L_mkt"] = loglosses(d.p_mkt, d.y)
    d["net"] = d.L_us - d.L_mkt          # >0 = we are WORSE than the market

    rows = []
    veto = ((d.gp_min < EARLY_GP) | (d.chaos >= CHAOS_M5)
            | (d.fresh >= FRESH_MIN))
    elig = (d.conf_us >= CONF_MIN) & d.same_side
    strata = {
        "ALL games": np.ones(len(d), bool),
        "  worst 1% of games": np.zeros(len(d), bool),   # filled below
        "A1+A2 eligible": elig.values,
        "A1+A2 + tail VETO": (elig & ~veto).values,
        "  vetoed only": (elig & veto).values,
        "conf_us > conf_mkt": (d.conf_us > d.conf_mkt).values,
        "conf_us < conf_mkt": (d.conf_us < d.conf_mkt).values,
    }
    k = max(1, int(round(0.01 * len(d))))
    worst = d.net.nlargest(k).index
    strata["  worst 1% of games"] = d.index.isin(worst)

    tot = float(d.net.sum())
    print(f"{'stratum':<24}{'n':>7}{'net/gm':>10}{'nats/gm':>10}"
          f"{'share of total deficit':>26}")
    for nm, m in strata.items():
        m = np.asarray(m)
        if m.sum() == 0:
            continue
        s = float(d.net[m].sum())
        rows.append({"stratum": nm, "n": int(m.sum()),
                     "net_per_game": float(d.net[m].mean()),
                     "share": s / tot if tot else np.nan})
        print(f"{nm:<24}{m.sum():>7}{d.net[m].mean():>10.5f}"
              f"{d.net[m].mean()*1000:>10.2f}{s/tot*100:>25.1f}%")
    print("\n  net/gm > 0 means WE ARE WORSE than the de-vigged close on that"
          " stratum.\n  A stratum with net/gm <= 0 is the only place a"
          " bet-at-close edge can exist.")
    return {"strata": rows, "total_net": tot, "n": int(len(d))}


def evaluate(df, fam, seasons_is, seasons_oos, label, results):
    print("\n" + "=" * 96)
    print(f"{label}   IS = {seasons_is}   OOS = {seasons_oos}")
    print("=" * 96)
    block = {}
    for tag, seas in (("IS", seasons_is), ("OOS", seasons_oos)):
        sub = df[df.season.isin(seas)].copy()
        print(f"\n--- {tag}  ({' + '.join(seas)}, {len(sub)} games) ---")
        print(HDR)
        betsets, roi_vig, roi_fair = {}, {}, {}
        for cfg in fam:
            st = Strategy(cfg)
            b = st.bets(sub)
            betsets[cfg.name] = b
            for sizing in ("kelly", "flat"):
                r = st.simulate(b, sizing=sizing)
                ci = boot_roi(r["stakes"], r["pnls"])
                pn = noise_compat(r["stakes"], r["decs"], r["roi"])
                print(row(cfg.name, sizing, r, ci, pn))
                key = f"{tag}|{cfg.name}|{sizing}"
                block[key] = {k: v for k, v in r.items()
                              if k not in ("curve", "dates", "stakes",
                                           "pnls", "pnls_fair", "decs")}
                block[key].update(ci_lo=ci[0], ci_hi=ci[1], p_noise=pn)
                if sizing == "kelly":
                    roi_vig[cfg.name] = r["roi"]
                    roi_fair[cfg.name] = r["roi_fair"]
        pfam, best = family_noise(sub, betsets, roi_fair)
        print(f"\n  FAMILY-WISE noise test ({len(fam)} pre-registered "
              f"configs, game-level null 'the de-vigged close IS the truth'):"
              f"\n    best FAIR ROI observed = {best*100:+.2f}%   "
              f"P(max fair ROI >= that | market is truth) = {pfam:.3f}")
        block[f"{tag}|FAMILY"] = {"p_family": pfam, "best_fair_roi": best}
    results[label] = block
    return block


def per_season_table(df, fam):
    print("\n" + "=" * 96)
    print("PER-SEASON (quarter-Kelly, vig 4.5%) — every config, every season")
    print("=" * 96)
    seasons = ["2021-22"] + list(ALL_SEASONS)
    hdr = f"{'config':<18}" + "".join(f"{s:>22}" for s in seasons)
    print(hdr)
    print(f"{'':<18}" + "".join(f"{'n / ROI% / DD%':>22}" for _ in seasons))
    out = {}
    for cfg in fam:
        st = Strategy(cfg)
        cells, rec = [], {}
        for s in seasons:
            sub = df[df.season == s]
            if s == "2021-22" and cfg.tail_veto:
                cells.append(f"{'(no inactives)':>22}")
                continue
            b = st.bets(sub)
            r = st.simulate(b)
            cells.append(f"{r['n']:>5} {r['roi']*100 if r['n'] else 0:>7.2f}"
                         f" {r['maxdd']*100:>6.1f}")
            rec[s] = {"n": r["n"], "roi": r["roi"], "maxdd": r["maxdd"],
                      "hit": r["hit"]}
        print(f"{cfg.name:<18}" + "".join(cells))
        out[cfg.name] = rec
    print("\n  2021-22 has no game_inactives coverage, so the FRESH veto leg "
          "cannot be built there;\n  veto configs are shown as blank rather "
          "than silently run on a degraded filter.")
    return out


def season_decay(df, fam):
    """WHY does OOS fail?  Two candidate stories:
       (a) the strategy never had an edge (noise), or
       (b) it had one and the market closed it (decay).
    The betting series has no power to tell them apart at n~50/season, but
    the LOG-LOSS deficit on the strategy's own eligible stratum does."""
    print("\n" + "=" * 96)
    print("SEASON DECAY — is the OOS failure noise, or a closing market?")
    print("=" * 96)
    d = df[df.season.isin(ALL_SEASONS)].copy()
    d["net"] = loglosses(d.p_us, d.y) - loglosses(d.p_mkt, d.y)
    veto = ((d.gp_min < EARLY_GP) | (d.chaos >= CHAOS_M5)
            | (d.fresh >= FRESH_MIN))
    elig = (d.conf_us >= CONF_MIN) & d.same_side & ~veto
    print(f"{'season':<10}{'n_all':>7}{'net/gm ALL':>13}"
          f"{'n_elig':>8}{'net/gm on A1+A2+VETO stratum':>32}")
    rec = {}
    for s in ALL_SEASONS:
        ms = d.season == s
        a = float(d.net[ms].mean())
        e = float(d.net[ms & elig].mean())
        rec[s] = {"net_all": a, "net_elig": e, "n_elig": int((ms & elig).sum())}
        print(f"{s:<10}{ms.sum():>7}{a:>13.5f}{(ms & elig).sum():>8}"
              f"{e:>32.5f}")
    print("\n  Rising net/gm = we are falling further behind the close.")

    print(f"\n{'config':<18}" + "".join(f"{s:>11}" for s in ALL_SEASONS)
          + f"{'trend/season':>15}{'seasons>0':>11}")
    for cfg in fam:
        st = Strategy(cfg)
        rois = []
        for s in ALL_SEASONS:
            r = st.simulate(st.bets(df[df.season == s]))
            rois.append(r["roi"] if r["n"] else np.nan)
        v = np.array(rois, float)
        ok = np.isfinite(v)
        slope = (np.polyfit(np.arange(4)[ok], v[ok], 1)[0]
                 if ok.sum() >= 3 else np.nan)
        print(f"{cfg.name:<18}"
              + "".join(f"{x*100:>11.2f}" if np.isfinite(x) else f"{'-':>11}"
                        for x in v)
              + f"{slope*100:>15.2f}{int(np.nansum(v > 0)):>11}")
        rec[cfg.name] = {"rois": [None if not np.isfinite(x) else float(x)
                                  for x in v],
                         "slope": float(slope) if np.isfinite(slope) else None}
    return rec


def breakeven_vig(df, fam, seasons):
    """The line-shopping question: what overround would this config need in
    order to break even?  V=1.045 is our baseline assumption; a real
    line-shopped best-of-book on NBA moneylines runs ~1.02-1.03."""
    print("\n" + "=" * 96)
    print(f"BREAK-EVEN VIG on {'+'.join(seasons)} "
          "(what price would this config need?)")
    print("=" * 96)
    print(f"{'config':<18}{'n':>5}{'ROI@4.5%':>10}{'ROI@3.0%':>10}"
          f"{'ROI@2.0%':>10}{'ROI@fair':>10}{'breakeven V':>14}")
    out = {}
    for cfg in fam:
        base = Strategy(cfg)
        b0 = base.bets(df[df.season.isin(seasons)])
        if len(b0) == 0:
            continue
        # The bet SET is HELD at the V=4.5% definition, so these columns
        # compare the SAME bets at different prices (not different bets).
        def roi_at(V):
            b = b0.copy()
            b["dec"] = np.maximum(1.0 / (b.p_mkt_side * V), MIN_DEC)
            c2 = StrategyConfig(**{**cfg.__dict__, "overround": V})
            return Strategy(c2).simulate(b)["roi"]

        row_v = [roi_at(V) for V in (1.045, 1.030, 1.020, 1.000)]
        lo, hi = 1.0, 1.30                       # bisect the breakeven V
        if roi_at(lo) <= 0:
            bev = float("nan")                   # never profitable, even fair
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if roi_at(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            bev = 0.5 * (lo + hi)
        print(f"{cfg.name:<18}{len(b0):>5}"
              + "".join(f"{x*100:>10.2f}" for x in row_v)
              + (f"{bev:>14.4f}" if np.isfinite(bev)
                 else f"{'never (<fair)':>14}"))
        out[cfg.name] = {"roi_by_V": [float(x) for x in row_v],
                         "breakeven_V": float(bev)}
    print("\n  breakeven V > 1.045 means the config is profitable at our "
          "assumed price;\n  V < 1.045 means it needs a better price than we"
          " assumed to survive.")
    return out


def selection_protocol(df, fam, seasons_is, seasons_oos, label):
    """The mechanical, pre-registered decision: choose the single best
    config by IS ROI at vig among those with IS n >= 30, then report its
    OOS.  No discretion, no post-hoc reading."""
    print("\n" + "=" * 96)
    print(f"SELECTION PROTOCOL — {label}")
    print("=" * 96)
    cand = []
    for cfg in fam:
        st = Strategy(cfg)
        r = st.simulate(st.bets(df[df.season.isin(seasons_is)]))
        if r["n"] >= 30:
            cand.append((r["roi"], cfg, r))
    if not cand:
        print("  no config reached the n>=30 IS floor")
        return {}
    cand.sort(key=lambda t: -t[0])
    roi_is, cfg, r_is = cand[0]
    st = Strategy(cfg)
    b_oos = st.bets(df[df.season.isin(seasons_oos)])
    r_oos = st.simulate(b_oos)
    ci = boot_roi(r_oos["stakes"], r_oos["pnls"])
    pn = noise_compat(r_oos["stakes"], r_oos["decs"], r_oos["roi"])
    print(f"  IS  ({'+'.join(seasons_is)}): SELECTED = {cfg.name}  "
          f"n={r_is['n']} ROI={roi_is*100:+.2f}% (best of "
          f"{len(cand)} eligible configs)")
    print(f"  OOS ({'+'.join(seasons_oos)}): n={r_oos['n']} "
          f"hit={r_oos['hit']*100:.1f}% ROI={r_oos['roi']*100:+.2f}% "
          f"fair={r_oos['roi_fair']*100:+.2f}% "
          f"CI95=[{ci[0]*100:+.2f},{ci[1]*100:+.2f}] "
          f"maxDD={r_oos['maxdd']*100:.1f}% "
          f"bankroll -> {r_oos['final']:.1f}")
    print(f"  P(OOS ROI >= observed | true breakeven) = {pn:.3f}")
    print(f"  IS -> OOS ROI swing: {(r_oos['roi']-roi_is)*100:+.2f} points")
    return {"selected": cfg.name, "roi_is": float(roi_is),
            "n_is": int(r_is["n"]), "roi_oos": float(r_oos["roi"]),
            "roi_oos_fair": float(r_oos["roi_fair"]),
            "n_oos": int(r_oos["n"]), "hit_oos": float(r_oos["hit"]),
            "maxdd_oos": float(r_oos["maxdd"]),
            "ci_oos": [ci[0], ci[1]], "p_noise_oos": pn}


def sensitivity(df, cfg, seasons):
    """Does the verdict depend on the sizing knobs?  (It should not — if it
    does, the result is a sizing artifact, which is the D75 'Kelly-
    consistency' failure signature.)"""
    print("\n" + "=" * 96)
    print(f"SIZING SENSITIVITY — {cfg.name} on {'+'.join(seasons)}")
    print("=" * 96)
    sub = df[df.season.isin(seasons)]
    b = Strategy(cfg).bets(sub)
    print(f"{'sizing':<28}{'n':>5}{'ROI%':>9}{'maxDD%':>9}{'final':>9}")
    out = {}
    for lbl, kw in (("flat 1u", {"sizing": "flat"}),
                    ("quarter-Kelly cap 2%", {}),
                    ("eighth-Kelly cap 2%",
                     {"cfg": StrategyConfig(**{**cfg.__dict__,
                                               "kelly_frac": 0.125})}),
                    ("half-Kelly cap 2%",
                     {"cfg": StrategyConfig(**{**cfg.__dict__,
                                               "kelly_frac": 0.5})}),
                    ("quarter-Kelly cap 1%",
                     {"cfg": StrategyConfig(**{**cfg.__dict__,
                                               "per_bet_cap": 0.01})}),
                    ("quarter-Kelly no cap",
                     {"cfg": StrategyConfig(**{**cfg.__dict__,
                                               "per_bet_cap": 1.0,
                                               "day_cap": 1.0})})):
        c2 = kw.pop("cfg", cfg)
        r = Strategy(c2).simulate(b, **kw)
        print(f"{lbl:<28}{r['n']:>5}{r['roi']*100:>9.2f}"
              f"{r['maxdd']*100:>9.1f}{r['final']:>9.1f}")
        out[lbl] = {"roi": float(r["roi"]), "maxdd": float(r["maxdd"]),
                    "final": float(r["final"])}
    return out


def verdict(df, fam, results):
    """The single question: does ANY configuration hold ROI out-of-sample?
    Operationalised without discretion — a config HOLDS only if its ROI at
    vig is positive in BOTH halves of the corpus.  One positive half is what
    a coin does half the time."""
    print("\n" + "=" * 96)
    print("VERDICT — does any configuration hold ROI out of sample?")
    print("=" * 96)
    A, B = list(WIN_A), list(WIN_B)
    print(f"{'config':<18}{'nA':>5}{'ROI_A%':>9}{'nB':>5}{'ROI_B%':>9}"
          f"{'both>0?':>9}{'pooled%':>9}{'poolFair%':>11}{'maxDD%':>8}")
    holds, table = [], {}
    for cfg in fam:
        st = Strategy(cfg)
        ra = st.simulate(st.bets(df[df.season.isin(A)]))
        rb = st.simulate(st.bets(df[df.season.isin(B)]))
        rp = st.simulate(st.bets(df[df.season.isin(ALL_SEASONS)]))
        ok = (ra["n"] > 0 and rb["n"] > 0 and ra["roi"] > 0 and rb["roi"] > 0)
        holds.append(ok)
        print(f"{cfg.name:<18}{ra['n']:>5}{ra['roi']*100:>9.2f}"
              f"{rb['n']:>5}{rb['roi']*100:>9.2f}{('YES' if ok else 'no'):>9}"
              f"{rp['roi']*100:>9.2f}{rp['roi_fair']*100:>11.2f}"
              f"{rp['maxdd']*100:>8.1f}")
        table[cfg.name] = {"roi_A": float(ra["roi"]), "n_A": int(ra["n"]),
                           "roi_B": float(rb["roi"]), "n_B": int(rb["n"]),
                           "both_positive": bool(ok),
                           "roi_pooled": float(rp["roi"]),
                           "roi_pooled_fair": float(rp["roi_fair"]),
                           "maxdd_pooled": float(rp["maxdd"])}
    print(f"\n  configs positive at vig in BOTH halves: "
          f"{sum(holds)} of {len(fam)}")

    # Does betting ROI track the STRUCTURAL deficit at all?  If the ROI
    # series were an edge, our best seasons vs the close should be our best
    # betting seasons.  If the correlation is zero or backwards, the ROI
    # series is noise riding on a deficit that never went away.
    d = df[df.season.isin(ALL_SEASONS)].copy()
    d["net"] = loglosses(d.p_us, d.y) - loglosses(d.p_mkt, d.y)
    veto = ((d.gp_min < EARLY_GP) | (d.chaos >= CHAOS_M5)
            | (d.fresh >= FRESH_MIN))
    elig = (d.conf_us >= CONF_MIN) & d.same_side & ~veto
    struct = np.array([d.net[(d.season == s) & elig].mean()
                       for s in ALL_SEASONS])
    print("\n  structural deficit vs betting ROI, by season "
          "(on the A1+A2+VETO stratum):")
    print(f"    {'season':<10}{'net/gm (lower=better)':>24}"
          + "".join(f"{c.name:>16}" for c in fam if c.direction == "FAV"))
    rois = {}
    for c in fam:
        rois[c.name] = np.array(
            [Strategy(c).simulate(Strategy(c).bets(df[df.season == s]))["roi"]
             for s in ALL_SEASONS], float)
    for i, s in enumerate(ALL_SEASONS):
        print(f"    {s:<10}{struct[i]:>24.5f}"
              + "".join(f"{rois[c.name][i]*100:>16.2f}"
                        for c in fam if c.direction == "FAV"))
    corrs = {}
    for c in fam:
        v = rois[c.name]
        ok = np.isfinite(v)
        corrs[c.name] = (float(np.corrcoef(struct[ok], v[ok])[0, 1])
                         if ok.sum() >= 3 else None)
    print("\n    corr(structural deficit, ROI) per config "
          "(a real edge would be NEGATIVE and large):")
    for k, v in corrs.items():
        print(f"      {k:<18}{v:+.3f}" if v is not None else f"      {k}: n/a")
    print(f"    mean across configs: "
          f"{np.mean([v for v in corrs.values() if v is not None]):+.3f}")

    # ---- POWER: can this experiment answer the question at all? ----------
    # Under the breakeven null (win prob q = 1/dec), a FLAT 1u bet has
    # E[pnl] = 0 and Var[pnl] = dec - 1 EXACTLY.  So ROI has
    # se = sqrt(mean(dec-1)/n), and the one-sided 5% detection threshold is
    # 1.645*se.  This is the honest bound on what n bets can ever show.
    print("\n  POWER — what ROI could this experiment even detect?")
    print(f"    {'config':<18}{'n':>6}{'mean dec':>10}{'se(ROI)%':>10}"
          f"{'ROI% needed':>13}{'observed%':>11}{'n for +2% @80%':>16}")
    power = {}
    for cfg in fam:
        st = Strategy(cfg)
        b = st.bets(df[df.season.isin(ALL_SEASONS)])
        if len(b) == 0:
            continue
        d_ = b.dec.values
        var1 = float(np.mean(d_ - 1.0))
        se = float(np.sqrt(var1 / len(b)))
        need = 1.645 * se
        obs = st.simulate(b, sizing="flat")["roi"]
        n80 = (1.645 + 0.8416) ** 2 * var1 / (0.02 ** 2)
        print(f"    {cfg.name:<18}{len(b):>6}{d_.mean():>10.3f}{se*100:>10.2f}"
              f"{need*100:>13.2f}{obs*100:>11.2f}{n80:>16.0f}")
        power[cfg.name] = {"n": int(len(b)), "se_roi": se,
                           "roi_needed_5pct": float(need),
                           "roi_observed_flat": float(obs),
                           "n_for_2pct_at_80pct_power": float(n80)}
    print("    'n for +2% @80%' = bets needed for a TRUE +2% ROI edge to be")
    print("    detected 80% of the time at one-sided 5%.  Compare with the")
    print("    ~40 (FAV) / ~200 (DOG) bets this strategy generates per SEASON.")

    # full-sample family-wise noise
    sub = df[df.season.isin(ALL_SEASONS)]
    bs, rf = {}, {}
    for cfg in fam:
        st = Strategy(cfg)
        b = st.bets(sub); bs[cfg.name] = b
        rf[cfg.name] = st.simulate(b)["roi_fair"]
    p_fam, best = family_noise(sub, bs, rf)
    print(f"\n  FULL-SAMPLE family-wise: best fair ROI {best*100:+.2f}%, "
          f"P(max >= that | de-vigged close is the truth) = {p_fam:.3f}")
    results["verdict"] = {"table": table, "n_holding": int(sum(holds)),
                          "struct_by_season": [float(x) for x in struct],
                          "corr_struct_roi": corrs, "power": power,
                          "p_family_full": p_fam,
                          "best_fair_roi_full": float(best)}
    return results["verdict"]


def d112_convergence(df, results):
    """CROSS-THREAD (D112, landed the same day).  D112's W49 forensic found
    that on a same-side bet `edge == conf_us - conf_mkt` EXACTLY, so an
    upper conf-excess cap IS our DIV_CAP under another name; it reported
    R4_LOWT going -4.51% -> +2.47% pooled at X=0.08.  Our pre-registered cap
    is 0.10.  This sweeps the FAV arm at {0.10, 0.08, 0.06} to see whether
    the two threads converge, and then asks the only question that matters:
    is the number of surviving configs MORE than chance produces?"""
    print("\n" + "=" * 96)
    print("D112 CONVERGENCE — their conf-excess cap X vs our DIV_CAP")
    print("=" * 96)
    specs = [("FAV.base", False, False), ("FAV.base.late", False, True),
             ("FAV.veto", True, False), ("FAV.veto.late", True, True)]
    caps = (0.10, 0.08, 0.06)
    print(f"{'config':<16}{'cap':>6}{'nA':>5}{'ROI_A%':>9}{'nB':>5}"
          f"{'ROI_B%':>9}{'both>0':>8}{'pooled%':>9}{'poolFair%':>11}"
          f"{'P(noise)':>10}")
    cfgs, obs_hold, table = [], 0, {}
    for nm, veto, late in specs:
        for cap in caps:
            cfg = StrategyConfig(name=nm, direction="FAV", tail_veto=veto,
                                 late_only=late, div_cap=cap)
            st = Strategy(cfg)
            ra = st.simulate(st.bets(df[df.season.isin(WIN_A)]))
            rb = st.simulate(st.bets(df[df.season.isin(WIN_B)]))
            rp = st.simulate(st.bets(df[df.season.isin(ALL_SEASONS)]))
            ok = (ra["n"] > 0 and rb["n"] > 0
                  and ra["roi"] > 0 and rb["roi"] > 0)
            obs_hold += int(ok)
            pn = noise_compat(rp["stakes"], rp["decs"], rp["roi"])
            print(f"{nm:<16}{cap:>6.2f}{ra['n']:>5}{ra['roi']*100:>9.2f}"
                  f"{rb['n']:>5}{rb['roi']*100:>9.2f}"
                  f"{('YES' if ok else 'no'):>8}{rp['roi']*100:>9.2f}"
                  f"{rp['roi_fair']*100:>11.2f}{pn:>10.3f}")
            cfgs.append((f"{nm}@{cap}", cfg, ok))
            table[f"{nm}@{cap}"] = {
                "n_A": int(ra["n"]), "roi_A": float(ra["roi"]),
                "n_B": int(rb["n"]), "roi_B": float(rb["roi"]),
                "both_positive": bool(ok), "roi_pooled": float(rp["roi"]),
                "roi_pooled_fair": float(rp["roi_fair"]), "p_noise": pn}
    print(f"\n  configs positive at vig in BOTH halves: {obs_hold} of "
          f"{len(cfgs)}")
    for k, _, ok in cfgs:
        if ok:
            print(f"    HOLDS: {k}")

    # How many WOULD hold under the null?  Game-level simulation with
    # y ~ Bernoulli(p_mkt) and FAIR pricing (so a config is exactly
    # breakeven), counting configs positive in both halves per replicate.
    rng = np.random.default_rng(SEED + 7)
    sub = df[df.season.isin(ALL_SEASONS)]
    pos = {g: i for i, g in enumerate(sub.game_id.values)}
    isA = sub.season.isin(WIN_A).values
    NREP = 4000
    Y = rng.random((NREP, len(sub))) < sub.p_mkt.values
    packs = []
    for _, cfg, _ in cfgs:
        b = Strategy(cfg).bets(sub)
        if len(b) == 0:
            continue
        j = np.array([pos[g] for g in b.game_id.values])
        packs.append((np.asarray(b.bet_home, bool), j, b.dec_fair.values,
                      isA[j]))
    cnt = np.zeros(NREP, int)
    for bh, j, dfair, ina in packs:
        win = np.where(bh[None, :], Y[:, j], ~Y[:, j])
        pnl = np.where(win, dfair - 1, -1.0)
        ra = pnl[:, ina].mean(axis=1) if ina.sum() else np.zeros(NREP)
        rb = pnl[:, ~ina].mean(axis=1) if (~ina).sum() else np.zeros(NREP)
        cnt += ((ra > 0) & (rb > 0)).astype(int)
    print(f"\n  NULL CHECK (4000 replicates, y ~ Bernoulli(p_mkt), fair "
          f"pricing so every config is exactly breakeven):")
    print(f"    expected # of the {len(packs)} configs holding both halves "
          f"BY CHANCE = {cnt.mean():.2f}  (median {int(np.median(cnt))}, "
          f"90th pct {int(np.percentile(cnt, 90))})")
    print(f"    observed = {obs_hold}")
    print(f"    P(chance produces >= {obs_hold}) = "
          f"{float((cnt >= obs_hold).mean()):.3f}")
    results["d112_convergence"] = {
        "table": table, "observed_holding": int(obs_hold),
        "n_configs": len(packs), "null_mean_holding": float(cnt.mean()),
        "p_chance_ge_observed": float((cnt >= obs_hold).mean())}
    return results["d112_convergence"]


def curve_report(df, cfg, seasons, label):
    st = Strategy(cfg)
    b = st.bets(df[df.season.isin(seasons)])
    r = st.simulate(b)
    print(f"\n  {label}: {cfg.name} on {'+'.join(seasons)}")
    if r["n"] == 0:
        print("    no bets")
        return r
    print(f"    n={r['n']} hit={r['hit']*100:.1f}% ROI={r['roi']*100:+.2f}% "
          f"fair={r['roi_fair']*100:+.2f}% bankroll {BANKROLL0:.0f} -> "
          f"{r['final']:.1f} ({r['ret']*100:+.1f}%) maxDD={r['maxdd']*100:.1f}%")
    c = r["curve"]
    step = max(1, len(c) // 24)
    print("    equity curve (sampled): "
          + " ".join(f"{v:.0f}" for v in c[::step]))
    return r


# ==========================================================================
def main():
    print("=" * 96)
    print("TS-STRATEGY — bet-at-close strategy designed around our losses")
    print("=" * 96)
    df = build_frame(rebuild="--rebuild" in sys.argv)
    print(f"frame: {len(df)} games, seasons "
          f"{sorted(df.season.unique())}")
    print(f"veto legs on {len(df[df.season.isin(ALL_SEASONS)])} scored games: "
          f"EARLY {(df[df.season.isin(ALL_SEASONS)].gp_min < EARLY_GP).mean()*100:.1f}%"
          f"  CHAOS {(df[df.season.isin(ALL_SEASONS)].chaos >= CHAOS_M5).mean()*100:.1f}%"
          f"  FRESH {(df[df.season.isin(ALL_SEASONS)].fresh >= FRESH_MIN).mean()*100:.1f}%")

    results = {}
    results["diagnostic"] = tail_diagnostic(df)

    fam = config_family()
    print(f"\npre-registered family: {[c.name for c in fam]}")

    evaluate(df, fam, WIN_A, WIN_B, "DIRECTION 1 (the directive's split)",
             results)
    evaluate(df, fam, WIN_B, WIN_A, "DIRECTION 2 (reversed)", results)
    results["select_1"] = selection_protocol(
        df, fam, WIN_A, WIN_B, "DIRECTION 1: IS 22-23+23-24 -> OOS 24-25+25-26")
    results["select_2"] = selection_protocol(
        df, fam, WIN_B, WIN_A, "DIRECTION 2: IS 24-25+25-26 -> OOS 22-23+23-24")
    results["per_season"] = per_season_table(df, fam)
    results["decay"] = season_decay(df, fam)
    results["breakeven"] = breakeven_vig(df, fam, ALL_SEASONS)
    results["sensitivity"] = {
        c.name: sensitivity(df, c, ALL_SEASONS)
        for c in fam if c.name in ("FAV.veto", "FAV.veto.late")}
    verdict(df, fam, results)
    d112_convergence(df, results)

    print("\n" + "=" * 96)
    print("BANKROLL CURVES — full 4-season run of every config")
    print("=" * 96)
    curves = {}
    for cfg in fam:
        r = curve_report(df, cfg, ALL_SEASONS, "full sample")
        curves[cfg.name] = {"final": r["final"], "maxdd": r["maxdd"],
                            "n": r["n"], "roi": r["roi"],
                            "roi_fair": r["roi_fair"], "hit": r["hit"]}
    results["full_sample"] = curves

    # persist the actual bet log of every config for audit
    logs = []
    for cfg in fam:
        st = Strategy(cfg)
        b = st.bets(df[df.season.isin(ALL_SEASONS)])
        if len(b):
            b = b.copy(); b["config"] = cfg.name
            logs.append(b)
    if logs:
        pd.concat(logs).to_csv(OUT_BETS, index=False)
        print(f"\nwrote {os.path.relpath(OUT_BETS, ROOT)} "
              f"({sum(len(x) for x in logs)} bet rows)")

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"wrote {os.path.relpath(OUT_JSON, ROOT)}")


if __name__ == "__main__":
    main()
