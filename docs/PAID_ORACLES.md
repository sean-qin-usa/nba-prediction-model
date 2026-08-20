> **DATA-COVERAGE CAVEAT — READ BEFORE QUOTING ANY NUMBER IN THIS FILE.**
> The daily NBA injury report, which the model's availability leg depends on,
> begins **2018-12-17 — mid-way through 2018-19**. Coverage of regular-season
> game dates is **0% before that, 63.7% in 2018-19, and 95–100% from 2019-20
> onward**. Only **2019-20 → 2025-26 (7 seasons)** is fully covered, and that is
> the only frame in which the model runs as designed. Earlier seasons score a
> *crippled variant* whose availability leg is fed inputs it was never meant to
> have. Any figure here spanning seasons before 2019-20 — including every
> 14-season and 19-season figure — blends two different models and should be
> read as historical context, not as a description of the shipped system.
> (`D186`)


## Extended ledger — what a DraftKings-tier shop can actually buy (2026-07-30)
| # | Info | Purchasable? | Product | Oracle stand-in | Status |
|---|---|---|---|---|---|
| 7 | Full player tracking (XY, matchups/poss, shot quality qSQ) | YES — league-approved vendors | Sportradar / Second Spectrum license | D36 team on-ball margin term, walk-forward k (season hindsight; scripts/oracle_tracking.py) | MEASURED — pooled +0.0004 NS CI(-0.0009,+0.0018); heavy-fav n=519 +0.0059 SIG CI(+0.0026,+0.0090), all 3 seasons positive there; chaos-teams NS -0.0008 |
| 8 | News SPEED (scratch/lineup seconds after break) | YES | Sportradar live injury wire, FantasyLabs alerts | actual played-set (= #1) | RE-DENOMINATED D158 — the old "residual +0.0037" was measured against a baseline that was ITSELF the played set. Honest value = C1−T2 = **+0.00386 pooled / +0.00235 on full-feed seasons** |
| 9 | Biometric / load mgmt (Kinexon, Catapult) | NO — league-restricted, not sold to bettors | n/a | n/a — noted as non-purchasable | closed |
| 10 | TRUE insider info (pre-announcement intent) | NO — not a legal product; shops buy SPEED not secrets | n/a | bounded above by #1+#2 oracles combined | closed |
| 11 | Sharp market screens / steam (Don Best-style) | YES but FORBIDDEN as model input (market-blind rule) | odds feeds | n/a for model; CLV/timing only | policy |
| 12 | College/G-League full data (Synergy college) | YES | Synergy Sports college | free sports-reference CBB covers translation needs | free path |
| 13 | Team charter/arrival logistics | YES (FlightAware etc.) | flight trackers | schedule-derived travel (free) ≈ upper bound; delays marginal | skip (tiny) |

## COMMITTED WOULD-BUY (Sean 2026-07-30): two-tier model policy
### CORRECTED 2026-08-03 (D158) — read this before quoting the tiers below
This section used to say "THE HEADLINE CAPSTONE **IS** the bought-availability
tier already (oracle outs are its default)". That was true, and it was the
DEFECT: the certified number was a played-set-oracle number in violation of
docs/LEAKAGE.md:131. As of D158 the polarity is reversed and the register now
reads:
- **THE CERTIFIED CAPSTONE IS THE FREE TIER.** `prod_by_season.py` defaults to
  T2-HONEST (5PM injury report ∪ official pregame inactives). Certified D158:
  **0.60750 pooled, normalized gap 14.95%** (2021-22 0.63053 / 2022-23 0.63385 /
  2023-24 0.59906 / 2024-25 0.58857 / 2025-26 0.58553).
- **THE BOUGHT-AVAILABILITY TIER IS A LABELLED CEILING**, reachable only via
  `ORACLE_PLAYED_OUTS=1`, artifact `data/capstone_pergame_oracle_ceiling.csv`:
  0.60364 pooled, 11.13%. It is NOT a model result and must not be certified.
- **PRICE-WORTH OF THE AVAILABILITY WIRE, measured honestly** =
  the C1−T2 distance = **+0.00386 pooled log loss** on the 5-season corpus,
  **+0.00235** on the three seasons that actually carry both feeds (D156
  measured +0.00243 independently on its own frame — the two agree). Note this
  is a CEILING on the wire: it prices *perfect* availability, and D156 §11(b)
  found the residual after it is bought is still 10.42% normalized.
- Tracking tier unchanged (row 7 above), still measured, still not bought.
Every future capstone reports both tiers **and names the tier in its own run
header**. Purchases logged as commitments, not made — $0 constraint stands.
