"""D140 — what the neutral-site fix actually does to D136's margin readout.

D139 registered the bubble travel fiction (1,505.5 km/team-game vs a true 0)
right next to the finding that D136's two "SIG" coefficients live only in the
COVID-inclusive frame, which invites the reading that the fiction CAUSED the
significance. This script tests that reading directly instead of assuming it.

READ-ONLY. Writes data/tv_bubble_correction.json + the log.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.model import travel as TV
from tv_margin_fit import BASE, ARMS, PRED, ols, frame   # noqa: E402

OUT = ROOT / "data" / "tv_bubble_correction.json"


def fit_one(rows, y, cols):
    beta, se, sd, n = ols(rows, y, cols + ["qd"])
    names = ["const"] + cols + ["qd"]
    return {c: dict(beta=float(beta[i]), se=float(se[i]),
                    t=float(beta[i] / se[i]),
                    lo=float(beta[i] - 1.96 * se[i]),
                    hi=float(beta[i] + 1.96 * se[i]),
                    sig=bool((beta[i] - 1.96 * se[i]) * (beta[i] + 1.96 * se[i]) > 0))
            for i, c in enumerate(names)}, n


def main():
    con = connect(read_only=True)
    res = {}

    # ---- 1. how much of the frame does the bubble actually occupy? --------
    rows, y, seas = frame(con, None)
    st = TV.build_state(con)
    gm = con.execute("""
        WITH t AS (SELECT season, game_id, game_date, team_id, is_home
                   FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL)
        SELECT h.season, h.game_date, h.team_id, a.team_id
        FROM t h JOIN t a USING (game_id)
        WHERE h.is_home AND NOT a.is_home ORDER BY h.game_date""").fetchall()
    con.close()

    keep = []
    for season, d, ht, at_ in gm:
        d = d.date() if hasattr(d, "date") else d
        sh, sa = st.get((ht, d)), st.get((at_, d))
        if sh is None or sa is None:
            continue
        keep.append((season, d, sh, sa))
    assert len(keep) == len(rows), (len(keep), len(rows))

    is_bubble = np.array([TV.BUBBLE_FROM <= d <= TV.BUBBLE_TO for _, d, _, _ in keep])
    bad_geo = np.array([not (sh["travel_valid"] and sa["travel_valid"])
                        for _, _, sh, sa in keep])
    print(f"frame n={len(rows)}   bubble rows={int(is_bubble.sum())} "
          f"({is_bubble.mean():.2%})   unknown-geo rows={int(bad_geo.sum())} "
          f"({bad_geo.mean():.3%})")
    dtr = np.array([r["dtrav_kkm"] for r in rows])
    print(f"dtrav_kkm on bubble rows AFTER fix: max|.|={np.abs(dtr[is_bubble]).max():.6f} "
          f"(all teams share one venue)")
    print(f"dtrav_kkm sd, whole frame {dtr.std():.4f}; "
          f"non-bubble {dtr[~is_bubble].std():.4f}")
    res["frame"] = dict(n=len(rows), n_bubble=int(is_bubble.sum()),
                        n_unknown_geo=int(bad_geo.sum()),
                        dtrav_sd=float(dtr.std()))

    # ---- 2. the same arms on nested frames --------------------------------
    universes = {
        "FULL 2019-20..2025-26 (corrected)": np.ones(len(rows), bool),
        "FULL minus the 88 bubble games": ~is_bubble,
        "FULL minus bubble minus unknown-geo": ~(is_bubble | bad_geo),
        "SCORED 2021-26 (corrected)": np.isin(
            seas, ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]),
        "COVID ERAS ONLY 2019-20+2020-21": np.isin(seas, ["2019-20", "2020-21"]),
    }
    print()
    for label, msk in universes.items():
        sub = [r for r, m in zip(rows, msk) if m]
        ysub = y[msk]
        entry = {}
        for arm in ("A", "D", "B"):
            terms, n = fit_one(sub, ysub, BASE + ARMS[arm])
            for t in ARMS[arm]:
                entry[t] = terms[t]
                entry[t]["n"] = n
        res[label] = entry
        print(f"{label:38s} n={entry['dtrav_kkm']['n']:5d}  "
              + "  ".join(
                  f"{t} {entry[t]['beta']:+7.4f} t={entry[t]['t']:+5.2f} "
                  f"{'SIG' if entry[t]['sig'] else 'ns '}"
                  for t in ("dtrav_kkm", "d3in4")))

    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
