"""StateBank — the one shared state container every v3 head reads (V3_SPEC 2.1).

M0 scope: the container, its evolution step (generic per-dim AR(1) + process
noise + event-shock inflation), and DuckDB persistence (player_states table).
The player-level OBSERVATION updates (update_day EKF, update_stints ridge) are
M3 and raise NotImplementedError until then — the team-level M1 pilot lives in
team_dlm.py and persists through the same table.

State per player p: theta[p] in R^18 (diagonal blocks, V3_SPEC 2.1):
    impact   net_o, net_d            (pts/100poss)
    minutes  min_mu, min_lsd         (E[min|active], log sd)
    usage    u                       (softmax weight, D32 scale)
    shooting sh_rim, sh_mid, sh_thr, sh_ft          (logits)
    volume   r_rim, r_mid, r_thr, r_fta, r_tov, r_oreb, r_dreb, r_ast,
             r_stl, r_blk            (log rates per minute)
Per team: pace (poss/48) as (mean, var).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .hyper import HyperParams

DIMS = ("net_o", "net_d",
        "min_mu", "min_lsd",
        "u",
        "sh_rim", "sh_mid", "sh_thr", "sh_ft",
        "r_rim", "r_mid", "r_thr", "r_fta", "r_tov", "r_oreb", "r_dreb",
        "r_ast", "r_stl", "r_blk")

BLOCK_OF = {**{d: "impact" for d in DIMS[:2]},
            **{d: "minutes" for d in DIMS[2:4]},
            "u": "usage",
            **{d: "shooting" for d in DIMS[5:9]},
            **{d: "volume" for d in DIMS[9:]}}

_TEAM_DIMS = ("pace", "team_off", "team_def", "league_mu", "home_edge")


class StateBank:
    """theta/P: {player_id: np.ndarray(18)} posterior mean / variance (diagonal
    per-dim, V3_SPEC filter design); pace: {team_id: (mu, var)}; asof: the date
    the states are valid AT (post games of asof-1, pre games of asof)."""

    def __init__(self, asof: dt.date):
        self.asof = asof
        self.theta: dict[int, np.ndarray] = {}
        self.P: dict[int, np.ndarray] = {}
        self.anchor: dict[int, np.ndarray] = {}      # hierarchical long-run mean
        self.pace: dict[int, tuple[float, float]] = {}

    # ------------------------------------------------------------------ setup
    def add_player(self, pid: int, anchor: np.ndarray | None = None,
                   prior_var: np.ndarray | None = None) -> None:
        """Cold start: theta sits AT the anchor with wide P (D16 absorbed)."""
        a = np.zeros(len(DIMS)) if anchor is None else np.asarray(anchor, float)
        v = (np.ones(len(DIMS)) if prior_var is None
             else np.asarray(prior_var, float))
        self.anchor[int(pid)] = a.copy()
        self.theta[int(pid)] = a.copy()
        self.P[int(pid)] = v.copy()

    # -------------------------------------------------------------- evolution
    def predict_to(self, date: dt.date, hp: HyperParams,
                   shocks: dict | None = None) -> None:
        """theta_d <- anchor_d + phi_d^dt (theta_d - anchor_d);
        P_d <- P_d + Q_class(d) * dt * S(p, t).
        shocks: {player_id: lam} Q-inflation multipliers active over the gap
        (event-shock windows, detect_shocks output)."""
        days = (date - self.asof).days
        if days < 0:
            raise ValueError(f"predict_to moving backward: {self.asof} -> {date}")
        if days == 0:
            return
        phi = np.array([hp.phi[BLOCK_OF[d]] for d in DIMS])
        q = np.array([hp.Q[BLOCK_OF[d]] for d in DIMS])
        shrink = phi ** days
        for pid in self.theta:
            lam = 1.0 if not shocks else float(shocks.get(pid, 1.0))
            a = self.anchor[pid]
            self.theta[pid] = a + shrink * (self.theta[pid] - a)
            self.P[pid] = self.P[pid] + q * days * lam
        self.asof = date

    # ------------------------------------------------------------ observation
    def update_day(self, obs_df, hp: HyperParams) -> None:
        """Per-player-game EKF (binomial/Poisson/minutes likelihoods) — M3."""
        raise NotImplementedError("M3 — player-level observation update")

    def update_stints(self, stints_df, hp: HyperParams) -> None:
        """Weekly stint-margin ridge pseudo-obs on net dims — M3."""
        raise NotImplementedError("M3 — stint ridge update")

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, con, asof: dt.date) -> "StateBank":
        """Rebuild from player_states rows at `asof` (read-only connection)."""
        rows = con.execute(
            """SELECT player_id, team_id, dim, mean, var FROM player_states
               WHERE "asof" = ? AND dim NOT IN ('team_off','team_def',
                                              'league_mu','home_edge')""",
            [asof]).fetchall()
        bank = cls(asof)
        acc: dict[int, dict[str, tuple[float, float]]] = {}
        for pid, tid, dim, mean, var in rows:
            if dim == "pace":
                bank.pace[int(pid)] = (float(mean), float(var))
                continue
            acc.setdefault(int(pid), {})[dim] = (float(mean), float(var))
        for pid, dims in acc.items():
            th = np.zeros(len(DIMS)); pv = np.ones(len(DIMS))
            for i, d in enumerate(DIMS):
                if d in dims:
                    th[i], pv[i] = dims[d]
            bank.theta[pid] = th
            bank.P[pid] = pv
            bank.anchor[pid] = th.copy()      # anchors re-derived at fit time
        return bank

    def snapshot(self, con_rw) -> None:
        """Persist to player_states at self.asof (idempotent: replace-by-date).
        con_rw MUST come from schema.v3_writer() (single-writer discipline)."""
        con_rw.execute('DELETE FROM player_states WHERE "asof" = ? AND dim '
                       "NOT IN ('team_off','team_def','league_mu','home_edge')",
                       [self.asof])
        rows = []
        for pid, th in self.theta.items():
            pv = self.P[pid]
            for i, d in enumerate(DIMS):
                rows.append((self.asof, pid, None, d, float(th[i]), float(pv[i])))
        for tid, (mu, var) in self.pace.items():
            rows.append((self.asof, tid, tid, "pace", mu, var))
        if rows:
            con_rw.executemany(
                "INSERT INTO player_states VALUES (?, ?, ?, ?, ?, ?)", rows)
