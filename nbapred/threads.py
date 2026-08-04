"""Thread-budget control for a SHARED box.

Why this exists (2026-08-02): the machine is 16 cores and runs several projects
at once. A single numpy/BLAS call defaults to one thread per core, so a handful
of concurrent gate scripts oversubscribe badly — a small matrix-solve loop that
runs in 0.8s pinned to one thread was measured taking >120s at load average 56.
Oversubscription is not just slow, it is *non-linearly* slow: threads spin
waiting on each other.

Two ways to use it, and the ORDER MATTERS:

  1. Best — before numpy is imported (env vars are read at BLAS load time):
         from nbapred.threads import pin
         pin()                      # then import numpy
  2. Anytime — runtime override via threadpoolctl, which reaches into an
     already-loaded BLAS:
         from nbapred.threads import limit
         with limit(1):
             ...heavy linear algebra...

`pin()` is a no-op if the variables are already set, so an operator can always
override from the shell or from cron without editing code.

Rule of thumb for this repo: gate/backtest scripts that loop over many SMALL
solves want 1 thread (the parallelism should be across games/seasons, not
inside a 30x30 matrix). A single big fit wants more.
"""
from __future__ import annotations

import contextlib
import os

_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

DEFAULT = max(1, (os.cpu_count() or 4) // 8)   # 16 cores -> 2


def pin(n: int | None = None) -> int:
    """Set the BLAS thread env vars. Call BEFORE importing numpy.

    Existing values are never overwritten, so `OMP_NUM_THREADS=8 python ...`
    still wins. Returns the value in force.
    """
    n = DEFAULT if n is None else max(1, int(n))
    for v in _VARS:
        os.environ.setdefault(v, str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return int(os.environ["OMP_NUM_THREADS"])


@contextlib.contextmanager
def limit(n: int = 1):
    """Runtime thread cap around a block, effective even after numpy is loaded.

    Falls back to a no-op (with the env still pinned) when threadpoolctl is
    absent, so callers never need to guard the import.
    """
    n = max(1, int(n))
    try:
        from threadpoolctl import threadpool_limits
    except Exception:            # noqa: BLE001 - optional dependency
        pin(n)
        yield
        return
    with threadpool_limits(limits=n):
        yield
