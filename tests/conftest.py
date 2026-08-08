"""Test configuration (D229).

THE ONE THING THIS DOES: turn "the corpus database is not in this checkout" into
a SKIP rather than a failure, and nothing else.

`data/nba.duckdb` is 262 MB and is correctly not committed, so a fresh clone of
the public repository cannot run the corpus-level assertions.  Some test files
guard that themselves (`except Exception: pytest.skip`), but most call
`nbapred.db.connect()` inline and simply blew up — 25 failures that told a reader
the project was broken when the only thing missing was a file the repository
deliberately does not ship.

The conversion is deliberately NARROW.  It fires only on the sentinel
`FileNotFoundError` that `connect(read_only=True)` raises for an absent database,
and only in a checkout where that database is genuinely absent.  Every other
error, including any other FileNotFoundError, fails exactly as before — so this
cannot hide a real regression, and in the working tree (where the DB exists) the
hook is unreachable by construction.
"""
from __future__ import annotations

import pytest

from nbapred.config import DB_PATH

#: `connect(read_only=True)`'s message, and DuckDB's own for the few call sites
#: that open the file directly and so never reach our wrapper.
_SENTINELS = ("does not exist — nothing to read",
              "database does not exist")


def _is_absent_db(exc_value) -> bool:
    msg = str(exc_value)
    if not any(s in msg for s in _SENTINELS):
        return False
    return isinstance(exc_value, FileNotFoundError) or \
        type(exc_value).__name__.endswith("IOException")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if DB_PATH.exists() or report.when != "call" or not report.failed:
        return
    exc = getattr(call, "excinfo", None)
    if exc is None or not _is_absent_db(exc.value):
        return
    report.outcome = "skipped"
    report.longrepr = (str(item.fspath), item.location[1] + 1,
                       f"skipped: {DB_PATH.name} not in this checkout "
                       f"(262 MB, deliberately uncommitted)")
