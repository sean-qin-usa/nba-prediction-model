"""Kaggle dataset download via the logged-in Chrome session (no API token, $0).

Reads Kaggle cookies from the Chrome profile that's signed in (auto-detected)
and hits the web download endpoint. Falls back to the official `kaggle` CLI if
~/.kaggle/kaggle.json exists. Downloads land in data/raw/kaggle/<slug>/.

D177: the Chrome-cookie route is DEAD on this box (`no logged-in Kaggle Chrome
profile found`) and `docs/OPENING_LINES.md` §4 already recorded that. But
Kaggle's own **public API needs no auth at all** for search, metadata and the
full-dataset zip: `GET https://www.kaggle.com/api/v1/datasets/{list,view,
download}/...` all return 200 anonymously. `_anon_session` is therefore the
fallback, and `search`/`view` are exposed so a dataset can be EVALUATED before
72 MB is pulled. (kaggle.com serves no robots.txt — `/robots.txt` soft-404s to
the SPA shell — and `/api/v1/` is Kaggle's own documented public API, whose
entire purpose is dataset download.)
"""
from __future__ import annotations

import glob
import logging
import os
import zipfile

import browser_cookie3
import requests

from ..config import RAW

log = logging.getLogger("kaggle_web")

RAW_KAGGLE = RAW / "kaggle"
RAW_KAGGLE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"}


def _find_session() -> requests.Session:
    """Locate a Chrome profile with a live kaggle.com session; return a Session."""
    profiles = glob.glob(os.path.expanduser("~/.config/google-chrome/*/Cookies"))
    profiles += glob.glob(os.path.expanduser("~/.config/chromium/*/Cookies"))
    for cf in sorted(profiles):
        try:
            cj = browser_cookie3.chrome(cookie_file=cf, domain_name="kaggle.com")
        except Exception:  # noqa: BLE001
            continue
        names = {c.name: c.value for c in cj}
        if "__Host-KAGGLEID" not in names:
            continue
        s = requests.Session()
        s.cookies.update(names)
        s.headers.update(UA)
        r = s.post("https://www.kaggle.com/api/i/users.UsersService/GetCurrentUser",
                   json={}, headers={"X-XSRF-TOKEN": names.get("XSRF-TOKEN", "")}, timeout=20)
        if r.status_code == 200 and '"email"' in r.text:
            log.info("kaggle session via %s", cf.split("/")[-2])
            return s
    raise RuntimeError("no logged-in Kaggle Chrome profile found")


def _anon_session() -> requests.Session:
    """A plain session against Kaggle's public API - no cookie, no token."""
    s = requests.Session()
    s.headers.update(UA)
    return s


def _session() -> requests.Session:
    """Logged-in Chrome session if one exists, else the anonymous public API."""
    try:
        return _find_session()
    except Exception as exc:  # noqa: BLE001
        log.info("no Chrome kaggle session (%s); using anonymous public API", exc)
        return _anon_session()


def search(term: str, page: int = 1, page_size: int = 100) -> list[dict]:
    """Public dataset search. Works with no auth. Returns the raw JSON rows."""
    r = _anon_session().get("https://www.kaggle.com/api/v1/datasets/list",
                            params={"search": term, "page": page,
                                    "pageSize": page_size}, timeout=60)
    r.raise_for_status()
    return r.json()


def view(slug: str) -> dict:
    """Public dataset metadata (title, licence, size, description, versions).

    Use this to EVALUATE a candidate before downloading it. NOTE the file list
    comes back EMPTY from this endpoint for most datasets - the only reliable
    way to see file names is to download the zip and read its namelist.
    """
    r = _anon_session().get(f"https://www.kaggle.com/api/v1/datasets/view/{slug}",
                            timeout=60)
    r.raise_for_status()
    return r.json()


def download_dataset(slug: str) -> list[str]:
    """Download+extract a dataset (owner/name). Returns extracted file paths."""
    out_dir = RAW_KAGGLE / slug.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "dataset.zip"

    s = _session()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    with s.get(url, timeout=600, stream=True) as r:
        r.raise_for_status()
        if "zip" not in r.headers.get("content-type", ""):
            raise RuntimeError(f"unexpected content-type for {slug}: {r.headers.get('content-type')}")
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
        members = z.namelist()
    log.info("%s: %d files (%d bytes zip)", slug, len(members), zip_path.stat().st_size)
    return [str(out_dir / m) for m in members]
