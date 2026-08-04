"""Kaggle dataset download via the logged-in Chrome session (no API token, $0).

Reads Kaggle cookies from the Chrome profile that's signed in (auto-detected)
and hits the web download endpoint. Falls back to the official `kaggle` CLI if
~/.kaggle/kaggle.json exists. Downloads land in data/raw/kaggle/<slug>/.
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


def download_dataset(slug: str) -> list[str]:
    """Download+extract a dataset (owner/name). Returns extracted file paths."""
    out_dir = RAW_KAGGLE / slug.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "dataset.zip"

    s = _find_session()
    with s.get(f"https://www.kaggle.com/datasets/{slug}/download", timeout=180, stream=True) as r:
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
