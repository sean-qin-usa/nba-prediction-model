"""2K ratings scrape from 2kratings.com (Cloudflare -> cloudscraper).

Contract (handoff III.1): archive raw HTML (site markup is fragile), version by
scrape date. Parsing rides the schema.org JSON-LD embedded in every page —
team pages carry the current roster (athlete URLs), player pages carry all
~38 attributes as PropertyValue pairs. If JSON-LD disappears in a redesign we
re-parse from the archived HTML.

Note: 2kratings has NO tendency data (attributes + badges only). Tendencies
(II.3 action-selection priors) need a separate source — flagged in README.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import random
import re
import time

import cloudscraper

from ..config import RAW_2K

log = logging.getLogger("ratings_2k")

TEAM_SLUGS = [
    "atlanta-hawks", "boston-celtics", "brooklyn-nets", "charlotte-hornets",
    "chicago-bulls", "cleveland-cavaliers", "dallas-mavericks", "denver-nuggets",
    "detroit-pistons", "golden-state-warriors", "houston-rockets", "indiana-pacers",
    "los-angeles-clippers", "los-angeles-lakers", "memphis-grizzlies", "miami-heat",
    "milwaukee-bucks", "minnesota-timberwolves", "new-orleans-pelicans",
    "new-york-knicks", "oklahoma-city-thunder", "orlando-magic",
    "philadelphia-76ers", "phoenix-suns", "portland-trail-blazers",
    "sacramento-kings", "san-antonio-spurs", "toronto-raptors", "utah-jazz",
    "washington-wizards",
]

_LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)


def _scraper():
    return cloudscraper.create_scraper()


def _fetch_and_archive(scraper, url: str, scrape_date: str) -> tuple[str, str]:
    """GET url, archive raw HTML under RAW_2K/<scrape_date>/, return (html, relpath)."""
    r = scraper.get(url, timeout=30)
    r.raise_for_status()
    day_dir = RAW_2K / scrape_date
    day_dir.mkdir(parents=True, exist_ok=True)
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    path = day_dir / f"{slug}.html"
    path.write_text(r.text)
    time.sleep(1.2 + random.uniform(0, 1.5))  # polite crawl
    return r.text, str(path.relative_to(RAW_2K))


def _ld_nodes(html: str) -> list[dict]:
    nodes = []
    for block in _LD_RE.findall(html):
        try:
            d = json.loads(block)
        except json.JSONDecodeError:
            continue
        nodes.extend(d.get("@graph", [d]))
    return nodes


def parse_team(html: str) -> list[dict]:
    """Current roster: [{name, url, overall}] from the SportsTeam JSON-LD node."""
    for node in _ld_nodes(html):
        if node.get("@type") == "SportsTeam":
            out = []
            for a in node.get("athlete", []):
                overall = None
                for p in a.get("additionalProperty", []):
                    if "Rating" in p.get("name", ""):
                        overall = p.get("value")
                out.append({"name": a.get("name"), "url": a.get("url"), "overall": overall})
            return out
    return []


def parse_player(html: str) -> dict | None:
    """{name, edition, overall, attributes{}} from the Person JSON-LD node."""
    for node in _ld_nodes(html):
        if node.get("@type") == "Person":
            attrs, overall, edition = {}, None, None
            for p in node.get("additionalProperty", []):
                name, val = p.get("name", ""), p.get("value")
                m = re.match(r"(NBA 2K\d+) Rating", name)
                if m:
                    edition, overall = m.group(1), val
                elif name.endswith(" Attribute"):
                    attrs[name[: -len(" Attribute")]] = val
            import html as _html
            return {"name": _html.unescape(node.get("name") or ""), "edition": edition,
                    "overall": overall, "attributes": attrs}
    return None


def slugify(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[-\s]+", " ", s.casefold())
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return "-".join(s.split())


def scrape_free_agents(con, names: list[str]) -> int:
    """Targeted fallback: fetch player pages directly by slugified name (players
    off any roster — free agents — have pages but appear on no team page).
    Stored with team_slug='free-agent'."""
    scrape_date = dt.date.today().isoformat()
    scraper = _scraper()
    n = 0
    for name in names:
        url = f"https://www.2kratings.com/{slugify(name)}"
        try:
            phtml, relpath = _fetch_and_archive(scraper, url, scrape_date)
        except Exception as e:  # noqa: BLE001
            log.info("no page for %s (%s)", name, e)
            continue
        player = parse_player(phtml)
        if not player or not player.get("attributes"):
            continue
        con.execute(
            "DELETE FROM ratings_2k WHERE scrape_date = ? AND player_name = ? AND team_slug = ?",
            [scrape_date, player["name"], "free-agent"])
        con.execute(
            "INSERT INTO ratings_2k VALUES (?,?,?,?,?,?,?,?,?,?)",
            [scrape_date, player["edition"] or "unknown", player["name"], "free-agent",
             player["overall"], json.dumps(player["attributes"]), None,
             url, relpath, dt.datetime.now(dt.timezone.utc)])
        n += 1
    return n


def scrape_all(con, teams: list[str] | None = None) -> int:
    """Full crawl: 30 team pages -> player pages -> ratings_2k rows. Idempotent
    per (scrape_date, player): re-running the same day overwrites."""
    scrape_date = dt.date.today().isoformat()
    scraper = _scraper()
    n = 0
    for slug in teams or TEAM_SLUGS:
        try:
            html, _ = _fetch_and_archive(
                scraper, f"https://www.2kratings.com/teams/{slug}", scrape_date)
        except Exception:
            log.exception("team page failed: %s", slug)
            continue
        roster = parse_team(html)
        log.info("%s: %d players", slug, len(roster))
        for entry in roster:
            try:
                phtml, relpath = _fetch_and_archive(scraper, entry["url"], scrape_date)
            except Exception:
                log.exception("player page failed: %s", entry["url"])
                continue
            player = parse_player(phtml)
            if not player or not player.get("attributes"):
                log.warning("no attributes parsed: %s", entry["url"])
                continue
            con.execute(
                "DELETE FROM ratings_2k WHERE scrape_date = ? AND player_name = ? AND team_slug = ?",
                [scrape_date, player["name"], slug])
            con.execute(
                "INSERT INTO ratings_2k VALUES (?,?,?,?,?,?,?,?,?,?)",
                [scrape_date, player["edition"] or "unknown", player["name"], slug,
                 player["overall"], json.dumps(player["attributes"]), None,
                 entry["url"], relpath, dt.datetime.now(dt.timezone.utc)])
            n += 1
    return n
