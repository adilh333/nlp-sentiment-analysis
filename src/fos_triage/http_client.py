"""Polite HTTP client for the FOS decisions register.

One request at a time, a fixed gap between requests, on-disk cache, and a
short retry on 429/503. robots.txt is fetched once and the few disallowed
form PDFs are refused.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from fos_triage.config import ROBOTS_URL, USER_AGENT

ALLOWED_HOST = "www.financial-ombudsman.org.uk"


class PoliteClient:
    def __init__(self, delay_s: float = 0.75, cache_dir: Path | None = None) -> None:
        self.delay_s = delay_s
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
            }
        )
        self._last = 0.0
        self._robots = RobotFileParser()
        self.requests_made = 0
        self.cache_hits = 0

    def check_robots(self) -> None:
        response = self.session.get(ROBOTS_URL, timeout=30)
        response.raise_for_status()
        self._robots.parse(response.text.splitlines())
        self._last = time.monotonic()

    def allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host and host != ALLOWED_HOST:
            return False
        if not self._robots.default_entry and not self._robots.entries:
            return True
        return bool(self._robots.can_fetch(USER_AGENT, url))

    def get(self, url: str, params: list[tuple[str, str]] | None = None, cache_key: str | None = None) -> bytes:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        path = self._cache_path(cache_key) if cache_key and self.cache_dir else None
        if path is not None and path.exists() and path.stat().st_size > 0:
            self.cache_hits += 1
            return path.read_bytes()

        self._wait()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=60)
                self.requests_made += 1
                if response.status_code in {429, 503}:
                    time.sleep(20 * (attempt + 1))
                    last_error = requests.HTTPError(f"{response.status_code} from {url}")
                    continue
                response.raise_for_status()
                content = response.content
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                return content
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last = time.monotonic()

    def _cache_path(self, cache_key: str) -> Path:
        assert self.cache_dir is not None
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        if cache_key.lower().endswith(".pdf") or cache_key.startswith("pdf:"):
            name = cache_key.split(":", 1)[-1]
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            return self.cache_dir / "pdf" / f"{safe}.pdf"
        return self.cache_dir / "html" / f"{digest}.html"
