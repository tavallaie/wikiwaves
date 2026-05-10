"""WikiFetcher client — all external Wikipedia / Wikimedia API calls."""

from __future__ import annotations

import datetime
import hashlib
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger

from .exceptions import FetcherError, MalformedResponseError, NetworkError, PageNotFoundError, RateLimitError
from .models import OnThisDayEvent, PageMetrics, TrendingEdit, WikiPage


class WikiFetcher:
    """Dumb pipe for Wikipedia data. No business logic, no ranking, no filtering.

    Fetches from three API families:
    * Wikimedia REST API (featured / on-this-day feeds)
    * MediaWiki Action API (page content, search, recent changes)
    * Wikimedia PageViews API (traffic statistics)
    """

    # Endpoints
    REST_BASE = "https://api.wikimedia.org/feed/v1/wikipedia/en"
    API_BASE = "https://en.wikipedia.org/w/api.php"
    PAGEVIEWS_BASE = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia/all-access/all-agents"
    )

    # Tunable defaults
    DEFAULT_RETRIES = 3
    DEFAULT_BACKOFF = 1.0
    DEFAULT_TIMEOUT = 30.0
    MAX_TITLES_PER_QUERY = 50
    MAX_WORKERS = 5

    def __init__(
        self,
        access_token: str | None = None,
        app_name: str | None = None,
        contact: str | None = None,
        env_path: str | None = None,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialise the fetcher.

        Credentials are read from explicit arguments first, then environment
        variables. At minimum *app_name* and *contact* are required so that
        Wikipedia receives a descriptive ``User-Agent``.
        """
        if env_path:
            load_dotenv(env_path)
        elif os.path.exists(".env"):
            load_dotenv()

        self.access_token = access_token or os.getenv("WM_ACCESS_TOKEN")
        self.app_name = app_name or os.getenv("WM_APP_NAME")
        self.contact = contact or os.getenv("WM_CONTACT")

        missing: list[str] = []
        if not self.app_name:
            missing.append("WM_APP_NAME")
        if not self.contact:
            missing.append("WM_CONTACT")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        self._retries = retries
        self._backoff = backoff
        self._timeout = timeout

        self._session = requests.Session()
        self._session.headers.update(self._build_headers())
        logger.debug("WikiFetcher initialised")

    # --------------------------------------------------------------------- #
    # Internal plumbing
    # --------------------------------------------------------------------- #

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": f"{self.app_name} ({self.contact})",
            "Accept": "application/json",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Execute an HTTP request with retry, backoff, and rate-limit handling."""
        attempt = 0
        max_attempts = self._retries

        while attempt < max_attempts:
            try:
                resp = self._session.request(
                    method, url, timeout=self._timeout, **kwargs
                )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = (
                        int(retry_after)
                        if retry_after
                        else int(self._backoff * (2**attempt))
                    )
                    logger.warning(
                        f"Rate limited on {url}, waiting {wait}s "
                        f"(retry {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(wait)
                    attempt += 1
                    if attempt >= max_attempts:
                        raise RateLimitError(
                            f"Rate limited after {max_attempts} attempts: {url}"
                        )
                    continue

                resp.raise_for_status()
                return resp

            except requests.Timeout as exc:
                logger.warning(
                    f"Timeout on {url} (attempt {attempt + 1}/{max_attempts})"
                )
                if attempt >= max_attempts - 1:
                    raise NetworkError(
                        f"Network timeout after {max_attempts} attempts: {exc}"
                    ) from exc
                wait = self._backoff * (2**attempt)
                time.sleep(wait)
                attempt += 1

            except requests.HTTPError as exc:
                status = exc.response.status_code
                if status in (500, 502, 503, 504):
                    logger.warning(
                        f"Server error {status} on {url} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    if attempt >= max_attempts - 1:
                        raise NetworkError(
                            f"Server error {status} after {max_attempts} attempts: {exc}"
                        ) from exc
                    wait = self._backoff * (2**attempt)
                    time.sleep(wait)
                    attempt += 1
                elif status == 404:
                    raise PageNotFoundError(f"Resource not found: {url}")
                else:
                    raise FetcherError(
                        f"HTTP {status}: {exc.response.text[:500]}"
                    ) from exc

        raise NetworkError(f"Failed after {max_attempts} attempts: {url}")

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        return self._request("GET", url, **kwargs)

    def _get_json(self, url: str, **kwargs: Any) -> Any:
        resp = self._get(url, **kwargs)
        try:
            return resp.json()
        except Exception as exc:
            snippet = resp.text[:500]
            logger.error(f"Malformed JSON from {url}: {snippet}")
            raise MalformedResponseError(
                f"Could not parse JSON response from {url}: {exc}"
            ) from exc

    @staticmethod
    def _date_to_mediawiki(dt: datetime.date) -> str:
        """ISO 8601 UTC string suitable for MediaWiki API parameters."""
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def fetch_on_this_day(
        self, date: datetime.date | None = None
    ) -> list[OnThisDayEvent]:
        """Fetch the "On This Day" feed for *date* (defaults to today)."""
        date = date or datetime.date.today()
        url = (
            f"{self.REST_BASE}/featured/"
            f"{date.year}/{date.month:02d}/{date.day:02d}"
        )

        logger.info(f"Fetching On This Day for {date.isoformat()}")
        data = self._get_json(url)

        events: list[OnThisDayEvent] = []
        onthisday = data.get("onthisday", {})

        # The API may return onthisday as a dict (featured endpoint) or a flat list.
        if isinstance(onthisday, list):
            raw_items = [("selected", item) for item in onthisday]
        elif isinstance(onthisday, dict):
            raw_items = [
                (event_type, item)
                for event_type in ("selected", "births", "deaths", "events", "holidays")
                for item in onthisday.get(event_type, [])
            ]
        else:
            raw_items = []

        for event_type, item in raw_items:
            try:
                year = item.get("year")
                text = item.get("text", "")
                pages = item.get("pages", [])
                related = [
                    p.get("titles", {}).get("canonical", p.get("title", ""))
                    for p in pages
                ]
                related = [r for r in related if r]

                if not text and pages:
                    text = pages[0].get("extract", "") or pages[0].get(
                        "description", ""
                    )

                event_id = hashlib.sha256(
                    f"{event_type}:{year}:{text}".encode("utf-8")
                ).hexdigest()[:12]
                events.append(
                    OnThisDayEvent(
                        event_id=event_id,
                        year=int(year) if year is not None else None,
                        description=text,
                        related_titles=related,
                        event_type=event_type,  # type: ignore[arg-type]
                    )
                )
            except Exception:
                logger.warning(f"Skipping malformed OnThisDay item: {item}")
                continue

        logger.info(f"Fetched {len(events)} On This Day events")
        return events

    def fetch_recent_changes(
        self, hours: int = 24, limit: int = 500
    ) -> list[TrendingEdit]:
        """Fetch recent main-space edits from the last *hours* (max 500)."""
        end = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)

        params: dict[str, str | int] = {
            "action": "query",
            "list": "recentchanges",
            "rcnamespace": 0,
            "rclimit": min(limit, 500),
            "rcprop": "title|user|timestamp|sizes|comment|flags|tags",
            "rctype": "edit|new",
            "rcend": self._date_to_mediawiki(end),
            "format": "json",
        }

        logger.info(f"Fetching recent changes for last {hours}h")
        data = self._get_json(self.API_BASE, params=params)

        edits: list[TrendingEdit] = []
        for item in data.get("query", {}).get("recentchanges", []):
            try:
                flags = item.get("flags", [])
                tags = item.get("tags", [])
                edits.append(
                    TrendingEdit(
                        title=item.get("title", ""),
                        editor=item.get("user", ""),
                        timestamp=datetime.datetime.fromisoformat(
                            item["timestamp"].replace("Z", "+00:00")
                        ),
                        bytes_changed=item.get("newlen", 0) - item.get("oldlen", 0),
                        summary=item.get("comment", ""),
                        is_bot="bot" in flags or "bot" in tags,
                        is_minor="minor" in flags,
                    )
                )
            except Exception:
                logger.warning(f"Skipping malformed recent change: {item}")
                continue

        logger.info(f"Fetched {len(edits)} recent changes")
        return edits

    def fetch_page(self, title: str) -> WikiPage | None:
        """Fetch a single Wikipedia page with full metadata and HTML."""
        pages = self._fetch_pages_batch([title])
        page = pages.get(title)
        if page is None:
            return None

        # Augment with HTML via action=parse (single-page only)
        try:
            html = self._fetch_page_html(title)
            return WikiPage(
                title=page.title,
                html=html,
                plain_text=page.plain_text,
                internal_links=page.internal_links,
                external_links=page.external_links,
                source_url=page.source_url,
                word_count=page.word_count,
                is_featured=page.is_featured,
                is_good=page.is_good,
                fetch_timestamp=page.fetch_timestamp,
            )
        except Exception as exc:
            logger.warning(f"Could not fetch HTML for '{title}': {exc}")
            return page

    def fetch_pages(self, titles: list[str]) -> dict[str, WikiPage | None]:
        """Fetch multiple pages in batches of 50.  No HTML is populated."""
        if not titles:
            return {}

        seen: set[str] = set()
        unique: list[str] = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        results: dict[str, WikiPage | None] = {}
        for i in range(0, len(unique), self.MAX_TITLES_PER_QUERY):
            batch = unique[i : i + self.MAX_TITLES_PER_QUERY]
            results.update(self._fetch_pages_batch(batch))
        return results

    def _fetch_pages_batch(self, titles: list[str]) -> dict[str, WikiPage | None]:
        """Query up to :attr:`MAX_TITLES_PER_QUERY` pages via ``action=query``."""
        params: dict[str, str | int] = {
            "action": "query",
            "prop": "extracts|links|extlinks|info|categories",
            "titles": "|".join(titles),
            "explaintext": 1,
            "exintro": 1,
            "exlimit": "max",
            "pllimit": "max",
            "ellimit": "max",
            "inprop": "url|displaytitle",
            "clcategories": (
                "Category:Featured articles|Category:Good articles"
            ),
            "cllimit": "max",
            "redirects": 1,
            "format": "json",
        }

        logger.debug(f"Fetching batch of {len(titles)} pages")
        data = self._get_json(self.API_BASE, params=params)

        query = data.get("query", {})
        normalizations = {
            n["from"]: n["to"] for n in query.get("normalized", [])
        }
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        pages_by_title: dict[str, Any] = {}
        for page in query.get("pages", {}).values():
            pages_by_title[page.get("title", "")] = page

        results: dict[str, WikiPage | None] = {}
        for title in titles:
            norm_title = normalizations.get(title, title)
            final_title = redirects.get(norm_title, norm_title)
            page = pages_by_title.get(final_title)

            if page is None or "missing" in page:
                results[title] = None
                continue

            try:
                internal = [
                    link["title"]
                    for link in page.get("links", [])
                    if link.get("ns") == 0
                ]
                categories = [
                    cat.get("title", "") for cat in page.get("categories", [])
                ]
                text = page.get("extract", "")

                results[title] = WikiPage(
                    title=page.get("title", title),
                    plain_text=text,
                    internal_links=internal,
                    external_links=page.get("extlinks", []),
                    source_url=page.get("fullurl", ""),
                    word_count=len(text.split()) if text else 0,
                    is_featured="Category:Featured articles" in categories,
                    is_good="Category:Good articles" in categories,
                )
            except Exception as exc:
                logger.error(f"Error parsing page '{title}': {exc}")
                results[title] = None

        return results

    def _fetch_page_html(self, title: str) -> str:
        """Fetch raw HTML for a single page via ``action=parse``."""
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
        }
        data = self._get_json(self.API_BASE, params=params)
        return data.get("parse", {}).get("text", {}).get("*", "")

    def fetch_page_views(self, title: str, days: int = 30) -> int | None:
        """Return the *average* daily page views over the last *days*.

        Returns ``None`` if data is unavailable or the request fails.
        """
        end = datetime.date.today() - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=days - 1)

        encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
        url = (
            f"{self.PAGEVIEWS_BASE}/{encoded}/"
            f"daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
        )

        logger.debug(f"Fetching page views for '{title}' ({days} days)")
        try:
            data = self._get_json(url)
            items = data.get("items", [])
            if not items:
                return None
            total = sum(item.get("views", 0) for item in items)
            return total // len(items)
        except (PageNotFoundError, FetcherError) as exc:
            logger.warning(f"Failed to fetch page views for '{title}': {exc}")
            return None

    def search(self, query: str) -> str | None:
        """Search Wikipedia and return the canonical title of the top result."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        }
        logger.debug(f"Searching for '{query}'")
        data = self._get_json(self.API_BASE, params=params)
        results = data.get("query", {}).get("search", [])
        return results[0].get("title") if results else None

    def check_article_quality(self, title: str) -> tuple[bool, bool]:
        """Return ``(is_featured, is_good)`` for *title*."""
        pages = self.fetch_pages([title])
        page = pages.get(title)
        if page is None:
            return (False, False)
        return (page.is_featured, page.is_good)

    def fetch_page_revisions(
        self, title: str, days: int = 30
    ) -> int | None:
        """Count revisions for *title* in the last *days*.

        .. note:: The count is capped at 500 because that is the API limit
           for unprivileged requests.
        """
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=days
        )

        params: dict[str, str | int] = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvlimit": "max",
            "rvprop": "timestamp",
            "rvdir": "older",
            "redirects": 1,
            "format": "json",
        }

        logger.debug(f"Fetching revisions for '{title}' (last {days} days)")
        data = self._get_json(self.API_BASE, params=params)

        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                return None

            revisions = page.get("revisions", [])
            count = 0
            for rev in revisions:
                ts_str = rev.get("timestamp", "")
                try:
                    ts = datetime.datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    )
                    if ts >= cutoff:
                        count += 1
                except Exception:
                    continue
            return count

        return None

    def enrich_pages(
        self, titles: list[str], days: int = 30
    ) -> dict[str, PageMetrics | None]:
        """Fetch content, page views, and edit counts for *titles*.

        Returns a mapping from the original title to a :class:`PageMetrics`
        object, or ``None`` if the page does not exist.
        """
        if not titles:
            return {}

        logger.info(f"Enriching {len(titles)} page(s) (metrics window: {days} days)")

        # 1. Batch-fetch page content
        pages = self.fetch_pages(titles)

        # 2. Parallel-fetch views & revisions
        results: dict[str, PageMetrics | None] = {}
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            view_futures = {
                t: executor.submit(self.fetch_page_views, t, days)
                for t in titles
            }
            rev_futures = {
                t: executor.submit(self.fetch_page_revisions, t, days)
                for t in titles
            }

            for title in titles:
                page = pages.get(title)
                if page is None:
                    results[title] = None
                    continue

                views = view_futures[title].result()
                revs = rev_futures[title].result()

                results[title] = PageMetrics(
                    title=page.title,
                    word_count=page.word_count,
                    internal_links=len(page.internal_links),
                    external_links=len(page.external_links),
                    page_views_30d=views,
                    edit_count_30d=revs,
                    is_featured=page.is_featured,
                    is_good=page.is_good,
                    source_url=page.source_url,
                )

        logger.info(f"Enrichment complete for {len(titles)} page(s)")
        return results
