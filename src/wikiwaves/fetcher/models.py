"""Data models for Wikipedia content returned by the Fetcher."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class OnThisDayEvent:
    """A single event from Wikipedia's "On this day" feed.

    Attributes:
        event_id: Stable unique identifier (hash of year + description).
        year: The year the event occurred. ``None`` for undated items (e.g. holidays).
        description: Human-readable summary of the event.
        related_titles: Canonical titles of related Wikipedia articles.
        event_type: Category of the event in the feed.
    """

    event_id: str
    year: int | None
    description: str
    related_titles: list[str] = field(default_factory=list)
    event_type: Literal["selected", "births", "deaths", "events", "holidays"] = "selected"


@dataclass(frozen=True, slots=True)
class TrendingEdit:
    """A significant edit from Wikipedia's recent-changes stream.

    Attributes:
        title: Article that was edited.
        editor: Username or IP address that made the edit.
        timestamp: UTC timestamp of the edit.
        bytes_changed: Net byte difference (positive = addition, negative = removal).
        summary: Editor-provided edit summary / comment.
        is_bot: Whether the edit was made by a bot account.
        is_minor: Whether the edit was marked as minor.
    """

    title: str
    editor: str
    timestamp: datetime.datetime
    bytes_changed: int
    summary: str
    is_bot: bool = False
    is_minor: bool = False


@dataclass(frozen=True, slots=True)
class WikiPage:
    """A complete Wikipedia article with extracted content and metadata.

    Attributes:
        title: Canonical article title.
        html: Raw HTML of the article (populated on single-page fetches; empty in bulk).
        plain_text: Cleaned plain-text extract.
        internal_links: List of internal article links (namespace 0 only).
        external_links: List of external URLs referenced in the article.
        source_url: Full canonical URL to the article.
        word_count: Approximate word count of the plain-text extract.
        is_featured: ``True`` if the article is a Featured Article.
        is_good: ``True`` if the article is a Good Article.
        fetch_timestamp: UTC time when the page was fetched.
    """

    title: str
    html: str = ""
    plain_text: str = ""
    internal_links: list[str] = field(default_factory=list)
    external_links: list[str] = field(default_factory=list)
    source_url: str = ""
    word_count: int = 0
    is_featured: bool = False
    is_good: bool = False
    fetch_timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


@dataclass(frozen=True, slots=True)
class PageMetrics:
    """Enriched metrics for a Wikipedia page.

    Attributes:
        title: Canonical article title.
        word_count: Approximate word count.
        internal_links: Count of internal article links.
        external_links: Count of external URLs.
        page_views_30d: Average daily page views over the last 30 days.
        edit_count_30d: Number of edits in the last 30 days (capped at 500).
        is_featured: ``True`` if the article is a Featured Article.
        is_good: ``True`` if the article is a Good Article.
        source_url: Full canonical URL to the article.
    """

    title: str
    word_count: int = 0
    internal_links: int = 0
    external_links: int = 0
    page_views_30d: int | None = None
    edit_count_30d: int | None = None
    is_featured: bool = False
    is_good: bool = False
    source_url: str = ""
