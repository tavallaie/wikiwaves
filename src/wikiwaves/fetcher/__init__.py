"""WikiWaves Fetcher — Wikipedia data acquisition module."""

from .client import WikiFetcher
from .exceptions import (
    FetcherError,
    MalformedResponseError,
    NetworkError,
    PageNotFoundError,
    RateLimitError,
)
from .models import OnThisDayEvent, PageMetrics, TrendingEdit, WikiPage

__all__ = [
    "WikiFetcher",
    "OnThisDayEvent",
    "PageMetrics",
    "TrendingEdit",
    "WikiPage",
    "FetcherError",
    "NetworkError",
    "RateLimitError",
    "PageNotFoundError",
    "MalformedResponseError",
]
