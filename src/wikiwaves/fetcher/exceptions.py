"""Custom exceptions for the Fetcher module."""


class FetcherError(Exception):
    """Base exception for all fetcher-related errors."""


class NetworkError(FetcherError):
    """Network failure after all retries have been exhausted."""


class RateLimitError(FetcherError):
    """Rate limit hit and the Retry-After grace period has expired."""


class PageNotFoundError(FetcherError):
    """The requested Wikipedia page or resource does not exist."""


class MalformedResponseError(FetcherError):
    """The API returned data that could not be parsed or validated."""
