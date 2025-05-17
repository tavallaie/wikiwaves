import os
import sys
import datetime
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class WikimediaFeaturedFetcher:
    """
    Fetches the Featured Content feed for English Wikipedia
    and provides each section separately.
    """

    API_BASE = "https://api.wikimedia.org/feed/v1/wikipedia/en/featured"

    def __init__(self):
        self.token = os.getenv("WM_ACCESS_TOKEN")
        self.app = os.getenv("WM_APP_NAME")
        self.contact = os.getenv("WM_CONTACT")
        missing = [
            v
            for v in ("WM_ACCESS_TOKEN", "WM_APP_NAME", "WM_CONTACT")
            if not os.getenv(v)
        ]
        if missing:
            raise RuntimeError(f"Missing environment vars: {', '.join(missing)}")

        # Configure Loguru
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
        logger.info("Initialized WikimediaFeaturedFetcher")

        self._data = None

    def _build_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": f"{self.app} ({self.contact})",
        }

    def _today_path(self):
        return datetime.datetime.now().strftime("%Y/%m/%d")

    def _fetch_all(self):
        """Internal: fetch and cache the full JSON feed."""
        if self._data is None:
            url = f"{self.API_BASE}/{self._today_path()}"
            logger.debug(f"Requesting URL: {url}")
            resp = requests.get(url, headers=self._build_headers())
            try:
                resp.raise_for_status()
            except requests.HTTPError:
                logger.error(f"HTTP {resp.status_code}: {resp.text}")
                raise
            self._data = resp.json()
            logger.info("Fetched and cached full feed")
        return self._data

    def get_tfa(self):
        """Today's 'Today’s Featured Article' section."""
        return self._fetch_all().get("tfa")

    def get_mostread(self):
        """Today's 'Most Read' section."""
        return self._fetch_all().get("mostread")

    def get_image(self):
        """Today's featured image info."""
        return self._fetch_all().get("image")

    def get_news(self):
        """Today's news spotlight."""
        return self._fetch_all().get("news")

    def get_onthisday(self):
        """Events for 'On this day...'."""
        return self._fetch_all().get("onthisday")


if __name__ == "__main__":
    try:
        fetcher = WikimediaFeaturedFetcher()
        print("=== Today’s Featured Article ===")
        print(fetcher.get_tfa(), "\n")

        print("=== Most Read ===")
        print(fetcher.get_mostread(), "\n")

        print("=== Featured Image ===")
        print(fetcher.get_image(), "\n")

        print("=== News ===")
        print(fetcher.get_news(), "\n")

        print("=== On This Day ===")
        print(fetcher.get_onthisday())
    except Exception:
        logger.exception("Failed to retrieve one or more sections")
        sys.exit(1)
