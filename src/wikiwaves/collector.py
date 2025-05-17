import os
import datetime
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class WikimediaFeaturedFetcher:
    """
    Fetches the Featured Content feed for English Wikipedia
    using Wikimedia’s REST API.
    """

    API_BASE = "https://api.wikimedia.org/feed/v1/wikipedia/en/featured"

    def __init__(self):
        # Read and validate all required env vars
        self.token = os.getenv("WM_ACCESS_TOKEN")
        self.app = os.getenv("WM_APP_NAME")
        self.contact = os.getenv("WM_CONTACT")

        missing = [
            var
            for var in ("WM_ACCESS_TOKEN", "WM_APP_NAME", "WM_CONTACT")
            if not os.getenv(var)
        ]
        if missing:
            raise RuntimeError(f"Missing environment vars: {', '.join(missing)}")

        # Configure Loguru
        logger.remove()  # Remove default stderr handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
        )
        logger.info("Initialized WikimediaFeaturedFetcher")

    def _build_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": f"{self.app} ({self.contact})",
        }

    def _today_path(self):
        today = datetime.datetime.now()
        return today.strftime("%Y/%m/%d")

    def fetch(self):
        """
        Fetch the JSON feed for today’s featured content.
        Returns the parsed JSON on success, or raises on error.
        """
        date_path = self._today_path()
        url = f"{self.API_BASE}/{date_path}"
        headers = self._build_headers()

        logger.debug(f"Requesting URL: {url}")
        resp = requests.get(url, headers=headers)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            logger.error(f"HTTP error {resp.status_code}: {resp.text}")
            raise
        logger.info("Fetch successful")
        return resp.json()


if __name__ == "__main__":
    import sys

    try:
        fetcher = WikimediaFeaturedFetcher()
        data = fetcher.fetch()
        # Pretty-print the keys of the response as a quick sanity check
        logger.info(f"Available sections: {list(data.keys())}")
    except Exception as e:
        logger.exception("Failed to fetch featured feed due to: \n" + str(e))
