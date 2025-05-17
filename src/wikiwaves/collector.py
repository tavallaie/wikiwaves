import os
import sys
import datetime
import json
import requests
from dotenv import load_dotenv
from loguru import logger


class WikimediaFeaturedFetcher:
    """
    Fetches the Featured Content feed for English Wikipedia
    and provides each section separately, with methods to save each
    section as its own JSON file.
    """

    API_BASE = "https://api.wikimedia.org/feed/v1/wikipedia/en/featured"

    def __init__(self, env_path: str = None):
        # Load environment variables from .env (if present) *every* time
        load_dotenv(env_path)

        # Read and validate required env vars
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
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            ),
            level="DEBUG",
        )
        logger.info("Initialized WikimediaFeaturedFetcher")

        # Cache for fetched JSON
        self._data = None

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": f"{self.app} ({self.contact})",
        }

    def _today_path(self) -> str:
        return datetime.datetime.now().strftime("%Y/%m/%d")

    def _fetch_all(self) -> dict:
        """Internal: fetch and cache the full JSON feed."""
        if self._data is None:
            url = f"{self.API_BASE}/{self._today_path()}"
            logger.debug(f"Requesting URL: {url}")
            resp = requests.get(url, headers=self._build_headers())
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                logger.error(f"HTTP {resp.status_code} error: {resp.text}")
                raise
            self._data = resp.json()
            logger.info("Fetched and cached full feed")
        return self._data

    def get_tfa(self) -> dict:
        """Return Today's Featured Article."""
        return self._fetch_all().get("tfa", {})

    def save_tfa(self, path: str = "tfa.json") -> None:
        """Save Today's Featured Article to a JSON file."""
        data = self.get_tfa()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved TFA to {path}")

    def get_mostread(self) -> dict:
        """Return Today's Most Read section."""
        return self._fetch_all().get("mostread", {})

    def save_mostread(self, path: str = "mostread.json") -> None:
        """Save Today's Most Read to a JSON file."""
        data = self.get_mostread()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Most Read to {path}")

    def get_image(self) -> dict:
        """Return Today's Featured Image section."""
        return self._fetch_all().get("image", {})

    def save_image(self, path: str = "image.json") -> None:
        """Save Today's Featured Image to a JSON file."""
        data = self.get_image()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Image to {path}")

    def get_news(self) -> dict:
        """Return Today's News spotlight."""
        return self._fetch_all().get("news", {})

    def save_news(self, path: str = "news.json") -> None:
        """Save Today's News to a JSON file."""
        data = self.get_news()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved News to {path}")

    def get_onthisday(self) -> dict:
        """Return Today's On This Day events."""
        return self._fetch_all().get("onthisday", {})

    def save_onthisday(self, path: str = "onthisday.json") -> None:
        """Save On This Day events to a JSON file."""
        data = self.get_onthisday()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved On This Day to {path}")


if __name__ == "__main__":
    try:
        fetcher = WikimediaFeaturedFetcher()

        # Example usage:
        tfa = fetcher.get_tfa()
        print("=== Today’s Featured Article ===")
        print(tfa, "\n")
        fetcher.save_tfa()

        mostread = fetcher.get_mostread()
        print("=== Most Read ===")
        print(mostread, "\n")
        fetcher.save_mostread()

        image = fetcher.get_image()
        print("=== Featured Image ===")
        print(image, "\n")
        fetcher.save_image()

        news = fetcher.get_news()
        print("=== News ===")
        print(news, "\n")
        fetcher.save_news()

        onthisday = fetcher.get_onthisday()
        print("=== On This Day ===")
        print(onthisday)
        fetcher.save_onthisday()

    except Exception:
        logger.exception("Failed to retrieve or save one or more sections")
        sys.exit(1)
