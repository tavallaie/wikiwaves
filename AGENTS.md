# AGENTS.md — wikiwaves

## Project Overview

`wikiwaves` is a Python CLI tool that fetches the **Wikimedia Featured Content feed** for English Wikipedia and saves each section (Featured Article, Most Read, Image, News, On This Day) as separate JSON files.

## Tech Stack

- **Language**: Python >=3.12
- **Build system**: Hatchling (`pyproject.toml`)
- **Package manager**: `uv` (lockfile: `uv.lock`)
- **Layout**: `src/` layout (`src/wikiwaves/`)

## Dependencies

- `requests` — HTTP requests to Wikimedia API
- `loguru` — structured logging
- `python-dotenv` — `.env` file support

## Project Structure

```
src/wikiwaves/
  __init__.py         # CLI entry point (`wikiwaves:main`)
  collector.py        # WikimediaFeaturedFetcher class — core logic
  fetcher/            # Wikipedia data acquisition
  curator/            # Topic selection and scoring
  enricher/           # LLM-powered content expansion
```

## Environment Variables

Required (see `.env.example`):

| Variable | Purpose |
|----------|---------|
| `WM_ACCESS_TOKEN` | Wikimedia API access token |
| `WM_APP_NAME` | App name for User-Agent header |
| `WM_CONTACT` | Contact email for User-Agent header |

Optional:

| Variable | Purpose |
|----------|---------|
| `WM_CLIENT_ID` | OAuth client ID |
| `WM_CLIENT_SECRET` | OAuth client secret |
| `LLM_API_KEY` | API key for the LLM provider (OpenAI-compatible) |
| `LLM_BASE_URL` | Base URL for the LLM API (default: `https://api.openai.com/v1`) |
| `LLM_MODEL` | Model name for link suggestion / dialogue generation (default: `gpt-4o-mini`) |

Load via `python-dotenv` (automatically in `WikimediaFeaturedFetcher`).

## Running

```bash
# Install dependencies
uv sync

# Run the module
uv run python -m wikiwaves.collector

# Or via CLI entry point (after install)
wikiwaves
```

## Core Class: `WikimediaFeaturedFetcher`

Located in `src/wikiwaves/collector.py`.

- Fetches from `https://api.wikimedia.org/feed/v1/wikipedia/en/featured/{YYYY}/{MM}/{DD}`
- Authenticates via Bearer token + custom User-Agent
- Sections: `tfa`, `mostread`, `image`, `news`, `onthisday`
- Each section has `get_*()` and `save_*()` methods
- Default save paths: `tfa.json`, `mostread.json`, `image.json`, `news.json`, `onthisday.json`

## Conventions

- Use `loguru` for all logging (not `print` in library code)
- Raise `RuntimeError` on missing env vars
- JSON output: `ensure_ascii=False`, indent=2
- Use type hints where practical
- Keep `__init__.py` minimal; core logic lives in `collector.py`
