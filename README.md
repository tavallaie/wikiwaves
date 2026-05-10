# WikiWaves

Automated, daily Wikipedia-to-podcast pipeline. Fetches historical events from Wikipedia's "On This Day" feed, curates the top stories, enriches them with related articles via LLM, and generates a structured podcast script — with TTS and assembly coming next.

---

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `WM_APP_NAME` | ✅ | App name for Wikipedia User-Agent header |
| `WM_CONTACT` | ✅ | Contact email for Wikipedia User-Agent header |
| `WM_ACCESS_TOKEN` | ✅ | Wikimedia API access token |
| `LLM_API_KEY` | ❌ | API key for LLM provider (OpenAI-compatible) |
| `LLM_BASE_URL` | ❌ | Base URL for LLM API (default: `https://api.openai.com/v1`) |
| `LLM_MODEL` | ❌ | Model name (default: `gpt-4o-mini`) |

> **Note:** The pipeline runs fine without an LLM key — enrichment will simply fall back to the base article alone for each topic.

### 3. Run the full pipeline

```bash
uv run python -m wikiwaves.orchestrator
```

Or for a specific date:

```bash
uv run python -m wikiwaves.orchestrator --date 2026-05-05 --topics 5
```

Artifacts are saved to:
- `output/YYYY-MM-DD/` — final outputs (enriched topics, report)
- `tmp/YYYY-MM-DD/` — intermediate artifacts (raw events, selection, base pages)

---

## Running Individual Modules

Each stage can be run standalone via the examples in `example/`:

```bash
# Fetch live Wikipedia data
uv run python example/fetcher.py

# Curate topics from fetched data
uv run python example/curator.py

# Enrich curated topics with related articles
uv run python example/enricher.py
```

---

## Project Structure

```
src/wikiwaves/
  __init__.py         # CLI entry point (placeholder)
  orchestrator.py     # Pipeline runner
  fetcher/            # Wikipedia API client
    client.py
    models.py
    exceptions.py
  curator/            # Topic scoring & selection
    aggregator.py
    selector.py
    models.py
  enricher/           # LLM-powered content expansion
    expander.py
    llm_client.py
    validators.py
    prompts.py
    models.py
  
example/
  fetcher.py          # Demo: fetch + enrich + curate
  curator.py          # Demo: curate from saved fetcher output
  enricher.py         # Demo: enrich from saved curator output

tests/
  test_fetcher.py
  test_curator.py
  test_enricher.py
  test_orchestrator.py
```

---

## Pipeline Stages

```
Fetcher  →  Curator  →  Enricher  →  [Scripter  →  TTS  →  Assembler]
```

| Stage | What it does |
|-------|--------------|
| **Fetcher** | Pulls "On This Day" events, recent changes, page content, views, and quality metrics from Wikipedia |
| **Curator** | Scores events by richness, recency, popularity, quality, and diversity; selects top 5 topics |
| **Enricher** | Uses an LLM to pick 3-5 related articles from each base page's internal links, then fetches them |
| **Scripter** | *(coming soon)* Generates conversational dialogue from enriched content |
| **TTS** | *(coming soon)* Converts script chunks to audio |
| **Assembler** | *(coming soon)* Stitches audio into a final podcast MP3 |

---

## Testing

```bash
uv run python -m unittest discover -s tests -v
```

All 59 tests should pass.

---

## Tech Stack

- **Python** ≥3.12
- **Build**: Hatchling (`pyproject.toml`)
- **Package manager**: `uv`
- **HTTP**: `requests`
- **Logging**: `loguru`
- **Env files**: `python-dotenv`
- **Testing**: `unittest`, `responses`

---

## License

MIT
