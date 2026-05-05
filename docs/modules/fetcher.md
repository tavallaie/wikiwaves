# Fetcher — Data Acquisition

## Purpose
Handle all external communication with Wikipedia and Wikimedia services. Zero business logic; only clean data retrieval and normalization.

## Input
- A date (for "On This Day" feed)
- A list of article titles (for full page content)
- Query parameters for search, page views, or recent changes

## Output
- Structured event feeds (events, births, deaths, holidays)
- Structured article objects (text, links, metadata)
- Structured recent-edits streams
- Page-view statistics and article-quality flags

## Boundaries
- Decides nothing about what is "interesting."
- Retries transient failures automatically.
- Returns empty results for missing pages instead of crashing.
- Never filters or ranks; passes raw data onward.

## Failure Mode
Log the failure, return the emptiest valid object possible, and let downstream modules decide whether to proceed or abort.
