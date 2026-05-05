"""Tests for the WikiWaves Pipeline Orchestrator."""

from __future__ import annotations

import datetime
import json
import os

import pytest

from wikiwaves.enricher.models import EnrichedTopic
from wikiwaves.fetcher.models import OnThisDayEvent, PageMetrics, TrendingEdit, WikiPage
from wikiwaves.orchestrator import PipelineOrchestrator, PipelineResult, StageReport


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Stub LLM client."""

    def __init__(self, response: dict | None = None, raise_error: Exception | None = None) -> None:
        self.response = response or {"suggestions": [], "explanation": "test"}
        self.raise_error = raise_error

    def chat(self, messages, temperature=0.3):
        if self.raise_error:
            raise self.raise_error
        import json as _json

        return _json.dumps(self.response)


class FakeFetcher:
    """Stub fetcher that returns deterministic data."""

    def __init__(
        self,
        events: list[OnThisDayEvent] | None = None,
        edits: list[TrendingEdit] | None = None,
        pages: dict[str, WikiPage | None] | None = None,
        metrics: dict[str, PageMetrics | None] | None = None,
    ) -> None:
        self._events = events or []
        self._edits = edits or []
        self._pages = pages or {}
        self._metrics = metrics or {}

    def fetch_on_this_day(self, date):
        return self._events

    def fetch_recent_changes(self, hours=24, limit=500):
        return self._edits

    def enrich_pages(self, titles):
        return {t: self._metrics.get(t) for t in titles}

    def fetch_pages(self, titles):
        return {t: self._pages.get(t) for t in titles}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(title: str, year: int = 2000) -> OnThisDayEvent:
    return OnThisDayEvent(
        event_id=f"evt-{title}",
        year=year,
        description=f"Something about {title}",
        related_titles=[title],
        event_type="selected",
    )


def _edit(title: str, bytes_changed: int = 1000) -> TrendingEdit:
    return TrendingEdit(
        title=title,
        editor="Editor1",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        bytes_changed=bytes_changed,
        summary="Edit summary",
        is_bot=False,
        is_minor=False,
    )


def _page(title: str, word_count: int = 1000, links: list[str] | None = None) -> WikiPage:
    return WikiPage(
        title=title,
        plain_text=f"Extract for {title}",
        internal_links=links or [],
        word_count=word_count,
    )


def _metrics(title: str, word_count: int = 1000) -> PageMetrics:
    return PageMetrics(
        title=title,
        word_count=word_count,
        internal_links=10,
        page_views_30d=5000,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_orchestrator_creates_dated_dirs(tmp_path):
    orch = PipelineOrchestrator(
        date=datetime.date(2024, 1, 15),
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
        fetcher=FakeFetcher(),
    )
    assert os.path.isdir(tmp_path / "out" / "2024-01-15")
    assert os.path.isdir(tmp_path / "tmp" / "2024-01-15")


# ---------------------------------------------------------------------------
# Full run — happy path
# ---------------------------------------------------------------------------


def test_run_success(tmp_path):
    events = [_event("Napoleon", year=1821)]
    edits = [_edit("Napoleon", bytes_changed=2400)]
    metrics = {"Napoleon": _metrics("Napoleon", word_count=2000)}
    pages = {"Napoleon": _page("Napoleon", word_count=2000, links=["France", "Waterloo"])}
    llm_response = {
        "suggestions": [{"title": "France", "reason": "Country"}],
        "explanation": "Adds context.",
    }
    fake_pages = {"France": _page("France", word_count=800)}

    fetcher = FakeFetcher(
        events=events,
        edits=edits,
        metrics=metrics,
        pages={**pages, **fake_pages},
    )
    llm = FakeLLMClient(response=llm_response)

    orch = PipelineOrchestrator(
        date=datetime.date(2024, 1, 15),
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
        fetcher=fetcher,
        llm_client=llm,
        topic_count=1,
    )
    result = orch.run()

    assert isinstance(result, PipelineResult)
    assert result.status == "success"
    assert result.date == "2024-01-15"
    assert len(result.topics) == 1
    assert result.topics[0].base_page.title == "Napoleon"
    assert len(result.topics[0].related_pages) == 1
    assert result.topics[0].related_pages[0].title == "France"
    assert len(result.fun_facts) == 1

    # Artifacts exist
    assert os.path.exists(tmp_path / "out" / "2024-01-15" / "report.txt")
    assert os.path.exists(tmp_path / "out" / "2024-01-15" / "enriched_topics.json")
    assert os.path.exists(tmp_path / "tmp" / "2024-01-15" / "raw_events.json")
    assert os.path.exists(tmp_path / "tmp" / "2024-01-15" / "selection.json")


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_run_fetch_fails(tmp_path):
    class BadFetcher:
        def fetch_on_this_day(self, date):
            raise RuntimeError("API down")

        def fetch_recent_changes(self, hours=24, limit=500):
            return []

    orch = PipelineOrchestrator(
        date=datetime.date(2024, 1, 15),
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
        fetcher=BadFetcher(),
    )
    result = orch.run()

    assert result.status == "failed"
    assert result.topics == []
    assert any("API down" in e for e in result.errors)


def test_run_curate_fails(tmp_path):
    class BadFetcher:
        def fetch_on_this_day(self, date):
            return [_event("X")]

        def fetch_recent_changes(self, hours=24, limit=500):
            return []

        def enrich_pages(self, titles):
            raise RuntimeError("Metrics failed")

    orch = PipelineOrchestrator(
        date=datetime.date(2024, 1, 15),
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
        fetcher=BadFetcher(),
    )
    result = orch.run()

    assert result.status == "failed"
    assert result.topics == []


def test_run_enrich_fails_gracefully(tmp_path):
    events = [_event("Napoleon", year=1821)]
    edits = []
    metrics = {"Napoleon": _metrics("Napoleon", word_count=2000)}
    pages = {"Napoleon": _page("Napoleon", word_count=2000, links=["France"])}

    fetcher = FakeFetcher(
        events=events,
        edits=edits,
        metrics=metrics,
        pages=pages,
    )
    llm = FakeLLMClient(raise_error=RuntimeError("LLM down"))

    orch = PipelineOrchestrator(
        date=datetime.date(2024, 1, 15),
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
        fetcher=fetcher,
        llm_client=llm,
        topic_count=1,
    )
    result = orch.run()

    # Fetch and curate succeed; enrichment fails internally (fail-soft) → success
    assert result.status == "success"
    assert len(result.topics) == 1
    assert result.topics[0].base_page.title == "Napoleon"
    assert result.topics[0].related_pages == []


# ---------------------------------------------------------------------------
# Fun facts
# ---------------------------------------------------------------------------


def test_extract_fun_facts_skips_bots():
    edits = [
        TrendingEdit(title="A", editor="bot", timestamp=datetime.datetime.now(datetime.timezone.utc), bytes_changed=1000, summary="", is_bot=True),
        TrendingEdit(title="B", editor="human", timestamp=datetime.datetime.now(datetime.timezone.utc), bytes_changed=500, summary="", is_bot=False),
    ]
    facts = PipelineOrchestrator._extract_fun_facts(edits)
    assert len(facts) == 1
    assert "human" in facts[0]


def test_extract_fun_facts_sorted_by_bytes_changed():
    edits = [
        TrendingEdit(title="Small", editor="e", timestamp=datetime.datetime.now(datetime.timezone.utc), bytes_changed=100, summary="", is_bot=False),
        TrendingEdit(title="Large", editor="e", timestamp=datetime.datetime.now(datetime.timezone.utc), bytes_changed=900, summary="", is_bot=False),
    ]
    facts = PipelineOrchestrator._extract_fun_facts(edits)
    assert "900" in facts[0]
    assert "100" in facts[1]


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def test_write_report_contains_all_sections(tmp_path):
    orch = PipelineOrchestrator(
        date=datetime.date(2024, 1, 15),
        output_dir=str(tmp_path / "out"),
        tmp_dir=str(tmp_path / "tmp"),
        fetcher=FakeFetcher(),
    )
    result = PipelineResult(
        date="2024-01-15",
        status="partial",
        topics=[],
        fun_facts=["Fact 1"],
        stage_reports=[
            StageReport(name="fetch", status="success", runtime_seconds=1.0),
            StageReport(name="enrich", status="failed", runtime_seconds=0.5, errors=["boom"]),
        ],
        total_runtime_seconds=2.0,
        output_dir=str(tmp_path / "out" / "2024-01-15"),
        errors=["something went wrong"],
    )
    orch._write_report(result)

    report_path = tmp_path / "out" / "2024-01-15" / "report.txt"
    assert report_path.exists()
    text = report_path.read_text()
    assert "partial" in text
    assert "Fact 1" in text
    assert "boom" in text
    assert "something went wrong" in text
