"""Pipeline orchestrator — wires fetcher, curator, and enricher into a daily run."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from loguru import logger

from wikiwaves.curator import aggregate_events, curate
from wikiwaves.curator.models import CuratedEpisode
from wikiwaves.enricher import LLMClient, enrich_pages
from wikiwaves.enricher.models import EnrichedTopic
from wikiwaves.fetcher import WikiFetcher
from wikiwaves.fetcher.models import OnThisDayEvent, TrendingEdit


@dataclass(frozen=True, slots=True)
class StageReport:
    """Performance and status report for a single pipeline stage."""

    name: str
    status: Literal["success", "partial", "failed"]
    runtime_seconds: float
    input_count: int = 0
    output_count: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Final result of a pipeline run."""

    date: str
    status: Literal["success", "partial", "failed"]
    topics: list[EnrichedTopic] = field(default_factory=list)
    fun_facts: list[str] = field(default_factory=list)
    stage_reports: list[StageReport] = field(default_factory=list)
    total_runtime_seconds: float = 0.0
    output_dir: str = ""
    errors: list[str] = field(default_factory=list)


class PipelineOrchestrator:
    """Run the WikiWaves pipeline end-to-end for a single calendar day."""

    def __init__(
        self,
        date: datetime.date | None = None,
        output_dir: str = "output",
        tmp_dir: str = "tmp",
        topic_count: int = 5,
        fetcher: WikiFetcher | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.date = date or datetime.date.today()
        self.date_str = self.date.isoformat()
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.topic_count = topic_count

        try:
            self.fetcher = fetcher or WikiFetcher()
        except RuntimeError as exc:
            logger.error(f"Cannot initialise fetcher: {exc}")
            raise

        self._llm_client = llm_client
        self._ensure_dirs()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_dirs(self) -> None:
        os.makedirs(os.path.join(self.output_dir, self.date_str), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, self.date_str), exist_ok=True)

    def _tmp_path(self, name: str) -> str:
        return os.path.join(self.tmp_dir, self.date_str, name)

    def _out_path(self, name: str) -> str:
        return os.path.join(self.output_dir, self.date_str, name)

    @staticmethod
    def _save_json(path: str, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @property
    def llm_client(self) -> LLMClient | None:
        if self._llm_client is None:
            try:
                self._llm_client = LLMClient()
            except Exception as exc:
                logger.warning(f"LLM client unavailable: {exc}")
        return self._llm_client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> PipelineResult:
        """Execute the full pipeline (fetch → curate → enrich)."""
        start = time.perf_counter()
        reports: list[StageReport] = []
        errors: list[str] = []

        # Stage 1: Fetch
        events, edits, report = self._run_fetch()
        reports.append(report)
        if report.status == "failed":
            errors.append("Fetch stage failed")
            errors.extend(report.errors)
            return self._build_result(reports, errors, start)

        # Stage 2: Curate
        episode, report = self._run_curate(events, edits)
        reports.append(report)
        if report.status == "failed":
            errors.append("Curation stage failed")
            errors.extend(report.errors)
            return self._build_result(reports, errors, start)

        # Stage 3: Enrich
        topics, report = self._run_enrich(episode)
        reports.append(report)
        if report.status == "failed":
            errors.append("Enrichment stage failed")
            errors.extend(report.errors)

        # TODO: Stage 4+ — Scripter, TTS, Assembler, Renderer

        fun_facts = self._extract_fun_facts(edits)
        result = self._build_result(
            reports=reports,
            errors=errors,
            start_time=start,
            topics=topics,
            fun_facts=fun_facts,
        )
        self._write_report(result)
        return result

    # ------------------------------------------------------------------ #
    # Stage implementations
    # ------------------------------------------------------------------ #

    def _run_fetch(
        self,
    ) -> tuple[list[OnThisDayEvent], list[TrendingEdit], StageReport]:
        stage_start = time.perf_counter()
        errors: list[str] = []
        events: list[OnThisDayEvent] = []
        edits: list[TrendingEdit] = []

        try:
            events = self.fetcher.fetch_on_this_day(self.date)
            self._save_json(self._tmp_path("raw_events.json"), [asdict(e) for e in events])
        except Exception as exc:
            logger.error(f"Fetch events failed: {exc}")
            errors.append(str(exc))

        try:
            edits = self.fetcher.fetch_recent_changes(hours=24, limit=500)
            self._save_json(self._tmp_path("raw_edits.json"), [asdict(e) for e in edits])
        except Exception as exc:
            logger.error(f"Fetch recent changes failed: {exc}")
            errors.append(str(exc))

        status: Literal["success", "partial", "failed"] = (
            "success"
            if not errors
            else ("failed" if not events else "partial")
        )
        report = StageReport(
            name="fetch",
            status=status,
            runtime_seconds=time.perf_counter() - stage_start,
            input_count=0,
            output_count=len(events) + len(edits),
            errors=errors,
        )
        return events, edits, report

    def _run_curate(
        self,
        events: list[OnThisDayEvent],
        edits: list[TrendingEdit],
    ) -> tuple[CuratedEpisode, StageReport]:
        stage_start = time.perf_counter()
        try:
            # Bulk metrics for all candidate pages
            unique_titles = list({t for e in events for t in e.related_titles})
            enriched = self.fetcher.enrich_pages(unique_titles)
            self._save_json(
                self._tmp_path("enriched_metrics.json"),
                {k: (asdict(v) if v else None) for k, v in enriched.items()},
            )

            # Filter out None values to satisfy type hints
            enriched_clean = {k: v for k, v in enriched.items() if v is not None}
            aggregated = aggregate_events(events, enriched_clean)
            self._save_json(
                self._tmp_path("aggregated_events.json"),
                [asdict(a) for a in aggregated],
            )

            episode = curate(aggregated, date=self.date_str, topic_count=self.topic_count)
            self._save_json(self._tmp_path("selection.json"), asdict(episode))

            report = StageReport(
                name="curate",
                status="success",
                runtime_seconds=time.perf_counter() - stage_start,
                input_count=len(events),
                output_count=len(episode.topics),
            )
            return episode, report
        except Exception as exc:
            logger.error(f"Curation failed: {exc}")
            report = StageReport(
                name="curate",
                status="failed",
                runtime_seconds=time.perf_counter() - stage_start,
                input_count=len(events),
                output_count=0,
                errors=[str(exc)],
            )
            return CuratedEpisode(date=self.date_str), report

    def _run_enrich(
        self,
        episode: CuratedEpisode,
    ) -> tuple[list[EnrichedTopic], StageReport]:
        stage_start = time.perf_counter()
        try:
            # Pick the longest related page as the base article for each topic
            base_titles: list[str] = []
            for topic in episode.topics:
                if topic.related_pages:
                    best = max(topic.related_pages, key=lambda p: p.word_count or 0)
                    base_titles.append(best.title)

            if not base_titles:
                return (
                    [],
                    StageReport(
                        name="enrich",
                        status="success",
                        runtime_seconds=time.perf_counter() - stage_start,
                        output_count=0,
                    ),
                )

            base_pages_map = self.fetcher.fetch_pages(base_titles)
            base_pages = [
                p for p in (base_pages_map.get(t) for t in base_titles) if p is not None
            ]
            self._save_json(
                self._tmp_path("base_pages.json"),
                [asdict(p) for p in base_pages],
            )

            enriched = enrich_pages(base_pages, self.fetcher, self.llm_client)
            self._save_json(
                self._tmp_path("enriched_pages.json"),
                [asdict(t) for t in enriched],
            )
            self._save_json(
                self._out_path("enriched_topics.json"),
                [asdict(t) for t in enriched],
            )

            report = StageReport(
                name="enrich",
                status="success",
                runtime_seconds=time.perf_counter() - stage_start,
                input_count=len(base_pages),
                output_count=sum(1 + len(t.related_pages) for t in enriched),
            )
            return enriched, report
        except Exception as exc:
            logger.error(f"Enrichment failed: {exc}")
            report = StageReport(
                name="enrich",
                status="failed",
                runtime_seconds=time.perf_counter() - stage_start,
                errors=[str(exc)],
            )
            return [], report

    # ------------------------------------------------------------------ #
    # Fun facts (placeholder — will move to dedicated module later)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_fun_facts(edits: list[TrendingEdit]) -> list[str]:
        candidates = [e for e in edits if not e.is_bot]
        candidates.sort(key=lambda e: abs(e.bytes_changed), reverse=True)
        facts: list[str] = []
        for edit in candidates[:3]:
            direction = "added" if edit.bytes_changed > 0 else "removed"
            facts.append(
                f"In the last 24 hours, '{edit.editor}' {direction} "
                f"{abs(edit.bytes_changed)} bytes to '{edit.title}'."
            )
        return facts

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def _build_result(
        self,
        reports: list[StageReport],
        errors: list[str],
        start_time: float,
        topics: list[EnrichedTopic] | None = None,
        fun_facts: list[str] | None = None,
    ) -> PipelineResult:
        total = time.perf_counter() - start_time
        failed_reports = [r for r in reports if r.status == "failed"]
        if not failed_reports:
            status: Literal["success", "partial", "failed"] = "success"
        elif any(r.name in {"fetch", "curate"} for r in failed_reports):
            status = "failed"
        else:
            status = "partial"
        return PipelineResult(
            date=self.date_str,
            status=status,
            topics=topics or [],
            fun_facts=fun_facts or [],
            stage_reports=reports,
            total_runtime_seconds=total,
            output_dir=self._out_path(""),
            errors=errors,
        )

    def _write_report(self, result: PipelineResult) -> None:
        lines = [
            f"WikiWaves Pipeline Report — {result.date}",
            f"Status: {result.status}",
            f"Total runtime: {result.total_runtime_seconds:.2f}s",
            "",
            "Stage Reports:",
        ]
        for r in result.stage_reports:
            lines.append(
                f"  {r.name}: {r.status} ({r.runtime_seconds:.2f}s) — "
                f"in:{r.input_count} out:{r.output_count}"
            )
            if r.errors:
                for e in r.errors:
                    lines.append(f"    ERROR: {e}")
        if result.errors:
            lines.append("")
            lines.append("Pipeline Errors:")
            for e in result.errors:
                lines.append(f"  - {e}")
        if result.fun_facts:
            lines.append("")
            lines.append("Fun Facts:")
            for f in result.fun_facts:
                lines.append(f"  • {f}")

        report_path = self._out_path("report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Pipeline report saved to {report_path}")


def main() -> None:
    """CLI entry point for the full pipeline."""
    parser = argparse.ArgumentParser(description="Run the WikiWaves pipeline")
    parser.add_argument(
        "--date",
        type=lambda s: datetime.date.fromisoformat(s),
        default=None,
        help="Episode date (ISO-8601). Defaults to today.",
    )
    parser.add_argument(
        "--topics",
        type=int,
        default=5,
        help="Number of topics to select (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for final artifacts.",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp",
        help="Temp directory for intermediate artifacts.",
    )
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        date=args.date,
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        topic_count=args.topics,
    )
    result = orchestrator.run()

    print(f"\n{'=' * 60}")
    print(f"Pipeline {result.status.upper()}")
    print(f"Runtime: {result.total_runtime_seconds:.2f}s")
    print(f"Topics: {len(result.topics)}")
    print(f"Output: {result.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
