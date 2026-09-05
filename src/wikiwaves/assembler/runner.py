"""Assembler Runner — concat audio segments and build visualizer video.

Usage:
    uv run python -m wikiwaves.assembler.runner 2026-05-11
    uv run python -m wikiwaves.assembler.runner 2026-05-11 --video
    uv run python -m wikiwaves.assembler.runner 2026-05-11 --video --title "WikiWaves" --subtitle "May 11, 2026"
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from loguru import logger

from wikiwaves.assembler.audio import build_episode, read_segment_wavs
from wikiwaves.assembler.video import build_visualizer_video, build_segment_video


def run(
    date_str: str | None = None,
    output_dir: str = "output",
    silence_sec: float = 0.8,
    build_video: bool = False,
    segment_videos: bool = False,
    title: str = "WikiWaves",
    subtitle: str = "",
    image_path: str | Path | None = None,
    bg_image_path: str | Path | None = None,
    logo_path: str | Path | None = None,
    fps: int = 30,
) -> Path:
    """Assemble episode audio and optionally build visualizer videos.

    Returns the path to episode.wav.
    """
    date_str = date_str or datetime.date.today().isoformat()
    audio_dir = Path(output_dir) / date_str / "audio"

    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        raise FileNotFoundError(audio_dir)

    # Assemble audio
    episode_wav = build_episode(audio_dir, silence_sec=silence_sec)

    # Build full episode video
    if build_video:
        episode_mp4 = audio_dir / "episode.mp4"
        build_visualizer_video(
            audio_path=episode_wav,
            output_path=episode_mp4,
            image_path=image_path,
            bg_image_path=bg_image_path,
            logo_path=logo_path,
            title=title,
            subtitle=subtitle or date_str,
            fps=fps,
        )

    # Build individual segment videos
    if segment_videos:
        segments = read_segment_wavs(audio_dir)
        for i, (wav_path, _wav, _sr) in enumerate(segments):
            seg_mp4 = wav_path.with_suffix(".mp4")
            seg_title = wav_path.stem.replace("_", " ")
            logger.info(f"[{i + 1}/{len(segments)}] Rendering segment video → {seg_mp4}")
            build_segment_video(
                audio_path=wav_path,
                output_path=seg_mp4,
                image_path=image_path,
                bg_image_path=bg_image_path,
                logo_path=logo_path,
                title=title,
                subtitle=seg_title,
                fps=fps,
            )

    return episode_wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble podcast episode and build video.")
    parser.add_argument("date", nargs="?", help="Date folder (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--silence", type=float, default=0.8, help="Silence between segments (sec).")
    parser.add_argument("--video", action="store_true", help="Build full episode visualizer video.")
    parser.add_argument("--segment-videos", action="store_true", help="Build a visualizer video for each segment.")
    parser.add_argument("--title", default="WikiWaves", help="Video title text.")
    parser.add_argument("--subtitle", default="", help="Video subtitle text (defaults to date).")
    parser.add_argument("--image", default=None, help="Cover image for circular visualizer.")
    parser.add_argument("--bg-image", default=None, help="Background image (blurred + grayscale).")
    parser.add_argument("--logo", default=None, help="Logo overlay image (top-left corner).")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    args = parser.parse_args()

    run(
        date_str=args.date,
        output_dir=args.output_dir,
        silence_sec=args.silence,
        build_video=args.video,
        segment_videos=args.segment_videos,
        title=args.title,
        subtitle=args.subtitle,
        image_path=args.image,
        bg_image_path=args.bg_image,
        logo_path=args.logo,
        fps=args.fps,
    )
