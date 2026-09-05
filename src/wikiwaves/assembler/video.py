"""Video assembly — build visualizer video from episode audio using VisGen."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

try:
    from visgen import AudioVisualizerVideo, ColorBackground, ImageBackground, ColorScheme
    from visgen.effects import GlowEffect, VignetteEffect
    from visgen.overlay import ImageOverlay
except ImportError as exc:
    _VSGEN_ERROR = exc
    AudioVisualizerVideo = None  # type: ignore[misc,assignment]
else:
    _VSGEN_ERROR = None


def _ensure_visgen() -> None:
    if _VSGEN_ERROR:
        raise RuntimeError(
            "visgen is required for video generation. Install it:\n"
            "  git clone https://github.com/yourname/visgen.git\n"
            "  cd visgen && pip install -e .\n"
            "  or: uv pip install -e ./visgen"
        ) from _VSGEN_ERROR


def _make_title_card(
    output_path: Path,
    title: str = "WikiWaves",
    subtitle: str = "",
    size: tuple[int, int] = (1080, 1080),
) -> Path:
    """Generate a simple cover image with PIL if none is provided."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", size, color=(15, 15, 30))
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Center title
    bbox = draw.textbbox((0, 0), title, font=font_large)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2 - 40
    draw.text((x, y), title, fill=(0, 200, 255), font=font_large)

    # Subtitle
    if subtitle:
        bbox2 = draw.textbbox((0, 0), subtitle, font=font_small)
        text_w2 = bbox2[2] - bbox2[0]
        x2 = (size[0] - text_w2) // 2
        y2 = y + text_h + 30
        draw.text((x2, y2), subtitle, fill=(200, 200, 200), font=font_small)

    img.save(str(output_path))
    return output_path


def build_visualizer_video(
    audio_path: str | Path,
    output_path: str | Path,
    image_path: str | Path | None = None,
    bg_image_path: str | Path | None = None,
    logo_path: str | Path | None = None,
    title: str = "WikiWaves",
    subtitle: str = "",
    fps: int = 30,
    duration: float | None = None,
) -> Path:
    """Create a circular FFT visualizer video from an audio file.

    Args:
        audio_path: Path to the episode WAV.
        output_path: Path for the output MP4.
        image_path: Cover image for the circular visualizer center.
                     If None, a title card is generated.
        bg_image_path: Optional background image. If None, uses solid dark color.
        logo_path: Optional logo image overlaid on the video (e.g., corner watermark).
        title: Title text for the generated cover image.
        subtitle: Subtitle text (e.g., date).
        fps: Video frame rate.
        duration: Max video length in seconds (default = full audio).

    Returns:
        Path to the generated MP4.
    """
    _ensure_visgen()

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path is None:
        card_path = output_path.with_suffix(".cover.png")
        image_path = _make_title_card(card_path, title=title, subtitle=subtitle)
        logger.info(f"Generated title card → {image_path}")

    # Background
    if bg_image_path:
        background = ImageBackground(
            str(bg_image_path),
            effect=["blur", "grayscale"],
            blur_radius=12.0,
            fit_mode="cover",
        )
    else:
        background = ColorBackground(color=(15, 15, 30))

    # Overlays
    overlays = []
    if logo_path:
        overlays.append(
            ImageOverlay(
                str(logo_path),
                position=(60, 60),
                size=(120, 120),
                anchor="lt",
            )
        )
        logger.info(f"Adding logo overlay → {logo_path}")

    logger.info("Rendering visualizer video (this may take a while)…")

    viz = AudioVisualizerVideo(
        audio_path=str(audio_path),
        image_path=str(image_path),
        output_path=str(output_path),
        duration=duration,
        fps=fps,
        circle_radius=180,
        bar_count=64,
        bar_max_length=220,
        colors=ColorScheme(bar=(0, 200, 255)),
        smooth_factor=0.3,
        background=background,
        frame_effects=[
            GlowEffect(strength=0.3, radius=8.0),
            VignetteEffect(strength=0.4),
        ],
    )
    viz.render_single(overlays=overlays)

    logger.info(f"Video saved → {output_path}")
    return output_path


def build_segment_video(
    audio_path: str | Path,
    output_path: str | Path,
    image_path: str | Path | None = None,
    bg_image_path: str | Path | None = None,
    logo_path: str | Path | None = None,
    title: str = "WikiWaves",
    subtitle: str = "",
    fps: int = 30,
) -> Path:
    """Build a visualizer video for a single audio segment.

    Same API as build_visualizer_video but with a shorter, simpler title card.
    """
    return build_visualizer_video(
        audio_path=audio_path,
        output_path=output_path,
        image_path=image_path,
        bg_image_path=bg_image_path,
        logo_path=logo_path,
        title=title,
        subtitle=subtitle,
        fps=fps,
        duration=None,
    )
