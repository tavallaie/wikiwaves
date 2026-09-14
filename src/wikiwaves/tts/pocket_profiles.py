"""PocketTTS variant profiles — registry and resolution.

Official Kyutai languages, community checkpoints (e.g. Farsi v2), and future
trained variants differ in text prep, voice rules, and generation defaults.
Keep those differences here so ``PocketTTSEngine`` stays a thin wrapper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PocketProfile:
    """Behaviour for one PocketTTS checkpoint family."""

    id: str
    # When set, a --config string matching this regex selects the profile.
    config_pattern: re.Pattern[str] | None = None
    # When set, a --language in this set selects the profile.
    language_names: frozenset[str] = field(default_factory=frozenset)
    # passthrough = send text as-is; farsi_g2p = phonemize Persian first.
    text_prep: str = "passthrough"
    # catalog_ok = alba/etc.; wav_required = local/hf/http prompt only.
    voice_mode: str = "catalog_ok"
    max_voice_sec: float | None = None
    # Applied when the caller still has the Supertonic-oriented runner defaults.
    default_steps: int | None = None
    default_temp: float | None = None
    default_eos_threshold: float | None = None
    default_frames_after_eos: int | None = None
    # Treat steps==5 / eos==-4 as "unset runner defaults" and replace them.
    replace_runner_step_default: bool = False
    replace_runner_eos_default: bool = False
    runaway_retry: bool = False
    warn_near_silent: bool = False


OFFICIAL = PocketProfile(
    id="official",
    language_names=frozenset(
        {
            "english",
            "english_2026-01",
            "english_2026-04",
            "french_24l",
            "german_24l",
            "portuguese",
            "italian",
            "spanish_24l",
        }
    ),
    text_prep="passthrough",
    voice_mode="catalog_ok",
)

FARSI_V2 = PocketProfile(
    id="farsi-v2",
    config_pattern=re.compile(r"farsi[-_]v2", re.IGNORECASE),
    text_prep="farsi_g2p",
    voice_mode="wav_required",
    max_voice_sec=5.0,
    default_steps=1,
    default_temp=0.3,
    default_eos_threshold=-2.0,
    default_frames_after_eos=0,
    replace_runner_step_default=True,
    replace_runner_eos_default=True,
    runaway_retry=True,
    warn_near_silent=True,
)

# Any --config that does not match a specialised profile.
COMMUNITY = PocketProfile(
    id="community",
    text_prep="passthrough",
    voice_mode="wav_required",
)

# Specialised profiles are matched before COMMUNITY / OFFICIAL fall-backs.
_SPECIALISED: tuple[PocketProfile, ...] = (FARSI_V2,)

PROFILES: dict[str, PocketProfile] = {
    OFFICIAL.id: OFFICIAL,
    FARSI_V2.id: FARSI_V2,
    COMMUNITY.id: COMMUNITY,
}


def list_profile_ids() -> list[str]:
    """Return registered profile ids in stable order."""
    return [OFFICIAL.id, FARSI_V2.id, COMMUNITY.id]


def resolve_profile(
    *,
    profile: str | None = None,
    config: str | None = None,
    language: str | None = None,
) -> PocketProfile:
    """Pick a profile: explicit id, then config/language match, then fall-back.

    Order:
      1. ``profile`` when set (must be a known id)
      2. first specialised profile whose ``config_pattern`` matches ``config``
      3. ``official`` when ``language`` is a known built-in name
      4. ``community`` when ``config`` is set but unmatched
      5. ``official`` otherwise
    """
    if profile:
        try:
            return PROFILES[profile]
        except KeyError as exc:
            known = ", ".join(list_profile_ids())
            raise ValueError(f"Unknown PocketTTS profile {profile!r}. Known: {known}") from exc

    if config:
        for candidate in _SPECIALISED:
            if candidate.config_pattern and candidate.config_pattern.search(config):
                return candidate
        return COMMUNITY

    if language and language in OFFICIAL.language_names:
        return OFFICIAL

    return OFFICIAL
