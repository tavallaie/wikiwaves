
 # WikiWaves — High-Level Design (HLD)

 > **Goal:** A fully automated, daily pipeline that transforms Wikipedia's "On this day" events and trending contributions into a polished, two-person dialogue podcast (and optionally video cast). The system runs hands-off every 24 hours, producing publish-ready audio content with zero human intervention in the core loop.
 > **Philosophy:** Keep it simple, modular, and Pythonic (Zen of Python). Explicit is better than implicit. Readability counts.

 ---

 ## Table of Contents
 1. [Definitions & Glossary](#1-definitions--glossary)
 2. [Goals & Objectives](#2-goals--objectives)
 3. [Architecture Overview](#3-architecture-overview)
 4. [Module Breakdown (Detailed)](#4-module-breakdown-detailed)
 5. [Data Flow](#5-data-flow)
 6. [Directory Structure](#6-directory-structure)
 7. [Key Design Decisions](#7-key-design-decisions)
 8. [API References](#8-api-references)
 9. [Execution Plan for Agents](#9-execution-plan-for-agents)
 10. [Sample Output](#10-sample-output)
 11. [Non-Goals](#11-non-goals)
 12. [Success Metrics](#12-success-metrics)

 ---

 ## 1. Definitions & Glossary

 | Term | Definition |
 |------|------------|
 | **Episode** | One complete podcast output for a single calendar day. Contains intro, 5 topic segments, and outro. |
 | **Topic** | One historical event or person selected from Wikipedia's "On this day" feed. The atomic unit of content. |
 | **Chunk** | A single line of dialogue assigned to one speaker. The atomic unit of audio generation. |
 | **Segment** | A collection of chunks covering one topic. An episode contains exactly 5 segments. |
 | **Fun Fact** | An interesting statistic or anecdote about Wikipedia contributions in the last 24 hours. Used as seasoning in intro/outro. |
 | **Enrichment** | The process of expanding a topic's content by fetching additional linked Wikipedia pages suggested by an LLM. |
 | **Base Page** | The primary Wikipedia article for a selected topic (e.g., the "Napoleon" page). |
 | **Related Page** | A secondary Wikipedia article fetched during enrichment to add depth (e.g., "Elba" when the base page is "Napoleon"). |
 | **Script** | The complete structured dialogue for an episode, represented as an ordered list of chunks in JSON format. |
 | **Bumpers** | Short musical intros/outros or transitions added to the beginning and end of the audio track. |
 | **Voice Map** | The configuration that assigns a specific TTS voice ID to each speaker role (speaker1 vs speaker2). |
 | **Pipeline** | The end-to-end sequence of stages that transforms raw Wikipedia data into a finished audio file. |
 | **Orchestrator** | The central controller that executes each pipeline stage in order and handles stage-to-stage data handoff. |
 | **Fail-Soft** | A design principle where individual stage failures are logged and bypassed rather than crashing the entire pipeline. |
 | **MVP** | Minimum Viable Product — the simplest version that produces a working podcast end-to-end. |
 | **TTS** | Text-to-Speech — the technology that converts written text into spoken audio. |
 | **LLM** | Large Language Model — the AI model used for link suggestion and script writing. |
 | **Curator** | The logic layer that scores, ranks, and selects topics from a larger pool of candidates. |
 | **Fetcher** | The module responsible for all external API calls to Wikipedia and Wikimedia services. |
 | **Scripter** | The module that converts aggregated factual content into conversational dialogue. |
 | **Assembler** | The module that combines individual audio files into a single continuous podcast track. |
 | **Renderer** | The module that creates video content by syncing audio with visual elements (Phase 2). |
 | **Cron** | A time-based job scheduler that triggers the pipeline to run automatically at a fixed time each day. |
 | **Grounding** | The constraint that all facts in the script must originate from Wikipedia content, preventing hallucination. |
 | **Dated Folder** | A filesystem directory named by date (e.g., `output/2026-05-05/`) that contains all artifacts for one episode. |
 | **Pause Marker** | A metadata value attached to each chunk indicating seconds of silence to insert after that line. |
 | **Emotion Tag** | A metadata label on each chunk (e.g., "excited", "curious") intended to guide TTS prosody or future animation. |
 | **Stub** | A very short or incomplete Wikipedia article. The curator avoids these to ensure rich content. |
 | **Featured Article** | A Wikipedia article recognized as high-quality. The curator boosts these in scoring. |
 | **Recent Changes** | The live feed of edits made to Wikipedia in the last 24 hours. Source for fun facts. |
 | **MediaWiki API** | The official programmatic interface for reading Wikipedia content. |
 | **Wikimedia REST API** | A modern RESTful interface providing structured feeds like "On this day". |
 | **Provider** | A vendor or service that supplies a capability (e.g., OpenAI as an LLM provider, ElevenLabs as a TTS provider). |
 | **Abstraction Layer** | A thin interface that hides provider-specific details, allowing easy swapping of vendors. |
 | **Artifact** | Any file produced by the pipeline (JSON, MP3, MP4, log file). |
 | **Idempotency** | The property that re-running the pipeline for the same date produces identical output. |
 | **Retry Logic** | Automatic re-attempt of failed API calls with exponential backoff. |
 | **Structured Logging** | Machine-readable log output (JSON lines) for debugging and monitoring. |
 | **Persona** | The character traits assigned to each speaker (e.g., "curious host" vs "knowledgeable co-host"). |
 | **Prompt Template** | A reusable text file with placeholders, used to construct LLM instructions. |
 | **pydub** | A Python library for audio manipulation (splicing, adding silence, format conversion). |
 | **ffmpeg** | A command-line tool for audio/video processing. Used as a fallback or for complex operations. |
 | **moviepy** | A Python library for video editing. Candidate for the renderer module. |
 | **LiteLLM** | A unified interface for calling multiple LLM providers with the same code. |
 | **Ollama** | A tool for running open-source LLMs locally. Optional provider for cost reduction. |
 | **Edge TTS** | Microsoft's free text-to-speech service. Low quality but zero cost, ideal for development. |
 | **ElevenLabs** | A premium TTS service known for natural-sounding voices. |
 | **RSS Feed** | A standardized XML format for distributing podcast episodes to platforms like Spotify or Apple Podcasts. |
 | **GitHub Actions** | A CI/CD service that can run the pipeline on a schedule (cron) and publish artifacts. |
 | **Docker** | A containerization platform for packaging the entire application with its dependencies. |
 | **Environment Variable** | A configuration value stored outside the codebase (e.g., API keys in `.env` file). |
 | **Dataclass** | A Python decorator that auto-generates boilerplate for data-holding classes. |
 | **Type Hint** | Python syntax for declaring expected data types, improving code clarity and IDE support. |

 ---

 ## 2. Goals & Objectives

 ### Primary Goal
 Build a fully autonomous system that, once per day, produces a publish-ready podcast episode exploring five historical events that occurred on that calendar date, enriched with Wikipedia's latest community activity.

 ### Secondary Goals
 | # | Objective | Why It Matters |
 |---|-----------|----------------|
 | 1 | **Zero-touch daily operation** | The pipeline must run on a schedule without human input, from data fetch to final audio file. |
 | 2 | **Factually grounded content** | Every claim in the script must trace back to a specific Wikipedia article. No hallucinated history. |
 | 3 | **Conversational, not robotic** | The dialogue should feel like two real people chatting, not an encyclopedia being read aloud. |
 | 4 | **Modular provider swapping** | Changing the LLM or TTS vendor should require editing only one file and one config value. |
 | 5 | **Easy maintenance for non-experts** | A new developer should understand the entire pipeline in under 30 minutes. |
 | 6 | **Cost-efficient at scale** | The MVP should run for pennies per episode. Premium voices are optional upgrades. |
 | 7 | **Observable and debuggable** | Every stage produces clear logs and intermediate artifacts for inspection when things go wrong. |
 | 8 | **Extensible to video** | The audio pipeline should be designed so that adding video visuals later is a plug-in, not a rewrite. |

 ### Listener Experience Goals
 | # | Objective |
 |---|-----------|
 | 1 | Listener learns something genuinely interesting they didn't know. |
 | 2 | The two voices are distinguishable and have personality. |
 | 3 | Pacing feels natural — not rushed, not dragging. |
 | 4 | Fun facts about Wikipedia itself add meta-interest for engaged listeners. |
 | 5 | Episode length is predictable (10–15 minutes). |

 ---

 ## 3. Architecture Overview

 ```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         WikiWaves Pipeline (Daily Cron)                     │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │                                                                             │
 │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
 │  │   Fetcher    │───▶│   Curator    │───▶│  Enricher    │───▶│  Scripter │  │
 │  │  (Wikipedia) │    │ (Pick topics)│    │ (LLM expand) │    │ (Dialogue)│  │
 │  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
 │                                                                             │
 │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
 │  │   TTS Engine │───▶│   Assembler  │───▶│   Renderer   │                  │
 │  │  (Audio gen) │    │ (Stitch MP3) │    │ (Video cast) │                  │
 │  └──────────────┘    └──────────────┘    └──────────────┘                   │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
 ```

 ### Stage Summary
 | Stage | Input | Output | Core Question Answered |
 |-------|-------|--------|------------------------|
 | **Fetcher** | Date | Raw Wikipedia JSON | What happened today? What's trending? |
 | **Curator** | Raw events + edits | 5 topic titles + 3 fun facts | Which stories are most interesting? |
 | **Enricher** | 5 base pages | 5 base pages + related pages | What context makes this story richer? |
 | **Scripter** | Enriched pages + fun facts | JSON script | How do two people talk about this? |
 | **TTS** | JSON script | ~60 individual MP3 chunks | What does it sound like? |
 | **Assembler** | MP3 chunks + pauses | Single MP3 podcast | How do we make one continuous show? |
 | **Renderer** | MP3 + slides | MP4 video cast | What does it look like? |

 ---

 ## 4. Module Breakdown (Detailed)

 ### 4.1 Fetcher — Data Acquisition

 **Responsibility:** All external communication with Wikipedia and Wikimedia. No business logic — just clean data retrieval and normalization.

 **Design Principle:** The fetcher is a "dumb pipe." It doesn't decide what's interesting; it just fetches what it's asked to fetch and returns structured objects.

 #### Capabilities

 | Capability | Purpose | Source |
 |------------|---------|--------|
 | On This Day feed | Events, births, deaths, holidays for a calendar date | Wikimedia REST API |
 | Recent changes | Significant edits in the last 24 hours | MediaWiki Action API |
 | Full page content | Article text, internal links, external links | MediaWiki Action API |
 | Page views | Article popularity over last 30 days | Wikimedia PageViews API |
 | Search | Find canonical article titles from fragments | MediaWiki Action API |
 | Article quality check | Verify if article is "Featured" or "Good" | MediaWiki Action API |
 | Bulk fetching | Concurrent, rate-limited fetching of multiple pages | Wrapper around single-page fetch |

 #### Data Models

 **WikiPage** — A complete Wikipedia article with extracted content and links.
 - Canonical title
 - Raw HTML and cleaned plain text
 - Internal and external link lists
 - Source URL
 - Word count
 - Featured article flag
 - Fetch timestamp

 **OnThisDayEvent** — A single event from the "On this day" feed.
 - Year of event
 - Human-readable description
 - Related article titles
 - Event type (selected, births, deaths, events, holidays)

 **TrendingEdit** — A significant edit from the recent changes feed.
 - Article title
 - Editor username
 - Edit timestamp
 - Bytes changed
 - Edit summary
 - Bot and minor edit flags

 #### Error Handling
 - Network timeout: Retry up to 3 times with exponential backoff (1s, 2s, 4s).
 - Rate limit (HTTP 429): Wait for Retry-After header, then retry.
 - Page not found: Return empty result instead of crashing. Caller decides fallback.
 - Malformed API response: Log the raw response, return empty list, and continue pipeline.

 ---

 ### 4.2 Curator — Topic Selection

 **Responsibility:** Transform a large pool of raw events into a tight, diverse, high-quality selection of 5 topics. Also extracts the most interesting community activity as "fun facts."

 **Design Principle:** The curator is the "editorial voice." It applies heuristics to simulate what a human producer would pick. All scoring must be transparent and adjustable via config.

 #### Capabilities

 | Capability | Purpose |
 |------------|---------|
 | Score events | Compute a composite quality score for each candidate event |
 | Pick top topics | Select the highest-scoring events, applying diversity filters |
 | Extract fun facts | Select the most interesting recent edits from the community |
 | Ensure diversity | Post-process selection to prevent similar themes or eras |
 | Fetch quality metadata | Retrieve article length and featured status for scoring |

 #### Scoring Dimensions

 | Dimension | Weight | How It's Measured | Rationale |
 |-----------|--------|-------------------|-----------|
 | **Content Richness** | 30% | Word count of primary article + number of linked pages | Longer articles with more connections provide more script material |
 | **Recency Bias** | 20% | Events from last 200 years score higher than ancient history | Modern events are more relatable to general audiences |
 | **Diversity** | 20% | Penalty if selected topics share categories or centuries | A good episode covers different eras and themes |
 | **Page Popularity** | 15% | 30-day average page views of primary article | Popular topics indicate broad audience interest |
 | **Article Quality** | 15% | Boost for Featured Articles or Good Articles | Higher quality means more reliable, richer content |

 #### Fun Fact Selection Criteria
 1. Exclude all bot edits.
 2. Exclude minor edits unless byte change is large.
 3. Prefer edits with descriptive comments.
 4. Prefer edits on well-known articles.
 5. Prefer large byte changes (major additions or rewrites).
 6. Format as conversational one-liners.

 ---

 ### 4.3 Enricher — Content Expansion via LLM

 **Responsibility:** For each selected topic, decide which additional Wikipedia articles would add the most valuable context, then fetch them. This is where the podcast goes beyond surface-level facts.

 **Design Principle:** The enricher is the "research assistant." It uses the LLM's reasoning to find connections a simple keyword search would miss, but it always grounds suggestions in actual Wikipedia links.

 #### Capabilities

 | Capability | Purpose |
 |------------|---------|
 | Suggest links | Ask LLM which internal links from the base page would deepen the story |
 | Validate suggestions | Filter out hallucinated or unavailable link suggestions |
 | Fetch related pages | Retrieve full content of validated suggestions |
 | Bundle enrichment | Package base page with related pages and LLM reasoning |
 | Rank suggestions | Optional post-ranking by relevance or length |

 #### Data Model

 **EnrichedPage** — A base page bundled with its LLM-suggested related pages.
 - Base WikiPage
 - 3-5 related WikiPages
 - LLM's explanation for why these were picked
 - Combined word count

 #### LLM Prompt Rules for Link Suggestion
 - ONLY pick from the available internal links list. Do not invent titles.
 - Prefer links that provide cause, consequence, or cultural impact.
 - Avoid links that are just lists or disambiguation pages.
 - Return only a list of article titles.

 #### Error Handling
 - LLM returns invalid format: Log the raw response, return empty enrichment, continue with base page only.
 - Suggested link is a redirect: Follow the redirect and fetch the target page.
 - Suggested link is a stub (very short): Log warning, skip it, try next suggestion.
 - All suggestions fail: Episode segment uses only the base page. Not a pipeline failure.

 ---

 ### 4.4 Scripter — Dialogue Script Generation

 **Responsibility:** Transform aggregated factual content into a natural, engaging two-person conversation. This is the creative heart of the pipeline.

 **Design Principle:** The scripter is the "writer's room." It must balance factual accuracy with conversational flow. Every claim must be traceable to a specific Wikipedia article.

 #### Capabilities

 | Capability | Purpose |
 |------------|---------|
 | Generate full script | Orchestrate intro, 5 segments, and outro creation |
 | Write intro | Opening banter including fun facts and host introductions |
 | Write segments | Dialogue for each topic, weaving base and related page facts |
 | Write outro | Closing remarks with callback to earlier topics |
 | Format to JSON | Convert script to structured chunk format |
 | Validate grounding | Check that every factual claim maps to a source article |

 #### Data Models

 **ScriptChunk** — A single line of dialogue.
 - Speaker identifier (speaker1 or speaker2)
 - Spoken text (max 2 sentences recommended)
 - Emotion tag (neutral, excited, curious, surprised, somber)
 - Pause duration after line
 - Source articles this line draws from
 - Segment index

 **Script** — The complete episode script.
 - Episode title with date
 - Publish date
 - Intro chunks
 - 5 segment chunk lists
 - Outro chunks
 - Fun facts used
 - Total chunk count
 - Estimated duration
 - Grounding report (any ungrounded claims)

 #### Dialogue Style Guidelines
 | Element | Guideline |
 |---------|-----------|
 | Sentence length | Max 25 words per line |
 | Vocabulary | Avoid jargon unless immediately explained |
 | Transitions | Each segment ends with a verbal bridge to the next topic |
 | Questions | Speaker 1 should ask 2-3 questions per segment |
 | Reactions | At least one genuine emotional reaction per segment |
 | Callback | Outro should reference one earlier topic for continuity |

 #### Speaker Personas
 - **Speaker 1 (Alex):** The curious, enthusiastic host. Asks questions, reacts with amazement, keeps energy up. Uses casual language.
 - **Speaker 2 (Jordan):** The knowledgeable, calm co-host. Provides facts, adds context, occasionally surprises Alex with little-known details. Speaks in concise sentences.

 ---

 ### 4.5 TTS Engine — Text-to-Speech Generation

 **Responsibility:** Convert each script chunk into a high-quality audio segment. Abstract the TTS provider so vendors can be swapped without touching pipeline logic.

 **Design Principle:** The TTS engine is the "recording studio." It handles voice selection, audio format, and quality. One chunk = one file = maximum flexibility.

 #### Capabilities

 | Capability | Purpose |
 |------------|---------|
 | Synthesize single chunk | Convert one dialogue line to audio |
 | Batch synthesis | Parallel generation of multiple chunks |
 | Voice mapping | Assign specific voices to speaker roles |
 | Voice validation | Check if a voice ID exists for current provider |
 | List available voices | Show all voices for setup and configuration |

 #### Provider Abstraction
 All TTS providers implement a common interface with three methods:
 1. Synthesize text to audio file
 2. List available voices with metadata
 3. Validate a voice ID exists

 #### Supported Providers
 | Provider | Cost | Quality | Best For |
 |----------|------|---------|----------|
 | Edge TTS | Free | Acceptable | Development, testing, MVP |
 | OpenAI TTS | Low | Good | Production MVP, natural prosody |
 | ElevenLabs | Medium | Excellent | Premium production, distinct voices |

 #### Audio Specifications
 - Format: MP3 (MVP), WAV optional for post-processing
 - Sample rate: 44.1 kHz
 - Bitrate: 192 kbps
 - Channels: Mono (MVP), Stereo optional
 - Naming: Segment index, chunk index, and speaker in filename

 ---

 ### 4.6 Assembler — Audio Stitching

 **Responsibility:** Combine all individual chunk audio files into one seamless podcast episode. Add structural silence and optional music bumpers.

 **Design Principle:** The assembler is the "sound engineer." It ensures consistent volume, proper pacing, and professional transitions.

 #### Capabilities

 | Capability | Purpose |
 |------------|---------|
 | Stitch chunks | Concatenate audio files with silence gaps |
 | Add silence | Generate precise silent segments |
 | Normalize volume | Adjust loudness to podcast standard (-16 LUFS) |
 | Add bumpers | Prepend intro music and append outro music |
 | Crossfade | Optional smooth transition between segments |
 | Write metadata | Add ID3 tags (title, artist, date, description) |

 #### Assembly Sequence
 1. For each chunk in script order: load audio, append to running total, append silence.
 2. Normalize final volume to podcast standard.
 3. If intro music provided: load, fade out over first 3 seconds of speech, prepend.
 4. If outro music provided: load, fade in over last 3 seconds of speech, append.
 5. Write final MP3 with ID3 tags.
 6. Return path to final file.

 ---

 ### 4.7 Renderer — Video Cast Generation (Phase 2)

 **Responsibility:** Turn the finished audio into a video cast with visual elements. This is an optional enhancement, not required for MVP.

 **Design Principle:** The renderer is the "video producer." It creates simple but professional visuals that complement the audio without distracting from it.

 #### Capabilities

 | Capability | Purpose |
 |------------|---------|
 | Render video | Master method producing final MP4 |
 | Generate slides | Create static images for each segment |
 | Generate title card | Create opening title screen |
 | Fetch topic images | Retrieve lead images from Wikipedia articles |
 | Sync slides to audio | Display each slide for duration of its segment |
 | Add subtitles | Optional burned-in subtitles for accessibility |

 #### Visual Style
 - Resolution: 1920x1080 (1080p)
 - Frame rate: 30 fps
 - Background: Dark navy with subtle wave pattern
 - Title font: Bold sans-serif, white, centered
 - Body text: Light gray, smaller, for context snippets
 - Image treatment: Slight vignette, rounded corners
 - Transitions: Simple fade, 0.5 seconds

 ---

 ### 4.8 Config — Centralized Settings

 **Responsibility:** Single source of truth for all configurable values. No hardcoded constants anywhere else in the codebase.

 **Design Principle:** The config is the "production bible." Changing one value here should change behavior everywhere without code edits.

 #### Configuration Categories

 | Category | What It Controls |
 |----------|------------------|
 | Wikipedia | API URLs, user agent, rate limits |
 | LLM | Provider, model, temperature, timeout, retries |
 | TTS | Provider, voice IDs, speed, output format |
 | Pipeline | Topics per episode, link limits, fun fact count, directories |
 | Content | Language, target duration, chunk constraints |
 | Assembly | Volume normalization, music paths, fade duration |
 | Video | Enable flag, resolution, frame rate, colors |
 | Scheduling | Daily run time, timezone |

 ---

 ### 4.9 Orchestrator — Pipeline Runner

 **Responsibility:** Wire all modules together in the correct order, handle data handoff between stages, manage the execution lifecycle, and produce the final artifact.

 **Design Principle:** The orchestrator is the "director." It knows the sequence but delegates all actual work to specialized modules. It handles errors gracefully and always produces a report.

 #### Capabilities

 | Capability | Purpose |
 |------------|---------|
 | Run full pipeline | Execute all stages for one calendar day |
 | Run single stage | Execute one stage with logging and timing |
 | Cleanup temp files | Remove intermediate files older than 7 days |
 | Generate report | Produce human-readable summary of the run |
 | Save artifacts | Organize all outputs into a dated folder |

 #### Pipeline Result
 - Episode date
 - Success or failure status
 - Path to final podcast file
 - Path to final video file (if enabled)
 - Generated script
 - Stage-by-stage performance reports
 - Any errors or warnings
 - Total runtime
 - Estimated API cost

 #### Stage Report
 - Stage name
 - Status (success, partial, failed)
 - Runtime in seconds
 - Input count
 - Output count
 - Any errors encountered

 ---

 ## 5. Data Flow

 ```
 Day N, 00:01 UTC
 │
 ├─▶ Fetcher: Get "On This Day" feed for today's date
 │   ├─▶ Parse into structured events
 │   └─▶ Save raw events to temporary storage
 │
 ├─▶ Fetcher: Get recent changes from last 24 hours
 │   ├─▶ Filter bots and minor edits
 │   ├─▶ Parse into structured edits
 │   └─▶ Save raw edits to temporary storage
 │
 ├─▶ Curator: Score all events
 │   ├─▶ Fetch page views for candidate pages
 │   ├─▶ Check featured article status
 │   ├─▶ Apply diversity filter
 │   ├─▶ Select top 5 topics
 │   ├─▶ Select top 3 fun facts
 │   └─▶ Save selection to temporary storage
 │
 ├─▶ Fetcher: Get full page content for 5 selected topics
 │   ├─▶ For each: text, links, external links, word count
 │   └─▶ Save base pages to temporary storage
 │
 ├─▶ Enricher: Expand content via LLM
 │   ├─▶ For each page, ask LLM which links add richest context
 │   ├─▶ Validate suggestions against actual page links
 │   ├─▶ Fetch full content of valid suggestions
 │   ├─▶ Bundle base pages with related pages
 │   └─▶ Save enriched pages to temporary storage
 │
 ├─▶ Scripter: Generate dialogue
 │   ├─▶ Aggregate all texts and fun facts
 │   ├─▶ Prompt LLM with speaker personas and rules
 │   ├─▶ Parse response into structured chunks
 │   ├─▶ Validate factual grounding
 │   ├─▶ Package into complete script
 │   └─▶ Save script to temporary storage
 │
 ├─▶ TTS: Generate audio per chunk
 │   ├─▶ For each chunk:
 │   │   ├─▶ Look up voice from voice map
 │   │   ├─▶ Call TTS provider API
 │   │   └─▶ Save audio file
 │   ├─▶ Log success or failure per chunk
 │   └─▶ Save audio manifest to temporary storage
 │
 ├─▶ Assembler: Combine audio
 │   ├─▶ Load all audio files in script order
 │   ├─▶ Insert silence between chunks
 │   ├─▶ Normalize volume to podcast standard
 │   ├─▶ Add intro and outro music if configured
 │   ├─▶ Write ID3 metadata tags
 │   └─▶ Save final podcast to dated output folder
 │
 └─▶ Renderer (Optional): Create video
     ├─▶ Generate title card
     ├─▶ Generate segment slides
     ├─▶ Fetch topic images from Wikipedia
     ├─▶ Sync slides to audio segment timing
     ├─▶ Add fade transitions
     └─▶ Save final video to dated output folder
 ```

 ---

 ## 6. Directory Structure

 ```
 wikiwaves/
 │
 ├── config.py                    # Centralized configuration
 ├── orchestrator.py              # Pipeline runner and lifecycle manager
 ├── run.py                       # CLI entry point
 │
 ├── fetcher/                     # Data acquisition
 │   ├── wiki_api.py              # Wikipedia API client
 │   ├── models.py                # Data models
 │   └── exceptions.py            # Fetcher errors
 │
 ├── curator/                     # Topic selection
 │   ├── selector.py              # Scoring and ranking
 │   ├── scoring.py               # Scoring functions
 │   └── models.py                # Selection models
 │
 ├── enricher/                    # Content expansion
 │   ├── expander.py              # LLM link suggestion
 │   ├── validators.py            # Suggestion validation
 │   └── prompts.py               # Prompt templates
 │
 ├── scripter/                    # Dialogue generation
 │   ├── writer.py                # Script writer
 │   ├── formatter.py             # JSON formatter
 │   ├── personas.py              # Speaker definitions
 │   └── prompts.py               # Dialogue prompts
 │
 ├── tts/                         # Text-to-speech
 │   ├── engine.py                # TTS orchestrator
 │   ├── models.py                # Voice models
 │   └── providers/               # Provider implementations
 │       ├── base.py              # Abstract base
 │       ├── edge.py              # Edge TTS
 │       ├── openai.py            # OpenAI TTS
 │       └── elevenlabs.py        # ElevenLabs TTS
 │
 ├── assembler/                   # Audio stitching
 │   ├── stitcher.py              # Concatenation
 │   ├── normalizer.py            # Volume normalization
 │   ├── bumper.py                # Music overlay
 │   └── metadata.py              # ID3 tags
 │
 ├── renderer/                    # Video generation (Phase 2)
 │   ├── video.py                 # Video orchestrator
 │   ├── slides.py                # Slide generation
 │   ├── images.py                # Image fetching
 │   └── subtitles.py             # Subtitle burn-in
 │
 ├── utils/                       # Shared utilities
 │   ├── text_cleaner.py          # HTML to plain text
 │   ├── retry.py                 # Exponential backoff
 │   ├── logger.py                # Structured logging
 │   └── file_manager.py          # Folder management
 │
 ├── prompts/                     # LLM prompt templates
 │   ├── suggest_links.txt
 │   ├── write_dialogue.txt
 │   ├── system_persona.txt
 │   └── fun_fact_intro.txt
 │
 ├── tests/                       # Test suite
 │   ├── test_fetcher.py
 │   ├── test_curator.py
 │   ├── test_enricher.py
 │   ├── test_scripter.py
 │   ├── test_tts.py
 │   └── test_assembler.py
 │
 ├── output/                      # Generated episodes (gitignored)
 │   └── YYYY-MM-DD/
 │       ├── podcast.mp3
 │       ├── video.mp4
 │       ├── script.json
 │       └── report.txt
 │
 ├── tmp/                         # Intermediate artifacts (gitignored)
 │   └── YYYY-MM-DD/
 │       ├── raw_events.json
 │       ├── raw_edits.json
 │       ├── selection.json
 │       ├── base_pages.json
 │       ├── enriched_pages.json
 │       ├── script.json
 │       └── audio/
 │           ├── chunk files
 │           └── manifest.json
 │
 ├── requirements.txt
 ├── requirements-dev.txt
 ├── .env.example
 ├── .gitignore
 ├── Dockerfile
 ├── docker-compose.yml
 └── README.md
 ```

 ---

 ## 7. Key Design Decisions

 | Decision | Rationale | Trade-off |
 |----------|-----------|-----------|
 | One audio file per chunk | Easier retry, precise pause control, parallel generation, granular debugging | More files to manage; assembly step required |
 | Dataclasses for all models | Immutable, self-documenting, easy JSON serialization, IDE-friendly | Slightly more boilerplate than dicts |
 | Prompts in separate text files | Non-coders can tweak tone without touching Python | More file I/O; need template engine |
 | TTS provider abstraction | Switch vendors for cost, quality, or availability | More initial setup; maintain multiple integrations |
 | No database (file-based) | Sufficient for MVP, zero ops overhead, easy to inspect | Harder to query history; add DB if scaling |
 | Pure functions where possible | Easier to test, debug, parallelize | May require passing more parameters |
 | Fail-soft on enrichment | Log warning and skip; never crash pipeline | Episode may be slightly shorter but always ships |
 | Volume normalization | Podcast listeners expect consistent loudness | Adds processing time |
 | Structured JSON logging | Enables monitoring and debugging in production | Slightly more verbose |
 | Dated output folders | Natural organization, easy archival, idempotent re-runs | More disk usage if not cleaned |
 | Grounding validation | Ensures factual integrity, prevents hallucinations | Adds validation step; may flag legitimate inferences |
 | Edge TTS for development | Zero cost during dev and testing | Lower voice quality during dev |
 | No real-time streaming | Batch daily is simpler, cheaper, sufficient | Episodes have delay between day start and publish |
 | English MVP only | Best feed and TTS quality in English | Limits audience; i18n is Phase 3 |

 ---

 ## 8. API References

 ### Wikipedia & Wikimedia APIs

 | API | Purpose | Rate Limit |
 |-----|---------|------------|
 | On This Day feed | Daily events, births, deaths | 200 req/sec |
 | Page Parse | Full article content and links | 200 req/sec |
 | Recent Changes | Live edit stream | 200 req/sec |
 | Page Views | Article popularity | 100 req/sec |
 | Search | Title search | 200 req/sec |
 | Page Props | Featured article check | 200 req/sec |

 **Important:** Always include a descriptive User-Agent header with contact info.

 ### LLM APIs (configurable)

 | Provider | Model Options | Cost Estimate |
 |----------|---------------|---------------|
 | OpenAI | gpt-4o, gpt-4o-mini | Low per 1K tokens |
 | Anthropic | claude-3-5-sonnet, claude-3-haiku | Low per 1K tokens |
 | Local (Ollama) | llama3, mistral | Hardware cost only |
 | LiteLLM | Any supported model | Varies by backend |

 ### TTS APIs (configurable)

 | Provider | Voice Options | Cost |
 |----------|---------------|------|
 | ElevenLabs | 1000+ voices, custom clones | Medium per 1K chars |
 | OpenAI TTS | alloy, echo, fable, onyx, nova, shimmer | Low per 1K chars |
 | Edge TTS | 50+ Microsoft voices | Free |

 ---

 ## 9. Execution Plan for Agents

 ### Phase 1 — Core Pipeline (MVP)
 *Target: Working podcast in 2 weeks*

 | # | Task | Module | Priority |
 |---|------|--------|----------|
 | 1 | Create config with all settings | Root | P0 |
 | 2 | Create environment variable template | Root | P0 |
 | 3 | Implement Wikipedia API client | Fetcher | P0 |
 | 4 | Define data models | Fetcher | P0 |
 | 5 | Implement topic scoring and selection | Curator | P0 |
 | 6 | Implement fun fact extraction | Curator | P0 |
 | 7 | Implement LLM link suggestion | Enricher | P0 |
 | 8 | Write link suggestion prompt | Prompts | P0 |
 | 9 | Implement dialogue generation | Scripter | P0 |
 | 10 | Implement JSON script formatter | Scripter | P0 |
 | 11 | Write dialogue prompt | Prompts | P0 |
 | 12 | Implement TTS provider abstraction | TTS | P0 |
 | 13 | Implement Edge TTS provider | TTS | P0 |
 | 14 | Implement audio stitching | Assembler | P0 |
 | 15 | Implement text cleaning utilities | Utils | P0 |
 | 16 | Implement retry logic | Utils | P1 |
 | 17 | Implement structured logging | Utils | P1 |
 | 18 | Implement pipeline orchestrator | Root | P0 |
 | 19 | Implement CLI entry point | Root | P0 |
 | 20 | Write tests for each module | Tests | P1 |
 | 21 | End-to-end integration test | Tests | P0 |
 | 22 | Write setup documentation | Root | P1 |

 ### Phase 2 — Polish & Video
 *Target: Premium audio + video in 1 additional week*

 | # | Task | Module | Priority |
 |---|------|--------|----------|
 | 23 | Implement OpenAI TTS provider | TTS | P1 |
 | 24 | Implement ElevenLabs TTS provider | TTS | P1 |
 | 25 | Add music bumper support | Assembler | P1 |
 | 26 | Add ID3 metadata writing | Assembler | P1 |
 | 27 | Implement video generation | Renderer | P2 |
 | 28 | Implement slide generation | Renderer | P2 |
 | 29 | Implement image fetching | Renderer | P2 |
 | 30 | Create Docker container | Root | P1 |
 | 31 | Set up GitHub Actions cron | CI/CD | P1 |

 ### Phase 3 — Scale & Distribution
 *Target: Production-ready system*

 | # | Task | Module | Priority |
 |---|------|--------|----------|
 | 32 | Add database for episode history | Data | P2 |
 | 33 | Build web dashboard | Web | P3 |
 | 34 | Generate RSS feed | Distribution | P2 |
 | 35 | Add subtitle generation | Renderer | P3 |
 | 36 | Implement multi-language support | International | P3 |
 | 37 | Add analytics | Analytics | P3 |

 ---

 ## 10. Sample Output

 ### Script JSON Structure

 ```json
 {
   "title": "WikiWaves — May 5, 2026",
   "publish_date": "2026-05-05",
   "estimated_duration": 12.5,
   "total_chunks": 67,
   "fun_facts_used": [
     "In the last 24 hours, user 'HistoryBuff42' added 2,400 bytes to the article on 'Cinco de Mayo' — basically a whole new section!",
     "Wikipedia editors fixed over 80 broken links yesterday.",
     "The article on 'SpaceX' saw 12 edits in one day."
   ],
   "intro": [
     {
       "speaker": "speaker1",
       "text": "Welcome to WikiWaves! I'm Alex.",
       "emotion": "excited",
       "pause_after": 0.5,
       "source_articles": [],
       "segment_index": 0
     },
     {
       "speaker": "speaker2",
       "text": "And I'm Jordan. Today is May 5th, and we've got five fascinating stories from Wikipedia.",
       "emotion": "neutral",
       "pause_after": 0.3,
       "source_articles": [],
       "segment_index": 0
     }
   ],
   "segments": [
     [
       {
         "speaker": "speaker2",
         "text": "Speaking of May 5th, did you know that in 1821, Napoleon Bonaparte died in exile on Saint Helena?",
         "emotion": "curious",
         "pause_after": 0.5,
         "source_articles": ["Napoleon"],
         "segment_index": 1
       },
       {
         "speaker": "speaker1",
         "text": "Wait, Saint Helena? That's in the middle of the Atlantic, right?",
         "emotion": "surprised",
         "pause_after": 0.3,
         "source_articles": ["Saint Helena"],
         "segment_index": 1
       }
     ]
   ],
   "outro": [
     {
       "speaker": "speaker1",
       "text": "That's all for today's WikiWaves!",
       "emotion": "excited",
       "pause_after": 0.3,
       "source_articles": [],
       "segment_index": 6
     },
     {
       "speaker": "speaker2",
       "text": "Thanks for listening. We'll be back tomorrow with five more stories.",
       "emotion": "neutral",
       "pause_after": 0.5,
       "source_articles": [],
       "segment_index": 6
     }
   ],
   "grounding_report": []
 }
 ```

 ### Final Episode Output

 ```
 output/2026-05-05/
 ├── podcast.mp3              # 12.5 min, normalized audio
 ├── script.json              # Full dialogue with metadata
 ├── report.txt               # Human-readable run summary
 ├── transcript.txt           # Plain text for accessibility
 └── (Phase 2)
     ├── video.mp4            # 1080p synced video
     ├── thumbnail.jpg        # Title card image
     └── chapters.json        # Timestamp markers
 ```

 ---

 ## 11. Non-Goals (Out of Scope)

 | # | Out-of-Scope Item | Rationale |
 |---|-------------------|-----------|
 | 1 | Real-time streaming or live generation | Batch daily is simpler, cheaper, sufficient |
 | 2 | Multi-language episodes in MVP | Best feed and TTS quality in English; i18n is Phase 3 |
 | 3 | User-generated topic selection or voting | Automated curation is the core value proposition |
 | 4 | Live Wikipedia editing or contribution | System is read-only |
 | 5 | Advanced video editing (motion graphics, animations) | Static slides are sufficient for MVP |
 | 6 | Interactive podcast features (Q&A, polls) | Adds complexity beyond scope |
 | 7 | Social media auto-posting | Distribution is separate concern |
 | 8 | Real-time analytics dashboard | Phase 3 consideration |
 | 9 | Custom voice cloning | ElevenLabs supports this but not required for MVP |
 | 10 | Music generation | Use royalty-free stock music or none |

 ---

 ## 12. Success Metrics

 | Metric | Target | How Measured |
 |--------|--------|--------------|
 | Pipeline runtime | Under 15 minutes end-to-end | Orchestrator timer |
 | TTS cost per episode | Under $1 using Edge or OpenAI | Provider billing API |
 | Script coherence | No hallucinated facts | Grounding validation report |
 | Audio quality | 44.1kHz stereo MP3, no glitches | Manual listening test |
 | Maintainability | New contributor adds TTS provider in under 30 min | Time trial |
 | Episode length | 10–15 minutes predictable | Script estimator |
 | Daily success rate | 95%+ episodes generate without manual intervention | Pipeline success logs |
 | Fact accuracy | 100% of claims traceable to Wikipedia | Grounding validation |
 | Voice distinction | Listeners can tell speaker1 from speaker2 | Listener survey |
 | Fun fact relevance | 80%+ of fun facts feel natural in dialogue | Editor review |

 ---

 *Document version: 2.0*
 *Last updated: 2026-05-05*
 *Status: Ready for agent implementation*