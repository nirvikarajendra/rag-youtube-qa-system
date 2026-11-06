"""YouTube helpers: URL parsing, transcript fetching (with cache) and metadata."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptError(RuntimeError):
    """Raised when a transcript can't be fetched; the message is safe to show users."""


def extract_video_id(url_or_id: str) -> str:
    """Accept a bare id or any common YouTube URL form and return the 11-char video id."""
    value = url_or_id.strip()
    if _ID_RE.match(value):
        return value

    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    parts = [p for p in parsed.path.split("/") if p]

    candidate = None
    if host == "youtu.be" and parts:
        candidate = parts[0]
    elif host in {"youtube.com", "music.youtube.com", "youtube-nocookie.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in {"embed", "shorts", "live", "v"}:
            candidate = parts[1]

    if candidate and _ID_RE.match(candidate):
        return candidate
    raise ValueError(f"Could not find a YouTube video id in {url_or_id!r}")


def format_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def video_url(video_id: str, start: float = 0) -> str:
    return f"https://youtu.be/{video_id}?t={int(start)}"


@dataclass
class Segment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Transcript:
    video_id: str
    title: str
    language: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(
            video_id=data["video_id"],
            title=data["title"],
            language=data["language"],
            segments=[Segment(**s) for s in data["segments"]],
        )


def fetch_title(video_id: str) -> str:
    """Title via YouTube's public oEmbed endpoint (no API key). Falls back to the id."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("title") or video_id
    except (requests.RequestException, ValueError):
        log.warning("Could not fetch title for %s", video_id)
        return video_id


def fetch_transcript(video_id: str, languages: tuple[str, ...] = ("en",)) -> Transcript:
    """Fetch captions in a preferred language, else translate (or take) whatever exists."""
    api = YouTubeTranscriptApi()
    try:
        try:
            fetched = api.fetch(video_id, languages=list(languages))
        except NoTranscriptFound:
            available = list(api.list(video_id))
            if not available:
                raise
            transcript = available[0]
            if transcript.is_translatable and any(
                t.language_code == "en" for t in transcript.translation_languages
            ):
                transcript = transcript.translate("en")
            log.info("No %s captions for %s; using %s", languages, video_id, transcript.language_code)
            fetched = transcript.fetch()
    except TranscriptsDisabled as exc:
        raise TranscriptError(f"Captions are disabled for video {video_id}.") from exc
    except VideoUnavailable as exc:
        raise TranscriptError(f"Video {video_id} is unavailable.") from exc
    except NoTranscriptFound as exc:
        raise TranscriptError(f"No transcript found for video {video_id}.") from exc
    except YouTubeTranscriptApiException as exc:
        # Includes IP blocks (common on cloud hosts) and age restrictions.
        raise TranscriptError(f"Could not fetch transcript for {video_id}: {type(exc).__name__}") from exc

    segments = [
        Segment(text=" ".join(s.text.split()), start=float(s.start), duration=float(s.duration))
        for s in fetched.snippets
        if s.text.strip()
    ]
    if not segments:
        raise TranscriptError(f"Transcript for {video_id} is empty.")
    return Transcript(
        video_id=video_id,
        title=fetch_title(video_id),
        language=fetched.language_code,
        segments=segments,
    )


def load_transcript(video_id: str, cache_dir: Path, languages: tuple[str, ...] = ("en",)) -> Transcript:
    """Return the cached transcript if present, otherwise fetch and cache it."""
    path = cache_dir / f"{video_id}.json"
    if path.exists():
        return Transcript.from_dict(json.loads(path.read_text()))
    transcript = fetch_transcript(video_id, languages)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False))
    return transcript


def list_cached_transcripts(cache_dir: Path) -> list[dict]:
    """id, title and duration of every cached video (for the CLI list and the UI dropdown)."""
    videos = []
    for path in sorted(cache_dir.glob("*.json")):
        t = Transcript.from_dict(json.loads(path.read_text()))
        videos.append({"video_id": t.video_id, "title": t.title, "duration": t.duration})
    return videos
