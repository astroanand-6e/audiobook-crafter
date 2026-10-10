"""Stage 4: speak every segment of a chapter script.

Each TTS request is cached under a hash of everything that shapes its audio (engine, model version,
text, emotion, voice clip, settings, seed). Re-runs only generate what changed, so fixing one
speaker in a script or one voice in cast.yaml re-renders exactly those lines."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from ..analysis import drop_speech_tags
from ..audio.dsp import clean_clip
from ..casting.voices import resolve_voice
from ..models import Cast, ChapterScript, Segment
from ..project import Project, write_json
from ..text.normalize import normalize_for_tts, speak_heading
from ..text.sentences import chunk_text
from ..tts import EnginePool
from ..tts.base import Line, TTSEngine, Voice
from .qa import ClipChecker

log = logging.getLogger(__name__)
Progress = Callable[[int, int, str], None]


class Clip(BaseModel):
    segment: str
    piece: int
    kind: str
    block: int
    speaker: str = ""
    voice: str = ""
    text: str = ""
    path: str | None = None  # relative to the project; None for pauses
    seconds: float = 0.0
    pause_ms: int | None = None
    issue: str = ""


def clip_key(engine: TTSEngine, line: Line, voice: Voice, seed: int) -> str:
    options = {k: v for k, v in engine.options.items() if k not in ("python",)}
    payload = {
        "engine": engine.name,
        "version": engine.version,
        "options": options,
        "line": asdict(line),
        "voice": voice.fingerprint,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


def spoken_text(segment: Segment, language: str) -> str:
    if segment.kind == "heading":
        return speak_heading(segment.text, language)
    return normalize_for_tts(segment.text, language)


def manifest_path(project: Project, chapter: int) -> Path:
    return project.render_dir / f"ch{chapter:03d}.clips.json"


def load_manifest(project: Project, chapter: int) -> list[Clip]:
    path = manifest_path(project, chapter)
    if not path.exists():
        raise FileNotFoundError(
            f"No clips for chapter {chapter}; run `abcraft synth --chapter {chapter}` first"
        )
    return [Clip.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _render(
    engine: TTSEngine, line: Line, voice: Voice, checker: ClipChecker, retries: int
) -> tuple[np.ndarray, int, str]:
    best: tuple[np.ndarray, int, str, float] | None = None
    for attempt in range(retries + 1):
        audio, sr = engine.synthesize(line, voice, voice.seed + 7919 * attempt)
        audio = clean_clip(audio, sr)
        verdict = checker.check(line.text, audio, sr)
        if best is None or verdict.score < best[3]:
            best = (audio, sr, verdict.issue, verdict.score)
        if not verdict.issue:
            break
        log.info("Retaking %r (%s)", line.text[:50], verdict.issue)
    assert best is not None
    return best[0], best[1], best[2]


def synthesize_chapter(
    project: Project,
    script: ChapterScript,
    cast: Cast,
    pool: EnginePool,
    *,
    limit: int | None = None,
    progress: Progress | None = None,
) -> list[Clip]:
    config = project.config
    language = project.language
    checker = ClipChecker(config.qa, language, config.tts.device)
    segments = script.segments[:limit] if limit else script.segments
    if config.style.mode == "drama":
        segments = drop_speech_tags(segments)

    plan: list[tuple[Segment, str]] = []
    for segment in segments:
        text = "" if segment.kind == "pause" else spoken_text(segment, language)
        if segment.kind == "pause" or any(ch.isalnum() for ch in text):
            plan.append((segment, text))
    total = len(plan)

    voices: dict[tuple[str, str], Voice] = {}
    warned: set[str] = set()
    clips: list[Clip] = []
    for done, (segment, text) in enumerate(plan, start=1):
        if segment.kind == "pause":
            clips.append(
                Clip(
                    segment=segment.id, piece=0, kind="pause", block=segment.block, pause_ms=segment.pause_ms
                )
            )
            continue
        voice_id, profile = cast.resolve(segment.speaker)
        engine_name = profile.engine or config.tts.engine
        engine = pool.get(engine_name)
        if not engine.supports(language) and engine_name not in warned:
            log.warning("%s doesn't list language '%s'; expect an accent or errors", engine_name, language)
            warned.add(engine_name)
        voice = voices.get((voice_id, engine_name))
        if voice is None:
            voice = voices[(voice_id, engine_name)] = resolve_voice(
                project, voice_id, profile, engine_name, config.tts.seed
            )
        pieces = chunk_text(text, config.tts.max_chars or engine.default_max_chars)
        for index, piece in enumerate(pieces):
            line = Line(
                text=piece,
                language=language,
                emotion=segment.emotion,
                intensity=min(1.0, segment.intensity * config.style.emotion_strength),
                direction=segment.direction,
                tags=segment.tags if index == 0 else [],
            )
            key = clip_key(engine, line, voice, voice.seed)
            path = project.cache_dir / "tts" / key[:2] / f"{key}.wav"
            meta = path.with_suffix(".json")
            if path.exists() and meta.exists():
                info = json.loads(meta.read_text(encoding="utf-8"))
                seconds, issue = float(info["seconds"]), str(info.get("issue", ""))
            else:
                started = time.perf_counter()
                audio, sr, issue = _render(engine, line, voice, checker, config.qa.retries)
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), audio, sr, subtype="PCM_16")
                seconds = len(audio) / sr
                write_json(
                    meta,
                    {
                        "text": piece,
                        "voice": voice_id,
                        "engine": engine_name,
                        "seconds": seconds,
                        "gen_seconds": round(time.perf_counter() - started, 3),
                        "issue": issue,
                    },
                )
            clips.append(
                Clip(
                    segment=segment.id,
                    piece=index,
                    kind=segment.kind,
                    block=segment.block,
                    speaker=segment.speaker,
                    voice=voice_id,
                    text=piece,
                    path=project.relative(path),
                    seconds=round(seconds, 3),
                    pause_ms=segment.pause_ms if index == len(pieces) - 1 else None,
                    issue=issue,
                )
            )
        if progress:
            progress(done, total, segment.text[:60])
    checker.close()
    write_json(manifest_path(project, script.chapter), [c.model_dump() for c in clips])
    return clips
