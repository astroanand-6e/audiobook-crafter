"""The whole flow: ingest -> analyze (LLM) -> cast -> design voices -> synthesize -> assemble.

Each stage reads and writes files in the project folder and skips work that's already done, so a
run can be interrupted and resumed, and any hand edit (a speaker in scripts/ch003.json, a voice in
cast.yaml) only redoes what it affects."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .analysis import analyze_chapter, scan_characters, speaker_counts
from .assemble import ChapterAudio, assemble_chapter, build_audiobook
from .casting.cast import auto_cast
from .casting.design import design_voices
from .ingest import load_document
from .llm import create_llm
from .models import Book, ChapterScript
from .project import Project
from .synth.runner import Clip, synthesize_chapter
from .tts import EnginePool, engine_available

log = logging.getLogger(__name__)


class Reporter(Protocol):
    def stage(self, name: str, detail: str = "") -> None: ...
    def progress(self, task: str, done: int, total: int, note: str = "") -> None: ...
    def warn(self, message: str) -> None: ...


class QuietReporter:
    def stage(self, name: str, detail: str = "") -> None:
        log.info("%s %s", name, detail)

    def progress(self, task: str, done: int, total: int, note: str = "") -> None:
        pass

    def warn(self, message: str) -> None:
        log.warning(message)


@dataclass
class RunResult:
    scripts: list[ChapterScript] = field(default_factory=list)
    audio: list[ChapterAudio] = field(default_factory=list)
    m4b: Path | None = None
    cast_changes: list[str] = field(default_factory=list)
    designed: list[str] = field(default_factory=list)
    issues: list[Clip] = field(default_factory=list)


def ingest(project: Project, source: Path, pages: tuple[int, int] | None = None, force: bool = False) -> Book:
    if project.book_path.exists() and not force:
        return project.load_book()
    book = load_document(source, pages)
    config = project.config
    if config.language:
        book.language = config.language  # the user said so
    else:
        config.language = book.language  # detected
    config.title = config.title or book.title
    config.author = config.author or book.author
    config.source = str(Path(source).resolve())
    project.save_config()
    project.save_book(book)
    return book


def analyze(
    project: Project,
    book: Book,
    chapters: list[int],
    reporter: Reporter,
    *,
    reanalyze: bool = False,
    scan_book: bool = False,
) -> list[ChapterScript]:
    todo = [i for i in chapters if reanalyze or project.load_script(i) is None]
    scripts = {i: project.load_script(i) for i in chapters if i not in todo}
    if todo:
        llm = create_llm(project.config.llm)
        if llm is None:
            reporter.warn(
                "No LLM configured: speakers come from simple rules (`--llm ollama:<model>` is much better)"
            )
        try:
            if llm is not None and scan_book:
                reporter.stage("Characters", f"reading the whole book with {llm.name}")
                every = [c.index for c in book.chapters if c.kind == "chapter"]
                scan_characters(
                    project, book, llm, every, lambda task, n, total: reporter.progress(task, n, total)
                )
            for index in todo:
                reporter.stage("Analyze", f"chapter {index} with {llm.name if llm else 'rules'}")
                scripts[index] = analyze_chapter(
                    project,
                    book,
                    index,
                    llm,
                    scan=not scan_book,
                    progress=lambda task, n, total: reporter.progress(task, n, total),
                )
        finally:
            if llm is not None and project.config.llm.unload_after:
                llm.release()  # free the GPU before TTS loads
    return [scripts[i] for i in chapters if scripts.get(i) is not None]  # type: ignore[misc]


def cast(project: Project) -> list[str]:
    characters = project.load_characters()
    cast_ = project.load_cast()
    changes = auto_cast(project, characters, cast_, speaker_counts(project))
    project.save_cast(cast_)
    return changes


def design(project: Project, book: Book, reporter: Reporter, pool: EnginePool) -> list[str]:
    cast_ = project.load_cast()
    missing = [v for v, p in cast_.voices.items() if p.design and not p.reference and not p.same_as]
    if not missing:
        return []
    config = project.config.tts
    engine_name = "mock" if config.engine == "mock" else config.design_engine
    if not engine_available(engine_name, config):
        reporter.warn(
            f"{len(missing)} voices need designing ({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}) but "
            f"{engine_name} isn't installed. Add clips under voices/, set presets in cast.yaml, or install qwen-tts."
        )
        return []
    reporter.stage("Voices", f"designing {len(missing)} voices with {engine_name}")
    engine = pool.get(engine_name)
    try:
        return design_voices(
            project,
            book,
            cast_,
            engine,
            on_progress=lambda n, total, v: reporter.progress("voices", n, total, v),
        )
    finally:
        pool.release(engine_name)  # make room for the speaking model


def synthesize(
    project: Project,
    scripts: list[ChapterScript],
    reporter: Reporter,
    pool: EnginePool,
    preview: int | None = None,
) -> dict[int, list[Clip]]:
    cast_ = project.load_cast()
    results: dict[int, list[Clip]] = {}
    for script in scripts:
        reporter.stage("Speak", f"chapter {script.chapter} with {project.config.tts.engine}")
        results[script.chapter] = synthesize_chapter(
            project,
            script,
            cast_,
            pool,
            limit=preview,
            progress=lambda n, total, note: reporter.progress("lines", n, total, note),
        )
    return results


def run(
    project: Project,
    source: Path,
    chapters: list[int],
    reporter: Reporter | None = None,
    *,
    pages: tuple[int, int] | None = None,
    reanalyze: bool = False,
    preview: int | None = None,
    scan_book: bool = False,
) -> RunResult:
    reporter = reporter or QuietReporter()
    result = RunResult()
    book = ingest(project, source, pages)
    result.scripts = analyze(project, book, chapters, reporter, reanalyze=reanalyze, scan_book=scan_book)
    result.cast_changes = cast(project)
    pool = EnginePool(project.config.tts)
    try:
        result.designed = design(project, book, reporter, pool)
        clips = synthesize(project, result.scripts, reporter, pool, preview)
    finally:
        pool.close()
    reporter.stage("Master", "pacing, levels, loudness, MP3 + M4B")
    for index in chapters:
        result.audio.append(assemble_chapter(project, book, index))
        result.issues += [c for c in clips.get(index, []) if c.issue]
    result.m4b = build_audiobook(project, book)
    return result
