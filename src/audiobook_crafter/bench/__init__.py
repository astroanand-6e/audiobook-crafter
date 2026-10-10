"""Benchmarks: how well an LLM casts a book, and how well a TTS engine performs it.

Gold labels are JSON lines, one quote per line:
    {"chapter": 1, "text": "Is it from London?", "speaker": "Clara Hartwell"}
`speaker` may be a character id or any name/alias from the character map.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import analyze_chapter
from ..llm import create_llm
from ..models import Book, ChapterScript, CharacterMap, normalize_name
from ..project import Project
from ..synth.runner import load_manifest


@dataclass
class GoldQuote:
    chapter: int
    text: str
    speaker: str


@dataclass
class AttributionReport:
    label: str
    total: int = 0
    matched: int = 0
    correct: int = 0
    per_speaker: dict[str, list[int]] = field(default_factory=dict)  # gold speaker -> [correct, total]
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def coverage(self) -> float:
        return self.matched / self.total if self.total else 0.0


def load_gold(path: Path) -> list[GoldQuote]:
    quotes = []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        try:
            quotes.append(GoldQuote(int(item["chapter"]), str(item["text"]), str(item["speaker"])))
        except KeyError as err:
            raise ValueError(f"{path}:{n} is missing {err}") from err
    return quotes


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower().replace("’", "'"))).strip()


def _overlap(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _same_speaker(gold: str, predicted: str, characters: CharacterMap) -> bool:
    if gold == predicted:
        return True
    found = characters.find(gold)
    if found is not None:
        return found.id == predicted
    character = characters.get(predicted)
    return bool(character) and normalize_name(gold) in {normalize_name(n) for n in character.names()}


def score_attribution(
    scripts: dict[int, ChapterScript], characters: CharacterMap, gold: list[GoldQuote], label: str = ""
) -> AttributionReport:
    report = AttributionReport(label=label)
    used: set[str] = set()
    for quote in gold:
        report.total += 1
        stats = report.per_speaker.setdefault(quote.speaker, [0, 0])
        stats[1] += 1
        script = scripts.get(quote.chapter)
        if script is None:
            continue
        want = _norm(quote.text)
        best, best_score = None, 0.0
        for segment in script.segments:
            if segment.kind != "dialogue" or segment.id in used:
                continue
            have = _norm(segment.text)
            if have == want:
                score = 2.0
            elif len(min(have, want, key=len)) >= 8 and (want in have or have in want):
                score = 1.5
            else:
                score = _overlap(have, want)
            if score > best_score:
                best, best_score = segment, score
        if best is None or best_score < 0.6:
            continue
        used.add(best.id)
        report.matched += 1
        if _same_speaker(quote.speaker, best.speaker, characters):
            report.correct += 1
            stats[0] += 1
        else:
            report.confusions[(quote.speaker, best.speaker)] += 1
    return report


def compare_llms(
    project: Project, book: Book, chapters: list[int], gold: list[GoldQuote], llm_specs: list[str]
) -> list[AttributionReport]:
    """Analyse the same chapters with each LLM in a scratch copy of the project and score each run."""
    reports = []
    for spec in llm_specs:
        slug = re.sub(r"[^\w.-]+", "_", spec)
        scratch = Project(project.root / "bench" / slug)
        if scratch.root.exists():
            shutil.rmtree(scratch.root)
        scratch.config = project.config.model_copy(deep=True)
        scratch.config.llm.model = spec
        scratch.save_config()
        scratch.save_book(book)
        llm = create_llm(scratch.config.llm)
        try:
            scripts = {i: analyze_chapter(scratch, book, i, llm) for i in chapters}
        finally:
            if llm is not None:
                llm.release()
        reports.append(score_attribution(scripts, scratch.load_characters(), gold, label=spec))
    return reports


def tts_report(project: Project, chapter: int) -> dict[str, float | int | str]:
    """Clip-level numbers for a synthesized chapter: QA failures, length, real-time factor."""
    clips = [c for c in load_manifest(project, chapter) if c.path]
    seconds = sum(c.seconds for c in clips)
    generation = []
    for clip in clips:
        meta = project.resolve(clip.path or "").with_suffix(".json")
        if meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            if "gen_seconds" in info:
                generation.append(float(info["gen_seconds"]))
    report: dict[str, float | int | str] = {
        "clips": len(clips),
        "audio_minutes": round(seconds / 60, 2),
        "qa_failures": sum(1 for c in clips if c.issue),
        "voices": len({c.voice for c in clips}),
    }
    if generation:
        report["real_time_factor"] = round(sum(generation) / max(seconds, 1e-6), 3)
        report["median_gen_seconds"] = round(statistics.median(generation), 2)
    return report
