"""PDF ingest built on PyMuPDF.

Pipeline: text lines with font metrics -> drop running heads, page numbers, footnotes -> rebuild
paragraphs from geometry (indents, gaps, short last lines) -> find chapters (PDF outline first,
then heading typography) -> Book.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from ..models import Block, Book, Chapter
from .common import (
    classify_chapters,
    clean_text,
    detect_language,
    is_scene_break,
    join_lines,
    looks_like_chapter_heading,
    title_from_headings,
)


class ScannedPDFError(RuntimeError):
    pass


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    page: int
    page_width: float
    page_height: float


@dataclass
class Para:
    lines: list[Line] = field(default_factory=list)
    first_on_page: bool = False
    gap_before: float = 0.0
    heading: bool = False
    on_toc_page: bool = False

    @property
    def text(self) -> str:
        return clean_text(join_lines([line.text for line in self.lines]))

    @property
    def page(self) -> int:
        return self.lines[0].page

    @property
    def last_page(self) -> int:
        return self.lines[-1].page

    @property
    def size(self) -> float:
        return max(line.size for line in self.lines)


_PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,7})(?:\s+of\s+\d+)?\s*$|^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$",
    re.IGNORECASE,
)
_TOC_LINE = re.compile(r"^\D.{1,90}?(?:\s*\.{2,}\s*|\s+)\d{1,4}$")
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)
_FOOTNOTE_MARK = re.compile(r"^\s*[\d*†‡§¶]+\s*$")
_SENTENCE_END = (".", "!", "?", "”", '"', "’", "'", "…", ":", ")", "।", "。", "！", "？")
_LINE_END_HYPHEN = re.compile(r"[\u00ad\u2010\u2011]$")
_NOTE_START = re.compile(r"^\s*(?:\d{1,3}[.)]?|[*†‡§¶])\s*\S")


def _page_lines(page: pymupdf.Page, index: int) -> list[Line]:
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    data = page.get_text("dict", flags=flags, sort=True)
    width, height = page.rect.width, page.rect.height
    lines: list[Line] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for raw in block.get("lines", []):
            if abs(raw.get("dir", (1, 0))[1]) > 0.1:  # rotated margin text
                continue
            parts: list[str] = []
            sizes: Counter[float] = Counter()
            bold_chars = total_chars = 0
            for span in raw.get("spans", []):
                text = span.get("text", "")
                if span.get("flags", 0) & 1 and _FOOTNOTE_MARK.match(text):  # superscript note reference
                    continue
                parts.append(text)
                n = len(text.strip())
                if n:
                    sizes[round(span.get("size", 0), 1)] += n
                    total_chars += n
                    if span.get("flags", 0) & 16 or _BOLD_FONT.search(span.get("font", "")):
                        bold_chars += n
            # Soft (U+00AD) and Unicode (U+2010/2011) hyphens at a line end become "-" before cleaning
            # removes them, so join_lines() can re-join the word.
            text = clean_text(_LINE_END_HYPHEN.sub("-", "".join(parts).rstrip()))
            if not text or not total_chars:
                continue
            x0, y0, x1, y1 = raw["bbox"]
            lines.append(
                Line(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=sizes.most_common(1)[0][0],
                    bold=bold_chars / total_chars > 0.6,
                    page=index,
                    page_width=width,
                    page_height=height,
                )
            )
    return _merge_same_baseline(lines)


def _merge_same_baseline(lines: list[Line]) -> list[Line]:
    """PyMuPDF sometimes splits one visual line into pieces; glue pieces that share a baseline."""
    merged: list[Line] = []
    for line in lines:
        prev = merged[-1] if merged else None
        if prev and abs(prev.y1 - line.y1) < 0.35 * line.size and line.x0 >= prev.x1 - 1:
            prev.text = (
                f"{prev.text} {line.text}" if line.x0 - prev.x1 > 0.15 * line.size else prev.text + line.text
            )
            prev.x1 = line.x1
            prev.size = max(prev.size, line.size)
            prev.bold = prev.bold and line.bold
        else:
            merged.append(line)
    return merged


def _in_margin(line: Line) -> bool:
    return line.y1 < 0.1 * line.page_height or line.y0 > 0.9 * line.page_height


def _furniture_key(line: Line) -> str:
    return re.sub(r"\d+", "#", line.text.lower()).strip()


def _strip_page_furniture(pages: list[list[Line]]) -> None:
    """Remove running heads/feet (repeated margin lines) and bare page numbers, in place."""
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({_furniture_key(line) for line in lines if _in_margin(line)})
    repeat_threshold = 3 if len(pages) >= 6 else len(pages) + 1
    for i, lines in enumerate(pages):
        pages[i] = [
            line
            for line in lines
            if not (
                _in_margin(line)
                and (_PAGE_NUMBER.match(line.text) or counts[_furniture_key(line)] >= repeat_threshold)
            )
        ]


def _body_size(pages: list[list[Line]]) -> float:
    sizes: Counter[float] = Counter()
    for lines in pages:
        for line in lines:
            sizes[round(line.size)] += len(line.text)
    return float(sizes.most_common(1)[0][0]) if sizes else 11.0


_WORDS_AFTER_A_OR_I = {
    "was", "am", "had", "have", "could", "would", "should", "never", "will", "did", "do", "think", "remember",
    "know", "knew", "saw", "felt", "found", "man", "woman", "few", "long", "little", "great", "moment", "year",
    "day", "night", "week", "cold", "dark", "storm", "light", "voice", "single", "new", "good", "bad",
}  # fmt: skip


def _clean_page(lines: list[Line], body: float) -> list[Line]:
    """Drop footnotes and re-attach drop caps ("T" + "he night was dark").

    A footnote is small type in the lower part of the page that starts with a note marker ("1", "*", "†"),
    plus the small lines that follow it. Other small type (copyright lines, epigraphs, letters) is kept."""
    out: list[Line] = []
    in_notes = False
    for i, line in enumerate(lines):
        small_and_low = line.size < 0.8 * body and line.y0 > 0.65 * line.page_height
        if small_and_low and (in_notes or _NOTE_START.match(line.text)):
            in_notes = True
            continue
        in_notes = in_notes and small_and_low
        if len(line.text) == 1 and line.text.isalpha() and line.size >= 1.6 * body:
            following = next(
                (
                    j
                    for j in range(i + 1, min(i + 4, len(lines)))
                    if lines[j].y0 < line.y1 and lines[j].x0 > line.x0
                ),
                None,
            )
            if following is not None and lines[following].text[:1].isalpha():
                rest = lines[following].text
                first_word = re.split(r"\W", rest.lower(), maxsplit=1)[0]
                joiner = " " if line.text in "AI" and first_word in _WORDS_AFTER_A_OR_I else ""
                lines[following].text = line.text + joiner + rest
                continue
        out.append(line)
    return out


def _is_toc_page(lines: list[Line]) -> bool:
    return sum(1 for line in lines if _TOC_LINE.match(line.text)) >= 5


def _build_paragraphs(pages: list[list[Line]], body: float) -> list[Para]:
    paras: list[Para] = []
    current: Para | None = None
    prev_right = None
    for lines in pages:
        if not lines:
            continue
        body_lines = [line for line in lines if abs(line.size - body) < 1.0] or lines
        x0s = sorted(line.x0 for line in body_lines)
        x1s = sorted(line.x1 for line in body_lines)
        left = x0s[len(x0s) // 10]
        right = x1s[int(len(x1s) * 0.9)]
        gaps = [b.y0 - a.y1 for a, b in zip(body_lines, body_lines[1:]) if 0 <= b.y0 - a.y1 < 3 * body]
        normal_gap = statistics.median(gaps) if gaps else 0.25 * body
        toc_page = _is_toc_page(lines)

        for k, line in enumerate(lines):
            indented = line.x0 - left > 0.45 * line.size
            new_para = current is None
            gap = 0.0
            if current is not None:
                prev = current.lines[-1]
                prev_big = prev.size >= body * 1.15
                line_big = line.size >= body * 1.15
                if prev.page == line.page:
                    gap = line.y0 - prev.y1
                    short_prev = prev.x1 < right - 2 * line.size
                    new_para = (
                        gap > normal_gap + 0.6 * line.size
                        or indented
                        or (short_prev and prev.text.endswith(_SENTENCE_END))
                        or prev_big != line_big
                    )
                else:
                    short_prev = prev_right is not None and prev.x1 < prev_right - 2 * line.size
                    new_para = (
                        indented
                        or prev_big
                        or line_big
                        or (current.text.endswith(_SENTENCE_END) and short_prev)
                        or looks_like_chapter_heading(line.text)
                    )
            if new_para:
                current = Para(first_on_page=(k == 0), gap_before=gap, on_toc_page=toc_page)
                paras.append(current)
            current.lines.append(line)
        prev_right = right

    for para in paras:
        text = para.text
        words = len(text.split())
        centered = all(
            abs((line.x0 + line.x1) / 2 - line.page_width / 2) < 0.08 * line.page_width for line in para.lines
        )
        big = para.size >= body * 1.18
        spaced = para.first_on_page or para.gap_before > 1.5 * body
        para.heading = words <= 14 and (
            big
            or (looks_like_chapter_heading(text) and spaced)
            or (all(line.bold for line in para.lines) and centered and words <= 8)
        )
    return paras


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def _toc_starts(toc: list[list], paras: list[Para]) -> list[tuple[int, str]]:
    entries = [
        (level, clean_text(title), page) for level, title, page, *_ in toc if page >= 1 and title.strip()
    ]
    if len(entries) < 2:
        return []
    per_level = Counter(level for level, _, _ in entries)
    split_level = (
        1 if per_level.get(1, 0) >= 5 else max((lvl for lvl in per_level if lvl <= 2), key=per_level.get)
    )
    by_page: dict[int, list[int]] = {}
    for i, para in enumerate(paras):
        by_page.setdefault(para.page, []).append(i)
    max_page = max(by_page) if by_page else 0

    starts: list[tuple[int, str]] = []
    for level, title, page in entries:
        if level > split_level:
            continue
        page_index = page - 1
        while page_index not in by_page and page_index <= max_page:
            page_index += 1  # outline points at a blank or image-only page
        candidates = by_page.get(page_index, [])
        if not candidates:
            continue
        match = _locate_title(title, [i for i in candidates if not starts or i > starts[-1][0]], paras)
        if match is not None and (not starts or match > starts[-1][0]):
            starts.append((match, title))
    return starts if len(starts) >= 2 else []


def _locate_title(title: str, candidates: list[int], paras: list[Para]) -> int | None:
    """The paragraph where an outline entry's chapter begins: chapters often start mid-page."""
    if not candidates:
        return None
    want = _norm(title)
    want_words = set(want.split())
    best, best_score = None, 0.0
    for i in candidates:
        have = _norm(paras[i].text)
        if not have:
            continue
        if have == want:
            score = 3.0
        elif len(have) >= 3 and (want.startswith(have) or have.startswith(want)):
            score = 2.0
        else:
            words = set(have.split())
            overlap = len(words & want_words) / max(1, len(words | want_words))
            score = overlap if overlap >= 0.6 and len(have) <= 2 * len(want) + 10 else 0.0
        if paras[i].heading:
            score += 0.5
        if score > best_score:
            best, best_score = i, score
    if best is not None and best_score >= 1.0:
        return best
    headings = [i for i in candidates if paras[i].heading and looks_like_chapter_heading(paras[i].text)]
    return headings[0] if headings else candidates[0]


def _heading_starts(paras: list[Para], body: float) -> list[tuple[int, str]]:
    def collect(predicate) -> list[int]:
        found: list[int] = []
        for i, para in enumerate(paras):
            if para.on_toc_page or not para.heading or not predicate(para):
                continue
            if found and found[-1] == i - 1:
                continue  # "CHAPTER ONE" + "The Letter": one start
            found.append(i)
        return found

    starts = collect(lambda p: looks_like_chapter_heading(p.text))
    if len(starts) < 2:
        starts = collect(lambda p: p.first_on_page and p.size >= body * 1.3)
    return [(i, "") for i in starts]


def _chapter_from(paras: list[Para], title: str, index: int) -> Chapter:
    blocks: list[Block] = []
    leading_headings: list[str] = []
    for para in paras:
        text = para.text
        if not text:
            continue
        if is_scene_break(text):
            if blocks and blocks[-1].kind != "scene_break":
                blocks.append(Block(kind="scene_break"))
            continue
        if para.heading:
            if all(b.kind == "heading" for b in blocks):
                leading_headings.append(text)
            blocks.append(Block(kind="heading", text=text))
        else:
            if (
                blocks
                and blocks[-1].kind == "paragraph"
                and para.gap_before > 2.2 * para.size
                and not para.first_on_page
            ):
                blocks.append(Block(kind="scene_break"))  # untagged scene break: a big vertical gap
            blocks.append(Block(kind="paragraph", text=text))
    title = title or title_from_headings(leading_headings) or f"Section {index}"
    pages = (paras[0].page + 1, paras[-1].last_page + 1) if paras else None
    return Chapter(index=index, title=title, pages=pages, blocks=blocks)


def _cover_title(pages: list[list[Line]], body: float) -> str:
    """No title in the PDF metadata: use the biggest text on the first page that has any."""
    first = next((lines for lines in pages if lines), [])
    biggest = max(first, key=lambda line: line.size, default=None)
    return biggest.text if biggest and biggest.size >= 1.5 * body and len(biggest.text) <= 120 else ""


def load_pdf(path: Path, page_range: tuple[int, int] | None = None) -> Book:
    """Extract a book. `page_range` (1-based, inclusive) skips chapter detection: the pages become one chapter."""
    with pymupdf.open(path) as doc:
        metadata = doc.metadata or {}
        first, last = (1, doc.page_count) if page_range is None else page_range
        first, last = max(1, first), min(doc.page_count, last)
        if first > last:
            raise ValueError(f"Page range {page_range} is outside this PDF (1-{doc.page_count})")
        # Running heads are found by repetition, so look at a few pages around a short range too.
        lo, hi = max(1, first - 10), min(doc.page_count, last + 10)
        window = [_page_lines(doc[i], i) for i in range(lo - 1, hi)]
        toc = doc.get_toc(simple=True) if page_range is None else []

    _strip_page_furniture(window)
    pages = window[first - lo : last - lo + 1]
    total_chars = sum(len(line.text) for lines in pages for line in lines)
    if total_chars < 20 * len(pages):
        raise ScannedPDFError(
            f"{path.name} has almost no text layer (scanned?). OCR it first, e.g. `ocrmypdf {path.name} ocr.pdf`."
        )

    body = _body_size(pages)
    pages = [_clean_page(lines, body) for lines in pages]
    paras = _build_paragraphs(pages, body)

    cover_title = _cover_title(window, body) if lo == 1 else ""
    title = clean_text(metadata.get("title") or "") or cover_title or Path(path).stem.replace("_", " ")
    author = clean_text(metadata.get("author") or "")
    if page_range is not None:
        chapters = [_chapter_from(paras, f"Pages {first}-{last}", 1)]
    else:
        starts = _toc_starts(toc, paras) or _heading_starts(paras, body)
        if not starts:
            chapters = [_chapter_from(paras, title, 1)]
        else:
            chapters = []
            if starts[0][0] > 0:
                opening = _chapter_from(paras[: starts[0][0]], "Front matter", 0)
                opening.kind = "front_matter"  # title page, copyright, dedication...: never chapter 1
                chapters.append(opening)
            bounds = [s for s, _ in starts] + [len(paras)]
            for n, (start, chapter_title) in enumerate(starts):
                chapters.append(_chapter_from(paras[start : bounds[n + 1]], chapter_title, n + 1))
            chapters = classify_chapters(chapters)

    sample = " ".join(p.text for p in paras[:200])
    return Book(
        title=title, author=author, language=detect_language(sample), source=str(path), chapters=chapters
    )


def render_cover(path: Path, out: Path, dpi: int = 110) -> Path | None:
    """First page as cover art for the M4B."""
    try:
        with pymupdf.open(path) as doc:
            if not doc.page_count:
                return None
            doc[0].get_pixmap(dpi=dpi).save(str(out))
        return out
    except Exception:  # noqa: BLE001 - a missing cover is never fatal
        return None
