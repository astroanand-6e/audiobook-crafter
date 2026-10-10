"""Text cleanup and chapter-structure helpers shared by all input formats."""

from __future__ import annotations

import re
import unicodedata

from ..models import Block, Chapter

_CHAPTER_WORDS = (
    r"chapter|chap\.|part|book|volume|prologue|epilogue|interlude|introduction|preface|foreword|afterword|"
    r"appendix|kapitel|chapitre|cap[ií]tulo|capitolo|hoofdstuk|rozdzia[łl]|глава|часть|अध्याय|भाग"
)
_KEYWORD_HEADING = re.compile(rf"^(?:{_CHAPTER_WORDS})(?=$|[\s.:,—–-])", re.IGNORECASE)
_ROMAN = re.compile(r"^(?=[IVXLCDM])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?$")
_NUMBER = re.compile(r"^\d{1,3}\.?$")
_UNITS = "one|two|three|four|five|six|seven|eight|nine"
_TEENS = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
_TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_SPELLED = re.compile(rf"^(?:{_UNITS}|{_TEENS}|(?:{_TENS})(?:[- ](?:{_UNITS}))?)\.?$", re.IGNORECASE)
_CJK_HEADING = re.compile(r"^第[\d一二三四五六七八九十百千零〇]+[章回节]")
_SCENE_BREAK = re.compile(r"^\s*(?:[*•·⁂#~◆◇♦❖]\s*){1,6}$|^\s*(?:-\s*){3,}$")
_FRONT_MATTER = re.compile(
    r"^(?:contents|table of contents|copyright|dedication|title page|half title|also by|praise for|epigraph|"
    r"cover|imprint|colophon)\b",
    re.IGNORECASE,
)
_BACK_MATTER = re.compile(
    r"^(?:acknowledg(?:e)?ments?|about the author|about the publisher|index|bibliography|notes|endnotes|"
    r"glossary|other books|reading group guide|a note on the type)\b",
    re.IGNORECASE,
)

# Characters that confuse TTS front-ends or never belong in spoken text.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)


def clean_text(text: str) -> str:
    """Normalise ligatures, odd spaces and invisible characters; collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    text = text.replace(" ", " ").replace("\t", " ")
    return re.sub(r"[  -   　]+", " ", text).strip()


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines of one paragraph, repairing words hyphenated at line ends."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
        elif re.search(r"[A-Za-zÀ-ÿ][-\u2010\u2011\u00ad]$", out) and line[:1].islower():
            out = out[:-1] + line  # "extraordi-" + "nary"
        else:
            out = f"{out} {line}"
    return out


def is_scene_break(text: str) -> bool:
    return bool(_SCENE_BREAK.match(text.strip()))


def looks_like_chapter_heading(text: str) -> bool:
    text = text.strip()
    if not text or len(text.split()) > 12 or len(text) > 90:
        return False
    return bool(
        _KEYWORD_HEADING.match(text)
        or _ROMAN.match(text)
        or _NUMBER.match(text)
        or _SPELLED.match(text)
        or _CJK_HEADING.match(text)
    )


def display_title(text: str) -> str:
    """ "CHAPTER ONE" -> "Chapter One"; mixed-case titles are left alone."""
    text = clean_text(text)
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and not _ROMAN.match(text):
        words = text.split(" ")
        text = " ".join(w if _ROMAN.match(w.strip(".:,")) else w.capitalize() for w in words)
    return text


def title_from_headings(headings: list[str]) -> str:
    """["CHAPTER ONE", "The Letter"] -> "Chapter One: The Letter"."""
    parts = [display_title(h) for h in headings if h.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}: {' '.join(parts[1:])}"


def classify_chapters(chapters: list[Chapter]) -> list[Chapter]:
    """Tag front/back matter, fold heading-only stubs ("PART ONE") into the next chapter, renumber.

    Leading front matter becomes chapter 0 so that real chapters start at 1."""
    merged: list[Chapter] = []
    pending: list[Block] = []
    for chapter in chapters:
        chapter.blocks = pending + chapter.blocks
        pending = []
        if chapter.word_count < 40 and not _FRONT_MATTER.match(chapter.title) and chapter is not chapters[-1]:
            heading_blocks = [b for b in chapter.blocks if b.kind == "heading"]
            if heading_blocks and len(heading_blocks) == len([b for b in chapter.blocks if b.text.strip()]):
                pending = heading_blocks  # a "Part One" page: announce it at the top of the next chapter
                continue
        merged.append(chapter)

    def is_content(i: int, pattern: re.Pattern[str]) -> bool:
        # A titled chapter ("Chapter One", "Prologue") counts at any length; anything else needs real prose.
        chapter = merged[i]
        substantial = looks_like_chapter_heading(chapter.title) or chapter.word_count >= 150
        return chapter.kind == "chapter" and not pattern.match(chapter.title) and substantial

    first = next((i for i in range(len(merged)) if is_content(i, _FRONT_MATTER)), 0)
    last = max((i for i in range(len(merged)) if is_content(i, _BACK_MATTER)), default=len(merged) - 1)

    result: list[Chapter] = []
    if first:
        front = merged[:first]
        pages = (front[0].pages[0], front[-1].pages[1]) if front[0].pages and front[-1].pages else None
        blocks = [b for c in front for b in c.blocks]
        result.append(Chapter(index=0, title="Front matter", kind="front_matter", pages=pages, blocks=blocks))
    for i in range(first, len(merged)):
        chapter = merged[i]
        chapter.index = i - first + 1
        chapter.kind = "back_matter" if i > last else "chapter"
        result.append(chapter)
    return result


_STOPWORDS = {
    "en": {"the", "and", "of", "to", "was", "he", "she", "you", "that", "it", "said", "with", "his", "her"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "se", "del", "las", "por", "con", "una", "dijo"},
    "fr": {"le", "la", "de", "et", "les", "des", "est", "il", "elle", "que", "une", "dans", "pas", "dit"},
    "de": {"der", "die", "und", "das", "ist", "nicht", "sie", "er", "ich", "mit", "den", "ein", "sich"},
    "it": {"il", "di", "che", "e", "la", "non", "un", "per", "una", "sono", "del", "gli", "disse"},
    "pt": {"o", "de", "que", "e", "do", "da", "em", "um", "não", "uma", "os", "com", "disse", "ele"},
    "nl": {"de", "het", "een", "en", "van", "ik", "niet", "dat", "zijn", "op", "zei", "hij", "ze"},
}


def detect_language(text: str) -> str:
    """Cheap script + stopword guess; the user can always override with --language."""
    sample = text[:20000]
    scripts = {
        "hi": sum("ऀ" <= c <= "ॿ" for c in sample),
        "ar": sum("؀" <= c <= "ۿ" for c in sample),
        "he": sum("֐" <= c <= "׿" for c in sample),
        "ru": sum("Ѐ" <= c <= "ӿ" for c in sample),
        "el": sum("Ͱ" <= c <= "Ͽ" for c in sample),
        "ko": sum("가" <= c <= "힯" for c in sample),
        "ja": sum("぀" <= c <= "ヿ" for c in sample),
        "zh": sum("一" <= c <= "鿿" for c in sample),
    }
    letters = sum(c.isalpha() for c in sample) or 1
    best, count = max(scripts.items(), key=lambda kv: kv[1])
    if count / letters > 0.3:
        if best == "zh" and scripts["ja"] > 0.05 * count:
            return "ja"
        return best
    words = re.findall(r"[a-zà-ÿ']+", sample.lower())
    scores = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORDS.items()}
    lang, hits = max(scores.items(), key=lambda kv: kv[1])
    return lang if hits else "en"
