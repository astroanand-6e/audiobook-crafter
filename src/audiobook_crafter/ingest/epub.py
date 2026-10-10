"""EPUB ingest: spine order, headings, paragraphs and scene breaks straight from the XHTML."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

from ..models import Block, Book, Chapter
from .common import classify_chapters, clean_text, detect_language, is_scene_break, looks_like_chapter_heading

_BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dd", "dt",
    "figcaption", "div", "section", "article", "td", "aside",
}  # fmt: skip
_BLOCK_LIST = sorted(_BLOCK_TAGS)  # bs4's find() wants a list, not a set
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_HEADING_CLASS = re.compile(r"chapter|title|heading|head\b|^h\d", re.IGNORECASE)
_BREAK_CLASS = re.compile(r"break|separator|ornament|scene|dinkus", re.IGNORECASE)
_NOTE_TYPES = re.compile(r"noteref|footnote|endnote|rearnote|pagebreak", re.IGNORECASE)


def _read(path: Path) -> epub.EpubBook:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return epub.read_epub(str(path), options={"ignore_ncx": True})


def _metadata(book: epub.EpubBook, field: str) -> str:
    values = book.get_metadata("DC", field)
    return clean_text(values[0][0]) if values else ""


def _toc_titles(book: epub.EpubBook) -> dict[str, str]:
    titles: dict[str, str] = {}

    def walk(items) -> None:
        for item in items:
            if isinstance(item, tuple):
                section, children = item
                if getattr(section, "href", None):
                    titles.setdefault(section.href.split("#")[0], clean_text(section.title))
                walk(children)
            elif getattr(item, "href", None):
                titles.setdefault(item.href.split("#")[0], clean_text(item.title))

    walk(book.toc)
    return titles


def _is_heading(el: Tag, text: str) -> bool:
    if el.name in _HEADINGS:
        return True
    classes = " ".join(el.get("class", []))
    return bool(classes and _HEADING_CLASS.search(classes)) and len(text.split()) <= 12


def _blocks(html: bytes) -> list[Block]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "head", "title"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"epub:type": _NOTE_TYPES}):
        tag.decompose()
    for tag in soup.find_all("sup"):
        if tag.find("a") or tag.get_text(strip=True).isdigit():
            tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" ")

    blocks: list[Block] = []
    body = soup.body or soup
    for el in body.find_all(True):
        if el.name == "hr" or (el.name in _BLOCK_TAGS and _BREAK_CLASS.search(" ".join(el.get("class", [])))):
            if blocks and blocks[-1].kind != "scene_break" and not el.get_text(strip=True).isalnum():
                blocks.append(Block(kind="scene_break"))
            continue
        if el.name not in _BLOCK_TAGS or el.find(_BLOCK_LIST):
            continue  # only leaf blocks; containers are visited through their children
        text = clean_text(el.get_text())
        if not text:
            continue
        if is_scene_break(text):
            if blocks and blocks[-1].kind != "scene_break":
                blocks.append(Block(kind="scene_break"))
        elif _is_heading(el, text):
            blocks.append(Block(kind="heading", text=text))
        else:
            blocks.append(Block(kind="paragraph", text=text))
    while blocks and blocks[-1].kind == "scene_break":
        blocks.pop()
    return blocks


def _split_on_headings(blocks: list[Block]) -> list[list[Block]]:
    """Some EPUBs put several chapters in one file; split where a chapter-like heading starts."""
    starts = [
        i
        for i, b in enumerate(blocks)
        if b.kind == "heading"
        and looks_like_chapter_heading(b.text)
        and not (i and blocks[i - 1].kind == "heading")
    ]
    if len(starts) < 2:
        return [blocks]
    if starts[0] != 0:
        starts.insert(0, 0)
    return [blocks[a:b] for a, b in zip(starts, [*starts[1:], len(blocks)])]


def load_epub(path: Path) -> Book:
    book = _read(path)
    toc_titles = _toc_titles(book)
    chapters: list[Chapter] = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        blocks = _blocks(item.get_content())
        if not any(b.text for b in blocks):
            continue
        parts = _split_on_headings(blocks)
        for n, part in enumerate(parts):
            headings = []
            for b in part:
                if b.kind != "heading":
                    break
                headings.append(b.text)
            title = (toc_titles.get(item.get_name()) if n == 0 else None) or " ".join(headings[:2])
            chapters.append(
                Chapter(index=len(chapters) + 1, title=title or f"Section {len(chapters) + 1}", blocks=part)
            )

    chapters = classify_chapters(chapters)
    language = (_metadata(book, "language") or "").split("-")[0].lower()
    sample = " ".join(b.text for c in chapters[:3] for b in c.blocks[:80])
    return Book(
        title=_metadata(book, "title") or Path(path).stem,
        author=_metadata(book, "creator"),
        language=language or detect_language(sample),
        source=str(path),
        chapters=chapters,
    )


def extract_cover(path: Path, out: Path) -> Path | None:
    try:
        book = _read(path)
        items = list(book.get_items_of_type(ebooklib.ITEM_COVER))
        if not items:
            cover_meta = book.get_metadata("OPF", "cover")
            cover_id = cover_meta[0][1].get("content") if cover_meta else None
            item = book.get_item_with_id(cover_id) if cover_id else None
            items = [item] if item is not None else []
        if not items:
            items = [
                i for i in book.get_items_of_type(ebooklib.ITEM_IMAGE) if "cover" in i.get_name().lower()
            ]
        if not items:
            return None
        out.write_bytes(items[0].get_content())
        return out
    except Exception:  # noqa: BLE001 - a missing cover is never fatal
        return None
