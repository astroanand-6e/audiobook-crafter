from __future__ import annotations

from pathlib import Path

from ebooklib import epub

from audiobook_crafter.ingest import load_document
from audiobook_crafter.ingest.common import display_title, join_lines, looks_like_chapter_heading
from audiobook_crafter.models import NARRATOR


def test_pdf_chapters_furniture_and_hyphenation(book_pdf: Path) -> None:
    book = load_document(book_pdf)
    assert book.title == "The Green Seal"
    assert [c.kind for c in book.chapters] == ["front_matter", "chapter", "chapter"]
    one, two = book.chapter(1), book.chapter(2)
    assert one.title == "Chapter One: The Letter"
    assert two.title == "Chapter Two: London"
    text = " ".join(b.text for b in one.blocks)
    assert "THE GREEN SEAL" not in text  # running head removed
    assert "extraordinarily old family" in text  # "extraordi-" + "narily" rejoined
    paragraphs = [b.text for b in one.blocks if b.kind == "paragraph"]
    assert (
        paragraphs[1]
        == "“Is it from London?” asked Clara, leaning over the banister. Her voice was bright with hope."
    )
    assert not any(p.strip().isdigit() for p in paragraphs)  # page numbers removed


def test_pdf_outline_titles(book_pdf_with_outline: Path) -> None:
    book = load_document(book_pdf_with_outline)
    assert [c.title for c in book.chapters if c.kind == "chapter"] == [
        "Chapter One: The Letter",
        "Chapter Two: London",
    ]


def test_pdf_page_range_becomes_one_chapter(book_pdf: Path) -> None:
    book = load_document(book_pdf, pages=(5, 6))
    assert len(book.chapters) == 1
    assert book.chapters[0].blocks[0].text == "CHAPTER TWO"


def test_epub(tmp_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("id1")
    book.set_title("Tiny Tales")
    book.set_language("en")
    book.add_author("Jo Writer")
    body = "".join(
        f"<p>“Line {i},” said Ana. It was a quiet evening in the small town.</p>" for i in range(12)
    )
    docs = []
    for n in (1, 2):
        item = epub.EpubHtml(title=f"Chapter {n}", file_name=f"ch{n}.xhtml", lang="en")
        item.content = f"<html><body><h1>Chapter {n}</h1>{body}<hr/><p>After the break.</p></body></html>"
        book.add_item(item)
        docs.append(item)
    book.toc = docs
    book.spine = docs
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "tiny.epub"
    epub.write_epub(str(path), book)

    loaded = load_document(path)
    assert loaded.title == "Tiny Tales" and loaded.author == "Jo Writer"
    chapters = [c for c in loaded.chapters if c.kind == "chapter"]
    assert [c.title for c in chapters] == ["Chapter 1", "Chapter 2"]
    kinds = [b.kind for b in chapters[0].blocks]
    assert kinds[0] == "heading" and "scene_break" in kinds


def test_text_with_gutenberg_header_and_wrapped_lines(tmp_path: Path) -> None:
    filler = "The fog came in from the sea and settled over the harbour like a grey wool blanket. " * 5
    path = tmp_path / "story.txt"
    path.write_text(
        "Some licence text\n*** START OF THE PROJECT GUTENBERG EBOOK STORY ***\n\n"
        "CHAPTER I\n\n"
        f"{filler}\nwrapped line continues here.\n\n“Hello,” said Tom.\n\n"
        "CHAPTER II\n\n"
        f"{filler}\n\n* * *\n\nLater that night.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK STORY ***\nmore licence\n",
        encoding="utf-8",
    )
    book = load_document(path)
    assert [c.title for c in book.chapters] == ["Chapter I", "Chapter II"]
    first = book.chapter(1)
    assert "wrapped line continues here." in first.blocks[1].text
    assert "licence" not in " ".join(b.text for c in book.chapters for b in c.blocks)
    assert "scene_break" in [b.kind for b in book.chapter(2).blocks]


def test_script_transcript(tmp_path: Path) -> None:
    path = tmp_path / "episode.txt"
    path.write_text(
        "# Episode 1\n"
        "NARRATOR: The studio lights flicker on.\n"
        "MAYA: Welcome back to the show.\n"
        "DEV (laughing): Thanks for having me.\n"
        "MAYA: Let's get into it.\n"
        "this line wraps from Maya's turn.\n",
        encoding="utf-8",
    )
    book = load_document(path)
    blocks = [b for b in book.chapters[0].blocks if b.kind == "paragraph"]
    assert [(b.speaker, b.direction) for b in blocks] == [
        (NARRATOR, None),
        ("MAYA", None),
        ("DEV", "laughing"),
        ("MAYA", None),
    ]
    assert blocks[-1].text.endswith("wraps from Maya's turn.")


def test_vtt_voices(tmp_path: Path) -> None:
    path = tmp_path / "talk.vtt"
    path.write_text(
        "WEBVTT\n\n00:00.000 --> 00:02.000\n<v Ana>Hi there.</v>\n\n"
        "00:02.000 --> 00:04.000\n<v Ana>Still me.</v>\n\n"
        "00:04.000 --> 00:06.000\n<v Ben>My turn.</v>\n",
        encoding="utf-8",
    )
    blocks = load_document(path).chapters[0].blocks
    assert [(b.speaker, b.text) for b in blocks] == [("Ana", "Hi there. Still me."), ("Ben", "My turn.")]


def test_heading_helpers() -> None:
    assert looks_like_chapter_heading("CHAPTER XII")
    assert looks_like_chapter_heading("Chapter Seven: The Storm")
    assert looks_like_chapter_heading("XIV")
    assert looks_like_chapter_heading("Twenty-One")
    assert not looks_like_chapter_heading("Mild.")
    assert not looks_like_chapter_heading("He had a chapter of his life he never spoke about.")
    assert display_title("CHAPTER ONE") == "Chapter One"
    assert display_title("CHAPTER IV") == "Chapter IV"
    assert join_lines(["an extraordi-", "nary day"]) == "an extraordinary day"


def test_pdf_mid_page_chapters_long_front_matter_and_footnotes(tmp_path: Path) -> None:
    import pymupdf
    from conftest import _text

    doc = pymupdf.open()
    front = doc.new_page()
    _text(front, (72, 90), "The Tricky Book", 24)
    y = 140
    for _ in range(14):  # ~250 words of title-page / copyright prose: must not become "chapter 1"
        _text(front, (72, y), "All rights reserved. No part of this book may be reproduced in any form", 10)
        y += 14
    _text(front, (72, 760), "Printed in the United Kingdom by Example Press.", 8)  # small print, not a note
    page = doc.new_page()
    _text(page, (230, 80), "CHAPTER ONE", 20)
    y = 120
    for n in range(9):
        _text(page, (90, y), f"Paragraph {n} of the first chapter, which goes on for a while here.", 11)
        y += 18
    _text(page, (230, y + 30), "CHAPTER TWO", 20)  # starts mid-page, after nine paragraphs
    y += 70
    for n in range(4):
        _text(page, (90, y), f"Second chapter paragraph {n}, with more words to read aloud.", 11)
        y += 18
    _text(page, (72, 740), "1 A footnote that should not be read aloud.", 8)
    for n in range(3):  # enough body text for chapter two to count as a chapter
        extra = doc.new_page()
        for k in range(12):
            _text(
                extra,
                (90, 80 + 18 * k),
                f"More of chapter two, page {n}, line {k}, and the story continues.",
                11,
            )
    doc.set_toc([[1, "Chapter One", 2], [1, "Chapter Two", 2]])
    path = tmp_path / "tricky.pdf"
    doc.save(path)

    book = load_document(path)
    assert [(c.index, c.kind) for c in book.chapters] == [(0, "front_matter"), (1, "chapter"), (2, "chapter")]
    one, two = book.chapter(1), book.chapter(2)
    assert one.blocks[0].text == "CHAPTER ONE" and "CHAPTER TWO" not in " ".join(b.text for b in one.blocks)
    assert two.blocks[0].text == "CHAPTER TWO"
    assert all("footnote" not in b.text for c in book.chapters for b in c.blocks)
    assert "Printed in the United Kingdom" in " ".join(b.text for b in book.chapters[0].blocks)


def test_join_lines_handles_unicode_hyphens() -> None:
    assert join_lines(["an extraordi‐", "nary day"]) == "an extraordinary day"
    assert join_lines(["an extraordi­", "nary day"]) == "an extraordinary day"
