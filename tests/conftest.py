from __future__ import annotations

import textwrap
from pathlib import Path

import pymupdf
import pytest

CHAPTER_ONE = [
    "The rain had not stopped for three days when the letter finally arrived at the Hartwell house.",
    "“Is it from London?” asked Clara, leaning over the banister. Her voice was bright with hope.",
    "“It is,” said Mr. Hartwell. He turned the envelope over twice. “And it is addressed to you.”",
    "Clara ran down the stairs. “Then give it to me, Father, before I perish of curiosity!”",
    "He laughed and handed it over. The seal was deep green and bore the crest of an extraordinarily "
    "old family that neither of them had ever heard of before that grey morning.",
]
CHAPTER_TWO = [
    "London was louder than Clara had imagined, and far less kind.",
    "“Miss Hartwell?” A tall woman in black stepped from the crowd. “I am Mrs. Pell. Come quickly.”",
    "“Where are we going?” Clara asked.",
    "“Somewhere dry,” said Mrs. Pell, and that was all she would say for the rest of the journey.",
]


FONT = pymupdf.Font("cjk")  # built into PyMuPDF and covers curly quotes and dashes


def _text(page: pymupdf.Page, pos: tuple[float, float], text: str, size: float) -> None:
    writer = pymupdf.TextWriter(page.rect)
    writer.append(pos, text, font=FONT, fontsize=size)
    writer.write_text(page)


def _write_paragraphs(page: pymupdf.Page, y: float, paragraphs: list[str], width: int = 62) -> float:
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, width=width, break_long_words=False)
        for i, line in enumerate(lines):
            if line.endswith(" extraordinarily") and i + 1 < len(lines):
                # Exercise de-hyphenation: "extraordi-" / "narily".
                lines[i] = line[: -len("narily")] + "-"
                lines[i + 1] = "narily " + lines[i + 1]
        for i, line in enumerate(lines):
            _text(page, (90 if i == 0 else 72, y), line, 11)  # first-line indent
            y += 14
    return y


def _furniture(page: pymupdf.Page, number: int) -> None:
    _text(page, (230, 40), "THE GREEN SEAL", 8)
    _text(page, (300, 780), str(number), 9)


def make_book_pdf(path: Path, with_outline: bool = False) -> Path:
    doc = pymupdf.open()
    title = doc.new_page()
    _text(title, (150, 300), "The Green Seal", 28)
    _text(title, (200, 340), "A. N. Author", 14)

    contents = doc.new_page()
    _text(contents, (72, 100), "Contents", 18)
    for i, (name, pg) in enumerate(
        [
            ("Chapter One", 3),
            ("Chapter Two", 5),
            ("Chapter Three", 7),
            ("Chapter Four", 9),
            ("Chapter Five", 11),
        ]
    ):
        _text(contents, (72, 140 + 18 * i), f"{name} .......... {pg}", 11)

    page_no = 1
    for heading, subtitle, paragraphs in [
        ("CHAPTER ONE", "The Letter", CHAPTER_ONE),
        ("CHAPTER TWO", "London", CHAPTER_TWO),
    ]:
        page = doc.new_page()
        _text(page, (230, 160), heading, 20)
        _text(page, (250, 190), subtitle, 14)
        _write_paragraphs(page, 240, paragraphs * 3)
        _furniture(page, page_no)
        page_no += 1
        cont = doc.new_page()  # a continuation page so running heads repeat
        _write_paragraphs(cont, 80, paragraphs[1:3])
        _furniture(cont, page_no)
        page_no += 1
    if with_outline:
        doc.set_toc([[1, "Chapter One: The Letter", 3], [1, "Chapter Two: London", 5]])
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def book_pdf(tmp_path: Path) -> Path:
    return make_book_pdf(tmp_path / "green_seal.pdf")


@pytest.fixture
def book_pdf_with_outline(tmp_path: Path) -> Path:
    return make_book_pdf(tmp_path / "green_seal_outline.pdf", with_outline=True)
