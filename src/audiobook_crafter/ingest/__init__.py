"""Turn an input document into a Book: chapters made of headings, paragraphs and scene breaks."""

from __future__ import annotations

from pathlib import Path

from ..models import Book

SUPPORTED_SUFFIXES = (".pdf", ".epub", ".txt", ".text", ".md", ".markdown", ".srt", ".vtt")


def load_document(path: Path, pages: tuple[int, int] | None = None) -> Book:
    """Load a PDF, EPUB, text/Markdown book, or speaker-labelled script/transcript.

    `pages` (PDF only, 1-based inclusive) skips chapter detection and returns those pages as chapter 1."""
    path = Path(path)
    suffix = path.suffix.lower()
    if pages is not None and suffix != ".pdf":
        raise ValueError("--pages only applies to PDF input")
    if suffix == ".pdf":
        from .pdf import load_pdf

        return load_pdf(path, pages)
    if suffix == ".epub":
        from .epub import load_epub

        return load_epub(path)
    if suffix in SUPPORTED_SUFFIXES:
        from .text import load_text

        return load_text(path)
    raise ValueError(f"Unsupported input '{suffix}'. Use one of: {', '.join(SUPPORTED_SUFFIXES)}")


def extract_cover(source: Path, out: Path) -> Path | None:
    suffix = Path(source).suffix.lower()
    if suffix == ".pdf":
        from .pdf import render_cover

        return render_cover(Path(source), out)
    if suffix == ".epub":
        from .epub import extract_cover as epub_cover

        return epub_cover(Path(source), out)
    return None
