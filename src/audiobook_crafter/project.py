"""A project is one book's working folder. Every stage reads and writes plain files here,
so any step can be inspected, hand-edited and re-run without redoing the others.

    project.yaml        settings
    book.json           ingested chapters and paragraphs
    characters.json     character map built by the LLM            (editable)
    scripts/ch003.json  who says what, with emotion, per chapter   (editable)
    cast.yaml           which voice plays whom                     (editable)
    voices/             reference clips: yours, or designed by a voice-design model
    cache/tts/          every generated clip, keyed by a hash of its inputs
    render/             per-chapter voice tracks and clip manifests
    output/             finished MP3 / M4B files
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .config import ProjectConfig
from .models import Book, Cast, ChapterScript, CharacterMap, slugify

CONFIG_FILE = "project.yaml"


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file so an interrupted run never leaves a half-written JSON behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def primary_language(code: str | None) -> str:
    """ "en-GB" / "pt_BR" -> "en" / "pt": engines, number spelling and ASR all want the bare language."""
    return (code or "").strip().lower().replace("_", "-").split("-")[0]


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class Project:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        config_path = self.root / CONFIG_FILE
        self.config = ProjectConfig.load(config_path) if config_path.exists() else ProjectConfig()

    @staticmethod
    def default_root(source: Path) -> Path:
        return Path.cwd() / "audiobooks" / slugify(Path(source).stem)

    @property
    def exists(self) -> bool:
        return (self.root / CONFIG_FILE).exists()

    @property
    def language(self) -> str:
        return primary_language(self.config.language) or "en"

    # ----------------------------------------------------------------- paths

    @property
    def book_path(self) -> Path:
        return self.root / "book.json"

    @property
    def characters_path(self) -> Path:
        return self.root / "characters.json"

    @property
    def cast_path(self) -> Path:
        return self.root / "cast.yaml"

    @property
    def voices_dir(self) -> Path:
        return self.root / "voices"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def render_dir(self) -> Path:
        return self.root / "render"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    def script_path(self, chapter: int) -> Path:
        return self.root / "scripts" / f"ch{chapter:03d}.json"

    def resolve(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root / path

    def relative(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    # ----------------------------------------------------------------- load / save

    def save_config(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.config.save(self.root / CONFIG_FILE)

    def load_book(self) -> Book:
        if not self.book_path.exists():
            raise FileNotFoundError(f"{self.book_path} not found; run `abcraft ingest` first")
        return Book.model_validate_json(self.book_path.read_text(encoding="utf-8"))

    def save_book(self, book: Book) -> None:
        write_json(self.book_path, book.model_dump(mode="json"))

    def load_characters(self) -> CharacterMap:
        if not self.characters_path.exists():
            return CharacterMap()
        return CharacterMap.model_validate_json(self.characters_path.read_text(encoding="utf-8"))

    def save_characters(self, characters: CharacterMap) -> None:
        write_json(self.characters_path, characters.model_dump(mode="json"))

    def load_script(self, chapter: int) -> ChapterScript | None:
        path = self.script_path(chapter)
        if not path.exists():
            return None
        return ChapterScript.model_validate_json(path.read_text(encoding="utf-8"))

    def save_script(self, script: ChapterScript) -> None:
        write_json(self.script_path(script.chapter), script.model_dump(mode="json"))

    def load_cast(self) -> Cast:
        if not self.cast_path.exists():
            return Cast()
        data = yaml.safe_load(self.cast_path.read_text(encoding="utf-8")) or {}
        return Cast.model_validate(data)

    def save_cast(self, cast: Cast) -> None:
        data = cast.model_dump(mode="json", exclude_defaults=True)
        header = (
            "# Voice casting. Edit freely: changes only re-generate the lines they affect.\n"
            "#   reference: clip to clone (voices/...)   reference_text: what the clip says\n"
            "#   preset: engine speaker (Qwen3 CustomVoice: Ryan, Aiden, Vivian, Serena, ...)\n"
            "#   design: description for the voice-design model   same_as: reuse another voice\n"
            "#   engine / params / seed: per-voice overrides\n"
        )
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
        atomic_write_text(self.cast_path, header + body)
