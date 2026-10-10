from __future__ import annotations

from pathlib import Path

import pytest
from conftest import CHAPTER_ONE, CHAPTER_TWO
from fakes import FakeLLM, green_seal_handler

from audiobook_crafter.analysis import analyze_chapter, drop_speech_tags
from audiobook_crafter.analysis.characters import merge_character
from audiobook_crafter.llm.jsonutil import JSONParseError, parse_json_object
from audiobook_crafter.models import Block, Book, Chapter, CharacterMap
from audiobook_crafter.project import Project
from audiobook_crafter.text.normalize import normalize_for_tts, speak_heading
from audiobook_crafter.text.quotes import detect_quote_style, is_speech_tag, split_dialogue
from audiobook_crafter.text.sentences import chunk_text, split_sentences


def make_project(tmp_path: Path, mode: str = "audiobook") -> tuple[Project, Book]:
    project = Project(tmp_path / "proj")
    project.config.style.mode = mode  # type: ignore[assignment]
    project.save_config()
    blocks = [Block(kind="heading", text="CHAPTER ONE")] + [Block(text=t) for t in CHAPTER_ONE + CHAPTER_TWO]
    book = Book(title="The Green Seal", chapters=[Chapter(index=1, title="Chapter One", blocks=blocks)])
    project.save_book(book)
    return project, book


# ---------------------------------------------------------------- text


def test_split_dialogue_styles() -> None:
    spans = split_dialogue("“It is,” said Mr. Hartwell. “And it is yours.”")
    assert [(s.kind, s.text) for s in spans] == [
        ("dialogue", "“It is,”"),
        ("narration", "said Mr. Hartwell."),
        ("dialogue", "“And it is yours.”"),
    ]
    british = split_dialogue("‘I don’t know,’ she said.", "single")
    assert [s.kind for s in british] == ["dialogue", "narration"] and british[0].text == "‘I don’t know,’"
    assert split_dialogue("“And so it began", "double")[0].open_end
    assert [s.kind for s in split_dialogue("—Hola —dijo Juan—. ¿Cómo estás?", "dash")] == [
        "dialogue",
        "narration",
        "dialogue",
    ]
    assert detect_quote_style(["‘Yes,’ he said.", "‘No,’ she said."]) == "single"
    assert detect_quote_style(["«Bonjour», dit-il."]) == "guillemets"


def test_speech_tags() -> None:
    assert is_speech_tag("said Mr. Darcy,")
    assert is_speech_tag("she whispered softly.")
    assert not is_speech_tag("she said, turning away from the window")
    assert not is_speech_tag("He laughed and handed it over.")


def test_sentences_and_chunks() -> None:
    assert split_sentences("Mr. Hartwell left at 10 a.m. sharp. “Hello.” She waved… and then? Yes!") == [
        "Mr. Hartwell left at 10 a.m. sharp.",
        "“Hello.”",
        "She waved… and then?",
        "Yes!",
    ]
    assert split_sentences("राम घर गया। सीता आई।") == ["राम घर गया।", "सीता आई।"]
    chunks = chunk_text(
        "One short. Two short. " + "A much longer sentence, full of clauses; it goes on and on. " * 3, 80
    )
    assert all(len(c) <= 80 for c in chunks)
    assert chunks[0] == "One short. Two short."


def test_normalize() -> None:
    assert normalize_for_tts("“It cost $5.50 in 1923,” said Mr. Hart.") == (
        "It cost five dollars and fifty cents in nineteen twenty-three, said Mister Hart."
    )
    assert normalize_for_tts("GET OUT OF MY HOUSE NOW!") == "Get out of my house now!"
    assert normalize_for_tts("At 10:30 on the 3rd, 45% of 1,200 men.") == (
        "At ten thirty on the third, forty-five percent of one thousand two hundred men."
    )
    assert speak_heading("CHAPTER IV") == "Chapter Four."
    assert speak_heading("12") == "Chapter Twelve."
    assert speak_heading("Chapter One: The Letter") == "Chapter One. The Letter."


def test_json_repair() -> None:
    assert parse_json_object('<think>hmm</think>```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert parse_json_object('Sure! {"lines": [{"id": 1, "speaker": "x"}, {"id": 2, "spea') == {
        "lines": [{"id": 1, "speaker": "x"}]
    }
    with pytest.raises(JSONParseError):
        parse_json_object("no json here")


# ---------------------------------------------------------------- characters


def test_merge_keeps_ids_and_separates_similar_names() -> None:
    characters = CharacterMap()
    lizzy = merge_character(
        characters, {"id": "", "name": "Elizabeth Bennet", "aliases": ["Lizzy"], "gender": "female"}
    )
    assert lizzy and lizzy.id == "elizabeth_bennet"
    again = merge_character(
        characters, {"id": "", "name": "Lizzy", "aliases": [], "gender": "unknown", "age": "teen"}
    )
    assert again is lizzy and lizzy.age == "teen"
    mrs = merge_character(
        characters, {"id": "", "name": "Mrs. Bennet", "aliases": ["Bennet"], "gender": "female"}
    )
    mr = merge_character(
        characters, {"id": "", "name": "Mr. Bennet", "aliases": ["Bennet"], "gender": "male"}
    )
    assert mrs is not mr and mr.id == "mr_bennet"
    assert merge_character(characters, {"id": "", "name": "Narrator", "aliases": []}) is None


# ---------------------------------------------------------------- full analysis


def test_analyze_with_llm(tmp_path: Path) -> None:
    project, book = make_project(tmp_path)
    llm = FakeLLM(green_seal_handler)
    script = analyze_chapter(project, book, 1, llm)
    dialogue = [s for s in script.segments if s.kind == "dialogue"]
    assert [s.speaker for s in dialogue] == [
        "clara_hartwell", "mr_hartwell", "mr_hartwell", "clara_hartwell",
        "mrs_pell", "mrs_pell", "clara_hartwell", "mrs_pell",
    ]  # fmt: skip
    assert dialogue[3].emotion == "excited"
    characters = project.load_characters()
    assert {c.id: c.lines for c in characters.characters} == {
        "clara_hartwell": 3,
        "mr_hartwell": 2,
        "mrs_pell": 3,
    }
    assert project.load_script(1) is not None
    assert script.segments[0].kind == "heading"


def test_analyze_without_llm_uses_rules(tmp_path: Path) -> None:
    project, book = make_project(tmp_path)
    script = analyze_chapter(project, book, 1, None)
    speakers = [s.speaker for s in script.segments if s.kind == "dialogue"]
    assert speakers[:4] == ["clara", "mr_hartwell", "mr_hartwell", "clara"]


def test_llm_failure_falls_back_to_rules(tmp_path: Path) -> None:
    project, book = make_project(tmp_path)

    def broken(system, prompt, schema):
        if "dialogue editor" in system:
            return "I can't do JSON today"
        return green_seal_handler(system, prompt, schema)

    script = analyze_chapter(project, book, 1, FakeLLM(broken))
    assert [s.speaker for s in script.segments if s.kind == "dialogue"][0] == "clara_hartwell"


def test_drama_mode_drops_speech_tags(tmp_path: Path) -> None:
    project, book = make_project(tmp_path, mode="drama")
    script = analyze_chapter(project, book, 1, FakeLLM(green_seal_handler))
    texts = [s.text for s in script.segments]
    assert "Clara asked." not in texts
    assert any(t.startswith("said Mr. Hartwell. He turned") for t in texts)  # not a bare tag: kept
    assert drop_speech_tags([]) == []
