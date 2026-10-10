from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fakes import FakeLLM, green_seal_handler
from test_text_and_analysis import make_project
from typer.testing import CliRunner

from audiobook_crafter import bench
from audiobook_crafter.analysis import analyze_chapter
from audiobook_crafter.cli import app, parse_chapters

GOLD = [
    {"chapter": 1, "text": "Is it from London?", "speaker": "Clara Hartwell"},
    {"chapter": 1, "text": "It is,", "speaker": "Mr. Hartwell"},
    {"chapter": 1, "text": "Then give it to me, Father, before I perish of curiosity!", "speaker": "Clara"},
    {"chapter": 1, "text": "Miss Hartwell?", "speaker": "Mrs. Pell"},
]


def write_gold(path: Path) -> Path:
    path.write_text("\n".join(json.dumps(g) for g in GOLD) + "\n", encoding="utf-8")
    return path


def test_score_attribution(tmp_path: Path) -> None:
    project, book = make_project(tmp_path)
    scripts = {1: analyze_chapter(project, book, 1, FakeLLM(green_seal_handler))}
    report = bench.score_attribution(
        scripts, project.load_characters(), bench.load_gold(write_gold(tmp_path / "g.jsonl"))
    )
    assert (report.total, report.matched, report.correct) == (4, 4, 4)  # aliases ("Clara") count as correct


def test_compare_llms_ranks_rules_below_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project, book = make_project(tmp_path)
    monkeypatch.setattr(
        bench, "create_llm", lambda config: None if config.model == "none" else FakeLLM(green_seal_handler)
    )
    gold = bench.load_gold(write_gold(tmp_path / "g.jsonl"))
    reports = {r.label: r for r in bench.compare_llms(project, book, [1], gold, ["none", "fake:model"])}
    assert reports["fake:model"].accuracy == 1.0
    assert reports["none"].accuracy < 1.0  # "Miss Hartwell?" needs coreference the rules don't have


def test_parse_chapters() -> None:
    assert parse_chapters(["3"]) == [3]
    assert parse_chapters(["1-3", "5", "2,7"]) == [1, 2, 3, 5, 7]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_cli_run_and_bench(tmp_path: Path, book_pdf: Path) -> None:
    runner = CliRunner()
    listing = runner.invoke(app, ["chapters", str(book_pdf)])
    assert listing.exit_code == 0 and "Chapter One" in listing.output

    project_dir = tmp_path / "proj"
    args = [
        "run",
        str(book_pdf),
        "-c",
        "1",
        "-p",
        str(project_dir),
        "--engine",
        "mock",
        "--llm",
        "none",
        "--preview",
        "8",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert list((project_dir / "output").glob("* (preview).mp3"))
    assert not list((project_dir / "output").glob("*.m4b"))  # previews stay out of the audiobook
    assert "without --preview" in result.output

    report = runner.invoke(app, ["bench", "tts", "-c", "1", "-p", str(project_dir)])
    assert report.exit_code == 0 and "real_time_factor" in report.output

    cast = runner.invoke(app, ["voices", "list", "-p", str(project_dir)])
    assert cast.exit_code == 0 and "narrator" in cast.output

    missing = runner.invoke(
        app, ["run", str(book_pdf), "-c", "9", "-p", str(project_dir), "--engine", "mock"]
    )
    assert missing.exit_code == 1 and "No chapter 9" in missing.output


def test_rules_baseline_on_pride_and_prejudice(tmp_path: Path) -> None:
    """Chapter 1 is almost all untagged Bennet back-and-forth: the rules must at least keep the voices
    alternating correctly (Mr. Bennet by name, his wife as the one unnamed woman)."""
    from audiobook_crafter.pipeline import ingest
    from audiobook_crafter.project import Project

    examples = Path(__file__).parent.parent / "examples"
    project = Project(tmp_path / "pp")
    book = ingest(project, examples / "pride_and_prejudice_ch1.txt")
    script = analyze_chapter(project, book, 1, None)
    gold = bench.load_gold(examples / "pride_and_prejudice_ch1.gold.jsonl")
    report = bench.score_attribution({1: script}, project.load_characters(), gold)
    assert report.matched == len(gold)
    assert report.per_speaker["Mr. Bennet"] == [14, 14]
    assert report.confusions == {("Mrs. Bennet", "unknown_female"): 18}
