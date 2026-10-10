"""`abcraft` command line."""

from __future__ import annotations

import logging
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Annotated

import soundfile as sf
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import bench, pipeline
from .assemble import assemble_chapter, build_audiobook
from .audio.dsp import clean_clip
from .casting.design import design_voices
from .casting.voices import resolve_voice
from .ingest import load_document
from .llm import parse_spec
from .project import Project
from .tts import ENGINES, EnginePool, engine_available
from .tts.base import Line

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="PDF / EPUB / script -> full-cast audiobook."
)
voices_app = typer.Typer(no_args_is_help=True, help="Design, list and audition voices.")
app.add_typer(voices_app, name="voices")
bench_app = typer.Typer(no_args_is_help=True, help="Score LLM casting and TTS runs (testbench).")
app.add_typer(bench_app, name="bench")
console = Console()


# ----------------------------------------------------------------- helpers


class RichReporter:
    def __init__(self) -> None:
        self.bar = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[note]}"),
            console=console,
            transient=True,
        )
        self.tasks: dict[str, int] = {}

    def __enter__(self) -> RichReporter:
        self.bar.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.bar.stop()

    def stage(self, name: str, detail: str = "") -> None:
        for task in self.tasks.values():
            self.bar.remove_task(task)
        self.tasks.clear()
        console.print(f"[bold cyan]▸ {name}[/] {detail}")

    def progress(self, task: str, done: int, total: int, note: str = "") -> None:
        if task not in self.tasks:
            self.tasks[task] = self.bar.add_task(task, total=total, note="")
        self.bar.update(self.tasks[task], completed=done, total=total, note=note[:50])

    def warn(self, message: str) -> None:
        console.print(f"[yellow]⚠ {message}")


def parse_chapters(values: list[str]) -> list[int]:
    chapters: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if re.fullmatch(r"\d+-\d+", part):
                start, end = map(int, part.split("-"))
                chapters.extend(range(start, end + 1))
            elif part.isdigit():
                chapters.append(int(part))
            elif part:
                raise typer.BadParameter(f"'{part}' is not a chapter number or range")
    return sorted(dict.fromkeys(chapters))


def parse_pages(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"(\d+)-(\d+)", value.strip())
    if not match:
        raise typer.BadParameter("use FIRST-LAST, e.g. 12-30")
    return int(match[1]), int(match[2])


def open_project(project_dir: Path | None, book: Path | None = None) -> Project:
    if project_dir is None:
        if book is not None:
            project_dir = Project.default_root(book)
        elif (Path.cwd() / "project.yaml").exists():
            project_dir = Path.cwd()
        else:
            raise typer.BadParameter("pass --project (the folder `abcraft run` created)")
    return Project(project_dir)


def apply_options(
    project: Project,
    *,
    llm: str | None = None,
    engine: str | None = None,
    language: str | None = None,
    mode: str | None = None,
    asr: bool | None = None,
    device: str | None = None,
) -> None:
    config = project.config
    if llm:
        config.llm.model = llm
    if engine:
        if engine not in ENGINES:
            raise typer.BadParameter(f"unknown engine '{engine}'; see `abcraft engines`")
        config.tts.engine = engine
    if language:
        config.language = language.lower()
    if mode:
        if mode not in ("audiobook", "drama"):
            raise typer.BadParameter("--mode is audiobook or drama")
        config.style.mode = mode  # type: ignore[assignment]
    if asr is not None:
        config.qa.asr = asr
    if device:
        config.tts.device = device
    project.save_config()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    logging.getLogger("audiobook_crafter").setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def fail(message: str) -> None:
    console.print(f"[bold red]✗[/] {message}")
    raise typer.Exit(1)


ProjectOpt = Annotated[
    Path | None, typer.Option("--project", "-p", help="Project folder (default: ./audiobooks/<book>)")
]
ChapterOpt = Annotated[
    list[str], typer.Option("--chapter", "-c", help="Chapter number, range or list: 3, 3-5, 1,4")
]
LLMOpt = Annotated[
    str | None,
    typer.Option("--llm", help="provider:model, e.g. ollama:qwen3.5:9b, anthropic:claude-opus-5-5, none"),
]
EngineOpt = Annotated[str | None, typer.Option("--engine", "-e", help="TTS engine (see `abcraft engines`)")]


# ----------------------------------------------------------------- commands


@app.command()
def chapters(
    book: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="PDF, EPUB, TXT, MD, SRT or VTT")],
    pages: Annotated[str | None, typer.Option(help="PDF pages FIRST-LAST to treat as one chapter")] = None,
) -> None:
    """List the chapters found in a book, to pick one for `run --chapter`."""
    parsed = load_document(book, parse_pages(pages))
    table = Table(
        title=f"{parsed.title}" + (f" — {parsed.author}" if parsed.author else ""), show_lines=False
    )
    for column in ("#", "kind", "title", "words", "pages", "starts with"):
        table.add_column(column, overflow="fold")
    for chapter in parsed.chapters:
        pages_text = f"{chapter.pages[0]}-{chapter.pages[1]}" if chapter.pages else ""
        style = "dim" if chapter.kind != "chapter" else ""
        table.add_row(str(chapter.index), chapter.kind, chapter.title, str(chapter.word_count), pages_text,
                      chapter.preview(10), style=style)  # fmt: skip
    console.print(table)
    console.print(f"Language: [bold]{parsed.language}[/]. Next: [bold]abcraft run {book} --chapter N[/]")


@app.command()
def run(
    book: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="PDF, EPUB, TXT, MD, SRT or VTT")],
    chapter: ChapterOpt = [],  # noqa: B006 - typer needs a literal default
    project_dir: ProjectOpt = None,
    llm: LLMOpt = None,
    engine: EngineOpt = None,
    language: Annotated[str | None, typer.Option(help="Book language (ISO code); default: detected")] = None,
    mode: Annotated[
        str | None, typer.Option(help="audiobook (every word) or drama (drops 'he said' tags)")
    ] = None,
    pages: Annotated[
        str | None, typer.Option(help="PDF pages FIRST-LAST as the chapter, if detection fails")
    ] = None,
    preview: Annotated[int | None, typer.Option(help="Only the first N lines: a quick listen")] = None,
    reanalyze: Annotated[bool, typer.Option(help="Redo speaker attribution even if a script exists")] = False,
    scan_book: Annotated[
        bool, typer.Option(help="Build the character map from the whole book first")
    ] = False,
    asr: Annotated[bool | None, typer.Option("--asr/--no-asr", help="Check every clip with Whisper")] = None,
    device: Annotated[str | None, typer.Option(help="cuda, cpu, mps or auto")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Convert chapters end to end: text -> characters -> script -> voices -> mastered MP3 + M4B."""
    setup_logging(verbose)
    page_range = parse_pages(pages)
    selected = parse_chapters(chapter) or ([1] if page_range else [])
    project = open_project(project_dir, book)
    if not selected:
        console.print("Pick chapters with [bold]--chapter[/]. Here is what the book contains:")
        chapters(book, pages)
        raise typer.Exit(2)
    apply_options(project, llm=llm, engine=engine, language=language, mode=mode, asr=asr, device=device)
    console.print(f"[bold]Project[/] {project.root}")
    try:
        with RichReporter() as reporter:
            result = pipeline.run(
                project, book, selected, reporter, pages=page_range, reanalyze=reanalyze, preview=preview,
                scan_book=scan_book,
            )  # fmt: skip
    except KeyError as err:
        fail(f"{err.args[0]}. See `abcraft chapters {book}`.")
    except Exception as err:  # noqa: BLE001 - show a clean message, details with -v
        if verbose:
            raise
        fail(f"{type(err).__name__}: {err}")

    for change in result.cast_changes:
        console.print(f"  [green]cast[/] {change}")
    for audio in result.audio:
        minutes = audio.seconds / 60
        console.print(
            f"[bold green]✓[/] Chapter {audio.index}: {audio.title} — {minutes:.1f} min → {audio.mp3}"
        )
    if result.m4b:
        console.print(f"[bold green]✓[/] Audiobook so far: {result.m4b}")
    if preview:
        console.print(
            "This was a preview; run again without --preview for the full chapter (clips are reused)."
        )
    if result.issues:
        console.print(f"[yellow]⚠ {len(result.issues)} clips still failed QA after retries:")
        for clip in result.issues[:10]:
            console.print(f"   {clip.segment} [{clip.voice}] {clip.issue}: {clip.text[:70]}")
    console.print(
        "Tweak and re-run (only changed lines are regenerated):\n"
        f"  speakers/emotions: {project.root / 'scripts'}\n  voices: {project.cast_path}"
    )


@app.command()
def analyze(
    chapter: ChapterOpt,
    book: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Book (first run only)")
    ] = None,
    project_dir: ProjectOpt = None,
    llm: LLMOpt = None,
    reanalyze: Annotated[bool, typer.Option(help="Redo chapters that already have a script")] = True,
    scan_book: bool = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Character map + speaker/emotion script only (no audio)."""
    setup_logging(verbose)
    project = open_project(project_dir, book)
    apply_options(project, llm=llm)
    source = book or Path(project.config.source)
    loaded = pipeline.ingest(project, source)
    with RichReporter() as reporter:
        scripts = pipeline.analyze(project, loaded, parse_chapters(chapter), reporter, reanalyze=reanalyze,
                                   scan_book=scan_book)  # fmt: skip
    characters = project.load_characters()
    table = Table(title="Characters")
    for column in ("id", "name", "gender", "age", "lines", "voice"):
        table.add_column(column, overflow="fold")
    for c in sorted(characters.characters, key=lambda c: -c.lines):
        table.add_row(c.id, c.name, c.gender, c.age, str(c.lines), c.voice)
    console.print(table)
    for script in scripts:
        counts = script.speaker_counts()
        console.print(
            f"Chapter {script.chapter}: {len(script.segments)} segments, speakers {dict(counts.most_common(8))}"
        )
        console.print(f"  edit: {project.script_path(script.chapter)}")


@app.command(name="cast")
def cast_command(project_dir: ProjectOpt = None) -> None:
    """Give every speaking character a voice (keeps existing choices) and show the cast."""
    project = open_project(project_dir)
    for change in pipeline.cast(project):
        console.print(f"  [green]new[/] {change}")
    show_cast(project)


def show_cast(project: Project) -> None:
    cast = project.load_cast()
    characters = {c.id: c for c in project.load_characters().characters}
    table = Table(title=f"Cast ({project.cast_path})")
    for column in ("voice", "lines", "source"):
        table.add_column(column, overflow="fold")
    for voice_id, profile in cast.voices.items():
        if profile.same_as:
            source = f"same as {profile.same_as}"
        elif profile.reference:
            source = profile.reference
        elif profile.preset:
            source = f"preset {profile.preset}"
        else:
            source = f"[yellow]to design:[/] {profile.design or '?'}"
        lines = characters[voice_id].lines if voice_id in characters else ""
        table.add_row(voice_id, str(lines), source)
    console.print(table)


@voices_app.command("design")
def voices_design(
    project_dir: ProjectOpt = None,
    only: Annotated[list[str], typer.Option(help="Only these voice ids")] = [],  # noqa: B006
    force: Annotated[bool, typer.Option(help="Redesign voices that already have a clip")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Create reference clips for designed voices (Qwen3-TTS VoiceDesign)."""
    setup_logging(verbose)
    project = open_project(project_dir)
    book = project.load_book()
    cast = project.load_cast()
    pool = EnginePool(project.config.tts)
    name = "mock" if project.config.tts.engine == "mock" else project.config.tts.design_engine
    try:
        with RichReporter() as reporter:
            done = design_voices(
                project, book, cast, pool.get(name), only=set(only) or None, force=force,
                on_progress=lambda n, total, v: reporter.progress("voices", n, total, v),
            )  # fmt: skip
    finally:
        pool.close()
    console.print(f"Designed {len(done)} voices: {', '.join(done) or '-'}")


@voices_app.command("list")
def voices_list(project_dir: ProjectOpt = None) -> None:
    """Show who speaks with which voice."""
    show_cast(open_project(project_dir))


@voices_app.command("audition")
def voices_audition(
    project_dir: ProjectOpt = None,
    text: Annotated[str | None, typer.Option(help="Line every voice should say")] = None,
) -> None:
    """Render one line per voice into output/auditions/ to compare the cast quickly."""
    project = open_project(project_dir)
    cast = project.load_cast()
    out_dir = project.output_dir / "auditions"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = text or "I never thought I would see this place again. And yet, here we are."
    pool = EnginePool(project.config.tts)
    try:
        for voice_id, profile in cast.voices.items():
            if profile.same_as:
                continue
            engine_name = profile.engine or project.config.tts.engine
            voice = resolve_voice(project, voice_id, profile, engine_name, project.config.tts.seed)
            try:
                audio, sr = pool.get(engine_name).synthesize(
                    Line(text=sample, language=project.language), voice, voice.seed
                )
            except Exception as err:  # noqa: BLE001
                console.print(f"  [yellow]{voice_id}: {err}")
                continue
            path = out_dir / f"{voice_id}.wav"
            sf.write(str(path), clean_clip(audio, sr), sr)
            console.print(f"  {voice_id} → {path}")
    finally:
        pool.close()


@app.command()
def synth(
    chapter: ChapterOpt,
    project_dir: ProjectOpt = None,
    engine: EngineOpt = None,
    preview: Annotated[int | None, typer.Option(help="Only the first N lines")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Speak chapters whose script exists (no mastering)."""
    setup_logging(verbose)
    project = open_project(project_dir)
    apply_options(project, engine=engine)
    book = project.load_book()
    scripts = [s for s in (project.load_script(i) for i in parse_chapters(chapter)) if s]
    pool = EnginePool(project.config.tts)
    try:
        with RichReporter() as reporter:
            pipeline.design(project, book, reporter, pool)
            pipeline.synthesize(project, scripts, reporter, pool, preview)
    finally:
        pool.close()


@app.command()
def assemble(chapter: ChapterOpt, project_dir: ProjectOpt = None) -> None:
    """Master synthesized chapters and rebuild the M4B."""
    project = open_project(project_dir)
    book = project.load_book()
    for index in parse_chapters(chapter):
        audio = assemble_chapter(project, book, index)
        console.print(f"✓ {audio.mp3} ({audio.seconds / 60:.1f} min)")
    m4b = build_audiobook(project, book)
    if m4b:
        console.print(f"✓ {m4b}")


@app.command()
def engines() -> None:
    """TTS engines and whether they're installed here."""
    table = Table()
    for column in ("engine", "installed", "what it does"):
        table.add_column(column, overflow="fold")
    for name, info in ENGINES.items():
        table.add_row(name, "[green]yes" if engine_available(name) else "[dim]no", info.summary)
    console.print(table)


@app.command()
def doctor(project_dir: ProjectOpt = None) -> None:
    """Check ffmpeg, PyTorch/CUDA (incl. RTX 50-series support), TTS packages and the LLM endpoint."""
    ok = True

    def line(good: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        console.print(f"{'[green]✓' if good else '[red]✗'}[/] {label} [dim]{detail}")

    line(sys.version_info >= (3, 10), "Python", platform.python_version())
    for tool in ("ffmpeg", "ffprobe"):
        line(shutil.which(tool) is not None, tool, shutil.which(tool) or "not on PATH")
    try:
        import torch

        cuda = torch.cuda.is_available()
        line(True, "PyTorch", f"{torch.__version__} (CUDA build {torch.version.cuda or 'none'})")
        if cuda:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
            supported = (
                arch in torch.cuda.get_arch_list() or f"compute_{major}{minor}" in torch.cuda.get_arch_list()
            )
            memory = torch.cuda.get_device_properties(0).total_memory / 2**30
            line(supported, "GPU", f"{name}, {memory:.0f} GB, {arch}")
            if not supported:
                console.print(
                    f"   This PyTorch has no {arch} kernels. RTX 50-series needs torch>=2.7 built for CUDA 12.8: "
                    "`uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`"
                )
        else:
            line(False, "GPU", "CUDA not available: TTS will run on CPU (slow)")
    except ImportError:
        line(False, "PyTorch", "not installed (needed by every real TTS engine)")
    for name in ("qwen3-tts", "chatterbox-multilingual"):
        line(engine_available(name), name, ENGINES[name].summary)
    import importlib.util

    if importlib.util.find_spec("flash_attn") is None:
        console.print("[dim]• flash-attn not installed (optional; Qwen3-TTS uses less VRAM with it)")

    project = None
    if project_dir or (Path.cwd() / "project.yaml").exists():
        project = open_project(project_dir)
    config = project.config.llm if project else None
    spec = config.model if config else "ollama:qwen3.5:9b"
    provider, model = parse_spec(spec)
    if provider == "ollama":
        import httpx

        base = (config.base_url if config else None) or "http://localhost:11434"
        try:
            tags = httpx.get(f"{base.rstrip('/')}/api/tags", timeout=3).json()
            names = {m.get("name") for m in tags.get("models", [])}
            pulled = model in names or f"{model}:latest" in names
            line(pulled, f"Ollama model {model}", "ready" if pulled else f"run `ollama pull {model}`")
        except Exception:  # noqa: BLE001
            line(False, "Ollama", f"not reachable at {base} (start it with `ollama serve`)")
    else:
        console.print(f"• LLM: {spec}")
    raise typer.Exit(0 if ok else 1)


def _print_reports(reports: list[bench.AttributionReport]) -> None:
    table = Table(title="Speaker attribution")
    for column in ("run", "accuracy", "quotes found", "worst confusions"):
        table.add_column(column, overflow="fold")
    for report in sorted(reports, key=lambda r: -r.accuracy):
        confusions = ", ".join(f"{g}→{p} ×{n}" for (g, p), n in report.confusions.most_common(3))
        table.add_row(
            report.label, f"{report.accuracy:.1%}", f"{report.matched}/{report.total}", confusions or "-"
        )
    console.print(table)


@bench_app.command("attribution")
def bench_attribution(
    gold: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="JSONL: chapter, text, speaker")],
    project_dir: ProjectOpt = None,
) -> None:
    """Score this project's scripts against hand-labelled quotes."""
    project = open_project(project_dir)
    quotes = bench.load_gold(gold)
    scripts = {i: s for i in sorted({q.chapter for q in quotes}) if (s := project.load_script(i))}
    _print_reports(
        [bench.score_attribution(scripts, project.load_characters(), quotes, label="this project")]
    )


@bench_app.command("llms")
def bench_llms(
    gold: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="JSONL: chapter, text, speaker")],
    llm: Annotated[list[str], typer.Option("--llm", help="provider:model; repeat to compare")],
    project_dir: ProjectOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the same chapters through several LLMs and rank them on attribution accuracy."""
    setup_logging(verbose)
    project = open_project(project_dir)
    quotes = bench.load_gold(gold)
    chapters_ = sorted({q.chapter for q in quotes})
    _print_reports(bench.compare_llms(project, project.load_book(), chapters_, quotes, llm))


@bench_app.command("tts")
def bench_tts(chapter: ChapterOpt, project_dir: ProjectOpt = None) -> None:
    """Clip count, QA failures and real-time factor of synthesized chapters."""
    project = open_project(project_dir)
    for index in parse_chapters(chapter):
        console.print(f"Chapter {index}: {bench.tts_report(project, index)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
