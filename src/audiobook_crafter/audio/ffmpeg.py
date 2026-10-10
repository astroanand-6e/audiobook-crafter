"""ffmpeg wrappers: two-pass EBU R128 mastering, MP3 with ID3 tags, M4B with chapter markers and cover."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import soundfile as sf

from ..config import MasteringConfig


class FFmpegError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegError(
            "ffmpeg not found on PATH. Install it: `winget install ffmpeg` (Windows), "
            "`sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)."
        )
    return path


def run(args: list[str]) -> str:
    """Run ffmpeg; returns its stderr (where filters such as loudnorm report)."""
    cmd = [ffmpeg_binary(), "-hide_banner", "-nostdin", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({result.returncode}): {result.stderr[-1500:]}")
    return result.stderr


def duration(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except RuntimeError:
        out = run(["-i", str(path), "-f", "null", "-"])
        times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", out)
        if not times:
            return 0.0
        h, m, s = times[-1]
        return int(h) * 3600 + int(m) * 60 + float(s)


def resample(src: Path, dst: Path, sample_rate: int) -> Path:
    run(["-v", "error", "-i", str(src), "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)])
    return dst


def prepare_reference(src: Path, dst: Path, max_seconds: float | None) -> Path:
    """Mono 24 kHz, edge silence removed, loudness evened out, optionally capped in length.

    A short silence is appended: in-context cloning (Qwen3 Base) continues from the end of the
    reference and trims it off by an estimate, so any overshoot lands in silence, not in speech."""
    edge = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.1"
    filters = f"{edge},areverse,{edge},areverse,loudnorm=I=-20:TP=-2:LRA=11,apad=pad_dur=0.4"
    args = ["-v", "error", "-i", str(src), "-ac", "1", "-af", filters, "-ar", "24000"]
    if max_seconds:
        args += ["-t", f"{max_seconds:.2f}"]
    run([*args, "-c:a", "pcm_s16le", str(dst)])
    return dst


def _prefilters(config: MasteringConfig) -> str:
    chain = ["highpass=f=60"]
    if config.compressor:
        chain.append("acompressor=threshold=-24dB:ratio=2.5:attack=15:release=250:knee=4:makeup=1")
    return ",".join(chain)


def measure_loudness(src: Path, config: MasteringConfig) -> dict[str, str] | None:
    target = f"I={config.loudness_lufs}:TP={config.true_peak_db}:LRA={config.lra}"
    out = run(["-nostats", "-i", str(src), "-af", f"{_prefilters(config)},loudnorm={target}:print_format=json",
               "-f", "null", "-"])  # fmt: skip
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", out, re.DOTALL)
    if not match:
        return None
    stats = json.loads(match.group(0))
    return None if "inf" in stats.get("input_i", "") else stats


def master(src: Path, dst: Path, config: MasteringConfig) -> Path:
    """High-pass, gentle compression, two-pass linear loudness normalisation, final sample rate."""
    stats = measure_loudness(src, config)
    target = f"I={config.loudness_lufs}:TP={config.true_peak_db}:LRA={config.lra}"
    if stats:
        loudnorm = (
            f"loudnorm={target}:measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}:linear=true"
        )
    else:  # too short or silent to measure
        loudnorm = f"loudnorm={target}"
    filters = f"{_prefilters(config)},{loudnorm},aresample={config.sample_rate}"
    run(
        [
            "-v",
            "error",
            "-i",
            str(src),
            "-af",
            filters,
            "-ac",
            str(config.channels),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    return dst


def _tag_args(tags: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in tags.items():
        if value:
            args += ["-metadata", f"{key}={value}"]
    return args


def encode_mp3(src: Path, dst: Path, config: MasteringConfig, tags: dict[str, str]) -> Path:
    run(["-v", "error", "-i", str(src), "-c:a", "libmp3lame", "-b:a", config.mp3_bitrate,
         "-ar", str(config.sample_rate), "-ac", str(config.channels), "-id3v2_version", "3",
         *_tag_args(tags), str(dst)])  # fmt: skip
    return dst


def _concat_entry(path: Path) -> str:
    quoted = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{quoted}'\n"


def _escape_metadata(value: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def build_m4b(
    chapters: list[tuple[Path, str]],
    dst: Path,
    config: MasteringConfig,
    tags: dict[str, str],
    cover: Path | None = None,
) -> Path:
    """Concatenate mastered chapter WAVs into one AAC audiobook with chapter markers."""
    work = dst.parent / f".{dst.stem}.parts"
    work.mkdir(parents=True, exist_ok=True)
    concat_list = work / "inputs.txt"
    concat_list.write_text("".join(_concat_entry(path) for path, _ in chapters), encoding="utf-8")
    lines = [";FFMETADATA1"] + [f"{k}={_escape_metadata(v)}" for k, v in tags.items() if v]
    start = 0
    for path, title in chapters:
        length = int(round(duration(path) * 1000))
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={start + length}",
                  f"title={_escape_metadata(title)}"]  # fmt: skip
        start += length
    metadata = work / "chapters.txt"
    metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")

    args = ["-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(metadata)]
    if cover and cover.exists():
        args += [
            "-i",
            str(cover),
            "-map",
            "0:a",
            "-map",
            "2:v",
            "-c:v",
            "mjpeg",
            "-disposition:v:0",
            "attached_pic",
        ]
    else:
        args += ["-map", "0:a"]
    args += ["-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", config.m4b_bitrate,
             "-ar", str(config.sample_rate), "-ac", str(config.channels), "-movflags", "+faststart",
             "-f", "mp4", str(dst)]  # fmt: skip
    run(args)
    shutil.rmtree(work, ignore_errors=True)
    return dst
