"""Run a TTS engine in another Python environment.

TTS models pin conflicting dependencies (qwen-tts wants transformers 4.57, chatterbox-tts wants 5.2),
so each can live in its own venv. Point `tts.engines.<name>.python` at that venv's interpreter and the
pipeline talks to it over JSON lines on stdin/stdout. That venv needs this package installed too.

    python -m audiobook_crafter.tts.worker <engine> '<json options>'
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .base import Line, TTSEngine, TTSError, Voice


def voice_to_dict(voice: Voice) -> dict[str, Any]:
    data = asdict(voice)
    data["reference"] = str(voice.reference) if voice.reference else None
    return data


def voice_from_dict(data: dict[str, Any]) -> Voice:
    return Voice(**{**data, "reference": Path(data["reference"]) if data.get("reference") else None})


class WorkerEngine(TTSEngine):
    def __init__(self, name: str, options: dict[str, Any] | None = None, device: str = "auto"):
        super().__init__(name, options, device)
        self.python = str(self.options.pop("python"))
        self.proc: subprocess.Popen[str] | None = None
        self._version = ""

    @property
    def version(self) -> str:
        return self._version

    def load(self) -> None:
        if self.proc is not None:
            return
        payload = json.dumps({"options": self.options, "device": self.device})
        try:
            self.proc = subprocess.Popen(
                [self.python, "-m", "audiobook_crafter.tts.worker", self.name, payload],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as err:
            raise TTSError(f"Can't start the {self.name} worker with {self.python}: {err}") from err
        info = self._call({"op": "info"})
        self.default_max_chars = int(info["max_chars"])
        self.languages = frozenset(info["languages"]) if info.get("languages") else None
        self.can_design = bool(info.get("can_design"))
        self._version = str(info.get("version", ""))

    def _call(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise TTSError(f"The {self.name} worker isn't running")
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        reply_line = self.proc.stdout.readline()
        if not reply_line:
            raise TTSError(f"The {self.name} worker exited (code {self.proc.poll()}); its log is above")
        reply = json.loads(reply_line)
        if not reply.get("ok"):
            raise TTSError(f"{self.name}: {reply.get('error')}")
        return reply

    def _render(self, message: dict[str, Any]) -> tuple[np.ndarray, int]:
        self.load()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self._call({**message, "out": path})
            audio, sr = sf.read(path, dtype="float32")
            return audio, int(sr)
        finally:
            Path(path).unlink(missing_ok=True)

    def synthesize(self, line: Line, voice: Voice, seed: int) -> tuple[np.ndarray, int]:
        return self._render(
            {"op": "synthesize", "line": asdict(line), "voice": voice_to_dict(voice), "seed": seed}
        )

    def design(self, description: str, text: str, language: str, seed: int) -> tuple[np.ndarray, int]:
        return self._render(
            {"op": "design", "description": description, "text": text, "language": language, "seed": seed}
        )

    def unload(self) -> None:
        if self.proc is None:
            return
        try:
            self._call({"op": "close"})
            self.proc.wait(timeout=60)
        except (TTSError, OSError, subprocess.TimeoutExpired, ValueError):
            self.proc.kill()
        self.proc = None


def main() -> None:
    from . import create_local_engine

    name, payload = sys.argv[1], json.loads(sys.argv[2])
    # Model libraries print freely; keep the real stdout for the protocol and send everything else to stderr.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    engine = create_local_engine(name, payload.get("options", {}), payload.get("device", "auto"))
    loaded = False

    def reply(data: dict[str, Any]) -> None:
        protocol.write(json.dumps(data) + "\n")
        protocol.flush()

    for raw in sys.stdin:
        message = json.loads(raw)
        op = message.get("op")
        try:
            if op == "info":
                languages = sorted(engine.languages) if engine.languages else None
                reply({"ok": True, "max_chars": engine.default_max_chars, "languages": languages,
                       "can_design": engine.can_design, "version": engine.version})  # fmt: skip
            elif op in ("synthesize", "design"):
                if not loaded:
                    engine.load()
                    loaded = True
                if op == "synthesize":
                    audio, sr = engine.synthesize(
                        Line(**message["line"]), voice_from_dict(message["voice"]), message["seed"]
                    )
                else:
                    audio, sr = engine.design(
                        message["description"], message["text"], message["language"], message["seed"]
                    )
                sf.write(message["out"], audio, sr, subtype="FLOAT")
                reply({"ok": True, "sample_rate": sr})
            elif op == "close":
                engine.unload()
                reply({"ok": True})
                break
            else:
                reply({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as err:  # noqa: BLE001 - report every failure back to the pipeline
            reply({"ok": False, "error": f"{type(err).__name__}: {err}"})


if __name__ == "__main__":
    main()
