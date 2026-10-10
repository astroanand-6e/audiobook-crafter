"""Reference clips on disk -> Voice objects the engines can use.

voices/<character_id>.wav (+ optional .txt transcript)   a clip for one character (or "narrator")
voices/pool/<male|female>_<anything>.wav                  clips auto-assigned to new characters
voices/designed/<voice_id>.wav + .txt                     clips made by the voice-design model
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..audio import ffmpeg
from ..models import VoiceProfile
from ..project import Project
from ..tts.base import Voice

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
PREP_VERSION = "r1"  # bump when ffmpeg.prepare_reference changes, so old prepared clips aren't reused


def _audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def user_clips(project: Project) -> dict[str, Path]:
    """voices/<id>.<ext> -> {id: path}."""
    return {p.stem: p for p in _audio_files(project.voices_dir)}


def pool_clips(project: Project) -> list[tuple[str, Path]]:
    """voices/pool/male_*.wav etc. -> [(gender, path)]."""
    clips = []
    for path in _audio_files(project.voices_dir / "pool"):
        prefix = path.stem.lower().split("_")[0]
        gender = {
            "m": "male",
            "male": "male",
            "man": "male",
            "f": "female",
            "female": "female",
            "woman": "female",
        }
        clips.append((gender.get(prefix, "unknown"), path))
    return clips


def transcript_for(path: Path) -> str | None:
    sidecar = path.with_suffix(".txt")
    if sidecar.exists():
        text = " ".join(sidecar.read_text(encoding="utf-8").split())
        return text or None
    return None


def _digest(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_voice(
    project: Project, voice_id: str, profile: VoiceProfile, engine: str, default_seed: int
) -> Voice:
    reference: Path | None = None
    reference_text = profile.reference_text
    content = ""
    if profile.reference:
        source = project.resolve(profile.reference)
        if not source.exists():
            raise FileNotFoundError(f"Voice '{voice_id}': reference clip {source} does not exist")
        reference_text = reference_text or transcript_for(source)
        content = _digest(source)
        # A transcript must match the audio exactly, so only cap clips that have none.
        cap = None if reference_text else 15.0
        reference = (
            project.cache_dir / "refs" / f"{content[:16]}-{PREP_VERSION}{'' if cap is None else '-cap'}.wav"
        )
        if not reference.exists():
            reference.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg.prepare_reference(source, reference, cap)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "ref": content,
                "prep": PREP_VERSION,
                "text": reference_text,
                "preset": profile.preset,
                "params": profile.params,
                "engine": engine,
            },  # fmt: skip
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    return Voice(
        id=voice_id,
        reference=reference,
        reference_text=reference_text,
        preset=profile.preset,
        params=dict(profile.params),
        seed=profile.seed if profile.seed is not None else default_seed,
        fingerprint=fingerprint,
    )
