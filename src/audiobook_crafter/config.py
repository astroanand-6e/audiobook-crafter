"""Per-project settings, stored as project.yaml next to the book's working files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    # "<provider>:<model>", e.g. "ollama:qwen3.5:9b", "ollama:gemma4:12b", "gemini:<model>",
    # "anthropic:claude-opus-5-5". "none" skips the LLM and uses rule-based speaker attribution.
    model: str = "ollama:qwen3.5:9b"
    base_url: str | None = None  # override the provider's default endpoint
    api_key_env: str | None = None  # env var that holds the key (provider default if unset)
    temperature: float = 0.2
    context_tokens: int = 32768  # Ollama num_ctx (Ollama's own default of 4K silently truncates prompts)
    chunk_chars: int = 12000  # book text sent per request
    max_output_tokens: int = 4096
    timeout_s: float = 900
    think: bool | None = False  # Ollama reasoning toggle for thinking models (None = model default)
    effort: str | None = None  # Anthropic output_config.effort: low | medium | high | xhigh | max
    unload_after: bool = True  # free the GPU (Ollama keep_alive=0) before TTS starts


class TTSConfig(BaseModel):
    engine: str = "qwen3-tts"
    design_engine: str = "qwen3-voice-design"  # creates reference clips from character descriptions
    device: str = "auto"
    seed: int = 1234
    max_chars: int | None = None  # characters per TTS request (engine default if unset)
    # Per-engine options, e.g. {"qwen3-tts": {"model_size": "0.6B"},
    #                           "chatterbox-multilingual": {"python": "/path/.venv-chatterbox/bin/python"}}
    # "python" runs that engine in a separate interpreter (for engines with conflicting dependencies).
    engines: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CastingConfig(BaseModel):
    # Voice-design brief for the narrator (ignored once cast.yaml gives the narrator a clip or preset).
    narrator: str = (
        "A warm, clear, mature audiobook narrator with crisp diction, an even, unhurried pace and gentle "
        "expressiveness. Close-mic studio recording, no background noise."
    )
    min_lines_for_own_voice: int = 2  # characters with fewer lines share a pool voice ("extra_*")
    pov_uses_narrator_voice: bool = True  # first-person books: the "I" character speaks as the narrator


class QAConfig(BaseModel):
    duration_check: bool = True  # catch truncated / runaway generations by their length
    asr: bool = False  # transcribe every clip with faster-whisper and compare to the text
    asr_model: str = "small"
    asr_device: str = "auto"
    max_wer: float = 0.2
    retries: int = 2
    # Re-generate cached clips that still had a QA issue. Set per run with --retake; never saved.
    retake_failed: bool = Field(default=False, exclude=True)


class StyleConfig(BaseModel):
    # audiobook: unabridged, every word of the book is read.
    # drama: short speech tags between lines ("he said") are dropped so it plays like an audio drama.
    mode: Literal["audiobook", "drama"] = "audiobook"
    announce_chapter_titles: bool = True
    emotion_strength: float = 1.0  # scales how hard emotions push the engine (0 = flat read)
    narrator_mood: bool = True  # narration follows the scene's mood, gently


class PacingConfig(BaseModel):
    sentence_ms: int = 320  # between sentences of one passage
    clause_ms: int = 160  # where a long sentence had to be split
    same_paragraph_ms: int = 260  # dialogue <-> "she said" inside one paragraph
    paragraph_ms: int = 700
    dialogue_turn_ms: int = 480  # a different character answers
    scene_break_ms: int = 1800
    after_heading_ms: int = 1500
    lead_in_ms: int = 800  # silence at the start of each chapter file
    tail_ms: int = 2500  # silence at the end of each chapter file


class MasteringConfig(BaseModel):
    loudness_lufs: float = -19.0  # inside the ACX / Audible -23..-18 window
    true_peak_db: float = -3.0
    lra: float = 11.0
    sample_rate: int = 44100
    channels: int = 1
    mp3_bitrate: str = "192k"
    m4b_bitrate: str = "96k"
    compressor: bool = True  # gentle compression so whispers and shouts both stay intelligible
    level_speakers: bool = True  # match the average level of every voice before mastering
    room_tone_db: float | None = None  # e.g. -68 adds faint noise instead of digital silence


class ProjectConfig(BaseModel):
    title: str = ""
    author: str = ""
    language: str = ""  # ISO 639-1 code; empty = detect from the book
    source: str = ""
    # PDF chapters whose pages you set by hand (--pages FIRST-LAST --chapter N): {N: [FIRST, LAST]}
    chapter_pages: dict[int, list[int]] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    casting: CastingConfig = Field(default_factory=CastingConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)
    pacing: PacingConfig = Field(default_factory=PacingConfig)
    mastering: MasteringConfig = Field(default_factory=MasteringConfig)

    @classmethod
    def load(cls, path: Path) -> ProjectConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        text = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8")
