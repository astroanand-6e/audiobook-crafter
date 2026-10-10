# Architecture

Five stages, each reading and writing files in the project folder (`audiobooks/<book>/`). Each stage is
restartable and safe to hand-edit between runs.

| stage | module | input → output |
|---|---|---|
| 1. ingest | `ingest/` | PDF / EPUB / TXT / MD / SRT / VTT → `book.json` (chapters → headings, paragraphs, scene breaks) |
| 2. analyze | `analysis/`, `llm/` | chapter → `characters.json` + `scripts/chNNN.json` (segments with speaker, emotion, intensity, direction, tags) |
| 3. cast | `casting/` | characters → `cast.yaml`; voice design → `voices/designed/*.wav` |
| 4. synthesize | `synth/`, `tts/` | script + cast → cached clips in `cache/tts/` + `render/chNNN.clips.json` |
| 5. assemble | `assemble/`, `audio/` | clips → voice track → mastered WAV → `output/*.mp3` + `output/<book>.m4b` |

`pipeline.py` runs them in order and frees GPU memory between the LLM, the voice-design model and the speaking model.

## Decisions worth knowing

**The LLM labels, it never rewrites.** Quotes are found deterministically (`text/quotes.py`). The model
gets the passage with `<q id="N">` markers and returns `{id, speaker, emotion, intensity, direction, tags}`. The
speaker is constrained to an enum of known ids. The book's words can't drift, a missed id is detected and
re-asked, and a weak model degrades gracefully into the rule-based attributor (`analysis/heuristics.py`), which
also serves as the benchmark baseline.

**The character map is incremental with stable ids.** Passages are read in context-sized chunks, so a 9B local
model and a 1M-token API model use the same code. Ids never change once assigned. Merging is exact on names and
aliases, and a new character's alias may only match an existing *name*, which keeps "Mr. Bennet" and
"Mrs. Bennet" apart.

**Casting is sticky.** `auto_cast` only fills gaps. It never reassigns a voice, except to promote a minor character
off a shared "extra" voice once they have lines. That rule is what keeps voices consistent across a book
converted one chapter at a time.

**Design once, clone forever.** A voice description becomes one reference clip with a known transcript
(`casting/design.py`). Every line is then cloned from that clip with a fixed per-voice seed. That gives the
same timbre in every chapter, whatever engine performs it.

**The cache is content-addressed.** A clip's key hashes the engine, model version, options, text, emotion,
voice fingerprint (reference audio bytes + transcript + params) and seed. Edits re-render exactly the affected
lines. Identical lines are rendered once.

**QA retakes.** Each clip is checked for runaway, cut-off or silent output by duration. With `--asr`, Whisper
checks word error rate too. Failures are re-generated with new seeds and the best take is kept.

**Engines are isolated.** `TTSEngine` is a small interface (`synthesize`, optional `design`). Engines that
can't share an environment run through `tts/worker.py`: JSON lines over stdin/stdout to an interpreter in
another venv, with library stdout redirected so the protocol stays clean.

**Pacing and mastering.** Gaps come from structure, not fixed silence:
- inside a paragraph
- across a paragraph
- a new speaker answering
- interruptions (an em-dash cuts straight in)
- scene breaks and headings

Voices are levelled to a common average so a quiet clone doesn't vanish next to a loud one; each voice keeps
its own dynamics. Mastering is high-pass, then gentle compression, then two-pass linear EBU R128 loudness at
-19 LUFS with -3 dBTP (within the ACX/Audible window), then 44.1 kHz.

## Adding an engine

1. Subclass `tts.base.TTSEngine` and implement `synthesize(line, voice, seed) -> (float32 mono, sample_rate)`.
   Implement `design()` too if the model can invent voices. Map `line.emotion` / `line.intensity` /
   `line.direction` / `line.tags` onto whatever controls the model has (`tts/emotion.py` has helpers).
2. Register it in `tts/__init__.py:ENGINES` with the import name that must be present.
3. Add a stand-in test in `tests/test_engines.py` that copies the real library's signatures.

## Sound design (planned)

Sound design rides on the same timeline as the voice track. A director pass (LLM) would add cues to the script:

```json
{"segment": "c003-s0042", "cue": "sfx",      "query": "heavy wooden door creaks open", "at": "before", "gain_db": -14}
{"segment": "c003-s0040", "cue": "ambience", "query": "steady rain on a window", "until": "c003-s0077", "gain_db": -28, "duck_db": 8}
```

- **Resolving cues:** a local, tagged SFX library (CC0 packs) comes first; a generative model (e.g.
  MOSS-SoundEffect) fills gaps.
- **Rendering:** cues mix under the voice track with sidechain ducking (`sidechaincompress` in ffmpeg) before
  mastering. The output goes stereo once beds are involved.
- **Defaults:** off, and conservative when on, so effects support the narration and never cover it.
