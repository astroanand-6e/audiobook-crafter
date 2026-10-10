# audiobook-crafter

Turn a PDF, EPUB or script into a **full-cast audiobook**: a narrator plus a distinct, consistent voice for
every character, emotion on every line, natural pacing, and mastered MP3/M4B output with chapter markers.
It runs locally on one consumer GPU (built and tuned for a 16 GB RTX 5060 Ti). The LLM "brain" can be local
(Ollama) or bring-your-own-key.

```
PDF/EPUB/TXT/SRT ──▶ chapters ──▶ character map ──▶ script: who says what, how ──▶ cast ──▶ voices ──▶ clips ──▶ mastered MP3 + M4B
      ingest           (LLM, incremental)   (LLM labels numbered quotes)   (never re-cast)  (design→clone) (cached, QA'd)  (pacing, loudness)
```

Every stage writes plain files into a project folder. You can read and hand-edit each one, then re-run.
Only the lines your edit affects are regenerated.

## Convert one chapter

### 1. Install (Windows or Linux)

You need [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/download.html) (`winget install ffmpeg` /
`sudo apt install ffmpeg`), a recent NVIDIA driver and about 15 GB of disk for the models.

```bash
git clone https://github.com/astroanand-6e/audiobook-crafter
cd audiobook-crafter
uv sync --extra qwen          # Qwen3-TTS + PyTorch built for CUDA 12.8 (RTX 50-series / sm_120 ready)
uv run abcraft doctor         # checks ffmpeg, the GPU and its kernels, the TTS packages and Ollama
```

The LLM that reads the book defaults to a local model through [Ollama](https://ollama.com):

```bash
ollama pull qwen3.5:9b        # ~7 GB; unloaded before TTS starts, so both fit in 16 GB of VRAM
```

### 2. Warm up on the included sample

`examples/` holds chapter 1 of *Pride and Prejudice* (public domain), which is almost all untagged back-and-forth
between Mr. and Mrs. Bennet, plus hand-labelled speakers for every quote:

```bash
uv run abcraft run examples/pride_and_prejudice_ch1.txt --chapter 1
uv run abcraft bench llms --gold examples/pride_and_prejudice_ch1.gold.jsonl \
  --llm none --llm ollama:qwen3.5:9b -p audiobooks/pride_and_prejudice_ch1
```

The rules alone (`--llm none`) score 44% here. They get every one of Mr. Bennet's lines right and keep
Mrs. Bennet's lines on one consistent "unnamed woman" voice, because the text only calls her "his wife". An LLM
should name her.

### 3. Pick a chapter

```bash
uv run abcraft chapters "My Book.pdf"
```

The command prints every chapter it found, with its index, title, word count, pages and opening words.
Front matter is chapter 0. If detection is off for your PDF, give the pages directly with `--pages 41-58`.

### 4. Convert it

```bash
uv run abcraft run "My Book.pdf" --chapter 3 --preview 40   # first 40 lines: a 2-minute listening test
uv run abcraft run "My Book.pdf" --chapter 3                # the whole chapter (reuses the preview's clips)
```

Output lands in `audiobooks/my_book/output/`: `My Book - 03 - <title>.mp3` plus `My Book.m4b`. The M4B holds every
chapter converted so far, with chapter markers and the cover.

On the first run, `run` does the following:
1. Reads the chapter, builds the character map and labels every quote with a speaker, emotion, intensity and delivery note.
2. Casts each speaking character.
3. **Designs** a voice per character from its description with Qwen3-TTS VoiceDesign, saving the clip to `voices/designed/`.
4. Speaks every line by cloning those clips with Qwen3-TTS Base, then masters the result.

The model weights download from Hugging Face on first use. Speed depends on the GPU and the engine, so start
with `--preview`. `abcraft bench tts -c 3` then reports the real-time factor on your machine.

### 5. Make it sound right

| To change… | edit | then |
|---|---|---|
| who says a line, its emotion/intensity | `scripts/ch003.json` (`speaker`, `emotion`, `intensity`, `direction`) | re-run `run` |
| a character's voice | `cast.yaml`: drop your clip at `voices/<id>.wav` (+ `voices/<id>.txt` transcript), set `preset: Aiden`, or rewrite `design:` and run `abcraft voices design --only <id> --force` | re-run `run` |
| how the whole cast sounds | `abcraft voices audition` (one line per voice into `output/auditions/`) | — |
| audio-drama feel | `--mode drama` drops bare "she said" tags between lines (scripts keep every word) | re-run `run` |
| catch mumbled or wrong takes | `uv sync --extra qwen --extra qa`, then `--asr` (Whisper checks each clip, retakes failures) | — |
| pauses, loudness, narrator | `project.yaml` (`pacing`, `mastering`, `casting.narrator`) | re-run |

Voices are **never re-cast** once assigned, so a character sounds the same in chapter 30 as in chapter 1. The
exception: a character who started with one line on a shared "extra" voice gets their own voice once they
matter. Run `--scan-book` once to build the character map from the whole book before the first chapter.

## How it decides who's talking

The book's text is split deterministically into narration and quoted lines. It handles “curly”, "straight",
‘British’, «guillemet», „German“ and em-dash dialogue, plus quotes that span paragraphs. The LLM never rewrites text:
it sees each quote as `<q id="12">…</q>` in context and returns labels per id. These are checked against an
enum of known character ids, so it can't invent speakers. Anything it skips falls back to rule-based attribution:
speech tags, action beats, pronouns and "his wife"-style role nouns, and who is on stage to answer an untagged reply.
The character map grows passage by passage with stable ids, and only
exact name/alias matches merge (so "Mr. Bennet" never swallows "Mrs. Bennet"). The rules alone work with
`--llm none`.

## Engines

| engine | voices | emotion control | languages | notes |
|---|---|---|---|---|
| `qwen3-tts` (default) | clone any clip (designed or yours); 9 presets | presets: natural-language instruction; clones: from the text itself | zh en ja ko de fr ru pt es it | Apache-2.0; VoiceDesign creates voices from descriptions |
| `chatterbox-multilingual` | clone any clip | `exaggeration` / `cfg_weight` per line | 23 incl. **Hindi** | MIT, Perth watermark kept |
| `chatterbox` | clone | same | English | |
| `chatterbox-turbo` | clone (≥5 s clip) | `[sigh] [laugh] [gasp] …` tags | English | fast drafts |
| `voxcpm2` | clone; designs voices | a style prompt per line **on the cloned voice** ("(very angry, …)") | 30 incl. **Hindi** | Apache-2.0, 48 kHz, ~8 GB; `uv sync --extra voxcpm` (Python ≤ 3.12) |
| `mock` | tones | — | any | dry runs, tests, no GPU |

`uv run abcraft engines` shows what's installed. `docs/TTS_MODELS.md` surveys the October 2026 model landscape.
VoxCPM2 is wired in because it adds what Qwen3 cloning lacks: per-line emotion on a consistent cloned voice.
IndexTTS-2 is next for the same reason.

**Running two engines together.** qwen-tts and chatterbox-tts pin incompatible `transformers` versions, so give
the second one its own environment and the pipeline will drive it as a worker process:

```bash
UV_PROJECT_ENVIRONMENT=.venv-chatterbox uv sync --extra chatterbox
```

```yaml
# audiobooks/my_book/project.yaml
tts:
  engines:
    chatterbox-multilingual:
      python: /path/to/audiobook-crafter/.venv-chatterbox/bin/python   # Windows: ...\.venv-chatterbox\Scripts\python.exe
```

Then use `--engine chatterbox-multilingual` for everything, or set `engine: chatterbox-multilingual` on single
voices in `cast.yaml`. One voice can be designed with Qwen and performed with Chatterbox's emotion knob.

**Hindi and other languages Qwen3-TTS lacks:** use `--engine chatterbox-multilingual --language hi` with Hindi
reference clips in `voices/`. Clips designed in another language carry its accent. Setting `params: {cross_language: true}`
on a voice softens that.

## LLM brain: local or bring your own key

`--llm provider:model` (or `llm.model` in `project.yaml`). Keys come from environment variables and are never
written to project files.

| provider | example | key |
|---|---|---|
| `ollama` (default) | `ollama:qwen3.5:9b`, `ollama:gemma4:12b` | — |
| `lmstudio`, `llamacpp`, `vllm` | `lmstudio:<model>` | — |
| `anthropic` | `anthropic:claude-opus-5-5` | `ANTHROPIC_API_KEY` (needs `uv sync --extra anthropic`) |
| `gemini` | `gemini:<model id>` | `GEMINI_API_KEY` |
| `openai` | `openai:<model id>` | `OPENAI_API_KEY` |
| `openrouter`, `groq`, `deepseek`, `together` | `openrouter:<vendor/model>` | `<PROVIDER>_API_KEY` |
| any OpenAI-compatible server | `--llm myserver:<model>` + `llm.base_url` | `llm.api_key_env` |

Ollama is called through its native API so the context window (`llm.context_tokens`, default 32K) is really
applied: Ollama's own default of 4K silently truncates long prompts. With `llm.unload_after` the model is unloaded
afterwards, so the GPU is free for TTS. Thinking models run with `think: false` by default for speed.

## Testbench

The same pipeline doubles as a benchmark for LLMs and TTS engines:

```bash
uv run abcraft bench llms --gold my_chapter_gold.jsonl --llm none --llm ollama:qwen3.5:9b --llm ollama:gemma4:12b -p audiobooks/my_book
uv run abcraft bench attribution --gold my_chapter_gold.jsonl -p audiobooks/my_book
uv run abcraft bench tts -c 3 -p audiobooks/my_book        # clips, QA failures, real-time factor
```

Gold labels are JSON lines: `{"chapter": 3, "text": "Is it from London?", "speaker": "Clara"}`. `docs/BENCHMARK.md`
has the full metric plan and test-book protocol: attribution accuracy, intelligibility, voice consistency and
distinctness, naturalness, speed.

## Project folder

```
audiobooks/my_book/
  project.yaml          settings (LLM, engine, pacing, mastering, casting)
  book.json             chapters and paragraphs from the PDF
  characters.json       character map: names, aliases, gender, age, voice description, line counts
  scripts/ch003.json    every segment: text, speaker, emotion, intensity, direction, tags
  cast.yaml             who is voiced how (clip / preset / design / same_as / engine / params)
  voices/               your clips (<id>.wav), voices/pool/ (male_*.wav, female_*.wav), voices/designed/
  cache/tts/            every clip, keyed by a hash of its inputs
  render/               voice track, mastered WAV and clip manifest per chapter
  output/               MP3 per chapter, the M4B, auditions
```

## Roadmap

- Sound design: per-scene ambience beds and spot effects with ducking. The cue schema is sketched in
  `docs/ARCHITECTURE.md`. Effects would come from a tagged local library first, then a generative SFX model.
- A director pass: scene-level pacing and emotion arcs, and paralinguistic cues where the text earns them.
- More engines: IndexTTS-2 (emotion vectors on cloned voices). Batched generation for speed.
- Podcast mode: two hosts discussing a document, using the existing script → cast → render path.
- Listening QA with an audio-capable LLM (Gemma 4 12B locally, Gemini via BYOK) for emotion and pronunciation.

## Use it responsibly

Convert books you own, for your own listening. Nothing here removes DRM, and nothing should. Only clone voices
you have permission to use. Designed voices avoid that question entirely. Chatterbox output keeps Resemble's
Perth watermark. This is a local tool. Hosting conversions of copyrighted books for others is a different thing
and isn't what it's for.

## Development

```bash
uv sync --extra dev
uv run pytest            # no GPU or model downloads needed: engines are tested against stand-ins
uv run ruff check src tests
```
