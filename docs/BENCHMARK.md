# Benchmark plan

Two tracks, matching the two "brains" of the pipeline. Each metric is automatic first. Listening tests come last
and are only used to confirm the automatic metrics.

## Track A: LLM casting brain

| metric | how | status |
|---|---|---|
| **Speaker attribution accuracy** | gold quotes → match the predicted dialogue segment → correct if the predicted id equals the gold speaker (names and aliases resolve through the character map) | `abcraft bench attribution`, `abcraft bench llms` |
| Quote coverage | gold quotes that found a predicted segment (catches quote-splitting bugs) | reported alongside |
| Top confusions | most frequent gold → predicted pairs (twins, narrator vs. POV character, …) | reported alongside |
| Character recall / duplicate rate | gold cast vs. `characters.json`; two ids for one person count as a duplicate | planned |
| Emotion agreement | gold emotion labels on a subset, macro-F1 | planned |
| Cost and latency | tokens, seconds per chapter, VRAM peak | planned |

**Gold data:**
- Your own book: hand-label one chapter (`{"chapter": 3, "text": "…", "speaker": "…"}` per line). About an hour
  of work for ~150 quotes. This is the set that matters for *your* listening.
- A public set: the Project Dialogism Novel Corpus (github.com/Priya22/project-dialogism-novel-corpus). It has
  quote spans, speakers and aliases for 28 novels. Use only its public-domain titles (Austen, Dickens, *Alice*,
  *Anne of Green Gables*, …) and check each title's status where you live. A converter to the JSONL above is
  planned.
- Baselines to beat: `--llm none` (this repo's rules), and published zero-shot LLM results on PDNC (around 90% for
  8B-class models) plus the ModernBookNLP results.

Run every LLM on the same chapters with the same prompts (`bench llms` does that in scratch copies of the
project). Report accuracy with the number of quotes.

## Track B: TTS performance

| metric | how | status |
|---|---|---|
| **Intelligibility** | Whisper transcript vs. text: WER / CER per clip, per voice | per clip with `--asr`; summary planned |
| **Failure rate** | clips failing duration/ASR checks after retakes | `abcraft bench tts` |
| **Speed** | real-time factor (generation seconds ÷ audio seconds), VRAM peak | RTF in `bench tts`; VRAM planned |
| **Voice consistency** | cosine similarity of an *independent* speaker-embedding model (e.g. WavLM-SV, ECAPA) between each line and its character's reference; report the mean and the 5th percentile per voice | planned |
| **Cast distinctness** | the same embedding between different characters' references; lower is better, and pairs closer than a threshold get flagged | planned |
| Naturalness | automatic MOS predictor (UTMOS-style), then a blind A/B on 20 lines × engines with a few listeners | planned |
| Emotion fit | an audio-capable judge (Gemini, or Gemma 4 locally) rates "does this read match the label?" | experimental |

**Fixed protocol:**
1. Use the same 3 chapters: dialogue-heavy, narration-heavy and mixed, from public-domain novels.
2. Use the same reference clips for every engine (designed once, frozen).
3. Use the same seeds, with retakes disabled for the raw scores and enabled for the "shipping" scores.
4. Fix the loudness before any listening test.
5. Publish per-engine JSON (model, version, settings, hardware) next to the audio, so results can be reproduced
   and compared over time.

**A note on the speaker encoder:** don't score voice consistency with the encoder the TTS model conditions on
(Chatterbox's own voice encoder, for example). It would grade its own homework.
