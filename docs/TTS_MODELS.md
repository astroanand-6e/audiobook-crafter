# Open TTS models for full-cast audiobooks (snapshot: October 2026)

This survey covers what matters for this project: **consistent cloned voices**, **per-line emotion control**,
**voice design from a description**, long-form stability, and whether the model fits a **16 GB RTX 5060 Ti**.
Model releases move fast, so re-check licences and versions before you depend on them. Items marked
*(unverified)* came only from secondary sources.

## What we use today

**Qwen3-TTS** (Alibaba, Apache-2.0, Jan 2026; `qwen-tts` 0.1.1). This is the default.
- *Base* (0.6B/1.7B) clones from about 3 s of audio; with a transcript it uses in-context mode and copies the voice closely.
- *VoiceDesign* (1.7B) invents a voice from a free-text description. Its README recommends design → clone, which
  is how this project keeps every character consistent.
- *CustomVoice* has 9 preset speakers that follow natural-language instructions ("very angry").
- Languages: zh, en, ja, ko, de, fr, ru, pt, es, it. **No Hindi.**
- **Limitation:** Base takes no emotion instruction, so a cloned voice acts only from the text. Presets and
  VoiceDesign take instructions.
- **Known issue:** clones can echo the end of the reference clip (QwenLM/Qwen3-TTS#341). This project appends
  silence to every prepared reference so the overshoot lands in silence, and trims it.
- Some listeners hear a slight Chinese accent in VoiceDesign English.

**Chatterbox** (Resemble AI, MIT). The alternative engine.
- *Multilingual* (0.5B) covers 23 languages including Hindi. Emotion is a single `exaggeration` knob, paired with
  `cfg_weight` for pacing.
- *English* (0.5B) has the same controls.
- *Turbo* (350M, Dec 2025) is English only. It takes inline tags (`[laugh] [chuckle] [gasp] [cough] [sigh]
  [groan] [sniff] [shush] [clear throat]`; others are accepted but do nothing), needs a reference of 5 s or more,
  and is unstable past about 25–30 s per call.
- Newer releases since PyPI 0.1.7: Multilingual V3 (Jun 2026, `t3_model="v3"`, from git), a dedicated Hindi pack,
  and Nano (110M) *(unverified)*.
- `chatterbox-tts` pins `torch==2.6.0`, which has **no RTX 50-series (sm_120) kernels**. This repo's
  `pyproject.toml` overrides the pin to torch 2.7–2.8 from the CUDA 12.8 index and drops the unused Gradio
  dependency.
- Resemble has pivoted to deepfake detection, so expect thin maintenance.

## Worth adding next

| model | why | caveats |
|---|---|---|
| **IndexTTS-2** (bilibili; 2.5 released Aug 2026) | Speaker and emotion are disentangled. Freeze one clip per character and set each line's emotion with an 8-value vector or a text description. Duration control. Runs on cu128. | Custom licence: royalty-free, with limits above 100M MAU or RMB 1B revenue. No Hindi. Installed from git. |
| **VoxCPM2** (OpenBMB, Apache-2.0, Apr 2026, 2B) | Style prompts such as `(tense whisper)` that work **on a cloned voice**, plus voice design and 30 languages **including Hindi**. 48 kHz output. ~8 GB. | Output varies between runs, so plan for 1–3 takes (our QA loop fits). Glitches on very short phrases. |
| MOSS-TTS family (Apache-2.0) | v1.5 has 31 languages including Hindi, a voice generator, a dialogue model (MOSS-TTSD), and **MOSS-SoundEffect** for the SFX stage. | Pins torch 2.9.1. |

## Others, briefly

- **Fish Audio S2-Pro:** top open-weight on Artificial Analysis, but non-commercial and needs 24 GB.
- **Higgs Audio v3:** non-commercial, borderline at 16 GB.
- **Breeze TTS 2:** non-commercial.
- **Step-Audio-EditX** (Apache): edits emotion and style of existing audio, a possible post-processing pass.
- **ZONOS2:** 8B MoE. Tight at 16 GB unless quantized.
- **Voxtral TTS:** non-commercial, and the open weights can't clone.
- **OmniVoice:** 646 languages and attribute-based design. VRAM creeps up on a 5060 Ti; call `torch.cuda.empty_cache()`.
- **VibeVoice:** community fork, long multi-speaker form. Sometimes adds music.
- **Dia2, Orpheus, Kokoro, Kyutai Pocket TTS, NeuTTS:** English or preset-only. Fine for previews.

## Voice-conversion modules (a separate "cloning" stage, if ever needed)

- **Chatterbox VC** (MIT): `ChatterboxVC.generate(wav, target_voice_path=...)`.
- **MeanVC2** (Apache-2.0): 18M parameters, real-time on CPU.
- **Seed-VC:** GPL-3.0, archived.
- **OpenVoice V2:** MIT, dormant.
- **RVC:** needs per-voice training.

Design → clone inside the TTS model gives cleaner results than converting voices after the fact, so this project
doesn't use VC by default.

## Recommendation for a 16 GB RTX 5060 Ti

- **English full cast:** Qwen3-TTS design → clone now. Benchmark IndexTTS-2 next for per-line emotion on cloned voices.
- **Hindi:** Chatterbox Multilingual (or its Hindi pack) with Hindi reference clips, A/B tested against VoxCPM2.
- **Designing character voices:** Qwen3-TTS VoiceDesign for English. VoxCPM2 for strong accents or Hindi.
- **Fast drafts:** Chatterbox Turbo with the same reference clips, or Kokoro for narrator-only previews.

Run each engine in its own environment. Their pins conflict (transformers 4.57 vs 5.2, torch 2.9.1, Python < 3.13),
so the worker mode in `audiobook_crafter/tts/worker.py` exists for exactly this.
