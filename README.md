# Inflect Studio

A **local desktop app** for zero-shot voice cloning and expressive text-to-speech
with a per-span **inflection editor**. Clone a voice from an MP4/MP3/WAV, type
text, highlight any phrase and change *how it is spoken* — emotion, delivery
description, speed, pauses — then render one seamless track and export WAV/MP3.

Everything runs locally. The only network access is the first-time model
download from Hugging Face.

> **Status:** under active construction. The document model, segmenter and
> project I/O are complete and tested. See "Build phases" below.

## Highlights

- **Voice cloning from video** — extract audio, isolate vocals (Demucs),
  auto-pick the cleanest speech window (Silero VAD), save a reusable profile.
- **Two engines** — Chatterbox (fast "Draft") and IndexTTS-2 (best "Final"),
  loaded one at a time to respect a 12 GB GPU.
- **Inflection editor** — 8 emotion sliders, a natural-language delivery box,
  speed and pause controls, applied to highlighted spans drawn as colored
  underlines.
- **Per-segment caching** — editing one phrase re-renders only that segment.

## Requirements

- Windows (primary target), Linux/macOS best-effort.
- Python 3.10–3.11.
- NVIDIA GPU with CUDA (RTX 4070 / 12 GB reference). CPU works for Chatterbox
  only and is slow.
- **ffmpeg** on your PATH.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate

# 1) Install a CUDA build of torch matching your driver, e.g.:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) Install the rest:
pip install -r requirements.txt
```

Models download automatically on first use into the app data directory.

## Run

```bash
python -m inflect
# or use the launchers:
run.bat        # Windows
./run.sh       # Linux/macOS
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Build phases

1. **Skeleton & document model** ✅ — span model, segmenter, project I/O, config (tested).
2. **Ingest** — ffmpeg extract, VAD candidate selection, profile library, import wizard.
3. **Chatterbox path** — engine adapter, worker, assembly, timeline, export.
4. **IndexTTS-2** — emotion vector / emo-text / alpha / speed, fp16, engine swap.
5. **Polish** — per-segment re-render, tooltips, settings, packaging.
6. **Fish S2 hybrid** (optional) — performance-transfer mode.

## License

MIT for the application code. Bundled/downloaded model weights carry their own
licenses (notably Fish Speech is research/personal-use unless separately
licensed) — review them before any commercial use.
