# Inflect Studio

A **local desktop app** for zero-shot voice cloning and expressive text-to-speech
with a per-span **inflection editor**. Clone a voice from an MP4/MP3/WAV, type a
script, highlight any phrase and change *how it is spoken* — emotion, a
natural-language delivery description, speed and pauses — then render one
seamless track and export WAV/MP3.

Everything runs locally. The only network access is the first-time model
download from Hugging Face.

---

## Features

- **Clone a voice from video/audio** — ffmpeg extracts audio, optional Demucs
  isolates vocals, Silero VAD auto-selects the cleanest 10–20 s of speech, and
  you audition/trim before saving a reusable **Voice Profile**. Zero-shot, no
  training.
- **Two built-in engines**
  - **Chatterbox** — fast **Draft** previews (`exaggeration` / `cfg_weight`).
  - **IndexTTS-2** — best **Final** quality with an 8-dim emotion vector,
    natural-language emotion text, emotion-reference audio and emotion strength.
  - **Fish Speech / S2** + a **Hybrid "performance transfer"** mode (Phase 6,
    optional).
- **Inflection editor** — highlight text and set 8 emotion sliders, a "describe
  delivery" box, emotion strength, speed and a trailing pause. Styled spans are
  drawn as colored wavy underlines (derived from the model, never baked into the
  text) and hovering shows a summary tooltip.
- **Seamless assembly** — per-segment renders are joined with 15 ms equal-power
  crossfades, pauses inserted, and the mix normalized to −16 LUFS / −1 dBTP.
- **Per-segment caching** — every segment is cached by a content hash, so
  editing one phrase only re-renders that one segment.
- **Timeline** — pyqtgraph waveform with playhead, click-to-seek, space to
  play/pause, colored segment regions and a right-click "re-render segment".
- **Projects** — save/load `.inflect` files; a persistent voice library.

## Requirements

- **OS:** Windows (primary target). Linux/macOS are best-effort.
- **Python:** 3.10 – 3.11.
- **GPU:** NVIDIA with CUDA (reference: RTX 4070 / 12 GB). IndexTTS-2 and Fish
  effectively require CUDA; Chatterbox can run on CPU (slowly).
- **ffmpeg** on your `PATH` (or set its path in Settings).

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate

# 1) Install a CUDA build of torch matching your driver FIRST, e.g. CUDA 12.1:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) Install everything else:
pip install -r requirements.txt
```

Notes:
- **IndexTTS-2** installs from GitHub (`index-tts/index-tts`) via
  `requirements.txt`. Its weights (`IndexTeam/IndexTTS-2`, several GB) download
  automatically on first Final-mode render.
- **Chatterbox** pulls its own weights on first Draft-mode render.
- **Fish Speech** (Phase 6) is heavy and research-licensed — install it manually
  when you want it: `pip install fish-speech`.

## Run

```bash
python -m inflect
# or the launchers (they create .venv and install deps on first run):
run.bat        # Windows
./run.sh       # Linux/macOS
```

## Usage

1. **Import a voice** — Voice Library (left) → *＋ Import from video / audio…*.
   Pick a file; the wizard extracts, optionally isolates vocals, and offers the
   three cleanest clips. Audition, trim, name it, tick *"I have permission to
   clone this voice"*, Save.
2. **Pick the voice** in the toolbar combo, and an **engine**: *Draft
   (Chatterbox)* for fast iteration or *Final (IndexTTS-2)* for quality.
3. **Type** your script in the center editor.
4. **Inflect** — select a phrase, set emotion sliders / "describe delivery" /
   speed / pause in the Inspector (right), then **Apply to selection**. Use
   **Set as document default** for the baseline delivery, or **Preview this
   segment** to hear just the selection.
5. **Synthesize All** in the toolbar. Watch the segment progress; **Cancel**
   stops between segments.
6. **Play** in the timeline (space / click to seek). Right-click a segment to
   re-render just it.
7. **Export…** to WAV or MP3.

## Where data lives

A per-user app data folder (override with the `INFLECT_HOME` env var):

- Windows: `%APPDATA%\InflectStudio`
- Linux: `~/.local/share/InflectStudio` (or `$XDG_DATA_HOME/InflectStudio`)

Inside it: `models/` (downloaded weights), `voices/` (profiles + `index.json`),
`project_cache/` (per-segment wavs), `logs/inflect.log`, `settings.json`.

## VRAM discipline (12 GB)

- Exactly **one** TTS engine is resident at a time; switching engines unloads
  the previous one (`del` + `empty_cache()` + `gc.collect()`).
- Demucs loads, runs and unloads within a single call — it never shares the GPU
  with a TTS engine.
- IndexTTS-2 runs in **fp16** with the CUDA kernel enabled (toggle in Settings).
- `torch.cuda.empty_cache()` runs after every segment; the status bar shows a
  live VRAM gauge.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The core (span model, segmenter, project I/O, VAD scoring, profiles, assembly
DSP, engine parameter mapping) is unit-tested. If PySide6 + its system libs are
available, offscreen GUI smoke tests also run; otherwise they skip.

## Troubleshooting

- **"ffmpeg not found"** — install ffmpeg and ensure it's on `PATH`, or set its
  full path in *Settings → ffmpeg path*. Required for importing media and MP3
  export.
- **CUDA out of memory** — close other GPU apps; keep fp16 on; split long text
  into shorter highlighted spans (each span renders separately); avoid running
  Demucs and synthesis back to back without letting the app unload.
- **Very slow first render** — the first Final render downloads several GB of
  IndexTTS-2 weights and warms up the model; later renders are much faster, and
  unchanged segments are served from the cache.
- **No CUDA GPU** — Chatterbox (Draft) still works on CPU; IndexTTS-2/Fish will
  be impractically slow. The status bar warns at startup.
- **Model download fails / is slow** — set a Hugging Face mirror in *Settings →
  HF endpoint* (e.g. `https://hf-mirror.com`).
- **No sound / wrong output device** — choose the device in *Settings → Output
  device*.
- **Diagnostics** — error dialogs include a *Copy diagnostics* button (recent
  log tail); the full log is at `logs/inflect.log`.

## Build phases

1. **Skeleton & document model** ✅ — span model, segmenter, project I/O, config.
2. **Ingest** ✅ — ffmpeg extract, VAD candidate selection, profiles, wizard.
3. **Chatterbox path** ✅ — engine, worker, assembly, timeline, export, GUI.
4. **IndexTTS-2** ✅ — emotion vector / emo-text / alpha / speed, fp16, swap.
5. **Polish** ✅ — per-segment re-render, tooltips, dark theme, settings, README.
6. **Fish S2 hybrid** ✅ (optional) — performance-transfer mode.

## Hybrid "performance transfer" (Phase 6)

In the Inspector, set a span's **Engine** to *Hybrid*. On synthesis the span is
rendered twice: **Fish** performs it expressively (driven by tag-translated
emotion) in its own voice, then **IndexTTS-2** re-renders your text in the
cloned voice using Fish's take as the *emotion reference*. The result carries
Fish's expressiveness in your speaker's timbre. Use *Audition performance* to
hear the Fish stage before committing. Jobs are batched by engine
(Fish → IndexTTS-2 → Chatterbox) so a mixed document needs at most ~2–3 model
swaps, and both stages are cached by hash.

Fish S2 is **research/personal-use licensed** and VRAM-hungry (24 GB is
recommended for S2); enable it in Settings and expect slow first tokens.

## License

MIT for the application code. Downloaded model weights carry their own licenses
— notably **Fish Speech is research/personal-use** unless separately licensed.
Review each model's license before any commercial use, and only clone voices you
have permission to use.
