# Contributing to TachiDUBB

Thanks for thinking about contributing — this is a small project and every PR makes a real difference.

## Quick orientation

The pipeline is intentionally modular. Each stage in `pipeline/` is a single self-contained file. Most contributions are a drop-in replacement or a new backend behind an existing base class.

```
pipeline/
├── downloader.py      # URL → local file (yt-dlp)
├── audio.py           # video → 16/48 kHz mono WAV (ffmpeg)
├── transcriber.py     # WAV → segments + word timestamps (whisper)
├── diarizer.py        # WAV → speaker turns (pyannote, optional)
├── translator.py      # source text → target text (Ollama)
├── synthesizer.py     # text → speech (VoxCPM2, edge-tts fallback)
├── tts_qa.py          # post-TTS quality check (whisper roundtrip)
├── tts_worker.py      # synthesis orchestration + retry policy
├── segment_post.py    # post-synthesis processing (loudness, pace)
├── vad.py             # silence trimming
├── assembler.py       # time-align dubbed segments + render
└── models.py          # system checks + model catalog
```

## Setup

```bash
git clone https://github.com/TachikomaRed/tachidubb
cd tachidubb
./install.sh           # or install.bat on Windows
source venv/bin/activate
pip install -r requirements.txt
```

For development you'll usually also want:

```bash
pip install ruff pytest
```

Run the server in dev mode:

```bash
python server.py --reload
```

## Good first PRs

- **New TTS backend** — add a class in `pipeline/synthesizer.py` next to the existing `VoxCPM2Synth` / `EdgeTTSSynth`. Implement `load()`, `unload()`, `synth(text, ref_wav, lang) -> np.ndarray`. Register it in the synthesizer factory.
- **New translation backend** — same shape, in `pipeline/translator.py`. We support Ollama; an OpenAI-compatible HTTP backend or vLLM would be welcome.
- **Voice presets** — add reference WAVs to `presets/voices/` (only with the original speaker's consent + a `LICENSE.txt` next to the file).
- **Fix a troubleshooting item** — anything in the README's Troubleshooting section we say is hard is fair game to make easier.

## Code style

- **Python 3.10+**, type hints where they help reading
- **Run `ruff check .`** before opening a PR — CI runs the same
- **No global state** added to existing modules
- **Logger names**: use `logging.getLogger("tachidubb.<module>")`, no print statements in pipeline code
- **No new dependencies** without discussing in an issue first — we prefer fewer, well-tested packages over many fashionable ones

## Tests

There's no full test suite yet (most of the project is integration-tested by running it). For new pure-functions, please add a small `tests/test_<module>.py` using pytest. For pipeline changes, a screenshot or short clip of the result attached to the PR is the most useful evidence.

## Commit style

- Imperative subject ("fix QA retry seed mutation", not "fixed" or "fixing")
- One concern per commit when reasonable
- Reference the issue: `Fixes #123`
- We don't sign commits — but if you do, that's fine

## Pull requests

1. Open an issue first for anything non-trivial — saves you wasted work if we'd reject it
2. Branch from `main`
3. Keep the diff focused — don't bundle a refactor with a feature
4. Update the README if user-visible behavior changes
5. Be patient — this is maintained by two people in evenings

## What we won't merge

- Features that defeat AI-disclosure / watermarking / consent guardrails (see [SECURITY.md](SECURITY.md))
- Cloud-only integrations that don't preserve a local-only mode
- Telemetry / phone-home of any kind
- Anything that requires accepting non-OSS license terms by default

## Code of conduct

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md). It's the Contributor Covenant. Be kind. Be specific. Receipts beat opinions.
