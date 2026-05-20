<div align="center">

# 🎙️ TachiDUBB Studio

**Local, agent-controllable AI video dubbing.**
YouTube link in → voice-cloned dub in 28 languages out. No cloud, no per-minute fees, no upload of your face to anyone's server.

*by [@smolekoma](https://x.com/smolekoma) and [@smolemaru](https://x.com/smolemaru) &mdash; built with [Claude](https://claude.ai)*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0+-76B900.svg)](https://developer.nvidia.com/cuda-downloads)
[![MCP enabled](https://img.shields.io/badge/MCP-enabled-7B61FF.svg)](https://modelcontextprotocol.io)
[![GitHub stars](https://img.shields.io/github/stars/TachikomaRed/tachidubb?style=social)](https://github.com/TachikomaRed/tachidubb/stargazers)

[**Quickstart**](#-30-second-quickstart) ·
[**Demo**](#-demo) ·
[**MCP / Agent use**](#-agent-control-mcp--cli) ·
[**Languages**](#-supported-languages) ·
[**FAQ**](#-faq) ·
[**Troubleshooting**](#-troubleshooting)

![demo](docs/demo.gif)

</div>

---

## ✨ Why TachiDUBB

| | TachiDUBB | ElevenLabs Dubbing | Heygen | Rask |
|---|---|---|---|---|
| **Cost** | Free (your GPU) | $0.30/min and up | $0.15+/min | $0.07+/min |
| **Runs offline** | ✅ 100% local | ❌ cloud | ❌ cloud | ❌ cloud |
| **Voice cloning** | ✅ VoxCPM2 | ✅ | ✅ | ✅ |
| **Languages** | 28 | 29 | 40+ | 130+ |
| **Multi-speaker diarization** | ✅ (pyannote) | ✅ | ✅ | ✅ |
| **Background music preservation** | ✅ (audio-separator) | ✅ | ✅ | ✅ |
| **YouTube URL → MP4** | ✅ in one step | ❌ | ❌ | ❌ |
| **Stitched multilingual reel** | ✅ built-in | ❌ | ❌ | ❌ |
| **MCP / agent control** | ✅ first-class | ❌ | ❌ | ❌ |
| **Open source** | ✅ MIT | ❌ | ❌ | ❌ |
| **No upload of your data** | ✅ | ❌ | ❌ | ❌ |
| **API key required** | ❌ none | ✅ paid | ✅ paid | ✅ paid |

If you're dubbing a 10-minute video weekly across 5 languages, this saves you about **$1,800/year** vs cloud tools — and the dub never leaves your machine.

---

## 🚀 30-second quickstart

### Windows (one click)

```text
1. Clone or unzip the repo
2. Double-click install.bat   ← installs everything (~5-10 min)
3. Double-click start.bat     ← browser opens at http://localhost:8910
4. Paste YouTube URL → pick language → Start
```

### Linux / macOS

```bash
git clone https://github.com/TachikomaRed/tachidubb && cd tachidubb
chmod +x install.sh
./install.sh    # installs everything + creates start.sh
./start.sh
```

First dubbing run downloads the VoxCPM2 model (~5 GB) — one time.

---

## 🤖 Agent control (MCP + CLI)

This is what makes TachiDUBB different. You don't have to touch the UI to use it.

### Tell Claude Code (or any MCP-aware agent) what you want

```text
You:    Dub https://youtu.be/abc into French, Spanish and Japanese,
        then stitch them into one 60-second showcase reel.

Claude: [calls tachidubb_showcase(...)]
        [polls tachidubb_get_showcase(...)]
        Done — http://localhost:8910/outputs/showcase_sc_2f1a.../showcase.mp4
```

Add the MCP server in 10 seconds:

```bash
claude mcp add tachidubb python /path/to/tachidubb/tools/tachidubb_mcp.py
```

Or paste into `~/.claude.json`:

```json
{
  "mcpServers": {
    "tachidubb": {
      "command": "/path/to/tachidubb/venv/Scripts/python.exe",
      "args": ["/path/to/tachidubb/tools/tachidubb_mcp.py"],
      "env": { "TACHIDUBB_URL": "http://localhost:8910" }
    }
  }
}
```

The repo ships a Claude Code skill at [`.claude/skills/tachidubb/SKILL.md`](.claude/skills/tachidubb/SKILL.md). Copy it to `~/.claude/skills/` and Claude knows when and how to drive the pipeline.

### CLI — works from any shell, any OS, any cron

```bash
# Single language, blocking
python tools/tachidubb_cli.py dub https://youtu.be/abc --lang fr --wait

# Compare 5 languages side-by-side
python tools/tachidubb_cli.py compare ./clip.mp4 --langs es,fr,de,ja,pt --trim 60

# Stitched multilingual showcase reel
python tools/tachidubb_cli.py showcase https://youtu.be/abc \
  --langs es,fr,de,ja,pt --trim 60 --wait

# Re-dub an existing job into new languages — skips re-upload
python tools/tachidubb_cli.py redub 5038e404 --langs ja,it --mode showcase --wait

# Health, status, history
python tools/tachidubb_cli.py system
python tools/tachidubb_cli.py jobs --limit 20
python tools/tachidubb_cli.py status <job_id>
```

Drive a remote box: `set TACHIDUBB_URL=http://192.168.0.10:8910`

See [`examples/`](examples/) for ready-to-run scripts.

---

## 🎬 Demo

| What | Length | Languages | Time on RTX 3080 Ti |
|---|---|---|---|
| Single-speaker YouTube short → French | 60 s | 1 | ~2 min |
| Compare 5 languages | 60 s × 5 | 5 | ~10-15 min |
| Showcase reel (stitched) | 60 s | 5 | ~12-18 min |
| Multi-speaker podcast (diarized) | 5 min | 1 | ~8-10 min |

> 📺 [Watch the full demo](docs/demo.mp4) (no audio, ~2 min) — submit a YouTube URL, pick 5 languages, get a stitched showcase reel.

---

## 🏗️ How it works

```
YouTube URL or local file
        │
        ▼
   yt-dlp ───────────────────────► (downloads source)
        │
        ▼
   FFmpeg ───────────────────────► (extracts audio)
        │
        ▼
  faster-whisper ───────────────► (transcript + word timestamps)
        │
        ▼
   pyannote ─────────────────────► (speaker diarization, optional)
        │
        ▼
   Ollama (Qwen3 / Gemma3 / Aya) ► (translation, length-matched)
        │
        ▼
   VoxCPM2 ──────────────────────► (voice cloning per speaker, 48 kHz)
        │
        ▼
   FFmpeg ───────────────────────► (time-align, mix bg music, render)
        │
        ▼
   Dubbed MP4 + SRT subtitles
```

Every step is modular, swappable, and runs on your hardware.

---

## 🌍 Supported languages

28 target languages out of the box (via VoxCPM2 + edge-tts fallback):

| Code | Language |     | Code | Language |     | Code | Language |     | Code | Language |
|---|---|---|---|---|---|---|---|---|---|---|
| `en` | English | | `ru` | Russian | | `es` | Spanish | | `fr` | French |
| `de` | German | | `it` | Italian | | `pt` | Portuguese | | `pl` | Polish |
| `tr` | Turkish | | `ja` | Japanese | | `ko` | Korean | | `zh` | Chinese |
| `ar` | Arabic | | `hi` | Hindi | | `nl` | Dutch | | `uk` | Ukrainian |
| `sv` | Swedish | | `th` | Thai | | `vi` | Vietnamese | | `cs` | Czech |
| `ro` | Romanian | | `hu` | Hungarian | | `bg` | Bulgarian | | `el` | Greek |
| `fi` | Finnish | | `id` | Indonesian | | `no` | Norwegian | | `da` | Danish |

Source detection is automatic (Whisper). Translation goes through whatever Ollama model you have — `aya-expanse:8b` is the default for best multilingual quality.

---

## 🖥️ Hardware

| | Minimum | Recommended | Why |
|---|---|---|---|
| **VRAM** | 8 GB | 12 GB+ | VoxCPM2 + Whisper + a translation LLM coexist |
| **RAM** | 16 GB | 32 GB | Audio-separator (background preservation) is hungry |
| **Disk** | 20 GB | 40 GB+ | Models + outputs |
| **GPU** | Any CUDA 12.0+ | RTX 30/40 series | CPU fallback works but ~15× slower |
| **Python** | 3.10–3.12 | 3.11 | |
| **OS** | Win 10+, Linux, macOS | — | macOS requires CPU mode |

No GPU? It still runs — just expect long jobs. The pipeline auto-falls back to `edge-tts` (Microsoft cloud TTS) if VoxCPM2 won't load, which sacrifices voice cloning but produces intelligible output fast.

### Disk budget (what gets downloaded)

| Component | Size | When |
|---|---|---|
| Python deps (PyTorch + transformers + faster-whisper + ...) | ~4 GB | At `install.bat` / `./install.sh` |
| FFmpeg + yt-dlp (Windows static build) | ~100 MB | At install |
| VoxCPM2 model weights | ~5 GB | First dubbing run, cached forever |
| Whisper `large-v3` weights | ~3 GB | First dubbing run, cached forever |
| Ollama translation model (e.g. `qwen3:8b`) | ~5 GB | At install (you pick it) |
| pyannote diarization weights (optional) | ~500 MB | First multi-speaker run |
| audio-separator UVR weights (optional) | ~250 MB | First background-preserve run |

**Total for full setup: ~18 GB.** Skinny single-language setup without diarization or BGM preservation: ~12 GB.

---

## 🔑 Tokens & API keys

**Required tokens: NONE.** The default install runs 100% offline once dependencies are downloaded. No OpenAI / ElevenLabs / Anthropic key needed — translation is local (Ollama), TTS is local (VoxCPM2), ASR is local (Whisper).

| Token | Required? | What for | Where to get |
|---|---|---|---|
| Hugging Face token (`HF_TOKEN`) | Only for multi-speaker diarization | Downloading pyannote diarization weights — gated by free terms-of-use acceptance | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — also accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| YouTube cookies (`YT_DLP_COOKIES_FROM_BROWSER`) | Only for age-restricted / member-only YouTube videos | yt-dlp downloads via your existing browser session | Auto — set to `chrome`, `firefox`, `edge` etc. |
| OpenAI / ElevenLabs / Anthropic keys | **Never.** | — | — |

What "phones home" by default:
- `yt-dlp` reaches YouTube/Vimeo/etc. — only when you submit a URL
- `huggingface.co` for model downloads — first run only, then cached
- `ollama.com` for translation model pulls — first install only
- `edge-tts` for the cloud TTS fallback — only triggers if VoxCPM2 fails to load on your GPU

There's no telemetry, no analytics, no phone-home from TachiDUBB itself. Audit the network calls: search the repo for `httpx.` / `requests.` — only the integrations above.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
# Speaker diarization (multi-speaker videos)
HF_TOKEN=hf_xxxxx                  # from huggingface.co/settings/tokens

# TTS model selection
VOXCPM_MODEL=openbmb/VoxCPM2       # or openbmb/VoxCPM1.5 (lighter)
VOXCPM_CFG=2.0                     # 1.5-3.0, higher = closer to reference voice
VOXCPM_STEPS=10                    # 5-20, lower = faster

# Translation backend
OLLAMA_URL=http://localhost:11434

# UI behavior
TACHIDUBB_OPEN_BROWSER=1           # 0 to disable auto-open
TACHIDUBB_QA_THRESHOLD=0.4         # stricter (lower) = more re-rolls on bad TTS
```

### Optional dependencies

| Feature | Install | Notes |
|---|---|---|
| Multi-speaker diarization | `pip install pyannote.audio` + HF token | Auto-detects N speakers, clones each |
| Background music preservation | `pip install audio-separator` | Demuxes vocals, keeps original BGM |
| Faster Whisper on GPU | (already in requirements) | If CUDA isn't found, falls back to CPU |

---

## 🧠 The agent skill

If you use Claude Code, copy `.claude/skills/tachidubb/SKILL.md` into your global skills folder (`~/.claude/skills/tachidubb/`). After that, just say:

- *"Dub this YouTube short into French and German"*
- *"Make a showcase reel of this clip in 5 languages"*
- *"Re-dub job 5038e404 into Japanese and Italian"*
- *"What's the status of my dub?"*

The skill teaches Claude which tool to call, what arguments to use, how to poll, how to recover from errors, and when to suggest a comparison vs a showcase. Read [`SKILL.md`](.claude/skills/tachidubb/SKILL.md) for the full trigger map.

Works with any MCP-compatible agent — Cursor, Cline, Continue, custom agents. The MCP tool schema is auto-discovered.

---

## 🛟 Troubleshooting

<details>
<summary><b>Ollama shows a red dot in the UI</b></summary>

Run `ollama serve` in a separate terminal, or restart the app — `start.bat` auto-starts Ollama. If you've never installed Ollama, the System panel has an install button.

</details>

<details>
<summary><b>Ollama has no models installed</b></summary>

Open the System tab → Models → click "Install" on `aya-expanse:8b` (best multilingual, ~5 GB) or `qwen3:8b` (good general, ~5 GB). Or from CLI: `ollama pull aya-expanse:8b`.

</details>

<details>
<summary><b>YouTube download fails / SSL error</b></summary>

Update yt-dlp: `venv\Scripts\activate && pip install -U yt-dlp`. If it's an age-restricted or region-blocked video, set `YT_DLP_COOKIES_FROM_BROWSER=chrome` in `.env`. For SSL errors, check firewall/VPN/corporate proxy.

</details>

<details>
<summary><b>VoxCPM2 runs out of VRAM</b></summary>

Three knobs, easiest first:

1. System tab → switch Whisper to `small` (frees ~3 GB)
2. `.env` → `VOXCPM_STEPS=6` (faster, less VRAM)
3. `.env` → `VOXCPM_MODEL=openbmb/VoxCPM1.5` (smaller model, slight quality drop)

</details>

<details>
<summary><b>Voice sounds like two different people mid-video</b></summary>

This was a real bug we fixed: in cross-lingual cloning, QA retries were mutating the random seed mid-job, producing different timbres for failed-then-retried segments. Make sure you're on the latest commit — the fix is in `pipeline/tts_worker.py`.

If you still hit it: try `VOXCPM_CFG=2.5` (more reference-anchored) or upload a longer, cleaner reference voice in the speaker tab.

</details>

<details>
<summary><b>First VoxCPM2 call is slow</b></summary>

Normal. The model downloads ~5 GB on first use; progress is in the terminal. Subsequent runs use the cached weights.

</details>

<details>
<summary><b>Hugging Face 401 / "access denied"</b></summary>

You need to (1) create a token at https://huggingface.co/settings/tokens, (2) accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1 (and https://huggingface.co/pyannote/segmentation-3.0), (3) put `HF_TOKEN=hf_…` in `.env`.

</details>

<details>
<summary><b>No GPU detected even though I have one</b></summary>

Verify CUDA is visible: `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, reinstall PyTorch matching your CUDA — see https://pytorch.org/get-started/locally/. On Windows make sure you're using the venv Python, not the system one.

</details>

<details>
<summary><b>Audio is out of sync with video</b></summary>

Usually a duration-mismatch in translation (target language is much longer/shorter than source). The pipeline time-aligns automatically, but extreme cases (German → Japanese, etc.) can drift. Try:

- Translation prompt is length-aware by default — make sure you didn't disable it in the UI
- Use a higher-quality translation model (`qwen3:14b` if you have the VRAM)
- For very long videos, dub in 2-3 minute chunks

</details>

<details>
<summary><b>FFmpeg not found</b></summary>

Linux/macOS: `sudo apt install ffmpeg` or `brew install ffmpeg`. Windows: the installer downloads a static build into `bin/` automatically — if it failed, re-run `install.bat`.

</details>

<details>
<summary><b>Showcase reel renders all black / no audio</b></summary>

Usually one of the child dubs failed silently. `python tools/tachidubb_cli.py showcase-status <batch_id>` shows which language failed. Rerun with `tachidubb showcase-rebuild <batch_id>` after fixing the failing job — it skips re-dubbing the successful ones.

</details>

<details>
<summary><b>Background-preserve toggle does nothing</b></summary>

Install the optional dep: `pip install audio-separator`. The UI shows a yellow warning if it's missing. First demux is slow (~30 s on GPU); subsequent ones are cached.

</details>

<details>
<summary><b>Linux ALSA / pulse errors during TTS</b></summary>

We don't play audio — these are warnings from a transitive dep. Ignore unless they actually break the run. `export ALSA_CARD=-1` silences them.

</details>

<details>
<summary><b>The server is on a different machine — how do I point the CLI at it?</b></summary>

`export TACHIDUBB_URL=http://192.168.0.10:8910` (or set `TACHIDUBB_URL` in your MCP config `env` block). The CLI and MCP server respect the same variable.

</details>

<details>
<summary><b>How do I run it headless / on a server?</b></summary>

`python server.py --host 0.0.0.0 --port 8910` and point your browser (or CLI / MCP) at it. Make sure port 8910 is accessible. There's no auth out of the box — put it behind nginx/Tailscale/Cloudflare Tunnel if exposed publicly.

</details>

---

## ❓ FAQ

**Is this really free?**
Yes. MIT licensed. The only "cost" is your electricity and GPU. No telemetry, no phone-home.

**Do I need an NVIDIA GPU?**
For reasonable speeds, yes. CPU works but a 1-minute dub takes ~30 minutes instead of ~2.

**Does it work on Apple Silicon (M1/M2/M3)?**
Yes via CPU + MPS fallback. Expect about 4-8× slower than a discrete GPU. PyTorch MPS support for VoxCPM2 is experimental — `edge-tts` fallback is reliable.

**Can I voice-clone a specific person?**
Yes — drop a 5-30 second clean WAV/MP3 into `presets/voices/` and pick it as the reference. Please don't do this without that person's consent. See [SECURITY.md](SECURITY.md).

**What's the quality vs ElevenLabs?**
On clean source audio, VoxCPM2 is genuinely close. On noisy / multi-speaker content, ElevenLabs still wins (their diarization is better). For 95% of one-speaker YouTube content, you won't tell the difference.

**Does it preserve emotion / tone?**
Partially. VoxCPM2 picks up energy and pacing from the reference. It doesn't model fine emotion the way some closed models do. If the source is a calm explainer, the dub is calm; if it's a hype reel, the dub is hype.

**Can I run multiple dubs in parallel?**
The server queues GPU work serially (one VoxCPM2 invocation at a time) to avoid OOM. CPU stages (download, transcribe with CPU Whisper, ffmpeg) overlap automatically.

**Does it work for animated content / games / non-real voices?**
Yes — anything VoxCPM2 can fit as a reference (usually 5+ s of clean speech) clones fine. Singing is not supported.

**Why VoxCPM2 instead of XTTS / OpenVoice / F5-TTS?**
VoxCPM2 has the best cross-lingual cloning quality we tested at the 5 GB weight class. The architecture is swappable — `pipeline/synthesizer.py` has a base class; PRs for other backends welcome.

**Can agents trigger this without my approval?**
Each MCP tool call requires user confirmation by default (per the MCP spec). Tachidubb doesn't bypass that.

---

## 🗺️ Roadmap

- [x] MCP server + CLI
- [x] Stitched multilingual showcase reels
- [x] Multi-speaker diarization
- [x] Background music preservation
- [x] Deterministic voice across cross-lingual segments
- [ ] Subtitle burn-in toggle (currently SRT sidecar only)
- [ ] Speaker labelling UI (assign names to detected speakers)
- [ ] Browser-only mode (no Ollama dependency, use llama.cpp WASM)
- [ ] Batch processing folder watcher
- [ ] Docker image with everything pre-baked
- [ ] Hardware-accelerated diarization (NVIDIA NeMo)
- [ ] Apple Silicon MLX backend

Vote / suggest features in [Discussions](https://github.com/TachikomaRed/tachidubb/discussions).

---

## 🛡️ Responsible use

Voice cloning is powerful and easily misused. **TachiDUBB is built for legitimate creators dubbing their own content or content they have rights to.** Please:

- Don't clone someone's voice without their explicit, informed consent.
- Don't impersonate real people (politicians, celebrities, your boss) for deception, fraud, or harassment.
- Disclose AI-generated speech when publishing — most platforms now require this, and it's the right thing to do.
- Comply with your local laws on synthetic media (EU AI Act, US state laws, etc.).

We refuse to add features that defeat watermarking, anti-cloning safeguards, or platform AI-disclosure requirements. See [SECURITY.md](SECURITY.md) for the threat model and how to report abuse.

---

## 🤝 Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and the modular pipeline design — most contributions are a single drop-in file in `pipeline/`.

Good first issues:
- Add a TTS backend (XTTS, F5-TTS, OpenVoice)
- Add a translation backend (OpenAI-compatible HTTP, vLLM, mlx_lm)
- New language voices in the edge-tts fallback map
- Improve the duration-matching prompt for hard language pairs

---

## 💖 Credits

Built by **[TachikomaRed](https://x.com/smolekoma)** and **[smolemaru](https://x.com/smolemaru)** &mdash; in collaboration with **[Claude](https://claude.ai)** (Anthropic).

Follow the build on X: [@smolekoma](https://x.com/smolekoma) &middot; [@smolemaru](https://x.com/smolemaru)

Standing on shoulders:
- [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — voice cloning TTS (Apache-2.0)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — ASR (MIT)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — diarization (MIT)
- [Ollama](https://ollama.com) — local LLM serving (MIT)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — universal downloader (Unlicense)
- [edge-tts](https://github.com/rany2/edge-tts) — cloud TTS fallback (GPL-3.0)
- [audio-separator](https://github.com/karaokenerds/python-audio-separator) — stem separation (MIT)
- [Model Context Protocol](https://modelcontextprotocol.io) — agent integration (Anthropic)

## 📜 License

MIT — see [LICENSE](LICENSE). VoxCPM2 is Apache-2.0. edge-tts is GPL-3.0; using it doesn't require this project to be GPL because it's a runtime dependency invoked as a process.

---

<div align="center">

**If TachiDUBB saved you a Heygen subscription, smash that ⭐ — that's how more people find it.**

[![Star History Chart](https://api.star-history.com/svg?repos=TachikomaRed/tachidubb&type=Date)](https://star-history.com/#TachikomaRed/tachidubb&Date)

</div>
