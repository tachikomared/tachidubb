# Security & Responsible Use

## Reporting vulnerabilities

If you find a security issue (code execution, path traversal, SSRF, auth bypass, exfiltration of local files via the API, etc.), please **do not open a public issue**.

Open a private security advisory on GitHub:
https://github.com/TachikomaRed/tachidubb/security/advisories/new

Or DM us on X: [@smolekoma](https://x.com/smolekoma) / [@smolemaru](https://x.com/smolemaru).

We aim to acknowledge within 72 hours and ship a fix within 14 days for critical issues.

## Threat model

TachiDUBB is designed to run on a **trusted local network**. Specifically:

| Scenario | Supported? |
|---|---|
| Single user on their own machine | ✅ Yes |
| Trusted home / office LAN behind a firewall | ✅ Yes |
| Behind a reverse proxy with auth (nginx + basic, Tailscale, Cloudflare Tunnel) | ✅ Yes |
| Public internet without auth | ❌ No — the API has no built-in authentication |
| Multi-tenant SaaS hosting | ❌ No — jobs share an output directory |

If you expose port `8910` to the public internet, anyone can submit dubbing jobs that use your GPU, download arbitrary YouTube content under your IP, and read all past outputs. Put it behind auth.

## What the server does and doesn't do

✅ It will:
- Download URLs via yt-dlp
- Read files you upload via `/api/upload`
- Run ffmpeg, Whisper, Ollama, pyannote, VoxCPM2 on its own machine
- Write all output to the project's `outputs/` directory

❌ It will not:
- Send any data to a remote server (other than yt-dlp downloads, Hugging Face model downloads if missing, Ollama model pulls, and edge-tts if the VoxCPM2 fallback triggers — all initiated by you)
- Phone home with telemetry
- Open ports other than the one you tell it to

## Voice cloning ethics

This is the more important section. Voice cloning at this quality is a dual-use technology.

### What this tool is for

- Creators dubbing their own content into other languages
- Studios with explicit voice-actor consent and a contract
- Researchers, language learners, accessibility work (audiobook generation, etc.)
- Educators dubbing their own lectures

### What this tool is not for

- Impersonating real people without their consent
- Fraud, scams, deceptive robocalls
- Non-consensual sexual content
- Election / political misinformation
- Defeating platform AI-disclosure requirements
- Bypassing voice authentication on banks, phones, or any account

### Guardrails we keep in the codebase

- The pipeline does not strip or block AI-disclosure metadata in output files
- We will not merge PRs that defeat watermarking or platform disclosure requirements
- We will not add a "remove watermark" feature
- Reference voices in `presets/voices/` should ship with a `LICENSE.txt` documenting consent

### Guardrails we ask of you

- Get written, informed consent before cloning someone's voice
- Disclose AI-generated speech when publishing — YouTube, TikTok, Instagram, Meta, and X all now require this
- Don't use this to create content of minors
- Comply with the EU AI Act (if you're in the EU), US state laws (Tennessee ELVIS Act, California AB 730, etc.), and any other jurisdiction you operate in

### If you see abuse

If you encounter content created with TachiDUBB being used to impersonate, defraud, or harass someone, please report:

- To the platform hosting the content (DMCA / abuse reports)
- To us via a GitHub security advisory or X DM ([@smolekoma](https://x.com/smolekoma) / [@smolemaru](https://x.com/smolemaru)) — we'll publicly document misuse patterns to help defenders
- To law enforcement if it constitutes a crime in your jurisdiction

We can't prevent every misuse — the model weights are downloadable from Hugging Face independent of this UI — but we will not make abuse easier, and we will spotlight it when it happens.
