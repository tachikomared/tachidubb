#!/usr/bin/env python
"""TachiDUBB Studio — CLI for local AI video dubbing.

Talks to a running TachiDUBB server via HTTP. Server must already be up
(start.bat or `python server.py`). Default URL = http://localhost:8910,
override with env var `TACHIDUBB_URL`.

Examples:
    # Dub a YouTube short into French
    python tools/tachidubb_cli.py dub https://youtu.be/abc --lang fr --wait

    # Re-dub job 5038e404 into 5 languages, stitched as showcase
    python tools/tachidubb_cli.py redub 5038e404 --langs es,fr,de,ja,pt --mode showcase --wait

    # Quick-test (separate side-by-side dubs)
    python tools/tachidubb_cli.py compare ./clip.mp4 --langs es,fr --trim 60

    # Show what's available
    python tools/tachidubb_cli.py system
    python tools/tachidubb_cli.py jobs --status complete
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

# Allow running from anywhere by inserting our own dir into sys.path
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# Job labels and dub output may contain non-Latin chars (Russian, CJK, …).
# Russian Windows consoles default to cp1251 which can't render them, so
# reconfigure stdout to UTF-8 with a replacement fallback. Harmless on
# UTF-8 systems.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from tachidubb_client import TachiDUBBClient, TachiDUBBError, DEFAULT_URL


# ─────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────
def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _print_job(j: dict, *, short: bool = False) -> None:
    if short:
        line = f"{j.get('id', '?')[:8]}  {j.get('status', '?'):>12}  -> {j.get('target_lang', '?'):>3}  {j.get('source_label', '')[:60]}"
        if j.get("error"):
            line += f"  err: {j['error'][:80]}"
        print(line)
    else:
        _print_json(j)


# ─────────────────────────────────────────────────────────────────────
# Subcommand handlers — each is async, takes args namespace
# ─────────────────────────────────────────────────────────────────────
async def cmd_dub(c: TachiDUBBClient, a) -> None:
    res = await c.submit_dub(
        a.source, a.lang,
        source_lang=a.source_lang, model=a.model,
        whisper_model=a.whisper, voice_preset=a.voice,
        tts_speed=a.tts_speed, speaker_mode=a.speakers,
        keep_bg=a.keep_bg, context_hint=a.context or "",
    )
    job_id = res.get("job_id")
    _print_json(res)
    if a.wait and job_id:
        print(f"[wait] polling job {job_id}…", file=sys.stderr)
        final = await c.wait_for_job(job_id, timeout=a.wait_timeout)
        if final.get("status") == "complete":
            print(f"[done] {c.output_url(job_id)}")
        else:
            print(f"[fail] status={final.get('status')} err={final.get('error', '?')}",
                  file=sys.stderr)
            sys.exit(2)


async def cmd_compare(c: TachiDUBBClient, a) -> None:
    res = await c.submit_compare(
        a.source, a.langs, trim_seconds=a.trim,
        source_lang=a.source_lang, model=a.model,
        whisper_model=a.whisper, voice_preset=a.voice,
        tts_speed=a.tts_speed, keep_bg=a.keep_bg,
    )
    _print_json(res)
    if a.wait and res.get("batch_id"):
        bid = res["batch_id"]
        print(f"[wait] polling batch {bid}…", file=sys.stderr)
        jobs = await c.wait_for_batch(bid, timeout=a.wait_timeout)
        for j in jobs:
            _print_job(j, short=True)


async def cmd_showcase(c: TachiDUBBClient, a) -> None:
    res = await c.submit_showcase(
        a.source, a.langs, trim_seconds=a.trim,
        source_lang=a.source_lang, model=a.model,
        whisper_model=a.whisper, voice_preset=a.voice,
        tts_speed=a.tts_speed, keep_bg=a.keep_bg,
    )
    _print_json(res)
    if a.wait and res.get("batch_id"):
        bid = res["batch_id"]
        print(f"[wait] dubbing {len(res.get('target_langs', []))} langs…", file=sys.stderr)
        await c.wait_for_batch(bid, timeout=a.wait_timeout)
        print(f"[wait] stitching showcase reel…", file=sys.stderr)
        info = await c.wait_for_showcase(bid, timeout=a.wait_timeout)
        print(f"[done] {c.showcase_url(bid)}")
        if a.json:
            _print_json(info)


async def cmd_redub(c: TachiDUBBClient, a) -> None:
    res = await c.redub(a.job_id, a.langs, mode=a.mode)
    _print_json(res)
    if a.wait:
        if res.get("batch_id"):
            bid = res["batch_id"]
            await c.wait_for_batch(bid, timeout=a.wait_timeout)
            if a.mode == "showcase":
                info = await c.wait_for_showcase(bid, timeout=a.wait_timeout)
                print(f"[done] {c.showcase_url(bid)}")
            else:
                jobs = await c.list_jobs(batch_id=bid, limit=10)
                for j in jobs:
                    _print_job(j, short=True)
        elif res.get("job_id"):
            final = await c.wait_for_job(res["job_id"], timeout=a.wait_timeout)
            if final.get("status") == "complete":
                print(f"[done] {c.output_url(res['job_id'])}")
            else:
                print(f"[fail] {final.get('error', '?')}", file=sys.stderr)
                sys.exit(2)


async def cmd_status(c: TachiDUBBClient, a) -> None:
    j = await c.get_job(a.job_id)
    if a.json:
        _print_json(j)
    else:
        _print_job(j)
        if j.get("status") == "complete":
            print(f"      url: {c.output_url(a.job_id)}")


async def cmd_jobs(c: TachiDUBBClient, a) -> None:
    jobs = await c.list_jobs(limit=a.limit, status=a.status, batch_id=a.batch)
    if a.json:
        _print_json(jobs)
        return
    if not jobs:
        print("(no jobs)")
        return
    for j in jobs:
        _print_job(j, short=True)


async def cmd_wait(c: TachiDUBBClient, a) -> None:
    final = await c.wait_for_job(a.job_id, timeout=a.timeout)
    if a.json:
        _print_json(final)
    else:
        _print_job(final)
    if final.get("status") != "complete":
        sys.exit(2)


async def cmd_showcase_status(c: TachiDUBBClient, a) -> None:
    info = await c.get_showcase(a.batch_id)
    _print_json(info)


async def cmd_showcase_rebuild(c: TachiDUBBClient, a) -> None:
    res = await c.rebuild_showcase(a.batch_id)
    _print_json(res)
    if a.wait:
        info = await c.wait_for_showcase(a.batch_id, timeout=a.wait_timeout)
        print(f"[done] {c.showcase_url(a.batch_id)}")


async def cmd_cancel(c: TachiDUBBClient, a) -> None:
    _print_json(await c.cancel_job(a.job_id))


async def cmd_delete(c: TachiDUBBClient, a) -> None:
    _print_json(await c.delete_job(a.job_id))


async def cmd_system(c: TachiDUBBClient, a) -> None:
    s = await c.system_status()
    if a.json:
        _print_json(s)
        return
    # Pretty-printed health summary. Use ASCII so it works in cp1251/cp866
    # consoles (Russian Windows etc.) without UnicodeEncodeError.
    def ok(d):
        return "[ok]" if (isinstance(d, dict) and d.get("ok")) else "[--]"
    print(f"{ok(s.get('python'))} python   {s.get('python', {}).get('version', '?')}")
    print(f"{ok(s.get('ffmpeg'))} ffmpeg")
    print(f"{ok(s.get('gpu'))} gpu      {s.get('gpu', {}).get('name', '?')} "
          f"({s.get('gpu', {}).get('vram_gb', 0)} GB)")
    print(f"{ok(s.get('whisper'))} whisper  {s.get('whisper', {}).get('type', '?')}")
    print(f"{ok(s.get('voxcpm'))} voxcpm")
    print(f"{ok(s.get('demucs'))} demucs")
    ollama = s.get("ollama") or s.get("ollama_binary") or {}
    print(f"{ok(ollama)} ollama")
    models = ollama.get("models", []) if isinstance(ollama, dict) else []
    if models:
        names = [m if isinstance(m, str) else m.get("name", "") for m in models]
        print(f"          models: {', '.join(names[:6])}{'...' if len(names) > 6 else ''}")


async def cmd_languages(c: TachiDUBBClient, a) -> None:
    print(",".join(await c.list_languages()))


async def cmd_models(c: TachiDUBBClient, a) -> None:
    for m in await c.list_models():
        print(m)


async def cmd_voices(c: TachiDUBBClient, a) -> None:
    for v in await c.list_voices():
        if isinstance(v, dict):
            print(f"{v.get('name', '?'):<20} {v.get('lang', '?'):<6} {v.get('description', '')}")
        else:
            print(v)


# ─────────────────────────────────────────────────────────────────────
# argparse setup
# ─────────────────────────────────────────────────────────────────────
def _add_common_dub_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source-lang", default="auto", help="Source language code or 'auto'")
    p.add_argument("--model", help="Ollama translation model (e.g. aya-expanse:8b)")
    p.add_argument("--whisper", default="large-v3", help="Whisper model size")
    p.add_argument("--voice", default="auto", help="Voice preset name")
    p.add_argument("--tts-speed", default="balanced",
                   choices=["fast", "balanced", "quality"])
    p.add_argument("--keep-bg", action="store_true", help="Keep background music")
    p.add_argument("--wait", action="store_true", help="Block until job(s) finish")
    p.add_argument("--wait-timeout", type=float, default=1800.0,
                   help="Seconds before --wait gives up (default 1800)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tachidubb",
        description="Command-line interface for TachiDUBB Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Server URL: env TACHIDUBB_URL (default %s)" % DEFAULT_URL,
    )
    p.add_argument("--url", default=DEFAULT_URL, help="Override server URL for this call")
    sub = p.add_subparsers(dest="cmd", required=True)

    # dub
    s = sub.add_parser("dub", help="Submit a single-language dub")
    s.add_argument("source", help="Video URL or local file path")
    s.add_argument("--lang", required=True, help="Target language code (e.g. fr, es, de)")
    s.add_argument("--context", help="Optional translator hint (genre, tone)")
    s.add_argument("--speakers", default="main", choices=["main", "all", "single"])
    _add_common_dub_opts(s)
    s.set_defaults(handler=cmd_dub)

    # compare (quick test)
    s = sub.add_parser("compare", help="N separate dubs side-by-side (Quick Test)")
    s.add_argument("source")
    s.add_argument("--langs", required=True, help="Comma-separated 2-6 codes")
    s.add_argument("--trim", type=int, default=60, help="Trim seconds (15-120)")
    _add_common_dub_opts(s)
    s.set_defaults(handler=cmd_compare)

    # showcase
    s = sub.add_parser("showcase", help="Multilingual stitched reel (one video, N segments)")
    s.add_argument("source")
    s.add_argument("--langs", required=True, help="Comma-separated 2-6 codes")
    s.add_argument("--trim", type=int, default=60)
    s.add_argument("--json", action="store_true")
    _add_common_dub_opts(s)
    s.set_defaults(handler=cmd_showcase)

    # redub
    s = sub.add_parser("redub", help="Re-dub an existing job's source in new languages")
    s.add_argument("job_id", help="Original job ID")
    s.add_argument("--langs", required=True, help="Comma-separated target languages")
    s.add_argument("--mode", default="compare", choices=["single", "compare", "showcase"])
    s.add_argument("--wait", action="store_true")
    s.add_argument("--wait-timeout", type=float, default=1800.0)
    s.set_defaults(handler=cmd_redub)

    # status / jobs / wait
    s = sub.add_parser("status", help="Show one job's status")
    s.add_argument("job_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_status)

    s = sub.add_parser("jobs", help="List jobs (default: 50 newest)")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--status", help="Filter by status (queued/complete/error/...)")
    s.add_argument("--batch", help="Filter by batch_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_jobs)

    s = sub.add_parser("wait", help="Block until a job finishes")
    s.add_argument("job_id")
    s.add_argument("--timeout", type=float, default=1800.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_wait)

    # showcase utils
    s = sub.add_parser("showcase-status", help="Showcase assembly status by batch_id")
    s.add_argument("batch_id")
    s.set_defaults(handler=cmd_showcase_status)

    s = sub.add_parser("showcase-rebuild", help="Re-run showcase stitch step")
    s.add_argument("batch_id")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--wait-timeout", type=float, default=600.0)
    s.set_defaults(handler=cmd_showcase_rebuild)

    # admin
    s = sub.add_parser("cancel", help="Cancel a running job"); s.add_argument("job_id"); s.set_defaults(handler=cmd_cancel)
    s = sub.add_parser("delete", help="Delete a job and its files"); s.add_argument("job_id"); s.set_defaults(handler=cmd_delete)
    s = sub.add_parser("system", help="System health check"); s.add_argument("--json", action="store_true"); s.set_defaults(handler=cmd_system)
    s = sub.add_parser("languages", help="List supported language codes"); s.set_defaults(handler=cmd_languages)
    s = sub.add_parser("models", help="List installed Ollama models"); s.set_defaults(handler=cmd_models)
    s = sub.add_parser("voices", help="List voice presets"); s.set_defaults(handler=cmd_voices)
    return p


async def _main_async(args) -> int:
    async with TachiDUBBClient(base_url=args.url) as c:
        try:
            await args.handler(c, args)
            return 0
        except TachiDUBBError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
