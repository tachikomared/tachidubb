#!/usr/bin/env python
"""TachiDUBB Studio — MCP server for Claude Code and other agents.

Exposes the running TachiDUBB HTTP API as Model Context Protocol tools.
The server must be running separately (start.bat or `python server.py`);
this MCP layer is a thin wrapper that translates tool calls into HTTP.

Setup:
    pip install "mcp[cli]"

Add to Claude Code MCP config (~/.claude.json or via `claude mcp add`):
    {
      "mcpServers": {
        "tachidubb": {
          "command": "python",
          "args": ["/path/to/tachidubb/tools/tachidubb_mcp.py"],
          "env": { "TACHIDUBB_URL": "http://localhost:8910" }
        }
      }
    }

Then from Claude Code:
    "Dub https://youtu.be/abc into French and show me the result"
    → calls tachidubb_dub(source=..., target_lang="fr", wait=True)
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: mcp SDK not installed. Run: pip install \"mcp[cli]\"",
          file=sys.stderr)
    sys.exit(1)

from tachidubb_client import TachiDUBBClient, TachiDUBBError, DEFAULT_URL


mcp = FastMCP(
    "tachidubb",
    instructions=(
        "TachiDUBB Studio — local AI video dubbing. The user has a running "
        "TachiDUBB server. Use these tools to: submit dubs (one language or "
        "many), check status, build multilingual showcase reels (one video "
        "that cycles through N languages), or re-dub existing jobs without "
        "re-uploading. Always prefer `wait=True` when the user expects a "
        "finished video; otherwise return the job_id so they can poll. "
        "Source can be a YouTube/direct URL or a local file path."
    ),
)


# A single client instance is reused across tool calls. asyncio handles
# concurrency. Recreated lazily so tools work even if the server starts
# AFTER the MCP process.
_client: Optional[TachiDUBBClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> TachiDUBBClient:
    global _client
    async with _client_lock:
        if _client is None:
            _client = TachiDUBBClient(base_url=os.environ.get("TACHIDUBB_URL", DEFAULT_URL))
        return _client


# ─────────────────────────────────────────────────────────────────────
# Submission tools — start work
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def tachidubb_dub(
    source: str,
    target_lang: str,
    source_lang: str = "auto",
    model: Optional[str] = None,
    voice_preset: str = "auto",
    tts_speed: str = "balanced",
    keep_bg: bool = False,
    context_hint: str = "",
    wait: bool = False,
    wait_timeout: float = 1800.0,
) -> dict:
    """Dub one video into a single target language.

    Args:
        source: YouTube/direct URL or absolute local file path.
        target_lang: ISO code like 'fr', 'es', 'de', 'ja', 'pt', 'ru'.
        source_lang: 'auto' (default) or explicit code.
        model: Ollama translation model name. Default = server's default.
        voice_preset: Voice preset key, or 'auto' for source-cloned voice.
        tts_speed: 'fast' | 'balanced' | 'quality'.
        keep_bg: Preserve background music under the new dub.
        context_hint: Free-text hint for translator (e.g. "tech podcast").
        wait: If True, block until job finishes and return final status + url.
        wait_timeout: Max seconds to wait when wait=True.

    Returns dict with `job_id`. If wait=True, also `status` and `url`.
    """
    c = await _get_client()
    res = await c.submit_dub(
        source, target_lang,
        source_lang=source_lang, model=model,
        voice_preset=voice_preset, tts_speed=tts_speed,
        keep_bg=keep_bg, context_hint=context_hint,
    )
    job_id = res.get("job_id")
    if wait and job_id:
        final = await c.wait_for_job(job_id, timeout=wait_timeout)
        res["status"] = final.get("status")
        res["error"] = final.get("error")
        if final.get("status") == "complete":
            res["url"] = c.output_url(job_id)
    return res


@mcp.tool()
async def tachidubb_compare(
    source: str,
    target_langs: list[str],
    trim_seconds: int = 60,
    source_lang: str = "auto",
    model: Optional[str] = None,
    voice_preset: str = "auto",
    tts_speed: str = "balanced",
    keep_bg: bool = False,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> dict:
    """Quick-test mode: produce N separate dubs (one per language) for
    side-by-side comparison. Best for evaluating settings.

    target_langs: 2-6 language codes.
    trim_seconds: 15-120, default 60.
    """
    c = await _get_client()
    res = await c.submit_compare(
        source, target_langs, trim_seconds=trim_seconds,
        source_lang=source_lang, model=model,
        voice_preset=voice_preset, tts_speed=tts_speed, keep_bg=keep_bg,
    )
    if wait and res.get("batch_id"):
        jobs = await c.wait_for_batch(res["batch_id"], timeout=wait_timeout)
        res["jobs"] = [{
            "id": j.get("id"), "lang": j.get("target_lang"),
            "status": j.get("status"), "error": j.get("error"),
            "url": c.output_url(j["id"]) if j.get("status") == "complete" else None,
        } for j in jobs]
    return res


@mcp.tool()
async def tachidubb_showcase(
    source: str,
    target_langs: list[str],
    trim_seconds: int = 60,
    source_lang: str = "auto",
    model: Optional[str] = None,
    voice_preset: str = "auto",
    tts_speed: str = "balanced",
    keep_bg: bool = False,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> dict:
    """Multilingual showcase reel: N language segments stitched into ONE
    continuous video with a corner badge showing the current language.

    Use this when the user wants a single deliverable that demos voice
    cloning across languages (vs. tachidubb_compare which gives N files).

    target_langs: 2-6 codes. trim_seconds: 15-120.
    """
    c = await _get_client()
    res = await c.submit_showcase(
        source, target_langs, trim_seconds=trim_seconds,
        source_lang=source_lang, model=model,
        voice_preset=voice_preset, tts_speed=tts_speed, keep_bg=keep_bg,
    )
    if wait and res.get("batch_id"):
        bid = res["batch_id"]
        await c.wait_for_batch(bid, timeout=wait_timeout)
        info = await c.wait_for_showcase(bid, timeout=wait_timeout)
        res["showcase_status"] = info.get("status")
        res["url"] = c.showcase_url(bid)
        res["manifest"] = info.get("manifest")
    return res


@mcp.tool()
async def tachidubb_redub(
    job_id: str,
    target_langs: list[str],
    mode: str = "compare",
    model: Optional[str] = None,
    voice_preset: Optional[str] = None,
    tts_speed: Optional[str] = None,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> dict:
    """Re-dub an existing job's source into new language(s). Reuses the
    source file/URL — no upload. Inherits settings from the original.

    mode: 'single' (1 lang), 'compare' (2-6 separate dubs),
          'showcase' (2-6 stitched into one reel).
    """
    c = await _get_client()
    res = await c.redub(
        job_id, target_langs, mode=mode,
        model=model, voice_preset=voice_preset, tts_speed=tts_speed,
    )
    if wait:
        if res.get("batch_id"):
            bid = res["batch_id"]
            jobs = await c.wait_for_batch(bid, timeout=wait_timeout)
            res["jobs"] = [{
                "id": j["id"], "lang": j.get("target_lang"),
                "status": j.get("status"),
                "url": c.output_url(j["id"]) if j.get("status") == "complete" else None,
            } for j in jobs]
            if mode == "showcase":
                info = await c.wait_for_showcase(bid, timeout=wait_timeout)
                res["showcase_url"] = c.showcase_url(bid)
                res["showcase_manifest"] = info.get("manifest")
        elif res.get("job_id"):
            final = await c.wait_for_job(res["job_id"], timeout=wait_timeout)
            res["status"] = final.get("status")
            res["url"] = c.output_url(res["job_id"]) if final.get("status") == "complete" else None
    return res


# ─────────────────────────────────────────────────────────────────────
# Inspection — read-only
# ─────────────────────────────────────────────────────────────────────
@mcp.tool()
async def tachidubb_get_job(job_id: str) -> dict:
    """Get one job's full state (status, progress, error, etc.)."""
    c = await _get_client()
    j = await c.get_job(job_id)
    if j.get("status") == "complete":
        j["url"] = c.output_url(job_id)
    return j


@mcp.tool()
async def tachidubb_list_jobs(
    limit: int = 20,
    status: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> list[dict]:
    """List recent jobs, newest first.

    status: filter by 'complete'|'error'|'queued'|'running'|...
    batch_id: filter to a specific batch (quick_test, showcase, redub).
    """
    c = await _get_client()
    return await c.list_jobs(limit=limit, status=status, batch_id=batch_id)


@mcp.tool()
async def tachidubb_get_showcase(batch_id: str) -> dict:
    """Get assembly status of a showcase batch. Returns url when ready."""
    c = await _get_client()
    info = await c.get_showcase(batch_id)
    if info.get("status") == "ready":
        info["url"] = c.showcase_url(batch_id)
    return info


@mcp.tool()
async def tachidubb_rebuild_showcase(batch_id: str, wait: bool = False) -> dict:
    """Re-run the stitch step for an existing showcase batch. Useful if the
    auto-assembly failed but all per-language dubs are complete."""
    c = await _get_client()
    res = await c.rebuild_showcase(batch_id)
    if wait:
        info = await c.wait_for_showcase(batch_id, timeout=600.0)
        res["status"] = info.get("status")
        res["url"] = c.showcase_url(batch_id)
    return res


@mcp.tool()
async def tachidubb_cancel_job(job_id: str) -> dict:
    """Request cancellation of a queued or running job."""
    c = await _get_client()
    return await c.cancel_job(job_id)


@mcp.tool()
async def tachidubb_delete_job(job_id: str) -> dict:
    """Delete a job and its output files. Cannot be undone."""
    c = await _get_client()
    return await c.delete_job(job_id)


@mcp.tool()
async def tachidubb_system_status() -> dict:
    """Get server health: ffmpeg, GPU, Ollama, models, whisper, voxcpm, etc.

    Use this first if the user reports an unexpected failure — it surfaces
    missing dependencies (e.g. 'voxcpm.ok=False' or 'ollama not reachable')."""
    c = await _get_client()
    return await c.system_status()


@mcp.tool()
async def tachidubb_list_languages() -> list[str]:
    """Supported target language codes."""
    c = await _get_client()
    return await c.list_languages()


@mcp.tool()
async def tachidubb_list_models() -> list[str]:
    """Installed Ollama translation models on the user's machine."""
    c = await _get_client()
    return await c.list_models()


@mcp.tool()
async def tachidubb_list_voices() -> list[dict]:
    """Available voice presets registered in the server."""
    c = await _get_client()
    return await c.list_voices()


if __name__ == "__main__":
    # FastMCP defaults to stdio transport — exactly what Claude Code expects.
    mcp.run()
