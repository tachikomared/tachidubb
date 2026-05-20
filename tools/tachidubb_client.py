"""TachiDUBB Studio — async HTTP client shared by the CLI and MCP server.

Wraps the FastAPI endpoints in `server.py` with a typed Python interface.
Both `tachidubb_cli.py` and `tachidubb_mcp.py` use this so we keep the
HTTP contract in one place.

Server URL is configurable via env var `TACHIDUBB_URL` (default
http://localhost:8910). The TachiDUBB server must already be running —
this client does not start it.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx


DEFAULT_URL = os.environ.get("TACHIDUBB_URL", "http://localhost:8910")

# Active statuses — used by wait_for_completion to know when to stop polling.
_ACTIVE = {
    "queued", "scheduled", "running", "downloading", "extracting",
    "transcribing", "translating", "synthesizing", "assembling", "merging",
}
_TERMINAL = {"complete", "error", "cancelled"}


class TachiDUBBError(RuntimeError):
    """Raised when the server returns a non-2xx response or a JSON error field."""


class TachiDUBBClient:
    """Async client. Use as `async with TachiDUBBClient() as c:` or call
    `await c.aclose()` manually."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "TachiDUBBClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── low-level helpers ────────────────────────────────────────────
    @staticmethod
    def _is_url(s: str) -> bool:
        return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

    async def _request(self, method: str, path: str, **kw) -> dict:
        r = await self._http.request(method, f"{self.base_url}{path}", **kw)
        try:
            data = r.json()
        except Exception:
            r.raise_for_status()
            return {}
        if r.status_code >= 400:
            err = data.get("error") or data.get("detail") or f"HTTP {r.status_code}"
            raise TachiDUBBError(err)
        return data

    @staticmethod
    def _source_fields(source: str) -> tuple[Optional[dict], dict]:
        """Build (files, form) tuple from a source path or URL.

        URLs go into the `source` form field (yt-dlp downloads them).
        Local file paths become a multipart `video` upload.
        """
        if TachiDUBBClient._is_url(source):
            return None, {"source": source}
        p = Path(source).expanduser().resolve()
        if not p.exists():
            raise TachiDUBBError(f"Source file not found: {p}")
        # httpx will close the file handle when the request finishes
        f = open(p, "rb")
        return {"video": (p.name, f, "application/octet-stream")}, {}

    # ── dub / batch / quick test / showcase / redub ───────────────────
    async def submit_dub(
        self,
        source: str,
        target_lang: str,
        *,
        source_lang: str = "auto",
        model: Optional[str] = None,
        whisper_model: str = "large-v3",
        voice_preset: str = "auto",
        voice_style: str = "",
        tts_speed: str = "balanced",
        speaker_mode: str = "main",
        keep_bg: bool = False,
        auto_denoise: bool = False,
        context_hint: str = "",
        wizard_mode: str = "auto",
    ) -> dict:
        """Submit a single-language dub. Returns dict with `job_id`."""
        files, form = self._source_fields(source)
        form.update({
            "source_lang": source_lang,
            "target_lang": target_lang,
            "whisper_model": whisper_model,
            "voice_preset": voice_preset,
            "voice_style": voice_style,
            "tts_speed": tts_speed,
            "speaker_mode": speaker_mode,
            "keep_bg": str(bool(keep_bg)).lower(),
            "auto_denoise": str(bool(auto_denoise)).lower(),
            "context_hint": context_hint,
            "wizard_mode": wizard_mode,
        })
        if model:
            form["model"] = model
        return await self._request("POST", "/api/dub", data=form, files=files)

    async def submit_compare(
        self,
        source: str,
        target_langs: list[str] | str,
        *,
        trim_seconds: int = 60,
        source_lang: str = "auto",
        model: Optional[str] = None,
        whisper_model: str = "large-v3",
        voice_preset: str = "auto",
        tts_speed: str = "balanced",
        keep_bg: bool = False,
    ) -> dict:
        """Submit N separate dubs (Quick Test mode). 2-6 target_langs."""
        if isinstance(target_langs, (list, tuple)):
            target_langs = ",".join(target_langs)
        files, form = self._source_fields(source)
        form.update({
            "target_langs": target_langs,
            "trim_seconds": str(int(trim_seconds)),
            "source_lang": source_lang,
            "whisper_model": whisper_model,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "keep_bg": str(bool(keep_bg)).lower(),
        })
        if model:
            form["model"] = model
        return await self._request("POST", "/api/quick_test", data=form, files=files)

    async def submit_showcase(
        self,
        source: str,
        target_langs: list[str] | str,
        *,
        trim_seconds: int = 60,
        source_lang: str = "auto",
        model: Optional[str] = None,
        whisper_model: str = "large-v3",
        voice_preset: str = "auto",
        tts_speed: str = "balanced",
        keep_bg: bool = False,
    ) -> dict:
        """Submit a multilingual showcase reel. 2-6 target_langs are
        dubbed independently then stitched into one continuous video."""
        if isinstance(target_langs, (list, tuple)):
            target_langs = ",".join(target_langs)
        files, form = self._source_fields(source)
        form.update({
            "target_langs": target_langs,
            "trim_seconds": str(int(trim_seconds)),
            "source_lang": source_lang,
            "whisper_model": whisper_model,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "keep_bg": str(bool(keep_bg)).lower(),
        })
        if model:
            form["model"] = model
        return await self._request("POST", "/api/showcase", data=form, files=files)

    async def redub(
        self,
        job_id: str,
        target_langs: list[str] | str,
        *,
        mode: str = "compare",     # 'single' | 'compare' | 'showcase'
        model: Optional[str] = None,
        voice_preset: Optional[str] = None,
        tts_speed: Optional[str] = None,
    ) -> dict:
        """Re-dub an existing job's source into new language(s) without
        re-uploading. Inherits settings from the original; overrides allowed."""
        if isinstance(target_langs, (list, tuple)):
            target_langs = ",".join(target_langs)
        form = {"target_langs": target_langs, "mode": mode}
        for k, v in (("model", model), ("voice_preset", voice_preset),
                     ("tts_speed", tts_speed)):
            if v is not None:
                form[k] = v
        return await self._request("POST", f"/api/job/{job_id}/redub", data=form)

    # ── status / inspection ───────────────────────────────────────────
    async def get_job(self, job_id: str) -> dict:
        return await self._request("GET", f"/api/job/{job_id}")

    async def list_jobs(self, *, limit: int = 50,
                        status: Optional[str] = None,
                        batch_id: Optional[str] = None) -> list[dict]:
        data = await self._request("GET", "/api/jobs")
        # /api/jobs returns {"jobs": [...]}
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        if status:
            jobs = [j for j in jobs if j.get("status") == status]
        if batch_id:
            jobs = [j for j in jobs if j.get("batch_id") == batch_id]
        # Newest first
        jobs.sort(key=lambda j: j.get("created", 0), reverse=True)
        return jobs[:limit]

    async def get_showcase(self, batch_id: str) -> dict:
        return await self._request("GET", f"/api/showcase/{batch_id}")

    async def rebuild_showcase(self, batch_id: str) -> dict:
        return await self._request("POST", f"/api/showcase/{batch_id}/rebuild")

    async def cancel_job(self, job_id: str) -> dict:
        return await self._request("POST", f"/api/dub/{job_id}/cancel")

    async def delete_job(self, job_id: str) -> dict:
        return await self._request("DELETE", f"/api/job/{job_id}")

    async def system_status(self) -> dict:
        return await self._request("GET", "/api/system")

    async def list_languages(self) -> list[str]:
        """Static list of supported language codes (matches server's known set)."""
        return [
            "en", "ru", "es", "pt", "fr", "de", "it", "pl", "tr",
            "ja", "ko", "zh", "ar", "hi", "nl",
        ]

    async def list_models(self) -> list[str]:
        """Installed Ollama models (from /api/system)."""
        s = await self.system_status()
        models = s.get("ollama", {}).get("models", []) or []
        return [m if isinstance(m, str) else m.get("name", "") for m in models]

    async def list_voices(self) -> list[dict]:
        """Voice presets registered in the server."""
        try:
            d = await self._request("GET", "/api/voice_presets")
            return d.get("presets", []) if isinstance(d, dict) else []
        except TachiDUBBError:
            return []

    # ── result / output URLs ─────────────────────────────────────────
    def output_url(self, job_id: str, filename: str = "dubbed_video.mp4") -> str:
        """Absolute URL of a file in the job's output directory."""
        return f"{self.base_url}/outputs/{job_id}/{filename}"

    def showcase_url(self, batch_id: str) -> str:
        return f"{self.base_url}/outputs/showcase_{batch_id}/showcase.mp4"

    # ── high-level: wait until done ──────────────────────────────────
    async def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float = 1800.0,
        poll: float = 2.0,
    ) -> dict:
        """Poll until job reaches a terminal state. Returns the final job dict.

        Raises TachiDUBBError on timeout. Status 'error' is NOT raised — the
        caller inspects `result["status"]` and `result["error"]`.
        """
        start = time.monotonic()
        last: dict = {}
        while time.monotonic() - start < timeout:
            try:
                last = await self.get_job(job_id)
            except TachiDUBBError:
                # Job might not be flushed to disk yet; keep trying briefly
                pass
            if last.get("status") in _TERMINAL:
                return last
            await asyncio.sleep(poll)
        raise TachiDUBBError(
            f"Timeout after {timeout}s waiting for {job_id} (last status={last.get('status')})")

    async def wait_for_batch(
        self,
        batch_id: str,
        *,
        timeout: float = 3600.0,
        poll: float = 3.0,
    ) -> list[dict]:
        """Poll a batch (quick_test or showcase) until all child jobs are
        terminal. Returns list of final job dicts."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            jobs = await self.list_jobs(batch_id=batch_id, limit=10)
            if jobs and all(j.get("status") in _TERMINAL for j in jobs):
                return jobs
            await asyncio.sleep(poll)
        raise TachiDUBBError(f"Timeout after {timeout}s waiting for batch {batch_id}")

    async def wait_for_showcase(
        self,
        batch_id: str,
        *,
        timeout: float = 3600.0,
        poll: float = 3.0,
    ) -> dict:
        """Wait for showcase assembly to finish. Returns the final showcase info
        dict with `status` == 'ready' (success) or raises on timeout/error."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            info = await self.get_showcase(batch_id)
            if info.get("status") == "ready":
                return info
            if info.get("status") == "error":
                raise TachiDUBBError(f"Showcase assembly failed: {info.get('error', '?')}")
            await asyncio.sleep(poll)
        raise TachiDUBBError(f"Timeout after {timeout}s waiting for showcase {batch_id}")
