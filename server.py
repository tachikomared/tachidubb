"""
TachiDUBB Studio - Plug-and-Play AI Video Dubbing
==================================================
Created by TachikomaRed and smolemaru
Run: python server.py
Open: http://localhost:8910
"""
# ═══════════════════════════════════════════════════════════════════
# SPEECHBRAIN / K2 WORKAROUND — MUST RUN BEFORE ANY OTHER IMPORT
# ═══════════════════════════════════════════════════════════════════
# speechbrain 1.x uses lazy modules for `integrations.k2_fsa` and a few
# deprecated-redirect paths. On Windows the `k2` wheel doesn't exist, so
# these lazy imports fail the moment anything walks speechbrain's
# namespace (e.g. inspect.getmembers during TTS). We pre-populate
# sys.modules with empty stubs so importlib.import_module returns the
# stub instead of trying to actually load the broken chain.
#
# IMPORTANT: this block runs BEFORE pipeline imports so that WhisperX
# and pyannote (which transitively import speechbrain) see the stubs
# from the very first load.
import sys as _sys
import types as _types


def _tachidubb_stub_module(_name: str) -> None:
    if _name in _sys.modules:
        return
    m = _types.ModuleType(_name)
    m.__file__ = f"<tachidubb-stub:{_name}>"
    m.__path__ = []
    _sys.modules[_name] = m


for _n in (
    "k2",
    "speechbrain.k2_integration",
    "speechbrain.integrations.k2_fsa",
    "speechbrain.integrations.k2_fsa.ctc_loss",
    "speechbrain.integrations.k2_fsa.graph_compiler",
    "speechbrain.integrations.k2_fsa.lattice_decoder",
    "speechbrain.integrations.k2_fsa.lexicon",
    "speechbrain.integrations.k2_fsa.losses",
    "speechbrain.integrations.k2_fsa.prepare_lang",
    "speechbrain.integrations.k2_fsa.utils",
    "speechbrain.wordemb",
    "speechbrain.lobes.models.huggingface_transformers",
):
    _tachidubb_stub_module(_n)

del _sys, _types, _tachidubb_stub_module, _n

# ═══════════════════════════════════════════════════════════════════

import asyncio
import json
import re
import logging
import os
import shutil
import sys
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Load .env file if python-dotenv is installed (HF_TOKEN, etc)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        print(f"[env] Loaded {_env_path}")
except ImportError:
    pass  # dotenv optional


# ─────────────────────────────────────────────────────────────
# Windows: help torchcodec find FFmpeg DLLs
# ─────────────────────────────────────────────────────────────
# torchcodec ships libtorchcodec_coreN.dll which in turn loads avformat/
# avcodec/avutil DLLs from wherever they happen to live. On Windows those
# DLLs come from ffmpeg's bin/ folder (e.g. installed via winget or
# C:\ffmpeg\bin). If that folder isn't on PATH *for DLL search*, the load
# fails with the "Could not find module" cascade seen in prior logs.
#
# Python 3.8+ requires os.add_dll_directory() explicitly — just having it
# on PATH is no longer enough. We scan typical install locations and
# register any that contain avformat*.dll. No-op if torchcodec is absent.
if sys.platform == "win32":
    try:
        import os as _os
        _ffmpeg_candidates = []
        # 1. FFMPEG_DIR / FFMPEG_PATH env override
        for _env_var in ("FFMPEG_DIR", "FFMPEG_PATH"):
            _p = _os.environ.get(_env_var, "").strip()
            if _p and _os.path.isdir(_p):
                _ffmpeg_candidates.append(_p)
                _bin = _os.path.join(_p, "bin")
                if _os.path.isdir(_bin):
                    _ffmpeg_candidates.append(_bin)
        # 2. Locate via `where ffmpeg` PATH lookup
        _ffmpeg_on_path = shutil.which("ffmpeg")
        if _ffmpeg_on_path:
            _ffmpeg_candidates.append(_os.path.dirname(_ffmpeg_on_path))
        # 3. Common install roots
        for _root in (
            r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
            _os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
        ):
            if _os.path.isdir(_root):
                _ffmpeg_candidates.append(_root)
        _added = set()
        for _d in _ffmpeg_candidates:
            if _d in _added or not _os.path.isdir(_d):
                continue
            # Only add if it actually has avformat (real ffmpeg DLLs)
            try:
                _has_av = any(
                    f.lower().startswith("avformat") and f.lower().endswith(".dll")
                    for f in _os.listdir(_d)
                )
                if _has_av:
                    _os.add_dll_directory(_d)
                    _added.add(_d)
                    print(f"[ffmpeg] Registered DLL dir for torchcodec: {_d}")
            except Exception:
                continue
        if not _added:
            # Not fatal — torchcodec is optional; pyannote has a fallback.
            # Just note it in dev logs.
            pass
    except Exception as _e:
        print(f"[ffmpeg] DLL registration skipped: {_e}")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from pipeline.downloader import download_video
from pipeline.audio import extract_audio, extract_audio_hq, separate_background, get_duration
from pipeline.transcriber import transcribe
from pipeline.diarizer import (
    diarize_speakers, assign_speakers_to_segments,
    extract_speaker_audio, extract_fallback_reference,
)
from pipeline.translator import translate_segments, check_ollama, ollama_pull_stream, unload_ollama_model
from pipeline.synthesizer import VoxCPMSynthesizer, F5TTSEngine, EdgeTTSFallback
from pipeline.assembler import assemble_dubbed_audio, merge_audio_video, write_srt
from pipeline.models import get_system_status, MODEL_CATALOG
from pipeline.vad import apply_vad_filter

from app.config import cfg, BASE, UPLOAD_DIR, OUTPUT_DIR, JOBS_DB, STATIC_DIR, CONFIG_FILE
from app.db import init_db, save_job_sync, load_all_jobs, delete_job_db


# Load .env manually (no python-dotenv dependency) so HF_TOKEN etc. are picked up
def _load_dotenv_simple():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv_simple()

# Force UTF-8 stdout for foreign-language transcripts on Windows cp1252 consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Paths come from app.config (already created at import time)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("tachidubb.server")


# Silence extremely repetitive polling endpoints that flood the console
# (UI polls /api/system and /api/job/<id> every few seconds, httpx logs
# every Ollama /api/tags healthcheck call).
class _QuietPolling(logging.Filter):
    _QUIET_SUBSTRINGS = (
        "/api/system",
        "/api/tags",
        "/api/job/",
        "/api/voices",
        "/outputs/",       # range-requests for video playback after completion
    )
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(q in msg for q in self._QUIET_SUBSTRINGS)


for _logger_name in ("uvicorn.access", "httpx", "httpcore"):
    logging.getLogger(_logger_name).addFilter(_QuietPolling())
# httpx logs Ollama calls at INFO, demote to WARNING so only errors show
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Windows ProactorEventLoop spams harmless WinError 10054 ("connection
# forcibly closed by remote host") every time a browser tab is closed or
# the video player seeks. These are not actionable errors — the tab went
# away, not a real problem. Filter them out so real errors stand out.
class _WindowsConnResetFilter(logging.Filter):
    _NOISE_SUBSTRINGS = (
        "WinError 10054",
        "ConnectionResetError",
        "_call_connection_lost",
    )
    def filter(self, record):
        try:
            msg = record.getMessage()
            if any(n in msg for n in self._NOISE_SUBSTRINGS):
                return False
            # Also check exception info (WinError 10054 often logged via exc_info)
            if record.exc_info:
                exc_str = str(record.exc_info[1])
                if any(n in exc_str for n in self._NOISE_SUBSTRINGS):
                    return False
        except Exception:
            pass
        return True


logging.getLogger("asyncio").addFilter(_WindowsConnResetFilter())

jobs: dict = {}
_tts_engine = None


def _free_gpu_memory():
    """Best-effort GPU memory cleanup. Call before loading a heavy model
    when another one may have left VRAM cached. Safe to call even if
    torch isn't imported — fails silently.

    What this does:
      1. Force Python GC so any dead tensor references get collected
      2. torch.cuda.empty_cache() releases PyTorch's cached allocator back
         to the driver (Ollama / llama.cpp don't share this cache so their
         unloads are separate)
      3. torch.cuda.ipc_collect() releases handles from forked processes
    """
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            log.info(f"[gpu] Cleared torch cache; free VRAM ~{free_mb:.0f} MB")
    except Exception as e:
        log.debug(f"_free_gpu_memory: {e}")

# ═══════════════════════════════════════════════════════════════════════
#  Job queue — ensures only one GPU-heavy job runs at a time
# ═══════════════════════════════════════════════════════════════════════
# Multiple concurrent dubs would OOM a 12GB GPU (WhisperX + VoxCPM +
# pyannote ≈ 10GB peak). Instead we queue. User can submit 5 videos in
# a row — they all get job_ids immediately and appear in History with
# status="queued". The scheduler processes them serially.
# User-facing benefit: "fire and forget" — drop 3 videos on the app,
# walk away, come back to 3 finished dubs.
# ═══════════════════════════════════════════════════════════════════════
_job_queue: "asyncio.Queue[tuple]" = None  # set in lifespan
_queue_worker_task = None
_scheduler_task = None


async def _scheduler_loop():
    """Background loop that moves scheduled jobs into the live queue when
    their scheduled_at time arrives. Polls every 30 seconds — good enough
    resolution for "start at 2 AM" use cases, and doesn't thrash CPU.

    Survives server restarts because the job state (status='scheduled' +
    scheduled_at timestamp + _pending_args) is persisted to disk. If the
    server was down when the time passed, jobs whose scheduled_at is in
    the past get enqueued immediately on next poll.
    """
    log.info("[scheduler] Loop started (polls every 30s)")
    while True:
        try:
            await asyncio.sleep(30)
            now = time.time()
            ready = [
                j for j in jobs.values()
                if j.get("status") == "scheduled"
                and j.get("scheduled_at", 0) > 0
                and j.get("scheduled_at", 0) <= now
            ]
            for j in ready:
                args = j.get("_pending_args")
                if not args:
                    j["status"] = "error"
                    j["error"] = "Scheduled job missing pipeline args"
                    save_job(j)
                    continue
                log.info(f"[scheduler] Job {j['id']} reached scheduled time — enqueueing")
                j.pop("_pending_args", None)
                await enqueue_job(j["id"], args)
        except asyncio.CancelledError:
            log.info("[scheduler] Loop cancelled, exiting")
            return
        except Exception as e:
            log.warning(f"[scheduler] Iteration failed (continuing): {e}")


async def enqueue_job_stub_removed_marker():
    pass


# ═══════════════════════════════════════════════════════════════════════
#  Cancellation — make the Cancel button actually work
# ═══════════════════════════════════════════════════════════════════════
# cancel_job() sets `j["cancel_requested"] = True` on the job dict.
# The pipeline checks this flag at every stage boundary (via update())
# and at per-segment progress callbacks, raising JobCancelled when set.
# The pipeline wrapper catches it and marks status=cancelled cleanly.
# ═══════════════════════════════════════════════════════════════════════
class JobCancelled(Exception):
    """Raised inside pipeline stages when the user has requested cancel."""
    pass


def _cancel_requested(job_id: str) -> bool:
    return bool(job_id in jobs and jobs[job_id].get("cancel_requested"))


def _maybe_terminate_tts_worker():
    """Best-effort kill of the persistent VoxCPM TTS subprocess.
    Called when we detect cancel mid-synthesis so the next iteration
    of stdout.readline() exits immediately instead of waiting for the
    current segment to finish rendering."""
    try:
        tts = _tts_engine
        if tts is None:
            return
        proc = getattr(tts, "_worker_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                log.info("[cancel] TTS worker terminated")
            except Exception as e:
                log.debug(f"[cancel] TTS worker terminate failed: {e}")
            try:
                tts._worker_proc = None
            except Exception:
                pass
    except Exception as e:
        log.debug(f"[cancel] _maybe_terminate_tts_worker error: {e}")


async def enqueue_job(job_id, pipeline_args):
    """Add a job to the queue; scheduler will pick it up. If queue empty
    and scheduler idle, starts processing immediately.

    pipeline_args: dict of keyword args for run_pipeline (preferred),
        or legacy tuple of positional args (old enqueue sites).
    """
    global _job_queue
    if _job_queue is None:
        # Queue not initialized yet (shouldn't happen after startup)
        _job_queue = asyncio.Queue()
    # Mark job as queued in the store so UI shows "queued" badge
    if job_id in jobs:
        jobs[job_id]["status"] = "queued"
        jobs[job_id]["queue_position"] = _job_queue.qsize() + 1
        save_job(jobs[job_id])
    await _job_queue.put((job_id, pipeline_args))
    log.info(f"[queue] Job {job_id} enqueued (position {_job_queue.qsize()})")
    # Auto-activate sleep prevention when the queue has jobs waiting.
    # This ensures long unattended runs don't halt because Windows slept.
    _apply_sleep_prevention(True)


# ═══════════════════════════════════════════════════════════════════════
#  Windows sleep prevention — keeps PC awake during long batch runs
# ═══════════════════════════════════════════════════════════════════════
# Night-mode workflow: user drops 5 courses in the queue, goes to bed.
# Without this, Windows would go to sleep ~20 min into the first video
# and pause everything. We call SetThreadExecutionState to tell Windows
# "keep the system awake AND keep CPU busy" while we have work.
# ES_CONTINUOUS=0x80000000, ES_SYSTEM_REQUIRED=0x00000001
# The flag is persistent until cleared (not a timer), so we must clear
# it when the queue empties. Called from both enqueue and worker-idle.
# ═══════════════════════════════════════════════════════════════════════
_sleep_lock_active = False


def _apply_sleep_prevention(keep_awake: bool):
    """Toggle Windows sleep prevention. Safe no-op on non-Windows OS."""
    global _sleep_lock_active
    try:
        import ctypes
        if not hasattr(ctypes, "windll"):
            return  # not Windows
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        # ES_AWAYMODE_REQUIRED could be added on desktop to keep CPU active
        # even with lid closed; we leave it off to allow screen sleep but
        # prevent system sleep.
        if keep_awake and not _sleep_lock_active:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
            _sleep_lock_active = True
            log.info("[power] Sleep prevention ON — PC stays awake during queue")
        elif not keep_awake and _sleep_lock_active:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            _sleep_lock_active = False
            log.info("[power] Sleep prevention OFF — PC can sleep again")
    except Exception as e:
        log.debug(f"[power] Sleep prevention toggle failed: {e}")


async def _job_queue_worker():
    """Background worker — runs pipelines serially from the queue."""
    global _job_queue
    log.info("[queue] Worker started")
    while True:
        try:
            job_id, pipeline_args = await _job_queue.get()
        except asyncio.CancelledError:
            log.info("[queue] Worker cancelled, exiting")
            return
        try:
            # Update queue positions for waiting jobs so UI reflects movement
            for j in jobs.values():
                if j.get("status") == "queued":
                    # Decrement: this one just left the queue,
                    # others move up
                    pos = j.get("queue_position", 1) - 1
                    j["queue_position"] = max(pos, 1)
            log.info(f"[queue] Processing job {job_id}")
            # Mark actual start time so elapsed/ETA measurements are
            # accurate (created = enqueue time, which can be much earlier
            # in a big batch). See "#3 started_at is never recorded" audit.
            if job_id in jobs:
                jobs[job_id]["started_at"] = time.time()
                save_job(jobs[job_id])
            # Queue stores args as a dict keyword, not positional tuple — so we
            # can add new pipeline params without breaking old enqueue sites
            if isinstance(pipeline_args, dict):
                await run_pipeline(job_id, **pipeline_args)
            else:
                # Legacy positional tuple (kept for back-compat with older queued jobs)
                await run_pipeline(job_id, *pipeline_args[:12],
                                   wizard_mode=pipeline_args[12])
        except JobCancelled:
            # Cancel is a user action, not an error. Status was already
            # set by the exception raiser; just log cleanly.
            log.info(f"[queue] Job {job_id} cancelled by user")
            if job_id in jobs:
                jobs[job_id]["status"] = "cancelled"
                jobs[job_id].pop("cancel_requested", None)
                save_job(jobs[job_id])
        except Exception as e:
            log.error(f"[queue] Job {job_id} crashed: {e}", exc_info=True)
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e) or type(e).__name__
                save_job(jobs[job_id])
        else:
            # Job completed cleanly. Run post-success hooks:
            #   1. Auto lip-sync (Wav2Lip) if the job opted in.
            #   2. Showcase stitch if this was the last sibling in a showcase batch.
            # Both are best-effort — failures log but don't fail the job.
            try:
                if job_id in jobs and jobs[job_id].get("lip_sync"):
                    log.info(f"[queue] post-hook: running auto lip-sync on {job_id}")
                    await asyncio.get_event_loop().run_in_executor(
                        None, _run_wav2lip_sync, job_id)
            except Exception as e:
                log.warning(f"[lipsync] auto hook failed: {e}", exc_info=True)
            try:
                if job_id in jobs and jobs[job_id].get("batch_kind") == "showcase":
                    await _maybe_assemble_showcase(jobs[job_id].get("batch_id", ""))
            except Exception as e:
                log.warning(f"[showcase] post-process hook failed: {e}", exc_info=True)
        finally:
            _job_queue.task_done()
            # Release sleep lock once nothing else is pending.
            # Next enqueue will re-acquire automatically.
            if _job_queue.empty():
                _apply_sleep_prevention(False)


def load_jobs_from_disk():
    loaded = load_all_jobs()
    jobs.update(loaded)
    # Mark any jobs that appear to still be "running" as error+resumable.
    # After a server restart the in-memory queue is empty, so these jobs
    # aren't actually being processed anymore. Without this fix, the
    # History tab shows them as permanently "transcribing..." / "queued".
    # The user can still click Resume (if a checkpoint exists) to pick
    # up where the pipeline left off.
    _active_statuses = {
        "queued", "running", "downloading", "extracting",
        "transcribing", "translating", "synthesizing",
        "assembling", "merging",
    }
    stale_count = 0
    for jid, job in jobs.items():
        if job.get("status") in _active_statuses:
            job["status"] = "error"
            job["error"] = (
                job.get("error")
                or "Interrupted by server restart — click Resume to continue"
            )
            job["stale_from_restart"] = True
            save_job(job)
            stale_count += 1
    if stale_count:
        log.info(
            f"Marked {stale_count} stale job(s) as 'error' "
            f"(left over from previous server run)"
        )
    log.info(f"Loaded {len(jobs)} jobs from disk")


def _job_checkpoint_info(job_id: str) -> dict:
    """Return which checkpoint stages exist on disk for a job.

    Used by list_jobs so the History UI knows whether an errored or
    cancelled job is resumable (and from where). Purely filesystem
    inspection — cheap enough to do on every /api/jobs poll."""
    work_dir = OUTPUT_DIR / job_id
    if not work_dir.exists():
        return {"has_checkpoint": False, "latest_checkpoint_stage": None}
    # Check most-advanced first so 'latest' reflects how far the
    # pipeline got before stopping.
    for stage in ("tts_done", "translation_done", "transcription_done"):
        if (work_dir / f"checkpoint_{stage}.json").exists():
            return {"has_checkpoint": True, "latest_checkpoint_stage": stage}
    # Legacy single-file pipeline state
    if (work_dir / "pipeline_state.json").exists():
        try:
            with open(work_dir / "pipeline_state.json", "r", encoding="utf-8") as f:
                d = json.load(f)
            return {
                "has_checkpoint": True,
                "latest_checkpoint_stage": d.get("stage") or "unknown",
            }
        except Exception:
            pass
    return {"has_checkpoint": False, "latest_checkpoint_stage": None}


def save_job(job: dict):
    save_job_sync(job)


# ─────────────────────────────────────────────────────────────
# Pipeline checkpoint system
# ─────────────────────────────────────────────────────────────
# Instead of one big pipeline_state.json, we save multiple checkpoint
# files — one per completed stage. This lets the user rewind to any
# earlier stage (e.g., "regenerate translation from scratch") without
# redoing the stages before it.
#
# File naming:
#   pipeline_state.json              — legacy; last-stage checkpoint
#   checkpoint_transcription_done.json
#   checkpoint_translation_done.json
#   checkpoint_tts_done.json
#
# Each file is self-contained — loading it is enough to resume from
# the corresponding stage. "stage" field says which stage just finished.

def _save_checkpoint(job_id: str, work_dir: Path, stage: str, data: dict) -> None:
    data["stage"] = stage
    data["job_id"] = job_id
    data["saved_at"] = time.time()

    # Named checkpoint (never overwritten by later stages)
    cpath = work_dir / f"checkpoint_{stage}.json"
    try:
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"[checkpoint] Saved: {stage} -> {cpath.name}")
    except Exception as e:
        log.warning(f"[checkpoint] Save failed ({stage}): {e}")

    # Legacy pipeline_state.json — always the MOST RECENT checkpoint
    try:
        with open(work_dir / "pipeline_state.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"[checkpoint] Legacy save failed: {e}")


def _load_checkpoint(job_id: str, stage: str) -> Optional[dict]:
    """Load a specific checkpoint. Returns None if not found."""
    work_dir = OUTPUT_DIR / job_id
    cpath = work_dir / f"checkpoint_{stage}.json"
    if not cpath.exists():
        # Fallback to legacy pipeline_state.json if it matches the stage
        legacy = work_dir / "pipeline_state.json"
        if legacy.exists():
            try:
                with open(legacy, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("stage") == stage:
                    return d
            except Exception:
                pass
        return None
    try:
        with open(cpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"[checkpoint] Load failed ({stage}): {e}")
        return None


def _latest_checkpoint(job_id: str) -> Optional[dict]:
    """Return the most advanced checkpoint available for a job."""
    for stage in ("tts_done", "translation_done", "transcription_done"):
        cp = _load_checkpoint(job_id, stage)
        if cp:
            return cp
    return None


def get_tts_engine():
    """TTS engine factory. Priority: VoxCPM2 → F5-TTS → Edge-TTS.

    Selection follows cfg.tts_engine preference, falling back through
    the tier chain when a higher-tier engine isn't installed or fails to load.
    """
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    _free_gpu_memory()

    requested = cfg.tts_engine  # "voxcpm" | "f5tts" | "edge-tts"

    # Tier 1: VoxCPM2
    if requested in ("voxcpm", "auto"):
        try:
            import voxcpm  # noqa
            _tts_engine = VoxCPMSynthesizer(
                model_id=cfg.voxcpm_model,
                load_denoiser=False,
                cfg_value=cfg.voxcpm_cfg,
                inference_timesteps=cfg.voxcpm_steps,
            )
            log.info("TTS engine: VoxCPM2 (voice cloning)")
            import atexit
            atexit.register(lambda: _tts_engine.unload() if _tts_engine else None)
            return _tts_engine
        except ImportError:
            log.info("VoxCPM2 not installed, trying F5-TTS...")

    # Tier 2: F5-TTS
    if requested in ("f5tts", "auto", "voxcpm"):
        try:
            import f5_tts  # noqa
            _tts_engine = F5TTSEngine()
            log.info("TTS engine: F5-TTS (voice cloning, lighter than VoxCPM2)")
            return _tts_engine
        except ImportError:
            log.info("F5-TTS not installed, falling back to Edge-TTS")

    # Tier 3: Edge-TTS (always available)
    _tts_engine = EdgeTTSFallback()
    log.warning("TTS engine: Edge-TTS fallback (no voice cloning)")
    return _tts_engine


# ─────────────────────────────────────────────────────────────
# Voice Preset Library
# ─────────────────────────────────────────────────────────────
# Each preset = a voice-design description + a fixed seed.
# The seed locks VoxCPM's random state so all segments in a job
# sound like the SAME voice (voice design is non-deterministic
# by default per the VoxCPM docs).
VOICE_PRESETS = {
    "auto": {
        "name": "Auto (use video voice if possible)",
        "style": "",
        "seed": None,  # random per-job
    },
    "male_warm": {
        "name": "Male — warm, middle-aged, calm",
        "style": "middle-aged male voice, warm and calm, clear articulation",
        "seed": 101,
    },
    "male_deep": {
        "name": "Male — deep, authoritative narrator",
        "style": "deep mature male voice, authoritative narrator, slow pace",
        "seed": 202,
    },
    "male_young": {
        "name": "Male — young, energetic",
        "style": "young adult male voice, energetic and friendly",
        "seed": 303,
    },
    "male_sports": {
        "name": "Male — sports instructor",
        "style": "grown adult male sports instructor, clear and steady, confident",
        "seed": 404,
    },
    "female_calm": {
        "name": "Female — warm, gentle",
        "style": "warm female voice, gentle and soothing, mid-tone",
        "seed": 505,
    },
    "female_narrator": {
        "name": "Female — professional narrator",
        "style": "professional female narrator, clear articulation, neutral tone",
        "seed": 606,
    },
    "female_young": {
        "name": "Female — young, cheerful",
        "style": "young adult female voice, friendly and cheerful",
        "seed": 707,
    },
}


VOICE_PRESETS_DIR = BASE / "presets" / "voices"
VOICE_PRESETS_DIR.mkdir(parents=True, exist_ok=True)

# User-editable glossary overrides. Loaded by translator.py when building
# prompts; editable via /api/glossary endpoints in the Settings tab.
USER_GLOSSARY_FILE = BASE / "presets" / "user_glossary.json"

# User preferences — persisted across sessions. Simple JSON blob edited
# by the UI; not schema-validated server-side (it's just a KV store).
PREFS_FILE = BASE / "user_prefs.json"


_VOICE_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def _voice_metadata_path(audio_path: Path) -> Path:
    """Sidecar metadata file for a voice preset (<name>.json next to audio)."""
    return audio_path.with_suffix(".json")


def _read_voice_metadata(audio_path: Path) -> dict:
    """Load JSON sidecar with structured metadata, falling back to legacy
    `<name>.txt` description if JSON doesn't exist. Always returns a dict."""
    j = _voice_metadata_path(audio_path)
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[voice_presets] bad JSON in {j.name}: {e}")
    # Backward-compat: legacy .txt description file
    txt = audio_path.with_suffix(".txt")
    if txt.exists():
        try:
            return {"description": txt.read_text(encoding="utf-8").strip()[:300]}
        except Exception:
            pass
    return {}


def scan_file_presets() -> dict:
    """Scan presets/voices/ folder for voice references + their metadata.

    Each `<name>.{wav,mp3,flac,ogg,m4a}` becomes a file-based preset.
    Optional `<name>.json` sidecar carries structured metadata
    (description, gender, language, tags, created_at). Legacy `<name>.txt`
    is still read as the description for backward-compat.

    Re-scanned on every endpoint call — drop a file into the folder and
    it's available immediately, no restart needed.
    """
    presets = {}
    if not VOICE_PRESETS_DIR.exists():
        return presets
    for path in sorted(VOICE_PRESETS_DIR.iterdir()):
        if path.suffix.lower() not in _VOICE_AUDIO_EXTS:
            continue
        meta = _read_voice_metadata(path)
        pid = f"file:{path.stem}"
        presets[pid] = {
            "id": pid,
            "name": meta.get("display_name") or path.stem,
            "style": meta.get("style", ""),
            "seed": meta.get("seed"),
            "reference_file": str(path),
            "description": meta.get("description", ""),
            # Structured metadata for the Voices tab UI
            "gender": meta.get("gender", ""),         # 'male' | 'female' | 'neutral' | ''
            "language": meta.get("language", ""),     # iso code or empty
            "tags": meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
            "created_at": meta.get("created_at"),
            # File facts (computed, not stored)
            "file_size": path.stat().st_size if path.exists() else 0,
            "file_ext": path.suffix.lower().lstrip("."),
            "audio_url": f"/api/voice_presets/{pid}/audio",
        }
    return presets


def resolve_voice_config(voice_preset: str, voice_style: str, job_id: str):
    """Return (effective_voice_style, voice_seed, reference_file) for a run.

    reference_file is set ONLY when user picked a file-based preset
    from presets/voices/ folder. Otherwise it's empty string.
    """
    import hashlib
    # Check file presets first (they live in a folder, re-scanned each call)
    file_presets = scan_file_presets()
    if voice_preset in file_presets:
        p = file_presets[voice_preset]
        return "", 0, p["reference_file"]

    preset = VOICE_PRESETS.get(voice_preset, VOICE_PRESETS["auto"])
    # Priority: explicit voice_style beats preset style (lets user override)
    eff_style = (voice_style or "").strip() or preset["style"]
    # Priority: preset seed > hash of voice_style > hash of job_id (random-ish)
    if preset.get("seed") is not None:
        seed = preset["seed"]
    elif eff_style:
        seed = int(hashlib.md5(eff_style.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
    else:
        seed = int(hashlib.md5(job_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
    return eff_style, seed, ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init SQLite store (creates table + migrates legacy JSON files)
    init_db(BASE / "tachidubb.db")
    load_jobs_from_disk()

    # Initialize job queue + start serial worker. Using a single worker
    # ensures GPU-heavy pipelines don't collide and OOM the card.
    global _job_queue, _queue_worker_task, _scheduler_task
    _job_queue = asyncio.Queue()
    _queue_worker_task = asyncio.create_task(_job_queue_worker())
    # Scheduler: polls every 30s looking for jobs with status='scheduled'
    # whose scheduled_at time has arrived. Survives restarts — status and
    # scheduled_at persist in the job dict on disk.
    _scheduler_task = asyncio.create_task(_scheduler_loop())

    if os.getenv("TACHIDUBB_OPEN_BROWSER", "1") == "1" and not os.getenv("DOCKER"):
        async def open_browser():
            await asyncio.sleep(1.5)
            try:
                webbrowser.open("http://localhost:8910")
            except Exception:
                pass
        asyncio.create_task(open_browser())

    # Warm up the TTS subprocess so the first dub doesn't pay the 40-60s
    # model-load tax. We send a tiny dummy job to the daemon worker which
    # loads VoxCPM and then idles waiting for real jobs. Runs in background;
    # failures are non-fatal (engine will load lazily on first real use).
    # VoxCPM warmup is now OFF by default. Reasoning: VoxCPM holds ~4 GB of
    # VRAM for its lifetime, and on 12 GB cards that prevents Ollama from
    # loading larger translation models (gemma4:e4b = 9.6 GB). By deferring
    # VoxCPM load until AFTER translation, Ollama gets the full 12 GB to
    # itself, translates fast, unloads via keep_alive, then VoxCPM loads
    # for TTS. Cost: first dub is ~20s slower (one-time VoxCPM load).
    # To opt back in (fast multi-user servers with abundant VRAM): set
    # TACHIDUBB_WARMUP=1 in .env or environment.
    if os.getenv("TACHIDUBB_WARMUP", "0") == "1":
        async def warmup_tts():
            await asyncio.sleep(2.0)  # let server finish binding port
            try:
                import tempfile as _tmp
                import wave as _wave
                import struct as _struct
                log.info("[warmup] Pre-spawning persistent TTS worker...")
                t0 = time.time()
                tts = get_tts_engine()
                if not isinstance(tts, VoxCPMSynthesizer):
                    return  # Edge-TTS fallback doesn't need warmup
                # Create a minimal valid WAV file to serve as dummy reference
                dummy_dir = _tmp.mkdtemp(prefix="tachidubb_warmup_")
                dummy_ref = os.path.join(dummy_dir, "dummy_ref.wav")
                with _wave.open(dummy_ref, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    # 1 second of low-volume noise (so ref validation passes)
                    w.writeframes(_struct.pack("<" + "h"*16000,
                                               *([0]*16000)))
                dummy_segs = [{
                    "idx": 0, "start": 0.0, "end": 1.0,
                    "text": "привет", "translated_text": "привет",
                    "speaker": "SPEAKER_00",
                }]
                # Run in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: tts.synthesize_segments(
                    dummy_segs, dummy_dir,
                    speaker_refs={"SPEAKER_00": dummy_ref},
                    speaker_transcripts={"SPEAKER_00": ""},
                    tts_speed="balanced",
                ))
                log.info(f"[warmup] TTS worker ready in {time.time()-t0:.1f}s "
                         f"— first dub will skip model-load")
                # Clean up dummy artifacts silently
                try:
                    shutil.rmtree(dummy_dir, ignore_errors=True)
                except Exception:
                    pass
            except Exception as e:
                log.warning(f"[warmup] Pre-load failed "
                            f"(engine will load on first real use): {e}")
        asyncio.create_task(warmup_tts())

    yield

    # Shutdown: cancel queue worker + scheduler so they don't hang the process
    for t in (_queue_worker_task, _scheduler_task):
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(title="TachiDUBB Studio", version="2.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ─────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────
async def run_pipeline(
    job_id: str,
    source: str,
    source_lang: str,
    target_lang: str,
    model: str,
    keep_bg: bool,
    whisper_model: str,
    reference_audio: str = "",
    speaker_mode: str = "main",
    context_hint: str = "",
    voice_style: str = "",
    voice_preset: str = "auto",
    tts_speed: str = "balanced",
    wizard_mode: str = "auto",  # "auto" | "review_translation" | "review_transcript"
    auto_denoise: bool = True,  # apply ffmpeg denoise before WhisperX
):
    """Main dubbing pipeline. When wizard_mode != 'auto', pauses at the
    specified checkpoint with status='awaiting_review' so the user can
    inspect/edit intermediate results before continuing."""
    job = jobs[job_id]
    work = OUTPUT_DIR / job_id
    work.mkdir(exist_ok=True)
    job["wizard_mode"] = wizard_mode

    # Resolve final voice config once, store on job so UI can display it
    eff_style, voice_seed, preset_ref_file = resolve_voice_config(voice_preset, voice_style, job_id)
    job["voice_preset"] = voice_preset
    job["voice_style_effective"] = eff_style
    job["voice_seed"] = voice_seed

    # If a file-based preset was selected, it acts like user-uploaded reference
    if preset_ref_file and os.path.exists(preset_ref_file):
        log.info(f"[ref] File-preset selected: {preset_ref_file}")
        reference_audio = preset_ref_file

    def update(**kwargs):
        # Check the cancel flag at every stage transition. Any pipeline
        # path that calls update() will raise JobCancelled within ~1
        # instruction of the user clicking Cancel, and the outer handler
        # in _job_queue_worker will mark status=cancelled cleanly.
        if job.get("cancel_requested"):
            _maybe_terminate_tts_worker()
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs)
        save_job(job)

    try:
        # 1. Acquire video
        update(status="downloading", progress=2, step_detail="Getting video...")
        video_path = download_video(source, str(work))
        duration = get_duration(video_path)
        update(duration=round(duration, 1), progress=8)

        # 2. Extract audio
        update(status="extracting", progress=10, step_detail="Extracting audio tracks...")
        audio_16k = str(work / "audio_16k.wav")
        extract_audio(video_path, audio_16k)

        # 2b. Optional denoise for noisy source audio.
        # BJJ/cooking/sports videos often have mat noise, background music,
        # crowd, or equipment hum that WhisperX mistakes for words. We
        # apply an ffmpeg filter chain to clean the audio WITHOUT removing
        # voice quality. Enabled via auto_denoise flag (default True on
        # high-duration videos where noise can compound error rate).
        # Filter chain reasoning:
        #   - afftdn: FFT-based noise reduction (safe, preserves speech)
        #   - highpass=80: drop sub-bass rumble (room noise, AC)
        #   - lowpass=10000: drop tweeter noise (mic hiss, digital artifacts)
        # This is conservative; aggressive denoise can hurt Whisper accuracy.
        if auto_denoise:
            try:
                import subprocess
                denoise_start = time.time()
                audio_clean = str(work / "audio_16k_clean.wav")
                update(progress=12, step_detail="Cleaning audio...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_16k,
                     "-af", "afftdn=nr=10:nf=-25,highpass=f=80,lowpass=f=10000",
                     "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                     audio_clean],
                    check=True, capture_output=True, timeout=180,
                )
                log.info(
                    f"[audio] Denoised in {time.time()-denoise_start:.1f}s "
                    f"(audio_16k_clean.wav) — feeding to WhisperX"
                )
                audio_16k = audio_clean
            except Exception as e:
                log.warning(f"Denoise failed (using raw audio): {e}")

        bg_audio_path = ""
        if keep_bg:
            try:
                audio_hq = str(work / "audio_hq.wav")
                extract_audio_hq(video_path, audio_hq)
                _, bg_audio_path = separate_background(audio_hq, str(work))
            except Exception as e:
                log.warning(f"BG separation skipped: {e}")
        update(progress=15)

        # 2c. VAD filtering — strip long silence/music before Whisper.
        # silero-vad is optional (graceful fallback to full audio).
        if cfg.vad_enabled:
            try:
                vad_out = str(work / "audio_16k_vad.wav")
                update(progress=16, step_detail="Filtering non-speech regions...")
                audio_16k, speech_ratio = apply_vad_filter(
                    audio_16k, vad_out, threshold=cfg.vad_threshold
                )
                if speech_ratio < 0.15:
                    log.warning(
                        f"[vad] Low speech ratio ({speech_ratio:.0%}) — "
                        f"consider disabling VAD or checking audio source"
                    )
            except Exception as e:
                log.warning(f"VAD skipped: {e}")

        # 3. Transcribe
        # 3. Transcribe with elapsed-time progress hint.
        # WhisperX doesn't expose internal progress, so during long transcription
        # (e.g. 20-min podcasts take 5-6min) the UI would just show "Transcribing..."
        # forever. We spawn a watchdog that updates step_detail with elapsed
        # seconds so the user can see it's still alive.
        update(status="transcribing", progress=18, step_detail="Transcribing speech...")
        _t_trans_start = time.time()
        _trans_done_flag = {"done": False}
        async def _trans_watchdog():
            while not _trans_done_flag["done"]:
                elapsed = int(time.time() - _t_trans_start)
                if elapsed > 10:  # only show after 10s to avoid noise on short videos
                    mins, secs = divmod(elapsed, 60)
                    hint = f"Transcribing ({duration:.0f}s audio)... elapsed {mins}m{secs:02d}s"
                    if elapsed > 120:
                        hint += " · try smaller whisper model for faster transcribe"
                    update(step_detail=hint)
                await asyncio.sleep(5)
        _watchdog_task = asyncio.create_task(_trans_watchdog())
        try:
            segments, detected_lang = transcribe(audio_16k, source_lang, whisper_model)
        finally:
            _trans_done_flag["done"] = True
            _watchdog_task.cancel()
            try:
                await _watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
        log.info(
            f"[transcribe] Completed in {time.time() - _t_trans_start:.1f}s "
            f"({duration:.0f}s audio, ratio {duration / max(time.time() - _t_trans_start, 1):.1f}x realtime)"
        )
        effective_src = detected_lang if source_lang == "auto" else source_lang
        update(
            source_lang_detected=detected_lang,
            segment_count=len(segments),
            progress=35,
        )

        if not segments:
            raise RuntimeError("No speech detected in video")

        # 4. Diarize
        update(status="diarizing", progress=38, step_detail="Identifying speakers...")
        hf_token = os.getenv("HF_TOKEN", "")
        speaker_turns = diarize_speakers(audio_16k, hf_token=hf_token)
        segments = assign_speakers_to_segments(segments, speaker_turns)

        # Store raw transcript preview for UI
        transcript_preview_raw = [
            {
                "idx": i,
                "start": s["start"], "end": s["end"],
                "text": s["text"],
                "speaker": s.get("speaker", ""),
            }
            for i, s in enumerate(segments)
        ]
        update(transcript_raw=transcript_preview_raw)

        speaker_refs = {}
        speaker_transcripts = {}
        # Always-preserved copy of refs extracted from the SOURCE video.
        # Even when user picks a preset/upload (which overrides speaker_refs),
        # we still stash the original source refs here so that "retry TTS"
        # without re-choosing a ref can go back to the source voice. Without
        # this, the uploaded preset got baked into the checkpoint and retry
        # would silently reuse it forever.
        source_speaker_refs = {}

        # Case A: user uploaded a reference voice -> use it for ALL speakers.
        # Pure Controllable Cloning: just reference_wav_path, nothing else.
        # VoxCPM2 README: "Clone any voice from a short reference clip".
        if reference_audio and os.path.exists(reference_audio):
            log.info(f"[ref] Using USER-UPLOADED reference: {reference_audio}")
            unique_speakers = {s.get("speaker", "SPEAKER_00") for s in segments}
            for sp in unique_speakers:
                speaker_refs[sp] = reference_audio
                # NOT populating speaker_transcripts — we want clean Controllable
                # Cloning (reference_wav_path only), not Ultimate Cloning which
                # needs an exact transcript of the reference audio and is
                # stricter about audio continuation.
                speaker_transcripts[sp] = ""
            # Also extract source refs for potential retry-with-source-voice
            if speaker_turns:
                try:
                    refs_dir = str(work / "speaker_refs")
                    source_speaker_refs = extract_speaker_audio(
                        audio_16k, speaker_turns, refs_dir, main_only=False,
                    ) or {}
                    log.info(f"[ref] Also extracted {len(source_speaker_refs)} "
                             f"source-voice refs for potential retry use")
                except Exception as e:
                    log.warning(f"[ref] Source-ref extraction failed (ok to skip): {e}")

        # Case B: diarization worked -> per-speaker refs from the source
        elif speaker_turns:
            log.info("[ref] No user upload; extracting speaker refs from source video")
            refs_dir = str(work / "speaker_refs")
            main_only = speaker_mode == "main"
            speaker_refs = extract_speaker_audio(
                audio_16k, speaker_turns, refs_dir, main_only=main_only,
            )
            # Source refs = speaker_refs in this case (same origin)
            source_speaker_refs = dict(speaker_refs)
            if speaker_refs:
                # Populate speaker_transcripts (enables Tier 1 Ultimate Cloning)
                # ONLY for same-language dubbing. Cross-lingual Tier 1 makes
                # VoxCPM "continue" the source-language phonetics, so Russian
                # text comes out with English phonemes = gibberish. For
                # cross-lingual we want Tier 2 Controllable Cloning which just
                # clones timbre without audio-continuation.
                same_lang = (effective_src == target_lang)
                if same_lang:
                    for spk in speaker_refs:
                        texts = [s["text"] for s in segments
                                 if s.get("speaker") == spk]
                        if texts:
                            speaker_transcripts[spk] = " ".join(texts[:3])
                else:
                    log.info(f"[ref] Cross-lingual dub ({effective_src}→{target_lang}); "
                             f"clearing prompt_text to force Controllable Cloning")
                    for spk in speaker_refs:
                        speaker_transcripts[spk] = ""

            # If main_only: remap EVERY segment to the sole extracted speaker
            if main_only and speaker_refs:
                primary = next(iter(speaker_refs))
                for s in segments:
                    s["speaker"] = primary

        # Case C: diarization failed -> build ONE clean reference from long segments
        if not speaker_refs:
            log.info("[ref] No user upload + diarization unavailable - "
                     "building fallback single-speaker reference from source")
            fb_path = str(work / "speaker_refs" / "ref_fallback.wav")
            (work / "speaker_refs").mkdir(exist_ok=True)
            fb = extract_fallback_reference(audio_16k, segments, fb_path, duration=30.0)
            if fb:
                speaker_refs["SPEAKER_00"] = fb
                source_speaker_refs["SPEAKER_00"] = fb  # same source as above
                # Same cross-lingual guard as Case B
                if effective_src == target_lang:
                    speaker_transcripts["SPEAKER_00"] = " ".join(
                        s["text"] for s in segments[:5]
                    )
                else:
                    speaker_transcripts["SPEAKER_00"] = ""
                for s in segments:
                    s["speaker"] = "SPEAKER_00"

        # ─── POST-PROCESS SEGMENTS ───────────────────────────────
        # WhisperX cuts on VAD (breath) boundaries, not sentence boundaries,
        # so natural sentences often get split at pauses. The resulting
        # micro-fragments (10-20 chars, <2s) give TTS too little context to
        # clone voice correctly and produce "кашка" output. This pass
        # merges continuations, absorbs orphan fragments, and splits
        # monster segments. See pipeline/segment_post.py for details.
        try:
            from pipeline.segment_post import postprocess_segments
            segments = postprocess_segments(segments)
        except Exception as e:
            log.warning(f"Segment postprocess failed (continuing with raw): {e}")

        n_speakers = len(set(s.get("speaker", "?") for s in segments))
        update(speaker_count=n_speakers, progress=42)

        # ─── CHECKPOINT 1: after transcription+diarization+speaker_refs ──
        # Speaker refs are built now, so /continue from this checkpoint has
        # everything it needs to run translate → TTS → merge.
        _save_checkpoint(job_id, work, stage="transcription_done", data={
            "video_path": video_path,
            "audio_16k": audio_16k,
            "bg_audio_path": bg_audio_path,
            "duration": duration,
            "effective_src": effective_src,
            "target_lang": target_lang,
            "keep_bg": keep_bg,
            "model": model,
            "context_hint": context_hint,
            "speaker_mode": speaker_mode,
            "reference_audio": reference_audio,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "speaker_refs": {k: v for k, v in speaker_refs.items()},
            "source_speaker_refs": {k: v for k, v in source_speaker_refs.items()},
            "speaker_transcripts": {k: v for k, v in speaker_transcripts.items()},
            "segments": [
                {
                    "idx": i, "start": s["start"], "end": s["end"],
                    "text": s["text"],
                    "speaker": s.get("speaker", "SPEAKER_00"),
                }
                for i, s in enumerate(segments)
            ],
        })

        if wizard_mode == "review_transcript":
            update(
                status="awaiting_transcript_review", progress=43,
                step_detail="Review transcription — edit or approve to continue",
                checkpoint_stage="transcription_done",
            )
            log.info(f"[wizard] Paused at transcript review for job {job_id}")
            return

        # 5. Translate
        update(status="translating", progress=45, step_detail=f"Translating to {target_lang}...")
        def _translate_progress(done, total, eta_sec):
            # Map translation progress into overall pipeline 45→62% range
            pct = 45 + int((done / max(total, 1)) * 17)
            eta_str = f" · ~{eta_sec // 60}m{eta_sec % 60}s left" if eta_sec > 30 else ""
            update(
                progress=min(pct, 62),
                step_detail=f"Translating batch {done}/{total}{eta_str}",
            )
        segments = await translate_segments(
            segments, effective_src, target_lang, model,
            context_hint=context_hint,
            progress_callback=_translate_progress,
        )

        # Sanity check: if a significant fraction of segments have
        # untranslated (source-language) text still in translated_text,
        # stop here instead of letting VoxCPM try to speak English with
        # Russian cross-lingual cfg (which crashes the worker). This
        # happens when Ollama times out on every request and per-line
        # fallback also fails.
        untranslated_count = 0
        for s in segments:
            tt = (s.get("translated_text") or "").strip()
            src_text = (s.get("text") or "").strip()
            if not tt or tt == src_text:
                untranslated_count += 1
        if untranslated_count == len(segments):
            raise RuntimeError(
                f"Translation completely failed — all {len(segments)} segments "
                f"still in source language. Check Ollama: run `ollama ps` and "
                f"try `ollama run {model} 'hi'` manually. If it hangs, the "
                f"model may be incompatible with your setup; try "
                f"`ollama pull qwen2.5:7b` and pick it in the UI."
            )
        if untranslated_count > len(segments) // 2:
            log.warning(
                f"[translate] {untranslated_count}/{len(segments)} segments "
                f"did not translate successfully — TTS quality may suffer"
            )

        # Unload Ollama model from VRAM before TTS. Without this, Ollama's
        # 9+ GB model sits in VRAM during TTS, leaving too little room for
        # VoxCPM (also ~4 GB). On 12 GB cards this causes VoxCPM to swap
        # to system RAM → slow inference. keep_alive=0 tells Ollama to
        # drop the model immediately after the next request; we pair it
        # with a cheap 1-token request to actually trigger the unload.
        try:
            await unload_ollama_model(model)
        except Exception as e:
            log.warning(f"Failed to unload Ollama model (non-fatal): {e}")

        transcript_preview = [
            {
                "start": s["start"], "end": s["end"],
                "text": s["text"],
                "translated": s.get("translated_text", ""),
                "speaker": s.get("speaker", ""),
            }
            for s in segments
        ]
        update(transcript=transcript_preview, progress=62)

        # ─── CHECKPOINT 2: after translation ──────────────────────────
        # Full state dump — retry_tts can load this and skip stages 1-5.
        srt_path = str(work / "subtitles.srt")
        write_srt(segments, srt_path)
        update(srt_url=f"/outputs/{job_id}/subtitles.srt")

        _save_checkpoint(job_id, work, stage="translation_done", data={
            "video_path": video_path,
            "audio_16k": audio_16k,
            "bg_audio_path": bg_audio_path,
            "duration": duration,
            "effective_src": effective_src,
            "target_lang": target_lang,
            "keep_bg": keep_bg,
            "speaker_refs": {k: v for k, v in speaker_refs.items()},
            "source_speaker_refs": {k: v for k, v in source_speaker_refs.items()},
            "speaker_transcripts": {k: v for k, v in speaker_transcripts.items()},
            "reference_audio": reference_audio,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "segments": [
                {
                    "idx": i, "start": s["start"], "end": s["end"],
                    "text": s["text"],
                    "translated_text": s.get("translated_text", ""),
                    "speaker": s.get("speaker", "SPEAKER_00"),
                }
                for i, s in enumerate(segments)
            ],
        })

        if wizard_mode == "review_translation":
            update(
                status="awaiting_translation_review",
                progress=63,
                step_detail="Review translation — edit, retranslate, or approve to continue",
                checkpoint_stage="translation_done",
            )
            log.info(f"[wizard] Paused at translation review for job {job_id}")
            return

        # 6. Synthesize
        tts = get_tts_engine()
        tts_dir = str(work / "tts_segments")

        # ─── VOICE MODE ROUTING (see _run_tts_and_merge_stage) ──────────
        # VoxCPM has 3 mutually-exclusive modes; pick one based on user's
        # choice. NEVER mix style prefix with speaker refs — the model will
        # literally read the style description out loud in the cloned voice.
        has_ref = any(speaker_refs.values())
        if reference_audio and os.path.exists(reference_audio):
            first_pipeline_mode = "file_ref"   # user uploaded / file preset
        elif eff_style and eff_style.strip() and not has_ref:
            first_pipeline_mode = "voice_design"
        elif eff_style and eff_style.strip() and has_ref:
            # User picked a style preset but we already extracted refs from
            # the source video. The user presumably wants a fresh designed
            # voice — drop the source refs.
            log.info("[pipeline] Style preset + source refs → dropping refs "
                     "for Voice Design")
            speaker_refs = {}
            speaker_transcripts = {}
            first_pipeline_mode = "voice_design"
        else:
            first_pipeline_mode = "source_refs"

        update(status="synthesizing", progress=65,
               step_detail=f"Generating speech (mode={first_pipeline_mode}, "
                           f"preset={voice_preset}, seed={voice_seed})...",
               voice_mode=("upload" if first_pipeline_mode == "file_ref" else
                           ("custom" if first_pipeline_mode == "voice_design"
                            else "source")))

        # Apply Voice Design prefix ONLY in voice_design mode
        if first_pipeline_mode == "voice_design" and isinstance(tts, VoxCPMSynthesizer):
            style = eff_style.strip().strip("()")
            for s in segments:
                base = s.get("translated_text") or s.get("text", "")
                if base and not base.startswith("("):
                    s["translated_text"] = f"({style}){base}"

        def synth_progress(done, total):
            pct = 65 + int((done / max(total, 1)) * 20)
            update(progress=min(pct, 85), step_detail=f"Synthesizing: {done}/{total}")

        if isinstance(tts, VoxCPMSynthesizer):
            segments = tts.synthesize_segments(
                segments, tts_dir,
                speaker_refs=speaker_refs,
                speaker_transcripts=speaker_transcripts,
                progress_callback=synth_progress,
                voice_seed=voice_seed,
                tts_speed=tts_speed,
                is_cross_lingual=(effective_src != target_lang),
                target_lang=target_lang,
            )
        else:
            segments = await tts.synthesize_segments_async(
                segments, tts_dir, target_lang,
                progress_callback=synth_progress,
            )

        synth_ok = sum(1 for s in segments if s.get("audio_path"))
        update(progress=85, step_detail=f"Synthesized {synth_ok}/{len(segments)}")

        if synth_ok == 0:
            raise RuntimeError("All TTS synthesis failed - check model/GPU")

        # ─── CHECKPOINT 3: after TTS (for per-segment regen) ──────────
        # Each segment now has an audio_path; store that so /regenerate_segment
        # can pick up where we left off without re-synthesizing everything.
        # QA score and tier are surfaced so the UI can flag problematic
        # segments with coloured badges in the review panel.
        _save_checkpoint(job_id, work, stage="tts_done", data={
            "video_path": video_path,
            "audio_16k": audio_16k,
            "bg_audio_path": bg_audio_path,
            "duration": duration,
            "effective_src": effective_src,
            "target_lang": target_lang,
            "keep_bg": keep_bg,
            "speaker_refs": {k: v for k, v in speaker_refs.items()},
            "speaker_transcripts": {k: v for k, v in speaker_transcripts.items()},
            "reference_audio": reference_audio,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "sample_rate": tts.sample_rate if hasattr(tts, "sample_rate") else 48000,
            "segments": [
                {
                    "idx": i, "start": s["start"], "end": s["end"],
                    "text": s["text"],
                    "translated_text": s.get("translated_text", ""),
                    "speaker": s.get("speaker", "SPEAKER_00"),
                    "audio_path": s.get("audio_path", ""),
                    "qa_score": s.get("qa_score"),
                    "tts_tier": s.get("tts_tier"),
                }
                for i, s in enumerate(segments)
            ],
        })

        # 7. Assemble (with loudness normalization)
        update(status="assembling", progress=88, step_detail="Assembling dubbed audio...")
        dubbed_wav = str(work / "dubbed_audio.wav")
        assemble_dubbed_audio(segments, duration, dubbed_wav, tts.sample_rate, apply_loudnorm=True)
        _save_placements(work, segments)

        # 8. Merge with video
        update(status="merging", progress=93, step_detail="Rendering final video...")
        output_mp4 = str(work / "dubbed_video.mp4")
        merge_audio_video(video_path, dubbed_wav, output_mp4, bg_audio_path)

        update(
            status="complete",
            progress=100,
            output_url=f"/outputs/{job_id}/dubbed_video.mp4",
            completed_at=time.time(),
            step_detail="Done!",
        )
        log.info(f"Pipeline complete: {output_mp4}")

    except JobCancelled:
        # Re-raise so _job_queue_worker marks the job as 'cancelled'
        # rather than 'error'. Keep the exception on the stack — logging
        # is handled upstream.
        log.info(f"Pipeline cancelled for {job_id}")
        raise
    except Exception as e:
        update(status="error", error=str(e))
        log.exception(f"Pipeline failed: {e}")


# ─────────────────────────────────────────────────────────────
# API: System & Models
# ─────────────────────────────────────────────────────────────

@app.get("/api/system")
async def system_status():
    status = get_system_status()
    ollama_ok, ollama_models = await check_ollama()
    status["ollama"] = {
        "ok": ollama_ok,
        "models": ollama_models,
        "binary": status["ollama_binary"]["ok"],
    }
    status["catalog"] = MODEL_CATALOG

    tts_ready = status["voxcpm"]["ok"] or status["edge_tts"]["ok"]
    ready = (
        status["python"]["ok"] and
        status["ffmpeg"]["ok"] and
        status["yt_dlp"]["ok"] and
        status["whisper"]["ok"] and
        tts_ready and
        ollama_ok and
        len(ollama_models) > 0
    )
    status["ready"] = ready
    return status


@app.post("/api/models/pull")
async def pull_model(model: str = Form(...)):
    async def stream():
        try:
            async for event in ollama_pull_stream(model):
                total = event.get("total", 0)
                completed = event.get("completed", 0)
                st = event.get("status", "")
                pct = int(completed / total * 100) if total else 0
                payload = {"status": st, "completed": completed, "total": total, "percent": pct}
                yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'status': 'success', 'percent': 100})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/models/delete")
async def delete_model(model: str = Form(...)):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.delete(
                f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/delete",
                json={"name": model},
            )
            if r.status_code == 200:
                return {"ok": True}
            return JSONResponse({"error": r.text}, 400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/api/voxcpm/warmup")
async def voxcpm_warmup():
    try:
        tts = get_tts_engine()
        if isinstance(tts, VoxCPMSynthesizer):
            if not tts.is_loaded:
                await asyncio.get_event_loop().run_in_executor(None, tts.load)
            return {"ok": True, "loaded": True}
        return {"ok": True, "loaded": False, "fallback": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ─────────────────────────────────────────────────────────────
# API: Dubbing
# ─────────────────────────────────────────────────────────────

@app.post("/api/dub")
async def start_dub(
    source: str = Form(""),
    video: Optional[UploadFile] = File(None),
    reference: Optional[UploadFile] = File(None),
    source_lang: str = Form("auto"),
    target_lang: str = Form("ru"),
    model: str = Form("gemma4:e4b"),
    keep_bg: bool = Form(False),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),   # "main" | "all"
    context_hint: str = Form(""),
    voice_style: str = Form(""),
    voice_preset: str = Form("auto"),
    tts_speed: str = Form("balanced"),
    wizard_mode: str = Form("auto"),  # "auto" | "review_translation" | "review_transcript"
    auto_denoise: bool = Form(False),
    lip_sync: bool = Form(False),  # if True, auto-run Wav2Lip after pipeline completes
):
    # Validate translation model exists in Ollama - fall back gracefully otherwise.
    _ok, _installed = await check_ollama()
    if _ok and model not in _installed:
        # Preference order: fast non-thinking translation-specialized
        # models first, then general purpose, then thinking models last.
        # gemma4:e4b/e2b work but hang on 12 GB GPUs due to thinking
        # mode — kept as last-resort fallback only.
        _preferred = [
            "aya-expanse:8b",      # Cohere multilingual, best EN↔RU
            "mistral-nemo:12b",    # Mistral, strong for European langs
            "qwen2.5:14b",         # Qwen non-thinking, very good
            "qwen3:8b",            # Qwen3, thinking optional
            "qwen2.5:7b",          # Qwen smaller, fast
            "gemma3:12b",          # Gemma3 (no thinking) — good quality
            "gemma3:4b",           # Gemma3 small
            "llama3.2:3b",         # Tiny fallback
            "qwen3:14b",           # Larger qwen3 (thinking optional)
            "gemma4:e4b",          # Thinking — heavy on 12 GB GPU
            "gemma4:e2b",          # Thinking — smaller but same issue
        ]
        _fallback = next((m for m in _preferred if m in _installed), None)
        if _fallback is None and _installed:
            _fallback = _installed[0]
        if _fallback:
            log.warning(f"Requested model '{model}' not installed; using '{_fallback}' instead")
            model = _fallback
        else:
            return JSONResponse({
                "error": f"No translation model installed. Pull one via 'ollama pull aya-expanse:8b' "
                         f"or use the Models panel."
            }, 400)

    job_id = uuid.uuid4().hex[:8]
    work = OUTPUT_DIR / job_id
    work.mkdir(exist_ok=True)

    if video and video.filename:
        ext = Path(video.filename).suffix or ".mp4"
        vid_path = str(UPLOAD_DIR / f"{job_id}{ext}")
        with open(vid_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        actual_source = vid_path
        source_type = "upload"
        source_label = video.filename
    elif source:
        actual_source = source
        source_type = "url" if source.startswith("http") else "path"
        source_label = source
    else:
        return JSONResponse({"error": "Provide a YouTube URL or upload a video"}, 400)

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "source": actual_source,
        "source_type": source_type,
        "source_label": source_label,
        "target_lang": target_lang,
        "model": model,
        "speaker_mode": speaker_mode,
        "context_hint": context_hint,
        "voice_style": voice_style,
        "voice_preset": voice_preset,
        "voice_mode": ("upload" if ref_path else
                       ("custom" if voice_style.strip() else "preset")),
        "tts_speed": tts_speed,
        "wizard_mode": wizard_mode,
        "lip_sync": bool(lip_sync),
        "created": time.time(),
        "step_detail": "Queued...",
    }
    save_job(jobs[job_id])

    # Dispatch to pipeline — via queue if GPU is busy, else directly.
    # Multiple simultaneous dub requests would OOM the 12GB 3080 Ti
    # (WhisperX large-v3 + VoxCPM + pyannote = 9-10GB each). The queue
    # ensures only ONE GPU-heavy job runs at a time; others wait.
    await enqueue_job(job_id, {
        "source": actual_source,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": model,
        "keep_bg": keep_bg,
        "whisper_model": whisper_model,
        "reference_audio": ref_path,
        "speaker_mode": speaker_mode,
        "context_hint": context_hint,
        "voice_style": voice_style,
        "voice_preset": voice_preset,
        "tts_speed": tts_speed,
        "wizard_mode": wizard_mode,
        "auto_denoise": auto_denoise,
    })

    return {"job_id": job_id}


@app.post("/api/dub/batch")
async def start_batch_dub(
    sources: str = Form(""),  # newline-separated URLs OR json list
    videos: Optional[list[UploadFile]] = File(None),
    reference: Optional[UploadFile] = File(None),
    source_lang: str = Form("auto"),
    target_lang: str = Form("ru"),
    model: str = Form("aya-expanse:8b"),
    keep_bg: bool = Form(False),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),
    context_hint: str = Form(""),
    voice_style: str = Form(""),
    voice_preset: str = Form("auto"),
    tts_speed: str = Form("balanced"),
    wizard_mode: str = Form("auto"),  # Usually "auto" for batch — no pauses
    auto_denoise: bool = Form(False),
    batch_label: str = Form(""),  # optional: "BJJ Course Week 1" for summary
    scheduled_at: float = Form(0.0),  # unix epoch seconds; 0 = start immediately
):
    """Enqueue multiple videos for night-mode processing.

    Intended flow: user drops 5-10 videos in UI, picks a preset, clicks
    "Queue all". Each video becomes a separate job sharing common
    settings (target lang, voice, context). Jobs run serially via the
    GPU queue. Sleep prevention auto-activates while queue is non-empty.

    When scheduled_at is set to a future timestamp, jobs are created
    with status="scheduled" and a background task enqueues them at the
    target time. Useful for "queue this now, start at 2 AM when
    electricity is cheap" workflows.

    Returns: {job_ids: [...], batch_id: str} so UI can track summary.
    """
    # Validate Ollama model once (not per-job)
    _ok, _installed = await check_ollama()
    if _ok and model not in _installed:
        _preferred = ["aya-expanse:8b", "mistral-nemo:12b", "qwen2.5:14b",
                      "qwen3:8b", "qwen2.5:7b", "gemma3:12b", "gemma3:4b",
                      "llama3.2:3b", "qwen3:14b", "gemma4:e4b", "gemma4:e2b"]
        _fallback = next((m for m in _preferred if m in _installed), None)
        if _fallback:
            log.warning(f"Batch: '{model}' not installed; using '{_fallback}'")
            model = _fallback
        else:
            return JSONResponse({
                "error": "No translation model installed. Run: ollama pull aya-expanse:8b"
            }, 400)

    # Save shared reference once — all batch jobs reuse it
    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"batch_{uuid.uuid4().hex[:8]}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        log.info(f"[batch] Saved shared reference: {ref_path}")

    # Collect sources: URLs from form + uploaded files
    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    job_ids = []

    # 1. URLs (newline-separated or JSON list)
    url_list = []
    if sources.strip():
        s = sources.strip()
        if s.startswith("["):
            try:
                url_list = json.loads(s)
            except Exception:
                url_list = [ln.strip() for ln in s.splitlines() if ln.strip()]
        else:
            url_list = [ln.strip() for ln in s.splitlines() if ln.strip()]

    # When a scheduled_at is in the future, jobs are parked in the jobs
    # dict with status='scheduled' and a background task wakes them up
    # at the target time. Otherwise we enqueue immediately as before.
    now = time.time()
    is_scheduled = scheduled_at > now + 10  # 10s grace for clock skew
    initial_status = "scheduled" if is_scheduled else "queued"

    async def _enqueue_or_defer(jid, pipeline_args):
        if not is_scheduled:
            await enqueue_job(jid, pipeline_args)
        else:
            # Just leave status=scheduled; the scheduler task will pick it up
            log.info(f"[schedule] Job {jid} deferred until {scheduled_at}")

    for url in url_list:
        if not url:
            continue
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = {
            "id": jid,
            "status": initial_status,
            "progress": 0,
            "source": url,
            "source_type": "url",
            "source_label": url[:60] + ("..." if len(url) > 60 else ""),
            "target_lang": target_lang,
            "model": model,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "voice_mode": ("upload" if ref_path else
                          ("custom" if voice_style.strip() else "preset")),
            "tts_speed": tts_speed,
            "whisper_model": whisper_model,
            "keep_bg": keep_bg,
            "wizard_mode": wizard_mode,
            "auto_denoise": auto_denoise,
            "batch_id": batch_id,
            "batch_label": batch_label,
            "created": time.time(),
            "scheduled_at": scheduled_at if is_scheduled else 0,
            # When scheduled, stash full pipeline args on the job so the
            # scheduler can re-hydrate and enqueue later
            "_pending_args": ({
                "source": url, "source_lang": source_lang,
                "target_lang": target_lang, "model": model,
                "keep_bg": keep_bg, "whisper_model": whisper_model,
                "reference_audio": ref_path, "speaker_mode": speaker_mode,
                "context_hint": context_hint, "voice_style": voice_style,
                "voice_preset": voice_preset, "tts_speed": tts_speed,
                "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
            } if is_scheduled else None),
        }
        save_job(jobs[jid])
        await _enqueue_or_defer(jid, {
            "source": url, "source_lang": source_lang,
            "target_lang": target_lang, "model": model,
            "keep_bg": keep_bg, "whisper_model": whisper_model,
            "reference_audio": ref_path, "speaker_mode": speaker_mode,
            "context_hint": context_hint, "voice_style": voice_style,
            "voice_preset": voice_preset, "tts_speed": tts_speed,
            "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
        })
        job_ids.append(jid)

    # 2. Uploaded files
    for video in (videos or []):
        if not video.filename:
            continue
        jid = uuid.uuid4().hex[:8]
        video_ext = Path(video.filename).suffix or ".mp4"
        dest = UPLOAD_DIR / f"{jid}{video_ext}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(video.file, f)
        jobs[jid] = {
            "id": jid,
            "status": initial_status,
            "progress": 0,
            "source": str(dest),
            "source_type": "file",
            "source_label": video.filename,
            "target_lang": target_lang,
            "model": model,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "voice_mode": ("upload" if ref_path else
                          ("custom" if voice_style.strip() else "preset")),
            "tts_speed": tts_speed,
            "whisper_model": whisper_model,
            "keep_bg": keep_bg,
            "wizard_mode": wizard_mode,
            "auto_denoise": auto_denoise,
            "batch_id": batch_id,
            "batch_label": batch_label,
            "created": time.time(),
            "scheduled_at": scheduled_at if is_scheduled else 0,
            "_pending_args": ({
                "source": str(dest), "source_lang": source_lang,
                "target_lang": target_lang, "model": model,
                "keep_bg": keep_bg, "whisper_model": whisper_model,
                "reference_audio": ref_path, "speaker_mode": speaker_mode,
                "context_hint": context_hint, "voice_style": voice_style,
                "voice_preset": voice_preset, "tts_speed": tts_speed,
                "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
            } if is_scheduled else None),
        }
        save_job(jobs[jid])
        await _enqueue_or_defer(jid, {
            "source": str(dest), "source_lang": source_lang,
            "target_lang": target_lang, "model": model,
            "keep_bg": keep_bg, "whisper_model": whisper_model,
            "reference_audio": ref_path, "speaker_mode": speaker_mode,
            "context_hint": context_hint, "voice_style": voice_style,
            "voice_preset": voice_preset, "tts_speed": tts_speed,
            "wizard_mode": wizard_mode, "auto_denoise": auto_denoise,
        })
        job_ids.append(jid)

    if is_scheduled:
        log.info(f"[batch] {batch_id}: SCHEDULED {len(job_ids)} jobs for "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(scheduled_at))} "
                 f"(label: {batch_label or 'untitled'})")
    else:
        log.info(f"[batch] {batch_id}: enqueued {len(job_ids)} jobs "
                 f"(label: {batch_label or 'untitled'})")
    return {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "count": len(job_ids),
        "label": batch_label,
    }


@app.get("/api/dub/batch/{batch_id}")
async def get_batch_summary(batch_id: str):
    """Summary of a batch run — how many complete, failed, still running.
    Used by UI to show 'night-mode' dashboard."""
    batch_jobs = [j for j in jobs.values() if j.get("batch_id") == batch_id]
    if not batch_jobs:
        return JSONResponse({"error": "Batch not found"}, 404)
    total = len(batch_jobs)
    complete = sum(1 for j in batch_jobs if j.get("status") == "complete")
    errored = sum(1 for j in batch_jobs if j.get("status") == "error")
    queued = sum(1 for j in batch_jobs if j.get("status") == "queued")
    running = total - complete - errored - queued
    started = min((j.get("created", 0) for j in batch_jobs), default=0)
    finished = max((j.get("completed_at", j.get("created", 0))
                    for j in batch_jobs if j.get("status") in ("complete", "error")),
                   default=0)
    elapsed = (finished - started) if finished > started else (time.time() - started)
    return {
        "batch_id": batch_id,
        "label": batch_jobs[0].get("batch_label", ""),
        "total": total,
        "complete": complete,
        "errored": errored,
        "queued": queued,
        "running": running,
        "started": started,
        "finished": finished if complete + errored == total else None,
        "elapsed_sec": int(elapsed),
        "jobs": [{
            "id": j["id"],
            "label": j.get("source_label", j["id"]),
            "status": j.get("status"),
            "progress": j.get("progress", 0),
            "error": j.get("error", ""),
            "dubbed_url": (f"/outputs/{j['id']}/dubbed_video.mp4"
                          if j.get("status") == "complete" else None),
        } for j in sorted(batch_jobs, key=lambda x: x.get("created", 0))],
    }


# ═══════════════════════════════════════════════════════════════════════
#  Waveform preview endpoint — visualize reference audio
# ═══════════════════════════════════════════════════════════════════════
# Lets the UI show a waveform of reference audio BEFORE the user commits
# to using it. Useful for spotting silences, clipping, or picking the
# cleanest 15-second window from a longer file. Uses ffmpeg's built-in
# `showwavespic` filter — fast, no Python audio libs required.
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/waveform")
async def generate_waveform(
    audio: UploadFile = File(...),
    width: int = Form(800),
    height: int = Form(120),
):
    """Returns a PNG of the audio waveform. Accepts any FFmpeg-readable
    audio file. Width/height in pixels; defaults suit a typical UI panel."""
    import subprocess
    tmp_id = uuid.uuid4().hex[:8]
    ext = Path(audio.filename or "in.wav").suffix or ".wav"
    src = UPLOAD_DIR / f"wf_{tmp_id}{ext}"
    dst = UPLOAD_DIR / f"wf_{tmp_id}.png"
    try:
        with open(src, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-filter_complex",
             f"aformat=channel_layouts=mono,"
             f"compand=0|0:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2,"
             f"showwavespic=s={width}x{height}:colors=0xfb923c:split_channels=0",
             "-frames:v", "1", str(dst)],
            check=True, capture_output=True, timeout=30,
        )
        with open(dst, "rb") as f:
            png_data = f.read()
        return Response(content=png_data, media_type="image/png")
    except subprocess.CalledProcessError as e:
        log.warning(f"[waveform] ffmpeg failed: {e.stderr[:200]}")
        return JSONResponse({"error": "Could not generate waveform"}, 500)
    finally:
        for p in (src, dst):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════
#  Per-speaker reference inspection — diagnostic tool for review screen
# ═══════════════════════════════════════════════════════════════════════
# After diarization the pipeline extracts ~30s of clean speech per
# detected speaker into speaker_refs/ref_SPEAKER_XX.wav. These are fed
# to VoxCPM as voice-cloning references — bad refs = bad dubbed voice.
#
# These endpoints let the review UI inspect/audition the extracted refs
# so the user can diagnose "why does SPEAKER_01 sound wrong" BEFORE
# running TTS. PNGs are cached under speaker_refs/wf_*.png to avoid
# re-rendering the same waveform on every UI open.
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/job/{job_id}/speakers")
async def list_job_speakers(job_id: str):
    """Return per-speaker reference metadata for a job: list of
    {speaker, ref_path, duration_sec, exists} so the UI can enumerate
    detected speakers and render a waveform panel per speaker."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    refs_dir = OUTPUT_DIR / job_id / "speaker_refs"
    if not refs_dir.exists():
        return {"speakers": [], "hint": "No speaker references extracted for this job"}

    out = []
    for p in sorted(refs_dir.glob("ref_*.wav")):
        name = p.stem  # "ref_SPEAKER_00" or "ref_fallback"
        speaker = name.replace("ref_", "", 1)
        duration = 0.0
        try:
            # Fast duration read via ffprobe
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
                capture_output=True, text=True, timeout=10,
            )
            duration = float(r.stdout.strip() or 0.0)
        except Exception:
            pass
        out.append({
            "speaker": speaker,
            "duration_sec": round(duration, 1),
            "audio_url": f"/api/job/{job_id}/speaker_ref/{speaker}/audio",
            "waveform_url": f"/api/job/{job_id}/speaker_ref/{speaker}/waveform",
        })
    return {"speakers": out}


@app.get("/api/job/{job_id}/speaker_ref/{speaker}/audio")
async def get_speaker_ref_audio(job_id: str, speaker: str):
    """Stream the speaker's reference WAV for in-browser playback.
    Lets the user quickly audition the extracted reference."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    # Strict filename validation — prevent path traversal via speaker id
    if not re.match(r"^[A-Za-z0-9_]+$", speaker):
        return JSONResponse({"error": "Invalid speaker id"}, 400)
    p = OUTPUT_DIR / job_id / "speaker_refs" / f"ref_{speaker}.wav"
    if not p.exists():
        return JSONResponse({"error": "Reference not found"}, 404)
    return FileResponse(str(p), media_type="audio/wav",
                        filename=f"ref_{speaker}.wav")


@app.get("/api/job/{job_id}/speaker_ref/{speaker}/waveform")
async def get_speaker_ref_waveform(
    job_id: str, speaker: str,
    width: int = 700, height: int = 80,
):
    """Return a cached PNG waveform for this speaker. Caches to
    speaker_refs/wf_{speaker}_{w}x{h}.png so repeated opens don't
    re-run ffmpeg. Cache invalidates only when the ref WAV's mtime
    changes (e.g. user re-uploaded via edit_speaker_ref)."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if not re.match(r"^[A-Za-z0-9_]+$", speaker):
        return JSONResponse({"error": "Invalid speaker id"}, 400)
    refs_dir = OUTPUT_DIR / job_id / "speaker_refs"
    src = refs_dir / f"ref_{speaker}.wav"
    if not src.exists():
        return JSONResponse({"error": "Reference not found"}, 404)

    cache = refs_dir / f"wf_{speaker}_{width}x{height}.png"
    # Cache hit only if cached file exists AND was modified after source.
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return FileResponse(str(cache), media_type="image/png")

    # Miss: render + cache
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-filter_complex",
             f"aformat=channel_layouts=mono,"
             f"compand=0|0:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2,"
             f"showwavespic=s={width}x{height}:colors=0xfb923c:split_channels=0",
             "-frames:v", "1", str(cache)],
            check=True, capture_output=True, timeout=30,
        )
    except subprocess.CalledProcessError as e:
        log.warning(f"[waveform] per-speaker render failed: {e.stderr[:200]}")
        return JSONResponse({"error": "Could not render waveform"}, 500)
    return FileResponse(str(cache), media_type="image/png")


# ═══════════════════════════════════════════════════════════════════════
#  Subtitle burn-in — overlay SRT onto dubbed video
# ═══════════════════════════════════════════════════════════════════════
# Optional post-processing: take the dubbed video + translated SRT and
# produce a version with hard-coded subtitles burned into the frame.
# Useful for YouTube where auto-generated CC is often wrong.
# Uses ffmpeg's `subtitles` filter (relies on libass).
# ═══════════════════════════════════════════════════════════════════════

# Subtitle styling presets — shared by burn-in and preview endpoints.
# ffmpeg 'subtitles' filter uses libass force_style syntax. BorderStyle=1
# is outline+shadow; 3 is opaque box.
SUB_STYLE_MAP = {
    "default": "Fontsize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=28",
    "large":   "Fontsize=30,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=3,Shadow=1,MarginV=40,Bold=1",
    "minimal": "Fontsize=18,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=20",
    # Yellow "classic cinema" style — high-legibility for action footage
    "yellow":  "Fontsize=24,PrimaryColour=&H00FFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=2,MarginV=30,Bold=1",
    # Opaque box for noisy backgrounds (e.g. bright snow, chaotic action)
    "boxed":   "Fontsize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BackColour=&HC0000000,BorderStyle=3,Outline=0,Shadow=0,MarginV=28",
}


@app.post("/api/dub/{job_id}/subs_preview")
async def preview_subtitle_style(
    job_id: str,
    style: str = Form("default"),
    timestamp: float = Form(-1.0),  # seconds into video; -1 = auto-pick
):
    """Render a single frame from the dubbed video with subs overlaid in
    the given style — lets the user preview styling instantly instead of
    waiting for a full re-encode of the whole video.

    Timestamp selection:
      - User can specify a timestamp (UI may tie this to a scrub bar)
      - If -1, we auto-pick the middle of a segment that has subtitle
        text so the preview actually shows text (not a silent frame)
    """
    import subprocess
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if style not in SUB_STYLE_MAP:
        return JSONResponse(
            {"error": f"Unknown style '{style}'. Options: {list(SUB_STYLE_MAP)}"}, 400)

    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    srt_file = work / "translated.srt"

    if not src_video.exists():
        return JSONResponse({"error": "Dubbed video not yet generated"}, 400)

    if not srt_file.exists():
        cp_path = work / "checkpoint_tts_done.json"
        if not cp_path.exists():
            cp_path = work / "checkpoint_translation_done.json"
        if cp_path.exists():
            try:
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                _write_srt_file(cp.get("segments", []), srt_file)
            except Exception as e:
                return JSONResponse({"error": f"Could not generate SRT: {e}"}, 500)
        else:
            return JSONResponse({"error": "No transcript data found"}, 400)

    # Auto-pick: find a segment with text that lasts at least 1s
    if timestamp < 0:
        try:
            cp_path = work / "checkpoint_tts_done.json"
            if not cp_path.exists():
                cp_path = work / "checkpoint_translation_done.json"
            if cp_path.exists():
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                for seg in cp.get("segments", []):
                    text = (seg.get("translated_text") or seg.get("text") or "").strip()
                    dur = seg.get("end", 0) - seg.get("start", 0)
                    if text and dur >= 1.0:
                        # Pick middle of segment so sub is guaranteed visible
                        timestamp = seg["start"] + dur / 2
                        break
        except Exception:
            pass
        if timestamp < 0:
            timestamp = 2.0  # fallback

    srt_arg = str(srt_file).replace("\\", "/").replace(":", r"\:")
    force_style = SUB_STYLE_MAP[style]

    # Render one frame at <timestamp> with subs overlaid. Use -ss BEFORE
    # -i for fast seek (less accurate but saves ~10x on long videos),
    # and -frames:v 1 to output just one PNG.
    out_png = work / f"subs_preview_{style}.png"
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", f"{timestamp:.2f}",
             "-i", str(src_video),
             "-vf", f"subtitles='{srt_arg}':force_style='{force_style}'",
             "-frames:v", "1",
             "-q:v", "3",  # good quality JPEG-equivalent
             str(out_png)],
            check=True, capture_output=True, timeout=30,
        )
        # Return a JSON URL so the browser can cache-bust the image.
        # The PNG is already accessible via the /outputs static mount.
        return JSONResponse({
            "url": f"/outputs/{job_id}/subs_preview_{style}.png?t={int(time.time())}"
        })
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
        log.warning(f"[subs_preview] ffmpeg failed: {err_msg}")
        return JSONResponse({"error": "Preview render failed",
                             "detail": err_msg[:300]}, 500)


@app.post("/api/dub/{job_id}/burn_subs")
async def burn_subtitles(
    job_id: str,
    style: str = Form("default"),  # "default" | "large" | "minimal" | "yellow" | "boxed"
):
    """Generate a version of the dubbed video with burned-in subtitles
    from the translated SRT. Produces dubbed_video_subs.mp4 in the job dir."""
    import subprocess
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    srt_file = work / "translated.srt"
    dst_video = work / "dubbed_video_subs.mp4"

    if not src_video.exists():
        return JSONResponse({"error": "Dubbed video not yet generated"}, 400)

    # Ensure SRT exists (it's written alongside translation checkpoint,
    # but regenerate if missing using current segments)
    if not srt_file.exists():
        cp_path = work / "checkpoint_tts_done.json"
        if not cp_path.exists():
            cp_path = work / "checkpoint_translation_done.json"
        if cp_path.exists():
            try:
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                segments = cp.get("segments", [])
                _write_srt_file(segments, srt_file)
            except Exception as e:
                return JSONResponse({"error": f"Could not generate SRT: {e}"}, 500)
        else:
            return JSONResponse({"error": "No transcript data found"}, 400)

    # Use the shared style map (preview + burn-in stay in sync)
    force_style = SUB_STYLE_MAP.get(style, SUB_STYLE_MAP["default"])

    # ffmpeg needs forward-slash path and escaped colons on Windows
    # (the subtitles filter parses its argument like a filter string)
    srt_arg = str(srt_file).replace("\\", "/").replace(":", r"\:")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_video),
             "-vf", f"subtitles='{srt_arg}':force_style='{force_style}'",
             "-c:a", "copy",  # don't re-encode audio
             "-preset", "fast",
             str(dst_video)],
            check=True, capture_output=True, timeout=600,
        )
        return {
            "ok": True,
            "url": f"/outputs/{job_id}/dubbed_video_subs.mp4?v={int(time.time())}",
        }
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
        log.warning(f"[burn_subs] ffmpeg failed for {job_id}: {err_msg}")
        return JSONResponse({
            "error": "Subtitle burn-in failed",
            "detail": err_msg[:300],
        }, 500)


def _write_srt_file(segments: list, dst: Path):
    """Write segments as an SRT file (SubRip format)."""
    def _fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60)
        s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _fmt(seg.get("start", 0.0))
        end = _fmt(seg.get("end", 0.0))
        text = (seg.get("translated_text") or seg.get("text") or "").strip()
        # Strip emotion tags like "(happy)" that are TTS-only and shouldn't
        # appear in subtitle text; keep only the spoken part
        text = re.sub(r"^\s*\([^)]+\)\s*", "", text).strip()
        if not text:
            continue
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    dst.write_text("\n".join(lines), encoding="utf-8")


def _trim_video(src: Path, dst: Path, seconds: int) -> Path:
    """Trim src to the first `seconds` seconds into dst.

    Tries stream-copy first (fast, requires keyframe alignment); on failure
    falls back to re-encode with libx264. Raises subprocess.CalledProcessError
    if both attempts fail. Returns dst on success.
    """
    import subprocess
    seconds = max(1, int(seconds))
    # Attempt 1 — stream copy. Works when keyframes align with the cut point.
    try:
        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", "0",
             "-i", str(src),
             "-t", str(seconds),
             "-c", "copy",
             "-avoid_negative_ts", "make_zero",
             str(dst)],
            check=True, capture_output=True, timeout=60,
        )
        log.info(f"[trim] {src.name} -> {dst.name} ({seconds}s, stream-copy)")
        return dst
    except subprocess.CalledProcessError as e1:
        log.warning(f"[trim] stream-copy failed for {src.name}: "
                    f"{(e1.stderr or b'').decode('utf-8', errors='replace')[-200:]}")
    # Attempt 2 — re-encode. Slower but works on any source.
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", str(src),
         "-t", str(seconds),
         "-c:v", "libx264", "-preset", "veryfast",
         "-c:a", "aac", "-b:a", "128k",
         str(dst)],
        check=True, capture_output=True, timeout=120,
    )
    log.info(f"[trim] {src.name} -> {dst.name} ({seconds}s, re-encoded)")
    return dst


# Default language picks for the Quick-Test feature. Frontend allows
# per-run override; this is just the pre-selection.
_QUICK_TEST_DEFAULT_LANGS = ("es", "fr", "de", "ja", "pt")
_QUICK_TEST_KNOWN_LANGS = {
    "en", "ru", "es", "pt", "fr", "de", "it", "pl", "tr",
    "ja", "ko", "zh", "ar", "hi", "nl",
}


@app.post("/api/quick_test")
async def start_quick_test(
    video: Optional[UploadFile] = File(None),
    source: str = Form(""),                # YouTube/direct URL
    reference: Optional[UploadFile] = File(None),
    trim_seconds: int = Form(60),
    target_langs: str = Form(""),          # comma-separated e.g. "es,fr,de,ja,pt"
    source_lang: str = Form("auto"),
    model: str = Form("aya-expanse:8b"),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),
    voice_preset: str = Form("auto"),
    voice_style: str = Form(""),
    tts_speed: str = Form("balanced"),
    keep_bg: bool = Form(False),
    auto_denoise: bool = Form(False),
    context_hint: str = Form(""),
    batch_label: str = Form(""),
):
    """Quick-test: trim a short clip and fan out into N normal dub jobs
    (one per target language) sharing a batch_id. The user gets a side-by-
    side comparison in the Batch view.
    """
    # ── Validate inputs ───────────────────────────────────────────────
    if not video and not source.strip():
        return JSONResponse({"error": "Provide either a video file or a URL"}, 400)
    if video and source.strip():
        return JSONResponse({"error": "Provide only one of video or url"}, 400)

    if trim_seconds < 15 or trim_seconds > 120:
        return JSONResponse(
            {"error": f"trim_seconds must be between 15 and 120 (got {trim_seconds})"},
            400,
        )

    langs = [c.strip() for c in target_langs.split(",") if c.strip()]
    if not (2 <= len(langs) <= 6):
        return JSONResponse(
            {"error": f"Pick 2-6 target languages (got {len(langs)})"}, 400)
    unknown = [c for c in langs if c not in _QUICK_TEST_KNOWN_LANGS]
    if unknown:
        return JSONResponse(
            {"error": f"Unknown language code(s): {unknown}"}, 400)
    if len(set(langs)) != len(langs):
        return JSONResponse({"error": "Duplicate language codes"}, 400)

    # ── Validate Ollama model (same fallback logic as start_batch_dub) ─
    _ok, _installed = await check_ollama()
    if _ok and model not in _installed:
        _preferred = ["aya-expanse:8b", "mistral-nemo:12b", "qwen2.5:14b",
                      "qwen3:8b", "qwen2.5:7b", "gemma3:12b", "gemma3:4b",
                      "llama3.2:3b", "qwen3:14b", "gemma4:e4b", "gemma4:e2b"]
        _fallback = next((m for m in _preferred if m in _installed), None)
        if _fallback:
            log.warning(f"[quick_test] '{model}' not installed; using '{_fallback}'")
            model = _fallback
        else:
            return JSONResponse({
                "error": "No translation model installed. Run: ollama pull aya-expanse:8b"
            }, 400)

    # ── Save shared reference (one upload, reused by all jobs) ────────
    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"qt_{uuid.uuid4().hex[:8]}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        log.info(f"[quick_test] Saved shared reference: {ref_path}")

    # ── Materialize the source file locally ───────────────────────────
    # File uploads write straight to uploads/. URLs go through yt-dlp first
    # so the trim step is centralized — we don't fan out N downloads.
    src_path: Path
    src_label: str
    if video and video.filename:
        ext = Path(video.filename).suffix or ".mp4"
        src_path = UPLOAD_DIR / f"qt_{uuid.uuid4().hex[:8]}{ext}"
        with open(src_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        src_label = video.filename
    else:
        from pipeline.downloader import download_video
        url = source.strip()
        src_label = url[:60] + ("..." if len(url) > 60 else "")
        try:
            dl_dir = UPLOAD_DIR / f"qt_{uuid.uuid4().hex[:8]}"
            dl_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(download_video(url, str(dl_dir)))
        except Exception as e:
            return JSONResponse(
                {"error": f"Could not download URL: {e}"}, 400)

    # ── Trim ──────────────────────────────────────────────────────────
    trimmed_path = src_path.parent / f"{src_path.stem}_qt{trim_seconds}s.mp4"
    try:
        _trim_video(src_path, trimmed_path, trim_seconds)
    except Exception as e:
        err_msg = ""
        if hasattr(e, "stderr") and getattr(e, "stderr", None):
            err_msg = e.stderr.decode("utf-8", errors="replace")[-300:]
        log.warning(f"[quick_test] trim failed: {e} :: {err_msg}")
        return JSONResponse(
            {"error": "Could not trim video", "detail": err_msg or str(e)}, 500)

    # ── Fan out: one job per language, all sharing one batch_id ───────
    batch_id = f"qt_{uuid.uuid4().hex[:8]}"
    label_final = batch_label or f"Quick test · {trim_seconds}s · {len(langs)} langs"
    job_ids: list = []

    for idx, lang in enumerate(langs):
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = {
            "id": jid,
            "status": "queued",
            "progress": 0,
            "source": str(trimmed_path),
            "source_type": "file",
            "source_label": f"{src_label} -> {lang.upper()}",
            "target_lang": lang,
            "model": model,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "voice_mode": ("upload" if ref_path else
                          ("custom" if voice_style.strip() else "preset")),
            "tts_speed": tts_speed,
            "whisper_model": whisper_model,
            "keep_bg": keep_bg,
            "wizard_mode": "auto",     # never pause in quick-test mode
            "auto_denoise": auto_denoise,
            "batch_id": batch_id,
            "batch_label": label_final,
            "batch_kind": "quick_test",
            "batch_position": idx,
            "batch_total": len(langs),
            "created": time.time(),
            "scheduled_at": 0,
            "_pending_args": None,
        }
        save_job(jobs[jid])
        await enqueue_job(jid, {
            "source": str(trimmed_path),
            "source_lang": source_lang,
            "target_lang": lang,
            "model": model,
            "keep_bg": keep_bg,
            "whisper_model": whisper_model,
            "reference_audio": ref_path,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "wizard_mode": "auto",
            "auto_denoise": auto_denoise,
        })
        job_ids.append(jid)

    log.info(f"[quick_test] {batch_id}: enqueued {len(job_ids)} jobs "
             f"({trim_seconds}s, langs={langs})")

    return {
        "ok": True,
        "batch_id": batch_id,
        "batch_kind": "quick_test",
        "job_ids": job_ids,
        "count": len(job_ids),
        "trimmed_file": f"/uploads/{trimmed_path.name}",
        "trim_seconds": trim_seconds,
        "target_langs": langs,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Multilingual Showcase — same source split into N language segments
#  stitched back into one continuous video with corner language badges.
# ═══════════════════════════════════════════════════════════════════════
# Flow:
#   1. Same as Quick Test — fan out N full-length dubs, one per language.
#   2. When all sibling jobs finish, _maybe_assemble_showcase() reads the
#      source transcript, picks ~equal time slices snapped to the nearest
#      sentence-end boundary, trims each dub to its slice, overlays a
#      "· LL ·" badge in the top-right, and concatenates into one mp4.

_SHOWCASE_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


def _find_drawtext_font() -> str:
    """Locate a usable TTF/TTC for ffmpeg drawtext. Empty string = let
    ffmpeg fall back to its default (may fail on some Windows builds)."""
    for c in _SHOWCASE_FONT_CANDIDATES:
        if Path(c).exists():
            return c
    return ""


def _snap_boundaries_to_sentences(segments: list, total_dur: float, n_parts: int) -> list:
    """Compute n_parts contiguous slices that together cover [0, total_dur].

    Slices are equal time chunks (`total_dur / n_parts`), with each interior
    boundary snapped to the nearest segment END time. Returns a list of
    (start, end) tuples — guaranteed contiguous, non-empty, sorted.
    """
    if n_parts <= 1 or not segments:
        return [(0.0, float(total_dur))]

    seg_ends = sorted({float(s.get("end", 0.0)) for s in segments if s.get("end")})
    seg_ends = [e for e in seg_ends if 0 < e < total_dur]

    target_each = total_dur / n_parts
    boundaries = []
    prev = 0.0
    for i in range(1, n_parts):
        target = i * target_each
        # Snap to the nearest segment end, but never go backwards past `prev`
        # (otherwise we'd get a zero-length or negative slice).
        candidates = [e for e in seg_ends if e > prev + 0.5]
        if candidates:
            snapped = min(candidates, key=lambda e: abs(e - target))
        else:
            snapped = target
        snapped = max(snapped, prev + 0.5)
        snapped = min(snapped, total_dur - (n_parts - i) * 0.5)
        boundaries.append(snapped)
        prev = snapped

    slices = []
    last = 0.0
    for b in boundaries:
        slices.append((last, b))
        last = b
    slices.append((last, float(total_dur)))
    return slices


def _save_placements(work_dir: Path, segments: list) -> None:
    """Write tts_placements.json next to dubbed_video.mp4. Records where each
    segment ended up in the final dubbed track (placed_start/end), which may
    differ from the source-time start/end due to overlap-pushing and atempo
    stretching in the assembler. Showcase reels need these to cut each dub
    at its OWN word boundaries instead of at fixed source timestamps."""
    try:
        rows = []
        for seg in segments:
            ps = seg.get("placed_start")
            pe = seg.get("placed_end")
            if ps is None or pe is None:
                continue
            rows.append({
                "idx": seg.get("idx"),
                "src_start": float(seg.get("start", 0.0)),
                "src_end": float(seg.get("end", 0.0)),
                "dub_start": float(ps),
                "dub_end": float(pe),
            })
        out = work_dir / "tts_placements.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if rows:
            log.info(f"[placements] saved {len(rows)} rows -> {out.name} "
                     f"(dub range [{rows[0]['dub_start']:.1f}–{rows[-1]['dub_end']:.1f}s])")
        else:
            log.warning(f"[placements] 0 rows written to {out} — "
                        "assembler did not set placed_start/end on any segment")
    except Exception as e:
        log.warning(f"[placements] failed to save: {e}")


def _load_placements(work_dir: Path) -> list:
    """Inverse of _save_placements. Returns [] if file missing/unparseable."""
    p = work_dir / "tts_placements.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning(f"[placements] failed to load {p}: {e}")
        return []


def _probe_duration(path: Path) -> float:
    """ffprobe a media file and return duration in seconds (0.0 on failure).

    Tries ffprobe first (fast), falls back to parsing `ffmpeg -i` stderr
    if ffprobe isn't available. Logs the reason on every failure so we
    don't get silent zero-returns.
    """
    import subprocess  # follow existing per-function-import pattern
    if not Path(path).exists():
        log.warning(f"[probe] file not found: {path}")
        return 0.0
    # Try ffprobe
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True, timeout=15,
        )
        d = float((r.stdout or "0").strip() or 0)
        if d > 0:
            return d
    except FileNotFoundError:
        log.warning(f"[probe] ffprobe not on PATH — falling back to ffmpeg -i")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        log.warning(f"[probe] ffprobe failed for {path}: {err[-200:].strip() or e}")
    except Exception as e:
        log.warning(f"[probe] ffprobe error for {path}: {type(e).__name__}: {e}")

    # Fallback: parse "Duration: HH:MM:SS.ss" from ffmpeg -i stderr.
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        import re as _re
        m = _re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", r.stderr or "")
        if m:
            h, mn, s = m.groups()
            return float(h) * 3600 + float(mn) * 60 + float(s)
        log.warning(f"[probe] ffmpeg fallback: no Duration in stderr for {path}")
    except FileNotFoundError:
        log.warning("[probe] ffmpeg not on PATH either — both probes failed")
    except Exception as e:
        log.warning(f"[probe] ffmpeg fallback failed: {type(e).__name__}: {e}")
    return 0.0


_showcase_assembling: set = set()  # in-progress batch IDs (de-dupe re-entry)
_showcase_tasks: set = set()       # strong refs so asyncio GC doesn't kill them


async def _maybe_assemble_showcase(batch_id: str) -> None:
    """Hook called after each job finishes. If all jobs in `batch_id` are
    complete and this batch is a 'showcase', assemble the combined reel.
    No-op otherwise. Safe to call multiple times — guarded by status check
    and an in-progress set."""
    log.info(f"[showcase] _maybe_assemble_showcase('{batch_id}') called")
    if not batch_id:
        log.info("[showcase] empty batch_id, skipping")
        return
    if batch_id in _showcase_assembling:
        log.info(f"[showcase] {batch_id} already assembling, skipping")
        return

    # Collect sibling jobs
    siblings = [j for j in jobs.values()
                if j.get("batch_id") == batch_id
                and j.get("batch_kind") == "showcase"]
    if not siblings:
        log.warning(f"[showcase] no siblings found for {batch_id}")
        return
    expected_total = max((j.get("batch_total") or 0) for j in siblings)
    if expected_total and len(siblings) < expected_total:
        log.info(f"[showcase] {batch_id}: only {len(siblings)}/{expected_total} jobs registered, waiting")
        return

    # All must be complete (not error/queued/running)
    statuses = [j.get("status") for j in siblings]
    if not all(s == "complete" for s in statuses):
        log.info(f"[showcase] {batch_id}: not all complete (statuses={statuses})")
        return

    # Already assembled? Bail.
    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    out_mp4 = showcase_dir / "showcase.mp4"
    if out_mp4.exists():
        log.info(f"[showcase] {batch_id}: already assembled at {out_mp4}")
        return

    log.info(f"[showcase] {batch_id}: all checks passed, kicking off ffmpeg")
    _showcase_assembling.add(batch_id)
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, _assemble_showcase_sync, batch_id, siblings, showcase_dir
        )
    except Exception as e:
        log.error(f"[showcase] {batch_id}: assembler crashed: {e}", exc_info=True)
    finally:
        _showcase_assembling.discard(batch_id)


def _assemble_showcase_sync(batch_id: str, siblings: list, showcase_dir: Path) -> None:
    """Synchronous worker for showcase assembly. Runs in a thread to keep
    the event loop responsive (ffmpeg is blocking)."""
    import subprocess  # follow existing per-function-import pattern
    siblings = sorted(siblings, key=lambda j: j.get("batch_position", 0))
    n = len(siblings)
    log.info(f"[showcase] {batch_id}: assembling {n} language segments…")

    # ── Load source segment times (any sibling has them; pick the first) ─
    segments: list = []
    for j in siblings:
        work = OUTPUT_DIR / j["id"]
        for cp_name in ("checkpoint_translation_done.json",
                        "checkpoint_transcription_done.json"):
            cp = work / cp_name
            if cp.exists():
                try:
                    data = json.loads(cp.read_text(encoding="utf-8"))
                    segments = data.get("segments", []) or []
                    if segments:
                        break
                except Exception as e:
                    log.warning(f"[showcase] couldn't parse {cp}: {e}")
        if segments:
            break

    # ── Verify all dub files exist ────────────────────────────────────
    missing = [OUTPUT_DIR / j["id"] / "dubbed_video.mp4"
               for j in siblings
               if not (OUTPUT_DIR / j["id"] / "dubbed_video.mp4").exists()]
    if missing:
        log.error(f"[showcase] {batch_id}: {len(missing)} dub file(s) missing — aborting")
        return

    # ── Pre-load ALL placements once, compute effective dub duration ──
    # The dubbed audio for each language is typically shorter than the source
    # video because TTS may run faster (atempo-stretched) and the assembler
    # trims trailing silence. The VIDEO container duration == source duration
    # (we pad the last frame), so probing the mp4 gives ~120s for a 120s
    # source — but the AUDIO ends at 95-106s. If we slice into [95-120s] of
    # a dub that has no audio there, we get silence in the showcase.
    # Solution: compute total_dur from the ACTUAL last placed segment in every
    # dub (max dub_end across placements), then take min so every language
    # has content throughout the full showcase.
    all_placements: dict = {}   # job_id -> list of placement rows
    effective_ends: list = []   # max dub_end per sibling
    for j in siblings:
        pl = _load_placements(OUTPUT_DIR / j["id"])
        all_placements[j["id"]] = pl
        if pl:
            effective_ends.append(max(p["dub_end"] for p in pl))
        else:
            # Fallback: probe dubbed audio track duration (not the mp4 container)
            # using ffprobe's stream-level query which returns audio stream duration.
            import subprocess as _sp
            audio_dur = 0.0
            try:
                r = _sp.run(
                    ["ffprobe", "-v", "error",
                     "-select_streams", "a:0",
                     "-show_entries", "stream=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1",
                     str(OUTPUT_DIR / j["id"] / "dubbed_video.mp4")],
                    check=True, capture_output=True, text=True, timeout=15,
                )
                audio_dur = float((r.stdout or "0").strip() or 0)
            except Exception:
                pass
            if audio_dur > 0:
                effective_ends.append(audio_dur)
            else:
                log.warning(f"[showcase] {j['id']}: no placements and audio probe failed; "
                            "showcase may include silent tail for this language")

    if not effective_ends:
        # True last-resort: use source video duration
        first_source = Path(siblings[0].get("source", ""))
        fallback_dur = _probe_duration(first_source) if first_source.exists() else 0.0
        if fallback_dur <= 0:
            log.error(f"[showcase] {batch_id}: could not determine any dub duration — aborting")
            return
        total_dur = fallback_dur
        log.warning(f"[showcase] using source duration as total_dur fallback ({total_dur:.1f}s)")
    else:
        total_dur = min(effective_ends)
        log.info(f"[showcase] effective dub durations: "
                 f"{[f'{d:.1f}' for d in effective_ends]}s → using min={total_dur:.1f}s")

    slices = _snap_boundaries_to_sentences(segments, total_dur, n)
    log.info(f"[showcase] source-time slices: {[f'{s:.1f}-{e:.1f}' for s, e in slices]}")

    # ── Map each source-time slice to per-dub time using placements ────
    # Key fix: use OVERLAP matching — a segment spans [src_start, src_end]
    # in source time and [dub_start, dub_end] in dub time. We want every
    # segment that OVERLAPS the slice [g_s, g_e], not just those whose start
    # falls inside. Without overlap matching, a long merged segment (e.g.
    # src [30-85s]) will cover several source slices but only be found for
    # the one whose boundary contains 30s. This caused 30s source slices to
    # map to only 7s of dub time (one tail segment found instead of all).
    LEAD = 0.05   # 50ms lead so we don't clip a word's onset
    TRAIL = 0.15  # 150ms trail for a clean release
    per_dub_ranges: list = []
    for slice_idx, (g_s, g_e) in enumerate(slices):
        job = siblings[slice_idx]
        placements = all_placements.get(job["id"], [])
        # Effective audio end for this dub (from placements or fallback)
        eff_end = (max(p["dub_end"] for p in placements)
                   if placements else effective_ends[slice_idx]
                   if slice_idx < len(effective_ends) else total_dur)
        # Overlap matching: segment overlaps slice if src_start < g_e AND src_end > g_s
        in_slice = [
            p for p in placements
            if p["src_start"] < g_e + 0.001 and p["src_end"] > g_s - 0.001
        ]
        if in_slice:
            # Key: include the SOURCE time range as well as the dub placement
            # range. In the dubbed video, non-speech gaps keep source timing
            # (assembler places segments at their source timestamps). So a
            # slice [67.5-79s] that has a speech segment placed at dub[70-72s]
            # should show dub[67.5-79s] — not just [70-72s] — to include the
            # gap/background content before and after the speech burst.
            d_start = max(0.0, min(min(p["dub_start"] for p in in_slice), g_s) - LEAD)
            d_end = min(eff_end, max(max(p["dub_end"] for p in in_slice), g_e) + TRAIL)
            # Sanity: never go past eff_end or before 0
            if d_end <= d_start + 0.1:
                d_start = max(0.0, g_s)
                d_end = min(eff_end, g_e)
            log.info(f"[showcase] slice {slice_idx} ({job.get('target_lang')}): "
                     f"src [{g_s:.2f}-{g_e:.2f}] -> dub [{d_start:.2f}-{d_end:.2f}] "
                     f"({len(in_slice)} segs, eff_end={eff_end:.1f}s)")
        else:
            # No speech in this slice — show source-equivalent gap content,
            # clamped to effective audio end.
            d_start = max(0.0, g_s)
            d_end = min(g_e, eff_end)
            if d_end <= d_start + 0.05:
                log.warning(f"[showcase] slice {slice_idx} ({job.get('target_lang')}): "
                            f"no speech and eff_end={eff_end:.1f}s < slice_start={g_s:.1f}s "
                            f"— skipping (will use 0.1s stub)")
                d_end = d_start + 0.1  # avoid zero-length segment crashing ffmpeg
            else:
                log.warning(f"[showcase] slice {slice_idx} ({job.get('target_lang')}): "
                            f"no speech in slice, showing gap [{d_start:.2f}-{d_end:.2f}]")
        per_dub_ranges.append((job, d_start, d_end))

    # ── Build ffmpeg filter_complex ───────────────────────────────────
    showcase_dir.mkdir(parents=True, exist_ok=True)
    font_path = _find_drawtext_font()
    # ffmpeg filter syntax requires escaped colons on Windows paths
    font_arg = font_path.replace("\\", "/").replace(":", r"\:") if font_path else ""

    filter_parts = []
    inputs = []
    for idx, (job, start, end) in enumerate(per_dub_ranges):
        src = OUTPUT_DIR / job["id"] / "dubbed_video.mp4"
        inputs.extend(["-i", str(src)])
        lang_code = (job.get("target_lang") or "").upper()
        label = f"· {lang_code} ·"  # · LL ·
        seg_dur = max(0.1, end - start)

        # Escape single quotes and special chars for drawtext text= field
        text_safe = label.replace("'", r"\'")

        drawtext = (
            f"drawtext="
            + (f"fontfile='{font_arg}':" if font_arg else "")
            + f"text='{text_safe}':"
            "fontsize=22:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=8:"
            "x=w-tw-24:y=24"
        )

        filter_parts.append(
            f"[{idx}:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS,{drawtext}[v{idx}]"
        )
        # apad+atrim guarantees the audio is EXACTLY seg_dur long: if the
        # dub's audio stream ends before `end` (TTS finished early) apad
        # fills with silence; atrim caps any overflow. Without this, ffmpeg
        # concat hands us a shorter audio stream than video and you get
        # the "audio cuts off at 50.6s while video runs 60s" bug.
        filter_parts.append(
            f"[{idx}:a]atrim=start={start:.3f}:end={end:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"apad=whole_dur={seg_dur:.3f},"
            f"atrim=duration={seg_dur:.3f}[a{idx}]"
        )

    # Concat all trimmed pieces
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[vout][aout]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(showcase_dir / "showcase.mp4"),
    ]

    out_dur = sum(e - s for _, s, e in per_dub_ranges)
    log.info(f"[showcase] running ffmpeg ({n} inputs, {out_dur:.1f}s out)")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-1200:]
        log.error(f"[showcase] ffmpeg failed:\n{err}")
        # Write a marker so the UI can surface the failure
        try:
            (showcase_dir / "error.txt").write_text(err, encoding="utf-8")
        except Exception:
            pass
        return

    # Write a manifest for the UI
    try:
        manifest = {
            "batch_id": batch_id,
            "created": time.time(),
            "n_segments": n,
            "slices": [
                {
                    "lang": j.get("target_lang"),
                    "src_start": float(src_s),
                    "src_end": float(src_e),
                    "dub_start": float(d_s),
                    "dub_end": float(d_e),
                    "job_id": j["id"],
                }
                for (j, d_s, d_e), (src_s, src_e) in zip(per_dub_ranges, slices)
            ],
            "total_seconds": float(out_dur),
        }
        (showcase_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        log.warning(f"[showcase] manifest write failed: {e}")

    log.info(f"[showcase] {batch_id}: done -> {showcase_dir / 'showcase.mp4'}")


# Default language picks for the Showcase feature — same defaults as
# Quick Test but kept separate so they can diverge if needed.
_SHOWCASE_DEFAULT_LANGS = _QUICK_TEST_DEFAULT_LANGS


@app.post("/api/showcase")
async def start_showcase(
    video: Optional[UploadFile] = File(None),
    source: str = Form(""),
    reference: Optional[UploadFile] = File(None),
    trim_seconds: int = Form(60),
    target_langs: str = Form(""),
    source_lang: str = Form("auto"),
    model: str = Form("aya-expanse:8b"),
    whisper_model: str = Form("large-v3"),
    speaker_mode: str = Form("main"),
    voice_preset: str = Form("auto"),
    voice_style: str = Form(""),
    tts_speed: str = Form("balanced"),
    keep_bg: bool = Form(False),
    auto_denoise: bool = Form(False),
    context_hint: str = Form(""),
    batch_label: str = Form(""),
):
    """Showcase: trim a short clip, fan out into N normal dub jobs, and
    when all finish, automatically stitch them into one multilingual reel
    (each segment in a different language with a corner badge)."""
    # ── Input validation — identical to /api/quick_test ───────────────
    if not video and not source.strip():
        return JSONResponse({"error": "Provide either a video file or a URL"}, 400)
    if video and source.strip():
        return JSONResponse({"error": "Provide only one of video or url"}, 400)
    if trim_seconds < 15 or trim_seconds > 120:
        return JSONResponse(
            {"error": f"trim_seconds must be between 15 and 120 (got {trim_seconds})"}, 400)

    langs = [c.strip() for c in target_langs.split(",") if c.strip()]
    if not (2 <= len(langs) <= 6):
        return JSONResponse({"error": f"Pick 2-6 target languages (got {len(langs)})"}, 400)
    unknown = [c for c in langs if c not in _QUICK_TEST_KNOWN_LANGS]
    if unknown:
        return JSONResponse({"error": f"Unknown language code(s): {unknown}"}, 400)
    if len(set(langs)) != len(langs):
        return JSONResponse({"error": "Duplicate language codes"}, 400)

    # Validate ollama model with fallback
    _ok, _installed = await check_ollama()
    if _ok and model not in _installed:
        _preferred = ["aya-expanse:8b", "mistral-nemo:12b", "qwen2.5:14b",
                      "qwen3:8b", "qwen2.5:7b", "gemma3:12b", "gemma3:4b",
                      "llama3.2:3b", "qwen3:14b", "gemma4:e4b", "gemma4:e2b"]
        _fallback = next((m for m in _preferred if m in _installed), None)
        if _fallback:
            log.warning(f"[showcase] '{model}' not installed; using '{_fallback}'")
            model = _fallback
        else:
            return JSONResponse({
                "error": "No translation model installed. Run: ollama pull aya-expanse:8b"
            }, 400)

    # Shared reference (one upload, reused by all jobs)
    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"sc_{uuid.uuid4().hex[:8]}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)
        log.info(f"[showcase] Saved shared reference: {ref_path}")

    # Materialize source (file or yt-dlp URL)
    src_path: Path
    src_label: str
    if video and video.filename:
        ext = Path(video.filename).suffix or ".mp4"
        src_path = UPLOAD_DIR / f"sc_{uuid.uuid4().hex[:8]}{ext}"
        with open(src_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        src_label = video.filename
    else:
        from pipeline.downloader import download_video
        url = source.strip()
        src_label = url[:60] + ("..." if len(url) > 60 else "")
        try:
            dl_dir = UPLOAD_DIR / f"sc_{uuid.uuid4().hex[:8]}"
            dl_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(download_video(url, str(dl_dir)))
        except Exception as e:
            return JSONResponse({"error": f"Could not download URL: {e}"}, 400)

    # Trim
    trimmed_path = src_path.parent / f"{src_path.stem}_sc{trim_seconds}s.mp4"
    try:
        _trim_video(src_path, trimmed_path, trim_seconds)
    except Exception as e:
        err_msg = ""
        if hasattr(e, "stderr") and getattr(e, "stderr", None):
            err_msg = e.stderr.decode("utf-8", errors="replace")[-300:]
        log.warning(f"[showcase] trim failed: {e} :: {err_msg}")
        return JSONResponse(
            {"error": "Could not trim video", "detail": err_msg or str(e)}, 500)

    # Fan out — one job per language, all sharing the batch_id
    batch_id = f"sc_{uuid.uuid4().hex[:8]}"
    label_final = batch_label or f"Showcase · {trim_seconds}s · {len(langs)} langs"
    job_ids: list = []

    for idx, lang in enumerate(langs):
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = {
            "id": jid,
            "status": "queued",
            "progress": 0,
            "source": str(trimmed_path),
            "source_type": "file",
            "source_label": f"{src_label} -> {lang.upper()} [showcase]",
            "target_lang": lang,
            "model": model,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "voice_mode": ("upload" if ref_path else
                          ("custom" if voice_style.strip() else "preset")),
            "tts_speed": tts_speed,
            "whisper_model": whisper_model,
            "keep_bg": keep_bg,
            "wizard_mode": "auto",
            "auto_denoise": auto_denoise,
            "batch_id": batch_id,
            "batch_label": label_final,
            "batch_kind": "showcase",
            "batch_position": idx,
            "batch_total": len(langs),
            "created": time.time(),
            "scheduled_at": 0,
            "_pending_args": None,
        }
        save_job(jobs[jid])
        await enqueue_job(jid, {
            "source": str(trimmed_path),
            "source_lang": source_lang,
            "target_lang": lang,
            "model": model,
            "keep_bg": keep_bg,
            "whisper_model": whisper_model,
            "reference_audio": ref_path,
            "speaker_mode": speaker_mode,
            "context_hint": context_hint,
            "voice_style": voice_style,
            "voice_preset": voice_preset,
            "tts_speed": tts_speed,
            "wizard_mode": "auto",
            "auto_denoise": auto_denoise,
        })
        job_ids.append(jid)

    log.info(f"[showcase] {batch_id}: enqueued {len(job_ids)} jobs "
             f"({trim_seconds}s, langs={langs})")

    return {
        "ok": True,
        "batch_id": batch_id,
        "batch_kind": "showcase",
        "job_ids": job_ids,
        "count": len(job_ids),
        "trimmed_file": f"/uploads/{trimmed_path.name}",
        "trim_seconds": trim_seconds,
        "target_langs": langs,
    }


@app.get("/api/showcase/{batch_id}")
async def get_showcase(batch_id: str):
    """Status + URL for an assembled showcase. Returns 404 if no showcase
    exists for this batch (either never started or still in progress)."""
    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    mp4 = showcase_dir / "showcase.mp4"
    manifest = showcase_dir / "manifest.json"
    err_file = showcase_dir / "error.txt"

    # Sibling jobs (for progress reporting)
    siblings = [j for j in jobs.values()
                if j.get("batch_id") == batch_id
                and j.get("batch_kind") == "showcase"]

    if mp4.exists():
        man = {}
        if manifest.exists():
            try:
                man = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "ok": True,
            "status": "ready",
            "url": f"/outputs/showcase_{batch_id}/showcase.mp4",
            "manifest": man,
            "sibling_count": len(siblings),
        }

    if err_file.exists():
        try:
            err = err_file.read_text(encoding="utf-8")[-800:]
        except Exception:
            err = "Assembly failed (see server logs)"
        return JSONResponse(
            {"status": "error", "error": err}, 500)

    # Still in progress — count how many sibling jobs are done
    done = sum(1 for j in siblings if j.get("status") == "complete")
    errored = sum(1 for j in siblings if j.get("status") == "error")
    if siblings:
        return {
            "ok": True,
            "status": "assembling" if (done == len(siblings) and batch_id in _showcase_assembling)
                       else ("waiting_for_jobs" if errored == 0 else "jobs_failed"),
            "completed_jobs": done,
            "errored_jobs": errored,
            "total_jobs": len(siblings),
        }

    return JSONResponse({"status": "not_found"}, 404)


@app.post("/api/job/{job_id}/redub")
async def redub_job(
    job_id: str,
    target_langs: str = Form(""),
    mode: str = Form("compare"),          # 'single' | 'compare' | 'showcase'
    model: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form(None),
    voice_preset: Optional[str] = Form(None),
    voice_style: Optional[str] = Form(None),
    tts_speed: Optional[str] = Form(None),
    keep_bg: Optional[bool] = Form(None),
    speaker_mode: Optional[str] = Form(None),
):
    """Re-dub an existing video into new language(s). Reuses the original
    source (file path or URL) — no re-upload required, just specify which
    new languages you want and the mode.

    Modes:
      single   — one new dub in one new language
      compare  — N new dubs (2-6), each in a different language (like Quick Test)
      showcase — N dubs, then stitched into one multilingual reel
    """
    orig = jobs.get(job_id)
    if not orig:
        return JSONResponse({"error": f"Job {job_id} not found"}, 404)

    # Validate mode + langs
    if mode not in ("single", "compare", "showcase"):
        return JSONResponse({"error": f"Invalid mode '{mode}'"}, 400)

    langs = [c.strip().lower() for c in target_langs.split(",") if c.strip()]
    if not langs:
        return JSONResponse({"error": "Specify at least one target_lang"}, 400)
    if mode == "single" and len(langs) != 1:
        return JSONResponse({"error": "mode=single requires exactly 1 language"}, 400)
    if mode in ("compare", "showcase") and not (2 <= len(langs) <= 6):
        return JSONResponse({"error": f"mode={mode} needs 2-6 langs (got {len(langs)})"}, 400)
    unknown = [c for c in langs if c not in _QUICK_TEST_KNOWN_LANGS]
    if unknown:
        return JSONResponse({"error": f"Unknown language codes: {unknown}"}, 400)
    if len(set(langs)) != len(langs):
        return JSONResponse({"error": "Duplicate language codes"}, 400)

    # ── Locate the source (file path or URL) ──────────────────────────
    # Preference order:
    #   1) Original source path if file still exists (uploads/...)
    #   2) source_video.mp4 in the original job's output dir (always copied)
    #   3) Original URL — yt-dlp will re-fetch (cached if possible)
    orig_source = orig.get("source", "")
    source_type = orig.get("source_type", "file")
    src_for_new_jobs: str

    if source_type == "file":
        if orig_source and Path(orig_source).exists():
            src_for_new_jobs = orig_source
        else:
            backup = OUTPUT_DIR / job_id / "source_video.mp4"
            if backup.exists():
                src_for_new_jobs = str(backup)
            else:
                return JSONResponse({
                    "error": "Original source file is gone — can't redub. "
                             "Re-upload it instead.",
                    "original_source": orig_source,
                }, 400)
    else:
        # URL source — pass through. download_video() should hit cache.
        if not orig_source:
            return JSONResponse({"error": "Original job has no source URL"}, 400)
        src_for_new_jobs = orig_source

    # ── Validate / fallback the Ollama model ──────────────────────────
    chosen_model = model or orig.get("model", "aya-expanse:8b")
    _ok, _installed = await check_ollama()
    if _ok and chosen_model not in _installed:
        _preferred = ["aya-expanse:8b", "mistral-nemo:12b", "qwen2.5:14b",
                      "qwen3:8b", "qwen2.5:7b", "gemma3:12b", "gemma3:4b",
                      "llama3.2:3b", "qwen3:14b", "gemma4:e4b", "gemma4:e2b"]
        _fallback = next((m for m in _preferred if m in _installed), None)
        if _fallback:
            log.warning(f"[redub] '{chosen_model}' not installed; using '{_fallback}'")
            chosen_model = _fallback
        else:
            return JSONResponse({
                "error": "No translation model installed",
            }, 400)

    # ── Build settings (inherit from original, accept overrides) ──────
    settings = {
        "model": chosen_model,
        "whisper_model": whisper_model or orig.get("whisper_model", "large-v3"),
        "voice_preset": voice_preset or orig.get("voice_preset", "auto"),
        "voice_style": voice_style if voice_style is not None else orig.get("voice_style", ""),
        "tts_speed": tts_speed or orig.get("tts_speed", "balanced"),
        "keep_bg": keep_bg if keep_bg is not None else bool(orig.get("keep_bg", False)),
        "speaker_mode": speaker_mode or orig.get("speaker_mode", "main"),
        "auto_denoise": bool(orig.get("auto_denoise", False)),
        "context_hint": orig.get("context_hint", ""),
        "source_lang": orig.get("source_lang", "auto"),
    }
    label_base = orig.get("source_label", "") or Path(orig_source).name or job_id

    def _build_job_dict(jid: str, lang: str, extra: dict | None = None) -> dict:
        d = {
            "id": jid,
            "status": "queued",
            "progress": 0,
            "source": src_for_new_jobs,
            "source_type": source_type,
            "source_label": f"{label_base} -> {lang.upper()} [redub]",
            "target_lang": lang,
            "model": settings["model"],
            "speaker_mode": settings["speaker_mode"],
            "context_hint": settings["context_hint"],
            "voice_style": settings["voice_style"],
            "voice_preset": settings["voice_preset"],
            "voice_mode": orig.get("voice_mode", "preset"),
            "tts_speed": settings["tts_speed"],
            "whisper_model": settings["whisper_model"],
            "keep_bg": settings["keep_bg"],
            "wizard_mode": "auto",
            "auto_denoise": settings["auto_denoise"],
            "redubbed_from": job_id,
            "created": time.time(),
            "scheduled_at": 0,
            "_pending_args": None,
        }
        if extra:
            d.update(extra)
        return d

    def _pipeline_args(lang: str) -> dict:
        return {
            "source": src_for_new_jobs,
            "source_lang": settings["source_lang"],
            "target_lang": lang,
            "model": settings["model"],
            "keep_bg": settings["keep_bg"],
            "whisper_model": settings["whisper_model"],
            "reference_audio": "",
            "speaker_mode": settings["speaker_mode"],
            "context_hint": settings["context_hint"],
            "voice_style": settings["voice_style"],
            "voice_preset": settings["voice_preset"],
            "tts_speed": settings["tts_speed"],
            "wizard_mode": "auto",
            "auto_denoise": settings["auto_denoise"],
        }

    # ── Single mode: one job, no batch wrapper ────────────────────────
    if mode == "single":
        jid = uuid.uuid4().hex[:8]
        lang = langs[0]
        jobs[jid] = _build_job_dict(jid, lang)
        save_job(jobs[jid])
        await enqueue_job(jid, _pipeline_args(lang))
        log.info(f"[redub] queued single job {jid} ({lang}) from {job_id}")
        return {"ok": True, "job_id": jid, "redubbed_from": job_id, "target_lang": lang}

    # ── Compare / Showcase: batched fan-out ───────────────────────────
    batch_kind = "showcase" if mode == "showcase" else "quick_test"
    prefix = "sc" if mode == "showcase" else "rd"
    batch_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    batch_label = f"Re-dub · {len(langs)} langs · from {job_id[:6]}"
    job_ids: list = []

    for idx, lang in enumerate(langs):
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = _build_job_dict(jid, lang, extra={
            "batch_id": batch_id,
            "batch_label": batch_label,
            "batch_kind": batch_kind,
            "batch_position": idx,
            "batch_total": len(langs),
        })
        save_job(jobs[jid])
        await enqueue_job(jid, _pipeline_args(lang))
        job_ids.append(jid)

    log.info(f"[redub] {batch_id} ({batch_kind}): enqueued {len(job_ids)} jobs "
             f"from {job_id} ({langs})")
    return {
        "ok": True,
        "batch_id": batch_id,
        "batch_kind": batch_kind,
        "job_ids": job_ids,
        "redubbed_from": job_id,
        "target_langs": langs,
        "mode": mode,
    }


@app.post("/api/showcase/{batch_id}/rebuild")
async def rebuild_showcase(batch_id: str):
    """Manually re-trigger showcase assembly. Useful when the auto-hook
    failed (e.g. ffprobe issue) and the user doesn't want to re-run all
    N dubs from scratch. Deletes any prior showcase.mp4 / error.txt
    first so _maybe_assemble_showcase() will actually rebuild."""
    siblings = [j for j in jobs.values()
                if j.get("batch_id") == batch_id
                and j.get("batch_kind") == "showcase"]
    if not siblings:
        return JSONResponse(
            {"error": f"No showcase batch with id {batch_id}"}, 404)

    incomplete = [j["id"] for j in siblings if j.get("status") != "complete"]
    if incomplete:
        return JSONResponse({
            "error": f"{len(incomplete)} of {len(siblings)} child jobs aren't "
                     f"complete yet — can't assemble",
            "incomplete_job_ids": incomplete,
        }, 400)

    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    for marker in ("showcase.mp4", "error.txt"):
        p = showcase_dir / marker
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                log.warning(f"[showcase] couldn't remove {p}: {e}")
    _showcase_assembling.discard(batch_id)

    # Fire-and-forget — keep a strong reference in _showcase_tasks so the
    # event loop's weak-ref GC doesn't kill the task mid-execution.
    task = asyncio.create_task(_maybe_assemble_showcase(batch_id))
    _showcase_tasks.add(task)
    task.add_done_callback(_showcase_tasks.discard)
    log.info(f"[showcase] {batch_id}: rebuild scheduled (task={task!r})")
    return {"ok": True, "status": "rebuilding", "batch_id": batch_id}


@app.post("/api/showcase/from_batch/{batch_id}")
async def stitch_batch_as_showcase(batch_id: str):
    """Convert any complete batch (quick_test, compare, …) into a showcase
    reel — no re-dubbing needed. Re-marks child jobs as batch_kind='showcase'
    so the normal assembly path picks them up, then triggers assembly.
    Idempotent: safe to call again if you want to re-stitch."""
    all_in_batch = [j for j in jobs.values() if j.get("batch_id") == batch_id]
    if not all_in_batch:
        return JSONResponse({"error": f"Batch {batch_id!r} not found"}, status_code=404)

    incomplete = [j["id"] for j in all_in_batch if j.get("status") != "complete"]
    if incomplete:
        return JSONResponse({
            "error": f"{len(incomplete)} of {len(all_in_batch)} jobs aren't complete yet",
            "incomplete_job_ids": incomplete,
        }, status_code=400)

    # Upgrade batch_kind so _maybe_assemble_showcase finds these siblings
    for j in all_in_batch:
        if j.get("batch_kind") != "showcase":
            j["batch_kind"] = "showcase"
            save_job(j)

    # Clear any stale showcase output so assembly runs fresh
    showcase_dir = OUTPUT_DIR / f"showcase_{batch_id}"
    for marker in ("showcase.mp4", "error.txt"):
        p = showcase_dir / marker
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                log.warning(f"[showcase] couldn't clear {p}: {e}")
    _showcase_assembling.discard(batch_id)

    task = asyncio.create_task(_maybe_assemble_showcase(batch_id))
    _showcase_tasks.add(task)
    task.add_done_callback(_showcase_tasks.discard)
    log.info(f"[showcase] {batch_id}: stitch-from-batch triggered "
             f"({len(all_in_batch)} jobs)")
    return {"ok": True, "status": "assembling", "batch_id": batch_id,
            "jobs": len(all_in_batch)}


# ═══════════════════════════════════════════════════════════════════════
#  Export for Platform — re-encode dubbed video for specific platforms
# ═══════════════════════════════════════════════════════════════════════
# Each preset defines an ffmpeg -vf filter chain, optional fps override,
# and whether to auto-burn translated subtitles into the frame.
#
# Crop/pad strategy:
#   16:9 outputs → letterbox (black bars) to preserve all content
#   9:16 / 1:1   → scale-to-fill + centre-crop (standard for social)
# ═══════════════════════════════════════════════════════════════════════

_EXPORT_PRESETS: dict = {
    "youtube_1080p": {
        "vf": "scale=1920:1080:force_original_aspect_ratio=decrease,"
              "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "fps": None, "burn_subs": False,
    },
    "youtube_4k": {
        "vf": "scale=3840:2160:force_original_aspect_ratio=decrease,"
              "pad=3840:2160:(ow-iw)/2:(oh-ih)/2:black",
        "fps": None, "burn_subs": False,
    },
    "tiktok": {
        "vf": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "fps": 30, "burn_subs": True,
    },
    "shorts": {
        "vf": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "fps": 30, "burn_subs": True,
    },
    "reels": {
        "vf": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        "fps": 30, "burn_subs": True,
    },
    "instagram_square": {
        "vf": "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080",
        "fps": None, "burn_subs": False,
    },
    "twitter": {
        "vf": "scale=1280:720:force_original_aspect_ratio=decrease,"
              "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
        "fps": None, "burn_subs": False,
    },
}


@app.post("/api/dub/{job_id}/export")
async def export_for_platform(
    job_id: str,
    preset: str = Form("youtube_1080p"),
    style: str = Form("default"),  # subtitle style — used when preset auto-burns subs
):
    """Re-encode the dubbed video for a specific platform preset.

    Returns JSON: {"ok": True, "url": "..."}  on success,
                  {"error": "...", "detail": "..."}  on failure.
    """
    import subprocess
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if preset not in _EXPORT_PRESETS:
        return JSONResponse(
            {"error": f"Unknown preset '{preset}'. Options: {list(_EXPORT_PRESETS)}"}, 400)

    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    if not src_video.exists():
        return JSONResponse({"error": "Dubbed video not yet generated"}, 400)

    pc = _EXPORT_PRESETS[preset]
    dst_video = work / f"export_{preset}.mp4"
    vf = pc["vf"]

    # For presets that burn subs, append the subtitle filter to the chain
    if pc["burn_subs"]:
        srt_file = work / "translated.srt"
        if not srt_file.exists():
            cp_path = work / "checkpoint_tts_done.json"
            if not cp_path.exists():
                cp_path = work / "checkpoint_translation_done.json"
            if cp_path.exists():
                try:
                    cp = json.loads(cp_path.read_text(encoding="utf-8"))
                    _write_srt_file(cp.get("segments", []), srt_file)
                except Exception as e:
                    return JSONResponse({"error": f"Could not generate SRT: {e}"}, 500)
        if srt_file.exists():
            force_style = SUB_STYLE_MAP.get(style, SUB_STYLE_MAP["default"])
            srt_arg = str(srt_file).replace("\\", "/").replace(":", r"\:")
            vf = f"{vf},subtitles='{srt_arg}':force_style='{force_style}'"

    cmd = [
        "ffmpeg", "-y", "-i", str(src_video),
        "-vf", vf,
        "-c:a", "copy",   # audio pass-through — no re-encode
        "-preset", "fast",
    ]
    if pc["fps"]:
        cmd += ["-r", str(pc["fps"])]
    cmd.append(str(dst_video))

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        return JSONResponse({
            "ok": True,
            "url": f"/outputs/{job_id}/export_{preset}.mp4?v={int(time.time())}",
            "preset": preset,
        })
    except subprocess.CalledProcessError as e:
        err_msg = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
        log.warning(f"[export] ffmpeg failed for {job_id}/{preset}: {err_msg}")
        return JSONResponse({
            "error": f"Export failed for preset '{preset}'",
            "detail": err_msg[:300],
        }, 500)


# ─────────────────────────────────────────────────────────────
# Stage helpers — reusable from run_pipeline AND from resume endpoints
# ─────────────────────────────────────────────────────────────

async def _run_translate_stage(
    job: dict, work: Path, segments: list, effective_src: str,
    target_lang: str, model: str, context_hint: str,
) -> list:
    """Run translation on raw segments + return segments with translated_text."""
    def update(**kwargs):
        if job.get("cancel_requested"):
            raise JobCancelled(f"Job {job['id']} cancelled by user")
        job.update(kwargs); save_job(job)
    update(status="translating", progress=45,
           step_detail=f"Translating to {target_lang}...")
    translated = await translate_segments(
        segments, effective_src, target_lang, model,
        context_hint=context_hint,
    )
    # See comment on unload in main pipeline — free VRAM for VoxCPM
    try:
        await unload_ollama_model(model)
    except Exception as e:
        log.warning(f"Failed to unload Ollama model (non-fatal): {e}")
    return translated


async def _run_tts_and_merge_stage(
    job: dict, work: Path, state: dict,
    voice_style: str, voice_preset: str, tts_speed: str,
    ref_path_override: str = "",
    audio_output_name: str = "dubbed_audio.wav",
    tts_subdir: str = "tts_segments",
    preserve_existing_audio_paths: bool = False,
) -> dict:
    """Run TTS + assemble + merge. Returns dict with 'segments' and status.
    - preserve_existing_audio_paths: if True, segments that already have
      a valid audio_path keep their existing file (for per-segment regen).
    """
    job_id = job["id"]
    def update(**kwargs):
        if job.get("cancel_requested"):
            _maybe_terminate_tts_worker()
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs); save_job(job)

    eff_style, voice_seed, preset_ref = resolve_voice_config(
        voice_preset, voice_style, job_id
    )
    ref_path = ref_path_override
    if preset_ref and os.path.exists(preset_ref):
        ref_path = preset_ref
        log.info(f"[stage] Using file-preset reference: {preset_ref}")

    # Force a FRESH voice_seed whenever this is NOT the first-time run:
    #   - retry_tts (audio_output_name is "dubbed_audio_retry.wav")
    #   - per-segment regen (preserve_existing_audio_paths=True)
    # Without re-seeding, identical inputs to VoxCPM produce byte-identical
    # outputs — so "click retry" would silently do nothing visible to user.
    is_rerun = (audio_output_name != "dubbed_audio.wav"
                or preserve_existing_audio_paths)
    if is_rerun:
        voice_seed = int(time.time() * 1000) % 2_147_483_647
        log.info(f"[stage] Re-run: rolled fresh voice_seed={voice_seed}")

    # ─── VOICE MODE ROUTING ──────────────────────────────────────────
    # VoxCPM has 3 mutually-exclusive modes — the user's UI choice maps
    # to ONE of them. Previously, mixing refs + style prefix was producing
    # garbage: VoxCPM would try to clone the video voice AND literally read
    # out the "(deep male voice, narrator)" style description as text.
    #
    #   Mode 1: File/uploaded reference → Controllable Cloning
    #           Clean reference_wav_path only, NO style prefix.
    #
    #   Mode 2: Style preset (e.g. "male_deep") → Voice Design
    #           "(style description)<text>" — NO reference at all.
    #
    #   Mode 3: No change (auto preset, no upload) → keep state refs
    #           Controllable/Ultimate Cloning from video refs, no prefix.
    mode = "source_refs"
    if ref_path and os.path.exists(ref_path):
        mode = "file_ref"
    elif eff_style and eff_style.strip():
        mode = "voice_design"

    update(
        status="synthesizing", progress=65,
        voice_preset=voice_preset, voice_style=voice_style,
        voice_style_effective=eff_style, voice_seed=voice_seed,
        tts_speed=tts_speed,
        voice_mode=("upload" if mode == "file_ref" else
                    ("custom" if mode == "voice_design" else "source")),
        step_detail="Generating speech...",
    )
    log.info(f"[stage] Voice mode: {mode} "
             f"(ref={bool(ref_path)}, style={bool(eff_style)})")

    # Start from state's refs, then override per mode
    speaker_refs = dict(state.get("speaker_refs", {}))
    speaker_transcripts = dict(state.get("speaker_transcripts", {}))

    if mode == "file_ref":
        log.info(f"[stage] Using file/upload ref for all speakers: {ref_path}")
        target_keys = list(speaker_refs.keys()) or ["SPEAKER_00"]
        for sp in target_keys:
            speaker_refs[sp] = ref_path
            speaker_transcripts[sp] = ""  # Controllable Cloning only
    elif mode == "voice_design":
        # CRITICAL: Voice Design needs NO reference. Without this clear,
        # VoxCPM sees ref + style prefix and produces broken output.
        log.info(f"[stage] Clearing speaker refs for Voice Design mode")
        speaker_refs = {}
        speaker_transcripts = {}
    else:
        # source_refs mode: use refs extracted from the source video.
        # CRITICAL: speaker_refs in the checkpoint may have been overwritten
        # by an earlier preset/upload (if the user previously dubbed with
        # zhirik.wav, speaker_refs contains zhirik paths). The real source
        # refs were stashed separately as source_speaker_refs — prefer those
        # when available so "retry without changing anything" truly falls
        # back to the original video's voice, not the last-used preset.
        source_refs_stash = state.get("source_speaker_refs") or {}
        if source_refs_stash and is_rerun:
            log.info(f"[stage] Retry: restoring ORIGINAL source refs "
                     f"(not previous preset): {list(source_refs_stash.keys())}")
            speaker_refs = dict(source_refs_stash)
            # Clear transcripts — cross-lingual/controllable cloning only
            for sp in speaker_refs:
                speaker_transcripts[sp] = ""
        else:
            log.info(f"[stage] Using original source speaker refs: "
                     f"{list(speaker_refs.keys())}")

    segments = [dict(s) for s in state["segments"]]
    tts = get_tts_engine()

    # Voice Design style prefix — ONLY when mode is voice_design.
    # If refs are used, the style description would literally be spoken.
    if mode == "voice_design" and isinstance(tts, VoxCPMSynthesizer):
        style = eff_style.strip().strip("()")
        for s in segments:
            base = s.get("translated_text") or s.get("text", "")
            if base and not base.startswith("("):
                s["translated_text"] = f"({style}){base}"

    # Preserve-mode: skip TTS for segments that already have valid audio
    if preserve_existing_audio_paths:
        todo, keep = [], []
        for s in segments:
            ap = s.get("audio_path", "")
            if ap and os.path.exists(ap):
                keep.append(s)
            else:
                todo.append(s)
        log.info(f"[stage] Preserving {len(keep)} existing segments, "
                 f"synthesizing {len(todo)} new")
        synth_input = todo
    else:
        synth_input = segments

    tts_dir = str(work / tts_subdir)
    total = len(synth_input)
    def synth_progress(done, total_inner):
        pct = 65 + int((done / max(total_inner, 1)) * 20)
        update(progress=min(pct, 85),
               step_detail=f"Synthesizing: {done}/{total_inner}")

    if total > 0:
        if isinstance(tts, VoxCPMSynthesizer):
            # Determine cross-lingual from state (may be missing from older
            # checkpoints — in that case assume cross-lingual as a safer default
            # since that's the common dubbing use-case)
            src_lang = state.get("effective_src") or state.get("source_lang", "en")
            tgt_lang = state.get("target_lang", "ru")
            synth_input = tts.synthesize_segments(
                synth_input, tts_dir,
                speaker_refs=speaker_refs,
                speaker_transcripts=speaker_transcripts,
                progress_callback=synth_progress,
                voice_seed=voice_seed,
                tts_speed=tts_speed,
                is_cross_lingual=(src_lang != tgt_lang),
                target_lang=tgt_lang,
            )
        else:
            synth_input = await tts.synthesize_segments_async(
                synth_input, tts_dir, state.get("target_lang", "ru"),
                progress_callback=synth_progress,
            )

    # Re-merge synth_input back into segments list if preserve-mode
    if preserve_existing_audio_paths:
        by_idx = {s.get("idx"): s for s in segments}
        for s in synth_input:
            if s.get("idx") in by_idx:
                by_idx[s["idx"]].update(s)
        segments = list(by_idx.values())

    synth_ok = sum(1 for s in segments if s.get("audio_path"))
    if synth_ok == 0:
        raise RuntimeError("All TTS synthesis failed - check model/GPU")
    update(progress=85, step_detail=f"Synthesized {synth_ok}/{len(segments)}")

    update(status="assembling", progress=88, step_detail="Assembling dubbed audio...")
    dubbed_wav = str(work / audio_output_name)
    assemble_dubbed_audio(segments, state["duration"], dubbed_wav,
                          tts.sample_rate, apply_loudnorm=True)
    _save_placements(work, segments)

    update(status="merging", progress=93, step_detail="Rendering final video...")
    output_mp4 = str(work / "dubbed_video.mp4")
    merge_audio_video(
        state["video_path"], dubbed_wav, output_mp4,
        state.get("bg_audio_path", "") if state.get("keep_bg") else "",
    )

    # Update the tts_done checkpoint so next regen starts from current audio.
    # Existing qa_score/tts_tier on regenerated segments is preserved from the
    # worker (s.get("qa_score") is set by synthesizer); for non-regenerated
    # segments, read from the previous checkpoint to keep QA badges intact.
    prev = _load_checkpoint(job_id, stage="tts_done") or {}
    prev_by_idx = {seg["idx"]: seg for seg in prev.get("segments", [])}

    def _meta(s, i):
        # Prefer fresh worker values on regenerated segments; else pull from prev
        pidx = s.get("idx", i)
        fallback = prev_by_idx.get(pidx, {})
        qa = s.get("qa_score")
        if qa is None: qa = fallback.get("qa_score")
        tier = s.get("tts_tier")
        if tier is None: tier = fallback.get("tts_tier")
        return qa, tier

    _save_checkpoint(job_id, work, stage="tts_done", data={
        **state,
        "segments": [
            (lambda qa_tier: {
                "idx": s.get("idx", i),
                "start": s["start"], "end": s["end"],
                "text": s["text"],
                "translated_text": s.get("translated_text", ""),
                "speaker": s.get("speaker", "SPEAKER_00"),
                "audio_path": s.get("audio_path", ""),
                "qa_score": qa_tier[0],
                "tts_tier": qa_tier[1],
            })(_meta(s, i))
            for i, s in enumerate(segments)
        ],
    })

    update(
        status="complete", progress=100,
        step_detail="Done!",
        output_url=f"/outputs/{job_id}/dubbed_video.mp4?v={int(time.time())}",
        completed_at=time.time(),
    )
    log.info(f"[stage] Pipeline complete: {output_mp4}")
    return {"segments": segments, "output_url": f"/outputs/{job_id}/dubbed_video.mp4"}


async def retry_tts_pipeline(job_id: str, voice_style: str, voice_preset: str,
                              tts_speed: str, ref_path: str):
    """Re-runs ONLY TTS + assemble + merge using the previously-saved
    translation/transcription. Much faster than re-running the full pipeline."""
    job = jobs.get(job_id)
    if not job:
        return
    work = OUTPUT_DIR / job_id
    state = _load_checkpoint(job_id, "translation_done") or \
            _load_checkpoint(job_id, "tts_done")
    if not state:
        job["status"] = "error"
        job["error"] = "No saved state to retry - run full pipeline once first"
        save_job(job)
        return
    try:
        await _run_tts_and_merge_stage(
            job, work, state,
            voice_style=voice_style, voice_preset=voice_preset, tts_speed=tts_speed,
            ref_path_override=ref_path,
            audio_output_name="dubbed_audio_retry.wav",
            tts_subdir="tts_segments_retry",
        )
    except Exception as e:
        log.error(f"[retry] Failed: {e}", exc_info=True)
        job.update(status="error", error=str(e)); save_job(job)





@app.post("/api/dub/{job_id}/retry_tts")
async def retry_tts(
    job_id: str,
    voice_style: str = Form(""),
    voice_preset: str = Form("auto"),
    tts_speed: str = Form("balanced"),
    reference: Optional[UploadFile] = File(None),
):
    """Re-runs only TTS + merge stages using saved state from a completed job.
    Much faster than re-running the full pipeline (no download, transcribe,
    translate steps)."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_retry_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    asyncio.create_task(retry_tts_pipeline(
        job_id, voice_style, voice_preset, tts_speed, ref_path,
    ))
    return {"ok": True, "job_id": job_id}


# ─────────────────────────────────────────────────────────────
# API: Wizard / Checkpoint Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/dub/{job_id}/checkpoint/{stage}")
async def get_checkpoint(job_id: str, stage: str):
    """Return the contents of a saved checkpoint for the UI to display.
    Useful for editable transcript/translation review screens."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, stage)
    if not cp:
        return JSONResponse({"error": f"Checkpoint '{stage}' not found"}, 404)

    # Expose speaker_refs as {speaker_id: basename} — the UI shows the
    # filename so the user knows which ref is active and can replace it.
    # We don't send absolute paths (server-internal).
    refs_summary = {}
    for spk, path in (cp.get("speaker_refs") or {}).items():
        if path and os.path.exists(path):
            try:
                import soundfile as _sf
                info = _sf.info(path)
                refs_summary[spk] = {
                    "filename": os.path.basename(path),
                    "duration": round(info.frames / info.samplerate, 1),
                    "is_user_upload": os.path.basename(path).startswith("user_"),
                }
            except Exception:
                refs_summary[spk] = {
                    "filename": os.path.basename(path),
                    "duration": None, "is_user_upload": False,
                }

    return {
        "stage": cp.get("stage"),
        "saved_at": cp.get("saved_at"),
        "target_lang": cp.get("target_lang"),
        "duration": cp.get("duration"),
        "segments": cp.get("segments", []),
        "speaker_refs": refs_summary,
    }


@app.post("/api/dub/{job_id}/edit_translations")
async def edit_translations(job_id: str, edits: str = Form(...)):
    """Update the translated_text for one or more segments in the saved
    checkpoint. `edits` is a JSON string: {"<idx>": "new translation", ...}
    After editing, the user should call /continue to proceed to TTS."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, "translation_done")
    if not cp:
        return JSONResponse({"error": "No translation checkpoint to edit"}, 404)
    try:
        edit_map = json.loads(edits)
        if not isinstance(edit_map, dict):
            raise ValueError("edits must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid edits JSON: {e}"}, 400)

    # Apply edits by segment index (string keys from the JSON)
    n_edited = 0
    for s in cp.get("segments", []):
        key = str(s.get("idx"))
        if key in edit_map:
            new_text = str(edit_map[key]).strip()
            if new_text and new_text != s.get("translated_text"):
                s["translated_text"] = new_text
                n_edited += 1
    # Re-save the translation_done checkpoint with the edits
    work = OUTPUT_DIR / job_id
    _save_checkpoint(job_id, work, stage="translation_done", data=cp)

    # Re-export SRT with the user's edits so .srt download always matches
    # what gets spoken. Also update tts_done if it exists (per-segment
    # regen panel uses it for translated_text display).
    if n_edited > 0:
        try:
            srt_path = str(work / "subtitles.srt")
            write_srt(cp.get("segments", []), srt_path)
        except Exception as e:
            log.warning(f"[edit] SRT re-export failed: {e}")
        tcp = _load_checkpoint(job_id, "tts_done")
        if tcp:
            tcp_segs = tcp.get("segments", [])
            tcp_by_idx = {s.get("idx"): s for s in tcp_segs}
            for seg in cp.get("segments", []):
                t = tcp_by_idx.get(seg.get("idx"))
                if t is not None:
                    t["translated_text"] = seg.get("translated_text", "")
            _save_checkpoint(job_id, work, stage="tts_done", data=tcp)

    log.info(f"[edit] Applied {n_edited} translation edits for job {job_id}")
    return {"ok": True, "edited": n_edited}


@app.post("/api/dub/{job_id}/edit_speaker_ref/{speaker_id}")
async def edit_speaker_ref(
    job_id: str, speaker_id: str,
    reference: UploadFile = File(...),
):
    """Replace the voice-cloning reference for one speaker.
    Used in wizard mode when diarization found a second speaker but only
    got 3-5 seconds of their audio (too short for clean cloning) — the
    user can upload a longer, cleaner clip of them from elsewhere."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    if not reference or not reference.filename:
        return JSONResponse({"error": "No file uploaded"}, 400)

    # Pick the earliest checkpoint that already has speaker_refs set; edit it
    # + later checkpoints so the new ref is picked up on /continue or regen.
    work = OUTPUT_DIR / job_id
    ref_dir = work / "speaker_refs"
    ref_dir.mkdir(exist_ok=True)
    ext = Path(reference.filename).suffix.lower() or ".wav"
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        return JSONResponse({"error": f"Unsupported format: {ext}"}, 400)

    # Always store the user's upload as a fresh file so we don't clobber
    # the diarizer-extracted one (allows user to revert later if needed).
    user_ref = str(ref_dir / f"user_{speaker_id}{ext}")
    with open(user_ref, "wb") as f:
        shutil.copyfileobj(reference.file, f)

    n_updated = 0
    for stage in ("transcription_done", "translation_done", "tts_done"):
        cp = _load_checkpoint(job_id, stage)
        if cp and "speaker_refs" in cp:
            cp["speaker_refs"][speaker_id] = user_ref
            _save_checkpoint(job_id, work, stage=stage, data=cp)
            n_updated += 1

    if n_updated == 0:
        return JSONResponse(
            {"error": "No checkpoint has speaker_refs yet"}, 400
        )
    log.info(f"[edit_spk] Replaced ref for {speaker_id} on job {job_id} "
             f"({n_updated} checkpoints updated)")
    return {"ok": True, "speaker_id": speaker_id,
            "new_ref_path": user_ref, "checkpoints_updated": n_updated}


# ═══════════════════════════════════════════════════════════════════════
#  Lip-sync (Wav2Lip) — placeholder endpoint
# ═══════════════════════════════════════════════════════════════════════
# Lip-sync is an optional post-processing step that recomputes mouth
# movements in the video to match the dubbed audio. This makes the
# output look much less like an obvious dub job.
#
# This endpoint currently returns an install-instructions error. When
# Wav2Lip is installed, replace the raise with actual model invocation.
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/dub/{job_id}/transcripts.txt")
async def download_transcripts_txt(job_id: str):
    """Export side-by-side transcript as plain text. Useful for content
    creators who want to copy/paste into captions, descriptions, etc.

    Format:
        === SEGMENT 1 (0.0s → 5.6s · SPEAKER_00) ===
        EN: Original source text
        RU: Translated text (with any user edits applied)
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)

    # Prefer tts_done (has user edits from per-segment regens), fall back
    # to translation_done. If neither exists, return 404.
    cp = _load_checkpoint(job_id, "tts_done") or _load_checkpoint(job_id, "translation_done")
    if not cp:
        return JSONResponse({"error": "No completed transcript yet"}, 404)

    segs = cp.get("segments", [])
    if not segs:
        return JSONResponse({"error": "No segments"}, 404)

    target = cp.get("target_lang", "ru").upper()
    lines = [
        f"TachiDUBB Studio transcript export · job {job_id}",
        f"Target language: {target} · {len(segs)} segments",
        "=" * 60, "",
    ]
    for s in segs:
        idx = s.get("idx", 0) + 1
        start = s.get("start", 0.0)
        end = s.get("end", 0.0)
        spk = s.get("speaker", "SPEAKER_00")
        lines.append(f"=== #{idx} · {start:.1f}s → {end:.1f}s · {spk} ===")
        lines.append(f"EN: {s.get('text', '').strip()}")
        lines.append(f"{target}: {s.get('translated_text', '').strip()}")
        lines.append("")

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content="\n".join(lines),
        headers={
            "Content-Disposition": f'attachment; filename="transcripts_{job_id}.txt"'
        },
    )


# ═══════════════════════════════════════════════════════════════════════
#  Lip-sync (Wav2Lip) — optional post-processing
# ═══════════════════════════════════════════════════════════════════════
# Wav2Lip (Rudrabha 2020) warps mouth regions in the source video to
# match the dubbed audio. Makes dubs look less obviously mismatched —
# especially helpful for talking-head footage.
#
# Not shipped as a pip install because:
#   1. ~400MB checkpoint needs to be downloaded manually (the repo's
#      Google Drive link requires a browser)
#   2. Its dependencies conflict with newer torch; safest to run in a
#      subprocess so it can have its own venv if needed
#   3. Only useful when faces are visible — probably <30% of use cases
#
# Detection: we look for the checkpoint file existing at one of several
# common paths. If absent, the endpoint returns a friendly install guide.
# If present, we invoke Wav2Lip via subprocess (its inference.py is the
# standard entry point).
# ═══════════════════════════════════════════════════════════════════════

def _find_wav2lip_setup():
    """Scan common paths for Wav2Lip repo + checkpoint. Returns dict with
    'repo_dir', 'checkpoint', 'python' — all strings, or None if missing.
    Checks paths relative to the TachiDUBB Studio install, and a few user-home
    fallbacks since people often clone repos in various places."""
    # Candidate directories where the repo might be cloned
    candidate_dirs = [
        BASE / "Wav2Lip",
        BASE / "external" / "Wav2Lip",
        BASE.parent / "Wav2Lip",
        Path.home() / "Wav2Lip",
    ]
    # Also check if TACHIDUBB_WAV2LIP_DIR env var was set
    env_dir = os.getenv("TACHIDUBB_WAV2LIP_DIR", "")
    if env_dir:
        candidate_dirs.insert(0, Path(env_dir))

    for d in candidate_dirs:
        if not d.exists():
            continue
        inference = d / "inference.py"
        if not inference.exists():
            continue
        # Find the checkpoint — Wav2Lip ships two models (wav2lip.pth and
        # wav2lip_gan.pth). Prefer GAN — sharper mouth detail.
        ckpt_candidates = [
            d / "checkpoints" / "wav2lip_gan.pth",
            d / "checkpoints" / "wav2lip.pth",
            d / "checkpoints" / "Wav2Lip.pth",
        ]
        ckpt = next((c for c in ckpt_candidates if c.exists()), None)
        if ckpt is None:
            continue
        return {
            "repo_dir": str(d),
            "checkpoint": str(ckpt),
            "python": sys.executable,  # reuse current venv — adjust via TACHIDUBB_WAV2LIP_PYTHON
            "checkpoint_name": ckpt.name,
        }
    return None


def _wav2lip_install_guide() -> dict:
    """Install guide returned when Wav2Lip isn't found. Structured so UI
    can render a copy-paste block + link to checkpoint download."""
    return {
        "error": "wav2lip_not_installed",
        "message": "Lip-sync requires Wav2Lip. It's optional — install it only if you "
                   "want mouth movements to match the dubbed audio. Works best on "
                   "talking-head footage; useless for action / wide shots.",
        "install_steps": [
            {"label": "1. Clone the Wav2Lip repo",
             "cmd": f"cd {BASE} && git clone https://github.com/Rudrabha/Wav2Lip.git"},
            {"label": "2. Install Wav2Lip's deps into your TachiDUBB Studio venv",
             "cmd": "pip install librosa==0.7.0 numba==0.48 opencv-contrib-python "
                    "face-detection tqdm"},
            {"label": "3. Download the GAN checkpoint (~400 MB)",
             "cmd": "Open https://github.com/Rudrabha/Wav2Lip in browser, "
                    "follow README link to wav2lip_gan.pth, "
                    f"save it at {BASE}\\Wav2Lip\\checkpoints\\wav2lip_gan.pth"},
            {"label": "4. Restart TachiDUBB Studio server. The button will light up automatically."},
        ],
        "note": "Wav2Lip is picky about input — only helps when faces are clear "
                "and roughly front-facing. It fails on fast cuts, extreme angles, "
                "and low-res video. Expect 2-5x the video duration to process.",
        "env_override": "If Wav2Lip is already cloned elsewhere, set "
                        "TACHIDUBB_WAV2LIP_DIR=<path> in .env",
    }


@app.get("/api/lip_sync/status")
async def lip_sync_status():
    """Quick detection endpoint — UI calls this to decide whether to show
    the lip-sync button lit (ready) or greyed (needs install)."""
    setup = _find_wav2lip_setup()
    if not setup:
        return {"installed": False, "guide": _wav2lip_install_guide()}
    return {
        "installed": True,
        "checkpoint": setup["checkpoint_name"],
        "repo_dir": setup["repo_dir"],
    }


def _run_wav2lip_sync(job_id: str) -> dict:
    """Synchronously apply Wav2Lip to a job's dubbed_video.mp4. Used by
    both the manual `POST /api/dub/{id}/lip_sync` endpoint AND the
    auto-lip-sync hook in the queue worker (when the job was submitted
    with `lip_sync=True`).

    Returns a dict with `ok`, `url`, `elapsed_sec` on success — or
    `error` + `message` (+ optional `stderr_tail`) on failure. Updates
    `jobs[job_id]['lipsync_status']` ('running' → 'done' | 'error') and
    `lipsync_url` so the UI can poll for completion.
    """
    import subprocess as _sp

    if job_id not in jobs:
        return {"error": "job_not_found", "message": f"Job {job_id} not found"}

    setup = _find_wav2lip_setup()
    if not setup:
        return {"error": "wav2lip_not_installed", "guide": _wav2lip_install_guide()}

    work = OUTPUT_DIR / job_id
    src_video = work / "dubbed_video.mp4"
    if not src_video.exists():
        return {"error": "no_dub", "message": "Dubbed video not generated yet"}

    dub_wav = work / "_lipsync_dub.wav"
    try:
        _sp.run(
            ["ffmpeg", "-y", "-i", str(src_video),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             str(dub_wav)],
            check=True, capture_output=True, timeout=120,
        )
    except _sp.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-400:]
        return {"error": "audio_extract_failed", "message": err}

    w2l_out = work / "_lipsync_raw.mp4"
    env = os.environ.copy()
    env["PYTHONPATH"] = setup["repo_dir"] + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        setup["python"], "inference.py",
        "--checkpoint_path", setup["checkpoint"],
        "--face", str(src_video),
        "--audio", str(dub_wav),
        "--outfile", str(w2l_out),
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "1",
        "--nosmooth",
    ]

    log.info(f"[lipsync] Running Wav2Lip for job {job_id}")
    jobs[job_id]["lipsync_status"] = "running"
    save_job(jobs[job_id])
    elapsed = 0.0
    try:
        t0 = time.time()
        result = _sp.run(
            cmd, cwd=setup["repo_dir"], env=env,
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            err_tail = (result.stderr or "")[-600:]
            log.warning(f"[lipsync] Wav2Lip failed: {err_tail}")
            jobs[job_id]["lipsync_status"] = "error"
            jobs[job_id]["lipsync_error"] = err_tail
            save_job(jobs[job_id])
            return {
                "error": "wav2lip_runtime_error",
                "message": "Wav2Lip ran but failed. Likely no face detected, "
                           "GPU OOM, or missing deps.",
                "stderr_tail": err_tail,
            }
        elapsed = time.time() - t0
        log.info(f"[lipsync] Wav2Lip done in {elapsed:.0f}s")
    except _sp.TimeoutExpired:
        jobs[job_id]["lipsync_status"] = "error"
        save_job(jobs[job_id])
        return {"error": "wav2lip_timeout", "message": "Didn't finish in 30 minutes"}

    if not w2l_out.exists():
        jobs[job_id]["lipsync_status"] = "error"
        save_job(jobs[job_id])
        return {"error": "wav2lip_no_output",
                "message": "Wav2Lip reported success but produced no output"}

    # Re-mux with the original full-quality dub audio (Wav2Lip's output
    # has 16kHz mono audio which sounds terrible).
    final_out = work / "dubbed_video_lipsync.mp4"
    try:
        _sp.run(
            ["ffmpeg", "-y",
             "-i", str(w2l_out),
             "-i", str(src_video),
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             str(final_out)],
            check=True, capture_output=True, timeout=300,
        )
    except _sp.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-400:]
        jobs[job_id]["lipsync_status"] = "error"
        save_job(jobs[job_id])
        return {"error": "final_mux_failed", "message": err}
    finally:
        for p in (dub_wav, w2l_out):
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass

    url = f"/outputs/{job_id}/dubbed_video_lipsync.mp4?v={int(time.time())}"
    jobs[job_id]["lipsync_status"] = "done"
    jobs[job_id]["lipsync_url"] = url
    jobs[job_id].pop("lipsync_error", None)
    save_job(jobs[job_id])
    return {"ok": True, "url": url, "elapsed_sec": round(elapsed, 1)}


@app.post("/api/dub/{job_id}/lip_sync")
async def lip_sync(job_id: str):
    """Apply Wav2Lip to the dubbed video (manual on-demand endpoint).

    Returns the new video URL when done. Same code path is also used
    automatically by the pipeline when a job was submitted with
    `lip_sync=true` on the original form.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    setup = _find_wav2lip_setup()
    if not setup:
        return JSONResponse(_wav2lip_install_guide(), 501)
    # Run in executor — wav2lip is heavy synchronous CPU/GPU work
    result = await asyncio.get_event_loop().run_in_executor(
        None, _run_wav2lip_sync, job_id)
    if "error" in result:
        return JSONResponse(result, 500)
    return result


@app.post("/api/dub/{job_id}/edit_transcript")
async def edit_transcript(job_id: str, edits: str = Form(...)):
    """Update the source `text` for one or more segments in the transcription
    checkpoint. Used in wizard review_transcript mode when the user spots
    ASR errors before they get baked into the translation.
    `edits` is a JSON string: {"<idx>": "corrected text", ...}"""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    cp = _load_checkpoint(job_id, "transcription_done")
    if not cp:
        return JSONResponse({"error": "No transcription checkpoint to edit"}, 404)
    try:
        edit_map = json.loads(edits)
        if not isinstance(edit_map, dict):
            raise ValueError("edits must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid edits JSON: {e}"}, 400)

    n_edited = 0
    for s in cp.get("segments", []):
        key = str(s.get("idx"))
        if key in edit_map:
            new_text = str(edit_map[key]).strip()
            if new_text and new_text != s.get("text"):
                s["text"] = new_text
                n_edited += 1
    work = OUTPUT_DIR / job_id
    _save_checkpoint(job_id, work, stage="transcription_done", data=cp)
    log.info(f"[edit] Applied {n_edited} transcript edits for job {job_id}")
    return {"ok": True, "edited": n_edited}


@app.post("/api/dub/{job_id}/continue")
async def continue_pipeline(
    job_id: str,
    voice_style: str = Form(""),
    voice_preset: str = Form(""),
    tts_speed: str = Form(""),
    reference: Optional[UploadFile] = File(None),
):
    """Continue the pipeline from the most recent checkpoint. Called after
    the user has reviewed (and possibly edited) the transcript/translation
    in wizard mode.

    - If stopped at translation_done: runs TTS + merge.
    - If stopped at transcription_done: runs translate + TTS + merge.

    Voice settings, if provided, override what was originally requested."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_continue_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    # Merge new voice settings with job defaults
    final_style = voice_style if voice_style else job.get("voice_style", "")
    final_preset = voice_preset if voice_preset else job.get("voice_preset", "auto")
    final_speed = tts_speed if tts_speed else job.get("tts_speed", "balanced")

    cp = _latest_checkpoint(job_id)
    if not cp:
        return JSONResponse({"error": "No checkpoint to continue from"}, 404)

    # Reset error/stale flags so the History UI immediately reflects that
    # the job is alive again. _continue_from_checkpoint will set status
    # to "translating"/"synthesizing" as it starts each stage.
    job["status"] = "resuming"
    job.pop("error", None)
    job.pop("stale_from_restart", None)
    save_job(job)

    asyncio.create_task(_continue_from_checkpoint(
        job_id, cp, final_style, final_preset, final_speed, ref_path,
    ))
    return {"ok": True, "job_id": job_id, "resuming_from": cp.get("stage")}


async def _continue_from_checkpoint(
    job_id: str, cp: dict,
    voice_style: str, voice_preset: str, tts_speed: str, ref_path: str,
):
    """Dispatch to the right stage(s) depending on which checkpoint we have."""
    job = jobs.get(job_id)
    if not job:
        return
    work = OUTPUT_DIR / job_id
    def update(**kwargs):
        if job.get("cancel_requested"):
            _maybe_terminate_tts_worker()
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs); save_job(job)

    stage = cp.get("stage", "")
    log.info(f"[continue] Resuming job {job_id} from stage '{stage}'")
    try:
        if stage == "transcription_done":
            # Need to translate first, then TTS
            update(status="translating", progress=45,
                   step_detail="Translating approved transcript...")
            effective_src = cp.get("effective_src", "en")
            target_lang = cp.get("target_lang", "ru")
            model = cp.get("model", "gemma4:e4b")
            context_hint = cp.get("context_hint", "")
            segments = await translate_segments(
                cp["segments"], effective_src, target_lang, model,
                context_hint=context_hint,
            )
            # Save translation_done checkpoint
            _save_checkpoint(job_id, work, stage="translation_done", data={
                **cp,
                "segments": [
                    {
                        "idx": i, "start": s["start"], "end": s["end"],
                        "text": s["text"],
                        "translated_text": s.get("translated_text", ""),
                        "speaker": s.get("speaker", "SPEAKER_00"),
                    }
                    for i, s in enumerate(segments)
                ],
            })
            cp = _load_checkpoint(job_id, "translation_done")

        # Now run TTS+merge from translation_done checkpoint
        await _run_tts_and_merge_stage(
            job, work, cp,
            voice_style=voice_style, voice_preset=voice_preset,
            tts_speed=tts_speed, ref_path_override=ref_path,
        )
    except Exception as e:
        log.error(f"[continue] Failed: {e}", exc_info=True)
        update(status="error", error=str(e))


@app.post("/api/dub/{job_id}/retranslate")
async def retranslate(
    job_id: str,
    model: str = Form(""),
    context_hint: str = Form(""),
    target_lang: str = Form(""),
):
    """Re-run ONLY the translation step using the saved transcription
    checkpoint. Useful when the user wants to try a different model,
    adjust the context hint, or switch target language mid-flight."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]
    cp = _load_checkpoint(job_id, "transcription_done")
    if not cp:
        return JSONResponse(
            {"error": "No transcription checkpoint — start with wizard_mode"}, 404
        )

    final_model = model or cp.get("model", "gemma4:e4b")
    final_context = context_hint if context_hint else cp.get("context_hint", "")
    final_target = target_lang or cp.get("target_lang", "ru")

    asyncio.create_task(_retranslate_stage(
        job_id, cp, final_model, final_context, final_target,
    ))
    return {"ok": True, "job_id": job_id}


async def _retranslate_stage(job_id: str, cp: dict, model: str,
                               context_hint: str, target_lang: str):
    job = jobs.get(job_id)
    if not job:
        return
    work = OUTPUT_DIR / job_id
    def update(**kwargs):
        if job.get("cancel_requested"):
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs); save_job(job)
    try:
        update(status="translating", progress=45, model=model,
               context_hint=context_hint, target_lang=target_lang,
               step_detail=f"Retranslating with {model}...")
        effective_src = cp.get("effective_src", "en")
        segments = await translate_segments(
            cp["segments"], effective_src, target_lang, model,
            context_hint=context_hint,
        )
        _save_checkpoint(job_id, work, stage="translation_done", data={
            **cp,
            "target_lang": target_lang,
            "segments": [
                {
                    "idx": i, "start": s["start"], "end": s["end"],
                    "text": s["text"],
                    "translated_text": s.get("translated_text", ""),
                    "speaker": s.get("speaker", "SPEAKER_00"),
                }
                for i, s in enumerate(segments)
            ],
        })
        update(
            status="awaiting_translation_review", progress=63,
            step_detail="Retranslated — review and continue",
            checkpoint_stage="translation_done",
        )
    except Exception as e:
        log.error(f"[retranslate] Failed: {e}", exc_info=True)
        update(status="error", error=str(e))


@app.post("/api/dub/{job_id}/regenerate_segment/{seg_idx}")
async def regenerate_segment(
    job_id: str, seg_idx: int,
    # Accept both 'translated_text' (UI form field name) and 'new_text'
    # (legacy) — whichever is populated wins.
    translated_text: str = Form(""),
    new_text: str = Form(""),
    voice_style: str = Form(""),
    voice_preset: str = Form(""),
    reference: Optional[UploadFile] = File(None),
):
    """Regenerate a SINGLE TTS segment. Optionally lets user edit the
    translated text and/or use a different voice for just this one line.
    After the new audio is rendered, the final video is rebuilt so the
    player immediately reflects the change."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    job = jobs[job_id]
    cp = _load_checkpoint(job_id, "tts_done")
    if not cp:
        return JSONResponse(
            {"error": "Per-segment regen requires tts_done checkpoint"}, 404
        )

    # Find the target segment
    segs = cp.get("segments", [])
    target = next((s for s in segs if s.get("idx") == seg_idx), None)
    if not target:
        return JSONResponse({"error": f"Segment {seg_idx} not found"}, 404)

    # Apply edits if any. Persist them to BOTH checkpoints immediately so
    # that even if the regen crashes, the user's edit isn't lost. Also
    # updates translation_done so a later full re-translate would only wipe
    # edits when the user explicitly requests it.
    edited = (translated_text or new_text).strip()
    if edited and edited != (target.get("translated_text") or "").strip():
        target["translated_text"] = edited
        # Also update the translation_done checkpoint so retranslate baseline
        # reflects the user's latest edit
        tcp = _load_checkpoint(job_id, "translation_done")
        if tcp:
            for ts in tcp.get("segments", []):
                if ts.get("idx") == seg_idx:
                    ts["translated_text"] = edited
                    break
            _save_checkpoint(job_id, OUTPUT_DIR / job_id,
                             stage="translation_done", data=tcp)
        log.info(f"[regen_seg] Edited segment {seg_idx} text: "
                 f"'{edited[:50]}{'...' if len(edited) > 50 else ''}'")

    ref_path = ""
    if reference and reference.filename:
        ref_ext = Path(reference.filename).suffix or ".wav"
        ref_path = str(UPLOAD_DIR / f"{job_id}_seg{seg_idx}_ref{ref_ext}")
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(reference.file, f)

    # Delete the old audio_path so _run_tts_and_merge_stage treats it as "to do"
    old_audio = target.get("audio_path", "")
    if old_audio and os.path.exists(old_audio):
        try:
            os.remove(old_audio)
        except Exception:
            pass
    target["audio_path"] = ""

    # Persist the updated tts_done checkpoint to disk BEFORE dispatching the
    # async regen task. The stage reloads checkpoints per-run, and the
    # background task runs against THIS modified cp dict in-memory anyway,
    # but saving now guards against server crash between dispatch and save.
    _save_checkpoint(job_id, OUTPUT_DIR / job_id, stage="tts_done", data=cp)

    final_style = voice_style if voice_style else job.get("voice_style", "")
    final_preset = voice_preset if voice_preset else job.get("voice_preset", "auto")
    final_speed = job.get("tts_speed", "balanced")

    asyncio.create_task(_regen_single_segment(
        job_id, cp, seg_idx, final_style, final_preset, final_speed, ref_path,
    ))
    return {"ok": True, "job_id": job_id, "seg_idx": seg_idx}


async def _regen_single_segment(
    job_id: str, cp: dict, seg_idx: int,
    voice_style: str, voice_preset: str, tts_speed: str, ref_path: str,
):
    job = jobs.get(job_id)
    if not job:
        return
    work = OUTPUT_DIR / job_id
    def update(**kwargs):
        if job.get("cancel_requested"):
            _maybe_terminate_tts_worker()
            raise JobCancelled(f"Job {job_id} cancelled by user")
        job.update(kwargs); save_job(job)
    try:
        update(status="synthesizing", progress=65,
               step_detail=f"Regenerating segment {seg_idx+1}...")
        # preserve_existing_audio_paths=True — only the cleared one will re-synth
        await _run_tts_and_merge_stage(
            job, work, cp,
            voice_style=voice_style, voice_preset=voice_preset,
            tts_speed=tts_speed, ref_path_override=ref_path,
            preserve_existing_audio_paths=True,
        )
    except Exception as e:
        log.error(f"[regen_seg] Failed: {e}", exc_info=True)
        update(status="error", error=str(e))


def _voice_preset_payload() -> dict:
    """Build the full presets list response used by both /api/voices
    and /api/voice_presets endpoints."""
    style_presets = [
        {"id": k, "name": v["name"], "style": v["style"], "type": "style"}
        for k, v in VOICE_PRESETS.items()
    ]
    file_presets = []
    for k, v in scan_file_presets().items():
        file_presets.append({
            "id": k, "name": v["name"], "style": v.get("style", ""),
            "type": "file",
            "description": v.get("description", ""),
            "reference_file": os.path.basename(v["reference_file"]),
            "gender": v.get("gender", ""),
            "language": v.get("language", ""),
            "tags": v.get("tags", []),
            "created_at": v.get("created_at"),
            "file_size": v.get("file_size", 0),
            "file_ext": v.get("file_ext", ""),
            "audio_url": v.get("audio_url"),
        })
    return {"presets": file_presets + style_presets}


@app.get("/api/voices")
async def list_voice_presets():
    """List all available voice presets (built-in styles + user file presets)."""
    return _voice_preset_payload()


@app.get("/api/voice_presets")
async def list_voice_presets_v2():
    """Alias for /api/voices — preferred name for new clients (CLI, MCP)."""
    return _voice_preset_payload()


def _file_preset_path(preset_id: str) -> Optional[Path]:
    """Resolve a `file:NAME` preset id back to its actual audio file on
    disk, or None if not found / not a file preset."""
    if not preset_id.startswith("file:"):
        return None
    name = preset_id[len("file:"):]
    # No path traversal — the name is just a basename, must not contain separators
    if "/" in name or "\\" in name or ".." in name:
        return None
    for ext in _VOICE_AUDIO_EXTS:
        p = VOICE_PRESETS_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


@app.get("/api/voice_presets/{preset_id}/audio")
async def get_voice_preset_audio(preset_id: str):
    """Stream the audio file behind a file-based preset. Used by the
    Voices tab's inline player and the dub form's preview."""
    from fastapi.responses import FileResponse
    p = _file_preset_path(preset_id)
    if not p:
        return JSONResponse({"error": "Preset not found or not a file preset"}, 404)
    media = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4",
    }.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media, filename=p.name)


_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-]{0,49}$")


def _sanitize_voice_name(raw: str) -> Optional[str]:
    """Normalize a user-provided voice name to a filesystem-safe stem.
    Returns None if invalid (would let through path traversal or weird chars)."""
    if not raw:
        return None
    s = raw.strip()
    # Collapse internal whitespace runs, strip trailing dots
    s = re.sub(r"\s+", " ", s).rstrip(".")
    if not _VOICE_NAME_RE.match(s):
        return None
    return s


@app.post("/api/voice_presets")
async def create_voice_preset(
    audio: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    gender: str = Form(""),
    language: str = Form(""),
    tags: str = Form(""),                  # comma-separated
    style: str = Form(""),
):
    """Upload a new voice reference. Saves the audio as
    `presets/voices/<name>.<ext>` and writes a JSON sidecar with the
    structured metadata. Re-upload with the same name overwrites.

    The new preset is immediately usable in any dub form (id = `file:<name>`).
    """
    clean = _sanitize_voice_name(name)
    if not clean:
        return JSONResponse({
            "error": "Name must be 1-50 chars, letters/digits/space/dash/underscore, "
                     "starting with a letter or digit."
        }, 400)

    if not audio.filename:
        return JSONResponse({"error": "No audio file provided"}, 400)
    ext = Path(audio.filename).suffix.lower()
    if ext not in _VOICE_AUDIO_EXTS:
        return JSONResponse({
            "error": f"Unsupported audio extension '{ext}'. Use one of: "
                     f"{', '.join(_VOICE_AUDIO_EXTS)}"
        }, 400)

    VOICE_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = VOICE_PRESETS_DIR / f"{clean}{ext}"
    # If a file with same stem but different ext already exists, remove the
    # old one so we don't keep duplicates (e.g. user re-uploads as mp3).
    for old_ext in _VOICE_AUDIO_EXTS:
        old = VOICE_PRESETS_DIR / f"{clean}{old_ext}"
        if old.exists() and old != audio_path:
            try:
                old.unlink()
            except Exception as e:
                log.warning(f"[voice_presets] couldn't remove old {old.name}: {e}")

    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        return JSONResponse({"error": f"Couldn't save audio: {e}"}, 500)

    # Normalize tags
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    meta = {
        "display_name": clean,
        "description": (description or "").strip()[:500],
        "gender": (gender or "").strip().lower(),
        "language": (language or "").strip().lower(),
        "tags": tag_list,
        "style": (style or "").strip()[:300],
        "created_at": time.time(),
    }
    try:
        _voice_metadata_path(audio_path).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"[voice_presets] couldn't write metadata: {e}")

    log.info(f"[voice_presets] created '{clean}' ({audio_path.stat().st_size} bytes)")
    pid = f"file:{clean}"
    return {"ok": True, "id": pid, "preset": scan_file_presets().get(pid, {})}


@app.put("/api/voice_presets/{preset_id}")
async def update_voice_preset(
    preset_id: str,
    name: Optional[str] = Form(None),       # rename
    description: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
):
    """Update metadata for a file-based preset. Optionally rename it
    (renames the audio file + sidecar). Fields left as None are preserved."""
    src_audio = _file_preset_path(preset_id)
    if not src_audio:
        return JSONResponse({"error": "File preset not found"}, 404)

    # Load existing metadata, then merge in updates
    meta = _read_voice_metadata(src_audio)
    if description is not None:
        meta["description"] = description.strip()[:500]
    if gender is not None:
        meta["gender"] = gender.strip().lower()
    if language is not None:
        meta["language"] = language.strip().lower()
    if tags is not None:
        meta["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if style is not None:
        meta["style"] = style.strip()[:300]

    # Rename if a new name was given
    final_audio = src_audio
    new_id = preset_id
    if name is not None:
        clean = _sanitize_voice_name(name)
        if not clean:
            return JSONResponse({"error": "Invalid name"}, 400)
        new_audio = VOICE_PRESETS_DIR / f"{clean}{src_audio.suffix}"
        if new_audio.exists() and new_audio != src_audio:
            return JSONResponse({"error": f"A preset named '{clean}' already exists"}, 409)
        if new_audio != src_audio:
            old_meta_path = _voice_metadata_path(src_audio)
            old_txt_path = src_audio.with_suffix(".txt")
            src_audio.rename(new_audio)
            if old_meta_path.exists():
                old_meta_path.rename(_voice_metadata_path(new_audio))
            if old_txt_path.exists():
                old_txt_path.rename(new_audio.with_suffix(".txt"))
            final_audio = new_audio
            new_id = f"file:{clean}"
            meta["display_name"] = clean

    try:
        _voice_metadata_path(final_audio).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        return JSONResponse({"error": f"Couldn't save metadata: {e}"}, 500)

    log.info(f"[voice_presets] updated '{final_audio.stem}'")
    return {"ok": True, "id": new_id, "preset": scan_file_presets().get(new_id, {})}


@app.delete("/api/voice_presets/{preset_id}")
async def delete_voice_preset(preset_id: str):
    """Delete a file-based preset (audio + metadata sidecars).
    Built-in style presets cannot be deleted."""
    p = _file_preset_path(preset_id)
    if not p:
        return JSONResponse({"error": "File preset not found"}, 404)
    try:
        p.unlink()
        for sidecar in (_voice_metadata_path(p), p.with_suffix(".txt")):
            if sidecar.exists():
                sidecar.unlink()
    except Exception as e:
        return JSONResponse({"error": f"Couldn't delete: {e}"}, 500)
    log.info(f"[voice_presets] deleted '{preset_id}'")
    return {"ok": True, "id": preset_id}


@app.get("/api/preferences")
async def get_preferences():
    """Return saved UI preferences (last-used models, voice, speed, etc).
    The UI uses localStorage as primary but falls back to this when
    localStorage is unavailable (private browsing, cross-device, etc)."""
    if not PREFS_FILE.exists():
        return {}
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read prefs: {e}")
        return {}


@app.post("/api/preferences")
async def set_preferences(prefs: str = Form(...)):
    """Update UI preferences. Merges into existing prefs; doesn't replace."""
    try:
        new_prefs = json.loads(prefs)
        if not isinstance(new_prefs, dict):
            raise ValueError("prefs must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid prefs JSON: {e}"}, 400)
    existing = {}
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(new_prefs)
    try:
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/config")
async def get_config():
    """Return current UserConfig as JSON. Editable fields shown in Settings tab."""
    return cfg.to_dict()


@app.patch("/api/config")
async def patch_config(body: str = Form(...)):
    """Update one or more UserConfig fields and persist to config-user.json."""
    try:
        updates = json.loads(body)
        if not isinstance(updates, dict):
            raise ValueError("body must be a JSON object")
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    try:
        cfg.update(**updates)
        return {"ok": True, "config": cfg.to_dict()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ═══════════════════════════════════════════════════════════════════════
#  Glossary — user-editable term overrides
# ═══════════════════════════════════════════════════════════════════════
# The translator ships with a built-in BJJ glossary baked into
# translator.py. Users doing non-BJJ courses (cooking, tech, music, etc)
# need a way to add their own domain terms without editing Python code.
# Solution: sidecar JSON file at presets/user_glossary.json. Translator
# loads it lazily on each call via _load_user_glossary().
#
# Format (validated lightly, not strictly):
#   { "domains": [
#       { "name": "Cooking EN→RU",
#         "triggers": ["cooking", "recipe"],
#         "target_lang": "ru",
#         "terms": { "sear": "запекать", ... } },
#       ...
#   ] }
# ═══════════════════════════════════════════════════════════════════════

_GLOSSARY_EXAMPLE = {
    "domains": [
        {
            "name": "Example: Cooking EN→RU",
            "triggers": ["cooking", "recipe", "food"],
            "target_lang": "ru",
            "terms": {
                "sear": "обжарить до корочки",
                "simmer": "томить",
                "al dente": "аль денте",
            },
        },
    ],
}


@app.get("/api/glossary")
async def get_glossary():
    """Return the current user glossary JSON + metadata. If the file
    doesn't exist, return an example structure so the UI has something
    sensible to show as the starting template."""
    if not USER_GLOSSARY_FILE.exists():
        return {
            "exists": False,
            "data": _GLOSSARY_EXAMPLE,
            "hint": "File will be created on first save. Built-in BJJ glossary stays active.",
        }
    try:
        with open(USER_GLOSSARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"exists": True, "data": data}
    except Exception as e:
        return JSONResponse({
            "exists": True,
            "error": f"Could not parse glossary file: {e}",
            "raw_text": USER_GLOSSARY_FILE.read_text(encoding="utf-8", errors="replace"),
        }, 500)


@app.post("/api/glossary")
async def set_glossary(body: str = Form(...)):
    """Replace the user glossary. Validates structure server-side so a
    malformed save doesn't break the translator.

    Accepts:
      { "domains": [ { name, triggers, target_lang, terms }, ... ] }
    """
    try:
        data = json.loads(body)
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON: {e}"}, 400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Top level must be an object"}, 400)
    domains = data.get("domains", [])
    if not isinstance(domains, list):
        return JSONResponse({"error": "'domains' must be an array"}, 400)
    # Light validation: each domain must have terms + triggers
    for i, d in enumerate(domains):
        if not isinstance(d, dict):
            return JSONResponse({"error": f"domains[{i}] must be an object"}, 400)
        if not isinstance(d.get("terms", {}), dict):
            return JSONResponse({"error": f"domains[{i}].terms must be an object"}, 400)
        if not isinstance(d.get("triggers", []), list):
            return JSONResponse({"error": f"domains[{i}].triggers must be an array"}, 400)

    # Ensure parent dir exists
    USER_GLOSSARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(USER_GLOSSARY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        total_terms = sum(len(d.get("terms", {})) for d in domains)
        log.info(f"[glossary] Saved {len(domains)} domain(s), {total_terms} term(s) total")
        return {
            "ok": True,
            "domains": len(domains),
            "total_terms": total_terms,
            "path": str(USER_GLOSSARY_FILE),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.delete("/api/glossary")
async def delete_glossary():
    """Remove the user glossary file entirely. Built-in BJJ terms still
    apply; this just clears user additions."""
    if USER_GLOSSARY_FILE.exists():
        try:
            USER_GLOSSARY_FILE.unlink()
            log.info("[glossary] User glossary file removed")
            return {"ok": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, 500)
    return {"ok": True, "note": "File didn't exist"}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Not found"}, 404)
    return jobs[job_id]


@app.post("/api/dub/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job.

    - Queued: removed from the asyncio queue immediately, status=cancelled.
    - Running: sets cancel_requested flag. The pipeline's update() closures
      check this flag at every stage boundary and raise JobCancelled, which
      the queue worker catches and marks as cancelled. For TTS synthesis we
      additionally terminate the persistent VoxCPM subprocess so cancel
      takes effect within 1-2 seconds instead of waiting for the current
      segment to finish rendering (can be 30+ seconds on a long segment).
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, 404)
    j = jobs[job_id]
    status = j.get("status")

    if status in ("complete", "error", "cancelled"):
        return {"ok": False, "reason": f"Job is already {status}"}

    if status == "scheduled":
        # Not yet in the queue — just flip status; scheduler loop will
        # see status != 'scheduled' and skip it on next poll.
        j["status"] = "cancelled"
        j["step_detail"] = "Cancelled before scheduled start"
        j.pop("_pending_args", None)
        save_job(j)
        log.info(f"[scheduler] Job {job_id} cancelled before scheduled start")
        return {"ok": True, "cancelled_from": "scheduled"}

    if status == "queued":
        # Drain queue, drop target job, push rest back. asyncio.Queue
        # doesn't support random removal directly.
        drained = []
        while not _job_queue.empty():
            try:
                item = _job_queue.get_nowait()
                if item[0] != job_id:
                    drained.append(item)
            except asyncio.QueueEmpty:
                break
        for item in drained:
            await _job_queue.put(item)
        j["status"] = "cancelled"
        j["step_detail"] = "Cancelled before start"
        save_job(j)
        log.info(f"[queue] Job {job_id} cancelled (removed from queue)")
        return {"ok": True, "cancelled_from": "queue"}

    # Running job — set flag, terminate TTS subprocess if mid-synth.
    # The pipeline's update() closures will pick up the flag at the next
    # stage boundary and raise JobCancelled cleanly.
    j["cancel_requested"] = True
    j["step_detail"] = "Cancelling..."
    save_job(j)
    # Proactively kill TTS worker so synth segments don't have to finish
    if status == "synthesizing":
        _maybe_terminate_tts_worker()
    log.info(f"[queue] Job {job_id} cancel requested (was {status})")
    return {"ok": True, "cancelled_from": "running",
            "message": "Cancel requested. Job will stop at the next stage boundary "
                       "(usually within 1-5 seconds)."}


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Not found"}, 404)
    work = OUTPUT_DIR / job_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    delete_job_db(job_id)
    jobs.pop(job_id, None)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════
#  Storage management — keep outputs/ from eating your drive
# ═══════════════════════════════════════════════════════════════════════
# Each dub leaves ~100-500 MB in outputs/{job_id}/ (source video, audio
# extracts, per-segment WAVs, dubbed video). Over a 5-course semester
# that's easily 20-50 GB.
#
# Design:
#   - Jobs can be "starred" to protect them from bulk cleanup. Stars
#     persist in the job dict (jobs[id].starred = True).
#   - /api/storage/stats returns total disk usage per job + aggregate.
#   - /api/storage/cleanup takes { older_than_days, mode } and deletes
#     jobs matching the criteria, skipping starred jobs. mode can be:
#       "all_files"  — rm -rf the whole outputs/{job_id} dir
#       "intermediate" — keep dubbed_video.mp4 + .srt, delete the rest
#         (source video, per-segment WAVs, intermediate audio ~90% saving)
#   - Dry-run by default; caller must pass dry_run=false to actually delete.
# ═══════════════════════════════════════════════════════════════════════

def _dir_size_bytes(path: Path) -> int:
    """Fast recursive directory size via os.scandir. Returns 0 on error."""
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += _dir_size_bytes(Path(entry.path))
            except OSError:
                continue
    except OSError:
        pass
    return total


# Files we keep when cleaning "intermediate" artifacts. These are the
# user-visible deliverables; everything else is regenerable from checkpoints.
_KEEP_ON_INTERMEDIATE_CLEAN = {
    "dubbed_video.mp4",
    "dubbed_video_subs.mp4",  # burn-in output
    "translated.srt",
    "checkpoint_translation_done.json",
    "checkpoint_tts_done.json",
    # Keep final so Resume stays possible even on cleaned jobs
}


@app.get("/api/storage/stats")
async def storage_stats():
    """Aggregate disk usage across all job outputs. Per-job breakdown so
    the UI can show a sortable list and identify the biggest offenders."""
    rows = []
    total_bytes = 0
    for jid, job in jobs.items():
        work = OUTPUT_DIR / jid
        if not work.exists():
            continue
        size = _dir_size_bytes(work)
        total_bytes += size
        rows.append({
            "id": jid,
            "label": job.get("source_label", jid),
            "status": job.get("status", ""),
            "target_lang": job.get("target_lang", ""),
            "created": job.get("created", 0),
            "completed_at": job.get("completed_at", 0),
            "starred": bool(job.get("starred")),
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 1),
            "age_days": round((time.time() - job.get("created", time.time())) / 86400, 1),
        })
    rows.sort(key=lambda r: -r["size_bytes"])
    return {
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 1),
        "total_gb": round(total_bytes / 1024 / 1024 / 1024, 2),
        "job_count": len(rows),
        "jobs": rows,
    }


@app.post("/api/storage/star/{job_id}")
async def toggle_star(job_id: str, starred: bool = Form(...)):
    """Star/unstar a job to protect it from bulk cleanup."""
    if job_id not in jobs:
        return JSONResponse({"error": "Not found"}, 404)
    jobs[job_id]["starred"] = bool(starred)
    save_job(jobs[job_id])
    return {"ok": True, "starred": jobs[job_id]["starred"]}


@app.post("/api/storage/cleanup")
async def cleanup_storage(
    older_than_days: int = Form(30),
    mode: str = Form("intermediate"),  # "intermediate" | "all_files"
    dry_run: bool = Form(True),
    include_errored: bool = Form(True),
    include_cancelled: bool = Form(True),
):
    """Bulk-remove old job outputs. Starred jobs are always skipped.

    - older_than_days=N — affects jobs created more than N days ago
    - mode=intermediate — keep the dubbed mp4 + srt + checkpoints, drop
      source video + per-segment WAVs + intermediate audio (~90% saving
      on typical jobs, job stays "viewable" but Regenerate may need
      re-download)
    - mode=all_files — rm -rf the whole job output dir (job record kept;
      UI will show "(files deleted)" and no View button)
    - dry_run=True — report what would be deleted, don't touch disk.
    """
    if mode not in ("intermediate", "all_files"):
        return JSONResponse({"error": f"Invalid mode: {mode}"}, 400)

    cutoff = time.time() - older_than_days * 86400
    candidates = []
    for jid, job in jobs.items():
        if job.get("starred"):
            continue
        if job.get("created", 0) > cutoff:
            continue
        status = job.get("status", "")
        # Only touch jobs in a settled state. Don't clean a running job.
        if status not in ("complete", "error", "cancelled"):
            continue
        if status == "error" and not include_errored:
            continue
        if status == "cancelled" and not include_cancelled:
            continue
        work = OUTPUT_DIR / jid
        if not work.exists():
            continue
        candidates.append((jid, job, work))

    bytes_freed = 0
    deleted_files = []
    errors = []
    for jid, job, work in candidates:
        try:
            if mode == "all_files":
                size = _dir_size_bytes(work)
                if not dry_run:
                    shutil.rmtree(work, ignore_errors=True)
                    # Mark job as files-deleted so UI can show it without
                    # trying to link to missing mp4. Keep db entry so
                    # history preserves the metadata.
                    job["files_deleted"] = True
                    save_job(job)
                deleted_files.append({"id": jid, "mode": "all_files",
                                       "size_mb": round(size/1024/1024, 1)})
                bytes_freed += size
            else:  # intermediate
                freed_here = 0
                for entry in list(os.scandir(work)):
                    if entry.name in _KEEP_ON_INTERMEDIATE_CLEAN:
                        continue
                    try:
                        if entry.is_file(follow_symlinks=False):
                            sz = entry.stat().st_size
                            if not dry_run:
                                Path(entry.path).unlink()
                            freed_here += sz
                        elif entry.is_dir(follow_symlinks=False):
                            sz = _dir_size_bytes(Path(entry.path))
                            if not dry_run:
                                shutil.rmtree(entry.path, ignore_errors=True)
                            freed_here += sz
                    except OSError as e:
                        errors.append(f"{jid}: {entry.name}: {e}")
                if freed_here > 0:
                    deleted_files.append({"id": jid, "mode": "intermediate",
                                          "size_mb": round(freed_here/1024/1024, 1)})
                    bytes_freed += freed_here
                if not dry_run:
                    job["intermediate_cleaned"] = True
                    save_job(job)
        except Exception as e:
            errors.append(f"{jid}: {e}")

    action = "would delete" if dry_run else "deleted"
    log.info(f"[cleanup] {action} {len(deleted_files)} jobs, "
             f"{round(bytes_freed/1024/1024/1024, 2)} GB freed "
             f"(mode={mode}, older_than={older_than_days}d, dry_run={dry_run})")
    return {
        "dry_run": dry_run,
        "mode": mode,
        "candidates": len(candidates),
        "affected": len(deleted_files),
        "bytes_freed": bytes_freed,
        "mb_freed": round(bytes_freed / 1024 / 1024, 1),
        "gb_freed": round(bytes_freed / 1024 / 1024 / 1024, 2),
        "details": deleted_files[:50],  # cap to keep payload small
        "errors": errors[:10],
    }


@app.get("/api/jobs")
async def list_jobs():
    # Annotate each job with checkpoint info so the History UI can
    # decide whether to show a Resume button. This is intentionally
    # done at read time (not stored on the job object) because users
    # can delete output dirs manually — reading live keeps UI honest.
    sorted_jobs = sorted(jobs.values(), key=lambda j: j.get("created", 0), reverse=True)
    enriched = []
    for j in sorted_jobs:
        info = _job_checkpoint_info(j["id"])
        # Shallow-copy so we don't mutate the in-memory job store
        enriched.append({**j, **info})
    return {"jobs": enriched}


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs or jobs[job_id].get("status") != "complete":
        return JSONResponse({"error": "Not ready"}, 400)
    path = str(OUTPUT_DIR / job_id / "dubbed_video.mp4")
    if not os.path.exists(path):
        return JSONResponse({"error": "File missing"}, 404)
    return FileResponse(path, filename=f"dubbed_{job_id}.mp4")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    print("")
    print("+====================================================+")
    print("|  TachiDUBB Studio - AI Video Dubbing               |")
    print("|  Opening browser at http://localhost:8910...       |")
    print("|  Press Ctrl+C to stop                              |")
    print("+====================================================+")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8910, log_level="info")
