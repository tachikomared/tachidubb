"""System status + model catalog for plug-and-play UX."""
import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger("tachidubb.models")


# Curated catalog - shown in the UI's Models panel
MODEL_CATALOG = {
    "translation": [
        {
            "id": "gemma4:e4b",
            "name": "Gemma 4 E4B",
            "size": "9.6 GB",
            "vram": "~10 GB",
            "description": "Google's newest. Top multilingual quality. Needs Ollama 0.20+.",
        },
        {
            "id": "gemma4:e2b",
            "name": "Gemma 4 E2B",
            "size": "7.2 GB",
            "vram": "~8 GB",
            "description": "Smaller Gemma 4. Fast with strong quality. Needs Ollama 0.20+.",
        },
        {
            "id": "gemma4:26b",
            "name": "Gemma 4 26B (MoE)",
            "size": "18 GB",
            "vram": "~20 GB",
            "description": "Mixture-of-Experts. Best translation quality. Needs 20 GB+ VRAM.",
        },
        {
            "id": "gemma4:31b",
            "name": "Gemma 4 31B",
            "size": "20 GB",
            "vram": "~22 GB",
            "description": "Largest dense Gemma 4. Frontier quality.",
        },
        {
            "id": "qwen3:8b",
            "name": "Qwen3 8B",
            "size": "5.2 GB",
            "vram": "~6 GB",
            "description": "Recommended default. Fast, excellent quality across 100+ languages.",
            "recommended": True,
        },
        {
            "id": "qwen3:14b",
            "name": "Qwen3 14B",
            "size": "8.2 GB",
            "vram": "~10 GB",
            "description": "Best overall translation quality. Needs 10GB+ VRAM.",
        },
        {
            "id": "qwen3:4b",
            "name": "Qwen3 4B",
            "size": "2.6 GB",
            "vram": "~3 GB",
            "description": "Lightweight. Good for low-VRAM systems.",
        },
        {
            "id": "gemma3:12b",
            "name": "Gemma 3 12B",
            "size": "7.5 GB",
            "vram": "~8 GB",
            "description": "Previous Gemma gen. Strong on European languages.",
        },
        {
            "id": "gemma3:4b",
            "name": "Gemma 3 4B",
            "size": "3.3 GB",
            "vram": "~4 GB",
            "description": "Compact Gemma. Very fast.",
        },
        {
            "id": "llama3.2:3b",
            "name": "Llama 3.2 3B",
            "size": "2.0 GB",
            "vram": "~3 GB",
            "description": "Meta's compact model. Fast, decent quality.",
        },
    ],
    "voxcpm": [
        {
            "id": "openbmb/VoxCPM2",
            "name": "VoxCPM2 (2B)",
            "size": "~5 GB",
            "vram": "~8 GB",
            "description": "Latest. 30 languages, 48kHz output, voice cloning + voice design.",
            "recommended": True,
        },
        {
            "id": "openbmb/VoxCPM1.5",
            "name": "VoxCPM 1.5 (800M)",
            "size": "~2 GB",
            "vram": "~4 GB",
            "description": "Previous version, lighter. English + Chinese.",
        },
    ],
}


def check_python():
    major, minor = sys.version_info.major, sys.version_info.minor
    version = f"{major}.{minor}.{sys.version_info.micro}"
    ok = major == 3 and 10 <= minor <= 12
    return {
        "ok": ok,
        "version": version,
        "note": "" if ok else "VoxCPM2 requires Python 3.10-3.12",
    }


def check_ffmpeg():
    p = shutil.which("ffmpeg")
    version = ""
    if p:
        try:
            r = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=5)
            version = r.stdout.split("\n")[0] if r.stdout else ""
        except Exception:
            pass
    return {"ok": p is not None, "path": p or "", "version": version}


def check_ytdlp():
    # Check binary + python module
    binary = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    has_module = False
    try:
        import yt_dlp  # noqa
        has_module = True
    except ImportError:
        pass
    return {"ok": binary is not None or has_module, "path": binary or "", "module": has_module}


def check_whisper():
    try:
        import faster_whisper  # noqa
        return {"ok": True, "type": "faster-whisper"}
    except ImportError:
        pass
    except Exception as e:
        return {"ok": False, "type": "broken", "error": str(e)[:150]}
    try:
        import whisper  # noqa
        return {"ok": True, "type": "openai-whisper"}
    except ImportError:
        pass
    except Exception as e:
        return {"ok": False, "type": "broken", "error": str(e)[:150]}
    return {"ok": False, "type": "none"}


def check_voxcpm():
    try:
        import voxcpm  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False}
    except Exception as e:
        # CUDA init errors, DLL load errors, etc. - treat as installed-but-broken
        return {"ok": False, "error": str(e)[:150]}


def check_edge_tts():
    try:
        import edge_tts  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False}


def check_demucs():
    try:
        import demucs  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False, "hint": "pip install demucs"}
    except Exception as e:
        # demucs imports torch — broken torch raises OSError here
        return {"ok": False, "error": str(e)[:150]}


def check_f5tts():
    try:
        import f5_tts  # noqa
        return {"ok": True}
    except ImportError:
        return {"ok": False, "hint": "pip install f5-tts"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:150]}


def check_silero_vad():
    try:
        import torch  # noqa
        return {"ok": True}
    except (ImportError, OSError):
        return {"ok": False, "hint": "pip install torch"}


def check_pyannote():
    try:
        import pyannote.audio  # noqa
        has_token = bool(os.getenv("HF_TOKEN", ""))
        return {"ok": True, "has_token": has_token}
    except ImportError:
        return {"ok": False, "has_token": False}
    except Exception as e:
        return {"ok": False, "has_token": False, "error": str(e)[:150]}


def check_gpu():
    # NOTE: catches Exception (not just ImportError) because on Windows,
    # a broken torch install (mismatched CUDA wheel, missing VC++ runtime,
    # etc.) raises OSError WinError 127 from `import torch` itself —
    # which would otherwise crash /api/system on every poll.
    try:
        import torch
        if not torch.cuda.is_available():
            return {"ok": False, "name": "", "vram_gb": 0}
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Free VRAM estimate (may not be fully accurate but useful)
        free = 0
        try:
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free = free_bytes / 1024**3
        except Exception:
            free = total
        return {
            "ok": True,
            "name": name,
            "vram_gb": round(total, 1),
            "vram_free_gb": round(free, 1),
        }
    except Exception as e:
        return {
            "ok": False,
            "name": "",
            "vram_gb": 0,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def check_ollama_binary():
    return {"ok": shutil.which("ollama") is not None}


def get_system_status():
    """Aggregate all system checks for the UI."""
    return {
        "python": check_python(),
        "ffmpeg": check_ffmpeg(),
        "yt_dlp": check_ytdlp(),
        "whisper": check_whisper(),
        "voxcpm": check_voxcpm(),
        "f5tts": check_f5tts(),
        "demucs": check_demucs(),
        "silero_vad": check_silero_vad(),
        "edge_tts": check_edge_tts(),
        "pyannote": check_pyannote(),
        "gpu": check_gpu(),
        "ollama_binary": check_ollama_binary(),
    }
