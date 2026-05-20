"""Download videos from YouTube or validate local paths (Windows-safe)."""
import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger("tachidubb.downloader")


def _find_ytdlp() -> str:
    """Find yt-dlp executable across platforms."""
    exe_dir = os.path.dirname(sys.executable)
    # On Windows venv: python.exe lives in venv\Scripts\, yt-dlp.exe is a sibling.
    # On Linux venv:   python lives in venv/bin/, yt-dlp is a sibling.
    if sys.platform.startswith("win"):
        candidates = [
            os.path.join(exe_dir, "yt-dlp.exe"),      # inside venv\Scripts
            os.path.join(exe_dir, "..", "Scripts", "yt-dlp.exe"),  # from venv root
            "yt-dlp.exe",                              # PATH
            "yt-dlp",
        ]
    else:
        candidates = [
            os.path.join(exe_dir, "yt-dlp"),
            "yt-dlp",
        ]

    for c in candidates:
        # shutil.which accepts full paths
        found = shutil.which(c)
        if found:
            return found
        # also accept if it's just a direct file
        if os.path.isfile(c):
            return c

    # Last resort: use Python module invocation
    return None  # caller will use `python -m yt_dlp`


def download_video(source: str, output_dir: str) -> str:
    """
    Download from YouTube URL or copy local file.
    Returns path to the local video file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "source_video.mp4")

    # Local file
    if os.path.isfile(source):
        if os.path.abspath(source) != os.path.abspath(output_path):
            shutil.copy2(source, output_path)
        log.info(f"Local file ready: {output_path}")
        return output_path

    # YouTube / HTTP URL
    if not (source.startswith("http://") or source.startswith("https://")):
        raise ValueError(f"Not a file or URL: {source}")

    ytdlp = _find_ytdlp()

    if ytdlp is None:
        # Fall back to Python module invocation
        base_cmd = [sys.executable, "-m", "yt_dlp"]
        log.info("Using: python -m yt_dlp (no binary found)")
    else:
        base_cmd = [ytdlp]
        log.info(f"Using yt-dlp: {ytdlp}")

    cmd = base_cmd + [
        "--no-playlist",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-check-certificates",
        "--retries", "3",
        "--socket-timeout", "30",
        "--no-warnings",
        source,
    ]

    log.info(f"Downloading: {source}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        log.warning(f"First attempt failed: {result.stderr[:200]}")
        # Fallback with simpler format
        cmd_fallback = base_cmd + [
            "--no-playlist",
            "-f", "best[ext=mp4]/best",
            "-o", output_path,
            "--no-check-certificates",
            source,
        ]
        result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            raise RuntimeError(f"YouTube download failed: {result.stderr[:300]}")

    # yt-dlp sometimes adds extensions; find the actual file
    if not os.path.exists(output_path):
        for f in os.listdir(output_dir):
            if f.startswith("source_video") and f.endswith((".mp4", ".mkv", ".webm")):
                actual = os.path.join(output_dir, f)
                if actual != output_path:
                    os.rename(actual, output_path)
                break

    if not os.path.exists(output_path):
        raise RuntimeError("Download completed but video file not found")

    size_mb = os.path.getsize(output_path) / 1048576
    log.info(f"Downloaded: {output_path} ({size_mb:.1f} MB)")
    return output_path
