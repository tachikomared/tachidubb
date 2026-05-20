"""Voice Activity Detection — strip non-speech regions before Whisper.

WhisperX already uses Silero VAD internally, but running it explicitly
upfront lets us:
  1. Remove long silence / music intros that cause Whisper hallucinations
  2. Report how much of the audio is actually speech (diagnostic)
  3. Optionally gate transcription on minimum speech ratio

Uses silero-vad (ONNX-based, 1 MB model, no GPU required).
Falls back gracefully if silero-vad is not installed.
"""
import logging
import os
import subprocess
import tempfile

log = logging.getLogger("tachidubb.vad")

# Minimum ratio of speech to total audio — below this we warn the user
SPEECH_RATIO_WARNING = 0.15

# Padding added around each speech segment (seconds) to avoid clipping
SEGMENT_PAD = 0.1


def _run_ffmpeg(cmd, desc="", timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} failed: {r.stderr[:300]}")
    return r


def get_speech_timestamps(audio_path: str, threshold: float = 0.5) -> list[dict]:
    """Return Silero VAD timestamps as list of {start, end} dicts (seconds).

    Falls back to [{start: 0, end: duration}] if silero-vad not installed,
    so callers don't need to special-case the missing-dependency path.
    """
    try:
        import torch
        import torchaudio  # noqa — needed by silero-vad load_silero_vad

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            verbose=False,
        )
        get_ts = utils[0]  # get_speech_timestamps is utils[0]
        read_audio = utils[2]

        wav = read_audio(audio_path, sampling_rate=16000)
        timestamps = get_ts(wav, model, sampling_rate=16000, threshold=threshold)
        result = [
            {"start": t["start"] / 16000, "end": t["end"] / 16000}
            for t in timestamps
        ]
        log.info(f"VAD: {len(result)} speech segments detected")
        return result
    except ImportError:
        log.debug("silero-vad not installed — VAD skipped, using full audio")
        return []
    except Exception as e:
        log.warning(f"VAD failed ({e}) — using full audio")
        return []


def _get_duration_ffprobe(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def apply_vad_filter(audio_path: str, output_path: str,
                     threshold: float = 0.5) -> tuple[str, float]:
    """Extract only speech regions from audio_path into output_path.

    Returns (output_path, speech_ratio) where speech_ratio is the fraction
    of the original audio that contains detected speech (0.0-1.0).

    If silero-vad isn't installed or VAD finds no segments, copies the
    original audio unchanged and returns speech_ratio=1.0 (conservative).

    This helps Whisper in two ways:
      1. Removes long music intros that cause hallucinations like
         "Translated by XYZ" or repeated filler phrases.
      2. Reduces total audio length → faster transcription.
    """
    total_dur = _get_duration_ffprobe(audio_path)
    if total_dur <= 0:
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, 1.0

    timestamps = get_speech_timestamps(audio_path, threshold=threshold)
    if not timestamps:
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, 1.0

    # Add padding and clamp to audio bounds
    padded = []
    for t in timestamps:
        s = max(0.0, t["start"] - SEGMENT_PAD)
        e = min(total_dur, t["end"] + SEGMENT_PAD)
        padded.append((s, e))

    # Merge overlapping/adjacent segments
    merged = []
    for s, e in sorted(padded):
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])

    speech_seconds = sum(e - s for s, e in merged)
    speech_ratio = speech_seconds / total_dur if total_dur > 0 else 1.0

    if speech_ratio < SPEECH_RATIO_WARNING:
        log.warning(
            f"VAD: only {speech_ratio*100:.0f}% speech detected in audio. "
            f"Background music or silence may affect transcription quality."
        )

    if speech_ratio > 0.90:
        # Almost all speech — skip filtering, not worth the overhead
        log.info(
            f"VAD: {speech_ratio*100:.0f}% speech — audio is dense, skipping filter"
        )
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, speech_ratio

    # Build ffmpeg filter: select speech intervals + concatenate
    # atrim=start=X:end=Y, then concat all pieces
    pieces = []
    for i, (s, e) in enumerate(merged):
        pieces.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )

    n = len(merged)
    concat_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_complex = ";".join(pieces) + f";{concat_inputs}concat=n={n}:v=0:a=1[out]"

    try:
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            output_path,
        ], "VAD filter", timeout=300)
        log.info(
            f"VAD: filtered {total_dur:.0f}s → {speech_seconds:.0f}s "
            f"({speech_ratio*100:.0f}% speech, {n} segments)"
        )
        return output_path, speech_ratio
    except Exception as e:
        log.warning(f"VAD ffmpeg filter failed ({e}) — using full audio")
        import shutil
        shutil.copy2(audio_path, output_path)
        return output_path, 1.0
