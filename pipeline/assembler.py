"""Time-aligned audio assembly, per-segment + global loudness normalization."""
import logging
import os
import subprocess

log = logging.getLogger("tachidubb.assembler")

# Target peak after per-segment peak-normalization (avoids whisper-vs-shout jumps
# between consecutive TTS outputs from VoxCPM2 when fed different reference clips).
PER_SEGMENT_PEAK = 0.7

# EBU R128 loudnorm targets (broadcast-safe, closer to YouTube spec)
LN_I = -16      # integrated loudness LUFS
LN_TP = -1.5    # true peak dBTP
LN_LRA = 11     # loudness range


def _run(cmd, desc="", timeout=600):
    log.info(f"[{desc}]")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} failed: {r.stderr[:400]}")
    return r


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            text = seg.get("translated_text", seg["text"])
            f.write(f"{i+1}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{text}\n\n")


def _normalize_loudness_inplace(wav_path: str) -> bool:
    """Apply ffmpeg loudnorm + anti-screech chain to wav_path in place.

    VoxCPM occasionally produces brief inter-segment pops/clicks and spiky
    transients that sound like "screech" or "digital crunch" to the ear.
    We add a 3-stage post-processing chain BEFORE loudnorm:

    1. adeclick    — removes isolated clicks/pops (<50ms transients)
    2. highpass=20 — cuts DC offset and sub-sonic rumble that can amplify
                     after loudnorm gain-up
    3. alimiter    — brick-wall limiter, catches any remaining transient
                     peaks above -1 dBTP that would otherwise clip post-loudnorm

    Then loudnorm itself for broadcast-level consistency. Returns True on success.
    """
    try:
        tmp = wav_path + ".ln.wav"
        # Filter chain: declick → DC-block/rumble-cut → limit → normalize
        af = (
            "adeclick,"
            "highpass=f=20,"
            "alimiter=limit=0.95:level=disabled:attack=5:release=50,"
            f"loudnorm=I={LN_I}:TP={LN_TP}:LRA={LN_LRA}"
        )
        _run([
            "ffmpeg", "-y", "-i", wav_path,
            "-af", af,
            "-ar", "48000", tmp,
        ], "loudnorm+declick+limit")
        os.replace(tmp, wav_path)
        return True
    except Exception as e:
        log.warning(f"loudnorm+anti-screech failed, trying plain loudnorm: {e}")
        # Fallback: just loudnorm without the pre-chain
        try:
            tmp = wav_path + ".ln.wav"
            _run([
                "ffmpeg", "-y", "-i", wav_path,
                "-af", f"loudnorm=I={LN_I}:TP={LN_TP}:LRA={LN_LRA}",
                "-ar", "48000", tmp,
            ], "loudnorm-fallback")
            os.replace(tmp, wav_path)
            return True
        except Exception as e2:
            log.warning(f"loudnorm fallback also failed: {e2}")
            return False


def _atempo_stretch(wav_path: str, speed: float) -> str:
    """Time-stretch audio via ffmpeg atempo (preserves pitch, NOT np.interp which
    is pitch-shift = chipmunk effect). Returns path to stretched file."""
    if abs(speed - 1.0) < 0.02:
        return wav_path
    # atempo only accepts 0.5-2.0; chain for extremes (we cap at 1.15 anyway)
    out = wav_path + f".{speed:.2f}x.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-filter:a", f"atempo={speed:.3f}",
             out],
            check=True, capture_output=True, timeout=60,
        )
        return out
    except Exception as e:
        log.warning(f"atempo failed: {e}, using original")
        return wav_path


def assemble_dubbed_audio(segments, total_duration, output_path,
                          sample_rate=48000, apply_loudnorm=True):
    """Place each TTS segment at its original timestamp (numpy-based mix).

    Handling of overlong TTS segments (Russian/Spanish are often 20-30%
    longer than English):
    - Speed up via ffmpeg atempo (preserves pitch) up to 1.15x MAX.
    - Anything that still won't fit is allowed to spill into the next
      segment's slot — slight overlap sounds FAR better than the chipmunk
      effect from aggressive pitch-shift.
    - Total audio may exceed total_duration; caller should NOT use -shortest.
    """
    import numpy as np
    import soundfile as sf

    # Compute how long the mixed track actually needs to be based on placed segments.
    # We allow extending beyond the source video length (+ a small safety).
    est_extra = 0.0
    for seg in segments:
        ap = seg.get("audio_path")
        if ap and os.path.exists(ap):
            try:
                info = sf.info(ap)
                tts_dur = info.frames / info.samplerate
                slot_dur = seg["end"] - seg["start"]
                if tts_dur > slot_dur:
                    est_extra += (tts_dur - slot_dur)
            except Exception:
                pass
    target_duration = total_duration + min(est_extra, 15.0) + 1.0

    n_samples = int(target_duration * sample_rate)
    mix = np.zeros(n_samples, dtype=np.float32)

    valid_count = 0
    stretched_count = 0
    current_end = 0.0  # track cumulative end time to push next segments forward if needed

    for seg in segments:
        audio_path = seg.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            continue

        try:
            # Decide if we need to time-stretch (proper atempo, not pitch-shift)
            slot_dur = seg["end"] - seg["start"]
            # Quick duration read without full load
            info = sf.info(audio_path)
            tts_dur = info.frames / info.samplerate

            # Emotion-tagged segments are intentionally 10-20% longer (excited
            # speech draws out vowels, angry speech has more pauses, etc).
            # This is a *feature* not a bug — but it often pushes segments
            # out of their timing slot. When we detect an emotion tag, allow
            # a slightly more aggressive stretch to compensate.
            text = (seg.get("translated_text") or "").lstrip()
            has_emotion = text.startswith("(") and ")" in text[:30]
            max_stretch = 1.22 if has_emotion else 1.15

            stretched_path = audio_path
            if slot_dur > 0.2 and tts_dur > slot_dur * 1.05:
                # Need stretch. Cap preserves quality — beyond the cap we
                # let audio spill over into next slot (better than chipmunk).
                speed = min(tts_dur / slot_dur, max_stretch)
                if speed > 1.02:
                    stretched_path = _atempo_stretch(audio_path, speed)
                    stretched_count += 1

            data, sr = sf.read(stretched_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Resample if needed (linear interp OK here since sr should match already)
            if sr != sample_rate:
                ratio = sample_rate / sr
                new_len = int(len(data) * ratio)
                idx = np.linspace(0, len(data) - 1, new_len)
                data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)

            # Per-segment peak normalization -> consistent volume across segments
            peak = float(np.abs(data).max())
            if peak > 0.01:
                data = data * (PER_SEGMENT_PEAK / peak)

            # ─── Anti-click fades ──────────────────────────────────
            # VoxCPM outputs sometimes start/end with a tiny DC-step or an
            # abrupt waveform edge. When we mix dozens of those together the
            # boundaries produce audible clicks. A 5ms cosine fade at each
            # end eliminates those without being perceptible as a volume
            # change.
            fade_samples = int(0.005 * sample_rate)  # 5ms
            if len(data) > fade_samples * 2:
                # Use cosine-shaped fade for smoother transient than linear
                fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
                fade_out = fade_in[::-1]
                data[:fade_samples] *= fade_in
                data[-fade_samples:] *= fade_out

            # Position: use original start, but shift forward if it would overlap
            # an earlier segment that's still playing.
            start = max(seg["start"], current_end)
            offset = int(start * sample_rate)
            end = min(offset + len(data), n_samples)
            length = end - offset
            if length > 0:
                mix[offset:offset + length] += data[:length]
                valid_count += 1
                current_end = start + length / sample_rate
                # Record where this segment actually lives in the dubbed track
                # (post-stretch, post-shift). Showcase reels use these to cut
                # at real word boundaries in each language rather than at the
                # original source timestamps where words have drifted.
                seg["placed_start"] = float(start)
                seg["placed_end"] = float(current_end)

        except Exception as e:
            log.warning(f"Skipped segment: {e}")
            continue

    if stretched_count > 0:
        log.info(f"Time-stretched {stretched_count}/{valid_count} segments (pitch preserved)")

    # Trim trailing silence beyond last actual audio (keep small tail)
    if current_end > 0 and current_end + 0.5 < target_duration:
        mix = mix[:int((current_end + 0.5) * sample_rate)]

    # Prevent clipping before loudnorm
    max_val = float(np.abs(mix).max())
    if max_val > 1.0:
        mix = mix / max_val * 0.95

    sf.write(output_path, mix, sample_rate, subtype="PCM_16")
    actual_dur = len(mix) / sample_rate
    log.info(f"Assembled {valid_count} segments -> {output_path} ({actual_dur:.1f}s)")

    # Global loudness normalization so YouTube/TV playback matches broadcast levels
    if apply_loudnorm and valid_count > 0:
        _normalize_loudness_inplace(output_path)

    return output_path


def merge_audio_video(video_path, dubbed_audio_path, output_path,
                     background_audio_path="", bg_volume=0.15):
    """Merge dubbed audio with video. If dubbed audio is longer than source
    video, freeze the last video frame to match (better than cutting off
    the end of the dub).
    """
    # Check if audio is longer than video and we need to extend video
    try:
        import subprocess as _sp
        def _dur(path):
            r = _sp.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of",
                        "default=noprint_wrappers=1:nokey=1", path],
                       capture_output=True, text=True, timeout=30)
            return float(r.stdout.strip())
        v_dur = _dur(video_path)
        a_dur = _dur(dubbed_audio_path)
        need_extend = a_dur > v_dur + 0.3
    except Exception:
        need_extend = False
        v_dur = a_dur = 0

    # If audio longer -> extend video by holding last frame via tpad.
    # Otherwise use source video as-is (no -shortest; let it play to end of video).
    if need_extend:
        extend_by = a_dur - v_dur + 0.2
        log.info(f"Extending video by {extend_by:.2f}s "
                 f"(audio {a_dur:.1f}s vs video {v_dur:.1f}s)")
        tpad = f"tpad=stop_mode=clone:stop_duration={extend_by:.2f}"
    else:
        tpad = None

    if background_audio_path and os.path.exists(background_audio_path):
        if tpad:
            _run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", dubbed_audio_path,
                "-i", background_audio_path,
                "-filter_complex",
                f"[0:v]{tpad}[v];"
                f"[1:a]volume=1.0[dub];[2:a]volume={bg_volume}[bg];"
                f"[dub][bg]amix=inputs=2:duration=first:normalize=0[out]",
                "-map", "[v]", "-map", "[out]",
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ], "merge with bg + extend")
        else:
            _run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", dubbed_audio_path,
                "-i", background_audio_path,
                "-filter_complex",
                f"[1:a]volume=1.0[dub];[2:a]volume={bg_volume}[bg];"
                f"[dub][bg]amix=inputs=2:duration=first:normalize=0[out]",
                "-map", "0:v", "-map", "[out]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                output_path,
            ], "merge with bg")
    else:
        if tpad:
            _run([
                "ffmpeg", "-y",
                "-i", video_path, "-i", dubbed_audio_path,
                "-filter_complex", f"[0:v]{tpad}[v]",
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ], "merge a+v + extend")
        else:
            _run([
                "ffmpeg", "-y",
                "-i", video_path, "-i", dubbed_audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                output_path,
            ], "merge a+v")
    return output_path
