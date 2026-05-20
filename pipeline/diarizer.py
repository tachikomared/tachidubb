"""Speaker diarization with pyannote.audio.

Returns speaker turns [(start, end, speaker_id)] and per-speaker reference audio.
Compatible with pyannote.audio 3.x and 4.x (API differences auto-handled).
"""
import logging
import os

import numpy as np
import soundfile as sf

log = logging.getLogger("tachidubb.diarizer")

HF_TOKEN = os.getenv("HF_TOKEN", "")

REFERENCE_TARGET = 30.0   # aim for ~30 seconds of clean speech per speaker
REFERENCE_MIN = 6.0
MIN_CHUNK = 1.5
MAX_CHUNK = 12.0


def _load_pipeline(token: str):
    """Load pyannote diarization pipeline. Uses the new token= kwarg
    (pyannote.audio >= 3.0). `use_auth_token` was removed in 4.x."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.warning("pyannote.audio not installed - diarization disabled")
        return None

    if not token:
        log.warning(
            "HF_TOKEN is empty. Diarization needs a Hugging Face token. "
            "Set it in .env as HF_TOKEN=hf_xxx and install python-dotenv."
        )
        return None

    models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/speaker-diarization-community-1",
    ]
    errors = []
    for model in models:
        try:
            pipe = Pipeline.from_pretrained(model, token=token)
            if pipe is not None:
                log.info(f"Diarization pipeline loaded: {model}")
                return pipe
        except Exception as e:
            errors.append(f"  {model}: {type(e).__name__}: {e}")
            continue

    log.warning(
        "Could not load any diarization pipeline. Check that:\n"
        "  1) Your HF_TOKEN is valid: https://huggingface.co/settings/tokens\n"
        "  2) You accepted the model terms at:\n"
        "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "     - https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        "Attempt errors:\n" + "\n".join(errors)
    )
    return None


def diarize_speakers(audio_path: str,
                     min_speakers: int | None = None,
                     max_speakers: int | None = None,
                     hf_token: str = "") -> list[tuple]:
    """Run diarization. Returns list of (start, end, speaker) tuples, or [] on failure."""
    token = hf_token or HF_TOKEN
    if not token:
        log.warning("HF_TOKEN not set - diarization skipped (set it in .env)")
        return []

    pipe = _load_pipeline(token)
    if pipe is None:
        return []

    # Move to GPU if available
    try:
        import torch as _t
        if _t.cuda.is_available():
            pipe.to(_t.device("cuda"))
    except Exception:
        pass

    try:
        import torch
        # Try preloading audio as a tensor to bypass broken torchcodec on Windows.
        # pyannote accepts {"waveform": tensor, "sample_rate": int} per
        # https://huggingface.co/pyannote/speaker-diarization-3.1
        try:
            import soundfile as sf
            import numpy as np
            wav, sr = sf.read(audio_path, always_2d=False)
            if wav.ndim == 1:
                wav = wav[np.newaxis, :]  # (1, time)
            elif wav.ndim == 2:
                wav = wav.T  # -> (channels, time)
            waveform = torch.from_numpy(wav.astype(np.float32))
            audio_input = {"waveform": waveform, "sample_rate": sr}
        except Exception as e:
            log.debug(f"Could not preload audio via soundfile: {e}; using path")
            audio_input = audio_path

        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        diarization = pipe(audio_input, **kwargs)

        # pyannote 4.x: returns DiarizeOutput (has .speaker_diarization attribute
        # which is the classic Annotation). 3.x returned Annotation directly.
        if hasattr(diarization, "speaker_diarization"):
            ann = diarization.speaker_diarization
        elif hasattr(diarization, "itertracks"):
            ann = diarization
        else:
            log.warning(f"Unknown diarization output type: {type(diarization)}")
            return []

        turns = [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in ann.itertracks(yield_label=True)
        ]
        log.info(f"Diarization: {len(turns)} turns, "
                 f"{len(set(s for _,_,s in turns))} speakers")
        return turns
    except Exception as e:
        log.warning(f"Diarization inference failed: {e}")
        return []

    pipe = _load_pipeline(token)
    if pipe is None:
        return []

def assign_speakers_to_segments(segments: list[dict], speaker_turns: list[tuple]) -> list[dict]:
    """Attach a 'speaker' field to each transcript segment by max temporal overlap."""
    if not speaker_turns:
        for s in segments:
            s.setdefault("speaker", "SPEAKER_00")
        return segments

    for seg in segments:
        s, e = seg["start"], seg["end"]
        best_overlap = 0.0
        best_spk = "SPEAKER_00"
        for ts, te, spk in speaker_turns:
            ov = max(0.0, min(e, te) - max(s, ts))
            if ov > best_overlap:
                best_overlap = ov
                best_spk = spk
        seg["speaker"] = best_spk
    return segments


def _total_duration(speaker_turns, spk) -> float:
    return sum(te - ts for ts, te, s in speaker_turns if s == spk)


def extract_speaker_audio(audio_path: str,
                          speaker_turns: list[tuple],
                          out_dir: str,
                          main_only: bool = False,
                          max_speakers: int | None = None) -> dict:
    """Build per-speaker reference WAVs by concatenating the cleanest chunks up to ~30s.

    - main_only=True  : keep only the speaker with the most total speech.
    - max_speakers=N  : keep only the top N speakers by total speech duration.

    Returns {speaker_id: path_to_wav}.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not speaker_turns:
        return {}

    # Group by speaker
    per_speaker: dict = {}
    for ts, te, spk in speaker_turns:
        per_speaker.setdefault(spk, []).append((ts, te))

    # Sort speakers by total duration desc
    speakers_sorted = sorted(per_speaker.keys(),
                             key=lambda k: -_total_duration(speaker_turns, k))
    if main_only and speakers_sorted:
        speakers_sorted = speakers_sorted[:1]
    elif max_speakers is not None and max_speakers > 0:
        speakers_sorted = speakers_sorted[:max_speakers]

    try:
        wav, sr = sf.read(audio_path)
    except Exception as e:
        log.warning(f"Could not read audio for references: {e}")
        return {}
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    refs = {}
    for spk in speakers_sorted:
        turns = sorted(per_speaker[spk], key=lambda x: -(x[1] - x[0]))
        selected = []
        total = 0.0
        seen = set()

        # Pass 1: durations in the sweet spot (1.5-12s, cleanest)
        for ts, te in turns:
            d = te - ts
            if MIN_CHUNK <= d <= MAX_CHUNK:
                selected.append((ts, te))
                seen.add((ts, te))
                total += d
                if total >= REFERENCE_TARGET:
                    break

        # Pass 2: short turns fill gaps
        if total < REFERENCE_MIN:
            for ts, te in turns:
                d = te - ts
                if 0.4 <= d < MIN_CHUNK and (ts, te) not in seen:
                    selected.append((ts, te))
                    seen.add((ts, te))
                    total += d
                    if total >= REFERENCE_TARGET:
                        break

        # Pass 3: long turns (capped) if still under-target
        if total < REFERENCE_MIN:
            for ts, te in turns:
                d = te - ts
                if d > MAX_CHUNK and (ts, te) not in seen:
                    capped_end = ts + min(d, 15.0)
                    selected.append((ts, capped_end))
                    seen.add((ts, te))
                    total += capped_end - ts
                    if total >= REFERENCE_TARGET:
                        break

        if not selected:
            log.warning(f"Speaker {spk}: no usable speech found")
            continue

        # Chronological order for natural-sounding reference
        selected.sort(key=lambda x: x[0])
        pieces = []
        for ts, te in selected:
            a = int(ts * sr)
            b = int(te * sr)
            if b > a and b <= len(wav):
                pieces.append(wav[a:b])

        if not pieces:
            continue

        merged = np.concatenate(pieces).astype(np.float32)
        # Peak-normalize so reference isn't clipped or too quiet
        peak = float(np.abs(merged).max())
        if peak > 0.01:
            merged = merged * (0.9 / peak)

        ref_path = os.path.join(out_dir, f"ref_{spk}.wav")
        sf.write(ref_path, merged, sr)
        refs[spk] = ref_path
        log.info(f"Speaker {spk}: {total:.1f}s reference -> {os.path.basename(ref_path)}")

    return refs


def extract_fallback_reference(audio_path: str,
                               segments: list[dict],
                               out_path: str,
                               duration: float = 30.0) -> str | None:
    """When diarization failed: build a single-speaker reference from the longest clean
    transcript segments. Use this for the main (single) speaker in the video.
    """
    if not segments:
        return None

    try:
        wav, sr = sf.read(audio_path)
    except Exception as e:
        log.warning(f"Could not read audio for fallback ref: {e}")
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Prefer medium-length segments (typical clear speech)
    sorted_segs = sorted(
        segments,
        key=lambda s: -(s.get("end", 0) - s.get("start", 0)),
    )

    selected = []
    total = 0.0
    # Pass 1: sweet spot
    for seg in sorted_segs:
        d = seg.get("end", 0) - seg.get("start", 0)
        text = (seg.get("text") or "").strip()
        if not text or d < MIN_CHUNK or d > MAX_CHUNK:
            continue
        selected.append((seg["start"], seg["end"]))
        total += d
        if total >= duration:
            break
    # Pass 2: any reasonable segment
    if total < REFERENCE_MIN:
        for seg in sorted_segs:
            d = seg.get("end", 0) - seg.get("start", 0)
            text = (seg.get("text") or "").strip()
            if not text or d < 0.5:
                continue
            pair = (seg["start"], seg["end"])
            if pair in selected:
                continue
            selected.append(pair)
            total += d
            if total >= duration:
                break

    if not selected:
        # Last resort: first `duration` seconds of audio
        nfr = min(len(wav), int(duration * sr))
        sf.write(out_path, wav[:nfr], sr)
        log.warning(f"Fallback reference: using first {nfr/sr:.1f}s of source audio")
        return out_path

    selected.sort(key=lambda x: x[0])
    pieces = []
    for ts, te in selected:
        a = int(ts * sr)
        b = int(te * sr)
        if b > a and b <= len(wav):
            pieces.append(wav[a:b])

    if not pieces:
        return None

    merged = np.concatenate(pieces).astype(np.float32)
    peak = float(np.abs(merged).max())
    if peak > 0.01:
        merged = merged * (0.9 / peak)
    sf.write(out_path, merged, sr)
    log.info(f"Fallback single-speaker reference: {total:.1f}s of clean speech")
    return out_path
