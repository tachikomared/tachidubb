"""
WhisperX-based transcriber with forced alignment and word-level timestamps.
Uses:
  - faster-whisper (via WhisperX) for fast initial transcription with VAD
  - wav2vec2 forced alignment for accurate word timings and natural pauses
  - optional integrated pyannote diarization
  - sentence grouping into TTS-friendly segments using word-level timings
"""
import logging
import os
import gc
from typing import Optional

log = logging.getLogger("tachidubb.transcriber")


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_compute_type(device: str):
    # float16 on GPU, int8 on CPU
    return "float16" if device == "cuda" else "int8"


def transcribe(
    audio_path: str,
    language: Optional[str] = None,
    model_size: str = "large-v3",
    diarize: bool = False,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    batch_size: int = 16,
):
    """
    Transcribe audio using WhisperX pipeline:
      1. Fast transcription (faster-whisper + Silero VAD)
      2. Forced alignment (wav2vec2) -> word-level timestamps
      3. Optional speaker diarization (pyannote)
      4. Group words into natural TTS segments

    Returns list of segments:
      [
        {
          "start": float,
          "end": float,
          "text": str,
          "words": [{"word": str, "start": float, "end": float}, ...],
          "speaker": str (optional),
        },
        ...
      ]
    Also returns detected language code.
    """
    import whisperx

    # whisperx requires None for auto-detection; "auto" is a UI sentinel value
    if language == "auto":
        language = None

    device = _get_device()
    compute_type = _get_compute_type(device)
    log.info(f"WhisperX: device={device}, compute_type={compute_type}, model={model_size}")

    # ---- 1. Load audio ----
    audio = whisperx.load_audio(audio_path)

    # ---- 2. Transcribe with faster-whisper backend ----
    asr_options = {
        "temperatures": [0.0],
        "initial_prompt": None,
    }
    try:
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
        )
    except Exception as e:
        # Some GPUs don't support float16 - fall back to int8_float16
        log.warning(f"float16 load failed ({e}), falling back to int8_float16")
        compute_type = "int8_float16" if device == "cuda" else "int8"
        model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
        )

    log.info("Transcribing...")
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    detected_lang = result.get("language", language or "en")
    log.info(f"Detected language: {detected_lang}")

    # Free ASR model from VRAM before loading alignment model
    del model
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- 3. Forced alignment for word-level timestamps ----
    aligned_segments = result["segments"]
    try:
        log.info(f"Loading alignment model for '{detected_lang}'...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device,
        )
        log.info("Aligning word timestamps...")
        aligned = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        aligned_segments = aligned["segments"]

        # Free alignment model
        del align_model
        gc.collect()
        if device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception as e:
        log.warning(f"Alignment failed ({e}) - continuing with coarse timestamps")

    # ---- 4. Optional diarization ----
    speaker_segments = None
    if diarize and hf_token:
        try:
            log.info("Running speaker diarization...")
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device,
            )
            diarize_segments = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            # Assign speakers to words/segments
            aligned_segments_with_speakers = whisperx.assign_word_speakers(
                diarize_segments, {"segments": aligned_segments}
            )
            aligned_segments = aligned_segments_with_speakers["segments"]
            speaker_segments = diarize_segments

            del diarize_model
            gc.collect()
            if device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Diarization failed ({e}) - continuing without speakers")
    elif diarize and not hf_token:
        log.warning("Diarization requested but HF_TOKEN not provided - skipping")

    # ---- 5. Group words into TTS-friendly segments ----
    tts_segments = _group_into_tts_segments(aligned_segments, detected_lang)

    log.info(f"Produced {len(tts_segments)} segments in '{detected_lang}'")
    return tts_segments, detected_lang


def _group_into_tts_segments(
    segments: list,
    language: str,
    max_chars: int = 160,
    max_duration: float = 12.0,
    min_duration: float = 0.6,
    pause_threshold: float = 0.6,
) -> list:
    """
    Group WhisperX word-level segments into natural TTS chunks.
    Split on:
      - sentence-ending punctuation
      - pauses longer than pause_threshold seconds
      - max_chars or max_duration exceeded
      - speaker change
    """
    # Flatten all words from all segments
    all_words = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # No word-level info - use whole segment as one "word"
            all_words.append({
                "word": seg.get("text", "").strip(),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "speaker": seg.get("speaker", ""),
            })
            continue
        for w in words:
            all_words.append({
                "word": w.get("word", "").strip(),
                "start": w.get("start", seg.get("start", 0.0)),
                "end": w.get("end", seg.get("end", 0.0)),
                "speaker": w.get("speaker", seg.get("speaker", "")),
            })

    if not all_words:
        return []

    # Ensure each word has valid timings (some words lack start/end after alignment)
    last_end = 0.0
    for w in all_words:
        if w["start"] is None or (isinstance(w["start"], float) and w["start"] != w["start"]):
            w["start"] = last_end
        if w["end"] is None or (isinstance(w["end"], float) and w["end"] != w["end"]):
            w["end"] = w["start"] + 0.1
        last_end = w["end"]

    # Sentence-ending punctuation (cross-language)
    sentence_enders = {".", "!", "?", "。", "！", "？", "…"}

    chunks = []
    current = []
    current_speaker = None

    def flush():
        if not current:
            return
        start = current[0]["start"]
        end = current[-1]["end"]
        text = " ".join(w["word"] for w in current if w["word"]).strip()
        text = text.replace("  ", " ")
        if not text:
            return
        duration = max(end - start, 0.1)
        if duration < min_duration:
            # extend end to min_duration
            end = start + min_duration
        spk = current_speaker or current[0].get("speaker", "")
        chunks.append({
            "start": float(start),
            "end": float(end),
            "text": text,
            "words": list(current),
            "speaker": spk,
        })

    for i, w in enumerate(all_words):
        if not current:
            current = [w]
            current_speaker = w.get("speaker", "")
            continue

        # Speaker changed?
        w_speaker = w.get("speaker", "")
        if current_speaker and w_speaker and w_speaker != current_speaker:
            flush()
            current = [w]
            current_speaker = w_speaker
            continue

        # Pause since previous word?
        gap = w["start"] - current[-1]["end"]

        current.append(w)
        tentative_text = " ".join(x["word"] for x in current)
        tentative_dur = current[-1]["end"] - current[0]["start"]

        # Check whether to split AFTER this word
        prev_word_text = w["word"].rstrip()
        ends_sentence = prev_word_text and prev_word_text[-1] in sentence_enders

        should_split = False
        if ends_sentence and tentative_dur >= min_duration:
            should_split = True
        elif gap > pause_threshold and tentative_dur >= min_duration:
            should_split = True
        elif len(tentative_text) >= max_chars:
            should_split = True
        elif tentative_dur >= max_duration:
            should_split = True

        if should_split:
            flush()
            current = []
            current_speaker = None

    flush()

    return chunks


# Keep a legacy single-return API for backward compat where server calls it
# expecting just a list of segments.
def transcribe_simple(audio_path: str, **kwargs):
    segments, _lang = transcribe(audio_path, **kwargs)
    return segments
