"""VoxCPM subprocess worker.

Why this exists
───────────────
When VoxCPM's `model.generate()` is invoked with `reference_wav_path` AFTER
WhisperX/pyannote have already loaded `speechbrain` in the same process,
speechbrain's LazyModule for `speechbrain.integrations.k2_fsa` fails on
Windows (k2 has no Windows wheels). We can't fix this in-process — the
standalone `diagnose_voxcpm.py` script proves VoxCPM itself works perfectly
when speechbrain hasn't been pre-loaded.

Solution: run all TTS in a subprocess that never imports WhisperX/pyannote.
The parent process passes a JSON job file; the worker writes WAVs.

Usage (from parent):
    python tts_worker.py <job_json_path>

Where job_json_path points to a file like:
{
  "model_id": "openbmb/VoxCPM2",
  "cfg_value": 2.0,
  "inference_timesteps": 10,
  "voice_seed": 404,
  "segments": [
    {
      "idx": 0,
      "text": "Привет",
      "output_path": "C:/...seg_0000.wav",
      "reference_wav_path": "C:/.../ref.wav",
      "prompt_wav_path": "C:/.../ref.wav",
      "prompt_text": "The reference transcript."
    },
    ...
  ]
}

The worker prints one JSON line per segment to stdout so the parent can
display progress:
    {"event":"segment","idx":0,"ok":true,"tier":1}
    {"event":"segment","idx":1,"ok":true,"tier":1}
    ...
    {"event":"done","ok":7,"total":9}
"""
import json
import os
import random as _random
import sys
import time
import traceback

# ═══════════════════════════════════════════════════════════════════
# CRITICAL: suppress everything that writes to stderr BEFORE importing
# VoxCPM. Otherwise tqdm/warnings fill the stderr pipe (8KB on Windows),
# the subprocess blocks on write, parent never drains it, DEADLOCK.
# ═══════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Instead of mocking tqdm (which breaks huggingface_hub's thread_map
# that uses tqdm.get_lock/get_instances), just patch tqdm to default
# to disable=True. The real API stays intact.
try:
    import tqdm
    _orig_tqdm_init = tqdm.tqdm.__init__

    def _silent_init(self, *args, **kwargs):
        kwargs.setdefault("disable", True)
        kwargs.setdefault("leave", False)
        return _orig_tqdm_init(self, *args, **kwargs)

    tqdm.tqdm.__init__ = _silent_init

    # Also patch tqdm.auto.tqdm the same way
    try:
        import tqdm.auto
        if tqdm.auto.tqdm is not tqdm.tqdm:
            _orig_auto_init = tqdm.auto.tqdm.__init__

            def _silent_auto_init(self, *args, **kwargs):
                kwargs.setdefault("disable", True)
                kwargs.setdefault("leave", False)
                return _orig_auto_init(self, *args, **kwargs)

            tqdm.auto.tqdm.__init__ = _silent_auto_init
    except Exception:
        pass
except Exception:
    pass


def _log_event(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def _set_seed(seed):
    if seed is None:
        return
    try:
        import numpy as np
        import torch
        s = int(seed) & 0x7FFFFFFF
        _random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        # Force cuDNN to use deterministic algorithms so every segment
        # starts from the SAME noise sample → consistent voice identity
        # across all segments of the same dub. Without this, cuDNN uses
        # non-deterministic atomic ops even when the seed is identical,
        # making each segment sound like a slightly different speaker.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _synth_one(model, seg, base_kwargs, voice_seed, tier_policy="balanced",
               target_lang: str = "ru", enable_qa: bool = True):
    """Try selected tiers in order with optional post-synth QA.

    If QA is enabled and the generated audio fails quality check (CER too
    high, wrong language detected, etc.) we regenerate with a fresh seed
    up to 2 more times. This catches VoxCPM's occasional gibberish output
    that its own retry_badcase doesn't detect.

    Returns (ok, tier_used, err). On success sets seg['qa_score'] and
    seg['qa_transcript'] for diagnostic logging.
    """
    import soundfile as sf

    ref = seg.get("reference_wav_path") or ""
    prompt = seg.get("prompt_wav_path") or ""
    prompt_text = seg.get("prompt_text") or ""
    text = seg["text"]
    out_path = seg["output_path"]

    # Voice Design mode detection: no reference audio AND text starts
    # with a "(style description)" prefix. In this mode every retry with
    # a different seed produces a DIFFERENT voice timbre (not just a
    # different take of the same voice) — which causes the final dub to
    # sound like multiple different speakers stitched together. So we
    # DISABLE seed-mutation retries for voice_design: take the first
    # attempt as-is, even if QA is marginal. Consistency > single-segment
    # quality when there's no reference pinning the timbre.
    has_ref = bool(ref and os.path.exists(ref))
    voice_design_mode = (not has_ref) and text.lstrip().startswith("(")

    # Tier 1 — full cloning (prompt + ref + prompt_text) -- SLOW + sometimes retries
    tier1 = dict(base_kwargs, text=text)
    if prompt and os.path.exists(prompt):
        tier1["prompt_wav_path"] = prompt
        if prompt_text:
            tier1["prompt_text"] = prompt_text
        tier1["reference_wav_path"] = prompt
    elif ref and os.path.exists(ref):
        tier1["reference_wav_path"] = ref

    # Tier 2 — reference ONLY (no prompt_text — that triggers retry_badcase
    # which can blow up to 30+ seconds per segment). Reference alone clones
    # the voice well enough at this speed tier.
    tier2 = dict(base_kwargs, text=text)
    if ref and os.path.exists(ref):
        tier2["reference_wav_path"] = ref
    elif prompt and os.path.exists(prompt):
        tier2["reference_wav_path"] = prompt

    # Tier 3 — pure voice design (fastest, no cloning)
    tier3 = dict(base_kwargs, text=text)

    has_real_prompt = bool(prompt_text and prompt and os.path.exists(prompt))

    # Voice Design mode: no reference at all. Using tier1 here is wasted
    # compute (no prompt to work with) AND sometimes produces subtly
    # different voices than tier3 with the same seed because cfg/timestep
    # values differ between tiers — breaks consistency across segments.
    # Force pure tier3 path so EVERY segment uses identical params.
    if voice_design_mode:
        tiers = [(3, tier3)]
    elif tier_policy == "quality":
        tiers = [(1, tier1), (2, tier2), (3, tier3)]
    elif tier_policy == "balanced":
        if has_real_prompt:
            tiers = [(1, tier1), (2, tier2), (3, tier3)]
        else:
            tiers = [(2, tier2), (3, tier3)]
    elif tier_policy == "fast":
        tiers = [(3, tier3)]
    else:
        tiers = [(2, tier2), (3, tier3)]

    def _try_generate(kw, seed_for_this):
        """Single generate attempt with given kwargs and seed."""
        # In voice_design mode, we ALWAYS use voice_seed (not seed_for_this)
        # because timbre consistency across segments trumps per-segment
        # variation. Even if seed_for_this differs (from QA retry), we
        # override to keep the voice identical across all segments.
        if voice_design_mode:
            seed_for_this = voice_seed
        _set_seed(seed_for_this)
        dbg = {k: (os.path.basename(v) if isinstance(v, str) and os.path.isfile(v) else v)
               for k, v in kw.items() if k != "text"}
        _log_event(event="tier_attempt", idx=seg["idx"], tier=_current_tier,
                   args=dbg, seed=seed_for_this)
        wav = model.generate(**kw)
        if wav is None or len(wav) == 0:
            raise RuntimeError("empty audio")
        sf.write(out_path, wav, model.tts_model.sample_rate)

    # Max QA-retries: regenerate with fresh seed if quality score too high.
    # Fast mode: no QA (would be slower than just accepting output).
    MAX_QA_RETRIES = 0 if tier_policy == "fast" else 2
    # In Voice Design mode (no reference audio) different seeds yield
    # different voice timbres. To keep ONE consistent speaker across the
    # whole dub, we skip seed-mutating retries here — even if QA score
    # is marginal. User can always per-segment regen a specific bad one
    # from the completion UI if needed.
    if voice_design_mode:
        MAX_QA_RETRIES = 0
    # In cloning mode (reference audio provided), QA retries with mutated
    # seeds produce different voice timbres even with the same reference —
    # especially in cross-lingual dubbing where the model is under higher
    # generative pressure. Concretely: segments that pass QA on attempt 1
    # use voice_seed; segments that fail and get a retry use
    # voice_seed+1001, voice_seed+2001 etc — an audibly different voice.
    # Result: dub sounds like two or more different speakers.
    # Fix: disable seed-mutation retries in cloning mode. If QA fails,
    # fall through to the NEXT TIER (which resets to voice_seed) rather
    # than accepting a different-voiced retry on the same tier.
    if has_ref and not voice_design_mode:
        MAX_QA_RETRIES = 0

    last_err = None
    best_score = 2.0  # worse than any real score
    best_transcript = ""
    for idx, kw in tiers:
        _current_tier = idx  # closure-visible for _try_generate's log
        try:
            _try_generate(kw, voice_seed)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue

        # Post-synth QA. If acceptable, return. If bad, retry with fresh seed.
        if not enable_qa:
            return (True, idx, None)

        try:
            from pipeline.tts_qa import check_segment_quality, is_acceptable
        except ImportError:
            try:
                from tts_qa import check_segment_quality, is_acceptable
            except ImportError:
                # Module not deployed — skip QA gracefully
                return (True, idx, None)

        score, transcript, diag = check_segment_quality(
            out_path, text, target_lang=target_lang,
        )
        if score < best_score:
            best_score = score
            best_transcript = transcript
        _log_event(event="qa_check", idx=seg["idx"], tier=idx,
                   score=round(score, 3), cer=diag.get("cer"),
                   detected_lang=diag.get("detected_lang"),
                   transcript_preview=transcript[:60])

        if is_acceptable(score):
            seg["qa_score"] = score
            seg["qa_transcript"] = transcript
            return (True, idx, None)

        # Bad score — retry with different seeds on same tier before
        # falling through to next tier
        for retry_n in range(MAX_QA_RETRIES):
            new_seed = (voice_seed or 0) + 1000 * (retry_n + 1) + idx
            _log_event(event="qa_retry", idx=seg["idx"], tier=idx,
                       attempt=retry_n + 1, new_seed=new_seed,
                       prev_score=round(score, 3))
            try:
                _try_generate(kw, new_seed)
                score, transcript, diag = check_segment_quality(
                    out_path, text, target_lang=target_lang,
                )
                if score < best_score:
                    best_score = score
                    best_transcript = transcript
                if is_acceptable(score):
                    seg["qa_score"] = score
                    seg["qa_transcript"] = transcript
                    return (True, idx, None)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

    # All tiers + retries exhausted; if we produced *something*, accept it
    # with a warning. The alternative would be silence which is worse.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        seg["qa_score"] = best_score
        seg["qa_transcript"] = best_transcript
        _log_event(event="qa_fallback", idx=seg["idx"],
                   score=round(best_score, 3),
                   msg="all retries failed QA, using last output")
        # Use the tier from the final attempt; log as tier 0 to mark "degraded"
        return (True, 0, None)

    return (False, -1, last_err)


def _load_model(job):
    """Load VoxCPM once per process (expensive, ~15-35s)."""
    from voxcpm import VoxCPM
    tier_policy = job.get("tier_policy", "balanced")
    use_denoiser = (tier_policy == "quality")
    _log_event(event="loading", model=job.get("model_id", "openbmb/VoxCPM2"),
               denoiser=use_denoiser, tier_policy=tier_policy)
    t0 = time.time()
    model = VoxCPM.from_pretrained(
        job.get("model_id", "openbmb/VoxCPM2"),
        load_denoiser=use_denoiser,
    )
    _log_event(event="loaded", seconds=round(time.time() - t0, 1),
               sample_rate=model.tts_model.sample_rate)
    return model


def _process_job(model, job):
    """Run TTS for all segments in a single job spec (already loaded model)."""
    base_kwargs = {
        "cfg_value": job.get("cfg_value", 2.0),
        "inference_timesteps": job.get("inference_timesteps", 10),
        "retry_badcase": job.get("retry_badcase", True),
        "retry_badcase_max_times": job.get("retry_badcase_max_times", 2),
    }
    voice_seed = job.get("voice_seed")
    tier_policy = job.get("tier_policy", "balanced")
    target_lang = job.get("target_lang", "ru")
    # Enable Whisper QA only for cross-lingual or when explicitly requested.
    # Same-language cloning very rarely produces gibberish, so skipping QA
    # there saves ~1-2s per segment of ASR overhead.
    enable_qa = bool(
        job.get("enable_qa", job.get("is_cross_lingual", False))
    )
    segments = job["segments"]
    total = len(segments)
    ok = 0
    qa_regens = 0  # count of segments that needed QA-triggered regen
    tier_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    if enable_qa:
        _log_event(event="qa_enabled", target_lang=target_lang)

    for seg in segments:
        try:
            _log_event(event="segment_start", idx=seg["idx"],
                       text_preview=seg["text"][:40])
            success, tier, err = _synth_one(
                model, seg, base_kwargs, voice_seed,
                tier_policy=tier_policy,
                target_lang=target_lang,
                enable_qa=enable_qa,
            )
            if success:
                ok += 1
                tier_stats[tier] = tier_stats.get(tier, 0) + 1
                if tier == 0:
                    qa_regens += 1  # degraded = QA couldn't recover
                _log_event(event="segment", idx=seg["idx"], ok=True, tier=tier,
                           text_preview=seg["text"][:40],
                           qa_score=seg.get("qa_score"))
            else:
                _log_event(event="segment", idx=seg["idx"], ok=False, error=err,
                           text_preview=seg["text"][:40])
        except Exception as e:
            _log_event(event="segment", idx=seg["idx"], ok=False,
                       error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())

    _log_event(event="done", ok=ok, total=total, tier_stats=tier_stats,
               qa_regens=qa_regens)


def main(job_path):
    """Single-shot mode: load model, process one job, exit."""
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    model = _load_model(job)
    _process_job(model, job)


def main_daemon():
    """Daemon mode: load model ONCE, then wait on stdin for job paths.
    Parent writes a line `<path_to_job.json>\\n` to our stdin; we process
    the job and print `{"event": "job_done"}` when finished. Exits on EOF
    or on the special line `SHUTDOWN`. First job path comes from argv[2]
    (same as single-shot mode, so first job bootstraps the model load)."""
    # First job bootstraps model + processes itself
    first_job_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not first_job_path or not os.path.exists(first_job_path):
        _log_event(event="fatal", error="daemon mode requires first job path as argv[2]")
        return
    with open(first_job_path, "r", encoding="utf-8") as f:
        first_job = json.load(f)
    model = _load_model(first_job)
    _process_job(model, first_job)
    _log_event(event="job_done")

    # Subsequent jobs arrive via stdin lines (one path per line)
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line == "SHUTDOWN":
            break
        if not os.path.exists(line):
            _log_event(event="job_error", error=f"job file not found: {line}")
            continue
        try:
            with open(line, "r", encoding="utf-8") as f:
                job = json.load(f)
            _process_job(model, job)
            _log_event(event="job_done")
        except Exception as e:
            _log_event(event="job_error", error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"event": "fatal",
                          "error": "usage: tts_worker.py [--daemon] <job.json>"}))
        sys.exit(2)
    try:
        if sys.argv[1] == "--daemon":
            main_daemon()
        else:
            main(sys.argv[1])
    except Exception as e:
        _log_event(event="fatal", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc())
        sys.exit(1)
