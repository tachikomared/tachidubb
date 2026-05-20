"""
Standalone VoxCPM diagnostic — runs generate() with reference_wav_path
and prints the full exception traceback, so we can see WHAT actually
fails in Tier 1 (the server swallows it).

Run with:  python diagnose_voxcpm.py
"""
import glob
import os
import sys
import traceback
import warnings


def main():
    print("=" * 70)
    print("VoxCPM Tier-1 diagnostic")
    print("=" * 70)

    # Show ALL warnings verbatim with full source
    warnings.simplefilter("always")

    # Find any existing reference audio from past jobs.
    # Looks in <project-root>/outputs/*/speaker_refs/*.wav relative to this file.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = sorted(
        glob.glob(os.path.join(_project_root, "outputs", "*", "speaker_refs", "*.wav")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        print("ERROR: no reference WAV found under outputs/*/speaker_refs/")
        print("Run at least one dubbing job first so a fallback ref is generated.")
        sys.exit(1)
    ref = candidates[0]
    print(f"Using reference: {ref}")
    print(f"Size: {os.path.getsize(ref)} bytes")
    print()

    # Import and load
    print("Loading VoxCPM...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print("Loaded.\n")

    # Test 1: voice design (should work)
    print("-" * 70)
    print("TEST 1: voice design (no reference) -- expected to PASS")
    print("-" * 70)
    try:
        wav = model.generate(
            text="(adult male voice)This is a voice design test.",
            cfg_value=2.0, inference_timesteps=10,
        )
        print(f"PASS -- got {len(wav)} samples")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

    # Test 2: reference-only cloning (Tier 2 in our pipeline)
    print("-" * 70)
    print("TEST 2: reference_wav_path only (Tier 2) -- checking")
    print("-" * 70)
    try:
        wav = model.generate(
            text="This is a reference cloning test.",
            reference_wav_path=ref,
            cfg_value=2.0, inference_timesteps=10,
        )
        print(f"PASS -- got {len(wav)} samples")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

    # Test 3: ultimate cloning with prompt_text (Tier 1 in our pipeline)
    print("-" * 70)
    print("TEST 3: prompt_wav_path + prompt_text + reference (Tier 1)")
    print("-" * 70)
    try:
        wav = model.generate(
            text="This is an ultimate cloning test.",
            prompt_wav_path=ref,
            prompt_text="This is some sample speech used as reference.",
            reference_wav_path=ref,
            cfg_value=2.0, inference_timesteps=10,
        )
        print(f"PASS -- got {len(wav)} samples")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

    print("=" * 70)
    print("Diagnostic complete. Copy ALL output above and send it.")
    print("=" * 70)


if __name__ == "__main__":
    main()
