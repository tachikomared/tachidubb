"""
Segment post-processing — cleanup of WhisperX + pyannote output.
================================================================

The raw output from WhisperX + pyannote has three systemic problems for
dubbing use:

1. **Sentence fragmentation**: WhisperX cuts on VAD (breath) boundaries,
   not sentence boundaries. A natural phrase like "who's the most spaz?
   Nicky" often becomes TWO segments split at the pause before "Nicky"
   — the TTS model then tries to clone a voice from 13 characters of
   context, which produces garbled output.

2. **Speaker-turn noise**: pyannote's segmentation model has ~200ms
   resolution and occasionally emits spurious speaker flips (one frame
   of wrong label). Those create micro-segments (<1s) that nobody
   intended and which TTS cannot handle well.

3. **Long monologue segments**: rare but harmful — when WhisperX misses
   a VAD break it can output 20-30s single segments that overflow the
   timing window and force aggressive time-stretching in the assembler.

This module runs three passes over the segment list to fix all of these:
`merge_continuation_segments` → `absorb_micro_segments` → `split_very_long`.
Apply it AFTER speaker assignment but BEFORE translation.
"""

import logging
import re
from typing import Dict, List

log = logging.getLogger("tachidubb.segment_post")

# Sentence-final punctuation that indicates a complete thought.
# We treat `…` (ellipsis) and `,` as "still going" to merge aggressively.
_SENTENCE_END_RE = re.compile(r'[.!?]["\']?\s*$')
# For splitting, we need a boundary followed by whitespace or string end.
_STRONG_BREAK_RE = re.compile(r'[.!?]["\']?\s+')


def postprocess_segments(
    segments: List[Dict],
    merge_gap: float = 0.5,
    max_merge_duration: float = 12.0,
    max_merge_chars: int = 240,
    split_threshold: float = 15.0,
    micro_duration: float = 1.0,
    micro_chars: int = 40,
) -> List[Dict]:
    """Run the full cleanup chain. Returns a new list (input not mutated).

    Defaults are tuned for TTS dubbing: we want segments between ~2s and
    ~12s, ideally ending on punctuation. Knobs can be tweaked per-job if
    the user picks a different TTS speed tier.
    """
    if not segments:
        return segments
    original_count = len(segments)

    # Pass 1 — join unfinished-sentence continuations into their neighbors.
    # This is by far the highest-impact pass; most "kasha" comes from
    # sentences chopped at pauses.
    out = _merge_continuation_segments(
        segments, merge_gap, max_merge_duration, max_merge_chars
    )

    # Pass 2 — absorb orphan <1s bits that couldn't merge in pass 1 but
    # are too short for TTS anyway. These usually come from spurious
    # speaker-turn flips in pyannote.
    out = _absorb_micro_segments(out, micro_duration, micro_chars)

    # Pass 3 — split monsters that would otherwise force brutal time
    # compression in the assembler. Only splits if we can find a clean
    # sentence boundary; otherwise leaves as-is.
    out = _split_very_long_segments(out, split_threshold)

    if len(out) != original_count:
        log.info(
            f"[post] Segment count: {original_count} → {len(out)} "
            f"(cleaner segments, better TTS quality)"
        )
    return out


def _merge_continuation_segments(
    segments: List[Dict], merge_gap: float, max_dur: float, max_chars: int
) -> List[Dict]:
    """Merge adjacent segments that look like one continuing sentence.

    Criteria (ALL must be true to merge):
      - Same speaker (or speaker info missing on both)
      - Gap between them ≤ merge_gap seconds
      - Previous segment doesn't end in `.!?` — OR gap is very tight (<0.2s)
        which indicates Whisper just chopped mid-phrase
      - Combined segment stays under duration/chars limits
    """
    if len(segments) < 2:
        return [dict(s) for s in segments]

    out = [dict(segments[0])]
    joined = 0
    for curr in segments[1:]:
        prev = out[-1]

        # Speaker match: identical IDs, OR either side missing speaker info
        prev_spk = prev.get("speaker")
        curr_spk = curr.get("speaker")
        same_speaker = (
            (prev_spk == curr_spk) or not prev_spk or not curr_spk
        )

        gap = curr["start"] - prev["end"]
        prev_text = (prev.get("text") or "").rstrip()
        prev_unfinished = not _SENTENCE_END_RE.search(prev_text)

        combined_dur = curr["end"] - prev["start"]
        combined_chars = (
            len(prev_text) + 1 + len((curr.get("text") or "").strip())
        )

        should_merge = (
            same_speaker
            and gap <= merge_gap
            and (prev_unfinished or gap < 0.2)
            and combined_dur <= max_dur
            and combined_chars <= max_chars
        )

        if should_merge:
            prev["end"] = curr["end"]
            prev["text"] = (
                prev_text + " " + (curr.get("text") or "").lstrip()
            ).strip()
            # Preserve word-level alignment (from WhisperX) if both have it
            if "words" in prev and "words" in curr:
                prev["words"] = prev["words"] + curr["words"]
            joined += 1
        else:
            out.append(dict(curr))

    if joined > 0:
        log.info(
            f"[post] Pass 1: joined {joined} continuation segments "
            f"(incomplete sentences merged with next)"
        )
    return out


def _absorb_micro_segments(
    segments: List[Dict], threshold_sec: float, threshold_chars: int
) -> List[Dict]:
    """Absorb remaining tiny segments into their neighbors.

    These are usually orphan bits (~13 chars, <1s) left over after pass 1
    because they had a sentence-ending marker somewhere. TTS hates them
    — too little context to clone voice, too short to even finish a
    diffusion step without padding. Better to merge into prev or next.
    """
    if len(segments) < 2:
        return [dict(s) for s in segments]

    out = []
    absorbed = 0
    i = 0
    while i < len(segments):
        seg = segments[i]
        dur = seg["end"] - seg["start"]
        text = (seg.get("text") or "").strip()

        is_micro = dur < threshold_sec and len(text) < threshold_chars

        if is_micro and out:
            # Try previous first (usually same speaker, continues naturally)
            prev = out[-1]
            prev_gap = seg["start"] - prev["end"]
            prev_same_spk = (
                prev.get("speaker") == seg.get("speaker")
                or not prev.get("speaker") or not seg.get("speaker")
            )
            if prev_same_spk and prev_gap < 1.5:
                prev["end"] = seg["end"]
                prev["text"] = (
                    (prev.get("text") or "").rstrip() + " " + text
                ).strip()
                if "words" in prev and "words" in seg:
                    prev["words"] = prev["words"] + seg["words"]
                absorbed += 1
                i += 1
                continue

        # Also consider absorbing into NEXT if prev didn't work
        if is_micro and i + 1 < len(segments):
            nxt = segments[i + 1]
            next_gap = nxt["start"] - seg["end"]
            next_same_spk = (
                nxt.get("speaker") == seg.get("speaker")
                or not nxt.get("speaker") or not seg.get("speaker")
            )
            if next_same_spk and next_gap < 1.5:
                # Build merged next in-place; don't advance, let loop pick it
                merged = dict(nxt)
                merged["start"] = seg["start"]
                merged["text"] = (
                    text + " " + (nxt.get("text") or "").lstrip()
                ).strip()
                if "words" in seg and "words" in nxt:
                    merged["words"] = seg["words"] + nxt["words"]
                segments = segments[: i + 1] + [merged] + segments[i + 2:]
                segments[i] = merged  # replace current with merged
                absorbed += 1
                i += 1
                continue

        out.append(dict(seg))
        i += 1

    if absorbed > 0:
        log.info(
            f"[post] Pass 2: absorbed {absorbed} micro-segments "
            f"(<{threshold_sec}s and <{threshold_chars} chars)"
        )
    return out


def _split_very_long_segments(
    segments: List[Dict], threshold: float
) -> List[Dict]:
    """Split segments >threshold seconds at sentence boundaries.

    Only splits if we can find a `.!?` break somewhere in the middle
    (not too close to either edge). Otherwise leaves alone — better one
    long segment than two weird halves.
    """
    out = []
    split_count = 0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        text = seg.get("text") or ""

        if dur <= threshold or len(text) < 40:
            out.append(dict(seg))
            continue

        # Find candidate split points (positions after .!? followed by space)
        boundaries = [
            m.end() for m in _STRONG_BREAK_RE.finditer(text)
            # Don't split too close to either edge
            if 20 < m.end() < len(text) - 20
        ]
        if not boundaries:
            out.append(dict(seg))
            continue

        # Pick the boundary closest to the middle for balanced halves
        middle = len(text) / 2
        best = min(boundaries, key=lambda b: abs(b - middle))
        frac = best / len(text)
        split_time = seg["start"] + dur * frac

        first = dict(seg)
        first["end"] = split_time
        first["text"] = text[:best].strip()

        second = dict(seg)
        second["start"] = split_time
        second["text"] = text[best:].strip()

        # Naive word-level split proportional to char position, if words exist
        if "words" in seg:
            n = len(seg["words"])
            cut = int(n * frac)
            first["words"] = seg["words"][:cut]
            second["words"] = seg["words"][cut:]

        out.extend([first, second])
        split_count += 1

    if split_count > 0:
        log.info(
            f"[post] Pass 3: split {split_count} long segments "
            f"(>{threshold}s, broken at sentence boundaries)"
        )
    return out
