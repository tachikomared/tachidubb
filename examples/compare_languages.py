"""Generate N separate dubs of the same source for A/B/C listening.

Unlike `showcase_reel.py`, this produces N full-length mp4 files — one per
language. Use it when you want to evaluate quality across languages.

Usage:
    python examples/compare_languages.py <source> --langs es,fr,de,ja,pt --trim 60
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from tachidubb_client import TachiDUBBClient  # noqa: E402


async def run(source: str, langs: list[str], trim: int) -> int:
    async with TachiDUBBClient() as c:
        submitted = await c.submit_compare(
            source=source, target_langs=langs, trim_seconds=trim
        )
        batch_id = submitted.get("batch_id") or submitted.get("id")
        if not batch_id:
            print(f"No batch_id in response: {submitted}")
            return 1
        print(f"Submitted batch {batch_id} — running {len(langs)} dubs...")
        finals = await c.wait_for_batch(batch_id)
        for j in finals:
            status = j.get("status")
            lang = j.get("target_lang")
            url = c.output_url(j.get("job_id", "")) if status == "complete" else "—"
            marker = "OK  " if status == "complete" else "FAIL"
            print(f"{marker} {lang}: {url}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--langs", required=True)
    ap.add_argument("--trim", type=int, default=60)
    args = ap.parse_args()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    return asyncio.run(run(args.source, langs, args.trim))


if __name__ == "__main__":
    raise SystemExit(main())
