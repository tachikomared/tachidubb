"""Build a stitched multilingual showcase reel from one source.

Output is a single mp4 that switches language every ~12 s with a small
`· LL ·` badge in the corner. Great for posting on socials.

Usage:
    python examples/showcase_reel.py <source> --langs es,fr,de,ja,pt --trim 60
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
        submitted = await c.submit_showcase(
            source=source, target_langs=langs, trim_seconds=trim
        )
        batch_id = submitted.get("batch_id") or submitted.get("id")
        if not batch_id:
            print(f"No batch_id in response: {submitted}")
            return 1
        print(f"Submitted batch {batch_id} — building reel...")
        info = await c.wait_for_showcase(batch_id)
        url = info.get("url") or c.showcase_url(batch_id)
        print(f"OK  {url}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="YouTube URL or local file")
    ap.add_argument("--langs", required=True,
                    help="Comma-separated lang codes, e.g. es,fr,de,ja,pt")
    ap.add_argument("--trim", type=int, default=60,
                    help="Trim source to N seconds first (default 60)")
    args = ap.parse_args()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    if not 2 <= len(langs) <= 6:
        print("Pick 2-6 languages.")
        return 2
    return asyncio.run(run(args.source, langs, args.trim))


if __name__ == "__main__":
    raise SystemExit(main())
