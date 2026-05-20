"""Dub one source into one language and print the output URL.

Usage:
    python examples/single_dub.py https://youtu.be/<id> fr
    python examples/single_dub.py ./clip.mp4 ja
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from tachidubb_client import TachiDUBBClient  # noqa: E402


async def run(source: str, lang: str) -> int:
    async with TachiDUBBClient() as c:
        submitted = await c.submit_dub(source=source, target_lang=lang)
        job_id = submitted["job_id"]
        print(f"Submitted {job_id} — polling...")
        final = await c.wait_for_job(job_id)
        if final.get("status") == "complete":
            print(f"OK  {c.output_url(job_id)}")
            return 0
        print(f"FAIL  {final.get('error') or final}")
        return 1


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    return asyncio.run(run(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
