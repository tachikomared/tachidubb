"""Watch a folder and auto-dub anything dropped in.

Polls the folder every N seconds. Each new video gets dubbed into the
configured languages. The dubbed mp4 stays on the server under
`outputs/<job_id>/`; this script prints the URL for each result.

Usage:
    python examples/watch_folder.py ./inbox --langs fr,es --poll 10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from tachidubb_client import TachiDUBBClient  # noqa: E402

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}


async def run(folder: Path, langs: list[str], poll: int) -> int:
    seen: set[Path] = set()
    print(f"Watching {folder} for new videos (langs={langs}, poll={poll}s) — Ctrl+C to stop")
    async with TachiDUBBClient() as c:
        try:
            while True:
                for p in sorted(folder.iterdir()):
                    if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                        continue
                    if p in seen:
                        continue
                    seen.add(p)
                    for lang in langs:
                        print(f"> {p.name} -> {lang}")
                        try:
                            submitted = await c.submit_dub(
                                source=str(p), target_lang=lang
                            )
                            job_id = submitted["job_id"]
                            final = await c.wait_for_job(job_id)
                            if final.get("status") == "complete":
                                print(f"  OK   {c.output_url(job_id)}")
                            else:
                                print(f"  FAIL {final.get('error')}")
                        except Exception as e:
                            print(f"  ERR  {e}")
                await asyncio.sleep(poll)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--poll", type=int, default=10)
    args = ap.parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 2
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    return asyncio.run(run(folder, langs, args.poll))


if __name__ == "__main__":
    raise SystemExit(main())
