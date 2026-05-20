# Agentic recipes

Once the MCP server is wired up in your agent (`claude mcp add tachidubb …`),
just talk to the agent. These are prompts that work reliably with the
bundled `.claude/skills/tachidubb/SKILL.md`.

## Single dub, fire-and-forget

> Dub https://youtu.be/dQw4w9WgXcQ into French. Don't wait — just give me
> the job ID and I'll check on it later.

The agent calls `tachidubb_dub(source=..., target_lang="fr", wait=False)`
and reports the job id.

## Single dub, blocking

> Dub https://youtu.be/abc into French and link me the MP4 when it's done.

The agent calls `tachidubb_dub(..., wait=True)` and reports the URL.

## Showcase reel for social posting

> Build a 60-second multilingual showcase of https://youtu.be/abc in
> Spanish, French, German, Japanese and Portuguese.

The agent calls `tachidubb_showcase(source=..., target_langs=[...],
trim_seconds=60, wait=True)`.

## Compare quality across languages

> I want to A/B listen to this clip in 5 languages — give me 5 separate
> mp4s, not a stitched reel: https://youtu.be/abc

The agent calls `tachidubb_compare(...)` and returns the URLs in a table.

## Re-dub from history

> The French one I made yesterday — also do Japanese and Italian.

The agent calls `tachidubb_list_jobs(status="complete")`, finds the
matching job, and calls `tachidubb_redub(job_id, target_langs=["ja","it"],
mode="compare")`. No re-upload, no re-transcribe.

## Pipeline health check

> What's the state of my TachiDUBB setup? Anything missing?

The agent calls `tachidubb_system_status()` and surfaces missing
dependencies (Ollama not running, no models pulled, GPU not detected,
VoxCPM2 not downloaded yet, etc).

## Smart language picking

> Dub this in whatever 3 languages would reach the largest audience.

The agent picks `en, zh, es` (or `en, hi, es` depending on its priors)
and calls `tachidubb_showcase(...)`.

## Long pipeline with cleanup

> Take the 5 most recent jobs in my history, re-dub each in Italian, then
> tell me the URLs as a markdown list.

The agent loops `tachidubb_list_jobs(limit=5)` → `tachidubb_redub(...)`
per job, polls each, then formats the output. This is where you actually
feel the difference between a one-shot tool and an agent.

## Recovery

> The showcase reel `sc_31623e75` failed at the stitch step. Can you fix
> it without redoing the dubs?

The agent calls `tachidubb_rebuild_showcase("sc_31623e75")` — re-runs
only the ffmpeg stitch, not the N dubs that already succeeded.
