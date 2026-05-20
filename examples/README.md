# Examples

Ready-to-run scripts. The TachiDUBB server must be running (`start.bat` /
`./start.sh`); these scripts talk to it via HTTP.

| File | What it does |
|---|---|
| [`single_dub.py`](single_dub.py) | Dub one video into one language, save the URL |
| [`showcase_reel.py`](showcase_reel.py) | Build a 5-language stitched showcase reel from one source |
| [`compare_languages.py`](compare_languages.py) | Generate N side-by-side dubs for A/B listening |
| [`watch_folder.py`](watch_folder.py) | Watch a directory and auto-dub anything dropped in |
| [`agentic.md`](agentic.md) | What to tell Claude Code / Cursor / Cline once the MCP server is wired up |

Set `TACHIDUBB_URL` if the server is on another machine:

```bash
export TACHIDUBB_URL=http://192.168.0.10:8910
```
