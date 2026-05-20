# Media for the repo

Drop the following files here and they show up in the README automatically.

| File | Used in | Notes |
|---|---|---|
| `demo.gif` | Hero image at the top of README | < 10 MB ideal, ≤ 1280 px wide |
| `screenshot-ui.png` | (optional) UI screenshot | Used if you add it to the README |
| `screenshot-mcp.png` | (optional) MCP / Claude chat screenshot | Used if you add it to the README |

## Converting your screen recording to a GitHub-ready GIF

Best size/quality balance, two-pass palette:

```bash
# 12 fps, 1080 px wide, ~5-8 MB for a 30-second clip
ffmpeg -i your_recording.mp4 \
  -vf "fps=12,scale=1080:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  -loop 0 docs/demo.gif
```

If the GIF is still > 10 MB:

```bash
# Drop to 10 fps and 900 px wide
ffmpeg -i your_recording.mp4 \
  -vf "fps=10,scale=900:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  -loop 0 docs/demo.gif
```

Or just commit the `.mp4` directly and embed it with a regular `<video>` tag
in the README — GitHub renders these on the rendered page.

## Trim before converting

```bash
ffmpeg -ss 00:00:03 -to 00:00:33 -i your_recording.mp4 -c copy trimmed.mp4
```

That clips seconds 3 through 33, no re-encode. Then feed `trimmed.mp4` to
the GIF command above.
