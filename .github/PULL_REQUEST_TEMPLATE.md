<!-- Thanks for the PR! A few things to make review faster -->

## What this changes

<!-- One or two sentences. Be specific. -->

## Why

<!-- Link the issue if there is one: `Fixes #123` -->

## How to test

<!-- Commands or steps a reviewer can run -->

```bash
# example
python tools/tachidubb_cli.py dub <some url> --lang fr --wait
```

## Screenshots / clips

<!-- For UI changes or audio output changes, attach evidence -->

## Checklist

- [ ] I ran `ruff check .` (or my changes don't touch Python)
- [ ] I updated the README / CHANGELOG if user-visible behavior changed
- [ ] I did not add new required dependencies (or I discussed it in an issue first)
- [ ] My change preserves the local-only / no-cloud default
- [ ] My change does not bypass AI-disclosure / consent guardrails
