# sub-D1-1 — empty-prediction baseline (format check)

Generated: 25 Sep 2026 · `src/make_empty_submission.py`

Every test Source 1 entity gets an empty match list. This is not a model. It exists to
prove the upload path end to end and to reveal the **public** singleton share, which
we cannot obtain any other way.

- Rows: 1,732,544 (+ header), all empty
- Official validator, with `--check-ids`: **PASS**
- Expected public score: about **0.056** if the public subset has the same singleton
  share as train (123,247 / 2,206,822 = 5.58%). A materially different score tells us
  the public split is not representative, which is worth knowing early.

Files are gzipped: 46 MB raw, and 15 uploads of that would bloat the repo with text we
can regenerate from the tagged commit at any time.
