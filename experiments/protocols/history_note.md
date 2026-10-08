# Note on commit hashes (2026-10-08)

Before the repository was made public, the author e-mail in every commit was replaced with the
GitHub no-reply address. Only the e-mail changed: file contents (git trees), author dates and
committer dates are identical, so every commit hash changed but no content or timestamp did.

The pre-registration of the real-data calibration study is therefore:

| | commit | committed |
|---|---|---|
| original hash (recorded in `results/realdata_cal_v2_meta.json`) | `4ae16e3d286f52feec6fe5253b5a67afc914af25` | 2026-10-08 14:37:28 +05:30 |
| hash after the e-mail rewrite | `7965826b81ab32a334beb3d148b27d037c742f3e` | 2026-10-08 14:37:28 +05:30 |

Both commits have the same tree, so the protocol and runner they contain are byte-identical.

Full mapping (old → new): `344c815 → f302682`, `f40c49b → 1c6e22a`, `fad075f → b456787`,
`ac5cb8d → 61765a2`, `4ae16e3 → 7965826`, `48b3552 → 86017d3`, `c403be0 → 61cf86b`,
`8ea9dbe → 69ef11f`.
