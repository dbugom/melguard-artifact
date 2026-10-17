# Research policy and guardrails

This artifact accompanies a defensive security study that attacks a detector in order to
measure it. The following guardrails are enforced in the code and in this release.

- **No live tool endpoints.** `harness/stub_vla.py` is inert by construction: every restricted
  tool is a stub that raises `NotImplementedError` if it is ever called, the harness measures
  by default (`execute=False`) and reaches a stub only through an explicit debug switch, and
  `tests/test_no_live_tools.py` asserts by AST inspection that no module under `harness/`
  imports `requests`, `socket`, `subprocess`, `smtplib`, `http`, `urllib`, `ftplib`, or
  `asyncio`. There is no runtime interception layer; there is simply no code path that
  performs a real network, filesystem, or process operation.
- **No working trigger waveforms or copy-paste payloads** in the artifact. Triggers are
  distributed as SHA-256 hashes plus the full optimizer configuration, so they can be
  regenerated under the pinned setup but not replayed from a supplied file.
- **Coordinated disclosure.** A 90-day coordinated-disclosure window (Day 0 = 2026-07-17)
  preceded this public release (window closed 2026-10-15).
- **Locked reproduction pins.** `env/*.lock.json` fix the models, numbers, and data
  manifest; the code reads these rather than hardcoding values.
- **One-shot sealed evaluation.** The held-out set was read exactly once
  (`out/phase7/SEAL_BROKEN.json` is the record); no result in this artifact post-dates or
  re-tunes against it.
