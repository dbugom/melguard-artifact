# Frozen evaluation harness — byte-exact snapshot

These are the thirteen files listed in `../HARNESS.sha256` exactly as they were when the sealed
held-out set was read (the one-shot evaluation recorded in `../sealed_result.json`, whose
`provenance.harness_sha256` block carries the same hashes for the code and lock files).

Verify all thirteen from this directory:

    sha256sum -c ../HARNESS.sha256

Nine of the thirteen are identical to the copies in their normal locations in this repository.
Four differ from their in-place copies because they were edited after the sealed run, for
non-scientific reasons: `tests/test_sealed_eval.py`, `tests/test_phase7_driver.py`, and
`tests/test_number_lock.py` gained `pytest.importorskip` guards so a lightweight CI job collects
cleanly, and `env/MODEL_PINS.lock.json` gained a "GPU load-tested" annotation on the Qwen2-Audio
pin used only in the exploratory transfer sub-study. No evaluated value changed; the in-place
copies are the ones the test suite runs. This snapshot exists so the frozen state is verifiable
without trusting that statement.
