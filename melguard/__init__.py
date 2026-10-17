"""MelGuard (YFI-17 / RQ3) — defender-side package.

A passive AST Mel-spectrogram detector deployed as a pre-LLM gateway against
sub-second adaptive audio-jailbreak triggers on SALMONN-13B.

All locked quantities are read from env/*.lock.json via melguard.locks — never
hardcode a NUMBER_LOCK value in code. See spec.lock.md and POLICY.md.
"""

__version__ = "0.0.0"
