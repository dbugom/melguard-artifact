"""Stub-VLA harness package.

GUARDRAIL (spec.lock.md sec10.3): no module under harness/ may import any of
{requests, socket, subprocess, smtplib, http}. No tool here resolves to a real
network / filesystem / process operation. tests/test_no_live_tools.py enforces
this on every push.
"""
