"""GUARDRAIL (spec.lock.md sec10.3 / POLICY.md): no module under harness/ may
import any live-I/O module. Enforced by AST, so a name inside a comment or
string cannot fool it, and a real `import subprocess` cannot slip through.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "harness"

FORBIDDEN = {"requests", "socket", "subprocess", "smtplib", "http", "urllib", "ftplib", "asyncio"}


def _harness_py_files() -> list[Path]:
    return sorted(HARNESS.rglob("*.py"))


def test_harness_dir_is_populated():
    # Non-vacuous: the harness package must exist with at least the stub module.
    assert HARNESS.is_dir(), "harness/ package is missing"
    files = {p.name for p in _harness_py_files()}
    assert "stub_vla.py" in files, "harness/stub_vla.py is missing"


def test_no_forbidden_imports_in_harness():
    offenders: list[str] = []
    for path in _harness_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in FORBIDDEN:
                        offenders.append(f"{path.name}: from {node.module} import ...")
    assert not offenders, "GUARDRAIL breach — live-I/O imports in harness/: " + "; ".join(offenders)


def test_stub_tools_raise_not_implemented():
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness.stub_vla import STUB_TOOLS, RESTRICTED_TOOLS

    assert set(STUB_TOOLS) == set(RESTRICTED_TOOLS)
    for name, fn in STUB_TOOLS.items():
        try:
            fn(target="[REDACTED]")
        except NotImplementedError:
            continue
        raise AssertionError(f"stub tool {name!r} did not raise NotImplementedError")
