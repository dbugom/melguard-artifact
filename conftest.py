"""Pytest bootstrap: make the repo root importable so `import melguard` works
without an editable install. Keeps CI and local runs identical.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
