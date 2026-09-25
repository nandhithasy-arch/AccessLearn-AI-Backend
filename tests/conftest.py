"""
Points the app at a throwaway sqlite file (not the dev accesslearn.db, and
not :memory: -- a real file so every TestClient request, which gets its own
DB session, reads/writes the same database) before app.main is imported.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_tmp_dir = tempfile.mkdtemp(prefix="accesslearn_test_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_dir}/test.db")
os.environ.setdefault("UPLOAD_DIR", f"{_tmp_dir}/uploads")
os.environ.setdefault("OUTPUT_DIR", f"{_tmp_dir}/outputs")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
