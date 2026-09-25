"""
Local disk storage for the MVP (swap for S3/GCS later behind this same
interface). Responsible-AI note from spec section 18: enforce
`file_retention_days` here -- a scheduled cleanup job or a check-on-read
that deletes anything past retention.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from app.core.config import get_settings


class FileStorage:
    def __init__(self) -> None:
        settings = get_settings()
        self.upload_dir = Path(settings.upload_dir)
        self.output_dir = Path(settings.output_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> str:
        key = f"{uuid.uuid4().hex}_{filename}"
        path = self.upload_dir / key
        path.write_bytes(content)
        return str(path)

    def save_output(self, document_id: str, name: str, content: bytes | str) -> str:
        doc_dir = self.output_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        path = doc_dir / name
        # `name` may include subdirectory components (e.g. "images/p1_0.png",
        # used by pdf_extractor for extracted embedded images) -- make sure
        # that subdirectory actually exists before writing, or this raises
        # FileNotFoundError instead of writing the file.
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return str(path)

    def delete_document_files(self, document_id: str) -> None:
        shutil.rmtree(self.output_dir / document_id, ignore_errors=True)
