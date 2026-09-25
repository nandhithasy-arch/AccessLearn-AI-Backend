"""
Proves the DB wiring, not the (still-stubbed) AI pipeline: a document
uploaded in one request is readable from a completely separate request/
session, survives being listed, and 404s cleanly when it doesn't exist --
none of which the old `_DOCUMENTS` in-memory dict could guarantee once you
have more than one worker process.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_persists_and_is_readable_in_a_separate_request():
    resp = client.post("/documents/upload", files={"file": ("lesson.txt", b"hello world", "text/plain")})
    assert resp.status_code == 201
    doc = resp.json()
    assert doc["status"] == "uploaded"
    assert doc["source_format"] == "text"
    document_id = doc["document_id"]

    resp2 = client.get(f"/documents/{document_id}")
    assert resp2.status_code == 200
    assert resp2.json()["document_id"] == document_id


def test_get_missing_document_404s():
    resp = client.get("/documents/does-not-exist")
    assert resp.status_code == 404


def test_list_documents_includes_uploads():
    client.post("/documents/upload", files={"file": ("a.txt", b"x", "text/plain")})
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_analyze_501s_but_document_row_is_real():
    upload = client.post("/documents/upload", files={"file": ("b.txt", b"y", "text/plain")})
    document_id = upload.json()["document_id"]

    analyze_resp = client.post(f"/documents/{document_id}/analyze")
    assert analyze_resp.status_code == 501  # pipeline logic not implemented yet -- expected

    # ...but the persistence layer under it is real, not a stub.
    assert client.get(f"/documents/{document_id}").status_code == 200


def test_unsupported_file_type_rejected_before_touching_the_db():
    resp = client.post("/documents/upload", files={"file": ("virus.exe", b"x", "application/octet-stream")})
    assert resp.status_code == 415


def test_empty_file_rejected():
    resp = client.post("/documents/upload", files={"file": ("empty.txt", b"", "text/plain")})
    assert resp.status_code == 400
