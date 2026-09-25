"""
Proves the thing STATUS.md calls out as the actual risk: that a real
StructuredDocument -- produced by real PyMuPDF extraction, not a hand-built
fixture -- survives a round trip through the DB with its schema intact
(no field that fails to serialize, no id that collides, no section that
comes back different from how it went in).

Every test builds its own throwaway PDF with `fitz` rather than shipping a
binary fixture, so the extraction code is exercised against a real (if
tiny) PDF file on disk, exactly like a real upload.
"""
from __future__ import annotations

import io

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_text_pdf() -> bytes:
    """One page: a large bold title (-> HEADING), a body paragraph
    (-> PARAGRAPH), and a bulleted list (-> LIST), at clearly different font
    sizes so the classifier's heuristics have something real to key off."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Photosynthesis", fontsize=24)
    page.insert_text(
        (72, 150),
        "Plants convert sunlight into chemical energy through a process\n"
        "called photosynthesis, which also releases oxygen as a byproduct.",
        fontsize=11,
    )
    page.insert_text(
        (72, 260),
        "- Requires sunlight\n- Requires water\n- Releases oxygen",
        fontsize=11,
    )
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_image_only_pdf() -> bytes:
    """One page containing only a rendered image and no extractable text,
    so `is_scanned()` should treat it as needing OCR."""
    doc = fitz.open()
    page = doc.new_page()
    # A tiny in-memory solid-color image, inserted as the page's only content.
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50))
    pixmap.clear_with(200)
    page.insert_image(fitz.Rect(72, 72, 300, 300), pixmap=pixmap)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _upload(pdf_bytes: bytes, filename: str = "lesson.pdf") -> dict:
    resp = client.post(
        "/documents/upload",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_pdf_extraction_round_trips_through_the_db():
    uploaded = _upload(_make_text_pdf())
    document_id = uploaded["document_id"]
    assert uploaded["status"] == "uploaded"
    assert uploaded["source_format"] == "pdf"

    analyze_resp = client.post(f"/documents/{document_id}/analyze")
    assert analyze_resp.status_code == 200, analyze_resp.text
    analyzed = analyze_resp.json()

    assert analyzed["status"] == "analyzed"
    assert analyzed["document_id"] == document_id
    assert len(analyzed["sections"]) >= 3

    types = [s["type"] for s in analyzed["sections"]]
    assert "heading" in types
    assert "paragraph" in types
    assert "list" in types

    heading = next(s for s in analyzed["sections"] if s["type"] == "heading")
    assert heading["heading_level"] is not None
    assert "Photosynthesis" in heading["text"]

    the_list = next(s for s in analyzed["sections"] if s["type"] == "list")
    assert len(the_list["list_items"]) == 3
    assert all(not item.startswith("-") for item in the_list["list_items"])  # marker stripped

    # Reading order preserved: heading before paragraph before list.
    orders = {s["type"]: s["order"] for s in analyzed["sections"]}
    assert orders["heading"] < orders["paragraph"] < orders["list"]

    # accessibility_score exists and every sub-score is a valid 0-100 int.
    score = analyzed["accessibility_score"]
    assert score is not None
    for key in ("overall", "text", "visual", "audio", "structural"):
        assert 0 <= score[key] <= 100

    # Every section id is unique -- exactly the kind of thing a round trip
    # through a real DB (not an in-memory dict) is what actually proves.
    section_ids = [s["id"] for s in analyzed["sections"]]
    assert len(section_ids) == len(set(section_ids))

    # Every accessibility_issue.element_id references a real section.
    for issue in analyzed["accessibility_issues"]:
        assert issue["element_id"] in section_ids

    # THE round-trip check: fetch the same document in a brand new request
    # (own DB session) and confirm it's byte-for-byte the same document the
    # analyze call just returned -- not a re-derived approximation.
    fetched = client.get(f"/documents/{document_id}").json()
    assert fetched == analyzed


def test_scanned_pdf_501s_instead_of_silently_returning_an_empty_document():
    uploaded = _upload(_make_image_only_pdf(), filename="scanned.pdf")
    document_id = uploaded["document_id"]

    resp = client.post(f"/documents/{document_id}/analyze")
    assert resp.status_code == 501
    assert "ocr" in resp.json()["detail"].lower()

    # Status was updated to reflect the failure, and the row is still there.
    doc = client.get(f"/documents/{document_id}").json()
    assert doc["status"] == "failed"


def test_corrupt_pdf_422s_cleanly():
    uploaded = _upload(b"not actually a pdf file", filename="broken.pdf")
    document_id = uploaded["document_id"]

    resp = client.post(f"/documents/{document_id}/analyze")
    assert resp.status_code == 422
    assert client.get(f"/documents/{document_id}").json()["status"] == "failed"


def test_non_pdf_format_still_501s():
    """The extraction path is deliberately scoped to pdf only for now --
    every other format should still 501, not silently no-op."""
    uploaded = _upload(b"hello world", filename="lesson.txt")
    resp = client.post(f"/documents/{uploaded['document_id']}/analyze")
    assert resp.status_code == 501
    assert "pdf" in resp.json()["detail"].lower()


def test_analyzing_twice_is_rejected_not_silently_redone():
    uploaded = _upload(_make_text_pdf())
    document_id = uploaded["document_id"]
    assert client.post(f"/documents/{document_id}/analyze").status_code == 200

    second = client.post(f"/documents/{document_id}/analyze")
    assert second.status_code == 409


def test_transform_still_409s_after_real_analysis_because_no_semantic_representation_yet():
    """Confirms the intentional boundary: extraction/classification now
    produce a real, persisted StructuredDocument, but semantic extraction
    (AI-dependent) is still not called, so /transform correctly refuses
    rather than fabricating a semantic representation that doesn't exist."""
    uploaded = _upload(_make_text_pdf())
    document_id = uploaded["document_id"]
    assert client.post(f"/documents/{document_id}/analyze").status_code == 200

    resp = client.post(
        f"/documents/{document_id}/transform",
        json={"document_id": document_id, "target_formats": ["simplified_text"]},
    )
    assert resp.status_code == 409
    assert "semantic representation" in resp.json()["detail"].lower()
