# AccessLearn AI — Backend

FastAPI service implementing the pipeline described in the project spec:
upload → extract → classify → detect issues → transform → validate → export.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`.

### Database (Stage 4)

**Default (SQLite)** — zero dependencies, file at `./accesslearn.db`:

```bash
# DATABASE_URL=sqlite:///./accesslearn.db  (already the default)
```

**Postgres** — production path, same code:

```bash
# from the submission/ root
docker compose up -d
# in backend/.env:
DATABASE_URL=postgresql://accesslearn:accesslearn@localhost:5432/accesslearn
```

What is persisted:

| Table | Contents |
|---|---|
| `documents` | Full `StructuredDocument` (sections = extracted elements, issues, scores) as JSON + queryable status/filename |
| `semantic_representations` | One canonical `SemanticRepresentation` per document |
| `transformed_versions` | Every transformation output (complete history; multiple rows per document) |
| `validation_results` | Semantic consistency checks + reviewer decisions (audit trail) |

FK cascade: deleting a document removes its semantic rep, versions, and validations.
`init_db()` runs on app startup and creates missing tables. The fully normalized
Postgres schema (separate `elements` / `accessibility_issues` tables) is documented
in `../accesslearn-data-model/database/schema.sql` for when JSON-blob queries become
a bottleneck.

## Where things stand

Persistence is real (SQLAlchemy + `db/repository.py`, see `STATUS.md`), and
the PDF extraction path is now real end-to-end: upload → `/analyze` →
`pdf_extractor` (PyMuPDF) → `classifier` (deterministic heuristics) →
`issue_detector` (detect + score) → persisted `StructuredDocument` →
`GET /documents/{id}`. Everything past that (semantic extraction,
transformation, validation) is still intentionally stubbed
(`raise NotImplementedError`) pending a live model call — see `STATUS.md`
for the exact boundary and why it's drawn where it is.

Every remaining stub under `app/services/` has a docstring explaining
exactly what it needs to do and which spec section it maps to, plus a
`TODO` comment for whoever picks it up.

## Suggested build order (matches spec section 16)

1. ~~`services/extraction/pdf_extractor.py` — get real text + bboxes out of a PDF.~~ **Done** (deterministic, PyMuPDF).
2. ~~`services/analysis/classifier.py` — type each block.~~ **Done** (deterministic heuristics only; the AI pass for genuinely ambiguous blocks is still a TODO in the same file).
3. ~~`services/analysis/issue_detector.py` — finish the scoring formula.~~ **Done** (deterministic weighted formula per category).
4. `services/analysis/semantic_extractor.py` — the canonical representation. **Next up** — first AI-dependent stage; `/analyze` deliberately stops just before this so the extraction round-trip isn't blocked by it.
5. `services/transformation/*.py` — simplify, alt text, HTML (HTML gen is
   already implemented, since it's deterministic).
6. `services/validation/*.py` — number_validator is done (deterministic);
   semantic_validator needs concept/relationship matching wired up.
7. ~~Wire routes in `api/routes/*.py` to the above, replace the in-memory
   `_DOCUMENTS` dict with real persistence via `db/database.py`.~~ **Done.**

## Design rule to keep

AI interprets language and images (simplification, alt text, translation,
concept extraction). Code enforces structure and checks exact values
(numbers, units, negation words, table headers, HTML validity). Don't let
an AI call quietly take over a job a regex or a schema check should do —
that's what keeps the semantic validator trustworthy.
