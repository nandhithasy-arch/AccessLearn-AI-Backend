from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, reviews, transform, validation
from app.core.config import get_settings
from app.db.database import init_db

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Content-understanding, accessibility-transformation, and semantic-validation platform.",
    version="0.1.0",
)

# MVP-permissive CORS; tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "https://access-learn-aa3nclm89-logic-loom3.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    # Creates documents / semantic_representations / transformed_versions /
    # validation_results tables if they don't exist yet (sqlite file or
    # Postgres, per settings.database_url). Safe to call on every boot.
    init_db()


app.include_router(documents.router)
app.include_router(transform.router)
app.include_router(validation.router)
app.include_router(reviews.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name}
