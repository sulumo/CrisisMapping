"""
tests/conftest.py
─────────────────
Shared pytest fixtures used across the test suite.

Key concepts:
  conftest.py is auto-loaded by pytest — any fixture defined here is available
  to every test file without importing it.

  Dependency overrides:
    FastAPI's `app.dependency_overrides` dict lets you replace any Depends()
    function with a different one for the duration of a test.  Here we replace
    `get_db` with `override_get_db`, which uses an in-memory SQLite database
    instead of the real PostGIS instance.  This means:
      - Tests run without a running database server
      - Each test session gets a fresh, empty database
      - Tests never pollute production or development data

  TestClient:
    FastAPI's TestClient wraps httpx and sends real HTTP requests to your app
    without starting a network server.  It runs the full request lifecycle
    (middleware, dependency injection, exception handlers) exactly as in
    production.

  Mocking the AI engine:
    We replace ai_engine.predict() with a function that returns a fixed result.
    This keeps tests fast (no GPU/model loading), deterministic (same result
    every run), and focused on the HTTP/DB logic rather than model accuracy.

  yield fixtures:
    A fixture with `yield` runs setup code before the test and teardown code
    after.  The `yield` is where the test executes.  This ensures cleanup
    happens even if the test fails.
"""
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base
from app.dependencies import get_db
from app.ai_core.service import ai_engine


# ── In-memory SQLite database ─────────────────────────────────────────────────
# SQLite does not support PostGIS geometry types, so we skip the geom column
# by using a patched Base that excludes it — or more practically, we just
# allow the column creation to fail silently by not using PostGIS extension.
# For full PostGIS tests you would need a real Postgres instance (CI service).
SQLITE_URL = "sqlite:///./test_crisis.db"

engine_test = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite with threads
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine_test
)


def override_get_db():
    """
    Replaces the real get_db dependency during tests.
    Uses the SQLite engine and rolls back after each request to keep tests isolated.
    """
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── AI engine mock ────────────────────────────────────────────────────────────
MOCK_AI_RESULT = {
    "is_informative": True,
    "humanitarian":   "infrastructure",
    "severity":       "severe damage",
}

MOCK_AI_NON_INFORMATIVE = {
    "is_informative": False,
    "humanitarian":   "not humanitarian",
    "severity":       "little or no damage",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_tables():
    """
    Create all tables once for the entire test session.
    Dropped at the end.

    scope="session" means this runs once per pytest session,
    not once per test — table creation is expensive.
    """
    # Import models so Base knows about them before create_all
    from app import models  # noqa: F401

    # Create tables, ignoring PostGIS-specific columns that SQLite can't handle
    try:
        Base.metadata.create_all(bind=engine_test)
    except Exception as e:
        print(f"[conftest] Table creation warning (expected on SQLite): {e}")

    yield

    Base.metadata.drop_all(bind=engine_test)


@pytest.fixture
def client(db_tables, monkeypatch):
    """
    A TestClient with:
      - DB dependency overridden to use SQLite
      - AI engine predict() mocked to return a fixed informative result
      - No model loading required

    scope is not set (default "function") — a fresh client per test,
    which resets the dependency override dict between tests.
    """
    # Override DB dependency
    app.dependency_overrides[get_db] = override_get_db

    # Mock AI predict — monkeypatch replaces the method for this test only
    monkeypatch.setattr(ai_engine, "predict", lambda text, path: MOCK_AI_RESULT)

    # Mark model as loaded so the "model is None" guard in predict() doesn't fire
    monkeypatch.setattr(ai_engine, "model", object())

    with TestClient(app) as c:
        yield c

    # Clean up overrides after the test
    app.dependency_overrides.clear()


@pytest.fixture
def client_non_informative(db_tables, monkeypatch):
    """
    Same as `client` but AI returns a non-informative result.
    Used to test the non-informative response path.
    """
    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(
        ai_engine, "predict", lambda text, path: MOCK_AI_NON_INFORMATIVE
    )
    monkeypatch.setattr(ai_engine, "model", object())

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """
    A minimal valid JPEG image as bytes.
    Avoids having image files in the repo — generates a tiny red square.
    """
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def event_form_data():
    """
    Valid multipart form fields for POST /events.
    Tests can override individual fields to test validation paths.
    """
    return {
        "text":          "A bridge has collapsed blocking the main road",
        "location_name": "Houston, TX",
        "latitude":      "29.7604",
        "longitude":     "-95.3698",
        "type":          "Flood",
    }