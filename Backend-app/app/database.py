"""
database.py
───────────
Single source of truth for the SQLAlchemy engine and session factory.

Why DB_HOST has a fallback:
  - Inside Docker the database container is reachable by its service name "db"
    (Docker's internal DNS resolves it automatically).
  - Outside Docker (local dev without docker-compose) "db" does not resolve.
    Setting DB_HOST=localhost lets you run the app directly against a local
    Postgres instance without editing this file.

  Export DB_HOST=localhost before running locally, or set it in a .env file.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ── Connection parameters from environment ────────────────────────────────────
USER     = os.getenv("POSTGRES_USER",     "crisis_user")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "secure_password_123")
DB_NAME  = os.getenv("POSTGRES_DB",       "crisis_db")
# "db" is the Docker service name; override with DB_HOST=localhost for local dev
HOST     = os.getenv("DB_HOST",           "db")
PORT     = os.getenv("DB_PORT",           "5432")

SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
)

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping=True makes SQLAlchemy test the connection before using it,
# which prevents "server closed the connection unexpectedly" errors after
# the database restarts or a long idle period.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
)

# ── Session factory ───────────────────────────────────────────────────────────
# autocommit=False  → we call db.commit() explicitly after each write
# autoflush=False   → SQLAlchemy won't auto-flush pending changes mid-query
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Base class ────────────────────────────────────────────────────────────────
# All ORM models (models.py) inherit from this.
Base = declarative_base()