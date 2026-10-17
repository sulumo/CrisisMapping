"""
app/main.py
───────────
Application entry point — keeps only what belongs here:
  1. FastAPI app instantiation
  2. Lifespan (startup / shutdown)
  3. Middleware (CORS)
  4. Static file mounting
  5. Router registration
  6. Database table creation

Everything else (routes, WebSocket logic, AI service, DB sessions) lives in
its own module.  This file should stay short and easy to scan.

FastAPI concepts here:
  lifespan          : asynccontextmanager that wraps startup + shutdown logic.
                      Code before `yield` runs on startup; after on shutdown.
  app.include_router: attaches a router's routes to the main app.  The router
                      can have its own prefix, tags, and dependencies.
  StaticFiles       : serves files from a directory under a URL prefix.
                      Here, images uploaded by users are served at /static/...
  CORSMiddleware    : tells browsers which origins are allowed to call this API.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import models
from .database import engine
from .ai_core.service import ai_engine
from .exceptions import register_exception_handlers
from .routers import events, simulation, websocket

# ── Logging configuration ─────────────────────────────────────────────────────
# basicConfig sets the root logger.  All `logging.getLogger(__name__)` calls
# in other modules inherit this level and format.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the first request, and shutdown logic after
    the last response.

    Startup order matters:
      1. Ensure the static image directory exists (StaticFiles will error
         if the directory is missing at mount time).
      2. Load the AI model — this takes several seconds on first run
         (downloading CLIP weights) but happens before any request arrives.
    """
    # Startup
    os.makedirs("app/static/image", exist_ok=True)
    logger.info("Static directories ready")

    ai_engine.load_model()

    yield  # ← application runs here; everything below runs on shutdown

    # Shutdown
    logger.info("Server shutting down — cleaning up")


# ── App instantiation ─────────────────────────────────────────────────────────
app = FastAPI(
    title="CrisisMapping API",
    description=(
        "Backend for the CrisisMapping platform. "
        "Accepts crisis reports, classifies them with CLIP_CrisisKAN, "
        "and serves them for real-time map display."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Exception handlers ───────────────────────────────────────────────────────
register_exception_handlers(app)

# ── CORS middleware ───────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) lets the browser make requests from
# one origin (e.g. localhost:3000) to a different origin (localhost:8000).
# Without this header, the browser blocks all API calls from the frontend.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://crisis-mapping-frontend.vercel.app",
    "http://192.168.1.47:3000",
    "http://203.252.106.25:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static files ──────────────────────────────────────────────────────────────
# Files saved to app/static/ are publicly accessible at /static/<filename>.
# Uploaded images land at /static/image/<filename>.
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ── Database table creation ───────────────────────────────────────────────────
# create_all is idempotent — it only creates tables that don't already exist.
# In production you would use Alembic migrations instead, but create_all is
# fine for development and first deployment.
models.Base.metadata.create_all(bind=engine)


# ── Router registration ───────────────────────────────────────────────────────
# Each router owns its routes.  Registering it here makes them part of the app.
# The `tags` parameter groups endpoints in the auto-generated /docs page.
app.include_router(events.router)
app.include_router(simulation.router)
app.include_router(websocket.router)

logger.info("CrisisMapping API ready")