"""
app/exceptions.py
─────────────────
Custom exception classes and FastAPI exception handlers.

Why custom exceptions?
  Python's built-in exceptions (ValueError, FileNotFoundError) are too generic
  — they carry no HTTP status code or structured message.  FastAPI's built-in
  HTTPException is fine for simple cases, but for application-specific errors
  (model not loaded, unsupported file type, geo validation) having named
  exception classes makes the code self-documenting and the handlers reusable.

How FastAPI exception handlers work:
  app.add_exception_handler(ExceptionClass, handler_function)
  When any code raises ExceptionClass (or a subclass), FastAPI catches it,
  calls the handler, and returns its JSONResponse — the normal request
  lifecycle is bypassed cleanly.  The caller always gets a structured JSON
  body, never a raw Python traceback.

Registration:
  register_exception_handlers(app) is called from main.py after the app
  is created.
"""
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# ── Custom exception classes ──────────────────────────────────────────────────

class AIServiceUnavailable(Exception):
    """Raised when the AI model is not loaded or inference fails."""
    pass


class UnsupportedFileType(Exception):
    """Raised when an uploaded file is not an accepted image type."""
    def __init__(self, filename: str, content_type: str):
        self.filename     = filename
        self.content_type = content_type
        super().__init__(f"Unsupported file type '{content_type}' for '{filename}'")


class GeolocationError(Exception):
    """Raised when GPS coordinates are invalid or incomplete."""
    pass


# ── Exception handlers ────────────────────────────────────────────────────────

async def ai_unavailable_handler(request: Request, exc: AIServiceUnavailable):
    """
    503 Service Unavailable — the model is not loaded or inference failed.
    The frontend should display a "classification unavailable" message rather
    than a generic error, so we include a machine-readable error_code.
    """
    logger.error(f"AI unavailable on {request.url}: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error_code": "AI_UNAVAILABLE",
            "detail": str(exc) or "The AI model is currently unavailable. Please try again later.",
        },
    )


async def unsupported_file_handler(request: Request, exc: UnsupportedFileType):
    """415 Unsupported Media Type — wrong file format uploaded."""
    logger.warning(f"Unsupported file: {exc.filename} ({exc.content_type})")
    return JSONResponse(
        status_code=415,
        content={
            "error_code": "UNSUPPORTED_FILE_TYPE",
            "detail": (
                f"File '{exc.filename}' has unsupported type '{exc.content_type}'. "
                f"Please upload JPEG, PNG, or WEBP images."
            ),
        },
    )


async def geo_error_handler(request: Request, exc: GeolocationError):
    """422 Unprocessable Entity — invalid GPS coordinates."""
    logger.warning(f"Geolocation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "GEOLOCATION_ERROR",
            "detail": str(exc),
        },
    )


async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    422 Unprocessable Entity — Pydantic validation failed.

    FastAPI already handles this by default, but the default response nests
    errors under a 'detail' list which can be verbose.  This handler flattens
    each error to a human-readable message — easier for the frontend to display.
    """
    errors = []
    for error in exc.errors():
        field   = " → ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")

    logger.warning(f"Validation error on {request.url}: {errors}")
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "detail": errors,
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Catch-all for HTTPException — ensures a consistent JSON structure
    even for 404s raised by FastAPI's routing (e.g. unknown URL).
    """
    logger.warning(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "detail": exc.detail,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    500 Internal Server Error — anything not caught above.
    We log the full traceback but return a safe message to the client
    (never expose internal details in production).
    """
    logger.critical(
        f"Unhandled exception on {request.url}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "detail": "An unexpected error occurred. The team has been notified.",
        },
    )


# ── Registration helper ───────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI):
    """
    Called once from main.py after app = FastAPI(...).

    Order matters for catch-all handlers — more specific exceptions
    should be registered before broader ones.
    """
    app.add_exception_handler(AIServiceUnavailable,   ai_unavailable_handler)
    app.add_exception_handler(UnsupportedFileType,    unsupported_file_handler)
    app.add_exception_handler(GeolocationError,       geo_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception,              unhandled_exception_handler)