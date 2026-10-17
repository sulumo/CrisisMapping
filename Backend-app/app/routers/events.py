"""
routers/events.py
─────────────────
All endpoints that deal with crisis events:
  POST /events   — submit a new report (image + text + GPS)
  GET  /events   — query stored events with filters

FastAPI concepts demonstrated here:
  APIRouter     : a mini-app that owns a subset of routes; registered in main.py
  Depends()     : injects a DB session per request (see dependencies.py)
  UploadFile    : FastAPI's wrapper for multipart file uploads
  Form()        : extracts text fields from a multipart/form-data request
  HTTPException : raises a structured error response (status code + detail)
  response_model: tells FastAPI which Pydantic schema to use for serialisation
  Union         : lets an endpoint return one of two different response shapes
"""
import logging
import os
import shutil
from datetime import datetime
from typing import List, Optional, Union

from fastapi import (
    APIRouter, Depends, File, Form,
    HTTPException, Query, UploadFile
)
from fastapi.responses import JSONResponse
from sqlalchemy import cast, func, or_
from sqlalchemy.orm import Session
from geoalchemy2 import Geography

from ..dependencies import get_db
from ..models import CrisisEvent
from ..schemas import (
    AIResult, DisasterType,
    EventResponse, EventSummary, NonInformativeResponse,
)
from ..ai_core.service import ai_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Events"])

UPLOAD_DIR = "app/static/image"


# ─────────────────────────────────────────────────────────────────────────────
# POST /events — submit a new crisis report
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/events",
    # Union return type: informative events return EventResponse,
    # non-informative ones return a lightweight message + id.
    response_model=Union[EventResponse, NonInformativeResponse],
    status_code=201,
    summary="Submit a new crisis report",
    description=(
        "Accepts one or more images plus text and GPS coordinates. "
        "The AI model classifies the report and the result is stored. "
        "Non-informative reports are saved but excluded from map queries."
    ),
)
async def create_event(
    # ── Multipart form fields ────────────────────────────────────────────────
    # FastAPI reads these directly from the multipart body.
    # `File(...)` means at least one file is required.
    # `Form(...)` means the field is required.
    files: List[UploadFile] = File(...,  description="One or more crisis images"),
    text:          str       = Form(...,  description="Description of the event"),
    location_name: str       = Form(...,  description="Human-readable place name"),
    latitude:      float     = Form(...,  description="GPS latitude  (-90 to 90)"),
    longitude:     float     = Form(...,  description="GPS longitude (-180 to 180)"),
    type:          str       = Form("Unknown", description="Disaster type"),
    db: Session = Depends(get_db),
):
    # ── 1. Validate coordinates ───────────────────────────────────────────────
    # Pydantic does not validate Form() fields automatically, so we do it here.
    if not (-90 <= latitude <= 90):
        raise HTTPException(status_code=422, detail="latitude must be between -90 and 90")
    if not (-180 <= longitude <= 180):
        raise HTTPException(status_code=422, detail="longitude must be between -180 and 180")

    # ── 2. Save uploaded images ───────────────────────────────────────────────
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_paths: List[str] = []
    image_urls:  List[str] = []

    for file in files:
        # Basic content-type guard — reject obvious non-images
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail=f"File '{file.filename}' is not an image (got {file.content_type})",
            )

        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        saved_paths.append(dest)
        image_urls.append(f"/static/image/{file.filename}")

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid image files were provided")

    # ── 3. AI inference ───────────────────────────────────────────────────────
    # We classify on the first image only (noted as a known limitation).
    # The result is validated against the AIResult schema to catch model errors early.
    try:
        raw_result = ai_engine.predict(text, saved_paths[0])
        ai_result  = AIResult(**raw_result)
    except Exception as e:
        logger.error(f"AI inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="AI model unavailable — try again later")

    # ── 4. Persist to database ────────────────────────────────────────────────
    combined_urls = ",".join(image_urls)

    new_event = CrisisEvent(
        text          = text,
        location_name = location_name,
        image_url     = combined_urls,
        latitude      = latitude,
        longitude     = longitude,
        # PostGIS WKT format: POINT(lon lat) — note longitude comes first
        geom          = f"SRID=4326;POINT({longitude} {latitude})",
        severity      = ai_result.severity,
        humanitarian  = ai_result.humanitarian,
        type          = type,
        is_informative= ai_result.is_informative,
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    logger.info(
        f"Event {new_event.id} saved — informative={new_event.is_informative}, "
        f"severity={new_event.severity}"
    )

    # ── 5. Return appropriate response ────────────────────────────────────────
    if not new_event.is_informative:
        # The event is stored (for audit purposes) but will not appear on the map.
        return JSONResponse(
            status_code=200,
            content={
                "message": (
                    "Report received and saved, but the AI classified it as "
                    "not informative. It will not appear on the map."
                ),
                "event_id": new_event.id,
            },
        )

    return new_event


# ─────────────────────────────────────────────────────────────────────────────
# GET /events — query stored events with filters
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/events",
    response_model=List[EventSummary],
    summary="Query crisis events",
    description=(
        "Returns only informative events. Supports full-text search, "
        "categorical filters, date ranges, PostGIS radius search, and pagination."
    ),
)
def get_events(
    db: Session = Depends(get_db),

    # ── Text search ──────────────────────────────────────────────────────────
    search: Optional[str] = Query(None, description="Keyword search in text or location"),

    # ── Categorical filters ──────────────────────────────────────────────────
    type:         Optional[str] = Query(None, description="Filter by disaster type"),
    humanitarian: Optional[str] = Query(None, description="Filter by humanitarian category"),
    severity:     Optional[str] = Query(None, description="Filter by damage severity"),

    # ── Date range ───────────────────────────────────────────────────────────
    start_date: Optional[datetime] = Query(None, description="Events from this date (ISO 8601)"),
    end_date:   Optional[datetime] = Query(None, description="Events up to this date (ISO 8601)"),

    # ── Geospatial radius search ─────────────────────────────────────────────
    lat:       Optional[float] = Query(None, description="Centre latitude for radius search"),
    lon:       Optional[float] = Query(None, description="Centre longitude for radius search"),
    radius_km: Optional[float] = Query(None, description="Search radius in kilometres", gt=0),

    # ── Pagination ───────────────────────────────────────────────────────────
    # Default 100, max 500. Always paginate — never return the whole table.
    limit:  int = Query(100, ge=1,  le=500, description="Maximum records to return"),
    offset: int = Query(0,   ge=0,          description="Records to skip (for paging)"),
):
    # Base query: only informative events reach the map
    query = db.query(CrisisEvent).filter(CrisisEvent.is_informative == True)  # noqa: E712

    # ── Text search (case-insensitive LIKE) ───────────────────────────────────
    if search:
        term = f"%{search}%"
        query = query.filter(
            or_(
                CrisisEvent.text.ilike(term),
                CrisisEvent.location_name.ilike(term),
            )
        )

    # ── Categorical exact-match filters ──────────────────────────────────────
    if type:
        query = query.filter(CrisisEvent.type == type)
    if humanitarian:
        query = query.filter(CrisisEvent.humanitarian == humanitarian)
    if severity:
        query = query.filter(CrisisEvent.severity == severity)

    # ── Date range ────────────────────────────────────────────────────────────
    if start_date:
        query = query.filter(CrisisEvent.created_at >= start_date)
    if end_date:
        query = query.filter(CrisisEvent.created_at <= end_date)

    # ── PostGIS radius search ─────────────────────────────────────────────────
    # ST_DWithin on Geography columns measures distance in metres over the
    # Earth's surface (not flat-map degrees), giving accurate results globally.
    if lat is not None and lon is not None and radius_km is not None:
        center = f"SRID=4326;POINT({lon} {lat})"
        query = query.filter(
            func.ST_DWithin(
                cast(CrisisEvent.geom, Geography),
                cast(center, Geography),
                radius_km * 1000,   # convert km → metres
            )
        )
    elif any(v is not None for v in (lat, lon, radius_km)):
        # Partial geo params — inform the caller rather than silently ignoring
        raise HTTPException(
            status_code=422,
            detail="Radius search requires lat, lon, AND radius_km together",
        )

    # ── Order + paginate ──────────────────────────────────────────────────────
    events = (
        query
        .order_by(CrisisEvent.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return events