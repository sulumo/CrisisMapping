"""
routers/simulation.py
─────────────────────
Endpoints for the mock-data simulation engine:

  POST /api/start-simulation  — trigger the simulation in the background
  GET  /api/active-events     — return the last 20 informative events (fixed)

Why BackgroundTasks?
  The simulation loop takes ~40 seconds (4s × 10 events).  If we ran it
  inside the route handler, the HTTP response would not arrive until it
  finished — the frontend would hang.

  BackgroundTasks tells FastAPI to return the HTTP response immediately,
  then run the task concurrently on the same async event loop.  The task
  can still write to the database and broadcast WebSocket messages as if
  it were a normal request handler.

Database sessions in background tasks:
  The `get_db` dependency only works inside route handlers (FastAPI manages
  its lifecycle).  Inside a background task we have no request context, so
  we open and close the session manually using SessionLocal() directly.
  This is the correct pattern for any code that runs outside a request.
"""
import asyncio
import json
import logging
import os
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..dependencies import get_db
from ..models import CrisisEvent
from ..schemas import EventSummary
from ..routers.websocket import manager   # the shared ConnectionManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Simulation"])


# ─────────────────────────────────────────────────────────────────────────────
# Simulation engine
# ─────────────────────────────────────────────────────────────────────────────
async def run_simulation():
    """
    Reads mock events from data/simulation.json, saves each to the database,
    broadcasts it over WebSocket, then waits 4 seconds before the next one.

    This runs as a background task — see start_simulation_endpoint below.
    """
    logger.info("Simulation Engine Started")
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "simulation.json")

    # ── Load mock data ────────────────────────────────────────────────────────
    try:
        with open(file_path) as f:
            mock_events = json.load(f)
    except FileNotFoundError:
        logger.error(f"Simulation data not found at {file_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Simulation data is malformed: {e}")
        return

    # ── Process each event ────────────────────────────────────────────────────
    for raw in mock_events:
        logger.info(f"[Simulation] Processing: {raw.get('type')} in {raw.get('source')}")

        # ── A. Save to database ───────────────────────────────────────────────
        # We open a fresh session here because we are outside a request context.
        db: Session = SessionLocal()
        saved_id = None
        try:
            lon = raw.get("lon", 0)
            lat = raw.get("lat", 0)

            new_event = CrisisEvent(
                text          = raw.get("text", "Simulated event"),
                location_name = raw.get("source", "Unknown"),
                image_url     = raw.get("image_url", ""),
                latitude      = lat,
                longitude     = lon,
                geom          = f"SRID=4326;POINT({lon} {lat})",
                severity      = raw.get("severity",     "Unknown"),
                humanitarian  = raw.get("humanitarian", "Unknown"),
                type          = raw.get("type",         "Unknown"),
                is_informative= True,   # simulation events are always informative
            )
            db.add(new_event)
            db.commit()
            db.refresh(new_event)
            saved_id = new_event.id
            logger.info(f"[Simulation] Saved event id={saved_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"[Simulation] DB save failed: {e}", exc_info=True)
        finally:
            db.close()

        # ── B. Broadcast over WebSocket ───────────────────────────────────────
        payload = {
            "type": "NEW_EVENT",
            "data": {**raw, "id": saved_id},
        }
        await manager.broadcast(payload)
        logger.info(f"[Simulation] Broadcasted event to {len(manager.active_connections)} clients")

        # ── C. Wait before next event ─────────────────────────────────────────
        await asyncio.sleep(4)

    logger.info("Simulation complete — all events processed")


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/start-simulation
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/api/start-simulation",
    summary="Trigger the mock event simulation",
    description=(
        "Starts replaying events from simulation.json in the background. "
        "Events are saved to the database and broadcast over WebSocket every 4 seconds."
    ),
)
async def start_simulation_endpoint(background_tasks: BackgroundTasks):
    """
    Returns immediately; the simulation runs concurrently in the background.

    BackgroundTasks vs asyncio.create_task:
      Both work here.  BackgroundTasks is the FastAPI-idiomatic choice — it
      integrates with the request lifecycle and is easier to test.
    """
    background_tasks.add_task(run_simulation)
    return {"message": "Simulation started. Events will appear on the map every 4 seconds."}


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/active-events  (fixed — was returning undefined `events_to_broadcast`)
# ─────────────────────────────────────────────────────────────────────────────
@router.get(
    "/api/active-events",
    response_model=List[EventSummary],
    summary="Get the most recent active events",
    description=(
        "Returns the 20 most recent informative events. "
        "Useful for populating the map when a new user connects before any "
        "WebSocket events have been received."
    ),
)
def get_active_events(db: Session = Depends(get_db)):
    """
    The original implementation returned `events_to_broadcast` which was
    never defined — a NameError on every call.  This replaces it with a
    simple DB query for the 20 most recent informative events.
    """
    events = (
        db.query(CrisisEvent)
        .filter(CrisisEvent.is_informative == True)   # noqa: E712
        .order_by(CrisisEvent.created_at.desc())
        .limit(20)
        .all()
    )
    return events