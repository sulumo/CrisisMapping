"""
schemas.py
──────────
Pydantic models used for:
  - Input validation  (what FastAPI checks before your code runs)
  - Output shaping    (what FastAPI serialises into JSON)
  - Internal typing   (AIResult keeps ai_core/service.py type-safe)

Why Pydantic schemas and SQLAlchemy models are separate:
  SQLAlchemy models describe HOW data is stored (columns, types, indexes).
  Pydantic schemas describe WHAT the API accepts and returns.
  Keeping them separate means you can change one without breaking the other.
  For example, you might store a comma-separated image_url string in the DB
  but return a proper list of URLs to the frontend — that translation lives here.

Pydantic v2 notes:
  - `model_config = ConfigDict(from_attributes=True)` replaces the old
    `class Config: orm_mode = True`.  It tells Pydantic to read values from
    ORM object attributes (not just dicts), which is needed when FastAPI
    converts a SQLAlchemy row into a response schema.
  - `Field(...)` marks a field as required.  `Field(default)` is optional.
  - Validators run before the field is stored — they raise 422 automatically.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Disaster type choices ─────────────────────────────────────────────────────
# Using Literal means FastAPI rejects any value not in this list with a 422.
DisasterType = Literal[
    "Flood", "Fire", "Earthquake", "Storm",
    "Landslide", "Hurricane", "Tsunami", "Unknown"
]


# ─────────────────────────────────────────────────────────────────────────────
# AI result — internal use only (not exposed directly to API callers)
# ─────────────────────────────────────────────────────────────────────────────
class AIResult(BaseModel):
    """
    Typed container for the output of CrisisAIService.predict().
    Using a schema here instead of a plain dict means any mismatch between
    what the model returns and what the endpoint expects is caught immediately.
    """
    is_informative: bool
    humanitarian:   str
    severity:       str


# ─────────────────────────────────────────────────────────────────────────────
# Event response — returned to the frontend on GET /events and POST /events
# ─────────────────────────────────────────────────────────────────────────────
class EventResponse(BaseModel):
    """
    Full event representation returned to the API caller.

    image_urls is a list so the frontend can iterate over multiple photos
    without having to split a comma-separated string itself.
    We derive it from the stored comma-separated image_url string via a
    field_validator below.
    """
    model_config = ConfigDict(from_attributes=True)

    id:            int
    created_at:    datetime
    text:          str
    location_name: str
    latitude:      float
    longitude:     float
    severity:      str
    humanitarian:  str
    type:          str
    is_informative: bool

    # The DB stores "/static/image/a.jpg,/static/image/b.jpg" as one string.
    # We expose it as a proper list to the frontend.
    image_urls: List[str] = Field(default_factory=list)

    @field_validator("image_urls", mode="before")
    @classmethod
    def split_image_url(cls, v):
        """
        Accept either a comma-separated string (from the ORM) or an
        already-split list (from tests / direct construction).
        """
        if isinstance(v, str):
            return [url.strip() for url in v.split(",") if url.strip()]
        return v or []


# ─────────────────────────────────────────────────────────────────────────────
# Event summary — lightweight version used in list responses
# ─────────────────────────────────────────────────────────────────────────────
class EventSummary(BaseModel):
    """
    Minimal event shape returned by GET /events.
    Omitting `text` (which can be long) keeps list payloads small when
    the frontend is rendering hundreds of map markers.
    """
    model_config = ConfigDict(from_attributes=True)

    id:            int
    created_at:    datetime
    location_name: str
    latitude:      float
    longitude:     float
    severity:      str
    humanitarian:  str
    type:          str
    is_informative: bool
    image_urls: List[str] = Field(default_factory=list)

    @field_validator("image_urls", mode="before")
    @classmethod
    def split_image_url(cls, v):
        if isinstance(v, str):
            return [url.strip() for url in v.split(",") if url.strip()]
        return v or []


# ─────────────────────────────────────────────────────────────────────────────
# Non-informative response — returned when AI marks a report as not informative
# ─────────────────────────────────────────────────────────────────────────────
class NonInformativeResponse(BaseModel):
    message: str
    event_id: int