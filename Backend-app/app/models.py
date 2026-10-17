"""
models.py
─────────
SQLAlchemy ORM models — define the database table structure.

Each class maps to one table.  The columns define the schema that
`models.Base.metadata.create_all(bind=engine)` will create on first run.

PostGIS note:
  The `geom` column stores a geographic POINT using the EPSG:4326 (GPS)
  coordinate reference system.  It is separate from the plain `latitude` /
  `longitude` floats because:
    - lat/lon floats are fast to read and display
    - geom enables PostGIS spatial functions like ST_DWithin (radius search)
      which are far more accurate than doing the maths in Python
"""
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime
from sqlalchemy.sql import func
from geoalchemy2 import Geometry

from .database import Base


class CrisisEvent(Base):
    __tablename__ = "crisis_events"

    # ── Identity & timing ────────────────────────────────────────────────────
    id         = Column(Integer, primary_key=True, index=True)
    # server_default=func.now() means Postgres sets this, not Python —
    # so it is always the DB server time regardless of app server timezone.
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # ── User-supplied inputs ─────────────────────────────────────────────────
    text          = Column(Text,   nullable=False)
    location_name = Column(String, nullable=False)
    # Comma-separated relative URLs, e.g. "/static/image/a.jpg,/static/image/b.jpg"
    image_url     = Column(String, nullable=False)

    # ── Geolocation ──────────────────────────────────────────────────────────
    latitude  = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    # PostGIS geometry column — POINT(longitude latitude) in SRID 4326
    geom = Column(Geometry(geometry_type="POINT", srid=4326))

    # ── AI classification results ─────────────────────────────────────────────
    is_informative = Column(Boolean, nullable=False)
    severity       = Column(String,  nullable=True)   # "little or no damage" etc.
    humanitarian   = Column(String,  nullable=True)   # "infrastructure" etc.
    type           = Column(String,  nullable=True)   # user-supplied disaster type