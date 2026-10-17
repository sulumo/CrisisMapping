"""
tests/test_schemas.py
─────────────────────
Unit tests for Pydantic schemas — no HTTP client or database needed.

These test the schemas in isolation, which is valuable because:
  - Schema validation is the first thing that runs on every request
  - A bug here affects every endpoint simultaneously
  - These tests run in milliseconds — no I/O involved

Testing schemas directly also teaches you how Pydantic validates:
  model_validate()  : build a model from a dict (replaces old .parse_obj())
  model_dump()      : convert model back to dict
  ValidationError   : raised when required fields are missing or invalid
"""
import pytest
from pydantic import ValidationError
from datetime import datetime

from app.schemas import EventResponse, EventSummary, AIResult


class TestEventResponseSchema:

    def _valid_data(self, **overrides):
        base = {
            "id":            1,
            "created_at":    datetime.now(),
            "text":          "Severe flooding in the downtown area",
            "location_name": "New Orleans, LA",
            "latitude":      29.95,
            "longitude":     -90.07,
            "severity":      "severe damage",
            "humanitarian":  "infrastructure",
            "type":          "Flood",
            "is_informative": True,
            "image_url":     "/static/image/a.jpg,/static/image/b.jpg",
        }
        return {**base, **overrides}

    def test_valid_data_constructs_correctly(self):
        event = EventResponse.model_validate(self._valid_data())
        assert event.id == 1
        assert event.is_informative is True

    def test_image_url_string_split_into_list(self):
        """
        The DB stores comma-separated URLs; the schema should split them
        into a proper list automatically.
        """
        event = EventResponse.model_validate(self._valid_data())
        assert isinstance(event.image_urls, list)
        assert len(event.image_urls) == 2
        assert "/static/image/a.jpg" in event.image_urls
        assert "/static/image/b.jpg" in event.image_urls

    def test_empty_image_url_gives_empty_list(self):
        event = EventResponse.model_validate(self._valid_data(image_url=""))
        assert event.image_urls == []

    def test_single_image_url_gives_one_item_list(self):
        event = EventResponse.model_validate(
            self._valid_data(image_url="/static/image/only.jpg")
        )
        assert event.image_urls == ["/static/image/only.jpg"]

    def test_missing_required_field_raises_validation_error(self):
        data = self._valid_data()
        del data["latitude"]
        with pytest.raises(ValidationError) as exc_info:
            EventResponse.model_validate(data)
        # Check that the error mentions the missing field
        assert "latitude" in str(exc_info.value)


class TestAIResultSchema:

    def test_valid_ai_result(self):
        result = AIResult(
            is_informative=True,
            humanitarian="infrastructure",
            severity="severe damage",
        )
        assert result.is_informative is True

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            AIResult(is_informative=True, humanitarian="infrastructure")
            # severity is missing → ValidationError

    def test_model_dump_returns_dict(self):
        result = AIResult(
            is_informative=False,
            humanitarian="not humanitarian",
            severity="little or no damage",
        )
        d = result.model_dump()
        assert d["is_informative"] is False
        assert d["severity"] == "little or no damage"


class TestEventSummarySchema:

    def test_summary_excludes_text_field(self):
        """EventSummary should not have a text field — keeps list payloads small."""
        assert not hasattr(EventSummary.model_fields, "text"), \
            "EventSummary should not include the text field"

    def test_summary_has_location_fields(self):
        """Summary must include coordinates and location_name for map rendering."""
        fields = EventSummary.model_fields
        assert "latitude"      in fields
        assert "longitude"     in fields
        assert "location_name" in fields
        assert "severity"      in fields
        assert "humanitarian"  in fields