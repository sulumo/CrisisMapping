"""
tests/test_events.py
────────────────────
Tests for POST /events and GET /events.

Test naming convention: test_<what>_<expected_outcome>
  e.g. test_create_event_returns_201 — clear what is being tested and what
  should happen, so a failure message is immediately informative.

Each test is independent — it does not rely on another test having run first.
The `client` fixture provides a fresh TestClient with mocked AI and SQLite DB.
"""
import io
import pytest
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def make_image_file(name="test.jpg") -> tuple:
    """Return a (name, fileobj, content_type) tuple for multipart upload."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 149, 237)).save(buf, format="JPEG")
    buf.seek(0)
    return (name, buf, "image/jpeg")


# ─────────────────────────────────────────────────────────────────────────────
# POST /events — happy path
# ─────────────────────────────────────────────────────────────────────────────
class TestCreateEvent:

    def test_returns_201_on_valid_submission(self, client, event_form_data):
        """A valid report with image + form fields returns 201 with full event data."""
        response = client.post(
            "/events",
            data=event_form_data,
            files={"files": make_image_file()},
        )
        assert response.status_code == 201

    def test_response_contains_ai_fields(self, client, event_form_data):
        """The response includes the AI classification result fields."""
        response = client.post(
            "/events",
            data=event_form_data,
            files={"files": make_image_file()},
        )
        body = response.json()
        assert "is_informative" in body
        assert "severity"       in body
        assert "humanitarian"   in body

    def test_ai_result_matches_mock(self, client, event_form_data):
        """The mock AI returns known values — verify they are in the response."""
        response = client.post(
            "/events",
            data=event_form_data,
            files={"files": make_image_file()},
        )
        body = response.json()
        assert body["is_informative"] is True
        assert body["severity"]       == "severe damage"
        assert body["humanitarian"]   == "infrastructure"

    def test_image_urls_is_a_list(self, client, event_form_data):
        """image_urls should be a list (the schema splits the stored string)."""
        response = client.post(
            "/events",
            data=event_form_data,
            files={"files": make_image_file("flood.jpg")},
        )
        body = response.json()
        assert isinstance(body.get("image_urls"), list)
        assert len(body["image_urls"]) >= 1

    def test_non_informative_returns_200_with_message(
        self, client_non_informative, event_form_data
    ):
        """
        Non-informative events return 200 (not 201) with a descriptive message.
        The event is saved but will not appear on the map.
        """
        response = client_non_informative.post(
            "/events",
            data=event_form_data,
            files={"files": make_image_file()},
        )
        assert response.status_code == 200
        body = response.json()
        assert "message"  in body
        assert "event_id" in body
        assert "not informative" in body["message"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# POST /events — validation failures (422)
# ─────────────────────────────────────────────────────────────────────────────
class TestCreateEventValidation:

    def test_missing_image_returns_422(self, client, event_form_data):
        """Submitting no file should be rejected."""
        response = client.post("/events", data=event_form_data)
        assert response.status_code == 422

    def test_missing_text_returns_422(self, client, event_form_data):
        """text is a required Form field."""
        data = {k: v for k, v in event_form_data.items() if k != "text"}
        response = client.post(
            "/events", data=data, files={"files": make_image_file()}
        )
        assert response.status_code == 422

    def test_invalid_latitude_returns_422(self, client, event_form_data):
        """Latitude outside [-90, 90] should be rejected."""
        data = {**event_form_data, "latitude": "999"}
        response = client.post(
            "/events", data=data, files={"files": make_image_file()}
        )
        assert response.status_code == 422

    def test_invalid_longitude_returns_422(self, client, event_form_data):
        """Longitude outside [-180, 180] should be rejected."""
        data = {**event_form_data, "longitude": "-999"}
        response = client.post(
            "/events", data=data, files={"files": make_image_file()}
        )
        assert response.status_code == 422

    def test_non_image_file_returns_415(self, client, event_form_data):
        """Uploading a PDF should be rejected with 415."""
        pdf_bytes = io.BytesIO(b"%PDF-1.4 fake content")
        response = client.post(
            "/events",
            data=event_form_data,
            files={"files": ("report.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 415


# ─────────────────────────────────────────────────────────────────────────────
# GET /events
# ─────────────────────────────────────────────────────────────────────────────
class TestGetEvents:

    def test_returns_list(self, client):
        """GET /events always returns a list (may be empty)."""
        response = client.get("/events")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_default_limit_respected(self, client):
        """Response never exceeds the default limit of 100."""
        response = client.get("/events")
        assert len(response.json()) <= 100

    def test_limit_above_500_rejected(self, client):
        """limit > 500 should be rejected by Pydantic query validation."""
        response = client.get("/events?limit=600")
        assert response.status_code == 422

    def test_limit_zero_rejected(self, client):
        """limit=0 violates ge=1 constraint."""
        response = client.get("/events?limit=0")
        assert response.status_code == 422

    def test_partial_geo_params_rejected(self, client):
        """Providing lat without lon and radius_km should return 422."""
        response = client.get("/events?lat=29.76")
        assert response.status_code == 422

    def test_created_event_appears_in_list(self, client, event_form_data):
        """An event submitted via POST should be retrievable via GET."""
        # Create
        post_resp = client.post(
            "/events",
            data=event_form_data,
            files={"files": make_image_file("visible.jpg")},
        )
        assert post_resp.status_code == 201

        # Retrieve
        get_resp = client.get("/events")
        assert get_resp.status_code == 200
        ids = [e["id"] for e in get_resp.json()]
        assert post_resp.json()["id"] in ids

    def test_non_informative_event_excluded_from_list(
        self, client_non_informative, event_form_data
    ):
        """Events classified as non-informative must not appear in GET /events."""
        post_resp = client_non_informative.post(
            "/events",
            data=event_form_data,
            files={"files": make_image_file("hidden.jpg")},
        )
        assert post_resp.status_code == 200   # non-informative path

        get_resp = client_non_informative.get("/events")
        ids = [e["id"] for e in get_resp.json()]
        assert post_resp.json()["event_id"] not in ids

    def test_search_filter_returns_matching_events(self, client, event_form_data):
        """Keyword search should match events containing the search term."""
        # Create a uniquely worded event
        data = {**event_form_data, "text": "UNIQUEFLOODKEYWORD bridge destroyed"}
        client.post("/events", data=data, files={"files": make_image_file("f.jpg")})

        # Search for it
        resp = client.get("/events?search=UNIQUEFLOODKEYWORD")
        assert resp.status_code == 200
        results = resp.json()
        assert any("UNIQUEFLOODKEYWORD" in e.get("text", "") for e in results)

    def test_pagination_offset(self, client, event_form_data):
        """offset shifts the result window."""
        resp_0 = client.get("/events?limit=1&offset=0")
        resp_1 = client.get("/events?limit=1&offset=1")
        assert resp_0.status_code == 200
        assert resp_1.status_code == 200
        # If there are at least 2 events, the two responses should differ
        if resp_0.json() and resp_1.json():
            assert resp_0.json()[0]["id"] != resp_1.json()[0]["id"]