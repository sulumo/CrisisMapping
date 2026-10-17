"""
tests/test_simulation.py
────────────────────────
Tests for:
  POST /api/start-simulation
  GET  /api/active-events

These were the two broken endpoints in the original codebase.
The tests verify that both now behave correctly.
"""


class TestStartSimulation:

    def test_returns_200_immediately(self, client):
        """
        The simulation runs in the background — the endpoint should return
        200 immediately without waiting for the simulation to complete.
        """
        response = client.post("/api/start-simulation")
        assert response.status_code == 200

    def test_response_contains_message(self, client):
        """The response body should have a 'message' key."""
        response = client.post("/api/start-simulation")
        assert "message" in response.json()


class TestGetActiveEvents:

    def test_returns_list(self, client):
        """
        GET /api/active-events was previously broken (NameError on
        undefined `events_to_broadcast`).  It should now return a list.
        """
        response = client.get("/api/active-events")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_returns_at_most_20(self, client):
        """The endpoint is capped at 20 recent events."""
        response = client.get("/api/active-events")
        assert len(response.json()) <= 20

    def test_only_informative_events_returned(self, client, event_form_data):
        """Non-informative events must not appear in active-events."""
        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="JPEG")
        buf.seek(0)

        # Create an informative event
        client.post(
            "/events",
            data=event_form_data,
            files={"files": ("ev.jpg", buf, "image/jpeg")},
        )

        response = client.get("/api/active-events")
        for event in response.json():
            assert event["is_informative"] is True