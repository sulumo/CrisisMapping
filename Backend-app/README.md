# CrisisMapping Backend API

FastAPI backend for the **CrisisMapping** application. It accepts user-submitted disaster reports, runs a trained CLIP-based multimodal classifier over the report text and an image, stores the event and its geolocation in PostgreSQL/PostGIS, and exposes REST and WebSocket interfaces for map clients. It also includes a mock-event simulation endpoint for demonstrating real-time map updates.

This README documents the code in this repository archive. Before deployment, review the configuration notes and known limitations near the end; several deployment details need to be aligned with the files in the repository.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [AI classification](#ai-classification)
- [API reference](#api-reference)
- [Event data model](#event-data-model)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Run with Docker Compose](#run-with-docker-compose)
- [Run locally](#run-locally)
- [Tests](#tests)
- [Repository structure](#repository-structure)
- [Operational notes and known limitations](#operational-notes-and-known-limitations)

## Features

- **Multimodal inference:** classifies report text and an image into informativeness, humanitarian category, and damage severity.
- **Event persistence:** stores report text, location, uploaded image paths, AI predictions, and timestamps in PostgreSQL.
- **Spatial queries:** uses PostGIS geometry to support radius searches by geographic coordinates.
- **Search and filtering:** queries informative reports by keyword, disaster type, humanitarian category, severity, date range, and distance.
- **Live map updates:** broadcasts simulated events to connected clients over WebSocket.
- **API documentation:** FastAPI provides interactive Swagger UI and OpenAPI documentation.
- **Automated tests:** tests cover event routes, schema validation, the simulation endpoints, and AI-service behavior. The route tests mock inference to avoid loading the real model.

## Architecture

```text
Browser / map frontend
       |
       | HTTP multipart/form-data / JSON responses
       v
FastAPI application (app/main.py)
       |--------------------|
       v                    v
Event routes             Simulation routes
       |                    |
       v                    v
AI service              Background simulation
       |                    |
       v                    +----> WebSocket broadcast (/ws/events)
CLIP-based model                    |
       |                            v
       +----> classification     Connected map clients
       |
       v
PostgreSQL + PostGIS
       |
       +----> filtered event results / radius search
```

### Main components

| Component                                 | Responsibility                                                                                                                                                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/main.py`                             | Creates the FastAPI app, configures CORS and static files, registers exception handlers and routers, loads the model during application startup, and creates database tables. |
| `app/routers/events.py`                   | Accepts reports, saves images, calls AI inference, persists events, and provides filtering/search endpoints.                                                                  |
| `app/routers/simulation.py`               | Starts a mock-event replay in a background task and returns recent informative events.                                                                                        |
| `app/routers/websocket.py`                | Maintains active WebSocket connections and broadcasts JSON event messages.                                                                                                    |
| `app/ai_core/service.py`                  | Selects the model architecture, loads the checkpoint and CLIP processor, and converts model logits into label names.                                                          |
| `app/ai_core/architectures/baseline.py`   | Baseline model using the final pooled CLIP visual representation.                                                                                                             |
| `app/ai_core/architectures/multilayer.py` | Task-conditioned gated visual fusion using early, middle, and final visual representations.                                                                                   |
| `app/schemas.py`                          | Pydantic input/output schemas and response serialization.                                                                                                                     |
| `app/models.py`                           | SQLAlchemy `CrisisEvent` database model.                                                                                                                                      |
| `app/database.py`                         | Database engine, session factory, and declarative base.                                                                                                                       |
| `app/dependencies.py`                     | Per-request database session dependency.                                                                                                                                      |
| `app/exceptions.py`                       | Structured JSON exception handlers.                                                                                                                                           |

## AI classification

The service uses `openai/clip-vit-base-patch32` as its pretrained image/text backbone and loads one of the trained CrisisMapping model variants. CLIP's image and text encoders are frozen in both architecture definitions; the task-specific projection, fusion, gating, and classification layers are part of the checkpoint.

### Supported variants

Set `MODEL_VARIANT` to one of the following exact values:

| Value        | Architecture                                                                                                                         | Expected checkpoint                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `baseline`   | `CLIP_CrisisKAN_Baseline`; final pooled image representation                                                                         | Checkpoint trained with the baseline architecture    |
| `multilayer` | `CLIP_CrisisKAN_TCG`; task-conditioned gates combine visual features from hidden states 3 and 7 plus the final pooled representation | Checkpoint trained with the multi-layer architecture |

`MODEL_WEIGHTS_PATH` selects the checkpoint file. The checkpoint architecture must match `MODEL_VARIANT`; loading an incompatible checkpoint should fail because the service uses strict state-dictionary loading.

At startup, the service loads the architecture and checkpoint, moves the model to CUDA when available (otherwise CPU), switches it to evaluation mode, and loads `CLIPProcessor`. The processor handles CLIP-compatible image normalization and text tokenization. Text is truncated to a maximum length of 77 tokens. Inference runs without gradient tracking.

### Output labels

The service maps the model's three output heads to these values:

| Output                          | Labels returned by the API                                                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Informativeness (`task1`)       | `not informative`, `informative` (returned as the Boolean `is_informative`)                                                         |
| Humanitarian category (`task2`) | `infrastructure`, `not humanitarian`, `other relevant information`, `rescue volunteering`, `vehicle damage`, `affected individuals` |
| Damage severity (`task3`)       | `little or no damage`, `mild damage`, `severe damage`                                                                               |

These label lists must remain consistent with the label-to-index mappings used during model training. If the training dataset's label mapping changes, update the mapping in `app/ai_core/service.py` and verify the checkpoint's output-head ordering.

### Inference behavior

For a submitted report with multiple images, **only the first image is passed to the model**. All accepted images are saved and their URLs are returned/stored. The model produces all three task predictions, but the event is only included in `GET /events` and `/api/active-events` when `is_informative` is true.

## API reference

The API has no global `/api` prefix. Routes are registered at the paths shown below.

Interactive documentation after startup:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### `GET /`

There is no explicit root (`/`) handler in the current code. Use `/docs` or the endpoints below to check the service.

### `POST /events` — submit a crisis report

Content type: `multipart/form-data`.

| Field           | Type           | Required | Description                                                                                                                                                                                                             |
| --------------- | -------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `files`         | file, repeated | Yes      | One or more image files. The first image is used for inference.                                                                                                                                                         |
| `text`          | string         | Yes      | Text describing the event.                                                                                                                                                                                              |
| `location_name` | string         | Yes      | Human-readable location.                                                                                                                                                                                                |
| `latitude`      | float          | Yes      | Latitude from `-90` to `90`.                                                                                                                                                                                            |
| `longitude`     | float          | Yes      | Longitude from `-180` to `180`.                                                                                                                                                                                         |
| `type`          | string         | No       | Disaster type; defaults to `Unknown`. The schema documents `Flood`, `Fire`, `Earthquake`, `Storm`, `Landslide`, `Hurricane`, `Tsunami`, and `Unknown`, though this route currently accepts the field as a plain string. |

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/events" \
  -F "text=Flood water has covered the main road" \
  -F "location_name=Example City" \
  -F "latitude=37.5665" \
  -F "longitude=126.9780" \
  -F "type=Flood" \
  -F "files=@./example.jpg;type=image/jpeg"
```

For multiple images, repeat the `-F "files=@..."` option for each image.

**Informative report:** returns an event object with fields such as `id`, `created_at`, `text`, `location_name`, `latitude`, `longitude`, `severity`, `humanitarian`, `type`, `is_informative`, and `image_urls`.

**Non-informative report:** the report is still saved in the database, but the endpoint returns a message and `event_id`; it is excluded from the map-query endpoints. The implementation returns HTTP `200` for this branch, despite the route's default declared status of `201` for informative submissions.

Validation includes latitude/longitude bounds and a basic check that each uploaded file's content type begins with `image/`. The route returns structured errors for invalid requests. This content-type check is not a substitute for validating file contents.

### `GET /events` — query map events

Returns a paginated list of **informative events only**, ordered by creation time descending. Available query parameters:

| Parameter      | Type              | Description                                                       |
| -------------- | ----------------- | ----------------------------------------------------------------- |
| `search`       | string            | Case-insensitive keyword search in report text or location name.  |
| `type`         | string            | Exact disaster-type filter.                                       |
| `humanitarian` | string            | Exact humanitarian-category filter.                               |
| `severity`     | string            | Exact damage-severity filter.                                     |
| `start_date`   | ISO 8601 datetime | Include events created at or after this time.                     |
| `end_date`     | ISO 8601 datetime | Include events created at or before this time.                    |
| `lat`          | float             | Centre latitude for a spatial radius search.                      |
| `lon`          | float             | Centre longitude for a spatial radius search.                     |
| `radius_km`    | positive float    | Radius in kilometres; must be supplied with `lat` and `lon`.      |
| `limit`        | integer           | Number of results; defaults to `100`, minimum `1`, maximum `500`. |
| `offset`       | integer           | Number of results to skip; defaults to `0`.                       |

Examples:

```bash
# Latest informative events
curl "http://localhost:8000/events"

# Search by keyword
curl "http://localhost:8000/events?search=flood"

# Filter by severity and type
curl "http://localhost:8000/events?severity=severe%20damage&type=Flood"

# Events within 10 km of a coordinate
curl "http://localhost:8000/events?lat=37.5665&lon=126.9780&radius_km=10"

# Pagination
curl "http://localhost:8000/events?limit=20&offset=20"
```

The spatial filter uses PostGIS `ST_DWithin` on geography-cast values, with the radius converted from kilometres to metres. Supply all three spatial parameters together; partial spatial parameters return HTTP `422`.

### `POST /api/start-simulation` — start mock event replay

Starts a FastAPI background task and immediately returns a message. The task reads mock events from `app/data/simulation.json`, inserts each into the database as informative, broadcasts each event over WebSocket, and waits four seconds before processing the next item.

```bash
curl -X POST "http://localhost:8000/api/start-simulation"
```

**Repository requirement:** `app/data/simulation.json` is referenced by the code but is not present in the supplied ZIP archive. Add the file before expecting the simulation to produce events, or change the route to point to the intended data source. If the file is absent, the task logs an error and exits.

### `GET /api/active-events` — get recent map events

Returns up to the 20 most recent informative events from the database. This is useful for initializing a map before it receives live WebSocket messages.

```bash
curl "http://localhost:8000/api/active-events"
```

### `WS /ws/events` — live event stream

Connect a frontend client to:

```text
ws://localhost:8000/ws/events
```

Use `wss://` when the API is served over HTTPS through a correctly configured reverse proxy. The server accepts the connection and broadcasts JSON payloads to all connected clients when the simulation emits events. The current WebSocket endpoint does not send an initial database snapshot; use `GET /api/active-events` for that purpose. It also does not require clients to send application messages; the server waits for incoming text to detect disconnection.

A broadcast message has this general shape:

```json
{
  "type": "NEW_EVENT",
  "data": {
    "id": 123,
    "type": "Flood",
    "source": "Example City",
    "lat": 37.5665,
    "lon": 126.978,
    "text": "Flood water has covered the main road"
  }
}
```

The actual `data` object is based on the corresponding record in `simulation.json`, plus the database ID. Fields depend on the simulation data file.

## Event data model

The SQLAlchemy model maps to the `crisis_events` table.

| Field                   | Meaning                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| `id`                    | Integer primary key.                                                                             |
| `created_at`            | Timestamp generated by the database.                                                             |
| `text`                  | Submitted report text.                                                                           |
| `location_name`         | Human-readable location.                                                                         |
| `image_url`             | Comma-separated list of relative static-image URLs in the database.                              |
| `latitude`, `longitude` | Numeric coordinates.                                                                             |
| `geom`                  | PostGIS `POINT` geometry using SRID 4326; coordinates are stored as `POINT(longitude latitude)`. |
| `is_informative`        | Boolean prediction from Task 1.                                                                  |
| `humanitarian`          | Task 2 prediction label.                                                                         |
| `severity`              | Task 3 prediction label.                                                                         |
| `type`                  | Disaster type supplied by the client or simulation source.                                       |

The response schemas expose stored image paths as an `image_urls` array rather than the comma-separated database string. Static images are served under `/static/`, so an uploaded image path such as `/static/image/example.jpg` is available from the API host at that path.

## Requirements

The project pins its main runtime dependencies in `requirements.txt`, including:

- Python compatible with the pinned package versions (Python 3.10 is a sensible starting point for the listed PyTorch 2.1.2 stack).
- FastAPI `0.111.0` and Uvicorn `0.29.0`.
- PyTorch `2.1.2` and torchvision `0.16.2`.
- Transformers `4.40.0` and Pillow `10.3.0`.
- SQLAlchemy `2.0.30`, GeoAlchemy2 `0.15.1`, and psycopg2-binary `2.9.9`.
- PostgreSQL with the PostGIS extension. The Compose configuration uses `postgis/postgis:15-3.3`.
- A trained checkpoint matching the selected architecture. The CLIP model and processor may be downloaded from Hugging Face at startup if they are not cached locally.

CUDA is optional in the Python service code: it selects CUDA when `torch.cuda.is_available()` is true, otherwise CPU. Docker Compose requests an NVIDIA GPU, so GPU-enabled deployment also requires a compatible NVIDIA driver and NVIDIA Container Toolkit. For CPU-only deployment, remove or disable the GPU reservation in `docker-compose.yml`.

## Configuration

The main variables are:

| Variable                   | Purpose                                                                                 |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `POSTGRES_USER`            | PostgreSQL username                                                                     |
| `POSTGRES_PASSWORD`        | PostgreSQL password                                                                     |
| `POSTGRES_DB`              | Database name                                                                           |
| `DB_HOST`                  | Database hostname                                                                       |
| `DB_PORT`                  | Database port                                                                           |
| `PGADMIN_DEFAULT_EMAIL`    | pgAdmin login email                                                                     |
| `PGADMIN_DEFAULT_PASSWORD` | pgAdmin login password                                                                  |
| `MODEL_VARIANT`            | Model architecture, exactly `baseline` or `multilayer`                                  |
| `MODEL_WEIGHTS_PATH`       | Path to the trained checkpoint, relative to the app's working directory unless absolute |

### Model checkpoint setup

Place the trained checkpoint at the path configured by `MODEL_WEIGHTS_PATH`, and ensure that `MODEL_VARIANT` matches the checkpoint architecture. The provided Compose file mounts `./app/ai_core/best.pt` into the container at `/code/app/ai_core/best.pt`, so that file must exist on the host for the default Compose configuration.

The service currently logs a warning rather than stopping startup if the checkpoint file is missing. In that case the model remains randomly initialized and any predictions should be considered invalid. For a real deployment, verify the checkpoint is present and loaded successfully before accepting user reports.

## Run with Docker Compose

1. Install Docker Engine or Docker Desktop and the Docker Compose plugin. If using GPU acceleration, configure the NVIDIA Container Toolkit.
2. Place the correct checkpoint at `app/ai_core/best.pt` or update the Compose volume and `MODEL_WEIGHTS_PATH` together.
3. Verify `MODEL_VARIANT` is set to `baseline` or `multilayer` and matches the checkpoint.
4. Build and start the stack:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up --build -d
```

View logs:

```bash
docker compose logs -f backend
```

Stop the services:

```bash
docker compose down
```

The Compose file defines:

| Service   | Purpose                    | Host port |
| --------- | -------------------------- | --------: |
| `backend` | FastAPI application        |    `8000` |
| `db`      | PostgreSQL/PostGIS         |    `5432` |
| `pgadmin` | Database administration UI |    `5050` |

The PostgreSQL data is stored in the named Docker volume `postgres_data`, so it persists when containers are stopped. To delete the database volume as well, use `docker compose down -v` (destructive: this removes persisted database data).

After startup, visit `http://localhost:8000/docs`.

## Run locally

Local execution requires a reachable PostgreSQL/PostGIS instance and the configured model checkpoint.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the environment variables in your shell or load `.env` using your preferred environment-management tool. The application does not automatically load `.env` by itself. When running outside Docker, set `DB_HOST=localhost` if PostgreSQL is running on the same machine. Ensure the database exists and PostGIS is available.

Start the API from the repository root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` is intended for development, not production. The model is loaded during FastAPI's startup lifespan, so the first startup can take longer while model assets are loaded or downloaded.

## Tests

The tests use FastAPI's `TestClient`, an SQLite test database, and a mocked AI prediction method to exercise API behavior without requiring real model inference. They cover event creation, validation, filtering/pagination, simulation endpoints, and AI-service behavior.

The supplied `pytest.ini` currently sets `testpaths = tests`, but the archive contains a directory named `test/` (singular). Align the configuration with the actual directory or rename the directory before running tests. Also ensure pytest and any test-only dependencies are installed; they are not listed in the supplied `requirements.txt`.

After correcting the test discovery configuration and installing test dependencies, run:

```bash
pytest
```

These tests validate application behavior with mocked inference; they do not evaluate the trained model's predictive accuracy.

## Repository structure

```text
Backend-app/
├── app/
│   ├── main.py
│   ├── database.py
│   ├── dependencies.py
│   ├── exceptions.py
│   ├── models.py
│   ├── schemas.py
│   ├── routers/
│   │   ├── events.py
│   │   ├── simulation.py
│   │   └── websocket.py
│   ├── ai_core/
│   │   ├── service.py
│   │   └── architectures/
│   │       ├── baseline.py
│   │       └── multilayer.py
│   └── static/image/
├── test/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```

The simulation router also expects `app/data/simulation.json`, which was not included in the supplied archive. The trained checkpoint is also intentionally excluded by `.gitignore` and must be supplied separately.

## Operational notes and known limitations

The following points are important before exposing this service to real users or the public internet:

1. **Checkpoint validation:** the service currently continues startup when the weight file is missing and can leave the model randomly initialized. Fail startup or add a readiness check that verifies a valid checkpoint was loaded before serving predictions.
2. **Upload filename safety:** uploaded files are written using the client-provided filename. Production code should generate server-side filenames, normalize/validate image formats, enforce file-size and image-dimension limits, and prevent path traversal or accidental overwrite.
3. **Inference error handling:** `CrisisAIService.predict()` catches exceptions and returns an `Unknown`/false fallback dictionary. Because the route validates that dictionary successfully, some inference failures may be stored as if they were predictions rather than returned as HTTP `503`. Consider raising a dedicated exception when inference fails and handling it consistently.
4. **Database initialization:** `Base.metadata.create_all()` is called during module import. This is convenient for a prototype, but database migrations (for example, Alembic) are preferable for managed deployments and schema evolution.
5. **CORS origins:** allowed origins are hardcoded in `app/main.py`. Update the list for the deployed frontend and remove origins that are no longer needed. CORS is not authentication and does not protect the API from non-browser clients.
6. **No authentication or rate limiting:** the supplied API does not implement authentication, authorization, or rate limiting. Add suitable controls before public deployment, particularly because report creation consumes storage and GPU/CPU inference time.
7. **Static image access:** uploaded images are served publicly under `/static/`. Do not upload private or sensitive imagery unless public access is intended and appropriate.
8. **Simulation is a demo facility:** it inserts mock records directly into the database and marks them informative without running model inference. The simulation file is missing from the supplied archive and must be added for this feature to work.
9. **Single-process WebSocket state:** the connection manager stores sockets in an in-memory list. It works for a single application process, but connections and broadcasts are not shared across multiple workers or containers. A shared broker/pub-sub layer is needed for horizontally scaled deployments.
10. **Compose checkpoint mount:** the Compose file binds `./app/ai_core/best.pt` to the container. Ensure the source file exists and matches the selected model variant; otherwise, the mount or model initialization may fail or use an unintended checkpoint.
11. **Simulation data and tests:** `app/data/simulation.json` is referenced but absent from the archive. The `pytest.ini` test directory setting also differs from the actual `test/` directory name.
12. **Database access:** Compose publishes PostgreSQL and pgAdmin ports on the host. For a deployment, avoid exposing these ports publicly unless necessary and protect pgAdmin with strong credentials and network controls.
13. **Prediction interpretation:** the model outputs should be treated as decision-support signals, not verified ground truth. Reports should be reviewed by humans before they inform emergency response decisions.

## Attribution

This backend serves the CrisisMapping multimodal crisis-event classification project and its trained CLIP-based model variants. Keep the model's training/data provenance and any applicable CrisisMMD dataset terms documented in the main project repository.
