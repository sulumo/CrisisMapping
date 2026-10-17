"""
routers/websocket.py
────────────────────
WebSocket endpoint and connection manager.

Why a ConnectionManager class?
  WebSocket connections are long-lived — a client connects once and stays
  connected, receiving pushes whenever new events arrive.  We need a central
  registry of all open connections so the simulation engine (and future
  event-creation logic) can broadcast to every connected frontend tab.

  Putting that registry in a class:
    - makes the broadcast logic reusable from any part of the app
    - prevents the list from being accidentally cleared on module reload
    - is the pattern recommended in FastAPI's official WebSocket docs

How the frontend uses this:
  1. On page load, Next.js opens: ws://your-server/ws/events
  2. The backend adds the socket to active_connections
  3. When a new event arrives, broadcast() sends JSON to all open sockets
  4. If the user closes the tab, WebSocketDisconnect is raised and we clean up

The manager instance is exported so simulation.py can call
  await manager.broadcast(payload)
without importing the whole router.
"""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """
    Keeps track of every open WebSocket connection and provides a
    broadcast helper that safely handles dead connections.
    """

    def __init__(self):
        # A plain list is fine here — WebSocket connections are created and
        # removed on the main async event loop, so there are no threading races.
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept the handshake and register the connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a connection from the registry."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, payload: dict):
        """
        Send a JSON payload to every connected client.

        Dead connections (browser tab closed without a clean disconnect)
        raise an exception when we try to write to them.  We collect those
        and remove them after the loop — never modify a list while iterating it.
        """
        dead: list[WebSocket] = []

        for connection in self.active_connections:
            try:
                await connection.send_json(payload)
            except Exception as e:
                logger.warning(f"Dead WebSocket detected, queued for removal: {e}")
                dead.append(connection)

        for conn in dead:
            self.disconnect(conn)


# Module-level singleton — imported by simulation.py to call broadcast()
manager = ConnectionManager()


@router.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket connection for real-time event updates.

    The client connects once; we keep the socket open by waiting for
    incoming messages in a loop.  The frontend does not need to send
    anything — the receive_text() call just keeps the coroutine alive.
    If the client disconnects, WebSocketDisconnect is raised and we clean up.
    """
    await manager.connect(websocket)
    try:
        while True:
            # We don't act on incoming messages, but receive_text() is required
            # to detect client disconnection (the exception it raises).
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)