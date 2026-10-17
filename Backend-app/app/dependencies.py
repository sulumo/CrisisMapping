"""
dependencies.py
───────────────
Shared FastAPI dependencies — injected into route functions via Depends().

What is a dependency?
  A dependency is any callable that FastAPI runs before your route handler.
  FastAPI reads the type annotation, calls the function, and passes the
  result as the argument.  This keeps boilerplate (like opening/closing a
  DB session) out of every route.

  Example:
      @app.get("/events")
      def get_events(db: Session = Depends(get_db)):
          ...

  FastAPI calls get_db(), gets the session, injects it, and runs the finally
  block automatically when the request finishes — even if an exception occurs.

Why a separate file?
  Both routers (events.py, simulation.py) import get_db.  Putting it here
  avoids circular imports and makes it trivial to override in tests:

      app.dependency_overrides[get_db] = get_test_db
"""
from sqlalchemy.orm import Session
from .database import SessionLocal


def get_db() -> Session:
    """
    Yield a database session for the duration of a single request,
    then close it automatically in the finally block.

    Using `yield` makes this a FastAPI "dependency with cleanup" —
    code before yield runs before the route handler, code after yield
    (the finally) runs after the response is sent.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()