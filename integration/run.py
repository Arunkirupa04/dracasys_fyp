"""
Entry point for the DracaSys integration backend.

Usage:
    python run.py

Runs on INTEGRATION_HOST:INTEGRATION_PORT (default 0.0.0.0:5000), configurable
via env vars. Module 1 must be started separately on its own port (default
8000, see module1/run.py) — this backend calls it over HTTP and does not
manage its lifecycle.
"""
import uvicorn

from backend.config import INTEGRATION_HOST, INTEGRATION_PORT

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=INTEGRATION_HOST, port=INTEGRATION_PORT, reload=False)
