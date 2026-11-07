"""
Backward-compatibility facade for backend.app.
Allows existing scripts and test harnesses to import `create_app` directly from `backend.app`.
"""
from backend.app.factory import create_app
from backend.api.search_routes import _run_stored_search_and_finalize

__all__ = ["create_app", "_run_stored_search_and_finalize"]
