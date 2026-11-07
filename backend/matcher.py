"""
Backward-compatibility facade for backend.services.matcher.LiveMatcher.
Allows existing code and test harnesses to import `LiveMatcher` directly from `backend.matcher`.
"""
from backend.services.matcher import LiveMatcher

__all__ = ["LiveMatcher"]
