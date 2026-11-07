"""
App package for AI Vision Search Platform.
Exports Flask application factory and extensions.
"""
from .factory import create_app

__all__ = ["create_app"]

