"""
API Routes Package for AI Vision Search Platform.
Provides modular blueprints for camera, search, history, drive, system, and media endpoints.
"""
from flask import current_app


def get_engine():
    return current_app.config.get("ENGINE")


def get_conn():
    return current_app.config.get("DB_CONN")


def get_matcher():
    return current_app.config.get("MATCHER")

