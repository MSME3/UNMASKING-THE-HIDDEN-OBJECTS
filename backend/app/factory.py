"""
Flask Application Factory for AI Vision Search Platform.
Assembles middleware, security headers, routing blueprints, and injected dependencies.
"""
import cv2
from flask import Flask, request, render_template

from backend.config import TEMPLATES_DIR, STATIC_DIR
from backend.config import TEMPLATES_DIR, STATIC_DIR, BASE_DIR
from backend.api.camera_routes import camera_bp
from backend.api.search_routes import search_bp
from backend.api.history_routes import history_bp
from backend.api.drive_routes import drive_bp
from backend.api.system_routes import system_bp
from backend.api.media_routes import media_bp


def create_app(engine, conn, matcher):
    """Factory creating and configuring the Flask application."""
    # Prevent OpenCV thread pool explosion across concurrent threads on Jetson ARM cores
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    app = Flask(
        "backend",
        root_path=BASE_DIR,
        static_url_path="/static",
        static_folder=STATIC_DIR,
        template_folder=TEMPLATES_DIR,
    )
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Inject shared services into application context
    app.config["ENGINE"] = engine
    app.config["DB_CONN"] = conn
    app.config["MATCHER"] = matcher

    @app.after_request
    def add_no_cache_headers(response):
        if request.path.startswith("/static/") or request.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # Web Dashboard Root Route
    @app.route("/")
    def index():
        return render_template("index.html")

    # Register API Blueprints
    app.register_blueprint(camera_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(drive_bp)
    app.register_blueprint(system_bp)
    app.register_blueprint(media_bp)

    return app

