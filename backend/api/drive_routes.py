"""
Google Drive Cloud Synchronization Routes.
"""
import threading
from flask import Blueprint, jsonify
from backend.gdrive import gdrive_storage
from . import get_conn, get_matcher

drive_bp = Blueprint("drive_api", __name__)


@drive_bp.route("/api/query/<query_id>/sync_drive", methods=["POST"])
def sync_query_drive_endpoint(query_id):
    """Syncs history records and files for a query to Google Drive in a background thread."""
    conn = get_conn()
    if not gdrive_storage.is_configured():
        return jsonify({"ok": False, "error": "Google Drive credentials not configured"}), 400

    threading.Thread(
        target=gdrive_storage.sync_query_history_to_drive,
        args=(conn, query_id),
        daemon=True,
        name=f"gdrive-sync-{query_id}",
    ).start()
    return jsonify({
        "ok": True,
        "message": f"Google Drive sync for query #{query_id} dispatched in background.",
        "query_id": query_id,
    })


@drive_bp.route("/api/history/sync_drive", methods=["POST"])
def sync_all_history_drive_endpoint():
    """Syncs all queries and detection history into Google Drive in a background thread."""
    conn = get_conn()
    if not gdrive_storage.is_configured():
        return jsonify({"ok": False, "error": "Google Drive credentials not configured"}), 400

    threading.Thread(
        target=gdrive_storage.sync_all_history_to_drive,
        args=(conn,),
        daemon=True,
        name="gdrive-sync-all",
    ).start()
    return jsonify({
        "ok": True,
        "message": "Full Google Drive history sync dispatched in background.",
    })


@drive_bp.route("/api/gdrive/sync_recordings", methods=["POST"])
def sync_drive_recordings():
    """Scans Google Drive across all dates, registers cloud recordings in SQLite, and triggers indexer."""
    conn = get_conn()
    matcher = get_matcher()
    if not gdrive_storage.is_configured():
        return jsonify({"ok": False, "error": "Google Drive is not configured"}), 400
    try:
        res = gdrive_storage.sync_all_drive_recordings_to_db(conn)
        if hasattr(matcher, "_scan_unprocessed_recordings"):
            matcher._scan_unprocessed_recordings()
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

