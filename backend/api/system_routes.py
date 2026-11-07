"""
System Health Telemetry, Queue Inspection, and Performance Audit Routes.
"""
import os
from flask import Blueprint, jsonify
from backend.database import _db_lock
from backend.gdrive import gdrive_storage
from . import get_conn, get_matcher

system_bp = Blueprint("system_api", __name__)


@system_bp.route("/api/system/status")
def api_system_status():
    conn = get_conn()
    matcher = get_matcher()

    with _db_lock:
        total_queries = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        total_live_matches = conn.execute("SELECT COUNT(*) FROM detections WHERE source='live'").fetchone()[0]
        total_stored_matches = conn.execute("SELECT COUNT(*) FROM detections WHERE source='stored'").fetchone()[0]

        # Searches breakdown (Persons vs Vehicles)
        queries = conn.execute("SELECT target_summary FROM queries").fetchall()
        person_searches = 0
        vehicle_searches = 0
        for q in queries:
            summary = (q["target_summary"] or "").lower()
            for part in summary.split(","):
                if ":" in part:
                    k, v = part.strip().split(":", 1)
                    try:
                        cnt = int(v)
                    except ValueError:
                        cnt = 1
                    if k == "person":
                        person_searches += cnt
                    elif k in ("car", "motorcycle", "bicycle", "vehicle"):
                        vehicle_searches += cnt
                else:
                    if "person" in part:
                        person_searches += 1
                    elif any(veh in part for veh in ("car", "motorcycle", "bicycle", "vehicle")):
                        vehicle_searches += 1

        # Findings breakdown (Persons vs Vehicles)
        person_findings = conn.execute("SELECT COUNT(*) FROM detections WHERE target_type='person'").fetchone()[0]
        vehicle_findings = conn.execute("SELECT COUNT(*) FROM detections WHERE target_type IN ('car', 'motorcycle', 'bicycle')").fetchone()[0]

        live_person_findings = conn.execute("SELECT COUNT(*) FROM detections WHERE source='live' AND target_type='person'").fetchone()[0]
        live_vehicle_findings = conn.execute("SELECT COUNT(*) FROM detections WHERE source='live' AND target_type IN ('car', 'motorcycle', 'bicycle')").fetchone()[0]

        stored_person_findings = conn.execute("SELECT COUNT(*) FROM detections WHERE source='stored' AND target_type='person'").fetchone()[0]
        stored_vehicle_findings = conn.execute("SELECT COUNT(*) FROM detections WHERE source='stored' AND target_type IN ('car', 'motorcycle', 'bicycle')").fetchone()[0]

        try:
            total_recordings = conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
            uploaded_recordings = conn.execute("SELECT COUNT(*) FROM recordings WHERE upload_status='uploaded'").fetchone()[0]
        except Exception:
            total_recordings = 0
            uploaded_recordings = 0

    type_counts = matcher.index_manager.type_counts() if hasattr(matcher, "index_manager") and matcher.index_manager else {}
    person_vectors = type_counts.get("person", 0)
    vehicle_vectors = sum(v for k, v in type_counts.items() if k in ("car", "motorcycle", "bicycle"))

    rec_path = matcher.recorder.current_recording_path() if hasattr(matcher, "recorder") and matcher.recorder else None

    return jsonify({
        "camera_running": matcher.is_running(),
        "camera_error": getattr(matcher, "camera_error", None),
        "recording_active": matcher.recorder.is_recording() if hasattr(matcher, "recorder") and matcher.recorder else False,
        "current_recording": os.path.basename(rec_path) if rec_path else None,
        "indexed_vectors": matcher.index_manager.total() if hasattr(matcher, "index_manager") and matcher.index_manager else 0,
        "type_counts": type_counts,
        "person_vectors": person_vectors,
        "vehicle_vectors": vehicle_vectors,
        "active_live_searches": len(matcher.active_query_ids()),
        "total_queries": total_queries,
        "person_searches": person_searches,
        "vehicle_searches": vehicle_searches,
        "total_matches": total_live_matches + total_stored_matches,
        "person_findings": person_findings,
        "vehicle_findings": vehicle_findings,
        "total_live_matches": total_live_matches,
        "live_person_findings": live_person_findings,
        "live_vehicle_findings": live_vehicle_findings,
        "total_stored_matches": total_stored_matches,
        "stored_person_findings": stored_person_findings,
        "stored_vehicle_findings": stored_vehicle_findings,
        "total_recordings": total_recordings,
        "uploaded_recordings": uploaded_recordings,
        "gdrive_url": (
            f"https://drive.google.com/drive/u/2/folders/{gdrive_storage.root_folder_id}"
            if (hasattr(gdrive_storage, "root_folder_id") and gdrive_storage.root_folder_id and gdrive_storage.root_folder_id != "root")
            else "https://drive.google.com/drive/u/2/my-drive"
        ),
    })


@system_bp.route("/api/system/performance")
def api_system_performance():
    """Real-time performance audit metrics: FPS, queues, latencies, CPU/RAM/Swap/IO."""
    matcher = get_matcher()
    if hasattr(matcher, "get_performance_stats"):
        return jsonify(matcher.get_performance_stats())
    return jsonify({"error": "performance stats unavailable"}), 500

