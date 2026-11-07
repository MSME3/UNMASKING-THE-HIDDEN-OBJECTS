"""
Camera Control and Real-Time Video Streaming Routes.
"""
import os
import cv2
from flask import Blueprint, jsonify, Response
from backend.database import _db_lock
from . import get_engine, get_conn, get_matcher

camera_bp = Blueprint("camera_api", __name__)


@camera_bp.route("/camera/start", methods=["POST"])
def camera_start():
    matcher = get_matcher()
    engine = get_engine()
    conn = get_conn()

    ok, msg = matcher.start_camera()
    if ok:
        # Re-register all queries configured for live search
        try:
            with _db_lock:
                active_qs = conn.execute(
                    "SELECT query_id, image_path, mode, live_status FROM queries "
                    "WHERE mode IN ('live', 'both') AND live_status != 'disabled'"
                ).fetchall()
            for q_row in active_qs:
                qid = q_row["query_id"]
                with matcher.lock:
                    already_active = qid in matcher.active_queries
                if not already_active and q_row["image_path"] and os.path.exists(q_row["image_path"]):
                    q_img = cv2.imread(q_row["image_path"])
                    if q_img is not None:
                        q_targets = engine.extract_targets(q_img, is_query=True)
                        if q_targets:
                            matcher.add_query(qid, q_targets)
                            print(f"[camera/start] Re-activated live query #{qid} ({len(q_targets)} targets)")
        except Exception as e:
            print(f"[camera/start] Warning re-registering live queries: {e}")
    return jsonify({"ok": ok, "message": msg, "running": matcher.is_running()})


@camera_bp.route("/camera/stop", methods=["POST"])
def camera_stop():
    matcher = get_matcher()
    matcher.stop_camera()
    return jsonify({"ok": True, "running": matcher.is_running()})


@camera_bp.route("/api/camera/status")
def camera_status():
    matcher = get_matcher()
    rec_path = matcher.recorder.current_recording_path() if hasattr(matcher, "recorder") and matcher.recorder else None
    idx_total = matcher.index_manager.total() if hasattr(matcher, "index_manager") and matcher.index_manager else 0
    idx_counts = matcher.index_manager.type_counts() if hasattr(matcher, "index_manager") and matcher.index_manager else {}

    rec_err = matcher.recorder.get_last_error() if hasattr(matcher, "recorder") and matcher.recorder else None

    return jsonify({
        "running": matcher.is_running(),
        "source_type": getattr(matcher, "camera_source_type", None) if matcher.is_running() else None,
        "source_name": getattr(matcher, "camera_source_name", None) if matcher.is_running() else None,
        "error": getattr(matcher, "camera_error", None),
        "recording_error": rec_err,
        "recording": matcher.recorder.is_recording() if hasattr(matcher, "recorder") and matcher.recorder else False,
        "recording_file": os.path.basename(rec_path) if rec_path else None,
        "index_size": idx_total,
        "type_counts": idx_counts,
        "active_queries": len(matcher.active_query_ids()),
    })


@camera_bp.route("/video_feed")
def video_feed():
    matcher = get_matcher()

    def generate():
        last_seq = -1
        try:
            while matcher.is_running():
                last_seq, jpeg = matcher.get_latest_stream_jpeg(last_seq, timeout=0.5)
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n" +
                        jpeg + b"\r\n"
                    )
        except GeneratorExit:
            pass

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

