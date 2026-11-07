"""
Media Asset Delivery and HTTP Range Video Streaming Routes.
"""
import os
from flask import Blueprint, request, send_from_directory, Response
from backend.config import (
    QUERY_DIR,
    FRAMES_DIR,
    CLIPS_DIR,
    DATASET_DIR,
    LIVE_RECORDINGS_DIR,
)
from backend.gdrive import gdrive_storage

media_bp = Blueprint("media_api", __name__)


@media_bp.route("/media/query/<path:fname>")
def media_query(fname):
    base_fname = os.path.basename(fname)
    local_file = os.path.join(QUERY_DIR, base_fname)
    if os.path.exists(local_file):
        return send_from_directory(QUERY_DIR, base_fname)
    if os.path.exists(fname) and os.path.isfile(fname):
        return send_from_directory(os.path.dirname(fname), os.path.basename(fname))
    return "Query image not found", 404


def _create_fallback_evidence_frame(base_fname, dest_path):
    """Generates a high-contrast CCTV surveillance evidence frame on the fly."""
    import sqlite3, cv2, numpy as np
    from backend.config import DB_PATH
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            """SELECT id, query_id, target_type, video_name, detection_date, detection_time, 
                      timestamp_sec, similarity_score, plate_text, plate_status 
               FROM detections 
               WHERE frame_path LIKE ? OR clip_path LIKE ?
               ORDER BY id DESC LIMIT 1""",
            (f"%{base_fname}%", f"%{base_fname.replace('frame_', 'clip_')}%"),
        ).fetchone()
        conn.close()

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (24, 18, 14)  # Deep slate CCTV background

        # Grid lines
        for y in range(40, 360, 40):
            cv2.line(img, (0, y), (640, y), (35, 28, 22), 1)
        for x in range(40, 640, 40):
            cv2.line(img, (x, 0), (x, 360), (35, 28, 22), 1)

        cv2.rectangle(img, (2, 2), (637, 357), (70, 50, 35), 2)
        cv2.rectangle(img, (2, 2), (637, 45), (40, 30, 20), -1)
        cv2.putText(img, "SURVEILLANCE EVIDENCE ARCHIVE", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

        if row:
            _id, qid, tt, vn, d_date, d_time, ts, score, plate, plate_st = row
            score_pct = float(score or 0.0) * 100.0
            score_col = (50, 205, 50) if score_pct >= 70 else ((0, 215, 255) if score_pct >= 50 else (0, 140, 255))
            cv2.putText(img, f"TARGET: {str(tt or 'MATCH').upper()}", (24, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"SIMILARITY: {score_pct:.1f}%", (24, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, score_col, 2, cv2.LINE_AA)
            dt_str = f"TIME: {d_date}  {d_time}  (+{float(ts or 0.0):.2f}s)"
            cv2.putText(img, dt_str, (24, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            v_str = f"SOURCE: {vn if vn else 'Live Stream'}"
            if len(v_str) > 48:
                v_str = v_str[:45] + "..."
            cv2.putText(img, v_str, (24, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
            if plate:
                cv2.putText(img, f"PLATE: {plate} ({plate_st})", (24, 265),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.rectangle(img, (2, 315), (637, 357), (30, 22, 16), -1)
            cv2.putText(img, f"EVIDENCE ID #{_id}  |  QUERY #{qid}  |  STATUS: ARCHIVED", (16, 342),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "HISTORICAL MATCH EVIDENCE", (24, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)

        cv2.imwrite(dest_path, img)
        return True
    except Exception as e:
        print(f"[media fallback] frame creation error: {e}")
        return False


@media_bp.route("/media/frame/<path:fname>")
def media_frame(fname):
    base_fname = os.path.basename(fname)
    local_file = os.path.join(FRAMES_DIR, base_fname)
    if os.path.exists(local_file) and os.path.getsize(local_file) > 100:
        return send_from_directory(FRAMES_DIR, base_fname)
    if os.path.exists(fname) and os.path.isfile(fname) and os.path.getsize(fname) > 100:
        return send_from_directory(os.path.dirname(fname), os.path.basename(fname))

    # Fallback 1: Extract snapshot frame from local clip if available
    clip_candidate = base_fname.replace("frame_", "clip_").rsplit(".", 1)[0] + ".mp4"
    clip_path = os.path.join(CLIPS_DIR, clip_candidate)
    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1024:
        try:
            import cv2
            cap = cv2.VideoCapture(clip_path)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                cv2.imwrite(local_file, frame)
                return send_from_directory(FRAMES_DIR, base_fname)
        except Exception as e:
            print(f"[media_frame fallback] could not extract from {clip_candidate}: {e}")

    # Fallback 2: Generate CCTV evidence snapshot on the fly
    if _create_fallback_evidence_frame(base_fname, local_file):
        return send_from_directory(FRAMES_DIR, base_fname)

    return "Frame not found", 404


@media_bp.route("/media/clip/<path:fname>")
def media_clip(fname):
    base_fname = os.path.basename(fname)
    local_file = os.path.join(CLIPS_DIR, base_fname)
    if os.path.exists(local_file) and os.path.getsize(local_file) > 1024:
        return send_from_directory(CLIPS_DIR, base_fname, mimetype="video/mp4", conditional=True)
    if os.path.exists(fname) and os.path.isfile(fname) and os.path.getsize(fname) > 1024:
        return send_from_directory(os.path.dirname(fname), os.path.basename(fname), mimetype="video/mp4", conditional=True)

    # Self-healing fallback: Generate on-the-fly 5-second video clip from the evidence frame
    frame_candidate = base_fname.replace("clip_", "frame_").rsplit(".", 1)[0] + ".jpg"
    frame_path = os.path.join(FRAMES_DIR, frame_candidate)
    if not (os.path.exists(frame_path) and os.path.getsize(frame_path) > 100):
        _create_fallback_evidence_frame(frame_candidate, frame_path)

    if os.path.exists(frame_path) and os.path.getsize(frame_path) > 100:
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", frame_path,
                "-c:v", "libx264",
                "-t", "5.0",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-loglevel", "error",
                local_file,
            ]
            subprocess.run(cmd, check=True, timeout=10)
            if os.path.exists(local_file) and os.path.getsize(local_file) > 1024:
                return send_from_directory(CLIPS_DIR, base_fname, mimetype="video/mp4", conditional=True)
        except Exception as e:
            print(f"[media_clip fallback] clip creation error: {e}")

    return "Clip not found locally", 404


@media_bp.route("/media/stored")
def media_stored():
    path = request.args.get("path", "")
    real_dataset = os.path.realpath(DATASET_DIR)
    real_recordings = os.path.realpath(LIVE_RECORDINGS_DIR)
    real_path = os.path.realpath(path)
    if not (real_path.startswith(real_dataset) or real_path.startswith(real_recordings)):
        return "forbidden", 403
    return send_from_directory(os.path.dirname(real_path), os.path.basename(real_path), mimetype="video/mp4", conditional=True)


@media_bp.route("/media/drive_clip/<file_id>")
def media_drive_clip(file_id):
    """Streams a video file directly from Google Drive using authorized backend credentials."""
    if not gdrive_storage.is_configured():
        return "Google Drive not configured", 404
    try:
        chunks, content_type = gdrive_storage.stream_file(file_id)
        if chunks:
            return Response(chunks, mimetype=content_type or "video/mp4")
    except Exception as e:
        print(f"[media_drive_clip] error: {e}")
    return "Could not stream file from Google Drive", 404

