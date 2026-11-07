"""
Historical Detections, Overview Statistics, and Temporal Filtering Routes.
"""
import os
from flask import Blueprint, jsonify, request
from backend.database import _db_lock
from . import get_conn

history_bp = Blueprint("history_api", __name__)


@history_bp.route("/api/history/dates")
def api_history_dates():
    """Returns available recording dates and detection dates from Google Drive / database."""
    conn = get_conn()
    with _db_lock:
        rec_dates = conn.execute(
            "SELECT MIN(SUBSTR(start_time, 1, 10)) as min_date, MAX(SUBSTR(start_time, 1, 10)) as max_date, COUNT(*) as count FROM recordings"
        ).fetchone()
        det_dates = conn.execute(
            "SELECT MIN(detection_date) as min_date, MAX(detection_date) as max_date, COUNT(*) as count FROM detections"
        ).fetchone()
        rec_list = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT SUBSTR(start_time, 1, 10) as d FROM recordings WHERE start_time IS NOT NULL ORDER BY d DESC"
            ).fetchall() if r[0]
        ]
        det_list = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT detection_date as d FROM detections WHERE detection_date IS NOT NULL ORDER BY d DESC"
            ).fetchall() if r[0]
        ]

    min_d = rec_dates["min_date"] or det_dates["min_date"] or ""
    max_d = rec_dates["max_date"] or det_dates["max_date"] or ""

    return jsonify({
        "recording_dates": rec_list,
        "detection_dates": det_list,
        "min_date": min_d,
        "max_date": max_d,
        "total_recordings": rec_dates["count"] or 0,
        "total_detections": det_dates["count"] or 0,
    })


@history_bp.route("/api/history")
def api_history():
    """Returns detection records, optionally filtered by source (live/stored), date range, query, or target type."""
    conn = get_conn()
    source = (request.args.get("source") or "").strip().lower()
    from_date = request.args.get("from_date", "").strip()
    to_date = request.args.get("to_date", "").strip()
    target_qid = request.args.get("query_id", "").strip()
    target_type = request.args.get("target_type", "").strip().lower()
    limit = request.args.get("limit", 500, type=int)

    if from_date and to_date and from_date > to_date:
        from_date, to_date = to_date, from_date

    sql = """
        SELECT d.*, q.query_name, q.target_summary, q.query_plate_text, q.image_path as query_image_path
        FROM detections d
        LEFT JOIN queries q ON d.query_id = q.query_id
        WHERE 1=1
    """
    params = []
    if source and source in ("live", "stored"):
        sql += " AND d.source=?"
        params.append(source)
    if target_qid:
        sql += " AND d.query_id=?"
        params.append(target_qid)
    if target_type:
        if target_type == "vehicle":
            sql += " AND LOWER(d.target_type) IN ('car', 'motorcycle', 'bicycle')"
        else:
            sql += " AND LOWER(d.target_type)=?"
            params.append(target_type)
    if from_date:
        sql += " AND SUBSTR(COALESCE(d.detection_date, d.created_at, ''), 1, 10) >= ?"
        params.append(from_date)
    if to_date:
        sql += " AND SUBSTR(COALESCE(d.detection_date, d.created_at, ''), 1, 10) <= ?"
        params.append(to_date)
    sql += " ORDER BY d.id DESC LIMIT ?"
    params.append(limit)

    with _db_lock:
        rows = conn.execute(sql, tuple(params)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if not d.get("plate_text") and d.get("query_plate_text"):
                d["plate_text"] = d["query_plate_text"]
            if d.get("frame_path") and not str(d["frame_path"]).startswith(("http://", "https://", "data:")):
                d["frame_path"] = os.path.basename(d["frame_path"])
            if d.get("clip_path") and not str(d["clip_path"]).startswith(("http://", "https://", "data:")):
                d["clip_path"] = os.path.basename(d["clip_path"])
            if d.get("query_image_path"):
                d["query_image_basename"] = os.path.basename(d["query_image_path"])
            out.append(d)
        return jsonify(out)


@history_bp.route("/api/history/stats")
def api_history_stats():
    """Returns overview statistics across all detection records for the history dashboard."""
    conn = get_conn()
    with _db_lock:
        total_dets = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        live_dets = conn.execute("SELECT COUNT(*) FROM detections WHERE source='live'").fetchone()[0]
        stored_dets = conn.execute("SELECT COUNT(*) FROM detections WHERE source='stored'").fetchone()[0]
        person_dets = conn.execute("SELECT COUNT(*) FROM detections WHERE LOWER(target_type)='person'").fetchone()[0]
        vehicle_dets = conn.execute("SELECT COUNT(*) FROM detections WHERE LOWER(target_type) IN ('car', 'motorcycle', 'bicycle')").fetchone()[0]
        verified_plates = conn.execute("SELECT COUNT(*) FROM detections WHERE plate_status='verified'").fetchone()[0]
        total_queries = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        active_matched_queries = conn.execute("SELECT COUNT(DISTINCT query_id) FROM detections").fetchone()[0]
        min_max = conn.execute("SELECT MIN(detection_date), MAX(detection_date) FROM detections").fetchone()

    return jsonify({
        "total_detections": total_dets,
        "live_detections": live_dets,
        "stored_detections": stored_dets,
        "person_detections": person_dets,
        "vehicle_detections": vehicle_dets,
        "verified_plates": verified_plates,
        "total_queries": total_queries,
        "queries_with_matches": active_matched_queries,
        "min_date": min_max[0] or "",
        "max_date": min_max[1] or "",
    })


@history_bp.route("/api/history/by_query")
def api_history_by_query():
    """Returns match history grouped separately by each query image strictly filtered by date range and query_id."""
    conn = get_conn()
    from_date = request.args.get("from_date", "").strip()
    to_date = request.args.get("to_date", "").strip()
    target_qid = request.args.get("query_id", "").strip()
    source_filter = (request.args.get("source") or "").strip().lower()
    type_filter = (request.args.get("target_type") or "").strip().lower()

    if from_date and to_date and from_date > to_date:
        from_date, to_date = to_date, from_date

    with _db_lock:
        if target_qid:
            queries_rows = conn.execute(
                "SELECT * FROM queries WHERE query_id=?", (target_qid,)
            ).fetchall()
        else:
            queries_rows = conn.execute(
                "SELECT * FROM queries ORDER BY uploaded_at DESC"
            ).fetchall()

        sql = "SELECT * FROM detections WHERE 1=1"
        params = []
        if target_qid:
            sql += " AND query_id=?"
            params.append(target_qid)
        if source_filter and source_filter in ("live", "stored"):
            sql += " AND source=?"
            params.append(source_filter)
        if type_filter:
            if type_filter == "vehicle":
                sql += " AND LOWER(target_type) IN ('car', 'motorcycle', 'bicycle')"
            else:
                sql += " AND LOWER(target_type)=?"
                params.append(type_filter)
        if from_date:
            sql += " AND SUBSTR(COALESCE(detection_date, created_at, ''), 1, 10) >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND SUBSTR(COALESCE(detection_date, created_at, ''), 1, 10) <= ?"
            params.append(to_date)
        sql += " ORDER BY id DESC"

        detections_rows = conn.execute(sql, tuple(params)).fetchall()

    detections_by_query = {}
    for r in detections_rows:
        d = dict(r)
        if d.get("frame_path") and not str(d["frame_path"]).startswith(("http://", "https://", "data:")):
            d["frame_path"] = os.path.basename(d["frame_path"])
        if d.get("clip_path") and not str(d["clip_path"]).startswith(("http://", "https://", "data:")):
            d["clip_path"] = os.path.basename(d["clip_path"])
        qid = d["query_id"]
        if qid not in detections_by_query:
            detections_by_query[qid] = []
        detections_by_query[qid].append(d)

    is_filtered = bool(from_date or to_date or source_filter or type_filter)
    out = []
    for q in queries_rows:
        qid = q["query_id"]
        dets = detections_by_query.get(qid, [])
        if is_filtered and not dets and not target_qid:
            continue

        q_plate = q["query_plate_text"] if "query_plate_text" in q.keys() and q["query_plate_text"] else ""
        for d in dets:
            if not d.get("plate_text") and q_plate:
                d["plate_text"] = q_plate
            d["query_plate_text"] = q_plate

        live_count = sum(1 for d in dets if d.get("source") == "live")
        stored_count = sum(1 for d in dets if d.get("source") == "stored")
        person_count = sum(1 for d in dets if str(d.get("target_type", "")).lower() == "person")
        vehicle_count = sum(1 for d in dets if str(d.get("target_type", "")).lower() in ("car", "motorcycle", "bicycle"))
        verified_plates_list = list({d.get("plate_text") for d in dets if d.get("plate_status") == "verified" and d.get("plate_text")})
        if not verified_plates_list and q_plate:
            verified_plates_list = [q_plate]

        scores = [float(d["similarity_score"]) for d in dets if d.get("similarity_score") is not None]
        highest_score = max(scores) if scores else 0.0

        dates = [d.get("detection_date", "") for d in dets if d.get("detection_date")]
        first_seen = min(dates) if dates else ""
        last_seen = max(dates) if dates else ""

        img_basename = os.path.basename(q["image_path"]) if q["image_path"] else ""
        q_name = q["query_name"] if "query_name" in q.keys() and q["query_name"] else "Target"
        q_mode = q["mode"] if "mode" in q.keys() and q["mode"] else "both"

        out.append({
            "query_id": qid,
            "query_name": q_name,
            "mode": q_mode,
            "drive_folder_id": q["drive_folder_id"] if "drive_folder_id" in q.keys() else "",
            "image_basename": img_basename,
            "image_url": f"/media/query/{img_basename}" if img_basename else "",
            "uploaded_at": q["uploaded_at"],
            "stored_status": q["stored_status"],
            "live_status": q["live_status"],
            "target_summary": q["target_summary"],
            "query_plate_text": q["query_plate_text"] if "query_plate_text" in q.keys() else "",
            "total_matches": len(dets),
            "live_matches": live_count,
            "stored_matches": stored_count,
            "person_matches": person_count,
            "vehicle_matches": vehicle_count,
            "highest_score": highest_score,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "verified_plates": verified_plates_list,
            "detections": dets,
            "is_date_filtered": is_filtered,
            "filter_from_date": from_date,
            "filter_to_date": to_date,
        })

    return jsonify(out)


@history_bp.route("/api/query/<query_id>/history/clear", methods=["POST"])
def clear_single_query_history(query_id):
    """Clears all detection records for a specific query."""
    conn = get_conn()
    with _db_lock:
        conn.execute("DELETE FROM detections WHERE query_id=?", (query_id,))
        conn.commit()
    return jsonify({"ok": True, "query_id": query_id})

