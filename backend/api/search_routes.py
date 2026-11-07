"""
Search Query Management, Target Upload, and Retrieval Routes.
"""
import os
import threading
import uuid
import base64
import sqlite3
import cv2
from flask import Blueprint, jsonify, request
from backend.config import QUERY_DIR
from backend.database import (
    db_add_query,
    db_set_query_mode,
    db_set_target_summary,
    db_set_stored_status,
    db_set_live_status,
    db_set_query_plate,
    db_delete_query,
    db_clear_all_queries,
    _db_lock,
)
from backend.engine import _target_summary, _print_target_breakdown
from backend.search_stored import search_stored_videos
from backend.gdrive import gdrive_storage
from . import get_engine, get_conn, get_matcher

search_bp = Blueprint("search_api", __name__)


def _run_stored_search_and_finalize(engine, conn, query_id, query_targets, index_manager):
    """Executes stored FAISS vector search in background and updates SQLite query status."""
    count = search_stored_videos(engine, conn, query_id, query_targets, index_manager)
    try:
        with _db_lock:
            q = conn.execute("SELECT stored_status FROM queries WHERE query_id=?", (query_id,)).fetchone()
            if q and q["stored_status"] == "disabled":
                return
        db_set_stored_status(conn, query_id, "matched" if count else "no_match")
    except (sqlite3.ProgrammingError, sqlite3.OperationalError, sqlite3.InterfaceError, Exception):
        return
    if gdrive_storage.is_configured():
        threading.Thread(
            target=gdrive_storage.sync_query_history_to_drive,
            args=(conn, query_id),
            daemon=True,
        ).start()


@search_bp.route("/upload", methods=["POST"])
def upload():
    engine = get_engine()
    conn = get_conn()
    matcher = get_matcher()

    f = request.files.get("query_image")
    existing_query_id = request.form.get("existing_query_id", "").strip()
    save_path = None
    fname = None
    query_id = None

    query_name = request.form.get("query_name", "").strip()
    search_source = request.form.get("search_source", "both").lower().strip()
    if search_source not in ("stored", "live", "both"):
        search_source = "both"

    run_live = search_source in ("live", "both")
    run_stored = search_source in ("stored", "both")
    initial_live_status = "monitoring" if (run_live and matcher.is_running()) else ("inactive" if run_live else "disabled")
    initial_stored_status = "searching" if run_stored else "disabled"

    if existing_query_id:
        with _db_lock:
            q_row = conn.execute("SELECT * FROM queries WHERE query_id=?", (existing_query_id,)).fetchone()
        if q_row and q_row["image_path"] and os.path.exists(q_row["image_path"]):
            query_id = existing_query_id
            save_path = q_row["image_path"]
            fname = os.path.basename(save_path)
            if not query_name:
                query_name = q_row["query_name"] or "Target"
            db_set_query_mode(conn, query_id, search_source)
            db_set_stored_status(conn, query_id, initial_stored_status)
            db_set_live_status(conn, query_id, initial_live_status)

    if not save_path:
        if not f or not f.filename:
            return jsonify({"error": "no file uploaded"}), 400

        if not query_name:
            base_name = os.path.splitext(f.filename)[0].replace("_", " ").strip()
            query_name = base_name if base_name else "Target"

        query_id = str(uuid.uuid4())[:8]
        fname = f"{query_id}_{f.filename}"
        save_path = os.path.join(QUERY_DIR, fname)
        f.save(save_path)

        db_add_query(
            conn,
            query_id,
            save_path,
            query_name=query_name,
            mode=search_source,
            stored_status=initial_stored_status,
            live_status=initial_live_status,
        )

    query_frame = cv2.imread(save_path)

    if query_frame is None:
        db_set_stored_status(conn, query_id, "no_match")
        return jsonify({
            "query_id": query_id,
            "query_name": query_name,
            "image_url": f"/media/query/{fname}",
            "target_summary": "could not read image",
            "search_source": search_source,
            "mode": search_source,
            "stored_status": "no_match",
            "live_status": "inactive" if run_live else "disabled",
        })

    query_targets = engine.extract_targets(query_frame, is_query=True)
    if not query_targets:
        db_set_stored_status(conn, query_id, "no_match")
        return jsonify({
            "query_id": query_id,
            "query_name": query_name,
            "image_url": f"/media/query/{fname}",
            "target_summary": "no person/car/bike detected",
            "search_source": search_source,
            "mode": search_source,
            "stored_status": "no_match",
            "live_status": "inactive" if run_live else "disabled",
        })

    vehicle_targets = [t for t in query_targets if t["type"] != "person"]
    person_targets = [t for t in query_targets if t["type"] == "person"]

    manual_plate = request.form.get("query_plate_text", "").strip()

    # Handle multiple vehicle disambiguation
    selected_idx_raw = request.form.get("selected_target_index")
    if len(vehicle_targets) > 1:
        if selected_idx_raw is not None and str(selected_idx_raw).strip() != "":
            try:
                sel_idx = int(selected_idx_raw)
                if 0 <= sel_idx < len(vehicle_targets):
                    query_targets = [vehicle_targets[sel_idx]] + person_targets
                    print(f"[upload] Multi-vehicle disambiguation: User selected vehicle #{sel_idx} ({vehicle_targets[sel_idx]['type']})")
            except (ValueError, TypeError):
                pass
        else:
            detected_list = []
            for idx, vt in enumerate(vehicle_targets):
                crop_b64 = ""
                if vt.get("crop") is not None and vt["crop"].size > 0:
                    try:
                        thumb = cv2.resize(vt["crop"], (160, 120))
                        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
                    except Exception:
                        pass
                detected_list.append({
                    "target_index": idx,
                    "type": vt["type"],
                    "bbox": vt["bbox"],
                    "conf": round(float(vt["conf"]), 2),
                    "plate_text": vt.get("plate_text", ""),
                    "crop_preview": crop_b64,
                    "is_primary": (idx == 0),
                })
            return jsonify({
                "ok": False,
                "status": "selection_required",
                "query_id": query_id,
                "query_name": query_name,
                "image_url": f"/media/query/{fname}",
                "message": f"Multiple vehicles detected ({len(vehicle_targets)} found). Please select target vehicle.",
                "detected_vehicles": detected_list,
                "search_source": search_source,
            })

    if manual_plate:
        for t in query_targets:
            if t["type"] != "person" and not t.get("plate_text"):
                t["plate_text"] = manual_plate

    summary = _target_summary(query_targets)
    query_plate_text = ", ".join([t["plate_text"] for t in query_targets if t.get("plate_text")])
    if not query_plate_text and manual_plate:
        query_plate_text = manual_plate

    db_set_target_summary(conn, query_id, summary)
    if query_plate_text:
        db_set_query_plate(conn, query_id, query_plate_text)
    _print_target_breakdown("query upload", query_targets)

    # Pre-create Google Drive History folder and upload initial target image
    if gdrive_storage.is_configured():
        threading.Thread(
            target=gdrive_storage.sync_query_history_to_drive,
            args=(conn, query_id),
            daemon=True,
        ).start()

    # 1. Register for live matching if selected and camera is active
    if run_live:
        if matcher.is_running():
            matcher.add_query(query_id, query_targets)
            live_status = "monitoring"
        else:
            live_status = "inactive"
            db_set_live_status(conn, query_id, "inactive")
    else:
        matcher.remove_query(query_id, status="disabled")
        live_status = "disabled"
        db_set_live_status(conn, query_id, "disabled")

    # 2. Asynchronous search through stored videos if selected
    if run_stored:
        stored_status = "searching"
        threading.Thread(
            target=_run_stored_search_and_finalize,
            args=(engine, conn, query_id, query_targets, matcher.index_manager),
            daemon=True,
        ).start()
    else:
        stored_status = "disabled"
        db_set_stored_status(conn, query_id, "disabled")

    return jsonify({
        "query_id": query_id,
        "query_name": query_name,
        "image_url": f"/media/query/{fname}",
        "target_summary": summary,
        "query_plate_text": query_plate_text,
        "search_source": search_source,
        "mode": search_source,
        "stored_status": stored_status,
        "live_status": live_status,
    })


@search_bp.route("/api/query/<query_id>/mode", methods=["POST"])
def change_query_mode(query_id):
    """Changes search mode (stored, live, or both) dynamically for an existing query."""
    engine = get_engine()
    conn = get_conn()
    matcher = get_matcher()

    data = request.get_json(silent=True) or request.form
    new_mode = (data.get("mode") or "both").lower().strip()
    if new_mode not in ("stored", "live", "both"):
        return jsonify({"error": "invalid mode"}), 400

    with _db_lock:
        q = conn.execute("SELECT * FROM queries WHERE query_id=?", (query_id,)).fetchone()
    if not q:
        return jsonify({"error": "unknown query"}), 404

    run_live = new_mode in ("live", "both")
    run_stored = new_mode in ("stored", "both")

    db_set_query_mode(conn, query_id, new_mode)

    qid_str = str(query_id).strip()
    query_targets = []
    with matcher.lock:
        if qid_str in matcher.active_queries:
            query_targets = matcher.active_queries[qid_str].get("targets", [])
    if not query_targets:
        image_path = q["image_path"]
        if image_path and os.path.exists(image_path):
            query_frame = cv2.imread(image_path)
            if query_frame is not None:
                query_targets = engine.extract_targets(query_frame, is_query=True)

    # 1. Live Mode handling
    if run_live:
        if matcher.is_running() and query_targets:
            matcher.add_query(query_id, query_targets)
            live_status = "monitoring"
        else:
            live_status = "inactive"
        db_set_live_status(conn, query_id, live_status)
    else:
        matcher.remove_query(query_id, status="disabled")
        live_status = "disabled"
        db_set_live_status(conn, query_id, "disabled")

    # 2. Stored Mode handling
    current_stored = q["stored_status"]
    if run_stored:
        if current_stored in ("disabled", "pending") and query_targets:
            stored_status = "searching"
            db_set_stored_status(conn, query_id, "searching")
            threading.Thread(
                target=_run_stored_search_and_finalize,
                args=(engine, conn, query_id, query_targets, matcher.index_manager),
                daemon=True,
            ).start()
        else:
            stored_status = current_stored if current_stored != "disabled" else "no_match"
    else:
        stored_status = "disabled"
        db_set_stored_status(conn, query_id, "disabled")

    return jsonify({
        "ok": True,
        "query_id": query_id,
        "mode": new_mode,
        "stored_status": stored_status,
        "live_status": live_status,
    })


@search_bp.route("/api/query/<query_id>/stop", methods=["POST"])
def stop_query(query_id):
    matcher = get_matcher()
    matcher.remove_query(query_id, status="stopped")
    return jsonify({"ok": True})


@search_bp.route("/api/query/<query_id>", methods=["DELETE", "POST"])
def delete_query_endpoint(query_id):
    matcher = get_matcher()
    conn = get_conn()
    matcher.remove_query(query_id, status="")
    deleted = db_delete_query(conn, query_id)
    return jsonify({"ok": deleted, "query_id": query_id})


@search_bp.route("/api/queries/clear", methods=["POST"])
def clear_all_queries_endpoint():
    matcher = get_matcher()
    conn = get_conn()
    for qid in matcher.active_query_ids():
        matcher.remove_query(qid, status="")
    db_clear_all_queries(conn)
    return jsonify({"ok": True, "message": "All search history cleared successfully."})


@search_bp.route("/api/results/<query_id>")
def api_results(query_id):
    conn = get_conn()
    matcher = get_matcher()
    engine = get_engine()

    with _db_lock:
        q = conn.execute("SELECT * FROM queries WHERE query_id=?", (query_id,)).fetchone()
        if not q:
            return jsonify({"error": "unknown query"}), 404
        rows = conn.execute(
            "SELECT * FROM detections WHERE query_id=? ORDER BY id DESC", (query_id,)
        ).fetchall()
    live, stored = [], []
    q_plate = q["query_plate_text"] if "query_plate_text" in q.keys() and q["query_plate_text"] else ""
    for r in rows:
        d = dict(r)
        if not d.get("plate_text") and q_plate:
            d["plate_text"] = q_plate
        d["query_plate_text"] = q_plate
        if d.get("frame_path") and not str(d["frame_path"]).startswith(("http://", "https://", "data:")):
            d["frame_path"] = os.path.basename(d["frame_path"])
        if d.get("clip_path") and not str(d["clip_path"]).startswith(("http://", "https://", "data:")):
            d["clip_path"] = os.path.basename(d["clip_path"])
        (live if d["source"] == "live" else stored).append(d)

    mode_val = q["mode"] if ("mode" in q.keys() and q["mode"]) else "both"
    return jsonify({
        "query_id": query_id,
        "mode": mode_val,
        "query_name": q["query_name"] if "query_name" in q.keys() and q["query_name"] else "Target",
        "drive_folder_id": q["drive_folder_id"] if "drive_folder_id" in q.keys() else "",
        "stored_status": q["stored_status"],
        "live_status": q["live_status"],
        "target_summary": q["target_summary"],
        "query_plate_text": q["query_plate_text"] if "query_plate_text" in q.keys() else "",
        "live": live,
        "stored": stored,
        "live_best_scores": matcher.get_best_scores(query_id) if q["live_status"] != "disabled" else {},
        "live_thresholds": {
            "person": engine.face_threshold,
            "car": engine.vehicle_threshold,
            "motorcycle": engine.vehicle_threshold,
            "bicycle": engine.vehicle_threshold,
        },
    })


@search_bp.route("/api/queries")
def api_queries():
    conn = get_conn()
    limit = request.args.get("limit", 20, type=int)
    with _db_lock:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY uploaded_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "query_id": r["query_id"],
            "mode": r["mode"] if ("mode" in r.keys() and r["mode"]) else "both",
            "query_name": r["query_name"] if "query_name" in r.keys() and r["query_name"] else "Target",
            "drive_folder_id": r["drive_folder_id"] if "drive_folder_id" in r.keys() else "",
            "image_basename": os.path.basename(r["image_path"]) if r["image_path"] else "",
            "uploaded_at": r["uploaded_at"],
            "stored_status": r["stored_status"],
            "live_status": r["live_status"],
            "target_summary": r["target_summary"],
            "query_plate_text": r["query_plate_text"] if "query_plate_text" in r.keys() else "",
        })
    return jsonify(out)
