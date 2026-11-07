"""
Database access and management module for Jetson Nano AI Vision Search.
"""
import os
import json
import sqlite3
import threading
from datetime import datetime
from .config import DB_PATH

_db_lock = threading.Lock()


def get_conn():
    """Returns a thread-safe SQLite connection with WAL mode and 30s busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass
    return conn


def _safe_commit(conn):
    """Safely commits transaction, ignoring operational errors if autocommit or no transaction is active."""
    try:
        conn.commit()
    except (sqlite3.OperationalError, sqlite3.ProgrammingError):
        pass


def init_db():
    """Initializes SQLite database schema and handles migrations."""
    conn = get_conn()
    with _db_lock:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                image_path TEXT,
                uploaded_at TEXT,
                stored_status TEXT,      -- searching | matched | no_match
                live_status TEXT,        -- inactive | monitoring | stopped
                target_summary TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                target_type TEXT,        -- person | car | motorcycle | bicycle
                source TEXT,             -- live | stored
                video_name TEXT,
                video_path TEXT,
                detection_date TEXT,
                detection_time TEXT,
                timestamp_sec REAL,
                similarity_score REAL,
                frame_path TEXT,
                clip_path TEXT,
                bbox TEXT,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                google_drive_file_id TEXT,
                google_drive_path TEXT,
                filename TEXT,
                local_cache_path TEXT,
                upload_status TEXT       -- pending | uploaded | failed
            )
        """)
        # Best-effort migration for schema evolution
        for stmt in (
            "ALTER TABLE queries ADD COLUMN stored_status TEXT DEFAULT 'pending'",
            "ALTER TABLE queries ADD COLUMN live_status TEXT DEFAULT 'inactive'",
            "ALTER TABLE queries ADD COLUMN target_summary TEXT",
            "ALTER TABLE queries ADD COLUMN query_name TEXT DEFAULT ''",
            "ALTER TABLE queries ADD COLUMN query_plate_text TEXT DEFAULT ''",
            "ALTER TABLE queries ADD COLUMN drive_folder_id TEXT DEFAULT ''",
            "ALTER TABLE detections ADD COLUMN plate_text TEXT DEFAULT ''",
            "ALTER TABLE detections ADD COLUMN plate_status TEXT DEFAULT 'not_applicable'",
            "ALTER TABLE recordings ADD COLUMN index_status TEXT DEFAULT 'pending'",
            "ALTER TABLE recordings ADD COLUMN retry_count INTEGER DEFAULT 0",
            "ALTER TABLE recordings ADD COLUMN next_retry_time REAL DEFAULT 0",
            "ALTER TABLE queries ADD COLUMN mode TEXT DEFAULT 'both'",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        _safe_commit(conn)
    return conn


def db_add_query(conn, query_id, image_path, query_name="", query_plate_text="", mode="both", stored_status="searching", live_status="inactive"):
    """Inserts a new search query."""
    with _db_lock:
        conn.execute(
            "INSERT INTO queries (query_id, image_path, uploaded_at, stored_status, live_status, query_name, query_plate_text, mode) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (query_id, image_path, datetime.now().isoformat(timespec="seconds"),
             stored_status, live_status, query_name, query_plate_text, mode),
        )
        _safe_commit(conn)


def db_set_query_mode(conn, query_id, mode):
    """Updates search mode ('stored', 'live', 'both') for a query."""
    with _db_lock:
        conn.execute("UPDATE queries SET mode=? WHERE query_id=?", (mode, query_id))
        _safe_commit(conn)


def db_set_query_name(conn, query_id, query_name):
    """Sets or updates query_name."""
    with _db_lock:
        conn.execute("UPDATE queries SET query_name=? WHERE query_id=?", (query_name, query_id))
        _safe_commit(conn)


def db_set_query_drive_folder(conn, query_id, folder_id):
    """Sets Google Drive history folder ID for a query."""
    with _db_lock:
        conn.execute("UPDATE queries SET drive_folder_id=? WHERE query_id=?", (folder_id, query_id))
        _safe_commit(conn)


def db_set_query_plate(conn, query_id, query_plate_text):
    """Sets extracted query plate string."""
    with _db_lock:
        conn.execute("UPDATE queries SET query_plate_text=? WHERE query_id=?", (query_plate_text, query_id))
        _safe_commit(conn)


def db_set_target_summary(conn, query_id, summary):
    """Updates target breakdown summary for a query."""
    with _db_lock:
        conn.execute("UPDATE queries SET target_summary=? WHERE query_id=?", (summary, query_id))
        _safe_commit(conn)


def db_set_stored_status(conn, query_id, status):
    """Updates stored-video search status."""
    with _db_lock:
        conn.execute("UPDATE queries SET stored_status=? WHERE query_id=?", (status, query_id))
        _safe_commit(conn)


def db_set_live_status(conn, query_id, status):
    """Updates live-camera search status."""
    with _db_lock:
        conn.execute("UPDATE queries SET live_status=? WHERE query_id=?", (status, query_id))
        _safe_commit(conn)


def db_add_detection(conn, **row):
    """
    Inserts a matched detection record.
    Only called on an actual match.
    """
    now = datetime.now()
    row.setdefault("detection_date", now.strftime("%Y-%m-%d"))
    row.setdefault("detection_time", now.strftime("%H:%M:%S"))
    row.setdefault("created_at", now.isoformat(timespec="seconds"))
    row.setdefault("plate_text", "")
    row.setdefault("plate_status", "not_applicable")
    row["bbox"] = json.dumps(row.get("bbox", []))
    cols = ",".join(row.keys())
    qs = ",".join(["?"] * len(row))
    with _db_lock:
        cur = conn.execute(f"INSERT INTO detections ({cols}) VALUES ({qs})", tuple(row.values()))
        _safe_commit(conn)
        return cur.lastrowid


def db_get_queries(conn, limit=50):
    """Retrieves recent queries ordered by upload time descending."""
    with _db_lock:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY uploaded_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def db_get_results(conn, query_id):
    """Retrieves query info and all associated detections."""
    with _db_lock:
        q = conn.execute("SELECT * FROM queries WHERE query_id=?", (query_id,)).fetchone()
        if not q:
            return None, None
        rows = conn.execute(
            "SELECT * FROM detections WHERE query_id=? ORDER BY id DESC", (query_id,)
        ).fetchall()
        return dict(q), [dict(r) for r in rows]


def db_get_history(conn, source, limit=50):
    """Retrieves detection history for a source ('live' or 'stored')."""
    with _db_lock:
        rows = conn.execute(
            "SELECT * FROM detections WHERE source=? ORDER BY id DESC LIMIT ?", (source, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def db_delete_query(conn, query_id):
    """
    Deletes a specific query, cleans up its uploaded query image,
    and removes all associated detection records and media files.
    """
    with _db_lock:
        # 1. Clean up query image file
        q = conn.execute("SELECT image_path FROM queries WHERE query_id=?", (query_id,)).fetchone()
        if q and q["image_path"] and os.path.exists(q["image_path"]):
            try:
                os.remove(q["image_path"])
            except OSError:
                pass

        # 2. Clean up associated detection frames and clips
        det_rows = conn.execute(
            "SELECT frame_path, clip_path FROM detections WHERE query_id=?", (query_id,)
        ).fetchall()
        for r in det_rows:
            for p in (r["frame_path"], r["clip_path"]):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        # 3. Delete from database
        conn.execute("DELETE FROM detections WHERE query_id=?", (query_id,))
        cur = conn.execute("DELETE FROM queries WHERE query_id=?", (query_id,))
        _safe_commit(conn)
        return cur.rowcount > 0


def db_clear_all_queries(conn):
    """
    Clears all search queries and all detection records,
    along with their query images, frames, and clips.
    """
    with _db_lock:
        # Remove all query images
        q_rows = conn.execute("SELECT image_path FROM queries").fetchall()
        for r in q_rows:
            if r["image_path"] and os.path.exists(r["image_path"]):
                try:
                    os.remove(r["image_path"])
                except OSError:
                    pass

        # Remove all detection frames and clips
        det_rows = conn.execute("SELECT frame_path, clip_path FROM detections").fetchall()
        for r in det_rows:
            for p in (r["frame_path"], r["clip_path"]):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM queries")
        _safe_commit(conn)
        return True


def db_clear_history(conn, source):
    """
    Deletes all detections for one source ('live' or 'stored')
    and their mirrored frame/clip files.
    """
    assert source in ("live", "stored")
    with _db_lock:
        rows = conn.execute(
            "SELECT frame_path, clip_path FROM detections WHERE source=?", (source,)
        ).fetchall()
        for r in rows:
            for p in (r["frame_path"], r["clip_path"]):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        cur = conn.execute("DELETE FROM detections WHERE source=?", (source,))
        _safe_commit(conn)
        return cur.rowcount


def db_add_recording(conn, recording_id, start_time, end_time, filename, local_cache_path,
                     upload_status="pending", index_status="pending", google_drive_file_id="", google_drive_path=""):
    """Inserts a 5-minute video segment record."""
    with _db_lock:
        conn.execute(
            """
            INSERT INTO recordings (
                recording_id, start_time, end_time, google_drive_file_id,
                google_drive_path, filename, local_cache_path, upload_status, index_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recording_id,
                start_time,
                end_time,
                google_drive_file_id,
                google_drive_path,
                filename,
                local_cache_path,
                upload_status,
                index_status,
            ),
        )
        _safe_commit(conn)


def db_update_recording_index_status(conn, recording_id, index_status):
    """Updates recording AI indexing status."""
    with _db_lock:
        conn.execute(
            "UPDATE recordings SET index_status = ? WHERE recording_id = ?",
            (index_status, recording_id),
        )
        _safe_commit(conn)


def db_update_recording_upload(conn, recording_id, upload_status, google_drive_file_id="",
                               google_drive_path="", local_cache_path="", retry_count=None, next_retry_time=None):
    """Updates recording status after Google Drive upload attempt with optional retry backoff."""
    with _db_lock:
        sql = """
            UPDATE recordings
            SET upload_status = ?,
                google_drive_file_id = CASE WHEN ? != '' THEN ? ELSE google_drive_file_id END,
                google_drive_path = CASE WHEN ? != '' THEN ? ELSE google_drive_path END,
                local_cache_path = ?
        """
        params = [
            upload_status,
            google_drive_file_id, google_drive_file_id,
            google_drive_path, google_drive_path,
            local_cache_path,
        ]
        if retry_count is not None:
            sql += ", retry_count = ?"
            params.append(retry_count)
        if next_retry_time is not None:
            sql += ", next_retry_time = ?"
            params.append(next_retry_time)
        sql += " WHERE recording_id = ?"
        params.append(recording_id)

        conn.execute(sql, tuple(params))
        _safe_commit(conn)


def db_get_recording_by_filename(conn, filename):
    """Retrieves recording metadata by video filename."""
    with _db_lock:
        row = conn.execute("SELECT * FROM recordings WHERE filename=?", (filename,)).fetchone()
        return dict(row) if row else None


def db_get_recording(conn, recording_id):
    """Retrieves recording metadata by recording_id."""
    with _db_lock:
        row = conn.execute("SELECT * FROM recordings WHERE recording_id=?", (recording_id,)).fetchone()
        return dict(row) if row else None


def db_get_pending_recordings(conn):
    """Retrieves all pending or failed recordings that need to be uploaded to Google Drive."""
    with _db_lock:
        rows = conn.execute(
            "SELECT * FROM recordings WHERE upload_status != 'uploaded' ORDER BY start_time ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def db_get_pending_index_recordings(conn):
    """Retrieves all recordings that still need AI indexing."""
    with _db_lock:
        rows = conn.execute(
            "SELECT * FROM recordings WHERE (index_status IS NULL OR index_status != 'indexed') ORDER BY start_time ASC"
        ).fetchall()
        return [dict(r) for r in rows]

