"""
Google Drive History Synchronization and SQLite Database Reconciliation.
"""
import os
import json
import uuid
from datetime import datetime, timedelta
from backend.config import get_video_start_time, VIDEO_BUFFER_DIR, RECORDING_SEGMENT_SECONDS


class DriveSyncMixin:
    """Provides query history backup, cloud recording indexing sync, and buffer uploads."""

    def get_or_create_history_root_folder(self):
        """Finds or creates top-level 'History' folder in Google Drive."""
        return self.get_or_create_folder("History", parent_id=self.root_folder_id or "root")

    def get_or_create_query_history_folder(self, query_id, query_name=""):
        """Creates/retrieves: Google Drive / History / <Query_Name>_<Query_ID> /."""
        history_root_id = self.get_or_create_history_root_folder()
        clean_name = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (query_name or "Target").strip())
        clean_name = "_".join(filter(None, clean_name.split("_"))) or "Target"
        folder_name = f"{clean_name}_{query_id}"
        folder_id = self.get_or_create_folder(folder_name, parent_id=history_root_id)
        return folder_id, folder_name

    def sync_query_history_to_drive(self, conn, query_id):
        """Synchronizes all history data for a query to Google Drive."""
        if not self.is_configured():
            return {"ok": False, "error": "Google Drive not configured"}

        try:
            with conn:
                q = conn.execute("SELECT * FROM queries WHERE query_id=?", (query_id,)).fetchone()
                if not q:
                    return {"ok": False, "error": "Query not found"}
                q_dict = dict(q)
                detections = [
                    dict(r) for r in conn.execute(
                        "SELECT * FROM detections WHERE query_id=? ORDER BY id ASC", (query_id,)
                    ).fetchall()
                ]

            q_name = q_dict.get("query_name") or "Target"
            folder_id, folder_name = self.get_or_create_query_history_folder(query_id, q_name)

            try:
                conn.execute("UPDATE queries SET drive_folder_id=? WHERE query_id=?", (folder_id, query_id))
                conn.commit()
            except Exception:
                pass

            import tempfile
            temp_dir = tempfile.mkdtemp(prefix=f"gdrive_sync_{query_id}_")

            img_path = q_dict.get("image_path")
            if img_path and os.path.exists(img_path):
                ext = os.path.splitext(img_path)[1] or ".jpg"
                target_fname = f"target_image_{query_id}{ext}"
                try:
                    self.upload_file(img_path, parent_folder_id=folder_id, target_filename=target_fname)
                except Exception as e:
                    print(f"[gdrive-sync] Warning uploading target image: {e}")

            report_data = {
                "query_id": query_id,
                "query_name": q_name,
                "uploaded_at": q_dict.get("uploaded_at"),
                "stored_status": q_dict.get("stored_status"),
                "live_status": q_dict.get("live_status"),
                "target_summary": q_dict.get("target_summary"),
                "total_matches": len(detections),
                "synced_at": datetime.now().isoformat(timespec="seconds"),
                "detections": detections,
            }
            json_report_path = os.path.join(temp_dir, "history_report.json")
            with open(json_report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            self.upload_file(json_report_path, parent_folder_id=folder_id, target_filename="history_report.json", mime_type="application/json")

            csv_report_path = os.path.join(temp_dir, "history_report.csv")
            import csv
            with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Query ID", "Query Name", "Detection ID", "Event Date", "Event Time",
                    "Source", "Target Type", "Match Score %", "Video Name", "Video Offset (s)",
                    "Frame File", "Clip File"
                ])
                for d in detections:
                    score_pct = f"{(d.get('similarity_score', 0) * 100):.1f}%"
                    writer.writerow([
                        query_id,
                        q_name,
                        d.get("id"),
                        d.get("detection_date"),
                        d.get("detection_time"),
                        d.get("source"),
                        d.get("target_type"),
                        score_pct,
                        d.get("video_name") or "",
                        d.get("timestamp_sec") or 0.0,
                        os.path.basename(d.get("frame_path") or ""),
                        os.path.basename(d.get("clip_path") or ""),
                    ])
            self.upload_file(csv_report_path, parent_folder_id=folder_id, target_filename="history_report.csv", mime_type="text/csv")

            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            print(f"[gdrive-sync] Synced history for Query #{query_id} ('{q_name}') -> Google Drive ({folder_name}/) [ID: {folder_id}]")
            return {
                "ok": True,
                "query_id": query_id,
                "query_name": q_name,
                "folder_id": folder_id,
                "folder_name": folder_name,
                "match_count": len(detections),
            }
        except Exception as e:
            print(f"[gdrive-sync] Error syncing query #{query_id} to Google Drive: {e}")
            return {"ok": False, "error": str(e)}

    def sync_all_history_to_drive(self, conn):
        """Syncs all queries and their match history into Google Drive / History /."""
        with conn:
            q_rows = conn.execute("SELECT query_id FROM queries").fetchall()
        results = []
        for r in q_rows:
            res = self.sync_query_history_to_drive(conn, r["query_id"])
            results.append(res)
        return {
            "ok": True,
            "synced_count": sum(1 for r in results if r.get("ok")),
            "total_queries": len(q_rows),
            "details": results,
        }

    def sync_all_drive_recordings_to_db(self, conn):
        """Scans Google Drive recordings, registers new entries in SQLite, and updates IDs."""
        if not self.is_configured():
            return {"ok": False, "error": "Google Drive not configured"}

        try:
            drive_videos = self.list_all_drive_videos()
            rec_videos = [
                v for v in drive_videos
                if v.get("name", "").startswith("REC_") and v.get("name", "").endswith(".mp4")
            ]

            with conn:
                existing_rows = {
                    r["filename"]: dict(r)
                    for r in conn.execute("SELECT * FROM recordings").fetchall()
                }

                new_count = 0
                updated_count = 0
                distinct_dates = set()

                for v in rec_videos:
                    fname = v.get("name", "")
                    file_id = v.get("id", "")
                    drive_url = f"https://drive.google.com/file/d/{file_id}/view"

                    start_dt = get_video_start_time(fname)
                    end_dt = start_dt + timedelta(seconds=300)
                    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                    distinct_dates.add(start_dt.strftime("%Y-%m-%d"))

                    if fname not in existing_rows:
                        rec_id = str(uuid.uuid4())[:12]
                        conn.execute(
                            """
                            INSERT INTO recordings (
                                recording_id, start_time, end_time, filename, local_cache_path,
                                upload_status, google_drive_file_id, google_drive_path,
                                index_status, retry_count
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rec_id, start_str, end_str, fname, "",
                                "uploaded", file_id, drive_url,
                                "pending", 0,
                            ),
                        )
                        new_count += 1
                    else:
                        existing = existing_rows[fname]
                        if not existing.get("google_drive_file_id") or existing.get("google_drive_file_id") != file_id:
                            conn.execute(
                                """
                                UPDATE recordings
                                SET google_drive_file_id=?, google_drive_path=?, upload_status='uploaded'
                                WHERE filename=?
                                """,
                                (file_id, drive_url, fname),
                            )
                            updated_count += 1

                conn.commit()

            print(
                f"[gdrive-sync] Registered {new_count} new cloud recordings, updated {updated_count} IDs "
                f"from {len(rec_videos)} Google Drive files across {len(distinct_dates)} days."
            )
            return {
                "ok": True,
                "total_drive_recordings": len(rec_videos),
                "new_recordings_registered": new_count,
                "updated_recordings": updated_count,
                "dates": sorted(list(distinct_dates)),
            }
        except Exception as e:
            print(f"[gdrive-sync] Error synchronizing Google Drive recordings to DB: {e}")
            return {"ok": False, "error": str(e)}

    def upload_pending_buffer_recordings(self, conn):
        """Scans pending recordings from DB and video_buffer, uploading them to Google Drive."""
        if not self.is_configured():
            print("[gdrive-upload] ERROR: Google Drive is not configured.")
            return {"ok": False, "error": "Google Drive not configured"}

        pending_rows = conn.execute(
            "SELECT * FROM recordings WHERE upload_status != 'uploaded' ORDER BY start_time ASC"
        ).fetchall()

        disk_files = {}
        if os.path.exists(VIDEO_BUFFER_DIR):
            for f in os.listdir(VIDEO_BUFFER_DIR):
                if f.endswith(".mp4") and not f.endswith((".tmp", ".part")) and not f.startswith("_raw_"):
                    fp = os.path.join(VIDEO_BUFFER_DIR, f)
                    if os.path.exists(fp) and os.path.getsize(fp) > 1024:
                        disk_files[f] = fp

        targets_to_upload = []
        seen_filenames = set()

        for r in pending_rows:
            fn = r["filename"]
            lp = r["local_cache_path"]
            if not lp or not os.path.exists(lp):
                lp = disk_files.get(fn)
            if lp and os.path.exists(lp) and os.path.getsize(lp) > 1024:
                targets_to_upload.append((fn, lp, r["recording_id"]))
                seen_filenames.add(fn)

        for fn, lp in disk_files.items():
            if fn not in seen_filenames:
                targets_to_upload.append((fn, lp, None))
                seen_filenames.add(fn)

        print(f"[gdrive-upload] Found {len(targets_to_upload)} video recordings to upload to Google Drive.")
        uploaded_count = 0

        for idx, (filename, full_path, rec_id) in enumerate(targets_to_upload, 1):
            try:
                start_dt = get_video_start_time(full_path)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                print(f"[gdrive-upload] [{idx}/{len(targets_to_upload)}] Uploading {filename} ({size_mb:.1f} MB)...")
                target_folder_id, hierarchy_path = self.get_or_create_folder_hierarchy(start_dt)
                file_id, drive_fname, web_link = self.upload_file(full_path, target_folder_id, target_filename=filename)

                if file_id:
                    drive_full_path = f"{hierarchy_path}/{drive_fname}"
                    end_dt = start_dt + timedelta(seconds=RECORDING_SEGMENT_SECONDS)
                    if not rec_id:
                        rec_id = str(uuid.uuid4())[:12]
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO recordings (
                                recording_id, start_time, end_time, filename, local_cache_path,
                                upload_status, google_drive_file_id, google_drive_path,
                                index_status, retry_count
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rec_id, start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                end_dt.strftime("%Y-%m-%d %H:%M:%S"), filename, "",
                                "uploaded", file_id, drive_full_path, "pending", 0,
                            )
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE recordings
                            SET upload_status='uploaded', google_drive_file_id=?, google_drive_path=?, local_cache_path=''
                            WHERE recording_id=? OR filename=?
                            """,
                            (file_id, drive_full_path, rec_id, filename)
                        )
                    conn.commit()
                    uploaded_count += 1
                    if os.path.exists(full_path):
                        try:
                            os.remove(full_path)
                            print(f"[gdrive-upload] Deleted local video file after cloud upload: {filename}")
                        except OSError as rm_err:
                            print(f"[gdrive-upload] Note deleting local file {filename}: {rm_err}")
                    print(f"[gdrive-upload] SUCCESS [{idx}/{len(targets_to_upload)}]: {filename} -> Drive ID: {file_id}")
            except Exception as e:
                print(f"[gdrive-upload] ERROR uploading {filename}: {e}")

        print(f"[gdrive-upload] Batch complete: {uploaded_count}/{len(targets_to_upload)} files successfully uploaded to Google Drive.")
        return {"ok": True, "uploaded_count": uploaded_count, "total": len(targets_to_upload)}

