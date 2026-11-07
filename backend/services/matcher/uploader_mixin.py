"""
Background Google Drive Upload Worker and Local Buffer Disk Retention Mixin.
"""
import os
import time
import queue
import uuid
from datetime import timedelta
from backend.config import (
    RECORDING_SEGMENT_SECONDS,
    VIDEO_BUFFER_DIR,
    get_video_start_time,
)
from backend.database import (
    db_get_recording_by_filename,
    db_add_recording,
    db_update_recording_upload,
    db_get_pending_recordings,
)
from backend.recorder import is_valid_mp4
from backend.gdrive import gdrive_storage


class UploaderMixin:
    """Manages background Google Drive uploads and local disk space retention pruning."""

    def _enqueue_upload_job(self, path, is_retry=False):
        """Adds a finalized recording to the Google Drive upload queue."""
        if not path or not os.path.exists(path) or path.endswith((".tmp", ".part")):
            return
        full_path = os.path.abspath(path)
        now = time.time()
        with self._upload_lock:
            if full_path in self._queued_or_active_upload:
                return
            if is_retry and full_path in self._upload_retry_schedule:
                retry_cnt, next_time = self._upload_retry_schedule[full_path]
                if now < next_time:
                    return

            self._queued_or_active_upload.add(full_path)
            try:
                self.upload_queue.put_nowait(full_path)
                print(f"[upload-queue] queued: {os.path.basename(full_path)} (queue size: {self.upload_queue.qsize()})")
            except queue.Full:
                self._queued_or_active_upload.discard(full_path)
                print(f"[upload-queue] WARNING: Upload queue full, dropped {os.path.basename(full_path)}")

    def _sync_and_enqueue_all_drive_videos(self):
        """Scans Google Drive across all dates and enqueues unindexed recordings."""
        if not gdrive_storage.is_configured():
            return
        try:
            print("[matcher] Synchronizing all historical Google Drive recordings...")
            res = gdrive_storage.sync_all_drive_recordings_to_db(self.conn)
            if res.get("ok"):
                dates = res.get("dates", [])
                print(f"[matcher] Synced {res.get('total_drive_recordings', 0)} Drive recordings across {len(dates)} days: {dates[:5]}...")
                self._scan_unprocessed_recordings()
        except Exception as e:
            print(f"[matcher] Error in background Drive sync: {e}")

    def _retry_pending_jobs(self):
        """Periodically scans database for pending uploads ready for backoff retry."""
        try:
            pending = db_get_pending_recordings(self.conn)
            for rec in pending:
                loc_path = rec.get("local_cache_path")
                if not loc_path or not os.path.exists(loc_path):
                    candidate = os.path.join(VIDEO_BUFFER_DIR, rec.get("filename", ""))
                    if os.path.exists(candidate):
                        loc_path = candidate
                if loc_path and os.path.exists(loc_path) and is_valid_mp4(loc_path):
                    self._enqueue_upload_job(loc_path, is_retry=True)
        except Exception as e:
            print(f"[matcher] error in periodic retry scan: {e}")

    def _upload_worker_loop(self):
        """Dedicated background worker thread for 8MB chunked Google Drive uploads."""
        import gc

        print("[drive-upload-worker] Background Google Drive upload worker ready.")
        backoff_delays = [30, 60, 120, 300, 600]

        while True:
            try:
                full_path = self.upload_queue.get()
            except Exception:
                break

            if full_path is None:
                break

            filename = os.path.basename(full_path)
            with self._upload_lock:
                self._active_upload_file = filename

            rec_id = None
            try:
                if not os.path.exists(full_path) or full_path.endswith((".tmp", ".part")):
                    continue
                if not is_valid_mp4(full_path):
                    continue

                start_dt = get_video_start_time(full_path)
                end_dt = start_dt + timedelta(seconds=RECORDING_SEGMENT_SECONDS)
                rec = db_get_recording_by_filename(self.conn, filename)
                if not rec:
                    rec_id = str(uuid.uuid4())[:12]
                    db_add_recording(
                        self.conn,
                        recording_id=rec_id,
                        start_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        end_time=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        filename=filename,
                        local_cache_path=full_path,
                        upload_status="pending",
                    )
                else:
                    rec_id = rec["recording_id"]

                if not gdrive_storage.is_configured():
                    print(f"[drive-uploader] Google Drive not configured. Retaining local segment: {filename}")
                    continue

                print(f"[drive-uploader] START: Uploading {filename} to Google Drive...")
                target_folder_id, hierarchy_path = gdrive_storage.get_or_create_folder_hierarchy(start_dt)
                file_id, drive_fname, web_link = gdrive_storage.upload_file(
                    full_path, target_folder_id, target_filename=filename
                )

                if not file_id:
                    raise RuntimeError("Google Drive upload returned no file ID")

                drive_full_path = f"{hierarchy_path}/{drive_fname}"
                db_update_recording_upload(
                    self.conn,
                    recording_id=rec_id,
                    upload_status="uploaded",
                    google_drive_file_id=file_id,
                    google_drive_path=drive_full_path,
                    local_cache_path="",
                )

                self.index_manager.update_video_drive_info(filename, file_id, drive_full_path)
                print(f"[drive-uploader] SUCCESS: {filename} uploaded to Google Drive ({drive_full_path}) [ID: {file_id}].")

                with self._upload_lock:
                    self._upload_retry_schedule.pop(full_path, None)

                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        print(f"[drive-uploader] Deleted local video file after cloud upload: {filename}")
                    except OSError as rm_err:
                        print(f"[drive-uploader] Note deleting local file {filename}: {rm_err}")

            except Exception as e:
                print(f"[drive-uploader] FAILED: error uploading {filename}: {e}")
                with self._upload_lock:
                    curr_retry = self._upload_retry_schedule.get(full_path, (0, 0))[0] + 1
                    delay = backoff_delays[min(curr_retry - 1, len(backoff_delays) - 1)]
                    next_time = time.time() + delay
                    self._upload_retry_schedule[full_path] = (curr_retry, next_time)
                print(f"[drive-uploader] Scheduled backoff retry #{curr_retry} for {filename} in {delay}s.")

                if rec_id:
                    db_update_recording_upload(
                        self.conn,
                        recording_id=rec_id,
                        upload_status="retry_wait",
                        local_cache_path=full_path,
                        retry_count=curr_retry,
                        next_retry_time=next_time,
                    )
            finally:
                with self._upload_lock:
                    self._active_upload_file = None
                    self._queued_or_active_upload.discard(full_path)
                gc.collect()

    def _prune_local_video_buffer_if_needed(self, min_free_gb=5.0, max_buffer_gb=15.0):
        """Manages local video retention, freeing space only for confirmed uploaded videos."""
        import shutil
        try:
            stat = shutil.disk_usage(VIDEO_BUFFER_DIR)
            free_gb = stat.free / (1024 ** 3)

            buffer_files = []
            total_size_bytes = 0
            for root, _, files in os.walk(VIDEO_BUFFER_DIR):
                for f in files:
                    if f.endswith(".mp4") and not f.endswith((".tmp", ".part")):
                        fp = os.path.join(root, f)
                        try:
                            sz = os.path.getsize(fp)
                            mtime = os.path.getmtime(fp)
                            buffer_files.append((mtime, fp, f, sz))
                            total_size_bytes += sz
                        except OSError:
                            pass

            buffer_gb = total_size_bytes / (1024 ** 3)
            if free_gb >= min_free_gb and buffer_gb <= max_buffer_gb:
                return

            buffer_files.sort(key=lambda x: x[0])
            for _, fp, fname, sz in buffer_files:
                if free_gb >= min_free_gb and buffer_gb <= max_buffer_gb:
                    break
                rec = db_get_recording_by_filename(self.conn, fname)
                if rec and rec.get("upload_status") == "uploaded":
                    try:
                        os.remove(fp)
                        buffer_gb -= (sz / (1024 ** 3))
                        free_gb += (sz / (1024 ** 3))
                        db_update_recording_upload(
                            self.conn,
                            recording_id=rec["recording_id"],
                            upload_status="uploaded",
                            google_drive_file_id=rec.get("google_drive_file_id", ""),
                            google_drive_path=rec.get("google_drive_path", ""),
                            local_cache_path="",
                        )
                        print(f"[retention] Pruned uploaded video {fname} to maintain disk budget (Free: {free_gb:.1f}GB)")
                    except OSError:
                        pass
        except Exception as e:
            print(f"[retention] error pruning buffer: {e}")

