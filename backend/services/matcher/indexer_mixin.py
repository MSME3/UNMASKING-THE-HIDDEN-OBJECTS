"""
Asynchronous Video Indexing Worker and FAISS Vector Persistence Mixin.
"""
import os
import time
import queue
import cv2
import uuid
from datetime import timedelta
from backend.config import (
    INDEX_SAMPLES_PER_SEC,
    RECORDING_SEGMENT_SECONDS,
    TEMP_DOWNLOADS_DIR,
    LIVE_RECORDINGS_DIR,
    DATASET_DIR,
    get_video_start_time,
)
from backend.database import (
    db_get_recording_by_filename,
    db_add_recording,
    db_update_recording_index_status,
    db_get_pending_index_recordings,
)
from backend.recorder import is_valid_mp4
from backend.gdrive import gdrive_storage


class IndexerMixin:
    """Manages the background indexing queue and worker loop for FAISS vector generation."""

    def _enqueue_index_job(self, path):
        """Adds a finalized recording to the AI index queue if not already queued or active."""
        if not path or not os.path.exists(path) or path.endswith((".tmp", ".part")):
            return
        full_path = os.path.abspath(path)
        with self._index_lock:
            if full_path in self._queued_or_active_index:
                return
            self._queued_or_active_index.add(full_path)
            try:
                self.index_queue.put_nowait(full_path)
                print(f"[index-queue] queued: {os.path.basename(full_path)} (queue size: {self.index_queue.qsize()})")
            except queue.Full:
                self._queued_or_active_index.discard(full_path)
                print(f"[index-queue] WARNING: Index queue full, dropped {os.path.basename(full_path)}")

    def _on_segment_closed(self, path):
        """Registers a closed video segment and enqueues it for indexing and uploading."""
        if path and os.path.exists(path) and not path.endswith((".tmp", ".part")):
            full_path = os.path.abspath(path)
            filename = os.path.basename(full_path)
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
                    index_status="pending",
                )
            self._enqueue_index_job(full_path)
            self._enqueue_upload_job(full_path)

    def _scan_unprocessed_recordings(self):
        """Scans video_buffer and database on startup for unindexed recordings."""
        from pathlib import Path

        buffer_path = Path(LIVE_RECORDINGS_DIR)
        if buffer_path.exists():
            now = time.time()
            try:
                for p in sorted(buffer_path.iterdir()):
                    fname = p.name
                    if (
                        p.is_file()
                        and fname.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
                        and not fname.endswith((".tmp", ".part"))
                        and not fname.startswith("_raw_")
                    ):
                        try:
                            if (now - p.stat().st_mtime) < 5.0:
                                continue
                        except OSError:
                            continue
                        full_path = str(p.resolve())
                        if is_valid_mp4(full_path):
                            self._enqueue_index_job(full_path)
                            self._enqueue_upload_job(full_path)
            except Exception as e:
                print(f"[matcher] error scanning video buffer: {e}")

        try:
            pending_indexes = db_get_pending_index_recordings(self.conn)
            for rec in pending_indexes:
                fname = rec.get("filename", "")
                loc_path = rec.get("local_cache_path")
                if not loc_path or not os.path.exists(loc_path):
                    candidate = os.path.join(LIVE_RECORDINGS_DIR, fname)
                    if os.path.exists(candidate):
                        loc_path = candidate
                if loc_path and os.path.exists(loc_path) and is_valid_mp4(loc_path):
                    self._enqueue_index_job(loc_path)
                elif rec.get("google_drive_file_id") or rec.get("upload_status") == "uploaded":
                    self._enqueue_index_job(fname)
        except Exception as e:
            print(f"[matcher] error scanning pending DB recordings: {e}")

        try:
            dataset_path = Path(DATASET_DIR)
            if dataset_path.exists():
                indexed_names = {m.get("video_name") for m in self.index_manager.metadata_snapshot()}
                for p in sorted(dataset_path.iterdir()):
                    fname = p.name
                    if (
                        p.is_file()
                        and fname.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
                        and not fname.endswith((".tmp", ".part"))
                        and fname not in indexed_names
                    ):
                        full_path = str(p.resolve())
                        if is_valid_mp4(full_path):
                            print(f"[matcher] Enqueuing unindexed dataset video: {fname}")
                            self._enqueue_index_job(full_path)
        except Exception as e:
            print(f"[matcher] error scanning dataset directory: {e}")

    def _index_worker_loop(self):
        """Dedicated background worker thread for embedding extraction & FAISS indexing."""
        import gc
        import torch

        print("[ai-index-worker] Background AI index worker ready.")
        while True:
            try:
                full_path = self.index_queue.get()
            except Exception:
                break

            if full_path is None:
                break

            filename = os.path.basename(full_path)
            with self._index_lock:
                self._active_index_file = filename

            rec = db_get_recording_by_filename(self.conn, filename)
            rec_id = rec["recording_id"] if rec else None
            drive_file_id = rec.get("google_drive_file_id") if rec else None
            drive_path = rec.get("google_drive_path") if rec else None

            is_local = os.path.exists(full_path) and not full_path.endswith((".tmp", ".part"))
            temp_dl_path = None

            try:
                metadata = self.index_manager.metadata_snapshot()
                already_indexed = any(m.get("video_name") == filename for m in metadata)
                if already_indexed:
                    if rec_id:
                        db_update_recording_index_status(self.conn, rec_id, "indexed")
                    continue

                if is_local:
                    if not is_valid_mp4(full_path):
                        continue
                    print(f"[auto-index] START (local): {filename}")
                    self._index_video_segment(
                        full_path,
                        original_filename=filename,
                        drive_file_id=drive_file_id,
                        drive_path=drive_path,
                    )
                elif (drive_file_id or gdrive_storage.is_configured()):
                    if not drive_file_id:
                        drive_file_id = gdrive_storage.get_file_id_by_name(filename)

                    if drive_file_id:
                        print(f"[auto-index] START (cloud index): {filename} (ID: {drive_file_id})")
                        temp_dl_path = os.path.join(TEMP_DOWNLOADS_DIR, f"idx_{filename}")
                        try:
                            gdrive_storage.download_file(drive_file_id, temp_dl_path, timeout=60)
                            if os.path.exists(temp_dl_path) and is_valid_mp4(temp_dl_path):
                                self._index_video_segment(
                                    temp_dl_path,
                                    original_filename=filename,
                                    drive_file_id=drive_file_id,
                                    drive_path=drive_path or f"https://drive.google.com/file/d/{drive_file_id}/view",
                                )
                        except Exception as dl_err:
                            print(f"[auto-index] Cloud index download error for {filename}: {dl_err}")
                    else:
                        print(f"[auto-index] Skipped {filename}: Drive file ID not found.")
                else:
                    print(f"[auto-index] Skipped {filename}: not available locally or in Drive.")

                if rec_id:
                    db_update_recording_index_status(self.conn, rec_id, "indexed")
            except Exception as e:
                print(f"[auto-index] FAILED: error indexing {filename}: {e}")
            finally:
                if temp_dl_path and os.path.exists(temp_dl_path):
                    try:
                        os.remove(temp_dl_path)
                    except OSError:
                        pass
                with self._index_lock:
                    self._active_index_file = None
                    self._queued_or_active_index.discard(full_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _index_video_segment(self, video_path, original_filename=None, drive_file_id=None, drive_path=None):
        """Extracts embeddings with cooperative CPU/GPU yields to prevent starving live AI."""
        import gc
        import torch

        if not os.path.exists(video_path) or video_path.endswith((".tmp", ".part")):
            return
        if not is_valid_mp4(video_path):
            return

        video_name = original_filename or os.path.basename(video_path)
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[auto-index] could not open video: {video_name}")
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.camera_args.get("fps", 30)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9000
            sample_step = max(1, int(round(fps / max(1, INDEX_SAMPLES_PER_SEC))))
            embeddings, metadatas = [], []

            frame_idx = 0
            while frame_idx < total_frames:
                if self.active_query_ids():
                    time.sleep(0.04)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                timestamp_sec = round(frame_idx / fps, 2)
                try:
                    with torch.inference_mode():
                        targets = self.engine.extract_targets(frame, extract_plates=False)
                except Exception:
                    targets = []

                for t in targets:
                    embeddings.append(t["emb"])
                    metadatas.append({
                        "video_path": os.path.abspath(video_path) if os.path.exists(video_path) else "",
                        "video_name": video_name,
                        "google_drive_file_id": drive_file_id or "",
                        "google_drive_path": drive_path or (f"https://drive.google.com/file/d/{drive_file_id}/view" if drive_file_id else ""),
                        "time": timestamp_sec,
                        "bbox": t["bbox"],
                        "type": t["type"],
                    })

                frame_idx += sample_step
                time.sleep(0.005)

            if embeddings:
                total = self.index_manager.add_vectors(embeddings, metadatas)
                print(f"[auto-index] SUCCESS: indexed {len(embeddings)} vectors from {video_name} (total: {total})")
            else:
                print(f"[auto-index] SUCCESS: 0 targets found in {video_name} (total: {self.index_manager.total()})")

        finally:
            if cap is not None:
                cap.release()
            del cap
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

