"""
Real-Time AI Inference Loop, Multi-Target Matching, and Query Lifecycle Mixin.
"""
import os
import time
import uuid
import queue
import threading
from datetime import datetime
import numpy as np
from backend.config import (
    LIVE_QUERY_TTL_SEC,
    LIVE_MATCH_COOLDOWN_SEC,
    is_compatible_target_type,
)
from backend.database import _db_lock, db_set_live_status, db_add_detection
from backend.recorder import save_matched_frame, save_clip_from_live_buffer
from backend.gdrive import gdrive_storage


class AIMixin:
    """Handles query registration, expiration, real-time AI inference, and detection persistence."""

    def add_query(self, query_id, targets):
        """Registers a query for live monitoring."""
        qid_str = str(query_id).strip()
        with self.lock:
            self.active_queries[qid_str] = {
                "targets": targets,
                "last_set": time.time(),
            }
        db_set_live_status(self.conn, qid_str, "monitoring")

    def remove_query(self, query_id, status="stopped"):
        """Removes a query from live monitoring."""
        qid_str = str(query_id).strip()
        with self.lock:
            self.active_queries.pop(qid_str, None)
            self.active_queries.pop(query_id, None)
            self.best_scores.pop(qid_str, None)
            self.best_scores.pop(query_id, None)
            self.last_saved.pop(qid_str, None)
            self.last_saved.pop(query_id, None)
        if status:
            db_set_live_status(self.conn, qid_str, status)

    def active_query_ids(self):
        with self.lock:
            return list(self.active_queries.keys())

    def get_best_scores(self, query_id):
        qid_str = str(query_id).strip()
        with self.lock:
            scores = self.best_scores.get(qid_str) or self.best_scores.get(query_id)
            return dict(scores) if scores else {}

    def _expire_stale_queries(self):
        now = time.time()
        expired = []
        with self.lock:
            for qid, qd in list(self.active_queries.items()):
                if now - qd["last_set"] > LIVE_QUERY_TTL_SEC:
                    expired.append(qid)
                    del self.active_queries[qid]
        for qid in expired:
            with _db_lock:
                q = self.conn.execute("SELECT live_status FROM queries WHERE query_id=?", (qid,)).fetchone()
                if q and q["live_status"] == "disabled":
                    continue
            db_set_live_status(self.conn, qid, "stopped")

    def _ai_loop(self):
        print("[live-matcher] AI worker started")
        while not self.stop_event.is_set():
            try:
                item = self.ai_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            if isinstance(item, tuple):
                frame, capture_ts = item
            else:
                frame, capture_ts = item, time.time()

            if not self.active_query_ids():
                continue
            try:
                self._match_frame(frame)
                now = time.time()
                with self._stats_lock:
                    self._ai_times.append(now)
                    self._ai_latencies.append((now - capture_ts) * 1000.0)
            except Exception as e:
                print(f"[live-matcher] AI worker error: {e}")
        print("[live-matcher] AI worker stopped")

    def _match_frame(self, frame):
        detected = self.engine.extract_targets(frame)
        if not detected:
            return

        with self.lock:
            queries_snapshot = {qid: qd["targets"] for qid, qd in self.active_queries.items()}

        for query_id, query_targets in queries_snapshot.items():
            for detected_target in detected:
                candidates = [
                    q
                    for q in query_targets
                    if is_compatible_target_type(q["type"], detected_target["type"])
                ]
                if not candidates:
                    continue

                best_query = max(
                    candidates,
                    key=lambda qt: float(np.dot(qt["emb"], detected_target["emb"])),
                )
                score = float(np.dot(best_query["emb"], detected_target["emb"]))

                with self.lock:
                    previous_best = self.best_scores[query_id][detected_target["type"]]
                    if score > previous_best:
                        self.best_scores[query_id][detected_target["type"]] = score

                is_match = False
                plate_text = ""
                plate_status = "not_applicable"

                if detected_target["type"] == "person":
                    is_match = (score >= detected_target["thresh"])
                    now_ts = time.time()
                    diag_key = f"{query_id}_person"
                    should_log = (now_ts - self._last_diag_log[diag_key] >= 2.0)
                    if should_log or is_match:
                        self._last_diag_log[diag_key] = now_ts
                        decision_str = "ACCEPTED" if is_match else f"REJECTED (< {detected_target['thresh']:.2f})"
                        print(f"[LIVE-PERSON] Query: {query_id} | Score: {score:.4f} (Thresh: {detected_target['thresh']:.2f}) | Decision: {decision_str} | BBox: {detected_target.get('bbox')}")
                else:
                    x1, y1, x2, y2 = detected_target["bbox"]
                    cand_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                    now_ts = time.time()
                    diag_key = f"{query_id}_{detected_target['type']}"
                    should_log = (now_ts - self._last_diag_log[diag_key] >= 2.0)

                    is_match, plate_text, plate_status, color_sim, decision_reason = (
                        self.engine.verify_vehicle_target_match(
                            best_query,
                            cand_crop,
                            score,
                            context_info="live-camera" if should_log else "",
                            query_id=query_id if should_log else "",
                        )
                    )
                    if should_log or is_match:
                        self._last_diag_log[diag_key] = now_ts

                if not is_match:
                    continue

                now = time.time()
                if now - self.last_saved[query_id][detected_target["type"]] < LIVE_MATCH_COOLDOWN_SEC:
                    continue
                self.last_saved[query_id][detected_target["type"]] = now

                detection_id = str(uuid.uuid4())[:8]
                now_dt = datetime.now()
                frame_path = save_matched_frame(frame, detected_target["bbox"], detection_id, event_dt=now_dt)
                clip_frames = list(self.frame_buffer)
                clip_path = save_clip_from_live_buffer(clip_frames, detection_id, event_dt=now_dt)

                recording_path = self.recorder.current_recording_path()
                recording_start = self.recorder.recording_start_time()
                recording_timestamp = (
                    round(time.time() - recording_start, 2) if recording_start is not None else 0.0
                )

                db_add_detection(
                    self.conn,
                    query_id=query_id,
                    target_type=detected_target["type"],
                    source="live",
                    video_name=os.path.basename(recording_path) if recording_path else "LIVE_CAMERA",
                    video_path=recording_path or clip_path,
                    timestamp_sec=recording_timestamp,
                    similarity_score=score,
                    frame_path=frame_path,
                    clip_path=clip_path,
                    bbox=detected_target["bbox"],
                    plate_text=plate_text or best_query.get("plate_text", ""),
                    plate_status=plate_status,
                )

                if gdrive_storage.is_configured():
                    threading.Thread(
                        target=gdrive_storage.sync_query_history_to_drive,
                        args=(self.conn, query_id),
                        daemon=True,
                    ).start()

                print(f"[live-matcher] MATCH query={query_id} {detected_target['type']} "
                      f"score={score * 100:.1f}%")

