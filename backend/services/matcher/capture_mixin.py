"""
Camera Capture Loop and Hardware Thread Lifecycle Mixin.
"""
import os
import time
import threading
import queue
import cv2
from backend.config import CAMERA_START_TIMEOUT_SEC, LIVE_AI_INTERVAL_SEC
from backend.camera import open_camera, read_bgr_frame
from backend.database import _db_lock, db_set_live_status


class CaptureMixin:
    """Manages camera capture thread, hardware frame acquisition, and thread lifecycle."""

    def is_running(self):
        return (
            self.running
            and self.capture_thread is not None
            and self.capture_thread.is_alive()
        )

    def start_camera(self):
        """Starts capture, AI worker, and stream encoder threads."""
        if self.is_running():
            return True, "already running"

        self.camera_error = None
        self.camera_source_type = None
        self.camera_source_name = None
        self.stop_event.clear()
        self.running = True
        self.frame_ready.clear()

        # Ensure SegmentedRecorder worker thread is active
        if not hasattr(self, "recorder") or self.recorder is None or not getattr(self.recorder, "running", False):
            from backend.recorder import SegmentedRecorder
            from backend.config import LIVE_RECORDINGS_DIR, RECORDING_SEGMENT_SECONDS
            self.recorder = SegmentedRecorder(
                output_dir=LIVE_RECORDINGS_DIR,
                segment_seconds=RECORDING_SEGMENT_SECONDS,
                on_segment_closed=getattr(self, "_on_segment_closed", None),
            )

        self.recorder.set_fps(self.camera_args.get("fps", 30))

        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera-capture"
        )
        self.ai_thread = threading.Thread(
            target=self._ai_loop, daemon=True, name="ai-matcher"
        )
        self.stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="stream-encoder"
        )

        self.ai_thread.start()
        self.stream_thread.start()
        self.capture_thread.start()

        waited = 0.0
        while waited < CAMERA_START_TIMEOUT_SEC:
            if self.camera_error:
                self.running = False
                self.stop_event.set()
                return False, self.camera_error
            if self.frame_ready.is_set():
                return True, "started"
            time.sleep(0.1)
            waited += 0.1

        return True, "starting"

    def stop_camera(self):
        """Stops capture, AI, and stream encoder threads cleanly."""
        if not self.running and self.capture_thread is None:
            return True

        self.running = False
        self.stop_event.set()

        with self.stream_lock:
            self.stream_condition.notify_all()

        try:
            self.ai_queue.put_nowait(None)
        except queue.Full:
            try:
                self.ai_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.ai_queue.put_nowait(None)
            except queue.Full:
                pass

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=3)
        if self.stream_thread is not None:
            self.stream_thread.join(timeout=2)
        if self.ai_thread is not None:
            self.ai_thread.join(timeout=5)

        self.capture_thread = None
        self.stream_thread = None
        self.ai_thread = None
        self.camera_source_type = None
        self.camera_source_name = None
        self.recorder.close()

        with self.frame_condition:
            self.latest_annotated_frame = None
            self.latest_stream_jpeg = None
            self.frame_ready.clear()
            self.frame_condition.notify_all()

        with self.lock:
            stopped_ids = list(self.active_queries.keys())
            self.active_queries.clear()

        for qid in stopped_ids:
            with _db_lock:
                q = self.conn.execute("SELECT live_status FROM queries WHERE query_id=?", (qid,)).fetchone()
                if q and q["live_status"] == "disabled":
                    continue
            db_set_live_status(self.conn, qid, "stopped")

        return True

    def _capture_loop(self):
        cap = None
        flip_val = self.camera_args.get("flip_method", 2)
        try:
            cap, cam_mode = open_camera(
                sensor_id=self.camera_args.get("sensor_id", 0),
                display_width=self.camera_args.get("width", 1920),
                display_height=self.camera_args.get("height", 1080),
                framerate=self.camera_args.get("fps", 30),
                flip_method=flip_val,
            )
        except Exception as e:
            self.camera_error = str(e)
            self.camera_source_type = None
            self.camera_source_name = None
            self.running = False
            self.stop_event.set()
            return

        sensor_cfg = self.camera_args.get("sensor_id", 0)
        if cam_mode in ("simulated_video", "video_loop"):
            self.camera_source_type = "simulated"
            self.camera_source_name = os.path.basename(str(sensor_cfg))
        elif cam_mode == "rtsp_stream":
            self.camera_source_type = "network"
            self.camera_source_name = str(sensor_cfg)
        else:
            self.camera_source_type = "physical"
            self.camera_source_name = f"Sensor {sensor_cfg}"

        width = self.camera_args.get("width", 1920)
        height = self.camera_args.get("height", 1080)
        last_expire_check = time.time()
        last_retry_check = time.time()
        consecutive_failures = 0

        print(f"[live-matcher] camera capture started (flip_method={flip_val})")

        try:
            while self.running and not self.stop_event.is_set():
                ret, frame = read_bgr_frame(cap, cam_mode, height, width)

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 90:  # ~3 seconds of no frames
                        print("[live-matcher] WARNING: Camera stream stalled (90 consecutive read failures). Attempting auto-recovery...")
                        try:
                            if cap is not None:
                                cap.release()
                        except Exception:
                            pass
                        time.sleep(1.0)
                        try:
                            cap, cam_mode = open_camera(
                                sensor_id=self.camera_args.get("sensor_id", 0),
                                display_width=width,
                                display_height=height,
                                framerate=self.camera_args.get("fps", 30),
                                flip_method=flip_val,
                            )
                            consecutive_failures = 0
                            print("[live-matcher] Camera auto-recovery successful. Resuming capture.")
                        except Exception as recov_err:
                            print(f"[live-matcher] Camera auto-recovery attempt failed: {recov_err}")
                            time.sleep(2.0)
                    else:
                        if self.stop_event.wait(0.03):
                            break
                    continue

                consecutive_failures = 0

                if cam_mode == "v4l2_raw" and flip_val != 0:
                    if flip_val in (1, 3, 5):
                        frame = cv2.flip(frame, 0)
                    elif flip_val == 4:
                        frame = cv2.flip(frame, 1)
                    elif flip_val in (2, 6):
                        frame = cv2.flip(frame, -1)

                self.recorder.write(frame)
                if self.camera_source_type == "physical":
                    self.recorder.write(frame)
                self.frame_buffer.append(frame.copy())

                now = time.time()
                with self._stats_lock:
                    self._capture_times.append(now)

                with self.stream_lock:
                    self.latest_raw_frame = (frame, now)
                    self.stream_condition.notify_all()

                if self.active_query_ids() and now - self.last_ai_submit >= LIVE_AI_INTERVAL_SEC:
                    self.last_ai_submit = now
                    frame_copy = frame.copy()
                    item = (frame_copy, now)
                    try:
                        self.ai_queue.put_nowait(item)
                    except queue.Full:
                        try:
                            self.ai_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.ai_queue.put_nowait(item)
                        except queue.Full:
                            pass

                if now - last_expire_check >= 5:
                    self._expire_stale_queries()
                    last_expire_check = now

                if now - last_retry_check >= 30:
                    self._retry_pending_jobs()
                    last_retry_check = now

        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.recorder.close()
            self.running = False
            print("[live-matcher] camera capture stopped")

