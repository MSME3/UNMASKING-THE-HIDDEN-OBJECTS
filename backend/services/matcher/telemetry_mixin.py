"""
Hardware and Queue Performance Telemetry Mixin.
"""
from backend.gdrive import gdrive_storage


class TelemetryMixin:
    """Computes real-time telemetry metrics for FPS, decoupled queues, latency, system, and storage."""

    def get_performance_stats(self):
        def _calc_fps(d):
            if len(d) < 2:
                return 0.0
            dt = d[-1] - d[0]
            return round((len(d) - 1) / dt, 1) if dt > 0 else 0.0

        def _calc_avg(d):
            return round(sum(d) / len(d), 1) if d else 0.0

        rec_stats = self.recorder.get_performance_stats()

        import psutil
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            mem = None
            swap = None
            cpu_pct = 0.0

        try:
            disk_io = psutil.disk_io_counters()
            disk_read_mb = round(disk_io.read_bytes / (1024 * 1024), 1) if disk_io else 0.0
            disk_write_mb = round(disk_io.write_bytes / (1024 * 1024), 1) if disk_io else 0.0
        except Exception:
            disk_read_mb = 0.0
            disk_write_mb = 0.0

        try:
            net_io = psutil.net_io_counters()
            net_sent_mb = round(net_io.bytes_sent / (1024 * 1024), 1) if net_io else 0.0
            net_recv_mb = round(net_io.bytes_recv / (1024 * 1024), 1) if net_io else 0.0
        except Exception:
            net_sent_mb = 0.0
            net_recv_mb = 0.0

        with self._stats_lock:
            cap_fps = _calc_fps(self._capture_times)
            stream_fps = _calc_fps(self._stream_times)
            ai_fps = _calc_fps(self._ai_times)
            stream_lat = _calc_avg(self._stream_latencies)
            ai_lat = _calc_avg(self._ai_latencies)

        with self._index_lock:
            active_idx = self._active_index_file
            unique_idx_cnt = len(self._queued_or_active_index)

        with self._upload_lock:
            active_upl = self._active_upload_file
            unique_upl_cnt = len(self._queued_or_active_upload)
            retry_cnt = len(self._upload_retry_schedule)

        return {
            "fps": {
                "camera_capture_fps": cap_fps,
                "videowriter_encoded_fps": rec_stats["encoded_fps"],
                "actual_recorded_fps": self.camera_args.get("fps", 30),
                "ai_inference_fps": ai_fps,
                "web_streaming_fps": stream_fps,
            },
            "queues": {
                "recorder_queue_size": rec_stats["queue_size"],
                "recorder_queue_maxsize": rec_stats["queue_maxsize"],
                "ai_queue_size": self.ai_queue.qsize(),
                "ai_queue_maxsize": self.ai_queue.maxsize,
                "index_queue_size": self.index_queue.qsize(),
                "upload_queue_size": self.upload_queue.qsize(),
                "unique_pending_indexes": unique_idx_cnt,
                "unique_pending_uploads": unique_upl_cnt,
                "frame_buffer_size": len(self.frame_buffer),
                "frame_buffer_maxsize": self.frame_buffer.maxlen,
            },
            "active_tasks": {
                "active_index_file": active_idx,
                "active_upload_file": active_upl,
                "retry_backoff_jobs": retry_cnt,
            },
            "latency": {
                "stream_latency_ms": stream_lat,
                "ai_latency_ms": ai_lat,
            },
            "system": {
                "cpu_percent": cpu_pct,
                "ram_total_mb": round(mem.total / (1024 * 1024), 1) if mem else 0.0,
                "ram_used_mb": round(mem.used / (1024 * 1024), 1) if mem else 0.0,
                "ram_free_mb": round(mem.free / (1024 * 1024), 1) if mem else 0.0,
                "ram_available_mb": round(mem.available / (1024 * 1024), 1) if mem else 0.0,
                "ram_percent": mem.percent if mem else 0.0,
                "swap_total_mb": round(swap.total / (1024 * 1024), 1) if swap else 0.0,
                "swap_used_mb": round(swap.used / (1024 * 1024), 1) if swap else 0.0,
                "swap_free_mb": round(swap.free / (1024 * 1024), 1) if swap else 0.0,
                "swap_percent": swap.percent if swap else 0.0,
                "disk_read_mb": disk_read_mb,
                "disk_write_mb": disk_write_mb,
                "net_sent_mb": net_sent_mb,
                "net_recv_mb": net_recv_mb,
            },
            "status": {
                "camera_running": self.is_running(),
                "camera_error": self.camera_error,
                "recording_active": rec_stats["is_recording"],
                "current_recording_file": rec_stats["current_file"],
                "active_queries": len(self.active_query_ids()),
                "gdrive_configured": gdrive_storage.is_configured(),
            }
        }

