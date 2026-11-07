"""
LiveMatcher Service for AI Vision Search Platform.
Coordinates camera ingestion, continuous recording, MJPEG streaming, real-time AI matching,
and asynchronous video indexing/upload workers.
"""
import queue
import threading
from collections import defaultdict, deque
from backend.config import (
    LIVE_CLIP_SECONDS,
    LIVE_RECORDINGS_DIR,
    RECORDING_SEGMENT_SECONDS,
)
from backend.index_manager import IndexManager
from backend.recorder import SegmentedRecorder
from .capture_mixin import CaptureMixin
from .streaming_mixin import StreamingMixin
from .ai_mixin import AIMixin
from .indexer_mixin import IndexerMixin
from .uploader_mixin import UploaderMixin
from .telemetry_mixin import TelemetryMixin


class LiveMatcher(
    CaptureMixin,
    StreamingMixin,
    AIMixin,
    IndexerMixin,
    UploaderMixin,
    TelemetryMixin,
):
    """
    Unified Live Matcher Engine.
    Coordinates:
      - Camera capture thread: frame acquisition + segmented recording + MJPEG publishing.
      - AI worker thread: live face/vehicle target matching against active queries.
      - Auto-indexing worker thread: background vector extraction of completed 5-min video chunks.
      - Drive upload worker thread: background resumable upload of completed recordings.
    """

    def __init__(self, engine, conn, camera_args, index_manager=None):
        self.engine = engine
        self.conn = conn
        self.camera_args = camera_args
        self.index_manager = index_manager if index_manager is not None else IndexManager()

        self.active_queries = {}
        self.last_saved = defaultdict(lambda: defaultdict(float))
        self.best_scores = defaultdict(lambda: defaultdict(float))
        self._last_diag_log = defaultdict(float)
        self.lock = threading.Lock()

        self.frame_buffer = deque(maxlen=int(LIVE_CLIP_SECONDS * 10))
        self.capture_thread = None
        self.ai_thread = None
        self.stream_thread = None
        self.running = False
        self.stop_event = threading.Event()
        self.camera_error = None

        self.latest_annotated_frame = None
        self.latest_stream_jpeg = None
        self.latest_raw_frame = None
        self.stream_lock = threading.Lock()
        self.stream_condition = threading.Condition(self.stream_lock)

        self.frame_lock = threading.Lock()
        self.frame_condition = threading.Condition(self.frame_lock)
        self.frame_seq = 0
        self.frame_ready = threading.Event()

        self.ai_queue = queue.Queue(maxsize=1)
        self.last_ai_submit = 0.0

        # Performance & Telemetry Tracking
        self._stats_lock = threading.Lock()
        self._capture_times = deque(maxlen=60)
        self._stream_times = deque(maxlen=60)
        self._ai_times = deque(maxlen=30)
        self._stream_latencies = deque(maxlen=30)
        self._ai_latencies = deque(maxlen=30)

        # Decoupled Background Queues & Unique Job Tracking
        self.index_queue = queue.Queue(maxsize=100)
        self.upload_queue = queue.Queue(maxsize=100)

        self._index_lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._queued_or_active_index = set()
        self._queued_or_active_upload = set()
        self._active_index_file = None
        self._active_upload_file = None
        self._upload_retry_schedule = {}

        # Unified 5-minute segmented recorder
        self.recorder = SegmentedRecorder(
            output_dir=LIVE_RECORDINGS_DIR,
            segment_seconds=RECORDING_SEGMENT_SECONDS,
            on_segment_closed=self._on_segment_closed,
        )

        # Separate Background Workers (Controlled Concurrency: 1 Indexer, 1 Uploader)
        self.index_thread = threading.Thread(
            target=self._index_worker_loop, daemon=True, name="ai-index-worker"
        )
        self.upload_thread = threading.Thread(
            target=self._upload_worker_loop, daemon=True, name="drive-upload-worker"
        )
        self.index_thread.start()
        self.upload_thread.start()

        print("[LiveMatcher] Background AI index worker and Drive upload worker started.")
        self._scan_unprocessed_recordings()
        threading.Thread(target=self._sync_and_enqueue_all_drive_videos, daemon=True, name="drive-historical-sync").start()


__all__ = ["LiveMatcher"]

