"""
Centralized Configuration and Path Management for AI Vision Search Platform.
Loads environment variables from .env with secure defaults.
"""
import os
import re
from datetime import datetime
from pathlib import Path
from .constants import (
    VEHICLE_YOLO_CLASSES,
    FOUR_WHEEL_TYPES,
    TWO_WHEEL_TYPES,
    is_compatible_target_type,
)


def load_env_file(env_path=None):
    """
    Lightweight, pure-Python .env file parser.
    Populates os.environ without requiring external packages like python-dotenv.
    Does not overwrite existing environment variables.
    """
    if env_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        env_path = base_dir / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.is_file():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Strip wrapping single or double quotes
                if len(val) >= 2 and (
                    (val[0] == '"' and val[-1] == '"')
                    or (val[0] == "'" and val[-1] == "'")
                ):
                    val = val[1:-1]
                if key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"[config] Notice: Could not read .env ({e})")


# Automatically load environment variables upon import
load_env_file()

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CORE_DIR)
BASE_DIR = os.path.dirname(BACKEND_DIR)

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")


def _resolve_dir(env_val, default_name):
    """Ensures a configured path is strictly absolute, anchored at BASE_DIR if relative."""
    val = env_val or default_name
    if os.path.isabs(val):
        return os.path.abspath(val)
    return os.path.abspath(os.path.join(BASE_DIR, val))


DATASET_DIR = _resolve_dir(os.getenv("DATASET_DIR"), "dataset")
INDEX_DIR = _resolve_dir(os.getenv("INDEX_STORE_DIR"), "index_store")
CLOUD_DIR = _resolve_dir(os.getenv("CLOUD_STORAGE_DIR"), "cloud_storage")
CLIPS_DIR = os.path.abspath(os.path.join(CLOUD_DIR, "clips"))
FRAMES_DIR = os.path.abspath(os.path.join(CLOUD_DIR, "frames"))
QUERY_DIR = _resolve_dir(os.getenv("QUERIES_DIR"), "queries")

CACHE_DIR = _resolve_dir(os.getenv("CACHE_DIR"), "cache")
VIDEO_BUFFER_DIR = _resolve_dir(os.getenv("VIDEO_BUFFER_DIR"), "video_buffer")
TEMP_RECORDINGS_DIR = VIDEO_BUFFER_DIR
TEMP_DOWNLOADS_DIR = os.path.abspath(os.path.join(CACHE_DIR, "downloads"))

# Google Drive / Cloud Sync Directory for Live 5-minute recordings
GDRIVE_DIR = os.getenv("GDRIVE_DIR") or os.getenv("DRIVE_RECORDINGS_DIR") or ""
if GDRIVE_DIR and os.path.exists(GDRIVE_DIR):
    LIVE_RECORDINGS_DIR = os.path.abspath(GDRIVE_DIR)
else:
    LIVE_RECORDINGS_DIR = VIDEO_BUFFER_DIR

# Local offline model caches
MODELS_DIR = _resolve_dir(os.getenv("MODELS_DIR"), "models")
INSIGHTFACE_ROOT = _resolve_dir(os.getenv("INSIGHTFACE_ROOT"), os.path.join("models", "insightface"))
TORCH_HOME = _resolve_dir(os.getenv("TORCH_HOME"), os.path.join("models", "torch"))
os.environ["TORCH_HOME"] = TORCH_HOME

DB_PATH = _resolve_dir(os.getenv("DATABASE_PATH"), "detections.db")
YOLO_MODEL_PATH = _resolve_dir(os.getenv("YOLO_MODEL_PATH"), "yolov8n.pt")

# Ensure all runtime directories exist
for _d in (
    DATASET_DIR,
    INDEX_DIR,
    CLOUD_DIR,
    CLIPS_DIR,
    FRAMES_DIR,
    QUERY_DIR,
    VIDEO_BUFFER_DIR,
    LIVE_RECORDINGS_DIR,
    CACHE_DIR,
    TEMP_RECORDINGS_DIR,
    TEMP_DOWNLOADS_DIR,
    TEMPLATES_DIR,
    STATIC_DIR,
    MODELS_DIR,
    INSIGHTFACE_ROOT,
    TORCH_HOME,
):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera Defaults
# ---------------------------------------------------------------------------
CAMERA_SENSOR_RAW = os.getenv("CAMERA_SENSOR_ID", "0")
CAMERA_SENSOR_ID = int(CAMERA_SENSOR_RAW) if CAMERA_SENSOR_RAW.isdigit() else CAMERA_SENSOR_RAW
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
CAMERA_FLIP_METHOD = int(os.getenv("CAMERA_FLIP_METHOD", "2"))

# ---------------------------------------------------------------------------
# Detection & Matching Defaults
# ---------------------------------------------------------------------------
FACE_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.52"))
VEHICLE_THRESHOLD = float(os.getenv("VEHICLE_THRESHOLD", "0.48"))
VEHICLE_COLOR_THRESHOLD = float(os.getenv("VEHICLE_COLOR_THRESHOLD", "0.42"))
VEHICLE_THRESHOLD = float(os.getenv("VEHICLE_THRESHOLD", "0.70"))
VEHICLE_COLOR_THRESHOLD = float(os.getenv("VEHICLE_COLOR_THRESHOLD", "0.65"))
VEHICLE_DINO_MIN = float(os.getenv("VEHICLE_DINO_MIN", "0.82"))
VEHICLE_PLATE_PROMOTION_THRESHOLD = float(os.getenv("VEHICLE_PLATE_PROMOTION_THRESHOLD", "0.45"))
VEHICLE_CONF_MIN = float(os.getenv("VEHICLE_CONF_MIN", "0.25"))
TOP_K = int(os.getenv("TOP_K", "2000"))

MATCH_BOX_PERSIST_SEC = float(os.getenv("MATCH_BOX_PERSIST_SEC", "0.15"))
LIVE_MATCH_COOLDOWN_SEC = float(os.getenv("LIVE_MATCH_COOLDOWN_SEC", "8"))
LIVE_QUERY_TTL_SEC = float(os.getenv("LIVE_QUERY_TTL_SEC", "600"))
LIVE_CLIP_SECONDS = float(os.getenv("LIVE_CLIP_SECONDS", "5"))
CAMERA_START_TIMEOUT_SEC = float(os.getenv("CAMERA_START_TIMEOUT_SEC", "4"))
RECORDING_SEGMENT_SECONDS = int(os.getenv("RECORDING_SEGMENT_SECONDS", "300"))
LIVE_AI_INTERVAL_SEC = float(os.getenv("LIVE_AI_INTERVAL_SEC", "0.40"))
INDEX_SAMPLES_PER_SEC = int(os.getenv("INDEX_SAMPLES_PER_SEC", "1"))

ENABLE_LPR = os.getenv("ENABLE_LPR", "True").lower() in ("true", "1", "yes")
LPR_SIMILARITY_THRESHOLD = float(os.getenv("LPR_SIMILARITY_THRESHOLD", "0.85"))

# Regex to extract YYYYMMDD_HHMMSS from video filenames
_FILENAME_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def get_video_start_time(video_path):
    """
    Recovers the real-world wall-clock start time of a video.
    1) Looks for YYYYMMDD_HHMMSS pattern anywhere in the filename.
    2) Falls back to the file's modification time if no pattern found.
    """
    fname = os.path.basename(video_path)
    m = _FILENAME_TS_RE.search(fname)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(video_path))
    except OSError:
        return datetime.now()

