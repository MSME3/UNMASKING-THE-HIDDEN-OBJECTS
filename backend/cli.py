"""
CLI commands and argument parser for Jetson Nano AI Vision Search.
"""
import argparse
import atexit
import glob
import os
import pickle
import signal
import sys
import time
import uuid
import warnings
from collections import defaultdict
import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from .config import (
    DATASET_DIR,
    INDEX_DIR,
    LIVE_RECORDINGS_DIR,
    FACE_THRESHOLD,
    VEHICLE_THRESHOLD,
    CAMERA_SENSOR_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
)
from .database import init_db, db_add_query, db_set_target_summary
from .engine import Engine, _target_summary, _print_target_breakdown
from .index_manager import IndexManager
from .camera import open_camera, read_bgr_frame, create_display_window
from .recorder import add_timestamp_overlay
from .matcher import LiveMatcher
from .app import create_app


def cmd_record(args):
    """OPTIONAL: manually record footage with the camera module."""
    os.makedirs(args.output_dir, exist_ok=True)
    cap, cam_mode = open_camera(args.sensor_id, args.width, args.height, args.fps)
    win_title = "Jetson Recorder - Camera Module"
    create_display_window(win_title, args.win_width, args.win_height)

    print("Press 'r' to start/stop recording, 'q' to quit.")
    writer, recording, clip_path, start_time = None, False, None, None

    try:
        while True:
            ret, frame = read_bgr_frame(cap, cam_mode, args.height, args.width)
            if not ret:
                print("ERROR: Could not capture frame")
                break
            display = frame.copy()
            if recording:
                elapsed = time.time() - start_time
                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC {elapsed:0.1f}s", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                rec_frame = frame.copy()
                add_timestamp_overlay(rec_frame)
                writer.write(rec_frame)
                if args.duration and elapsed >= args.duration:
                    print(f"Reached duration limit, stopping: {clip_path}")
                    writer.release()
                    writer, recording = None, False

            add_timestamp_overlay(display)
            cv2.imshow(win_title, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if not recording:
                    clip_path = os.path.join(args.output_dir, f"clip_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              args.fps, (args.width, args.height))
                    recording, start_time = True, time.time()
                    print(f"Recording started: {clip_path}")
                else:
                    writer.release()
                    writer, recording = None, False
                    print(f"Recording stopped: {clip_path}")
            elif key == ord("q"):
                break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped")


def cmd_index(args):
    """OPTIONAL: bulk-build FAISS index from an existing dataset/ folder."""
    engine = Engine(use_gpu=not args.cpu, skip_vehicle_reid=args.skip_vehicle_reid)
    import faiss

    video_files = sorted(
        glob.glob(os.path.join(args.dataset_dir, "**", "*.mp4"), recursive=True) +
        glob.glob(os.path.join(args.dataset_dir, "**", "*.avi"), recursive=True)
    )
    if not video_files:
        print(f"No videos found in {args.dataset_dir}")
        return

    stored_embeddings, metadata = [], []
    print(f"Indexing {len(video_files)} stored video(s)...")

    for vid_path in video_files:
        video_name = os.path.basename(vid_path)
        cap = cv2.VideoCapture(vid_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        sample_interval = max(1, fps // args.samples_per_sec)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_count % sample_interval == 0:
                timestamp_sec = round(frame_count / fps, 2)
                for t in engine.extract_targets(frame, det_thresh=args.det_thresh):
                    stored_embeddings.append(t["emb"])
                    metadata.append({
                        "video_path": os.path.abspath(vid_path),
                        "video_name": video_name,
                        "time": timestamp_sec,
                        "bbox": t["bbox"],
                        "type": t["type"],
                    })
            frame_count += 1
        cap.release()
        print(f"  {video_name}: scanned")

    if not stored_embeddings:
        print("No persons/vehicles detected in any stored video. Nothing to index.")
        return

    type_counts = defaultdict(int)
    for m in metadata:
        type_counts[m["type"]] += 1
    print("Indexed vector breakdown by type:")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    from .index_manager import IndexManager
    mgr = IndexManager(args.index_dir)
    total = mgr.add_vectors(stored_embeddings, metadata)
    print(f"Indexed {len(stored_embeddings)} new vectors from {len(video_files)} video(s) -> Total in index: {total} ({args.index_dir})")


def cmd_live(args):
    """Console-only live matching against one query image (debug mode)."""
    conn = init_db()
    engine = Engine(use_gpu=not args.cpu, skip_vehicle_reid=args.skip_vehicle_reid,
                     face_threshold=args.face_threshold, vehicle_threshold=args.vehicle_threshold,
                     enable_lpr=not args.disable_lpr)
    query_frame = cv2.imread(args.query_image)
    if query_frame is None:
        print(f"Could not read query image: {args.query_image}")
        return
    query_targets = engine.extract_targets(query_frame)
    if not query_targets:
        print("No person/car/bike detected in the query image.")
        return
    _print_target_breakdown("query image", query_targets)

    query_id = str(uuid.uuid4())[:8]
    db_add_query(conn, query_id, os.path.abspath(args.query_image))
    db_set_target_summary(conn, query_id, _target_summary(query_targets))

    index_manager = IndexManager(INDEX_DIR)
    matcher = LiveMatcher(engine, conn, {
        "sensor_id": args.sensor_id, "width": args.width,
        "height": args.height, "fps": args.fps,
        "flip_method": args.flip_method,
    }, index_manager=index_manager)

    cleaned = False
    def cleanup():
        nonlocal cleaned
        if not cleaned:
            cleaned = True
            print("\n[LIVE] Stopping camera capture cleanly...")
            matcher.stop_camera()
            print("[LIVE] Stopped.")

    atexit.register(cleanup)
    def sig_handler(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    ok, msg = matcher.start_camera()
    if not ok:
        print(f"Camera failed to start: {msg}")
        return
    matcher.add_query(query_id, query_targets)
    print("Live matching running (auto-recording + auto-indexing). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cleanup()


def cmd_serve(args):
    """Main Web Server: records + auto-indexes + dual live/stored searches."""
    conn = init_db()
    engine = Engine(use_gpu=not args.cpu, skip_vehicle_reid=args.skip_vehicle_reid,
                     face_threshold=args.face_threshold, vehicle_threshold=args.vehicle_threshold,
                     enable_lpr=not args.disable_lpr)
    index_manager = IndexManager(INDEX_DIR)
    matcher = LiveMatcher(engine, conn, {
        "sensor_id": args.sensor_id, "width": args.width,
        "height": args.height, "fps": args.fps,
        "flip_method": args.flip_method,
    }, index_manager=index_manager)

    cleaned = False
    def cleanup():
        nonlocal cleaned
        if not cleaned:
            cleaned = True
            print("\n[SERVER] Stopping camera capture and background workers cleanly...")
            matcher.stop_camera()
            print("[SERVER] Shutdown complete.")

    atexit.register(cleanup)

    def sig_handler(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Start camera automatically
    ok, msg = matcher.start_camera()
    print(f"[SERVER] Automatic camera initialization: {msg}")
    print(f"[SERVER] Auto-indexing worker: ACTIVE (5-min recordings saved to {LIVE_RECORDINGS_DIR})")

    app = create_app(engine, conn, matcher)
    print(f"\n=======================================================")
    print(f" Jetson AI Vision Search Platform Ready")
    print(f" Open in browser: http://0.0.0.0:{args.port} (or your LAN IP)")
    print(f"=======================================================\n")
    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cleanup()


def cmd_upload(args):
    """Bulk uploads all pending video recordings from video_buffer/ to Google Drive."""
    conn = init_db()
    from .gdrive import gdrive_storage
    print("=" * 60)
    print(" [Google Drive Upload] Scanning pending video buffer recordings...")
    print("=" * 60)
    res = gdrive_storage.upload_pending_buffer_recordings(conn)
    print(f"[Google Drive Upload] Finished: {res}")


def _parse_sensor_id(val):
    return int(val) if str(val).isdigit() else str(val)


def build_parser():
    parser = argparse.ArgumentParser(description="Jetson Nano AI Vision Search Platform")
    sub = parser.add_subparsers(dest="mode", required=True)
    cam = dict(sensor_id=CAMERA_SENSOR_ID, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, win_width=1280, win_height=720)

    p_upload = sub.add_parser("upload", help="Upload all pending recordings from video_buffer to Google Drive")
    p_upload.set_defaults(func=cmd_upload)

    p_record = sub.add_parser("record", help="OPTIONAL: manually record videos with camera module")
    p_record.add_argument("--output-dir", default=DATASET_DIR)
    p_record.add_argument("--sensor-id", type=_parse_sensor_id, default=cam["sensor_id"])
    p_record.add_argument("--width", type=int, default=cam["width"])
    p_record.add_argument("--height", type=int, default=cam["height"])
    p_record.add_argument("--win-width", type=int, default=cam["win_width"])
    p_record.add_argument("--win-height", type=int, default=cam["win_height"])
    p_record.add_argument("--fps", type=int, default=cam["fps"])
    p_record.add_argument("--duration", type=float, default=0)
    p_record.set_defaults(func=cmd_record)

    p_index = sub.add_parser("index", help="OPTIONAL: bulk-build FAISS index from dataset/ folder")
    p_index.add_argument("--dataset-dir", default=DATASET_DIR)
    p_index.add_argument("--index-dir", default=INDEX_DIR)
    p_index.add_argument("--samples-per-sec", type=int, default=5)
    p_index.add_argument("--det-thresh", type=float, default=0.6)
    p_index.add_argument("--cpu", action="store_true")
    p_index.add_argument("--skip-vehicle-reid", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_live = sub.add_parser("live", help="Console-only live matching debug mode")
    p_live.add_argument("--query-image", required=True)
    p_live.add_argument("--sensor-id", type=_parse_sensor_id, default=cam["sensor_id"])
    p_live.add_argument("--width", type=int, default=cam["width"])
    p_live.add_argument("--height", type=int, default=cam["height"])
    p_live.add_argument("--fps", type=int, default=cam["fps"])
    p_live.add_argument("--cpu", action="store_true")
    p_live.add_argument("--skip-vehicle-reid", action="store_true")
    p_live.add_argument("--face-threshold", type=float, default=FACE_THRESHOLD)
    p_live.add_argument("--vehicle-threshold", type=float, default=VEHICLE_THRESHOLD)
    p_live.add_argument("--disable-lpr", action="store_true", help="Disable second-level license plate verification for vehicles")
    p_live.add_argument("--flip-method", type=int, default=2,
                        help="Camera flip method: 0=no flip, 2=180deg (upright unmirrored), 4=horizontal mirror, 6=vertical flip (default: 2)")
    p_live.set_defaults(func=cmd_live)

    p_serve = sub.add_parser("serve", help="Run full web platform (Home page + Search + Live Streaming)")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--sensor-id", type=_parse_sensor_id, default=cam["sensor_id"])
    p_serve.add_argument("--width", type=int, default=cam["width"])
    p_serve.add_argument("--height", type=int, default=cam["height"])
    p_serve.add_argument("--fps", type=int, default=cam["fps"])
    p_serve.add_argument("--cpu", action="store_true")
    p_serve.add_argument("--skip-vehicle-reid", action="store_true")
    p_serve.add_argument("--face-threshold", type=float, default=FACE_THRESHOLD)
    p_serve.add_argument("--vehicle-threshold", type=float, default=VEHICLE_THRESHOLD)
    p_serve.add_argument("--disable-lpr", action="store_true", help="Disable second-level license plate verification for vehicles")
    p_serve.add_argument("--flip-method", type=int, default=2,
                         help="Camera flip method: 0=no flip, 2=180deg (upright unmirrored), 4=horizontal mirror, 6=vertical flip (default: 2)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
