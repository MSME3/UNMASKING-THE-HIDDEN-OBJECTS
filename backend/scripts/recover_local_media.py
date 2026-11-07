"""
Batch extraction and recovery utility for local detections.
Extracts missing high-resolution frames and 5-second video clips for all detections
whose source video is available on the local filesystem.
"""
import os
import sys
import json
import sqlite3
import subprocess
import cv2
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from backend.config import (
    DATASET_DIR,
    CACHE_DIR,
    VIDEO_BUFFER_DIR,
    CLIPS_DIR,
    FRAMES_DIR,
    DB_PATH,
)

def find_source_video(vn, vp):
    candidates = []
    if vp:
        candidates.extend([
            vp,
            os.path.join(DATASET_DIR, os.path.basename(vp)),
            os.path.join(VIDEO_BUFFER_DIR, os.path.basename(vp)),
            os.path.join(CACHE_DIR, 'downloads', os.path.basename(vp)),
            os.path.join(CACHE_DIR, 'downloads', 'cloud_dl_' + os.path.basename(vp)),
        ])
    if vn:
        candidates.extend([
            vn,
            os.path.join(DATASET_DIR, os.path.basename(vn)),
            os.path.join(VIDEO_BUFFER_DIR, os.path.basename(vn)),
            os.path.join(CACHE_DIR, 'downloads', os.path.basename(vn)),
            os.path.join(CACHE_DIR, 'downloads', 'cloud_dl_' + os.path.basename(vn)),
        ])
    for cand in candidates:
        if cand and os.path.exists(cand) and os.path.getsize(cand) > 1024:
            return cand
    return None

def recover_media():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, query_id, target_type, source, video_name, video_path, 
               detection_date, detection_time, timestamp_sec, similarity_score, 
               frame_path, clip_path, bbox, created_at, plate_text
        FROM detections
        ORDER BY id ASC
    """)
    rows = c.fetchall()

    frames_created = 0
    clips_created = 0
    db_updated = 0

    print(f"Total detections to evaluate: {len(rows)}")

    for row in rows:
        (
            _id, qid, tt, src_type, vn, vp,
            d_date, d_time, ts, score,
            fp, cp, bbox_raw, created_at, plate_text
        ) = row

        src_video = find_source_video(vn, vp)
        if not src_video:
            continue

        base_fp = os.path.basename(fp) if fp else f"frame_{d_date.replace('-', '')}_{_id}.jpg"
        base_cp = os.path.basename(cp) if cp else f"clip_{d_date.replace('-', '')}_{_id}.mp4"

        target_frame_path = os.path.join(FRAMES_DIR, base_fp)
        target_clip_path = os.path.join(CLIPS_DIR, base_cp)

        frame_needed = not (os.path.exists(target_frame_path) and os.path.getsize(target_frame_path) > 100)
        clip_needed = not (os.path.exists(target_clip_path) and os.path.getsize(target_clip_path) > 1024)

        if not frame_needed and not clip_needed:
            continue

        ts_sec = float(ts or 0.0)

        # 1. Extract frame if needed
        if frame_needed:
            try:
                cap = cv2.VideoCapture(src_video)
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts_sec * 1000.0))
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    # Draw bbox overlay if present
                    if bbox_raw:
                        try:
                            bbox = json.loads(bbox_raw) if isinstance(bbox_raw, str) and bbox_raw.startswith('[') else None
                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                x1 = max(0, min(w - 1, int(x1)))
                                y1 = max(0, min(h - 1, int(y1)))
                                x2 = max(0, min(w, int(x2)))
                                y2 = max(0, min(h, int(y2)))
                                if x2 > x1 and y2 > y1:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        except Exception:
                            pass

                    # Add timestamp text badge
                    dt_str = f"{d_date}  {d_time}" if (d_date and d_time) else "CCTV ARCHIVE"
                    cv2.putText(frame, dt_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, dt_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.imwrite(target_frame_path, frame)
                    frames_created += 1
            except Exception as e:
                print(f"Error extracting frame for #{_id}: {e}")

        # 2. Extract clip if needed
        if clip_needed:
            try:
                start_sec = max(0.0, ts_sec - 2.0)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_sec:.2f}",
                    "-i", src_video,
                    "-t", "5.0",
                    "-c", "copy",
                    "-avoid_negative_ts", "1",
                    "-loglevel", "error",
                    target_clip_path,
                ]
                subprocess.run(cmd, check=True, timeout=10)
                if os.path.exists(target_clip_path) and os.path.getsize(target_clip_path) > 1024:
                    clips_created += 1
                elif os.path.exists(target_clip_path):
                    os.remove(target_clip_path)
            except Exception as e:
                print(f"Error cutting clip for #{_id}: {e}")

        # Update database with clean basenames
        new_fp = base_fp if os.path.exists(target_frame_path) and os.path.getsize(target_frame_path) > 100 else fp
        new_cp = base_cp if os.path.exists(target_clip_path) and os.path.getsize(target_clip_path) > 1024 else cp

        if new_fp != fp or new_cp != cp:
            c.execute("UPDATE detections SET frame_path = ?, clip_path = ? WHERE id = ?", (new_fp, new_cp, _id))
            db_updated += 1

    conn.commit()
    conn.close()

    print(f"Recovery complete!")
    print(f"  Frames extracted: {frames_created}")
    print(f"  Clips extracted: {clips_created}")
    print(f"  Detections updated in DB: {db_updated}")

if __name__ == "__main__":
    recover_media()

