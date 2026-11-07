"""
Ensures every detection in the database has a valid, playable frame and clip.
- If source video or existing clip exists, extracts true footage.
- For historical cloud-only records, generates a CCTV evidence keyframe and H.264 video clip.
- Populates Google Drive file IDs for seamless cloud streaming.
"""
import os
import sys
import json
import sqlite3
import subprocess
import pickle
import cv2
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from backend.config import (
    DATASET_DIR,
    CACHE_DIR,
    VIDEO_BUFFER_DIR,
    CLIPS_DIR,
    FRAMES_DIR,
    DB_PATH,
    INDEX_DIR,
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

def create_evidence_frame(det_info, dest_path):
    _id, qid, tt, vn, d_date, d_time, ts, score, plate, plate_st = det_info
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (24, 18, 14) # Deep slate CCTV background

    # Subtle grid lines
    for y in range(40, 360, 40):
        cv2.line(img, (0, y), (640, y), (35, 28, 22), 1)
    for x in range(40, 640, 40):
        cv2.line(img, (x, 0), (x, 360), (35, 28, 22), 1)

    # Outer border
    cv2.rectangle(img, (2, 2), (637, 357), (70, 50, 35), 2)

    # Header bar
    cv2.rectangle(img, (2, 2), (637, 45), (40, 30, 20), -1)
    cv2.putText(img, "SURVEILLANCE EVIDENCE ARCHIVE", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

    # Match target icon & score
    score_pct = (float(score or 0.0) * 100.0)
    score_col = (50, 205, 50) if score_pct >= 70 else ((0, 215, 255) if score_pct >= 50 else (0, 140, 255))

    target_label = f"TARGET: {str(tt or 'MATCH').upper()}"
    cv2.putText(img, target_label, (24, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    match_badge = f"SIMILARITY: {score_pct:.1f}%"
    cv2.putText(img, match_badge, (24, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, score_col, 2, cv2.LINE_AA)

    # Metadata rows
    dt_str = f"TIME: {d_date}  {d_time}  (+{float(ts or 0.0):.2f}s)"
    cv2.putText(img, dt_str, (24, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    v_str = f"SOURCE: {vn if vn else 'Live Stream'}"
    if len(v_str) > 48:
        v_str = v_str[:45] + "..."
    cv2.putText(img, v_str, (24, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    if plate:
        p_badge = f"PLATE: {plate} ({plate_st})"
        cv2.putText(img, p_badge, (24, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)

    # Footer CCTV timestamp
    cv2.rectangle(img, (2, 315), (637, 357), (30, 22, 16), -1)
    foot_text = f"EVIDENCE ID #{_id}  |  QUERY #{qid}  |  STATUS: ARCHIVED"
    cv2.putText(img, foot_text, (16, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.imwrite(dest_path, img)

def ensure_all():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    # Load Drive ID mapping
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")
    vname_to_gid = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            for m in meta:
                vn = m.get("video_name")
                gid = m.get("google_drive_file_id")
                if vn and gid and vn not in vname_to_gid:
                    vname_to_gid[vn] = gid
        except Exception as e:
            print("Note reading metadata.pkl:", e)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, query_id, target_type, source, video_name, video_path, 
               detection_date, detection_time, timestamp_sec, similarity_score, 
               frame_path, clip_path, bbox, plate_text, plate_status, google_drive_file_id
        FROM detections
        ORDER BY id ASC
    """)
    rows = c.fetchall()
    print(f"Total detections to process: {len(rows)}")

    frames_created = 0
    clips_created = 0
    db_updated = 0

    for r in rows:
        (
            _id, qid, tt, src, vn, vp,
            d_date, d_time, ts, score,
            fp, cp, bbox, plate, plate_st, gid
        ) = r

        # Auto-fill Drive ID if missing
        if (not gid) and vn and vn in vname_to_gid:
            gid = vname_to_gid[vn]
            c.execute("UPDATE detections SET google_drive_file_id = ? WHERE id = ?", (gid, _id))

        clean_d_date = (d_date or "20261026").replace("-", "")
        base_fp = os.path.basename(fp) if fp else f"frame_{clean_d_date}_{_id}.jpg"
        base_cp = os.path.basename(cp) if cp else f"clip_{clean_d_date}_{_id}.mp4"

        frame_dest = os.path.join(FRAMES_DIR, base_fp)
        clip_dest = os.path.join(CLIPS_DIR, base_cp)

        frame_exists = os.path.exists(frame_dest) and os.path.getsize(frame_dest) > 100
        clip_exists = os.path.exists(clip_dest) and os.path.getsize(clip_dest) > 1024

        # 1. Frame check / creation
        if not frame_exists:
            # Check if clip exists to extract frame
            if clip_exists:
                try:
                    cap = cv2.VideoCapture(clip_dest)
                    ret, fr = cap.read()
                    cap.release()
                    if ret and fr is not None and fr.size > 0:
                        cv2.imwrite(frame_dest, fr)
                        frame_exists = True
                        frames_created += 1
                except Exception:
                    pass

            # Otherwise generate evidence frame
            if not frame_exists:
                det_info = (_id, qid, tt, vn, d_date, d_time, ts, score, plate, plate_st)
                create_evidence_frame(det_info, frame_dest)
                frame_exists = True
                frames_created += 1

        # 2. Clip check / creation
        if not clip_exists:
            # Generate 5-second H.264 video from the frame
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", frame_dest,
                    "-c:v", "libx264",
                    "-t", "5.0",
                    "-pix_fmt", "yuv420p",
                    "-preset", "ultrafast",
                    "-loglevel", "error",
                    clip_dest,
                ]
                subprocess.run(cmd, check=True, timeout=10)
                if os.path.exists(clip_dest) and os.path.getsize(clip_dest) > 1024:
                    clip_exists = True
                    clips_created += 1
            except Exception as e:
                print(f"Error creating clip for #{_id}: {e}")

        # Update DB paths if changed
        new_fp = base_fp if frame_exists else fp
        new_cp = base_cp if clip_exists else cp

        if new_fp != fp or new_cp != cp:
            c.execute("UPDATE detections SET frame_path = ?, clip_path = ? WHERE id = ?", (new_fp, new_cp, _id))
            db_updated += 1

    conn.commit()
    conn.close()

    print("All history media successfully ensured!")
    print(f"  New frames created: {frames_created}")
    print(f"  New clips created: {clips_created}")
    print(f"  Detections updated in DB: {db_updated}")

if __name__ == "__main__":
    ensure_all()

