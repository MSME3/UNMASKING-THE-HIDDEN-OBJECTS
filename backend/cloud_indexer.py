"""
Continuous background Google Drive cloud archive indexer.
Indexes all historical 5-minute video segments stored in Google Drive across all dates into the FAISS vector database.
"""
import os
import gc
import time
import sqlite3
import cv2
import torch
from datetime import datetime

from .config import DB_PATH, INDEX_DIR, TEMP_DOWNLOADS_DIR, INDEX_SAMPLES_PER_SEC
from .engine import Engine
from .index_manager import IndexManager
from .gdrive import gdrive_storage
from .matcher import is_valid_mp4


def index_all_cloud_recordings(limit=None, max_workers=1, progress_callback=None):
    """
    Indexes all unindexed Google Drive recordings into the FAISS vector database.
    """
    if not gdrive_storage.is_configured():
        print("[cloud-indexer] Google Drive is not configured.")
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 1. Sync any new files from Google Drive first
    try:
        gdrive_storage.sync_all_drive_recordings_to_db(conn)
    except Exception as e:
        print(f"[cloud-indexer] Note syncing drive: {e}")

    # 2. Get unindexed recordings
    with conn:
        query = (
            "SELECT * FROM recordings "
            "WHERE index_status != 'indexed' AND (google_drive_file_id != '' OR upload_status = 'uploaded') "
            "ORDER BY start_time DESC"
        )
        if limit:
            query += f" LIMIT {int(limit)}"
        unindexed_rows = [dict(r) for r in conn.execute(query).fetchall()]

    if not unindexed_rows:
        print("[cloud-indexer] All Google Drive recordings are already indexed!")
        conn.close()
        return 0

    print(f"[cloud-indexer] Starting indexing of {len(unindexed_rows)} cloud recordings...")
    engine = Engine()
    idx_mgr = IndexManager(INDEX_DIR)
    os.makedirs(TEMP_DOWNLOADS_DIR, exist_ok=True)

    indexed_count = 0
    total_to_index = len(unindexed_rows)

    for i, rec in enumerate(unindexed_rows, 1):
        fname = rec.get("filename", "")
        fid = rec.get("google_drive_file_id", "")
        rec_id = rec.get("recording_id", "")

        if not fid:
            fid = gdrive_storage.get_file_id_by_name(fname)
            if fid:
                try:
                    conn.execute("UPDATE recordings SET google_drive_file_id=? WHERE recording_id=?", (fid, rec_id))
                    conn.commit()
                except Exception:
                    pass

        if not fid:
            print(f"[cloud-indexer] [{i}/{total_to_index}] Skipping {fname}: Drive file ID not found.")
            continue

        temp_path = os.path.join(TEMP_DOWNLOADS_DIR, f"idx_{fname}")
        try:
            print(f"[cloud-indexer] [{i}/{total_to_index}] Downloading {fname} ({rec.get('start_time')})...")
            gdrive_storage.download_file(fid, temp_path, timeout=60)
            
            if not os.path.exists(temp_path) or not is_valid_mp4(temp_path):
                print(f"[cloud-indexer] [{i}/{total_to_index}] Download invalid for {fname}.")
                continue

            # Fast frame extraction
            cap = cv2.VideoCapture(temp_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9000
            sample_step = max(1, int(round(fps / max(1, INDEX_SAMPLES_PER_SEC))))
            embeddings, metadatas = [], []

            frame_idx = 0
            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_sec = round(frame_idx / fps, 2)
                try:
                    with torch.inference_mode():
                        targets = engine.extract_targets(frame, extract_plates=False)
                except Exception:
                    targets = []

                for t in targets:
                    embeddings.append(t["emb"])
                    metadatas.append({
                        "video_path": "",
                        "video_name": fname,
                        "google_drive_file_id": fid,
                        "google_drive_path": f"https://drive.google.com/file/d/{fid}/view",
                        "time": timestamp_sec,
                        "bbox": t["bbox"],
                        "type": t["type"],
                    })

                frame_idx += sample_step

            cap.release()

            if embeddings:
                total_vecs = idx_mgr.add_vectors(embeddings, metadatas)
                print(f"[cloud-indexer] [{i}/{total_to_index}] SUCCESS: {fname} -> {len(embeddings)} vectors (Total in index: {total_vecs})")
            else:
                print(f"[cloud-indexer] [{i}/{total_to_index}] SUCCESS: {fname} -> 0 targets found.")

            with conn:
                conn.execute("UPDATE recordings SET index_status='indexed' WHERE recording_id=?", (rec_id,))
                conn.commit()

            indexed_count += 1
            if progress_callback:
                progress_callback(indexed_count, total_to_index, fname)

        except Exception as e:
            print(f"[cloud-indexer] [{i}/{total_to_index}] Error processing {fname}: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    conn.close()
    print(f"[cloud-indexer] Finished! Successfully indexed {indexed_count}/{total_to_index} recordings.")
    return indexed_count


if __name__ == "__main__":
    index_all_cloud_recordings()

