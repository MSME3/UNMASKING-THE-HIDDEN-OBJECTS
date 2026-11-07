"""
Background Google Drive Bulk Video Indexer for Jetson Nano AI Vision Search.
Scans all historical 5-minute video chunks in Google Drive, extracts face and vehicle
embeddings at 1.5 fps, and saves them to the FAISS index with Google Drive file IDs.
"""
import os
import gc
import cv2
import torch
import time
from backend.engine import Engine
from backend.index_manager import IndexManager
from backend.gdrive import gdrive_storage
from backend.config import TEMP_DOWNLOADS_DIR, VIDEO_BUFFER_DIR, DATASET_DIR


def index_all_drive_recordings(sample_fps=1.5, max_videos=None, force_reindex=False):
    print("=" * 60)
    print(" [Drive Indexer] Initializing AI Engine and FAISS Index Manager...")
    print("=" * 60)
    
    os.makedirs(TEMP_DOWNLOADS_DIR, exist_ok=True)
    engine = Engine(use_gpu=True, skip_vehicle_reid=False)
    index_manager = IndexManager()
    
    if not gdrive_storage.is_configured():
        print("[Drive Indexer] ERROR: Google Drive is not configured.")
        return

    print("[Drive Indexer] Querying Google Drive for all video recordings...")
    drive_videos = gdrive_storage.list_all_drive_videos()
    mp4_recordings = [f for f in drive_videos if f.get("name", "").endswith(".mp4") and "REC_" in f.get("name", "")]
    print(f"[Drive Indexer] Found {len(mp4_recordings)} recording chunks in Google Drive.")

    # Check already indexed videos in metadata
    current_meta = index_manager.metadata_snapshot()
    already_indexed = set(m.get("video_name") for m in current_meta if m.get("video_name"))
    print(f"[Drive Indexer] Currently {len(already_indexed)} videos indexed in FAISS ({index_manager.total()} total vectors).")

    unindexed = mp4_recordings if force_reindex else [f for f in mp4_recordings if f["name"] not in already_indexed]
    print(f"[Drive Indexer] Remaining Drive videos to process: {len(unindexed)}")

    if max_videos:
        unindexed = unindexed[:max_videos]
        print(f"[Drive Indexer] Processing batch of {len(unindexed)} videos.")

    if not unindexed:
        print("[Drive Indexer] All Google Drive recordings are already indexed!")
        return

    indexed_count = 0
    total_to_process = len(unindexed)

    for idx, f in enumerate(unindexed, 1):
        vname = f["name"]
        file_id = f["id"]
        print(f"\n[Drive Indexer] [{idx}/{total_to_process}] Processing {vname} (ID: {file_id})...")

        # 1. Check if local copy exists
        local_candidates = [
            os.path.join(VIDEO_BUFFER_DIR, vname),
            os.path.join(DATASET_DIR, vname),
            os.path.join(TEMP_DOWNLOADS_DIR, f"cloud_dl_{vname}"),
        ]
        local_path = None
        for c in local_candidates:
            if os.path.exists(c) and os.path.getsize(c) > 1024:
                local_path = c
                break

        downloaded_temp = False
        if not local_path:
            local_path = os.path.join(TEMP_DOWNLOADS_DIR, f"cloud_dl_{vname}")
            try:
                print(f"[Drive Indexer] Downloading {vname} from Google Drive...")
                gdrive_storage.download_file(file_id, local_path, timeout=120)
                downloaded_temp = True
            except Exception as dl_err:
                print(f"[Drive Indexer] Failed downloading {vname}: {dl_err}")
                continue

        if not os.path.exists(local_path):
            print(f"[Drive Indexer] Skipped {vname}: file not accessible.")
            continue

        # 2. Extract targets
        try:
            cap = cv2.VideoCapture(local_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(fps / sample_fps))

            embeddings = []
            metas = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    sec = frame_idx / fps
                    targets = engine.extract_targets(frame)
                    for t in targets:
                        embeddings.append(t["emb"])
                        metas.append({
                            "video_path": vname,
                            "video_name": vname,
                            "google_drive_file_id": file_id,
                            "time": round(sec, 2),
                            "bbox": t["bbox"],
                            "type": t["type"],
                        })
                frame_idx += 1
            cap.release()

            if embeddings:
                new_total = index_manager.add_vectors(embeddings, metas)
                indexed_count += 1
                print(f"[Drive Indexer] SUCCESS: Indexed {len(embeddings)} targets from {vname}. (FAISS Index Total: {new_total} vectors)")
            else:
                print(f"[Drive Indexer] No targets detected in {vname}.")

        except Exception as proc_err:
            print(f"[Drive Indexer] Error scanning {vname}: {proc_err}")
        finally:
            if downloaded_temp and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f" [Drive Indexer] Batch completed! Successfully indexed {indexed_count} recordings.")
    print(f" [Drive Indexer] Total active vectors in FAISS index: {index_manager.total()}")
    print("=" * 60)


if __name__ == "__main__":
    index_all_drive_recordings()

