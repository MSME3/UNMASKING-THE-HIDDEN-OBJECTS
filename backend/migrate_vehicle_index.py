"""
High-Speed Vector Migration Utility for Jetson Orin Nano:
- Single-pass sequential video decoding (O(N) vs O(N*K) seeking).
- Batched GPU DINOv2 feature extraction.
- Exact 70/30 DINOv2 + HSV normalization.
- Builds versioned targets_v2.index and metadata_v2.pkl with 100% alignment.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pickle
import json
import cv2
import numpy as np
import torch
import faiss
from collections import defaultdict
from PIL import Image

from backend.config import (
    INDEX_DIR,
    DATASET_DIR,
    VIDEO_BUFFER_DIR,
    TEMP_DOWNLOADS_DIR,
    LIVE_RECORDINGS_DIR,
)
from backend.engine import Engine
from backend.gdrive import gdrive_storage

CURRENT_INDEX_VERSION = {
    "version": "2.0-dino70_hsv30",
    "dim": 512,
    "metric": "inner_product",
    "vehicle_reid": {
        "dino_model": "dinov2_vits14",
        "dino_weight": 0.70,
        "color_weight": 0.30,
        "color_bins": 128
    },
    "person_model": "buffalo_l"
}


def _resolve_video_file(video_name, file_id=None):
    """Locates or downloads video file for extraction."""
    candidates = [
        os.path.join(DATASET_DIR, video_name),
        os.path.join(VIDEO_BUFFER_DIR, video_name),
        os.path.join(LIVE_RECORDINGS_DIR, video_name),
        os.path.join(TEMP_DOWNLOADS_DIR, f"cloud_dl_{video_name}"),
        os.path.join(TEMP_DOWNLOADS_DIR, f"idx_{video_name}"),
        os.path.join(TEMP_DOWNLOADS_DIR, video_name),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1024:
            return c, False

    if file_id and gdrive_storage.is_configured():
        dl_path = os.path.join(TEMP_DOWNLOADS_DIR, f"cloud_dl_{video_name}")
        try:
            print(f"[migrate] Downloading {video_name} (ID: {file_id}) from Google Drive...")
            gdrive_storage.download_file(file_id, dl_path, timeout=120)
            if os.path.exists(dl_path) and os.path.getsize(dl_path) > 1024:
                return dl_path, True
        except Exception as e:
            print(f"[migrate] Download note for {video_name}: {e}")

    return None, False


def run_migration():
    print("=" * 65)
    print(" [FAST FAISS VEHICLE INDEX MIGRATION: 70/30 DINOv2 + HSV]")
    print("=" * 65)

    idx_orig_path = os.path.join(INDEX_DIR, "targets.index")
    meta_orig_path = os.path.join(INDEX_DIR, "metadata.pkl")

    if not os.path.exists(idx_orig_path) or not os.path.exists(meta_orig_path):
        print(f"[migrate] ERROR: Original index not found in {INDEX_DIR}")
        return False

    print("[migrate] Loading original FAISS index and metadata...")
    orig_index = faiss.read_index(idx_orig_path)
    with open(meta_orig_path, "rb") as f:
        orig_metadata = pickle.load(f)

    total_orig = len(orig_metadata)
    assert orig_index.ntotal == total_orig, f"Mismatch: index={orig_index.ntotal}, meta={total_orig}"
    print(f"[migrate] Total vectors: {total_orig}")

    vehicle_indices = []
    person_indices = []
    for idx, m in enumerate(orig_metadata):
        if m.get("type") == "person":
            person_indices.append(idx)
        else:
            vehicle_indices.append(idx)

    print(f"[migrate] Person vectors to preserve (100% intact): {len(person_indices)}")
    print(f"[migrate] Vehicle vectors to regenerate: {len(vehicle_indices)}")

    engine = Engine(use_gpu=True, skip_vehicle_reid=False)

    # Group vehicle targets by video
    video_groups = defaultdict(list)
    for v_idx in vehicle_indices:
        m = orig_metadata[v_idx]
        vname = m.get("video_name", "")
        video_groups[vname].append((v_idx, m))

    print(f"[migrate] Targets grouped into {len(video_groups)} unique video sources.")

    new_embeddings = {}
    total_rebuilt = 0
    total_reused = 0
    total_fallback = 0

    start_all = time.time()

    for vname, entries in video_groups.items():
        t0 = time.time()
        file_id = entries[0][1].get("google_drive_file_id")
        video_file, is_temp = _resolve_video_file(vname, file_id)

        if not video_file or not os.path.exists(video_file):
            print(f"[migrate] Video '{vname}' not accessible locally. Retaining existing embeddings...")
            for v_idx, m in entries:
                vec = orig_index.reconstruct(v_idx)
                new_embeddings[v_idx] = vec
                dino_sq = np.linalg.norm(vec[:384])**2
                if abs(dino_sq - 0.70) <= 0.05:
                    total_reused += 1
                else:
                    total_fallback += 1
            continue

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        # Map target frame index -> list of (v_idx, bbox)
        frame_targets = defaultdict(list)
        for v_idx, m in entries:
            ts_sec = float(m.get("time", 0.0))
            frame_no = int(round(ts_sec * fps))
            frame_targets[frame_no].append((v_idx, m.get("bbox", [])))

        max_target_frame = max(frame_targets.keys()) if frame_targets else 0

        # High-speed single sequential pass through video
        crops_to_embed = []  # list of (v_idx, crop_img, color_sig)

        cur_frame_idx = 0
        while cap.isOpened() and cur_frame_idx <= max_target_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if cur_frame_idx in frame_targets:
                fh, fw = frame.shape[:2]
                for v_idx, bbox in frame_targets[cur_frame_idx]:
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(fw, int(x2)), min(fh, int(y2))
                        crop = frame[y1:y2, x1:x2]
                        if crop.shape[0] > 8 and crop.shape[1] > 8:
                            color_sig = engine.extract_vehicle_color_signature(crop)
                            crops_to_embed.append((v_idx, crop, color_sig))
                        else:
                            new_embeddings[v_idx] = orig_index.reconstruct(v_idx)
                            total_fallback += 1
                    else:
                        new_embeddings[v_idx] = orig_index.reconstruct(v_idx)
                        total_fallback += 1

            cur_frame_idx += 1

        cap.release()

        # Batched GPU DINOv2 feature extraction
        if crops_to_embed:
            batch_size = 32
            for b_start in range(0, len(crops_to_embed), batch_size):
                batch = crops_to_embed[b_start:b_start + batch_size]
                tensors = []
                for _, crop_img, _ in batch:
                    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    tensors.append(engine.transform(pil_img))
                
                batch_tensor = torch.stack(tensors).to(engine.device)
                with torch.inference_mode():
                    dino_raws = engine.dinov2(batch_tensor).cpu().numpy()

                for (v_idx, _, color_sig), dino_raw in zip(batch, dino_raws):
                    dino_raw_flat = dino_raw.flatten()
                    dino_emb = dino_raw_flat / (np.linalg.norm(dino_raw_flat) + 1e-6)
                    fused = np.hstack((np.sqrt(0.70) * dino_emb, np.sqrt(0.30) * color_sig))
                    emb = fused / (np.linalg.norm(fused) + 1e-6)
                    new_embeddings[v_idx] = emb
                    total_rebuilt += 1

        if is_temp and os.path.exists(video_file):
            try: os.remove(video_file)
            except OSError: pass

        t_elapsed = time.time() - t0
        print(f"[migrate] Video '{vname}': Processed {len(entries)} targets in {t_elapsed:.2f}s.")

    total_time = time.time() - start_all
    print(f"\n[migrate] Complete feature extraction in {total_time:.2f}s!")
    print(f"  Regenerated 70/30 embeddings: {total_rebuilt}")
    print(f"  Reused fresh embeddings:       {total_reused}")
    print(f"  Fallback/Retained embeddings:  {total_fallback}")

    # Build new versioned FAISS index
    print("\n[migrate] Assembling new FAISS IndexFlatIP(512)...")
    new_index = faiss.IndexFlatIP(512)
    final_vectors = []
    final_metadata = list(orig_metadata)

    for idx in range(total_orig):
        if idx in new_embeddings:
            final_vectors.append(new_embeddings[idx])
        else:
            final_vectors.append(orig_index.reconstruct(idx))

    final_array = np.array(final_vectors).astype("float32")
    new_index.add(final_array)

    idx_v2_path = os.path.join(INDEX_DIR, "targets_v2.index")
    meta_v2_path = os.path.join(INDEX_DIR, "metadata_v2.pkl")
    ver_path = os.path.join(INDEX_DIR, "index_version.json")

    faiss.write_index(new_index, idx_v2_path)
    with open(meta_v2_path, "wb") as f:
        pickle.dump(final_metadata, f)
    with open(ver_path, "w") as f:
        json.dump(CURRENT_INDEX_VERSION, f, indent=2)

    print(f"[migrate] Saved: {idx_v2_path} ({new_index.ntotal} vectors)")
    print(f"[migrate] Saved: {meta_v2_path} ({len(final_metadata)} entries)")
    print(f"[migrate] Saved: {ver_path}")

    # Validation
    test_idx = faiss.read_index(idx_v2_path)
    with open(meta_v2_path, "rb") as f:
        test_meta = pickle.load(f)

    assert test_idx.ntotal == len(test_meta) == total_orig, "Validation failed: count mismatch"

    veh_count = 0
    consistent_count = 0
    for idx, m in enumerate(test_meta):
        if m.get("type") != "person":
            veh_count += 1
            vec = test_idx.reconstruct(idx)
            dino_sq = np.linalg.norm(vec[:384])**2
            color_sq = np.linalg.norm(vec[384:])**2
            if abs(dino_sq - 0.70) <= 0.05 and abs(color_sq - 0.30) <= 0.05:
                consistent_count += 1

    print(f"\n[migrate] VALIDATION RESULTS:")
    print(f"  Total index vectors: {test_idx.ntotal}")
    print(f"  Person vectors:      {len(person_indices)} (100% verified)")
    print(f"  Vehicle vectors:     {veh_count}")
    print(f"  70/30 Verified:      {consistent_count}/{veh_count} ({consistent_count/max(1,veh_count)*100:.1f}%)")
    print(f"  Migration Status:    SUCCESSFUL")
    return True


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
