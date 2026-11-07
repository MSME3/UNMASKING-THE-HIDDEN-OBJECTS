"""
Stored video FAISS vector search with real-world historical event timestamp matching.
"""
import os
import uuid
from collections import defaultdict
from datetime import timedelta
import cv2

from .config import (
    TOP_K,
    get_video_start_time,
    TEMP_DOWNLOADS_DIR,
    LIVE_RECORDINGS_DIR,
    DATASET_DIR,
    FRAMES_DIR,
    CLIPS_DIR,
    VEHICLE_THRESHOLD,
    VEHICLE_COLOR_THRESHOLD,
    VEHICLE_PLATE_PROMOTION_THRESHOLD,
    is_compatible_target_type,
)
from .database import db_add_detection, db_get_recording_by_filename, _db_lock
from .gdrive import gdrive_storage
from .recorder import save_matched_frame, save_clip_from_stored_video


def _resolve_local_video_path(video_path, video_name, conn=None):
    """Finds the actual local file path for a stored video if present on the filesystem."""
    if video_path and os.path.exists(video_path):
        return os.path.abspath(video_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(LIVE_RECORDINGS_DIR, video_name) if video_name else None,
        os.path.join(DATASET_DIR, video_name) if video_name else None,
        os.path.join(base_dir, "dataset", video_name) if video_name else None,
        os.path.join(base_dir, "video_buffer", video_name) if video_name else None,
        os.path.join(base_dir, "cache", "recordings", video_name) if video_name else None,
    ]
    if conn and video_name:
        try:
            rec = db_get_recording_by_filename(conn, video_name)
            if rec and rec.get("local_cache_path") and os.path.exists(rec["local_cache_path"]):
                candidates.insert(0, rec["local_cache_path"])
        except Exception:
            pass
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def search_stored_videos(engine, conn, query_id, query_targets, index_manager):
    """
    Performs FAISS nearest neighbor search across stored video embeddings.
    Extracts match clips and logs detection with actual historical wall-clock time.
    """
    metadata = index_manager.metadata_snapshot()
    if not metadata:
        print(
            "[stored-search] index is empty so far - stored videos are auto-indexed in the "
            "background as 5-minute recordings finish; or run index command to bulk index."
        )
        return 0

    indexed_types = defaultdict(int)
    for m in metadata:
        indexed_types[m["type"]] += 1
    vehicle_types_wanted = {t["type"] for t in query_targets if t["type"] != "person"}
    for vt in vehicle_types_wanted:
        matching_count = sum(
            cnt
            for t_name, cnt in indexed_types.items()
            if is_compatible_target_type(t_name, vt)
        )
        if matching_count == 0:
            print(
                f"[stored-search] WARNING: query has a '{vt}' target but the index has "
                f"0 matching vectors ({dict(indexed_types)})."
            )

    match_count = 0
    cached_cloud_downloads = {}  # video_name -> temp_local_path

    try:
        for qt in query_targets:
            query_vector = qt["emb"].reshape(1, -1).astype("float32")
            D, I = index_manager.search(query_vector, TOP_K)
            search_k = min(len(metadata), max(TOP_K, 5000))
            D, I = index_manager.search(query_vector, search_k)
            if D is None:
                continue

            best_per_video = {}
            for score, idx in zip(D[0], I[0]):
                if idx == -1 or idx >= len(metadata):
                    continue
                m = metadata[idx]
                if not is_compatible_target_type(m["type"], qt["type"]):
                    continue

                # Hybrid threshold: allows cross-view plate comparison if score >= 0.35
                min_score = (
                    0.35
                    if (
                        qt["type"] != "person"
                        and getattr(engine, "enable_lpr", False)
                        and qt.get("plate_text")
                    )
                    else qt["thresh"]
                )
                if score < min_score:
                    if qt["type"] == "person" and score >= (min_score - 0.08):
                        print(f"[STORED-PERSON-REJECT] Query: {query_id} | Video: {m.get('video_name')} @ {m.get('time', 0.0):.2f}s | Score: {score:.4f} < Thresh: {min_score:.2f} | Reason: below face similarity threshold")
                    continue
                key = m["video_name"]
                if key not in best_per_video or score > best_per_video[key][0]:
                    best_per_video[key] = (float(score), m)

            # Sort candidate matches by similarity score descending
            sorted_candidates = sorted(best_per_video.values(), key=lambda x: x[0], reverse=True)
            if sorted_candidates and qt["type"] != "person" and not qt.get("plate_text"):
                top_score = sorted_candidates[0][0]
                # Filter out background noise candidates that are significantly weaker than top match
                effective_min = max(0.60, top_score - 0.12)
                sorted_candidates = [c for c in sorted_candidates if c[0] >= effective_min]
            top_score = sorted_candidates[0][0] if sorted_candidates else 0.0
            if sorted_candidates and qt["type"] != "person":
                effective_min = max(0.70, top_score - 0.07)
                if not qt.get("plate_text"):
                    sorted_candidates = [c for c in sorted_candidates if c[0] >= effective_min]
                else:
                    # Allow candidates below effective_min only if within plate promotion range
                    sorted_candidates = [
                        c for c in sorted_candidates
                        if c[0] >= effective_min or c[0] >= VEHICLE_PLATE_PROMOTION_THRESHOLD
                    ]

            for score, m in sorted_candidates:
                with _db_lock:
                    q_row = conn.execute("SELECT stored_status FROM queries WHERE query_id=?", (query_id,)).fetchone()
                    if q_row and q_row["stored_status"] == "disabled":
                        print(f"[stored-search] Query {query_id} stored search was disabled/canceled. Stopping.")
                        return match_count

                video_path = m.get("video_path", "")
                match_timestamp = m.get("time", 0.0)
                video_name = m.get("video_name", "")
                if qt["type"] == "person":
                    print(f"[STORED-PERSON-MATCH] Query: {query_id} | Video: {video_name} @ {match_timestamp:.2f}s | Score: {score:.4f} >= Thresh: {qt['thresh']:.2f} | Decision: ACCEPTED | BBox: {m.get('bbox')}")

                resolved_local = _resolve_local_video_path(video_path, video_name, conn=conn)
                is_local = bool(resolved_local and os.path.exists(resolved_local))
                if is_local:
                    video_path = resolved_local

                # Google Drive Check
                drive_file_id = m.get("google_drive_file_id")
                if not drive_file_id:
                    rec = db_get_recording_by_filename(conn, video_name)
                    if rec and rec.get("google_drive_file_id"):
                        drive_file_id = rec["google_drive_file_id"]

                if not drive_file_id and gdrive_storage.is_configured():
                    try:
                        print(f"[DRIVE] Looking up Google Drive file ID for: {video_name}")
                        drive_file_id = gdrive_storage.get_file_id_by_name(video_name)
                    except Exception as lookup_err:
                        print(f"[DRIVE] Error finding {video_name} in Google Drive: {lookup_err}")
                        drive_file_id = None

                if not drive_file_id and not is_local:
                    print(f"[DRIVE-SEARCH] Skipping video (not found in Google Drive or active buffer): {video_name}")
                    continue

                if drive_file_id:
                    print(f"[DRIVE-SEARCH] Found Google Drive video: {video_name} (File ID: {drive_file_id})")
                else:
                    print(f"[DRIVE-SEARCH] Found active local buffer video: {video_name}")

                video_start = get_video_start_time(video_path if video_path else video_name)
                actual_event_time = video_start + timedelta(seconds=match_timestamp)
                detection_id = str(uuid.uuid4())[:8]

                frame_path = None
                clip_path = None
                drive_view_path = m.get("google_drive_path") or (f"https://drive.google.com/file/d/{drive_file_id}/view" if drive_file_id else video_name)
                plate_text = ""
                plate_status = "not_applicable"

                # 1. Process local copy if already present on disk (zero download time)
                if is_local:
                    try:
                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_MSEC, match_timestamp * 1000)
                        ret, frame = cap.read()
                        cap.release()

                        if ret and frame is not None:
                            # Secondary Target-Specific Verification for Vehicles
                            if qt["type"] != "person":
                                x1, y1, x2, y2 = m.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
                                cand_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                                is_match, plate_text, plate_status, color_sim, decision_reason = (
                                    engine.verify_vehicle_target_match(
                                        qt,
                                        cand_crop,
                                        score,
                                        context_info=f"local {video_name} @ {match_timestamp}s",
                                        query_id=query_id,
                                    )
                                )
                                if not is_match:
                                    continue
                                if plate_status != "verified" and score < max(0.70, top_score - 0.07):
                                    print(f"[stored-search] REJECTED {video_name} @ {match_timestamp}s: visual score {score:.3f} < relative margin {max(0.70, top_score - 0.07):.3f}")
                                    continue

                            frame_path = save_matched_frame(
                                frame, m.get("bbox"), detection_id, event_dt=actual_event_time
                            )
                            clip_path = save_clip_from_stored_video(
                                video_path,
                                match_timestamp,
                                m.get("bbox"),
                                qt["type"],
                                detection_id,
                                base_dt=video_start,
                            )
                    except Exception as e:
                        print(f"[stored-search] Local clip extraction warning: {e}")

                # 2. Fast HTTP Range extraction directly from Google Drive (downloads ~3MB instead of 340MB)
                if not clip_path and drive_file_id and gdrive_storage.is_configured():
                    try:
                        print(f"[DRIVE-FAST] Extracting 5s clip directly from Google Drive: {video_name} (ID: {drive_file_id}) @ {match_timestamp}s...")
                        target_clip_name = f"clip_{actual_event_time.strftime('%Y%m%d_%H%M%S')}_{detection_id}.mp4"
                        target_frame_name = f"frame_{actual_event_time.strftime('%Y%m%d_%H%M%S')}_{detection_id}.jpg"
                        target_clip_dest = os.path.join(CLIPS_DIR, target_clip_name)
                        target_frame_dest = os.path.join(FRAMES_DIR, target_frame_name)

                        extracted_clip, extracted_frame, clean_frame = gdrive_storage.extract_fast_clip_and_frame(
                            drive_file_id,
                            match_timestamp,
                            target_clip_dest,
                            target_frame_dest,
                            bbox=m.get("bbox"),
                            target_type=qt["type"],
                            base_dt=video_start,
                        )
                        if clean_frame is not None and qt["type"] != "person":
                            fh, fw = clean_frame.shape[:2]
                            x1, y1, x2, y2 = m.get("bbox", [0, 0, fw, fh])
                            cand_crop = clean_frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
                            is_match, plate_text, plate_status, color_sim, decision_reason = (
                                engine.verify_vehicle_target_match(
                                    qt,
                                    cand_crop,
                                    score,
                                    context_info=f"drive-fast {video_name} @ {match_timestamp}s",
                                    query_id=query_id,
                                )
                            )
                            if (not is_match) or (plate_status != "verified" and score < max(0.70, top_score - 0.07)):
                                if is_match:
                                    print(f"[stored-search] REJECTED {video_name} @ {match_timestamp}s: visual score {score:.3f} < relative margin {max(0.70, top_score - 0.07):.3f}")
                                if target_clip_dest and os.path.exists(target_clip_dest):
                                    try: os.remove(target_clip_dest)
                                    except OSError: pass
                                if target_frame_dest and os.path.exists(target_frame_dest):
                                    try: os.remove(target_frame_dest)
                                    except OSError: pass
                                continue

                        if extracted_clip:
                            clip_path = extracted_clip
                            print(f"[DRIVE-FAST] SUCCESS: Extracted clip: {clip_path}")
                        if extracted_frame:
                            frame_path = extracted_frame
                            print(f"[DRIVE-FAST] SUCCESS: Extracted frame: {frame_path}")
                    except Exception as fast_err:
                        print(f"[DRIVE-FAST] Fast range extraction note: {fast_err}")

                # 3. Fallback to full temporary download if fast extraction was unavailable
                if not clip_path and drive_file_id and gdrive_storage.is_configured():
                    temp_dl_path = cached_cloud_downloads.get(video_name)
                    if not temp_dl_path or not os.path.exists(temp_dl_path):
                        temp_dl_path = os.path.join(TEMP_DOWNLOADS_DIR, f"cloud_dl_{video_name}")
                        try:
                            print(f"[DRIVE] Downloading video from Google Drive fallback: {video_name} (ID: {drive_file_id})...")
                            gdrive_storage.download_file(drive_file_id, temp_dl_path, timeout=60)
                            if os.path.exists(temp_dl_path):
                                cached_cloud_downloads[video_name] = temp_dl_path
                                print(f"[DRIVE] Successfully cached: {temp_dl_path}")
                        except Exception as dl_err:
                            print(f"[DRIVE] Download error for {video_name}: {dl_err}")
                            temp_dl_path = None

                    if temp_dl_path and os.path.exists(temp_dl_path):
                        try:
                            cap = cv2.VideoCapture(temp_dl_path)
                            cap.set(cv2.CAP_PROP_POS_MSEC, match_timestamp * 1000)
                            ret, frame = cap.read()
                            cap.release()

                            if ret and frame is not None:
                                if qt["type"] != "person":
                                    x1, y1, x2, y2 = m.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
                                    cand_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                                    is_match, plate_text, plate_status, color_sim, decision_reason = (
                                        engine.verify_vehicle_target_match(
                                            qt,
                                            cand_crop,
                                            score,
                                            context_info=f"drive-full {video_name} @ {match_timestamp}s",
                                            query_id=query_id,
                                        )
                                    )
                                    if not is_match:
                                        continue
                                    if plate_status != "verified" and score < max(0.70, top_score - 0.07):
                                        print(f"[stored-search] REJECTED {video_name} @ {match_timestamp}s: visual score {score:.3f} < relative margin {max(0.70, top_score - 0.07):.3f}")
                                        continue

                                frame_path = save_matched_frame(
                                    frame, m.get("bbox"), detection_id, event_dt=actual_event_time
                                )
                                clip_path = save_clip_from_stored_video(
                                    temp_dl_path,
                                    match_timestamp,
                                    m.get("bbox"),
                                    qt["type"],
                                    detection_id,
                                    base_dt=video_start,
                                )
                        except Exception as err:
                            print(f"[stored-search] Cloud video fallback extraction error: {err}")

                # If no clip could be created or candidate was rejected, skip logging detection
                if not clip_path and not is_local:
                    continue

                # Log verified match detection with frame & clip
                db_add_detection(
                    conn,
                    query_id=query_id,
                    target_type=qt["type"],
                    source="stored",
                    video_name=video_name,
                    video_path=drive_view_path,
                    timestamp_sec=match_timestamp,
                    similarity_score=score,
                    frame_path=frame_path or "",
                    clip_path=clip_path or "",
                    bbox=m.get("bbox", []),
                    detection_date=actual_event_time.strftime("%Y-%m-%d"),
                    detection_time=actual_event_time.strftime("%H:%M:%S"),
                    plate_text=plate_text or qt.get("plate_text", ""),
                    plate_status=plate_status,
                )
                match_count += 1
                print(
                    f"[API] Verified Stored match #{match_count}: {qt['type']} in {video_name} @ {match_timestamp}s "
                    f"({score*100:.1f}%) -> Frame: {frame_path} | Clip: {clip_path}"
                )
    finally:
        # Cleanup cached temporary downloads
        for path in cached_cloud_downloads.values():
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    return match_count
