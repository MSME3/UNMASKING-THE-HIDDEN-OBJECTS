"""
Utility script to migrate any leftover local frames, clips, and recordings to Google Drive,
and clean up local storage on Jetson Nano.
"""

import os
import glob
from backend.gdrive import gdrive_storage

def sync_all():
    if not gdrive_storage.is_configured():
        print("[sync] Google Drive is not configured or authenticated. Please verify credentials.json / token.json.")
        return

    frames = glob.glob("cloud_storage/frames/*.jpg")
    clips = glob.glob("cloud_storage/clips/*.mp4")
    recordings = glob.glob("live_recordings/*.mp4") + glob.glob("cache/recordings/*.mp4")

    print(f"[sync] Starting sync: {len(frames)} frames, {len(clips)} clips, {len(recordings)} recordings.")

    for f in frames:
        try:
            gdrive_storage.upload_media_and_cleanup(f, folder_name="frames")
        except Exception as e:
            print(f"[sync] Failed uploading frame {f}: {e}")

    for c in clips:
        try:
            gdrive_storage.upload_media_and_cleanup(c, folder_name="clips")
        except Exception as e:
            print(f"[sync] Failed uploading clip {c}: {e}")

    for r in recordings:
        try:
            gdrive_storage.upload_recording_segment(r)
        except Exception as e:
            print(f"[sync] Failed uploading recording {r}: {e}")

    print("[sync] Sync completed successfully.")

if __name__ == "__main__":
    sync_all()

