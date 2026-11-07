"""
Google Drive File Storage, Hierarchy Management, and Media Streaming.
"""
import os
import time
import json
import requests
from datetime import datetime


class DriveStorageMixin:
    """Provides folder navigation, resumable file uploads, and media streaming."""

    def get_or_create_folder(self, folder_name, parent_id="root"):
        """Finds or creates a folder by name under parent_id with caching."""
        cache_key = (parent_id, folder_name)
        if cache_key in self.folder_cache:
            return self.folder_cache[cache_key]

        headers = self._get_auth_headers()
        query = (
            f"name = '{folder_name}' and "
            f"'{parent_id}' in parents and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"trashed = false"
        )
        url = f"{self.DRIVE_API_BASE}/files"
        params = {
            "q": query,
            "spaces": "drive",
            "fields": "files(id, name)",
        }
        
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            files = resp.json().get("files", [])
            if files:
                folder_id = files[0]["id"]
                self.folder_cache[cache_key] = folder_id
                return folder_id

        metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id] if parent_id else [],
        }
        create_resp = requests.post(url, headers={**headers, "Content-Type": "application/json"}, json=metadata, timeout=15)
        if create_resp.status_code in (200, 201):
            folder_id = create_resp.json().get("id")
            self.folder_cache[cache_key] = folder_id
            return folder_id
        else:
            raise RuntimeError(f"Failed to create Google Drive folder '{folder_name}': {create_resp.status_code} {create_resp.text}")

    def get_or_create_folder_hierarchy(self, dt=None):
        """Navigates or creates: Google Drive / YEAR / MONTH / DATE / HOUR."""
        dt = dt or datetime.now()
        year_str = dt.strftime("%Y")
        month_str = dt.strftime("%m")
        date_str = dt.strftime("%d")
        hour_str = dt.strftime("%H")

        current_parent = self.root_folder_id or "root"
        for seg in (year_str, month_str, date_str, hour_str):
            current_parent = self.get_or_create_folder(seg, parent_id=current_parent)

        hierarchy_path = f"{year_str}/{month_str}/{date_str}/{hour_str}"
        return current_parent, hierarchy_path

    def get_or_create_subfolder(self, folder_name, parent_id=None):
        """Finds or creates a top-level or subfolder in Google Drive."""
        parent = parent_id or self.root_folder_id or "root"
        return self.get_or_create_folder(folder_name, parent_id=parent)

    def upload_file(self, local_path, parent_folder_id=None, target_filename=None, mime_type=None):
        """Uploads a local file to Google Drive using resumable upload (>2MB) or multipart."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file does not exist: {local_path}")

        file_size = os.path.getsize(local_path)
        filename = target_filename or os.path.basename(local_path)
        if not mime_type:
            ext = os.path.splitext(filename)[1].lower()
            if ext in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            elif ext == ".png":
                mime_type = "image/png"
            elif ext in (".mp4", ".avi", ".mkv", ".mov"):
                mime_type = "video/mp4"
            else:
                mime_type = "application/octet-stream"

        headers = self._get_auth_headers()
        parent_id = parent_folder_id or self.root_folder_id or "root"

        if file_size > 2 * 1024 * 1024:
            init_url = f"{self.DRIVE_UPLOAD_BASE}/files?uploadType=resumable"
            metadata = {
                "name": filename,
                "parents": [parent_id] if parent_id and parent_id != "root" else [],
            }
            init_headers = {
                **headers,
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": mime_type,
                "X-Upload-Content-Length": str(file_size),
            }

            init_resp = requests.post(init_url, headers=init_headers, json=metadata, timeout=20)
            if init_resp.status_code not in (200, 201):
                raise RuntimeError(f"Failed to initiate Google Drive upload session: {init_resp.status_code} {init_resp.text}")

            upload_url = init_resp.headers.get("Location")
            if not upload_url:
                raise RuntimeError("Google Drive did not return a resumable upload Location URL.")

            chunk_size = 8 * 1024 * 1024  # 8MB chunk
            offset = 0
            with open(local_path, "rb") as f:
                while offset < file_size:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    chunk_len = len(chunk)
                    end_byte = offset + chunk_len - 1
                    upload_headers = {
                        "Content-Length": str(chunk_len),
                        "Content-Range": f"bytes {offset}-{end_byte}/{file_size}",
                        "Content-Type": mime_type,
                    }
                    
                    success = False
                    for attempt in range(3):
                        try:
                            upload_resp = requests.put(upload_url, headers=upload_headers, data=chunk, timeout=60)
                            if upload_resp.status_code in (200, 201):
                                res_data = upload_resp.json()
                                return res_data.get("id"), filename, res_data.get("webViewLink", "")
                            elif upload_resp.status_code == 308:
                                offset += chunk_len
                                success = True
                                time.sleep(0.02)
                                break
                            else:
                                time.sleep(1.0)
                        except Exception as chunk_err:
                            if attempt == 2:
                                raise chunk_err
                            time.sleep(1.0)
                    if not success and upload_resp.status_code not in (200, 201, 308):
                        raise RuntimeError(f"Failed uploading file chunk to Google Drive: {upload_resp.status_code} {upload_resp.text}")

            if upload_resp and upload_resp.status_code in (200, 201):
                res_data = upload_resp.json()
                return res_data.get("id"), filename, res_data.get("webViewLink", "")
            else:
                raise RuntimeError("Upload ended without receiving file metadata confirmation.")
        else:
            metadata = {
                "name": filename,
                "parents": [parent_id] if parent_id and parent_id != "root" else [],
            }
            files = {
                "data": ("metadata", json.dumps(metadata), "application/json; charset=UTF-8"),
                "file": (filename, open(local_path, "rb"), mime_type),
            }
            upload_url = f"{self.DRIVE_UPLOAD_BASE}/files?uploadType=multipart"
            resp = requests.post(upload_url, headers=headers, files=files, timeout=30)
            if resp.status_code in (200, 201):
                res_data = resp.json()
                return res_data.get("id"), filename, res_data.get("webViewLink", "")
            else:
                raise RuntimeError(f"Failed multipart upload to Google Drive: {resp.status_code} {resp.text}")

    def upload_media_and_cleanup(self, local_path, folder_name="frames"):
        """Uploads media and immediately unlinks local file from edge storage."""
        if not os.path.exists(local_path):
            return None, os.path.basename(local_path), ""

        filename = os.path.basename(local_path)
        folder_id = self.get_or_create_subfolder(folder_name)
        file_id, drive_fname, web_link = self.upload_file(local_path, parent_folder_id=folder_id, target_filename=filename)

        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass

        print(f"[gdrive] Uploaded {filename} -> Google Drive ({folder_name}/) [ID: {file_id}] and removed local copy.")
        return file_id, drive_fname, web_link

    def stream_file(self, file_id):
        """Streams a file directly from Google Drive chunk by chunk."""
        headers = self._get_auth_headers()
        url = f"{self.DRIVE_API_BASE}/files/{file_id}?alt=media"
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "application/octet-stream")
            return resp.iter_content(chunk_size=65536), content_type
        return None, None

    def download_file(self, file_id, local_dest_path, timeout=15):
        """Downloads a file from Google Drive to local destination path."""
        headers = self._get_auth_headers()
        url = f"{self.DRIVE_API_BASE}/files/{file_id}?alt=media"

        os.makedirs(os.path.dirname(os.path.abspath(local_dest_path)), exist_ok=True)
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Failed to download file {file_id} from Google Drive: {r.status_code} {r.text}")
            with open(local_dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        return local_dest_path

    def extract_fast_clip_and_frame(self, file_id, timestamp_sec, clip_dest_path, frame_dest_path, bbox=None, target_type="target", base_dt=None):
        """Extracts 5-second H.264 clip and single frame directly using HTTP Range seeking."""
        token = self._get_access_token()
        if not token:
            return None, None

        start_sec = max(0.0, float(timestamp_sec) - 2.0)
        relative_offset_sec = max(0.0, float(timestamp_sec) - start_sec)
        url = f"{self.DRIVE_API_BASE}/files/{file_id}?alt=media"
        headers_arg = f"Authorization: Bearer {token}\r\n"

        os.makedirs(os.path.dirname(os.path.abspath(clip_dest_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(frame_dest_path)), exist_ok=True)

        import subprocess
        try:
            cmd = [
                "ffmpeg", "-y",
                "-headers", headers_arg,
                "-ss", f"{start_sec:.2f}",
                "-i", url,
                "-t", "5.0",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-preset", "ultrafast",
                "-loglevel", "error",
                clip_dest_path,
            ]
            subprocess.run(cmd, check=True, timeout=30)
        except Exception as e:
            print(f"[gdrive] Fast range clip extraction note: {e}")

        clean_frame = None
        if os.path.exists(clip_dest_path):
            if os.path.getsize(clip_dest_path) <= 1024:
                try:
                    os.remove(clip_dest_path)
                except Exception:
                    pass
            else:
                try:
                    import cv2
                    import numpy as np
                    from datetime import timedelta
                    cap = cv2.VideoCapture(clip_dest_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0 or np.isnan(fps):
                        fps = 30.0

                    target_frame_idx = int(round(relative_offset_sec * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                    ret, frame = cap.read()

                    if not ret or frame is None or frame.size == 0:
                        cap.set(cv2.CAP_PROP_POS_MSEC, relative_offset_sec * 1000.0)
                        ret, frame = cap.read()

                    if ret and frame is not None and frame.size > 0:
                        clean_frame = frame.copy()
                        actual_dt = (base_dt + timedelta(seconds=float(timestamp_sec))) if base_dt else datetime.now()
                        h, w = frame.shape[:2]
                        annotated = frame.copy()

                        if bbox and len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            x1 = max(0, min(w - 1, int(x1)))
                            y1 = max(0, min(h - 1, int(y1)))
                            x2 = max(0, min(w, int(x2)))
                            y2 = max(0, min(h, int(y2)))
                            if x2 > x1 and y2 > y1:
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        ts_str = actual_dt.strftime("%d-%m-%Y  %H:%M:%S")
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.60 if w >= 1280 else 0.48
                        thick = 2 if w >= 1280 else 1
                        cv2.putText(annotated, ts_str, (20, 30), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
                        cv2.putText(annotated, ts_str, (20, 30), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                        cv2.imwrite(frame_dest_path, annotated)
                except Exception as e:
                    print(f"[gdrive-extract] Frame snapshot extraction error: {e}")

        valid_clip = os.path.exists(clip_dest_path) and os.path.getsize(clip_dest_path) > 1024
        valid_frame = os.path.exists(frame_dest_path) and os.path.getsize(frame_dest_path) > 100
        return (
            os.path.basename(clip_dest_path) if valid_clip else None,
            os.path.basename(frame_dest_path) if valid_frame else None,
            clean_frame,
        )

    def list_all_drive_videos(self, parent_id=None):
        """Lists all MP4 video files in Google Drive recursively with pagination."""
        headers = self._get_auth_headers()
        url = f"{self.DRIVE_API_BASE}/files"
        query = "(mimeType contains 'video/' or name contains '.mp4') and trashed = false"
        if parent_id and parent_id != "root":
            query = f"'{parent_id}' in parents and ({query})"

        all_videos = []
        page_token = None
        while True:
            params = {
                "q": query,
                "spaces": "drive",
                "fields": "nextPageToken, files(id, name, mimeType, size, createdTime, parents)",
                "pageSize": 100,
            }
            if page_token:
                params["pageToken"] = page_token
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    all_videos.extend(data.get("files", []))
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
                else:
                    break
            except Exception as e:
                print(f"[gdrive] Error querying drive files: {e}")
                break
        return all_videos

    def get_file_id_by_name(self, filename):
        """Looks up a Google Drive file ID by exact filename."""
        headers = self._get_auth_headers()
        url = f"{self.DRIVE_API_BASE}/files"
        query = f"name = '{filename}' and trashed = false"
        params = {
            "q": query,
            "spaces": "drive",
            "fields": "files(id, name, size)",
            "pageSize": 1,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 200:
                files = resp.json().get("files", [])
                if files:
                    return files[0]["id"]
        except Exception as e:
            print(f"[gdrive] Error searching file by name '{filename}': {e}")
        return None

    def delete_file(self, file_id):
        """Deletes a file from Google Drive."""
        headers = self._get_auth_headers()
        url = f"{self.DRIVE_API_BASE}/files/{file_id}"
        resp = requests.delete(url, headers=headers, timeout=15)
        return resp.status_code in (200, 204)

