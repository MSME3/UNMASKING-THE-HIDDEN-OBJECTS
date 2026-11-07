"""
Thread-safe FAISS Index Manager for Jetson Nano AI Vision Search.
"""
import os
import pickle
import threading
from collections import defaultdict
import numpy as np
from .config import INDEX_DIR


import json

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


class IndexManager:
    """
    Thread-safe manager for the FAISS vector index and metadata dictionary.
    Shared concurrently between the Flask request handler (search) and the
    background auto-indexing worker (adding new 5-minute video chunks).
    """

    def __init__(self, index_dir=INDEX_DIR):
        self.index_dir = index_dir
        self.lock = threading.Lock()
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        idx_path = os.path.join(self.index_dir, "targets.index")
        meta_path = os.path.join(self.index_dir, "metadata.pkl")
        ver_path = os.path.join(self.index_dir, "index_version.json")
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            import faiss
            with self.lock:
                try:
                    self.index = faiss.read_index(idx_path)
                    with open(meta_path, "rb") as f:
                        self.metadata = pickle.load(f)
                    
                    # Verify Index Version and Vector Dimensions
                    if os.path.exists(ver_path):
                        try:
                            with open(ver_path, "r") as vf:
                                ver_info = json.load(vf)
                            print(f"[IndexManager] Verified Index Version: {ver_info.get('version')} (dim={ver_info.get('dim', 512)})")
                        except Exception as ve:
                            print(f"[IndexManager] Version info read note: {ve}")
                    else:
                        print(f"[IndexManager] Note: index_version.json not found. Assuming schema version {CURRENT_INDEX_VERSION['version']}.")

                    print(f"[IndexManager] loaded {self.index.ntotal} vectors from {self.index_dir}")
                except Exception as e:
                    print(f"[IndexManager] WARNING: Corrupted index/metadata detected: {e}")
                    # Safely preserve corrupted file for inspection
                    import time
                    corrupt_suffix = f".corrupted_{int(time.time())}"
                    try:
                        if os.path.exists(idx_path):
                            os.rename(idx_path, idx_path + corrupt_suffix)
                        if os.path.exists(meta_path):
                            os.rename(meta_path, meta_path + corrupt_suffix)
                        print(f"[IndexManager] Moved corrupted index to *{corrupt_suffix} and initialized fresh index.")
                    except Exception as re_err:
                        print(f"[IndexManager] Could not move corrupted index files: {re_err}")
                    self.index = None
                    self.metadata = []
        else:
            print("[IndexManager] no existing index on disk - will auto-index live recordings.")

    def add_vectors(self, embeddings, metadatas):
        """Thread-safe append and persist to disk."""
        if not embeddings:
            return 0
        import faiss
        with self.lock:
            dim = len(embeddings[0])
            if self.index is None:
                self.index = faiss.IndexFlatIP(dim)
            self.index.add(np.array(embeddings).astype("float32"))
            self.metadata.extend(metadatas)
            total = self.index.ntotal
            self._save_locked()
        return total

    def _save_locked(self):
        """Atomically persist FAISS index and metadata to disk."""
        if self.index is None:
            return
        os.makedirs(self.index_dir, exist_ok=True)
        import faiss
        idx_final = os.path.join(self.index_dir, "targets.index")
        meta_final = os.path.join(self.index_dir, "metadata.pkl")
        ver_final = os.path.join(self.index_dir, "index_version.json")
        
        idx_tmp = os.path.join(self.index_dir, f"targets.index.tmp.{os.getpid()}")
        meta_tmp = os.path.join(self.index_dir, f"metadata.pkl.tmp.{os.getpid()}")
        ver_tmp = os.path.join(self.index_dir, f"index_version.json.tmp.{os.getpid()}")
        
        try:
            faiss.write_index(self.index, idx_tmp)
            with open(meta_tmp, "wb") as f:
                pickle.dump(self.metadata, f)
            with open(ver_tmp, "w") as vf:
                json.dump(CURRENT_INDEX_VERSION, vf, indent=2)
            
            # Atomic replacement (POSIX atomic rename)
            os.replace(idx_tmp, idx_final)
            os.replace(meta_tmp, meta_final)
            os.replace(ver_tmp, ver_final)
        except Exception as e:
            print(f"[IndexManager] ERROR: Failed saving index atomically: {e}")
            for p in (idx_tmp, meta_tmp, ver_tmp):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def search(self, query_vector, k):
        """Thread-safe FAISS search."""
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return None, None
            return self.index.search(query_vector, k)

    def metadata_snapshot(self):
        """Returns a snapshot copy of current metadata."""
        with self.lock:
            return list(self.metadata)

    def type_counts(self):
        """Returns target count per type in the index."""
        with self.lock:
            c = defaultdict(int)
            for m in self.metadata:
                c[m.get("type", "unknown")] += 1
            return dict(c)

    def total(self):
        """Returns total count of vectors currently indexed."""
        with self.lock:
            return self.index.ntotal if self.index is not None else 0

    def update_video_drive_info(self, video_name, drive_file_id, drive_path):
        """Updates Google Drive metadata for entries referencing video_name."""
        with self.lock:
            updated = False
            for m in self.metadata:
                if m.get("video_name") == video_name:
                    m["google_drive_file_id"] = drive_file_id
                    m["google_drive_path"] = drive_path
                    updated = True
            if updated:
                self._save_locked()


def load_index():
    """Standalone loader for offline CLI utilities."""
    idx_path = os.path.join(INDEX_DIR, "targets.index")
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None, None
    import faiss
    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
