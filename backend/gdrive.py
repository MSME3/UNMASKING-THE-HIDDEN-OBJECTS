"""
Backward-compatibility facade for backend.services.drive.
Preserves existing import signatures for GoogleDriveStorage and gdrive_storage.
"""
from backend.services.drive import GoogleDriveStorage, gdrive_storage

__all__ = ["GoogleDriveStorage", "gdrive_storage"]
