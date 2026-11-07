"""
Google Drive Services Package for AI Vision Search Platform.
Provides modular client authentication, storage operations, and synchronization.
"""
from .client import DriveClient
from .storage import DriveStorageMixin
from .sync import DriveSyncMixin


class GoogleDriveStorage(DriveClient, DriveStorageMixin, DriveSyncMixin):
    """
    Unified Google Drive Storage Service.
    Combines authentication, storage, and synchronization into a single interface.
    """
    pass


# Global singleton instance
gdrive_storage = GoogleDriveStorage()

__all__ = ["GoogleDriveStorage", "gdrive_storage", "DriveClient", "DriveStorageMixin", "DriveSyncMixin"]

