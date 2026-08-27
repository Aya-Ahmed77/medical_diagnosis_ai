"""
Model file persistence via MongoDB GridFS.

Trained model artifacts (pickled sklearn pipelines, Keras .h5/.keras
files, HuggingFace model directories zipped up, etc.) are serialized to
bytes, stored in GridFS, and the returned ObjectId is what gets written
into the `gridfs_id` field of the Models collection.
"""
import io
import os
import tempfile
import zipfile
from typing import Optional

try:
    import gridfs
    from bson import ObjectId
    _GRIDFS_AVAILABLE = True
except Exception:
    gridfs = None
    ObjectId = None
    _GRIDFS_AVAILABLE = False

from typing import Optional
import io
import os
import tempfile
import zipfile

from app.database.connection import get_db
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _fs():
    if not _GRIDFS_AVAILABLE:
        raise RuntimeError("GridFS (pymongo + gridfs) is not available in this environment")
    return gridfs.GridFS(get_db())


def store_bytes(data: bytes, filename: str, content_type: str = "application/octet-stream") -> str:
    """Store raw bytes in GridFS and return the new file's ObjectId as a string."""
    if not _GRIDFS_AVAILABLE:
        raise RuntimeError("GridFS not available: cannot store bytes")
    file_id = _fs().put(data, filename=filename, content_type=content_type)
    logger.info("Stored '%s' (%d bytes) in GridFS as %s", filename, len(data), file_id)
    return str(file_id)


def store_file(local_path: str, filename: Optional[str] = None) -> str:
    """Read a local file and store its bytes in GridFS."""
    if not _GRIDFS_AVAILABLE:
        raise RuntimeError("GridFS not available: cannot store file")
    filename = filename or os.path.basename(local_path)
    with open(local_path, "rb") as fh:
        return store_bytes(fh.read(), filename)


def store_directory_as_zip(local_dir: str, filename: str) -> str:
    """Zip an entire directory (used for HuggingFace transformer save_pretrained
    output, which is multiple files) and store the archive in GridFS."""
    if not _GRIDFS_AVAILABLE:
        raise RuntimeError("GridFS not available: cannot store directory")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(local_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, local_dir)
                zf.write(full_path, arcname)
    return store_bytes(buffer.getvalue(), filename, content_type="application/zip")


def load_bytes(gridfs_id: str) -> bytes:
    """Retrieve raw bytes for a stored file by its GridFS ObjectId."""
    if not _GRIDFS_AVAILABLE:
        raise RuntimeError("GridFS not available: cannot load bytes")
    grid_out = _fs().get(ObjectId(gridfs_id))
    return grid_out.read()


def load_to_file(gridfs_id: str, destination_path: str) -> str:
    """Download a GridFS file to a local path and return that path."""
    data = load_bytes(gridfs_id)
    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    with open(destination_path, "wb") as fh:
        fh.write(data)
    return destination_path


def load_zip_to_directory(gridfs_id: str, destination_dir: str) -> str:
    """Download a zipped directory (e.g. a saved transformer) and extract it."""
    data = load_bytes(gridfs_id)
    os.makedirs(destination_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(destination_dir)
    return destination_dir


def delete(gridfs_id: str) -> None:
    if not _GRIDFS_AVAILABLE:
        raise RuntimeError("GridFS not available: cannot delete")
    _fs().delete(ObjectId(gridfs_id))
    logger.info("Deleted GridFS file %s", gridfs_id)
