import base64
from pathlib import Path
from typing import Dict, Optional, Tuple
from google.cloud import storage
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

def get_audio_mime_type(file_path: str) -> str:
    """Get MIME type for audio file based on extension"""
    ext = Path(file_path).suffix.lower()
    return {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac'
    }.get(ext, 'audio/wav')

def parse_gcs_path(gcs_path: str) -> Tuple[str, str]:
    """Parse GCS path into bucket and blob names"""
    # Remove 'gs://' prefix
    path = gcs_path.replace('gs://', '')
    # Split into bucket and blob path
    bucket_name = path.split('/')[0]
    blob_path = '/'.join(path.split('/')[1:])
    return bucket_name, blob_path

def download_from_gcs(gcs_client: storage.Client, gcs_path: str) -> bytes:
    """Download file from GCS"""
    try:
        bucket_name, blob_path = parse_gcs_path(gcs_path)
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download to memory
        audio_data = blob.download_as_bytes()
        logger.info(f"Successfully downloaded audio from GCS: {gcs_path}")
        return audio_data
    
    except Exception as e:
        logger.error(f"Error downloading from GCS: {gcs_path}", exc_info=True)
        raise Exception(f"Failed to download audio from GCS: {str(e)}")

def create_audio_response(
    audio_data: bytes,
    file_path: str,
    source: str = 'local'
) -> Dict[str, str]:
    """Create standardized audio response"""
    return {
        'type': 'audio',
        'format': 'base64',
        'mime_type': get_audio_mime_type(file_path),
        'data': base64.b64encode(audio_data).decode('utf-8'),
        'source': source,
        'path': file_path
    }
    
def get_audio_path(gcs_path: str) -> tuple[str, str]:
    """
    Parse GCS path into bucket and blob path
    Args:
        gcs_path: GCS path in format 'gs://bucket-name/path/to/file'
    Returns:
        Tuple of (bucket_name, blob_path)
    """
    path = gcs_path.replace('gs://', '')
    bucket_name, *blob_parts = path.split('/')
    blob_path = '/'.join(blob_parts)
    return bucket_name, blob_path

