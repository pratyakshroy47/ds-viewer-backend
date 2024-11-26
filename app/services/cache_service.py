from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import logging
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class DatasetCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.lock_file = self.cache_dir / "cache.lock"
        self.metadata = self._load_metadata()
        self.cache_ttl = timedelta(hours=settings.cache.CACHE_TTL_HOURS)
        
        # Initialize cache if needed
        self._init_cache()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {e}")
                return {}
        return {}

    def _init_cache(self):
        """Initialize cache metadata if empty"""
        if not self.metadata:
            self.metadata = {
                "files": {},
                "last_cleanup": datetime.now().isoformat(),
                "cache_hits": 0,
                "cache_misses": 0
            }
            self._save_metadata()
        self._cleanup_old_cache()
        self._enforce_cache_size_limit()

    def _save_metadata(self):
        """Save cache metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def _cleanup_old_cache(self):
        """Remove expired cache files"""
        now = datetime.now()
        files_to_remove = []

        # Check files in metadata
        for cache_key, meta in self.metadata.get("files", {}).items():
            try:
                cache_time = datetime.fromisoformat(meta['timestamp'])
                if now - cache_time > self.cache_ttl:
                    files_to_remove.append(cache_key)
            except Exception as e:
                logger.error(f"Error processing cache entry {cache_key}: {e}")
                files_to_remove.append(cache_key)

        # Remove expired files
        for cache_key in files_to_remove:
            self._remove_cache_file(cache_key)

        # Update last cleanup time
        self.metadata["last_cleanup"] = now.isoformat()
        self._save_metadata()

    def _remove_cache_file(self, cache_key: str):
        """Remove a cache file and its metadata"""
        try:
            cache_file = self.get_cache_path(cache_key)
            if cache_file.exists():
                cache_file.unlink()
            if cache_key in self.metadata.get("files", {}):
                del self.metadata["files"][cache_key]
        except Exception as e:
            logger.error(f"Error removing cache file {cache_key}: {e}")

    def _enforce_cache_size_limit(self):
        """Enforce cache size limit by removing oldest files"""
        total_size = 0
        cache_files = []
        
        # Calculate total cache size and collect file info
        for cache_key, meta in self.metadata.get("files", {}).items():
            cache_file = self.get_cache_path(cache_key)
            if cache_file.exists():
                size = os.path.getsize(cache_file)
                last_accessed = datetime.fromisoformat(meta['last_accessed'])
                cache_files.append((cache_key, cache_file, size, last_accessed))
                total_size += size
        
        # Remove oldest files if cache size exceeds limit
        max_size_bytes = settings.cache.MAX_CACHE_SIZE_GB * 1024 * 1024 * 1024
        if total_size > max_size_bytes:
            cache_files.sort(key=lambda x: x[3])
            
            for cache_key, _, size, _ in cache_files:
                if total_size <= max_size_bytes:
                    break
                self._remove_cache_file(cache_key)
                total_size -= size

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the path for a cache file"""
        return self.cache_dir / f"{cache_key}.cache"

    def get_cache_key(self, file_path: str) -> str:
        """Generate a cache key from a file path"""
        from hashlib import md5
        return md5(file_path.encode()).hexdigest()

    def add_to_cache(self, cache_key: str, data: Any):
        """Add data to cache"""
        try:
            cache_path = self.get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                f.write(data)
            
            self.metadata.setdefault("files", {})[cache_key] = {
                "timestamp": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "size": os.path.getsize(cache_path)
            }
            self._save_metadata()
        except Exception as e:
            logger.error(f"Error adding to cache: {e}")
            raise

    def get_from_cache(self, cache_key: str) -> Optional[bytes]:
        """Get data from cache"""
        try:
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists() and cache_key in self.metadata.get("files", {}):
                # Update last accessed time
                self.metadata["files"][cache_key]["last_accessed"] = datetime.now().isoformat()
                self._save_metadata()
                
                with open(cache_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
        return None

    def is_cached(self, cache_key: str) -> bool:
        """Check if a key exists in cache"""
        cache_path = self.get_cache_path(cache_key)
        return cache_path.exists() and cache_key in self.metadata.get("files", {})
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if a cached file is valid and not expired
        Returns:
            bool: True if cache is valid and not expired, False otherwise
        """
        try:
            # First check if file exists in cache
            if not self.is_cached(cache_key):
                return False

            # Get metadata for the cache key
            cache_meta = self.metadata.get("files", {}).get(cache_key)
            if not cache_meta:
                return False

            # Check file existence
            cache_file = self.get_cache_path(cache_key)
            if not cache_file.exists():
                return False

            # Check cache expiration
            cache_time = datetime.fromisoformat(cache_meta['timestamp'])
            if datetime.now() - cache_time > self.cache_ttl:
                # Cache is expired, remove it
                self._remove_cache_file(cache_key)
                return False

            # Cache is valid
            return True

        except Exception as e:
            logger.error(f"Error checking cache validity for {cache_key}: {e}")
            return False
        
    def get_cached_file(self, cache_key: str) -> Optional[Path]:
        """
        Get the path to a cached file if it exists and is valid
        
        Args:
            cache_key (str): The cache key to look up
            
        Returns:
            Optional[Path]: Path to the cached file if it exists and is valid, None otherwise
        """
        try:
            if not self.is_cache_valid(cache_key):
                return None
                
            cache_path = self.get_cache_path(cache_key)
            if not cache_path.exists():
                return None
                
            # Update last accessed time
            self.metadata["files"][cache_key]["last_accessed"] = datetime.now().isoformat()
            self._save_metadata()
            
            return cache_path
            
        except Exception as e:
            logger.error(f"Error getting cached file for {cache_key}: {e}")
            return None