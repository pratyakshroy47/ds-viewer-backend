import multiprocessing
from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path
import os

class CacheSettings(BaseSettings):
    # Base Cache Settings
    BASE_DIR: str = "cache"
    DATASET_CACHE_DIR: str = "cache/datasets"
    AUDIO_CACHE_DIR: str = "cache/audio"
    
    # Cache TTL Settings
    CACHE_TTL_HOURS: int = 24
    AUDIO_TTL_HOURS: int = 1
    
    # Size Limits
    MAX_CACHE_SIZE_GB: float = 10.0
    DATASET_MAX_SIZE_GB: float = 8.0
    AUDIO_MAX_SIZE_GB: float = 2.0
    
    # Performance Settings
    COMPRESSION_ENABLED: bool = True
    COMPRESSION_LEVEL: int = 6
    CHUNK_SIZE: int = 50000
    USE_ASYNC_IO: bool = True
    
    # Monitoring
    CACHE_STATS_ENABLED: bool = True
    CLEANUP_INTERVAL_HOURS: int = 1
    MIN_FREE_SPACE_GB: float = 1.0

class MemorySettings(BaseSettings):
    MAX_MEMORY_USAGE_GB: float = 4.0
    MEMORY_WARNING_THRESHOLD_GB: float = 3.5
    FORCE_GARBAGE_COLLECTION: bool = True
    MONITOR_MEMORY_USAGE: bool = True
    MEMORY_CHECK_INTERVAL: int = 300  # seconds

class LoggingSettings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR: str = "logs"
    CACHE_LOG_FILE: str = "logs/cache.log"
    DATASET_LOG_FILE: str = "logs/dataset_service.log"
    ENABLE_FILE_LOGGING: bool = True
    ENABLE_CONSOLE_LOGGING: bool = True
    MAX_LOG_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_LOG_FILE_COUNT: int = 5

class Settings(BaseSettings):
    # Base settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Dataset Viewer API"
    DEBUG: bool = False
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # GCS settings
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GCS_RETRY_LIMIT: int = 3
    GCS_TIMEOUT: int = 30
    
    # Performance settings
    CPU_CORES: int = multiprocessing.cpu_count()
    MAX_WORKERS: int = CPU_CORES * 2
    SWIFTER_NPARTITIONS: int = CPU_CORES
    
    # Pagination settings
    MAX_PAGE_SIZE: int = 100
    MAX_ROWS_PER_PAGE: int = 50
    
    # Sub-configurations
    cache: CacheSettings = CacheSettings()
    memory: MemorySettings = MemorySettings()
    logging: LoggingSettings = LoggingSettings()
    
    class Config:
        env_file = ".env"
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        # Create cache directories
        Path(self.cache.DATASET_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.cache.AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        Path(self.logging.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.DEBUG
    
    def get_cache_dir(self, cache_type: str) -> str:
        """Get cache directory based on type"""
        if cache_type == "dataset":
            return self.cache.DATASET_CACHE_DIR
        elif cache_type == "audio":
            return self.cache.AUDIO_CACHE_DIR
        return self.cache.BASE_DIR

# Initialize settings
settings = Settings()