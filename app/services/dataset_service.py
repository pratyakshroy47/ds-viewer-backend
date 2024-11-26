import io
import math
from typing import Dict, List, Any, Optional
from fastapi import HTTPException
import pandas as pd
import swifter
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio
import numpy as np
import json
import aiofiles
from pathlib import Path
from google.cloud import storage
from ..schemas.filter import FilterConfig
from ..schemas.dataset import DatasetResponse
from .filter_service import FilterService
from .data_quality_service import DataQualityService
from ..utils.logger import setup_logger
from ..config import settings
from .cache_service import DatasetCache
import os
from ..utils.audio_utils import get_audio_path, download_from_gcs, create_audio_response

logger = setup_logger("dataset_service", "logs/dataset_service.log")

class DatasetService:
    def __init__(self):
        self.filter_service = FilterService()
        self.cache = DatasetCache(settings.cache.DATASET_CACHE_DIR)  # Updated to use new config structure
        self.data_quality_service = DataQualityService()
        self.gcs_client = None
        self._dataset_df = None
        self._current_dataset_path = None
        self._executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        self.n_cores = settings.CPU_CORES
        self.storage_client = storage.Client()
        
        try:
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                self.gcs_client = storage.Client()
                logger.info("Successfully initialized Google Cloud Storage client")
        except Exception as e:
            logger.error("Failed to initialize GCS client", exc_info=True)
            raise
        
    # async def _load_file(self, file_path: str) -> pd.DataFrame:
    #     """
    #     Load a dataset file into a pandas DataFrame.
        
    #     Args:
    #         file_path (str): Path to the dataset file
            
    #     Returns:
    #         pd.DataFrame: Loaded dataset
            
    #     Raises:
    #         ValueError: If file format is not supported or file doesn't exist
    #     """
    #     if not os.path.exists(file_path):
    #         raise ValueError(f"File not found: {file_path}")
        
        # file_extension = os.path.splitext(file_path)[1].lower()
        
        # try:
        #     if file_extension == '.csv':
        #         return pd.read_csv(file_path)
        #     elif file_extension in ['.xls', '.xlsx']:
        #         return pd.read_excel(file_path)
        #     elif file_extension == '.json':
        #         return pd.read_json(file_path)
        #     else:
        #         raise ValueError(f"Unsupported file format: {file_extension}")
        # except Exception as e:
        #     raise ValueError(f"Error reading file: {str(e)}")


    def _parse_gcs_path(self, gcs_path: str) -> tuple[str, str]:
        """Parse Google Cloud Storage path into bucket and blob path"""
        if not gcs_path.startswith("gs://"):
            raise ValueError("Invalid GCS path format")
        
        path = gcs_path.replace("gs://", "")
        bucket_name = path.split("/")[0]
        blob_path = "/".join(path.split("/")[1:])
        return bucket_name, blob_path

    async def _download_blob_to_cache(self, file_path: str, cache_key: str) -> Path:
        """Download blob to cache if not already cached"""
        try:
            # Check if valid cache exists
            if self.cache.is_cache_valid(cache_key):
                logger.info(f"Using cached file for {file_path}")
                return self.cache.get_cache_path(cache_key)

            # Download from GCS if not cached or cache invalid
            logger.info(f"Downloading file from GCS: {file_path}")
            bucket_name, blob_path = self._parse_gcs_path(file_path)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download to memory first
            data = await asyncio.to_thread(blob.download_as_bytes)
            
            # Save to cache
            self.cache.add_to_cache(cache_key, data)
            
            return self.cache.get_cache_path(cache_key)

        except Exception as e:
            logger.error(f"Error downloading blob to cache: {e}")
            raise Exception(f"Failed to download file: {str(e)}")

    async def _load_dataset_chunks(self, cache_file: Path) -> pd.DataFrame:
        """Load dataset in chunks using parallel processing"""
        try:
            async def read_chunks():
                chunks = []
                async with aiofiles.open(cache_file, mode='r') as f:
                    content = await f.read()
                    
                    # Split content into chunks for parallel processing
                    lines = content.splitlines()
                    chunk_size = max(1, len(lines) // self.n_cores)
                    content_chunks = [
                        lines[i:i + chunk_size] 
                        for i in range(0, len(lines), chunk_size)
                    ]
                    
                    def process_chunk(chunk_lines):
                        try:
                            return pd.DataFrame([
                                json.loads(line) 
                                for line in chunk_lines 
                                if line.strip()
                            ])
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            return pd.DataFrame()

                    # Process chunks in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                        chunk_dfs = list(executor.map(process_chunk, content_chunks))
                        
                    # Combine chunks
                    valid_chunks = [df for df in chunk_dfs if not df.empty]
                    if not valid_chunks:
                        raise ValueError("No valid data found in file")
                        
                    return pd.concat(valid_chunks, ignore_index=True)

            return await read_chunks()

        except Exception as e:
            logger.error(f"Error loading dataset chunks: {str(e)}", exc_info=True)
            raise

    async def _apply_filters_parallel(self, df: pd.DataFrame, filters: List[FilterConfig]) -> pd.DataFrame:
        """Apply filters in parallel using swifter"""
        try:
            if not filters:
                return df

            # Group filters by column for efficiency
            column_filters = {}
            for f in filters:
                if f.column not in column_filters:
                    column_filters[f.column] = []
                column_filters[f.column].append(f)

            filtered_df = df.copy()

            # Apply filters for each column
            for column, col_filters in column_filters.items():
                if column not in filtered_df.columns:
                    logger.warning(f"Column {column} not found in dataset")
                    continue

                col_type = self.filter_service._detect_column_type(filtered_df[column])
                logger.info(f"Applying filters for column {column} of type {col_type}")

                # Apply all filters for this column
                for filter_ in col_filters:
                    mask = filtered_df[column].swifter.apply(
                        lambda x: self.filter_service._apply_filter(x, filter_, col_type)
                        # lambda row: all(self.filter_service._apply_filter(row, f, data_type) for f in filters)
                    )
                    filtered_df = filtered_df[mask]
                    logger.info(f"After filter {column} {filter_.operator} {filter_.value}: {len(filtered_df)} rows")

            return filtered_df

        except Exception as e:
            logger.error(f"Error applying filters in parallel: {str(e)}")
            raise

    async def _get_columns_info_async(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Get column information asynchronously"""
        try:
            def process_column(column: str) -> Dict[str, str]:
                dtype = str(df[column].dtype)
                sample_mask = ~df[column].isna()
                sample_value = df[column].iloc[sample_mask.idxmax()] if sample_mask.any() else None
                
                if isinstance(sample_value, str):
                    if any(keyword in column.lower() for keyword in ['audio', 'wav', 'mp3', 'flac', 'fileuri']):
                        return {"name": column, "type": "audio"}
                    return {"name": column, "type": "string"}
                
                return {"name": column, "type": dtype}

            # Process columns in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                columns_info = list(executor.map(process_column, df.columns))
            
            return columns_info

        except Exception as e:
            logger.error(f"Error getting columns info: {str(e)}", exc_info=True)
            raise
        
    def _map_dtype_to_column_type(self, dtype: str) -> str:
        """Map pandas dtype to our column type enum"""
        dtype_str = str(dtype)
        
        if dtype_str.startswith('float'):
            return "float64"
        elif dtype_str.startswith('int'):
            return "int64"
        elif dtype_str == 'bool':
            return "boolean"
        elif dtype_str.startswith('datetime'):
            return "datetime"
        elif dtype_str == 'category':
            return "category"
        elif dtype_str == 'object':
            return "object"
        else:
            return "string"
        
    async def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a dataset file into a pandas DataFrame from local filesystem or Google Cloud Storage.
        
        Args:
            file_path (str): Path to the dataset file (local path or gs:// URL)
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        # Check if file is in Google Cloud Storage
        if file_path.startswith('gs://'):
            return await self._load_from_gcs(file_path)
        else:
            return await self._load_from_local(file_path)

    async def _load_from_gcs(self, gcs_path: str) -> pd.DataFrame:
        """Load file from Google Cloud Storage"""
        try:
            # Parse GCS path
            bucket_name = gcs_path.split('/')[2]
            blob_name = '/'.join(gcs_path.split('/')[3:])
            
            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Download content to memory
            content = blob.download_as_string()
            
            # Determine file type and read accordingly
            if gcs_path.endswith('.csv'):
                return pd.read_csv(io.BytesIO(content))
            elif gcs_path.endswith('.jsonl'):
                return pd.read_json(io.BytesIO(content), lines=True)
            elif gcs_path.endswith('.json'):
                return pd.read_json(io.BytesIO(content))
            elif gcs_path.endswith(('.xls', '.xlsx')):
                return pd.read_excel(io.BytesIO(content))
            else:
                raise ValueError(f"Unsupported file format: {gcs_path}")
                
        except Exception as e:
            raise ValueError(f"Error reading from GCS: {str(e)}")

    async def _load_from_local(self, file_path: str) -> pd.DataFrame:
        """Load file from local filesystem"""
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.jsonl'):
                return pd.read_json(file_path, lines=True)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
        
    


    async def load_dataset(
        self,
        file_path: str,
        page: int = 1,
        page_size: int = 10,
        filters: List[FilterConfig] = None
    ) -> DatasetResponse:
        """Load dataset with pagination and filtering"""
        try:
            # Use caching mechanism if it's a GCS file
            if file_path.startswith('gs://'):
                if self._current_dataset_path != file_path:
                    cache_key = self.cache.get_cache_key(file_path)
                    cache_file = await self._download_blob_to_cache(file_path, cache_key)
                    df = await self._load_dataset_chunks(cache_file)
                    self._dataset_df = df
                    self._current_dataset_path = file_path
                else:
                    df = self._dataset_df
            else:
                # For local files, use existing load mechanism
                if self._current_dataset_path != file_path:
                    df = await self._load_file(file_path)
                    self._dataset_df = df
                    self._current_dataset_path = file_path
                else:
                    df = self._dataset_df

            # Apply filters using existing parallel implementation
            filtered_df = await self._apply_filters_parallel(df, filters or [])
            
            # Calculate pagination
            total_rows = len(filtered_df)
            total_pages = math.ceil(total_rows / page_size)
            
            # Validate page number
            if page > total_pages:
                raise ValueError(f"Page {page} exceeds total pages {total_pages}")
            if page < 1:
                raise ValueError("Page number must be greater than 0")
                
            # Calculate slice indices
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            
            # Get page data
            page_data = filtered_df.iloc[start_idx:end_idx]
            
            # Use existing filter service for columns and filters
            columns_and_filters = self.filter_service.get_filterable_columns(filtered_df)
            
            return DatasetResponse(
                total_rows=total_rows,
                total_pages=total_pages,
                current_page=page,
                page_size=page_size,
                columns=columns_and_filters["columns"],
                filters=columns_and_filters["filters"],
                data=page_data.to_dict('records')
            )

        except ValueError as ve:
            logger.warning(f"Pagination error: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error loading dataset: {str(e)}")

    from .data_quality_service import DataQualityService, DataQualityReport
    

    async def check_data_quality(self, file_path: str) -> DataQualityReport:
        """Generate data quality report for a dataset"""
        try:
            # Load dataset if needed
            if self._current_dataset_path != file_path:
                cache_key = self.cache.get_cache_key(file_path)
                cache_file = await self._download_blob_to_cache(file_path, cache_key)
                self._dataset_df = await self._load_dataset_chunks(cache_file)
                self._current_dataset_path = file_path

            # Generate quality report
            return self.data_quality_service.generate_quality_report(
                self._dataset_df,
                file_path
            )

        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}", exc_info=True)
            raise
        
        
    async def get_audio(self, file_path: str) -> Dict[str, Any]:
        """Get audio file data"""
        try:
            cache_key = self.cache.get_cache_key(file_path)
            
            # Check cache first
            cache_file = self.cache.get_cached_file(cache_key)
            if cache_file is not None:
                logger.info(f"Using cached audio file: {file_path}")
                with open(cache_file, 'rb') as f:
                    audio_data = f.read()
                return create_audio_response(audio_data, file_path, 'cache')
            
            # Download from GCS if not cached
            logger.info(f"Downloading audio from GCS: {file_path}")
            bucket_name, blob_path = self._parse_gcs_path(file_path)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Download to memory
            audio_data = await asyncio.to_thread(blob.download_as_bytes)
            
            # Cache the audio data
            self.cache.add_to_cache(cache_key, audio_data)
            
            return create_audio_response(audio_data, file_path, 'gcs')
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error loading audio: {str(e)}"
            )