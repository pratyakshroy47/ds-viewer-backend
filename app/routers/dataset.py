from ast import List
from fastapi import APIRouter, HTTPException, Query, Depends
import io
from typing import Dict, Any, List, Optional, Union
from ..schemas.dataset import (
    DatasetResponse, 
    LoadDatasetRequest, 
    AudioResponse,
    ErrorResponse
)
from ..services.dataset_service import DatasetService
from ..schemas.filter import FilterConfig
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional
from pydantic import BaseModel, Field
from ..schemas.dataset import (
    DatasetResponse, 
    LoadDatasetRequest, 
    AudioResponse,
    ErrorResponse,
    DataQualityReport  # Add this import
)


# @router.post("/dataset/load", response_model=DatasetResponse)
# async def load_dataset(request: LoadDatasetRequest) -> DatasetResponse:
#     try:
#         data = await dataset_service.load_dataset(request.file_path)
        
#         # Create a proper DatasetResponse object
#         response = DatasetResponse(
#             total_rows=data["total_rows"],
#             columns=data["columns"],
#             sample_data=data["sample_data"],
#             message=data["message"]
#         )
#         return response
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Error loading dataset: {str(e)}"
#         )



router = APIRouter()
dataset_service = DatasetService()

class DatasetLoadRequest(BaseModel):
    file_path: str
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    filters: Optional[List[FilterConfig]] = Field(default=None)

    class Config:
        from_attributes = True

@router.post("/dataset/load", response_model=DatasetResponse)
async def load_dataset(
    request: DatasetLoadRequest = Body(...)
) -> DatasetResponse:
    try:
        data = await dataset_service.load_dataset(
            file_path=request.file_path,
            page=request.page,
            page_size=request.page_size,
            filters=request.filters
        )
        return data
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error loading dataset: {str(e)}"
        )

@router.post("/dataset/quality-check", response_model=DataQualityReport)
async def check_data_quality(
    file_path: str,
    dataset_service: DatasetService = Depends()
):
    """Generate data quality report for a dataset"""
    return await dataset_service.check_data_quality(file_path)
        
@router.get("/dataset/audio", response_model=AudioResponse)
async def get_audio(
    path: str = Query(..., description="Path to audio file (local or GCS)")
) -> AudioResponse:
    try:
        audio_data = await dataset_service.get_audio(path)
        return AudioResponse(**audio_data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error loading audio: {str(e)}"
        )

# @router.get("/dataset/audio")
# async def get_audio(
#     path: str = Query(..., description="Path to audio file (local or GCS)")
# ) -> AudioResponse:
#     try:
#         audio_data = await dataset_service.get_audio(path)
#         return AudioResponse(
#             io.BytesIO(audio_data),
#             media_type="audio/wav/flac"
#         )
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))