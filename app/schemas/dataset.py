from typing import Union
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from .filter import FilterableColumn
from datetime import datetime
from .filter import FilterOperator


class ColumnType(str, Enum):
    STRING = "string"
    FLOAT = "float64"
    INT = "int64"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORY = "category"
    AUDIO = "audio"
    OBJECT = "object"  

class ColumnInfo(BaseModel):
    name: str
    type: str

class LoadDatasetRequest(BaseModel):
    file_path: str
    page: Optional[int] = Field(default=1, ge=1)
    page_size: Optional[int] = Field(default=10, ge=1, le=100)

class FilterOperator(BaseModel):
    name: str
    description: str

class FilterMetadata(BaseModel):
    column: str
    type: str
    unique_values: Optional[Union[List[Any], Dict[str, Any]]]
    operators: List[FilterOperator]
    
class DatasetLoadRequest(BaseModel):
    file_path: str
    page: int = 1
    page_size: int = 10
    filters: Optional[List[Dict[str, Any]]] = None


class DatasetResponse(BaseModel):
    total_rows: int
    total_pages: int
    current_page: int
    page_size: int
    columns: List[Dict[str, str]]
    filters: List[FilterMetadata]
    data: List[Dict[str, Any]]

    class Config:
        from_attributes = True

class AudioResponse(BaseModel):
    """Response model for audio data"""
    type: str = "audio"
    format: str = "base64"
    mime_type: str
    data: str
    source: str
    path: str

class ErrorResponse(BaseModel):
    """Response model for errors"""
    detail: str
    
class DatasetError(Exception):
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class QualityCheckType(str, Enum):
    MISSING_VALUES = "missing_values"
    TYPE_CONSISTENCY = "type_consistency"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    FORMAT_VALIDATION = "format_validation"
    VALUE_RANGE = "value_range"
    UNIQUENESS = "uniqueness"
    CORRELATION = "correlation"
    CATEGORICAL_VALIDITY = "categorical_validity"

class QualitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class QualityIssue(BaseModel):
    check_type: QualityCheckType
    severity: QualitySeverity
    column: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

class DataQualityReport(BaseModel):
    dataset_path: str
    total_rows: int
    total_columns: int
    issues: List[QualityIssue]
    summary_stats: Dict[str, Any]
    generated_at: datetime = datetime.now()
    
class Operator(str, Enum):
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    
