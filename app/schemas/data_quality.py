from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

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