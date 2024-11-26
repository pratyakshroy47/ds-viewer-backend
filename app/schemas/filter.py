from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class FilterType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    NONE = "none"

class FilterOperator(str, Enum):
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
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

class FilterableColumn(BaseModel):
    name: str
    filter_type: FilterType
    unique_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_operators: List[FilterOperator]

class FilterConfig(BaseModel):
    column: str
    operator: FilterOperator
    value: Optional[Any] = None
    values: Optional[List[Any]] = None
    case_sensitive: bool = False

    class Config:
        use_enum_values = True

class FilterRequest(BaseModel):
    filters: List[FilterConfig]