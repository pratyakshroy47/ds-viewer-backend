from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum
import re
from datetime import datetime
from ..schemas.filter import FilterConfig, FilterOperator, FilterType
from ..utils.logger import setup_logger
from ..config import settings
from ..schemas.dataset import ColumnType
from pathlib import Path

logger = setup_logger("filter_service", "logs/filter.log")

class DataType(str, Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"

class FilterService:
    def __init__(self):
        self.type_handlers = {
            DataType.NUMERIC: self._handle_numeric,
            DataType.TEXT: self._handle_text,
            DataType.DATETIME: self._handle_datetime,
            DataType.BOOLEAN: self._handle_boolean,
            DataType.CATEGORICAL: self._handle_categorical
        }
        self.type_mapping = {
            DataType.TEXT: ColumnType.STRING,
            DataType.NUMERIC: ColumnType.FLOAT,
            DataType.DATETIME: ColumnType.DATETIME,
            DataType.BOOLEAN: ColumnType.BOOLEAN,
            DataType.CATEGORICAL: ColumnType.CATEGORY,
            DataType.UNKNOWN: ColumnType.OBJECT
        }
        
        self.operator_descriptions = {
            "eq": "Equals",
            "ne": "Not equals",
            "gt": "Greater than",
            "lt": "Less than",
            "ge": "Greater than or equal to",
            "le": "Less than or equal to",
            "contains": "Contains text",
            "not_contains": "Does not contain text",
            "starts_with": "Starts with",
            "ends_with": "Ends with",
            "in": "In list of values",
            "not_in": "Not in list of values",
            "is_null": "Is empty",
            "is_not_null": "Is not empty"
        }

        self.type_operators = {
            "string": ["eq", "ne", "contains", "not_contains", "starts_with", "ends_with", "is_null", "is_not_null"],
            "float64": ["eq", "ne", "gt", "lt", "ge", "le", "is_null", "is_not_null"],
            "int64": ["eq", "ne", "gt", "lt", "ge", "le", "is_null", "is_not_null"],
            "boolean": ["eq", "ne", "is_null", "is_not_null"],
            "datetime": ["eq", "ne", "gt", "lt", "ge", "le", "is_null", "is_not_null"],
            "category": ["eq", "ne", "in", "not_in", "is_null", "is_not_null"],
            "object": ["eq", "ne", "is_null", "is_not_null"]
        }

    def _detect_column_type(self, series: pd.Series) -> DataType:
        """Detect the data type of a column"""
        try:
            # Check for all null
            if series.isna().all():
                return DataType.UNKNOWN

            # Get non-null values
            non_null = series.dropna()
            if len(non_null) == 0:
                return DataType.UNKNOWN

            # Check dtype
            if pd.api.types.is_numeric_dtype(series):
                return DataType.NUMERIC
            elif pd.api.types.is_bool_dtype(series):
                return DataType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(series):
                return DataType.DATETIME
            elif pd.api.types.is_categorical_dtype(series):
                return DataType.CATEGORICAL
            
            # Check if string/object type
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                # Check for datetime strings
                try:
                    pd.to_datetime(series.iloc[0])
                    return DataType.DATETIME
                except:
                    pass
                
                # Check for categorical (if few unique values)
                unique_ratio = len(non_null.unique()) / len(non_null)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    return DataType.CATEGORICAL
                
                return DataType.TEXT

            return DataType.UNKNOWN

        except Exception as e:
            logger.error(f"Error detecting column type: {str(e)}")
            return DataType.UNKNOWN

    def _handle_numeric(self, value: Any, filter_value: Any, operator: str) -> bool:
        """Handle numeric comparisons"""
        try:
            value = float(value) if not isinstance(value, (int, float)) else value
            filter_value = float(filter_value) if not isinstance(filter_value, (int, float)) else filter_value
            
            if operator == "eq": return value == filter_value
            elif operator == "ne": return value != filter_value
            elif operator == "gt": return value > filter_value
            elif operator == "lt": return value < filter_value
            elif operator == "ge": return value >= filter_value
            elif operator == "le": return value <= filter_value
            return False
        except Exception as e:
            logger.error(f"Error in numeric comparison: {str(e)}")
            return False

    def _handle_text(self, value: Any, filter_value: Any, operator: str) -> bool:
        """Handle text comparisons"""
        try:
            value = str(value).lower() if value is not None else ""
            filter_value = str(filter_value).lower() if filter_value is not None else ""
            
            if operator == "eq": return value == filter_value
            elif operator == "ne": return value != filter_value
            elif operator == "contains": return filter_value in value
            elif operator == "not_contains": return filter_value not in value
            elif operator == "starts_with": return value.startswith(filter_value)
            elif operator == "ends_with": return value.endswith(filter_value)
            return False
        except Exception as e:
            logger.error(f"Error in text comparison: {str(e)}")
            return False

    def _handle_datetime(self, value: Any, filter_value: Any, operator: str) -> bool:
        """Handle datetime comparisons"""
        try:
            value = pd.to_datetime(value)
            filter_value = pd.to_datetime(filter_value)
            
            if operator == "eq": return value == filter_value
            elif operator == "ne": return value != filter_value
            elif operator == "gt": return value > filter_value
            elif operator == "lt": return value < filter_value
            elif operator == "ge": return value >= filter_value
            elif operator == "le": return value <= filter_value
            return False
        except Exception as e:
            logger.error(f"Error in datetime comparison: {str(e)}")
            return False

    def _handle_boolean(self, value: Any, filter_value: Any, operator: str) -> bool:
        """Handle boolean comparisons"""
        try:
            if isinstance(value, str):
                value = value.lower() in ['true', '1', 'yes', 'y']
            if isinstance(filter_value, str):
                filter_value = filter_value.lower() in ['true', '1', 'yes', 'y']
            
            if operator == "eq": return value == filter_value
            elif operator == "ne": return value != filter_value
            return False
        except Exception as e:
            logger.error(f"Error in boolean comparison: {str(e)}")
            return False

    def _handle_categorical(self, value: Any, filter_value: Any, operator: str) -> bool:
        """Handle categorical comparisons"""
        try:
            value = str(value).lower() if value is not None else ""
            filter_value = str(filter_value).lower() if filter_value is not None else ""
            
            if operator == "eq": return value == filter_value
            elif operator == "ne": return value != filter_value
            elif operator == "in": return value in filter_value
            elif operator == "not_in": return value not in filter_value
            return False
        except Exception as e:
            logger.error(f"Error in categorical comparison: {str(e)}")
            return False

    def _apply_filter(self, value: Any, filter_config: FilterConfig, data_type: DataType) -> bool:
        """Apply a single filter to a value"""
        try:
            # Handle null values
            if pd.isna(value):
                return filter_config.operator == "is_null"
            if filter_config.operator == "is_null":
                return pd.isna(value)
            if filter_config.operator == "is_not_null":
                return not pd.isna(value)

            # Get the appropriate handler for the data type
            handler = self.type_handlers.get(data_type, self._handle_text)
            return handler(value, filter_config.value, filter_config.operator)

        except Exception as e:
            logger.error(f"Error applying filter: {str(e)}")
            return False

    def apply_filters(self, df: pd.DataFrame, filters: List[FilterConfig]) -> pd.DataFrame:
        """Apply multiple filters to the dataframe"""
        try:
            if not filters:
                return df

            filtered_df = df.copy()
            
            # Group filters by column
            column_filters = {}
            for f in filters:
                if f.column not in column_filters:
                    column_filters[f.column] = []
                column_filters[f.column].append(f)

            # Apply filters for each column
            for column, col_filters in column_filters.items():
                if column not in filtered_df.columns:
                    logger.warning(f"Column {column} not found in dataset")
                    continue

                col_type = self._detect_column_type(filtered_df[column])
                logger.info(f"Applying filters for column {column} of type {col_type}")
                
                # Apply all filters for this column
                for filter_ in col_filters:
                    mask = filtered_df[column].apply(
                        lambda x: self._apply_filter(x, filter_, col_type)
                    )
                    filtered_df = filtered_df[mask]
                    logger.info(f"After filter {column} {filter_.operator} {filter_.value}: {len(filtered_df)} rows")

            return filtered_df

        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            raise
        
    def apply_filter_to_series(self, series: pd.Series, filter_config: FilterConfig, data_type: DataType) -> pd.Series:
        """Apply filter to a pandas Series and return boolean mask"""
        try:
            # Handle null values
            if filter_config.operator == "is_null":
                return series.isna()
            if filter_config.operator == "is_not_null":
                return ~series.isna()
                
            # Get the appropriate handler for the data type
            handler = self.type_handlers.get(data_type, self._handle_text)
            
            # Apply handler vectorized
            return series.apply(lambda x: handler(x, filter_config.value, filter_config.operator))
            
        except Exception as e:
            logger.error(f"Error applying filter to series: {str(e)}")
            return pd.Series([False] * len(series))
    
    async def _apply_filters_parallel(self, df: pd.DataFrame, filters: List[FilterConfig]) -> pd.DataFrame:
        """Apply filters in parallel using swifter"""
        try:
            if not filters:
                return df
                
            filtered_df = df.swifter.apply(
                lambda row: all(self.filter_service.apply_filter(row, f) for f in filters),
                axis=1
            )
            return df[filtered_df]
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            raise
    
    def _get_simple_type(self, data_type: DataType) -> str:
        """Convert internal DataType to simple string type"""
        type_mapping = {
            DataType.TEXT: "string",
            DataType.NUMERIC: "float64",
            DataType.DATETIME: "datetime",
            DataType.BOOLEAN: "boolean",
            DataType.CATEGORICAL: "category",
            DataType.UNKNOWN: "object"
        }
        return type_mapping.get(data_type, "string")



    def _get_allowed_operators(self, data_type: DataType) -> List[str]:
        """Get allowed operators for each data type"""
        common_operators = ["eq", "ne", "is_null", "is_not_null"]
        
        type_operators = {
            DataType.NUMERIC: common_operators + ["gt", "lt", "ge", "le"],
            DataType.TEXT: common_operators + ["contains", "not_contains", "starts_with", "ends_with"],
            DataType.DATETIME: common_operators + ["gt", "lt", "ge", "le"],
            DataType.BOOLEAN: common_operators,
            DataType.CATEGORICAL: common_operators + ["in", "not_in"],
            DataType.UNKNOWN: common_operators
        }
        
        return type_operators.get(data_type, common_operators)


    def get_filterable_columns(self, df: pd.DataFrame) -> Dict[str, List]:
        """Get columns and their filter metadata"""
        try:
            columns = []
            filters = []
            
            # Define audio file extensions and GCS prefix
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
            gcs_prefix = "gs://"
            
            for column in df.columns:
                # Determine column type
                column_type = str(df[column].dtype)
                
                # Check if this column contains GCS audio file paths
                is_audio_column = False
                if column_type == "object":
                    # Check a sample value from the column
                    sample_value = df[column].iloc[0] if not df[column].empty else ""
                    if isinstance(sample_value, str):
                        is_gcs_path = sample_value.startswith(gcs_prefix)
                        file_ext = Path(sample_value).suffix.lower()
                        is_audio_column = is_gcs_path and file_ext in audio_extensions
                
                # Assign filter type
                if is_audio_column:
                    filter_type = "audio"
                elif column_type == "object":
                    filter_type = "string"
                elif column_type in ["float64", "int64"]:
                    filter_type = column_type
                elif column_type == "category":
                    filter_type = "category"
                else:
                    filter_type = "string"
            
            
                
                # Add column info
                columns.append({
                    "name": column,
                    "type": filter_type
                })
                
                # Get unique values based on type, limit to 50 values
                unique_values = None
                if filter_type == "category":
                    unique_vals = df[column].unique()
                    unique_values = unique_vals[:50].tolist() if len(unique_vals) > 50 else unique_vals.tolist()
                elif filter_type in ["float64", "int64"]:
                    unique_values = {
                        "min": float(df[column].min()),
                        "max": float(df[column].max()),
                        "mean": float(df[column].mean())
                    }
                elif filter_type in ["string", "audio"]:
                    unique_vals = df[column].unique()
                    if len(unique_vals) > 50:
                        unique_values = {
                            "total_unique": len(unique_vals),
                            "sample_values": unique_vals[:50].tolist()
                        }
                    else:
                        unique_values = unique_vals.tolist()
                
                # Get operators for this type
                operators = self._get_operators_for_type(filter_type)
                
                # Add filter metadata
                filters.append({
                    "column": column,
                    "type": filter_type,
                    "unique_values": unique_values,
                    "operators": operators
                })
            
            return {
                "columns": columns,
                "filters": filters
            }
            
        except Exception as e:
            logger.error(f"Error getting filterable columns: {str(e)}")
            raise

    def _get_operators_for_type(self, filter_type: str) -> List[Dict]:
        """Get appropriate operators for each filter type"""
        base_operators = [
            {"name": "is_null", "description": "Is empty"},
            {"name": "is_not_null", "description": "Is not empty"}
        ]
        
        type_operators = {
            "string": [
                {"name": "eq", "description": "Equals"},
                {"name": "ne", "description": "Not equals"},
                {"name": "contains", "description": "Contains text"},
                {"name": "not_contains", "description": "Does not contain text"},
                {"name": "starts_with", "description": "Starts with"},
                {"name": "ends_with", "description": "Ends with"}
            ],
            "audio": [
                {"name": "eq", "description": "Equals"},
                {"name": "ne", "description": "Not equals"},
                {"name": "contains", "description": "Contains text"},
                {"name": "not_contains", "description": "Does not contain text"}
            ],
            "category": [
                {"name": "eq", "description": "Equals"},
                {"name": "ne", "description": "Not equals"},
                {"name": "in", "description": "In list"},
                {"name": "not_in", "description": "Not in list"}
            ],
            "float64": [
                {"name": "eq", "description": "Equals"},
                {"name": "ne", "description": "Not equals"},
                {"name": "gt", "description": "Greater than"},
                {"name": "lt", "description": "Less than"},
                {"name": "ge", "description": "Greater than or equal to"},
                {"name": "le", "description": "Less than or equal to"},
                {"name": "between", "description": "Between two values"}
            ]
        }
        
        return type_operators.get(filter_type, []) + base_operators
        
    def get_column_unique_values(self, df: pd.DataFrame, column: str, column_type: str) -> Union[List, Dict, None]:
        """Get unique values for a column based on its type"""
        try:
            if column_type == "category":
                return df[column].unique().tolist()
            
            elif column_type == "float64" or column_type == "int64":
                return {
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "mean": float(df[column].mean())
                }
                
            elif column_type == "string":
                # For string columns, get unique values but limit to prevent overwhelming response
                unique_values = df[column].unique()
                if len(unique_values) > 100:  # Limit to prevent too many values
                    return {
                        "total_unique": len(unique_values),
                        "sample_values": unique_values[:100].tolist()
                    }
                return unique_values.tolist()
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting unique values for column {column}: {str(e)}")
            return None
        
    def get_filters_metadata(self, df: pd.DataFrame) -> List[Dict]:
        """Get metadata about available filters including unique values"""
        filters = []
        
        for column in df.columns:
            column_type = str(df[column].dtype)
            
            # Map pandas types to our filter types
            if column_type == "object":
                filter_type = "string"
            elif column_type in ["float64", "int64"]:
                filter_type = column_type
            elif column_type == "category":
                filter_type = "category"
            else:
                filter_type = "string"
                
            # Get appropriate operators based on type
            operators = self._get_operators_for_type(filter_type)
            
            # Get unique values
            unique_values = self.get_column_unique_values(df, column, filter_type)
            
            filters.append({
                "column": column,
                "type": filter_type,
                "unique_values": unique_values,
                "operators": operators
            })
            
        return filters
    
    