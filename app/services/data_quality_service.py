import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from scipy import stats
from ..schemas.data_quality import (
    QualityCheckType, 
    QualitySeverity, 
    QualityIssue, 
    DataQualityReport
)
from ..utils.logger import setup_logger

logger = setup_logger("data_quality_service", "logs/data_quality.log")

class DataQualityService:
    def __init__(self):
        # Configure thresholds
        self.missing_threshold = 0.1  # 10% missing values
        self.outlier_threshold = 3  # Standard deviations for outlier detection
        self.correlation_threshold = 0.95  # High correlation threshold
        self.unique_ratio_threshold = 0.95  # Unique values ratio for potential IDs
        
        # Common patterns for format validation
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*$',
            'date': r'^\d{4}-\d{2}-\d{2}$'
        }

    def generate_quality_report(self, df: pd.DataFrame, dataset_path: str) -> DataQualityReport:
        """Generate a comprehensive data quality report"""
        try:
            issues = []
            
            # Run all quality checks
            issues.extend(self._check_missing_values(df))
            issues.extend(self._check_type_consistency(df))
            issues.extend(self._check_outliers(df))
            issues.extend(self._check_duplicates(df))
            issues.extend(self._check_format_validation(df))
            issues.extend(self._check_value_ranges(df))
            issues.extend(self._check_uniqueness(df))
            issues.extend(self._check_correlations(df))
            issues.extend(self._check_categorical_validity(df))
            
            # Generate summary statistics
            summary_stats = self._generate_summary_stats(df)
            
            return DataQualityReport(
                dataset_path=dataset_path,
                total_rows=len(df),
                total_columns=len(df.columns),
                issues=issues,
                summary_stats=summary_stats
            )
            
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}", exc_info=True)
            raise

    def _check_missing_values(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for missing values in each column"""
        issues = []
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_ratio = missing_count / len(df)
            
            if missing_ratio > 0:
                severity = QualitySeverity.INFO
                if missing_ratio > self.missing_threshold:
                    severity = QualitySeverity.WARNING
                if missing_ratio > 0.5:
                    severity = QualitySeverity.ERROR
                
                issues.append(QualityIssue(
                    check_type=QualityCheckType.MISSING_VALUES,
                    severity=severity,
                    column=column,
                    message=f"Contains {missing_count} missing values ({missing_ratio:.2%})",
                    details={
                        "missing_count": int(missing_count),
                        "missing_ratio": float(missing_ratio)
                    }
                ))
        
        return issues

    def _check_type_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for type consistency within columns"""
        issues = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if column contains mixed types
                types = df[column].dropna().apply(type).value_counts()
                if len(types) > 1:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.TYPE_CONSISTENCY,
                        severity=QualitySeverity.WARNING,
                        column=column,
                        message=f"Mixed data types detected",
                        details={"type_counts": {str(k): int(v) for k, v in types.items()}}
                    ))
        
        return issues

    def _check_outliers(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect outliers in numeric columns"""
        issues = []
        
        for column in df.select_dtypes(include=[np.number]).columns:
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers = z_scores > self.outlier_threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.OUTLIERS,
                    severity=QualitySeverity.WARNING,
                    column=column,
                    message=f"Contains {outlier_count} outliers",
                    details={
                        "outlier_count": int(outlier_count),
                        "outlier_ratio": float(outlier_count / len(df)),
                        "outlier_threshold": self.outlier_threshold
                    }
                ))
        
        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for duplicate rows and values"""
        issues = []
        
        # Check duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(QualityIssue(
                check_type=QualityCheckType.DUPLICATES,
                severity=QualitySeverity.WARNING,
                column="ALL",
                message=f"Contains {duplicate_rows} duplicate rows",
                details={"duplicate_count": int(duplicate_rows)}
            ))
        
        # Check duplicate values in each column
        for column in df.columns:
            duplicate_values = df[column].duplicated().sum()
            if duplicate_values > 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.DUPLICATES,
                    severity=QualitySeverity.INFO,
                    column=column,
                    message=f"Contains {duplicate_values} duplicate values",
                    details={"duplicate_count": int(duplicate_values)}
                ))
        
        return issues

    def _check_format_validation(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Validate format of text columns"""
        issues = []
        
        for column in df.select_dtypes(include=['object']).columns:
            sample_values = df[column].dropna().head(100)
            
            # Check for common patterns
            for pattern_name, pattern in self.patterns.items():
                matches = sample_values.str.match(pattern, na=False)
                match_ratio = matches.mean()
                
                if 0 < match_ratio < 1:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.FORMAT_VALIDATION,
                        severity=QualitySeverity.INFO,
                        column=column,
                        message=f"Possible {pattern_name} format with {match_ratio:.2%} match ratio",
                        details={
                            "pattern_type": pattern_name,
                            "match_ratio": float(match_ratio)
                        }
                    ))
        
        return issues

    def _check_value_ranges(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check value ranges for numeric columns"""
        issues = []
        
        for column in df.select_dtypes(include=[np.number]).columns:
            stats = df[column].describe()
            
            # Check for suspicious ranges
            if stats['min'] < 0 and column.lower().find('count') >= 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.VALUE_RANGE,
                    severity=QualitySeverity.WARNING,
                    column=column,
                    message=f"Negative values found in count column",
                    details={
                        "min_value": float(stats['min']),
                        "negative_count": int((df[column] < 0).sum())
                    }
                ))
        
        return issues

    def _check_uniqueness(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check column uniqueness"""
        issues = []
        
        for column in df.columns:
            unique_ratio = df[column].nunique() / len(df)
            
            if unique_ratio > self.unique_ratio_threshold:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.UNIQUENESS,
                    severity=QualitySeverity.INFO,
                    column=column,
                    message=f"High uniqueness ratio ({unique_ratio:.2%}), possible ID column",
                    details={"unique_ratio": float(unique_ratio)}
                ))
        
        return issues

    def _check_correlations(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check for highly correlated numeric columns"""
        issues = []
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            # Get pairs of highly correlated columns
            high_corr = np.where(np.abs(corr_matrix) > self.correlation_threshold)
            high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                        for x, y in zip(*high_corr) if x != y and x < y]
            
            for col1, col2, corr in high_corr:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.CORRELATION,
                    severity=QualitySeverity.WARNING,
                    column=f"{col1}, {col2}",
                    message=f"High correlation ({corr:.2f}) between columns",
                    details={
                        "correlation": float(corr),
                        "columns": [col1, col2]
                    }
                ))
        
        return issues

    def _check_categorical_validity(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check categorical columns for validity"""
        issues = []
        
        for column in df.select_dtypes(include=['object', 'category']).columns:
            value_counts = df[column].value_counts()
            
            # Check for rare categories
            rare_categories = value_counts[value_counts < len(df) * 0.01]
            if len(rare_categories) > 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.CATEGORICAL_VALIDITY,
                    severity=QualitySeverity.INFO,
                    column=column,
                    message=f"Contains {len(rare_categories)} rare categories (<1%)",
                    details={
                        "rare_categories": rare_categories.to_dict(),
                        "rare_count": len(rare_categories)
                    }
                ))
        
        return issues

    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset"""
        return {
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            "total_missing": df.isna().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "column_stats": {
                column: {
                    "dtype": str(df[column].dtype),
                    "unique_count": df[column].nunique(),
                    "missing_count": df[column].isna().sum()
                } for column in df.columns
            }
        }