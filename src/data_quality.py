"""
Comprehensive Data Quality Assurance for ML-TA System

This module provides automated data quality checks, anomaly detection,
data profiling, and quality reporting for cryptocurrency market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import structlog

from .config import get_config
from .exceptions import ValidationError, DataQualityError
from .utils import ensure_directory
from .logging_config import get_logger

logger = get_logger("data_quality").get_logger()


class QualityCheckType(Enum):
    """Types of data quality checks."""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"


class QualitySeverity(Enum):
    """Severity levels for quality issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """Represents a data quality issue."""
    check_type: QualityCheckType
    severity: QualitySeverity
    description: str
    affected_records: int
    affected_columns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Data quality metrics for a dataset."""
    total_records: int
    completeness_score: float
    consistency_score: float
    validity_score: float
    timeliness_score: float
    overall_score: float
    issues: List[QualityIssue] = field(default_factory=list)


class DataQualityFramework:
    """Main framework for comprehensive data quality assessment."""
    
    def __init__(self):
        """Initialize data quality framework."""
        self.logger = logger.bind(component="data_quality_framework")
    
    def check_completeness(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check data completeness."""
        issues = []
        
        # Check missing values
        null_counts = df.isnull().sum()
        for column, null_count in null_counts.items():
            if null_count > 0:
                percentage = (null_count / len(df)) * 100
                severity = QualitySeverity.CRITICAL if percentage > 10 else QualitySeverity.MEDIUM
                
                issues.append(QualityIssue(
                    check_type=QualityCheckType.COMPLETENESS,
                    severity=severity,
                    description=f"Missing values in {column}: {null_count} ({percentage:.1f}%)",
                    affected_records=null_count,
                    affected_columns=[column]
                ))
        
        return issues
    
    def check_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check data consistency."""
        issues = []
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['low'] > df['open']) |
                (df['low'] > df['high']) |
                (df['low'] > df['close']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close'])
            )
            
            if invalid_ohlc.any():
                count = invalid_ohlc.sum()
                issues.append(QualityIssue(
                    check_type=QualityCheckType.CONSISTENCY,
                    severity=QualitySeverity.HIGH,
                    description=f"Invalid OHLC relationships: {count} records",
                    affected_records=count,
                    affected_columns=['open', 'high', 'low', 'close']
                ))
        
        return issues
    
    def check_validity(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check data validity."""
        issues = []
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.VALIDITY,
                        severity=QualitySeverity.CRITICAL,
                        description=f"Non-positive values in {col}: {negative_count} records",
                        affected_records=negative_count,
                        affected_columns=[col]
                    ))
        
        return issues
    
    def check_timeliness(self, df: pd.DataFrame, max_age_hours: int = 24) -> List[QualityIssue]:
        """Check data timeliness."""
        issues = []
        
        if 'timestamp' in df.columns and not df.empty:
            latest = df['timestamp'].max()
            age_hours = (datetime.now() - latest).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.TIMELINESS,
                    severity=QualitySeverity.MEDIUM,
                    description=f"Data is stale: {age_hours:.1f} hours old",
                    affected_records=len(df),
                    affected_columns=['timestamp']
                ))
        
        return issues
    
    def assess_quality(self, df: pd.DataFrame, symbol: str = "unknown") -> QualityMetrics:
        """Perform comprehensive quality assessment."""
        self.logger.info(f"Assessing quality for {symbol}", records=len(df))
        
        all_issues = []
        all_issues.extend(self.check_completeness(df))
        all_issues.extend(self.check_consistency(df))
        all_issues.extend(self.check_validity(df))
        all_issues.extend(self.check_timeliness(df))
        
        # Calculate scores
        scores = self._calculate_scores(df, all_issues)
        
        metrics = QualityMetrics(
            total_records=len(df),
            completeness_score=scores['completeness'],
            consistency_score=scores['consistency'],
            validity_score=scores['validity'],
            timeliness_score=scores['timeliness'],
            overall_score=scores['overall'],
            issues=all_issues
        )
        
        self.logger.info(f"Quality assessment complete", 
                        overall_score=metrics.overall_score,
                        issues=len(all_issues))
        
        return metrics
    
    def _calculate_scores(self, df: pd.DataFrame, issues: List[QualityIssue]) -> Dict[str, float]:
        """Calculate quality scores."""
        scores = {
            'completeness': 100.0,
            'consistency': 100.0,
            'validity': 100.0,
            'timeliness': 100.0
        }
        
        # Deduct points for issues
        for issue in issues:
            deduction = min(20, (issue.affected_records / len(df)) * 100)
            
            if issue.severity == QualitySeverity.CRITICAL:
                deduction *= 2
            elif issue.severity == QualitySeverity.HIGH:
                deduction *= 1.5
            
            if issue.check_type == QualityCheckType.COMPLETENESS:
                scores['completeness'] = max(0, scores['completeness'] - deduction)
            elif issue.check_type == QualityCheckType.CONSISTENCY:
                scores['consistency'] = max(0, scores['consistency'] - deduction)
            elif issue.check_type == QualityCheckType.VALIDITY:
                scores['validity'] = max(0, scores['validity'] - deduction)
            elif issue.check_type == QualityCheckType.TIMELINESS:
                scores['timeliness'] = max(0, scores['timeliness'] - deduction)
        
        scores['overall'] = np.mean(list(scores.values()))
        return scores


class DataCleaner:
    """Automatically fixes common data quality issues."""
    
    def __init__(self):
        """Initialize data cleaner."""
        self.logger = logger.bind(component="data_cleaner")
    
    def clean_data(self, df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """
        Clean data by fixing common issues.
        
        Args:
            df: DataFrame to clean
            aggressive: Whether to apply aggressive cleaning
        
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        
        # Remove duplicate timestamps
        if 'timestamp' in df_cleaned.columns:
            before_count = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates(subset=['timestamp'])
            after_count = len(df_cleaned)
            
            if before_count != after_count:
                self.logger.info(f"Removed {before_count - after_count} duplicate timestamps")
        
        # Fix OHLC relationships
        if all(col in df_cleaned.columns for col in ['open', 'high', 'low', 'close']):
            # Ensure high >= max(open, close) and low <= min(open, close)
            df_cleaned['high'] = df_cleaned[['high', 'open', 'close']].max(axis=1)
            df_cleaned['low'] = df_cleaned[['low', 'open', 'close']].min(axis=1)
        
        # Remove records with non-positive prices
        if aggressive:
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df_cleaned.columns:
                    before_count = len(df_cleaned)
                    df_cleaned = df_cleaned[df_cleaned[col] > 0]
                    after_count = len(df_cleaned)
                    
                    if before_count != after_count:
                        self.logger.info(f"Removed {before_count - after_count} records with non-positive {col}")
        
        return df_cleaned


# Factory function
def create_quality_framework() -> DataQualityFramework:
    """Create data quality framework instance."""
    return DataQualityFramework()
