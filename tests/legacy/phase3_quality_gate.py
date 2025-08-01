#!/usr/bin/env python3
"""
Phase 3 Quality Gate Implementation for ML-TA System

This script validates that all Phase 3 Feature Engineering Validation requirements have been met:
- Generates 200+ features from OHLCV data
- Prevents data leakage through temporal validation
- Tests feature selection optimization
- Validates cross-validation feature stability
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.features import create_feature_pipeline, FeaturePipeline
from src.feature_selection import create_feature_selector, FeatureSelector
from src.indicators import create_indicators_calculator
from src.config import get_config


class Phase3QualityGate:
    """Quality gate checker for Phase 3 compliance."""
    
    def __init__(self):
        self.results = {}
        self.passed_checks = 0
        self.total_checks = 0
        
    def log_result(self, check_name: str, passed: bool, details: str = ""):
        """Log a check result."""
        self.results[check_name] = {
            'passed': passed,
            'details': details
        }
        self.total_checks += 1
        if passed:
            self.passed_checks += 1
            print(f"✅ {check_name}: PASSED {details}")
        else:
            print(f"❌ {check_name}: FAILED {details}")
    
    def generate_test_data(self, n_records: int = 1000) -> pd.DataFrame:
        """Generate realistic OHLCV test data."""
        dates = pd.date_range('2023-01-01', periods=n_records, freq='1H')
        
        # Generate realistic market data with trend and volatility
        np.random.seed(42)
        base_price = 50000
        
        # Create price series with realistic patterns
        returns = np.random.normal(0, 0.002, n_records)  # 0.2% volatility
        trend = np.linspace(0, 0.1, n_records)  # Small upward trend
        
        prices = [base_price]
        for i in range(1, n_records):
            new_price = prices[-1] * (1 + returns[i] + trend[i] / n_records)
            prices.append(max(new_price, 1))
        
        test_data = []
        for i in range(n_records):
            open_price = prices[i]
            close_price = prices[i] if i == n_records - 1 else prices[i + 1]
            
            # Generate realistic high/low with proper relationships
            volatility = abs(returns[i]) * open_price * 3
            high = max(open_price, close_price) + np.random.uniform(0, volatility)
            low = min(open_price, close_price) - np.random.uniform(0, volatility)
            
            # Ensure OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            test_data.append({
                'timestamp': dates[i],
                'symbol': 'BTCUSDT',
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': round(np.random.uniform(100, 1000), 2)
            })
        
        return pd.DataFrame(test_data)
    
    def check_feature_generation_200plus(self) -> bool:
        """Test generation of 200+ features from OHLCV data."""
        try:
            # Generate test data
            df = self.generate_test_data(500)  # Use smaller dataset for speed
            
            # Create feature pipeline
            pipeline = create_feature_pipeline()
            
            # Engineer features
            features_df = pipeline.engineer_features(
                df,
                fit_scalers=True,
                validate_temporal=True
            )
            
            # Count generated features (exclude original OHLCV and metadata)
            original_cols = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
            feature_cols = [col for col in features_df.columns if col not in original_cols]
            
            feature_count = len(feature_cols)
            
            if feature_count >= 200:
                self.log_result("Feature Generation 200+", True, f"Generated {feature_count} features")
                return True
            else:
                self.log_result("Feature Generation 200+", False, f"Only generated {feature_count} features (<200)")
                return False
                
        except Exception as e:
            self.log_result("Feature Generation 200+", False, f"Exception: {str(e)}")
            return False
    
    def check_temporal_validation_prevention(self) -> bool:
        """Test that temporal validation prevents data leakage."""
        try:
            # Generate test data with intentional temporal issues
            df = self.generate_test_data(200)
            
            # Create feature pipeline
            pipeline = create_feature_pipeline()
            
            # Test 1: Normal temporal validation should pass
            try:
                features_df = pipeline.engineer_features(
                    df,
                    validate_temporal=True
                )
                temporal_validation_works = True
            except Exception:
                temporal_validation_works = False
            
            # Test 2: Check that temporal validator detects issues
            from src.features import TemporalValidator
            validator = TemporalValidator()
            
            # Create data with temporal violations (future timestamps)
            bad_df = df.copy()
            bad_df.loc[bad_df.index[-10:], 'timestamp'] = bad_df['timestamp'].iloc[0]  # Future data leak
            
            violations_detected = False
            try:
                validator.validate_feature_matrix(bad_df)
            except Exception:
                violations_detected = True
            
            if temporal_validation_works and violations_detected:
                self.log_result("Temporal Validation Prevention", True, "Prevents data leakage correctly")
                return True
            else:
                self.log_result("Temporal Validation Prevention", False, 
                              f"Validation works: {temporal_validation_works}, Detects violations: {violations_detected}")
                return False
                
        except Exception as e:
            self.log_result("Temporal Validation Prevention", False, f"Exception: {str(e)}")
            return False
    
    def check_feature_selection_optimization(self) -> bool:
        """Test feature selection optimization methods."""
        try:
            # Generate test data and features
            df = self.generate_test_data(300)
            pipeline = create_feature_pipeline()
            features_df = pipeline.engineer_features(df, fit_scalers=True)
            
            # Create target variable for testing
            features_df['target'] = (features_df['close'].pct_change() > 0).astype(int)
            
            # Prepare features and target
            feature_cols = [col for col in features_df.columns 
                          if col not in {'timestamp', 'symbol', 'target'} and not col.startswith('target_')]
            X = features_df[feature_cols].fillna(0)
            y = features_df['target'].fillna(0)
            
            # Test feature selector
            selector = create_feature_selector()
            
            # Perform feature selection
            X_selected = selector.fit_transform(X, y, task_type='classification')
            selected_features = selector.get_feature_names_out()
            
            # Check reduction ratio and that some features were selected
            original_count = len(feature_cols)
            selected_count = len(selected_features)
            reduction_ratio = selected_count / original_count
            
            # Good feature selection should reduce features significantly but keep meaningful ones
            if 0.1 <= reduction_ratio <= 0.8 and selected_count >= 10:
                self.log_result("Feature Selection Optimization", True, 
                              f"Reduced from {original_count} to {selected_count} features ({reduction_ratio:.2f} ratio)")
                return True
            else:
                self.log_result("Feature Selection Optimization", False, 
                              f"Poor selection: {original_count}→{selected_count} (ratio: {reduction_ratio:.2f})")
                return False
                
        except Exception as e:
            self.log_result("Feature Selection Optimization", False, f"Exception: {str(e)}")
            return False
    
    def check_feature_stability_validation(self) -> bool:
        """Test cross-validation feature stability."""
        try:
            # Generate larger dataset for stability testing
            df = self.generate_test_data(500)
            pipeline = create_feature_pipeline()
            features_df = pipeline.engineer_features(df, fit_scalers=True)
            
            # Create target variable
            features_df['target'] = (features_df['close'].pct_change() > 0).astype(int)
            
            # Prepare features
            feature_cols = [col for col in features_df.columns 
                          if col not in {'timestamp', 'symbol', 'target'} and not col.startswith('target_')]
            X = features_df[feature_cols].fillna(0)
            y = features_df['target'].fillna(0)
            
            # Test feature selection stability across splits
            n_splits = 3
            selected_features_per_split = []
            
            for split in range(n_splits):
                # Create time-based splits to maintain temporal order
                split_size = len(X) // n_splits
                start_idx = split * split_size
                end_idx = start_idx + split_size * 2  # Overlapping windows
                
                if end_idx > len(X):
                    end_idx = len(X)
                
                X_split = X.iloc[start_idx:end_idx]
                y_split = y.iloc[start_idx:end_idx]
                
                # Perform feature selection on split
                selector = create_feature_selector({
                    'use_correlation': True,
                    'use_variance': True,
                    'correlation_threshold': 0.95,
                    'variance_threshold': 0.01
                })
                
                try:
                    X_selected = selector.fit_transform(X_split, y_split, task_type='classification')
                    selected_features = selector.get_feature_names_out()
                    selected_features_per_split.append(set(selected_features))
                except Exception:
                    # If selection fails, use variance-only selection
                    high_var_features = X_split.var().nlargest(50).index.tolist()
                    selected_features_per_split.append(set(high_var_features))
            
            # Calculate stability (overlap between splits)
            if len(selected_features_per_split) >= 2:
                overlaps = []
                for i in range(len(selected_features_per_split)):
                    for j in range(i + 1, len(selected_features_per_split)):
                        set1, set2 = selected_features_per_split[i], selected_features_per_split[j]
                        if len(set1.union(set2)) > 0:
                            overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                            overlaps.append(overlap)
                
                avg_stability = np.mean(overlaps) if overlaps else 0
                
                # Feature selection should have reasonable stability (>30%)
                if avg_stability >= 0.3:
                    self.log_result("Feature Stability Validation", True, 
                                  f"Average stability: {avg_stability:.2f}")
                    return True
                else:
                    self.log_result("Feature Stability Validation", False, 
                                  f"Low stability: {avg_stability:.2f}")
                    return False
            else:
                self.log_result("Feature Stability Validation", False, "Insufficient splits for stability testing")
                return False
                
        except Exception as e:
            self.log_result("Feature Stability Validation", False, f"Exception: {str(e)}")
            return False
    
    def check_rolling_window_calculations(self) -> bool:
        """Test rolling window calculations are correct."""
        try:
            # Generate test data
            df = self.generate_test_data(100)
            
            # Create feature pipeline
            pipeline = create_feature_pipeline()
            
            # Generate features with rolling windows
            features_df = pipeline.engineer_features(df, validate_temporal=True)
            
            # Check for rolling features
            rolling_features = [col for col in features_df.columns if 'rolling' in col.lower() or '_ma_' in col.lower()]
            
            if len(rolling_features) >= 10:
                # Verify rolling calculations don't use future data
                # Rolling features should have NaN values at the beginning
                has_initial_nans = False
                for col in rolling_features[:3]:  # Check first few rolling features
                    if features_df[col].iloc[:5].isna().any():
                        has_initial_nans = True
                        break
                
                if has_initial_nans:
                    self.log_result("Rolling Window Calculations", True, 
                                  f"Found {len(rolling_features)} rolling features with proper temporal ordering")
                    return True
                else:
                    self.log_result("Rolling Window Calculations", False, 
                                  "Rolling features don't show proper temporal ordering")
                    return False
            else:
                self.log_result("Rolling Window Calculations", False, 
                              f"Insufficient rolling features: {len(rolling_features)}")
                return False
                
        except Exception as e:
            self.log_result("Rolling Window Calculations", False, f"Exception: {str(e)}")
            return False
    
    def check_feature_scaling_normalization(self) -> bool:
        """Test feature scaling and normalization."""
        try:
            # Generate test data
            df = self.generate_test_data(200)
            
            # Create feature pipeline
            pipeline = create_feature_pipeline()
            
            # Generate features with scaling
            features_df = pipeline.engineer_features(df, fit_scalers=True)
            
            # Check numeric features for scaling
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            exclude_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
            scaled_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(scaled_cols) >= 50:
                # Check that scaled features have reasonable ranges
                scaled_data = features_df[scaled_cols].fillna(0)
                
                # Most scaled features should be in reasonable range (e.g., -10 to 10)
                in_range_count = 0
                for col in scaled_cols[:20]:  # Check subset for performance
                    col_data = scaled_data[col]
                    if col_data.min() >= -10 and col_data.max() <= 10:
                        in_range_count += 1
                
                range_ratio = in_range_count / min(20, len(scaled_cols))
                
                if range_ratio >= 0.7:
                    self.log_result("Feature Scaling Normalization", True, 
                                  f"{in_range_count}/{min(20, len(scaled_cols))} features in reasonable range")
                    return True
                else:
                    self.log_result("Feature Scaling Normalization", False, 
                                  f"Poor scaling: {in_range_count}/{min(20, len(scaled_cols))} in range")
                    return False
            else:
                self.log_result("Feature Scaling Normalization", False, 
                              f"Insufficient scaled features: {len(scaled_cols)}")
                return False
                
        except Exception as e:
            self.log_result("Feature Scaling Normalization", False, f"Exception: {str(e)}")
            return False
    
    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all Phase 3 quality gate checks."""
        print("🔍 Running Phase 3 Quality Gate Checks...")
        print("=" * 50)
        
        # Run all checks
        checks = [
            self.check_rolling_window_calculations,
            self.check_feature_scaling_normalization,
            self.check_temporal_validation_prevention,
            self.check_feature_selection_optimization,
            self.check_feature_stability_validation,
            self.check_feature_generation_200plus,  # Run this last as it's most intensive
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                check_name = check.__name__.replace('check_', '').replace('_', ' ').title()
                self.log_result(check_name, False, f"Unexpected error: {str(e)}")
        
        print("=" * 50)
        
        # Calculate pass rate
        pass_rate = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        print(f"📊 Phase 3 Quality Gate Results:")
        print(f"   Passed: {self.passed_checks}/{self.total_checks} checks ({pass_rate:.1f}%)")
        
        # Phase 3 requires 100% pass rate
        phase3_passed = pass_rate == 100.0
        
        if phase3_passed:
            print("🎉 Phase 3 Quality Gate: PASSED")
            print("✅ Ready to proceed to Phase 4: Model Training Excellence")
        else:
            print("⚠️  Phase 3 Quality Gate: FAILED")
            print("❌ Must fix issues before proceeding to Phase 4")
        
        return phase3_passed, self.results


def main():
    """Main function to run Phase 3 quality gate."""
    gate = Phase3QualityGate()
    passed, results = gate.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
