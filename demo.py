#!/usr/bin/env python3
"""
ML-TA System Demonstration Script

This script demonstrates the core functionality of the ML-TA system including
data fetching, feature engineering, model training, and prediction.
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_config
from src.logging_config import get_logger
from src.utils import set_deterministic_seed, ensure_directory
from src.exceptions import MLTAException

# Initialize logger
logger = get_logger("demo").get_logger()


class MLTADemo:
    """Demonstration of ML-TA system capabilities."""
    
    def __init__(self):
        """Initialize demo with configuration."""
        self.config = get_config()
        self.logger = logger.bind(component="demo")
        
        # Set deterministic seed for reproducibility
        set_deterministic_seed(self.config.app.seed)
        
        # Ensure directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup required directories."""
        for path_name, path_value in self.config.paths.dict().items():
            ensure_directory(path_value)
            self.logger.info(f"Directory ensured: {path_value}")
    
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data for demonstration."""
        self.logger.info("Generating sample OHLCV data")
        
        # Generate 1000 data points over the last 1000 hours
        n_points = 1000
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=n_points)
        
        # Create timestamp series
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_points)
        
        # Generate realistic price data using random walk
        np.random.seed(self.config.app.seed)
        base_price = 50000  # Starting price for BTCUSDT
        
        # Generate price movements with volatility
        returns = np.random.normal(0, 0.02, n_points)  # 2% hourly volatility
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Add some intrabar volatility
            volatility = abs(returns[i]) * price * 0.5
            
            open_price = price
            close_price = price * (1 + np.random.normal(0, 0.001))  # Small close variation
            
            high = max(open_price, close_price) + np.random.uniform(0, volatility)
            low = min(open_price, close_price) - np.random.uniform(0, volatility)
            
            # Ensure OHLC relationships are maintained
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            volume = np.random.lognormal(mean=5, sigma=1)  # Realistic volume distribution
            
            data.append({
                'timestamp': timestamp,
                'symbol': 'BTCUSDT',
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2)
            })
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} OHLCV records", 
                        start_price=df['close'].iloc[0],
                        end_price=df['close'].iloc[-1])
        
        return df
    
    def demonstrate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Demonstrate technical indicator calculations."""
        self.logger.info("Calculating technical indicators")
        
        # Simple moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        indicator_count = len([col for col in df.columns if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        self.logger.info(f"Calculated {indicator_count} technical indicators")
        
        return df
    
    def demonstrate_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Demonstrate feature engineering pipeline."""
        self.logger.info("Engineering features")
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Interaction features
        df['price_volume_interaction'] = df['close'] * df['volume']
        df['rsi_volume_interaction'] = df['rsi'] * df['volume_ratio']
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Target variables (for supervised learning)
        for horizon in [1, 3, 5]:
            df[f'target_return_{horizon}'] = df['close'].pct_change(periods=horizon).shift(-horizon)
            df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
        
        feature_count = len([col for col in df.columns if col.startswith(('sma_', 'ema_', 'rsi', 'bb_', 'volume_', 'close_', 'price_', 'target_'))])
        self.logger.info(f"Engineered {feature_count} features")
        
        return df
    
    def demonstrate_data_quality_checks(self, df: pd.DataFrame) -> dict:
        """Demonstrate data quality assessment."""
        self.logger.info("Performing data quality checks")
        
        quality_report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'price_statistics': {
                'min_price': df['close'].min(),
                'max_price': df['close'].max(),
                'mean_price': df['close'].mean(),
                'price_volatility': df['close'].std()
            }
        }
        
        # Check for data quality issues
        issues = []
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 5:
            issues.append(f"High missing value percentage: {missing_pct:.2f}%")
        
        # Check for duplicates
        if quality_report['duplicate_records'] > 0:
            issues.append(f"Found {quality_report['duplicate_records']} duplicate records")
        
        # Check for price anomalies
        price_changes = df['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.1).sum()  # >10% price changes
        if extreme_changes > len(df) * 0.01:  # More than 1% of records
            issues.append(f"Found {extreme_changes} extreme price changes (>10%)")
        
        quality_report['issues'] = issues
        quality_report['quality_score'] = max(0, 100 - len(issues) * 20)  # Simple scoring
        
        self.logger.info("Data quality assessment completed",
                        quality_score=quality_report['quality_score'],
                        issues_found=len(issues))
        
        return quality_report
    
    def demonstrate_model_simulation(self, df: pd.DataFrame) -> dict:
        """Simulate model training and prediction."""
        self.logger.info("Simulating model training and prediction")
        
        # Prepare feature matrix (using available features, dropping NaN)
        feature_cols = [col for col in df.columns if col.startswith(('sma_', 'ema_', 'rsi', 'bb_', 'volume_ratio', 'price_change'))]
        target_col = 'target_direction_1'
        
        # Create clean dataset
        model_data = df[feature_cols + [target_col]].dropna()
        
        if len(model_data) < 100:
            self.logger.warning("Insufficient data for model simulation")
            return {'status': 'insufficient_data', 'records': len(model_data)}
        
        # Split data (80/20 train/test)
        split_idx = int(len(model_data) * 0.8)
        train_data = model_data.iloc[:split_idx]
        test_data = model_data.iloc[split_idx:]
        
        # Simulate model training (using simple statistics)
        feature_importance = {}
        for feature in feature_cols:
            # Calculate correlation with target as proxy for importance
            correlation = abs(train_data[feature].corr(train_data[target_col]))
            feature_importance[feature] = correlation if not np.isnan(correlation) else 0
        
        # Simulate predictions (random but realistic)
        np.random.seed(self.config.app.seed)
        predictions = np.random.choice([0, 1], size=len(test_data), p=[0.45, 0.55])
        actual = test_data[target_col].values
        
        # Calculate metrics
        accuracy = (predictions == actual).mean()
        precision = np.sum((predictions == 1) & (actual == 1)) / max(np.sum(predictions == 1), 1)
        recall = np.sum((predictions == 1) & (actual == 1)) / max(np.sum(actual == 1), 1)
        f1_score = 2 * (precision * recall) / max((precision + recall), 1e-8)
        
        model_results = {
            'status': 'completed',
            'training_records': len(train_data),
            'test_records': len(test_data),
            'feature_count': len(feature_cols),
            'metrics': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4)
            },
            'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        }
        
        self.logger.info("Model simulation completed",
                        accuracy=model_results['metrics']['accuracy'],
                        feature_count=model_results['feature_count'])
        
        return model_results
    
    def save_results(self, df: pd.DataFrame, quality_report: dict, model_results: dict):
        """Save demonstration results."""
        self.logger.info("Saving demonstration results")
        
        # Save processed data
        data_path = Path(self.config.paths.data) / "processed_demo_data.parquet"
        df.to_parquet(data_path, compression='snappy')
        
        # Save quality report
        import json
        quality_path = Path(self.config.paths.artefacts) / "quality_report.json"
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        # Save model results
        model_path = Path(self.config.paths.artefacts) / "model_results.json"
        with open(model_path, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
        
        self.logger.info("Results saved",
                        data_path=str(data_path),
                        quality_path=str(quality_path),
                        model_path=str(model_path))
    
    async def run_demo(self):
        """Run complete demonstration."""
        try:
            self.logger.info("Starting ML-TA system demonstration")
            
            # Step 1: Generate sample data
            df = self.generate_sample_data()
            
            # Step 2: Calculate technical indicators
            df = self.demonstrate_technical_indicators(df)
            
            # Step 3: Engineer features
            df = self.demonstrate_feature_engineering(df)
            
            # Step 4: Assess data quality
            quality_report = self.demonstrate_data_quality_checks(df)
            
            # Step 5: Simulate model training
            model_results = self.demonstrate_model_simulation(df)
            
            # Step 6: Save results
            self.save_results(df, quality_report, model_results)
            
            # Summary
            self.logger.info("Demonstration completed successfully",
                           total_records=len(df),
                           features_engineered=len([col for col in df.columns if 'target' not in col]) - 6,  # Subtract OHLCV + timestamp + symbol
                           quality_score=quality_report['quality_score'],
                           model_accuracy=model_results.get('metrics', {}).get('accuracy', 0))
            
            print("\n" + "="*60)
            print("ML-TA SYSTEM DEMONSTRATION COMPLETED")
            print("="*60)
            print(f"✓ Generated {len(df):,} OHLCV records")
            print(f"✓ Calculated {len([col for col in df.columns if col.startswith(('sma_', 'ema_', 'rsi', 'bb_'))])} technical indicators")
            print(f"✓ Engineered {len([col for col in df.columns if 'target' not in col]) - 6} features")
            print(f"✓ Data quality score: {quality_report['quality_score']}/100")
            if model_results.get('status') == 'completed':
                print(f"✓ Model accuracy: {model_results['metrics']['accuracy']:.2%}")
            print(f"✓ Results saved to: {self.config.paths.artefacts}")
            print("="*60)
            
        except Exception as e:
            self.logger.error("Demonstration failed", error=str(e))
            raise MLTAException(f"Demo failed: {e}")


async def main():
    """Main entry point for demonstration."""
    try:
        demo = MLTADemo()
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
