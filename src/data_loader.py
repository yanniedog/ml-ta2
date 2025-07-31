"""
Advanced Data Loading for ML-TA System

This module provides comprehensive data loading capabilities with support for
multiple data sources, lazy loading, caching, versioning, and streaming.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Iterator, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Handle optional dependencies gracefully
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging
    structlog = logging

from .config import get_config
from .exceptions import DataFetchError, ValidationError, SystemResourceError
from .utils import ensure_directory, save_parquet, load_parquet, optimize_dataframe_memory
from .data_quality import DataQualityFramework
from .logging_config import get_logger

logger = get_logger("data_loader").get_logger()


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from source."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source."""
        pass


class FileDataSource(DataSource):
    """Data source for file-based data (Parquet, CSV, etc.)."""
    
    def __init__(self, file_path: Union[str, Path], file_format: str = "parquet"):
        """Initialize file data source."""
        self.file_path = Path(file_path)
        self.file_format = file_format.lower()
        self.logger = logger.bind(component="file_data_source")
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from file."""
        if not self.file_path.exists():
            raise DataFetchError(f"File not found: {self.file_path}")
        
        try:
            if self.file_format == "parquet":
                df = load_parquet(self.file_path, **kwargs)
            elif self.file_format == "csv":
                df = pd.read_csv(self.file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
            
            self.logger.info(f"Loaded {len(df)} records from {self.file_path}")
            return df
        
        except Exception as e:
            raise DataFetchError(f"Failed to load data from {self.file_path}: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get file metadata."""
        if not self.file_path.exists():
            return {}
        
        stat = self.file_path.stat()
        return {
            "file_path": str(self.file_path),
            "file_format": self.file_format,
            "file_size_mb": round(stat.st_size / 1024 / 1024, 2),
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }


class DatabaseDataSource(DataSource):
    """Data source for database connections."""
    
    def __init__(self, connection_string: str, table_name: str):
        """Initialize database data source."""
        self.connection_string = connection_string
        self.table_name = table_name
        self.logger = logger.bind(component="database_data_source")
    
    def load_data(self, query: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from database."""
        try:
            import sqlalchemy as sa
            
            engine = sa.create_engine(self.connection_string)
            
            if query:
                df = pd.read_sql(query, engine, **kwargs)
            else:
                df = pd.read_sql_table(self.table_name, engine, **kwargs)
            
            self.logger.info(f"Loaded {len(df)} records from database table {self.table_name}")
            return df
        
        except Exception as e:
            raise DataFetchError(f"Failed to load data from database: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get database metadata."""
        return {
            "connection_string": self.connection_string.split('@')[0] + "@***",  # Hide credentials
            "table_name": self.table_name
        }


class DataCatalog:
    """Maintains metadata about all datasets including schemas, lineage, and quality metrics."""
    
    def __init__(self, catalog_dir: str = "data/catalog"):
        """Initialize data catalog."""
        self.catalog_dir = Path(catalog_dir)
        ensure_directory(self.catalog_dir)
        self.logger = logger.bind(component="data_catalog")
        self._load_catalog()
    
    def _load_catalog(self) -> None:
        """Load existing catalog from disk."""
        catalog_file = self.catalog_dir / "catalog.json"
        
        if catalog_file.exists():
            try:
                with open(catalog_file, 'r') as f:
                    self.catalog = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load catalog: {e}")
                self.catalog = {}
        else:
            self.catalog = {}
    
    def _save_catalog(self) -> None:
        """Save catalog to disk."""
        catalog_file = self.catalog_dir / "catalog.json"
        
        try:
            with open(catalog_file, 'w') as f:
                json.dump(self.catalog, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save catalog: {e}")
    
    def register_dataset(
        self,
        dataset_id: str,
        metadata: Dict[str, Any],
        schema: Optional[Dict[str, str]] = None,
        quality_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a dataset in the catalog."""
        entry = {
            "dataset_id": dataset_id,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata,
            "schema": schema or {},
            "quality_metrics": quality_metrics or {},
            "access_count": 0,
            "last_accessed": None
        }
        
        self.catalog[dataset_id] = entry
        self._save_catalog()
        
        self.logger.info(f"Registered dataset: {dataset_id}")
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset."""
        return self.catalog.get(dataset_id)
    
    def update_access_stats(self, dataset_id: str) -> None:
        """Update access statistics for a dataset."""
        if dataset_id in self.catalog:
            self.catalog[dataset_id]["access_count"] += 1
            self.catalog[dataset_id]["last_accessed"] = datetime.now().isoformat()
            self._save_catalog()
    
    def list_datasets(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List datasets matching criteria."""
        if not filter_criteria:
            return list(self.catalog.keys())
        
        # Simple filtering implementation
        matching_datasets = []
        for dataset_id, entry in self.catalog.items():
            match = True
            for key, value in filter_criteria.items():
                if key not in entry["metadata"] or entry["metadata"][key] != value:
                    match = False
                    break
            
            if match:
                matching_datasets.append(dataset_id)
        
        return matching_datasets


class DataVersioning:
    """Tracks data changes and enables reproducible experiments."""
    
    def __init__(self, versions_dir: str = "data/versions"):
        """Initialize data versioning."""
        self.versions_dir = Path(versions_dir)
        ensure_directory(self.versions_dir)
        self.logger = logger.bind(component="data_versioning")
    
    def create_version(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        version_tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new version of a dataset."""
        # Generate version ID
        if version_tag:
            version_id = f"{dataset_id}_{version_tag}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = f"{dataset_id}_{timestamp}"
        
        # Calculate data hash for integrity
        data_hash = self._calculate_data_hash(df)
        
        # Save data
        version_path = self.versions_dir / f"{version_id}.parquet"
        save_parquet(df, version_path)
        
        # Save metadata
        version_metadata = {
            "version_id": version_id,
            "dataset_id": dataset_id,
            "created_at": datetime.now().isoformat(),
            "data_hash": data_hash,
            "record_count": len(df),
            "columns": list(df.columns),
            "file_path": str(version_path),
            "metadata": metadata or {}
        }
        
        metadata_path = self.versions_dir / f"{version_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Created data version: {version_id}")
        return version_id
    
    def load_version(self, version_id: str) -> pd.DataFrame:
        """Load a specific version of a dataset."""
        version_path = self.versions_dir / f"{version_id}.parquet"
        
        if not version_path.exists():
            raise DataFetchError(f"Version not found: {version_id}")
        
        df = load_parquet(version_path)
        self.logger.info(f"Loaded data version: {version_id}")
        return df
    
    def get_version_metadata(self, version_id: str) -> Dict[str, Any]:
        """Get metadata for a specific version."""
        metadata_path = self.versions_dir / f"{version_id}_metadata.json"
        
        if not metadata_path.exists():
            raise DataFetchError(f"Version metadata not found: {version_id}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_versions(self, dataset_id: Optional[str] = None) -> List[str]:
        """List available versions."""
        versions = []
        
        for metadata_file in self.versions_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if dataset_id is None or metadata.get("dataset_id") == dataset_id:
                    versions.append(metadata["version_id"])
            
            except Exception as e:
                self.logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
        
        return sorted(versions)
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for integrity checking."""
        # Convert DataFrame to string representation and hash
        df_string = df.to_string()
        return hashlib.sha256(df_string.encode()).hexdigest()


class LazyLoader:
    """Loads data on-demand to optimize memory usage."""
    
    def __init__(self, data_source: DataSource, chunk_size: Optional[int] = None):
        """Initialize lazy loader."""
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.logger = logger.bind(component="lazy_loader")
        self._cached_data = None
        self._loaded = False
    
    def load(self, force_reload: bool = False, **kwargs) -> pd.DataFrame:
        """Load data lazily."""
        if not self._loaded or force_reload:
            self.logger.info("Loading data from source")
            self._cached_data = self.data_source.load_data(**kwargs)
            self._loaded = True
        
        return self._cached_data
    
    def load_chunks(self, **kwargs) -> Iterator[pd.DataFrame]:
        """Load data in chunks."""
        if self.chunk_size is None:
            yield self.load(**kwargs)
            return
        
        df = self.load(**kwargs)
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size].copy()
            yield chunk
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the data without loading it."""
        metadata = self.data_source.get_metadata()
        metadata["loaded"] = self._loaded
        
        if self._loaded and self._cached_data is not None:
            metadata["record_count"] = len(self._cached_data)
            metadata["memory_usage_mb"] = self._cached_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        return metadata


class DataLoader:
    """Main data loader with support for multiple sources and advanced features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data loader."""
        from .config import get_model_dict
        self.config = config or get_model_dict(get_config())
        self.logger = logger.bind(component="data_loader")
        
        # Initialize components
        self.catalog = DataCatalog()
        self.versioning = DataVersioning()
        self.quality_framework = DataQualityFramework()
        
        # Cache for lazy loaders
        self._loaders_cache: Dict[str, LazyLoader] = {}
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        file_format: str = "parquet",
        lazy: bool = False,
        register_in_catalog: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, LazyLoader]:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            file_format: File format ('parquet', 'csv')
            lazy: Whether to use lazy loading
            register_in_catalog: Whether to register in catalog
            **kwargs: Additional arguments for loading
        
        Returns:
            DataFrame or LazyLoader instance
        """
        file_path = Path(file_path)
        data_source = FileDataSource(file_path, file_format)
        
        if lazy:
            loader = LazyLoader(data_source)
            
            # Cache the loader
            cache_key = str(file_path)
            self._loaders_cache[cache_key] = loader
            
            return loader
        else:
            df = data_source.load_data(**kwargs)
            
            # Optimize memory usage
            df = optimize_dataframe_memory(df)
            
            # Register in catalog if requested
            if register_in_catalog:
                dataset_id = file_path.stem
                metadata = data_source.get_metadata()
                schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
                # Assess quality
                quality_metrics = self.quality_framework.assess_quality(df, dataset_id)
                
                self.catalog.register_dataset(
                    dataset_id,
                    metadata,
                    schema,
                    quality_metrics.__dict__
                )
            
            return df
    
    def load_from_database(
        self,
        connection_string: str,
        table_name: str,
        query: Optional[str] = None,
        lazy: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, LazyLoader]:
        """Load data from database."""
        data_source = DatabaseDataSource(connection_string, table_name)
        
        if lazy:
            loader = LazyLoader(data_source)
            return loader
        else:
            df = data_source.load_data(query=query, **kwargs)
            return optimize_dataframe_memory(df)
    
    def load_multiple_files(
        self,
        file_pattern: str,
        file_format: str = "parquet",
        combine: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load multiple files matching a pattern.
        
        Args:
            file_pattern: Glob pattern for files
            file_format: File format
            combine: Whether to combine into single DataFrame
            **kwargs: Additional loading arguments
        
        Returns:
            Combined DataFrame or dictionary of DataFrames
        """
        from glob import glob
        
        file_paths = glob(file_pattern)
        
        if not file_paths:
            raise DataFetchError(f"No files found matching pattern: {file_pattern}")
        
        dataframes = {}
        
        for file_path in file_paths:
            try:
                df = self.load_from_file(
                    file_path,
                    file_format=file_format,
                    register_in_catalog=False,
                    **kwargs
                )
                
                file_key = Path(file_path).stem
                dataframes[file_key] = df
                
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
        
        if combine and dataframes:
            combined_df = pd.concat(dataframes.values(), ignore_index=True)
            self.logger.info(f"Combined {len(dataframes)} files into {len(combined_df)} records")
            return combined_df
        
        return dataframes
    
    def create_dataset_version(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        version_tag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a versioned dataset."""
        return self.versioning.create_version(df, dataset_id, version_tag, metadata)
    
    def load_dataset_version(self, version_id: str) -> pd.DataFrame:
        """Load a specific dataset version."""
        return self.versioning.load_version(version_id)
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset information from catalog."""
        return self.catalog.get_dataset_info(dataset_id)
    
    def list_datasets(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List available datasets."""
        return self.catalog.list_datasets(filter_criteria)
    
    def clear_cache(self) -> None:
        """Clear loader cache."""
        self._loaders_cache.clear()
        self.logger.info("Loader cache cleared")


# Factory function
def create_data_loader(config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """Create data loader instance."""
    return DataLoader(config)


# Example usage
if __name__ == "__main__":
    # Test data loader
    loader = create_data_loader()
    
    # Example: Load from file
    # df = loader.load_from_file("data/sample.parquet")
    
    # Example: Lazy loading
    # lazy_loader = loader.load_from_file("data/large_file.parquet", lazy=True)
    # df = lazy_loader.load()
    
    print("Data loader module ready")
