"""
Automated cleanup script for redundant test files.

This script safely removes duplicate and obsolete test files while preserving
essential testing functionality in the proper tests/ directory structure.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set
import sys
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Try to use structured logging, fall back to basic logging
try:
    from logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Files to be removed (redundant/obsolete test files)
REDUNDANT_TEST_FILES: Set[str] = {
    'simple_test.py',
    'quick_test.py',
    'fix_ab_testing.py', 
    'fix_ab_testing2.py',
    'simple_phase4_test.py'
}

# Phase test files to be consolidated into tests/
PHASE_TEST_FILES: List[str] = [
    'direct_phase4_test.py',
    'direct_phase5_test.py', 
    'direct_phase6_test.py',
    'direct_phase7_test.py',
    'direct_phase8_test.py',
    'direct_phase9_test.py',
    'phase1_quality_gate.py',
    'phase2_quality_gate.py',
    'phase3_quality_gate.py'
]

# Files to be moved to appropriate directories
RELOCATION_MAP = {
    'demo.py': 'examples/',
    'feature_engineering_demo.py': 'examples/',
    'launch.bat': 'scripts/',
    'launch_local.py': 'scripts/',
    'web_frontend.py': 'web_app/'
}


def create_backup_directory() -> Path:
    """Create backup directory for moved files."""
    backup_dir = Path('backup/cleanup_' + str(int(time.time())))
    backup_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created backup directory: {backup_dir}")
    return backup_dir


def backup_and_remove_file(file_path: Path, backup_dir: Path) -> bool:
    """Backup a file and remove it from original location."""
    try:
        if file_path.exists():
            # Create backup
            backup_path = backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up {file_path.name} to {backup_path}")
            
            # Remove original
            file_path.unlink()
            logger.info(f"Removed redundant file: {file_path.name}")
            return True
        else:
            logger.warning(f"File not found: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


def move_file_to_directory(source: Path, target_dir: str, backup_dir: Path) -> bool:
    """Move file to appropriate directory."""
    try:
        if not source.exists():
            logger.warning(f"Source file not found: {source}")
            return False
            
        # Create target directory if it doesn't exist
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Backup original
        backup_path = backup_dir / source.name
        shutil.copy2(source, backup_path)
        
        # Move to target
        final_path = target_path / source.name
        shutil.move(source, final_path)
        
        logger.info(f"Moved {source.name} to {final_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error moving {source} to {target_dir}: {e}")
        return False


def consolidate_phase_tests(phase_files: List[str], backup_dir: Path) -> bool:
    """Consolidate phase test files into proper test structure."""
    success_count = 0
    
    for phase_file in phase_files:
        file_path = Path(phase_file)
        if file_path.exists():
            try:
                # Backup the file
                backup_path = backup_dir / file_path.name
                shutil.copy2(file_path, backup_path)
                
                # The functionality from these files should be integrated into
                # the proper tests/ directory structure. For now, we'll move them
                # to a legacy directory for manual review.
                legacy_dir = Path('tests/legacy')
                legacy_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = legacy_dir / file_path.name
                shutil.move(file_path, target_path)
                
                logger.info(f"Moved {phase_file} to tests/legacy/ for review")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error consolidating {phase_file}: {e}")
    
    return success_count > 0


def cleanup_redundant_test_files() -> bool:
    """Main cleanup function."""
    import time
    
    logger.info("Starting redundant test file cleanup...")
    
    # Create backup directory
    backup_dir = create_backup_directory()
    
    success_count = 0
    total_operations = 0
    
    # Remove redundant files
    logger.info("Removing redundant test files...")
    for filename in REDUNDANT_TEST_FILES:
        file_path = Path(filename)
        total_operations += 1
        if backup_and_remove_file(file_path, backup_dir):
            success_count += 1
    
    # Consolidate phase test files
    logger.info("Consolidating phase test files...")
    total_operations += 1
    if consolidate_phase_tests(PHASE_TEST_FILES, backup_dir):
        success_count += 1
    
    # Relocate files to proper directories
    logger.info("Relocating files to proper directories...")
    for source_file, target_dir in RELOCATION_MAP.items():
        source_path = Path(source_file)
        total_operations += 1
        if move_file_to_directory(source_path, target_dir, backup_dir):
            success_count += 1
    
    # Create examples directory structure
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    
    # Create examples README
    examples_readme = examples_dir / 'README.md'
    if not examples_readme.exists():
        examples_readme.write_text("""# Examples Directory

This directory contains demonstration scripts and examples for the ML-TA system.

## Files

- `demo.py`: Basic system demonstration
- `feature_engineering_demo.py`: Feature engineering examples

## Usage

Run any example script from the project root:

```bash
python examples/demo.py
```
""")
    
    logger.info(f"Cleanup complete: {success_count}/{total_operations} operations successful")
    logger.info(f"Backup created at: {backup_dir}")
    
    return success_count == total_operations


if __name__ == "__main__":
    success = cleanup_redundant_test_files()
    if success:
        print("✅ Cleanup completed successfully")
        print("📁 Check the backup directory for moved files")
        print("🧪 Run the new consolidated tests with: pytest tests/test_core_functionality.py")
    else:
        print("❌ Some cleanup operations failed - check logs")
    sys.exit(0 if success else 1)
