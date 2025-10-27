#!/usr/bin/env python3
"""
Data Download Helper for Criteo CTR Dataset

This script provides instructions and helpers for downloading the Criteo
Display Advertising Challenge dataset from Kaggle.

Note: Kaggle API credentials are required for automated download.
"""

import logging
import sys
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_kaggle_api() -> bool:
    """
    Check if Kaggle API is installed and configured.

    Returns:
        True if Kaggle API is available, False otherwise
    """
    try:
        import kaggle
        return True
    except ImportError:
        return False


def print_manual_instructions():
    """Print manual download instructions."""
    logger.info("=" * 80)
    logger.info("Manual Download Instructions")
    logger.info("=" * 80)
    print("""
To download the Criteo CTR dataset manually:

1. Visit the Kaggle competition page:
   https://www.kaggle.com/c/criteo-display-ad-challenge/data

2. Accept the competition rules (you must have a Kaggle account)

3. Download the following files:
   - train.csv.gz (~11 GB compressed, ~40 GB uncompressed)
   - test.csv.gz (~2 GB compressed, ~6 GB uncompressed)

4. Extract the files:
   gunzip train.csv.gz
   gunzip test.csv.gz

5. Move the files to the data/ directory:
   mv train.csv data/
   mv test.csv data/

6. Verify the files:
   python scripts/verify_data.py
""")


def print_kaggle_api_setup():
    """Print Kaggle API setup instructions."""
    logger.info("=" * 80)
    logger.info("Kaggle API Setup Instructions")
    logger.info("=" * 80)
    print("""
To use the Kaggle API for automated downloads:

1. Install the Kaggle API:
   pip install kaggle

2. Get your Kaggle API credentials:
   a. Go to https://www.kaggle.com/account
   b. Scroll to "API" section
   c. Click "Create New API Token"
   d. This downloads kaggle.json

3. Place kaggle.json in the correct location:
   # Linux/macOS
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

   # Windows
   mkdir %USERPROFILE%\\.kaggle
   move %USERPROFILE%\\Downloads\\kaggle.json %USERPROFILE%\\.kaggle\\

4. Run this script again to download data automatically
""")


def download_with_kaggle_api(data_dir: Path) -> bool:
    """
    Download dataset using Kaggle API.

    Args:
        data_dir: Directory to save data

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Downloading data using Kaggle API...")
        logger.info("This may take a while (files are ~13 GB compressed)...")

        # Import here to handle case where it's not installed
        import kaggle

        # Create data directory
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset files
        logger.info("Downloading train.csv.gz...")
        kaggle.api.competition_download_file(
            'criteo-display-ad-challenge',
            'train.csv.gz',
            path=str(data_dir)
        )

        logger.info("Downloading test.csv.gz...")
        kaggle.api.competition_download_file(
            'criteo-display-ad-challenge',
            'test.csv.gz',
            path=str(data_dir)
        )

        logger.info("Download complete!")
        logger.info("Extracting files...")

        # Extract files
        import gzip
        import shutil

        # Extract train.csv
        train_gz = data_dir / 'train.csv.gz'
        train_csv = data_dir / 'train.csv'

        if train_gz.exists():
            logger.info("Extracting train.csv.gz...")
            with gzip.open(train_gz, 'rb') as f_in:
                with open(train_csv, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Extracted to {train_csv}")

        # Extract test.csv
        test_gz = data_dir / 'test.csv.gz'
        test_csv = data_dir / 'test.csv'

        if test_gz.exists():
            logger.info("Extracting test.csv.gz...")
            with gzip.open(test_gz, 'rb') as f_in:
                with open(test_csv, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Extracted to {test_csv}")

        logger.info("=" * 80)
        logger.info("Dataset downloaded and extracted successfully!")
        logger.info(f"  Train: {train_csv}")
        logger.info(f"  Test: {test_csv}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False


def create_sample_data(data_dir: Path, sample_size: int = 1_000_000):
    """
    Create a smaller sample dataset for testing.

    Args:
        data_dir: Directory containing data
        sample_size: Number of rows to include in sample
    """
    train_file = data_dir / 'train.csv'
    sample_file = data_dir / 'train_sample.csv'

    if not train_file.exists():
        logger.error(f"Train file not found: {train_file}")
        return

    logger.info(f"Creating sample dataset with {sample_size:,} rows...")

    try:
        # Use head command for efficiency
        result = subprocess.run(
            ['head', '-n', str(sample_size + 1), str(train_file)],
            capture_output=True,
            text=True,
            check=True
        )

        with open(sample_file, 'w') as f:
            f.write(result.stdout)

        logger.info(f"Sample dataset created: {sample_file}")

    except Exception as e:
        logger.error(f"Error creating sample: {e}")


def verify_data(data_dir: Path):
    """
    Verify that data files exist and have reasonable size.

    Args:
        data_dir: Directory containing data
    """
    logger.info("Verifying data files...")

    train_file = data_dir / 'train.csv'
    test_file = data_dir / 'test.csv'

    if train_file.exists():
        size_gb = train_file.stat().st_size / (1024 ** 3)
        logger.info(f"✓ train.csv exists ({size_gb:.2f} GB)")

        # Expected: ~40 GB
        if size_gb < 30 or size_gb > 50:
            logger.warning(
                f"Warning: train.csv size ({size_gb:.2f} GB) outside expected range (30-50 GB)"
            )
    else:
        logger.error("✗ train.csv not found")

    if test_file.exists():
        size_gb = test_file.stat().st_size / (1024 ** 3)
        logger.info(f"✓ test.csv exists ({size_gb:.2f} GB)")

        # Expected: ~6 GB
        if size_gb < 4 or size_gb > 8:
            logger.warning(
                f"Warning: test.csv size ({size_gb:.2f} GB) outside expected range (4-8 GB)"
            )
    else:
        logger.error("✗ test.csv not found")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download Criteo CTR dataset'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to save data (default: ./data)'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Create a sample dataset after downloading'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1_000_000,
        help='Sample size in rows (default: 1,000,000)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data files'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Criteo CTR Dataset Downloader")
    logger.info("=" * 80)

    # Verify only mode
    if args.verify_only:
        verify_data(data_dir)
        return

    # Check if Kaggle API is available
    has_kaggle = check_kaggle_api()

    if has_kaggle:
        logger.info("Kaggle API detected!")
        response = input("Download data using Kaggle API? (y/n): ")

        if response.lower() == 'y':
            success = download_with_kaggle_api(data_dir)

            if success:
                if args.sample:
                    create_sample_data(data_dir, args.sample_size)
                return
    else:
        logger.warning("Kaggle API not installed or not configured")
        print_kaggle_api_setup()
        print()

    # Fall back to manual instructions
    print_manual_instructions()

    # Verify if data already exists
    logger.info("\nChecking for existing data files...")
    verify_data(data_dir)


if __name__ == '__main__':
    main()
