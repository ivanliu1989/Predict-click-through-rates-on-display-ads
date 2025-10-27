"""
CSV to Vowpal Wabbit Format Converter

Converts Criteo CTR dataset from CSV format to Vowpal Wabbit format.

Original credit:
- __Author__: Triskelion <info@mlwave.com>
- Credit: Zygmunt Zając <zygmunt@fastml.com>

Modernized with:
- Python 3 compatibility
- Error handling and logging
- Progress reporting
- Configurable paths
- Better performance
"""

import logging
import sys
from datetime import datetime
from csv import DictReader
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def csv_to_vw(
    csv_path: Path,
    output_path: Path,
    is_train: bool = True,
    report_interval: int = 1_000_000
) -> None:
    """
    Convert CSV file to Vowpal Wabbit format.

    Vowpal Wabbit format:
    [label] ['tag] |namespace features

    Example train:
    1 'id123 |i I1:5 I2:10 |c C1 C2 C3

    Example test:
    1 'id456 |i I1:3 |c C5 C6

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output VW file
        is_train: Whether this is training data (includes labels)
        report_interval: How often to log progress

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    start_time = datetime.now()

    logger.info("=" * 80)
    logger.info(f"Converting CSV to Vowpal Wabbit format")
    logger.info(f"  Input: {csv_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Mode: {'Training' if is_train else 'Testing'}")
    logger.info("=" * 80)

    row_count = 0

    try:
        with open(csv_path, 'r', encoding='utf-8') as csv_file, \
             open(output_path, 'w', encoding='utf-8') as vw_file:

            reader = DictReader(csv_file)

            # Validate required fields
            if reader.fieldnames:
                if 'Id' not in reader.fieldnames:
                    raise ValueError("CSV must contain 'Id' column")
                if is_train and 'Label' not in reader.fieldnames:
                    raise ValueError("Training CSV must contain 'Label' column")

            for row_count, row in enumerate(reader, start=1):
                try:
                    # Create VW format line
                    vw_line = _convert_row_to_vw(row, is_train)
                    vw_file.write(vw_line + '\n')

                except Exception as e:
                    logger.warning(f"Error processing row {row_count}: {e}")
                    continue

                # Report progress
                if row_count % report_interval == 0:
                    elapsed = datetime.now() - start_time
                    rate = row_count / elapsed.total_seconds()
                    logger.info(
                        f"Processed {row_count:,} rows | "
                        f"Elapsed: {elapsed} | "
                        f"Rate: {rate:.0f} rows/sec"
                    )

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise

    elapsed = datetime.now() - start_time
    logger.info("=" * 80)
    logger.info(f"Conversion completed!")
    logger.info(f"  Rows processed: {row_count:,}")
    logger.info(f"  Total time: {elapsed}")
    logger.info(f"  Average rate: {row_count / elapsed.total_seconds():.0f} rows/sec")
    logger.info("=" * 80)


def _convert_row_to_vw(row: dict, is_train: bool) -> str:
    """
    Convert a single CSV row to Vowpal Wabbit format.

    Args:
        row: Dictionary from CSV DictReader
        is_train: Whether this is training data

    Returns:
        VW formatted string (without newline)
    """
    # Extract label and ID
    row_id = row.get('Id', 'unknown')

    if is_train:
        # VW uses 1 for positive, -1 for negative
        label = 1 if row.get('Label') == '1' else -1
    else:
        # Test data: use dummy label 1
        label = 1

    # Separate numerical and categorical features
    numerical_features = []
    categorical_features = []

    for key, value in row.items():
        # Skip label and ID
        if key in ['Label', 'Id']:
            continue

        # Skip empty values
        if not value or value.strip() == '':
            continue

        # Numerical features (start with 'I')
        if key.startswith('I'):
            numerical_features.append(f"{key}:{value}")

        # Categorical features (start with 'C')
        elif key.startswith('C'):
            # For categorical, just use the value (no key:value format)
            categorical_features.append(value)

    # Build VW format line
    # Format: label 'tag |namespace1 features1 |namespace2 features2
    numerical_str = ' '.join(numerical_features) if numerical_features else ''
    categorical_str = ' '.join(categorical_features) if categorical_features else ''

    vw_line = f"{label} '{row_id} |i {numerical_str} |c {categorical_str}"

    return vw_line


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert Criteo CSV to Vowpal Wabbit format'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input CSV file path'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Output VW file path'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Convert test data (no labels)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=1_000_000,
        help='Progress report interval (default: 1,000,000)'
    )

    args = parser.parse_args()

    # Convert paths
    csv_path = Path(args.input)
    output_path = Path(args.output)

    # Check if input exists
    if not csv_path.exists():
        logger.error(f"Input file not found: {csv_path}")
        sys.exit(1)

    # Warn if output exists
    if output_path.exists():
        logger.warning(f"Output file exists and will be overwritten: {output_path}")

    try:
        # Perform conversion
        csv_to_vw(
            csv_path=csv_path,
            output_path=output_path,
            is_train=not args.test,
            report_interval=args.interval
        )

        logger.info("Success!")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Example usage (uncomment to run):
    # csv_to_vw(
    #     csv_path=Path('train.csv'),
    #     output_path=Path('click.train.vw'),
    #     is_train=True
    # )
    # csv_to_vw(
    #     csv_path=Path('test.csv'),
    #     output_path=Path('click.test.vw'),
    #     is_train=False
    # )

    main()
