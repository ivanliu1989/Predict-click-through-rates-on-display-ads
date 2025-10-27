"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import csv


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    return [
        {'Id': '1', 'Label': '0', 'I1': '5', 'I2': '10', 'C1': 'abc123', 'C2': 'def456'},
        {'Id': '2', 'Label': '1', 'I1': '3', 'I2': '', 'C1': 'abc123', 'C2': 'xyz789'},
        {'Id': '3', 'Label': '0', 'I1': '', 'I2': '20', 'C1': 'ghi012', 'C2': 'def456'},
    ]


@pytest.fixture
def temp_csv_file(sample_csv_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_data.csv"

    fieldnames = ['Id', 'Label', 'I1', 'I2', 'C1', 'C2']

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_csv_data)

    return csv_file


@pytest.fixture
def sample_train_data():
    """Generate sample training data (X, y)."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
