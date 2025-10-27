"""
Tests for CSV to Vowpal Wabbit converter (csv_to_vw_modernized.py).
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from csv_to_vw_modernized import _convert_row_to_vw, csv_to_vw


class TestConvertRowToVW:
    """Test suite for row conversion function."""

    def test_convert_train_row_basic(self):
        """Test converting a basic training row."""
        row = {
            'Id': '123',
            'Label': '1',
            'I1': '5',
            'I2': '10',
            'C1': 'abc',
            'C2': 'def'
        }

        vw_line = _convert_row_to_vw(row, is_train=True)

        # Should start with label (1) and tag ('123)
        assert vw_line.startswith("1 '123")

        # Should contain numerical namespace
        assert '|i' in vw_line

        # Should contain categorical namespace
        assert '|c' in vw_line

        # Should contain feature values
        assert 'I1:5' in vw_line
        assert 'I2:10' in vw_line
        assert 'abc' in vw_line
        assert 'def' in vw_line

    def test_convert_train_row_negative_label(self):
        """Test converting row with negative label."""
        row = {
            'Id': '456',
            'Label': '0',
            'I1': '3',
            'C1': 'xyz'
        }

        vw_line = _convert_row_to_vw(row, is_train=True)

        # Label should be -1 for VW format
        assert vw_line.startswith("-1 '456")

    def test_convert_test_row(self):
        """Test converting a test row (no label)."""
        row = {
            'Id': '789',
            'I1': '7',
            'C1': 'test'
        }

        vw_line = _convert_row_to_vw(row, is_train=False)

        # Test rows get dummy label 1
        assert vw_line.startswith("1 '789")
        assert 'I1:7' in vw_line
        assert 'test' in vw_line

    def test_convert_row_with_missing_values(self):
        """Test converting row with missing values."""
        row = {
            'Id': '111',
            'Label': '1',
            'I1': '5',
            'I2': '',  # Missing
            'C1': 'abc',
            'C2': ''   # Missing
        }

        vw_line = _convert_row_to_vw(row, is_train=True)

        # Should include non-empty features
        assert 'I1:5' in vw_line
        assert 'abc' in vw_line

        # Should not include empty features
        assert 'I2:' not in vw_line

    def test_convert_row_with_spaces(self):
        """Test converting row with whitespace."""
        row = {
            'Id': '222',
            'Label': '0',
            'I1': '  5  ',  # With spaces
            'C1': '  abc  '
        }

        vw_line = _convert_row_to_vw(row, is_train=True)

        # Spaces should be handled (features present)
        assert '|i' in vw_line
        assert '|c' in vw_line

    def test_convert_row_empty_namespaces(self):
        """Test converting row with all values empty in a namespace."""
        row = {
            'Id': '333',
            'Label': '1',
            'I1': '',
            'I2': '',
            'C1': 'abc'
        }

        vw_line = _convert_row_to_vw(row, is_train=True)

        # Should still have namespace markers
        assert '|i' in vw_line
        assert '|c' in vw_line


class TestCSVToVWFunction:
    """Test suite for full CSV to VW conversion."""

    def test_csv_to_vw_file_not_found(self, tmp_path):
        """Test with non-existent input file."""
        csv_path = tmp_path / "nonexistent.csv"
        output_path = tmp_path / "output.vw"

        with pytest.raises(FileNotFoundError):
            csv_to_vw(csv_path, output_path)

    def test_csv_to_vw_basic(self, temp_csv_file, tmp_path):
        """Test basic CSV to VW conversion."""
        output_path = tmp_path / "output.vw"

        # Should not raise error
        csv_to_vw(temp_csv_file, output_path, is_train=True)

        # Output file should exist
        assert output_path.exists()

        # Check output content
        with open(output_path, 'r') as f:
            lines = f.readlines()

        # Should have same number of lines as CSV (minus header)
        assert len(lines) == 3  # 3 data rows from fixture

        # Each line should be VW format
        for line in lines:
            assert '|i' in line
            assert '|c' in line

    def test_csv_to_vw_test_mode(self, temp_csv_file, tmp_path):
        """Test CSV to VW conversion in test mode."""
        output_path = tmp_path / "output.vw"

        csv_to_vw(temp_csv_file, output_path, is_train=False)

        # Output file should exist
        assert output_path.exists()

        with open(output_path, 'r') as f:
            lines = f.readlines()

        # In test mode, all labels should be 1
        for line in lines:
            assert line.startswith('1 ')

    def test_csv_to_vw_creates_output_file(self, temp_csv_file, tmp_path):
        """Test that output file is created correctly."""
        output_path = tmp_path / "nested" / "dir" / "output.vw"

        # Parent directories don't exist
        assert not output_path.parent.exists()

        # This should fail since we don't create parent dirs
        # (could be enhanced to create them)
        with pytest.raises(FileNotFoundError):
            csv_to_vw(temp_csv_file, output_path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_csv(self, tmp_path):
        """Test handling of malformed CSV."""
        csv_file = tmp_path / "malformed.csv"

        # Create malformed CSV
        with open(csv_file, 'w') as f:
            f.write("Id,Label,I1\n")
            f.write("1,0,5\n")
            f.write("2,1\n")  # Missing column

        output_file = tmp_path / "output.vw"

        # Should handle gracefully
        csv_to_vw(csv_file, output_file, is_train=True)

        # Output should still be created
        assert output_file.exists()

    def test_empty_csv(self, tmp_path):
        """Test handling of empty CSV."""
        csv_file = tmp_path / "empty.csv"

        # Create empty CSV (header only)
        with open(csv_file, 'w') as f:
            f.write("Id,Label,I1,C1\n")

        output_file = tmp_path / "output.vw"

        # Should not raise error
        csv_to_vw(csv_file, output_file, is_train=True)

        # Output should be empty
        with open(output_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 0
