"""
Tests for Logistic Regression model (py_lh4_modernized.py).
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py_lh4_modernized import CTRPredictor


class TestCTRPredictor:
    """Test suite for CTRPredictor class."""

    def test_initialization(self):
        """Test model initialization."""
        model = CTRPredictor(dimension=1000, learning_rate=0.1)

        assert model.D == 1000
        assert model.alpha == 0.1
        assert len(model.w) == 1000
        assert len(model.n) == 1000
        assert all(w == 0.0 for w in model.w)
        assert all(n == 0.0 for n in model.n)

    def test_logloss_positive_label(self):
        """Test logloss calculation for positive label."""
        loss = CTRPredictor.logloss(0.8, 1.0)
        expected = -np.log(0.8)
        assert np.isclose(loss, expected)

    def test_logloss_negative_label(self):
        """Test logloss calculation for negative label."""
        loss = CTRPredictor.logloss(0.2, 0.0)
        expected = -np.log(0.8)
        assert np.isclose(loss, expected)

    def test_logloss_boundary_values(self):
        """Test logloss with boundary values (close to 0 or 1)."""
        # Should not raise error or return inf
        loss1 = CTRPredictor.logloss(0.99999, 1.0)
        loss2 = CTRPredictor.logloss(0.00001, 0.0)

        assert not np.isinf(loss1)
        assert not np.isinf(loss2)
        assert loss1 > 0
        assert loss2 > 0

    def test_get_features_basic(self):
        """Test feature hashing."""
        model = CTRPredictor(dimension=1000)

        csv_row = {'I1': '5', 'I2': '10', 'C1': 'abc'}
        features = model.get_features(csv_row)

        # Should include bias term (index 0)
        assert 0 in features
        assert len(features) > 1
        # All indices should be within dimension
        assert all(0 <= idx < model.D for idx in features)

    def test_get_features_empty_values(self):
        """Test feature hashing with empty values."""
        model = CTRPredictor(dimension=1000)

        csv_row = {'I1': '5', 'I2': '', 'C1': 'abc'}
        features = model.get_features(csv_row)

        # Should only have bias and non-empty features
        assert 0 in features
        assert len(features) >= 1

    def test_predict_probability_range(self):
        """Test that predictions are in valid probability range."""
        model = CTRPredictor(dimension=100)

        # Random features
        features = [0, 5, 10, 25]

        prob = model.predict_probability(features)

        # Should be between 0 and 1
        assert 0.0 <= prob <= 1.0

    def test_predict_probability_with_weights(self):
        """Test prediction with non-zero weights."""
        model = CTRPredictor(dimension=100)

        # Set some weights
        model.w[0] = 1.0
        model.w[5] = 2.0
        model.w[10] = -1.5

        features = [0, 5, 10]
        prob = model.predict_probability(features)

        # Expected: sigmoid(1.0 + 2.0 - 1.5) = sigmoid(1.5)
        expected = 1.0 / (1.0 + np.exp(-1.5))

        assert np.isclose(prob, expected)

    def test_update_weights(self):
        """Test weight update."""
        model = CTRPredictor(dimension=100, learning_rate=0.1)

        features = [0, 5, 10]
        p = 0.7  # Prediction
        y = 1.0  # True label

        # Store initial weights
        initial_weights = [model.w[i] for i in features]

        # Update weights
        model.update_weights(features, p, y)

        # Weights should have changed
        for i, idx in enumerate(features):
            assert model.w[idx] != initial_weights[i]

        # Counters should have incremented
        for idx in features:
            assert model.n[idx] == 1.0

    def test_train_file_not_found(self):
        """Test training with non-existent file."""
        model = CTRPredictor(dimension=100)

        with pytest.raises(FileNotFoundError):
            model.train(Path("nonexistent_file.csv"))

    def test_predict_file_not_found(self):
        """Test prediction with non-existent file."""
        model = CTRPredictor(dimension=100)

        with pytest.raises(FileNotFoundError):
            model.predict(
                Path("nonexistent_test.csv"),
                Path("output.csv")
            )

    def test_train_with_temp_file(self, temp_csv_file):
        """Test training with a small temporary file."""
        model = CTRPredictor(dimension=100, learning_rate=0.1)

        # Should not raise error
        model.train(temp_csv_file)

        # Weights should have been updated
        assert any(w != 0.0 for w in model.w)

    def test_numerical_stability(self):
        """Test that extreme values don't cause overflow."""
        model = CTRPredictor(dimension=100)

        # Set extreme weights
        model.w[5] = 1000.0
        model.w[10] = -1000.0

        features = [5, 10]
        prob = model.predict_probability(features)

        # Should not be nan or inf
        assert not np.isnan(prob)
        assert not np.isinf(prob)
        assert 0.0 <= prob <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dimension(self):
        """Test with zero dimension (should work but be useless)."""
        with pytest.raises(Exception):
            model = CTRPredictor(dimension=0)

    def test_negative_learning_rate(self):
        """Test with negative learning rate."""
        # Should still initialize (may behave oddly in training)
        model = CTRPredictor(dimension=100, learning_rate=-0.1)
        assert model.alpha == -0.1

    def test_very_large_dimension(self):
        """Test memory handling with large dimension."""
        # This might be slow or fail on memory-constrained systems
        # Using smaller value for testing
        try:
            model = CTRPredictor(dimension=10_000_000)
            assert len(model.w) == 10_000_000
        except MemoryError:
            pytest.skip("Insufficient memory for this test")
