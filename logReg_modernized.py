"""
Logistic Regression Training and Testing Module

Implements logistic regression with multiple optimization algorithms:
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Smooth Stochastic Gradient Descent (Smooth SGD)

Modernized from original logReg.py with:
- Python 3 compatibility
- Explicit imports (no wildcards)
- Type hints
- Error handling
- Better documentation
"""

import logging
import time
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Calculate the sigmoid (logistic) function.

    Args:
        x: Input array or scalar

    Returns:
        Sigmoid of input, bounded between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-x))


def train_logistic_regression(
    train_x: np.ndarray,
    train_y: np.ndarray,
    opts: Dict[str, Any]
) -> np.ndarray:
    """
    Train a logistic regression model using specified optimization algorithm.

    Args:
        train_x: Training features, shape (num_samples, num_features).
                 Should include bias term (column of ones) if needed.
        train_y: Training labels, shape (num_samples, 1)
        opts: Dictionary with training options:
            - 'alpha': Learning rate (float)
            - 'maxIter': Maximum iterations (int)
            - 'optimizeType': Optimization algorithm (str)
                Options: 'gradDescent', 'stocGradDescent', 'smoothStocGradDescent'

    Returns:
        weights: Trained weight vector, shape (num_features, 1)

    Raises:
        ValueError: If optimize type is not recognized
        TypeError: If input arrays have wrong shape
    """
    # Validate inputs
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: train_x has {train_x.shape[0]} samples, "
            f"train_y has {train_y.shape[0]} samples"
        )

    start_time = time.time()

    num_samples, num_features = train_x.shape
    alpha = opts.get('alpha', 0.01)
    max_iter = opts.get('maxIter', 1000)
    optimize_type = opts.get('optimizeType', 'gradDescent')

    logger.info(f"Training Logistic Regression:")
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  Features: {num_features}")
    logger.info(f"  Algorithm: {optimize_type}")
    logger.info(f"  Learning rate: {alpha}")
    logger.info(f"  Max iterations: {max_iter}")

    # Initialize weights
    weights = np.ones((num_features, 1))

    # Optimize using selected algorithm
    if optimize_type == 'gradDescent':
        weights = _gradient_descent(train_x, train_y, weights, alpha, max_iter)

    elif optimize_type == 'stocGradDescent':
        weights = _stochastic_gradient_descent(
            train_x, train_y, weights, alpha, max_iter
        )

    elif optimize_type == 'smoothStocGradDescent':
        weights = _smooth_stochastic_gradient_descent(
            train_x, train_y, weights, alpha, max_iter
        )

    else:
        raise ValueError(
            f"Unsupported optimize type: {optimize_type}. "
            f"Must be 'gradDescent', 'stocGradDescent', or 'smoothStocGradDescent'"
        )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")

    return weights


def _gradient_descent(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    max_iter: int
) -> np.ndarray:
    """
    Batch gradient descent optimization.

    Updates weights using all samples in each iteration.
    """
    for k in range(max_iter):
        # Forward pass
        output = sigmoid(train_x @ weights)

        # Calculate error
        error = train_y - output

        # Update weights using all samples
        weights = weights + alpha * (train_x.T @ error)

        # Log progress
        if (k + 1) % 100 == 0:
            loss = np.mean(-train_y * np.log(output + 1e-10) -
                          (1 - train_y) * np.log(1 - output + 1e-10))
            logger.info(f"Iteration {k+1}/{max_iter}, Loss: {loss:.6f}")

    return weights


def _stochastic_gradient_descent(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    max_iter: int
) -> np.ndarray:
    """
    Stochastic gradient descent optimization.

    Updates weights using one sample at a time.
    """
    num_samples = train_x.shape[0]

    for k in range(max_iter):
        for i in range(num_samples):
            # Get single sample
            x_i = train_x[i:i+1, :].T  # Shape: (num_features, 1)
            y_i = train_y[i, 0]

            # Forward pass
            output = sigmoid((x_i.T @ weights)[0, 0])

            # Calculate error
            error = y_i - output

            # Update weights
            weights = weights + alpha * x_i * error

        # Log progress
        if (k + 1) % 100 == 0:
            output_all = sigmoid(train_x @ weights)
            loss = np.mean(-train_y * np.log(output_all + 1e-10) -
                          (1 - train_y) * np.log(1 - output_all + 1e-10))
            logger.info(f"Iteration {k+1}/{max_iter}, Loss: {loss:.6f}")

    return weights


def _smooth_stochastic_gradient_descent(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    max_iter: int
) -> np.ndarray:
    """
    Smooth stochastic gradient descent optimization.

    Uses random sample selection and adaptive learning rate to reduce oscillations.
    """
    num_samples = train_x.shape[0]

    for k in range(max_iter):
        # Create random order of samples
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        for i, idx in enumerate(indices):
            # Adaptive learning rate that decreases over time
            adaptive_alpha = 4.0 / (1.0 + k + i) + 0.01

            # Get single sample
            x_i = train_x[idx:idx+1, :].T  # Shape: (num_features, 1)
            y_i = train_y[idx, 0]

            # Forward pass
            output = sigmoid((x_i.T @ weights)[0, 0])

            # Calculate error
            error = y_i - output

            # Update weights with adaptive learning rate
            weights = weights + adaptive_alpha * x_i * error

        # Log progress
        if (k + 1) % 100 == 0:
            output_all = sigmoid(train_x @ weights)
            loss = np.mean(-train_y * np.log(output_all + 1e-10) -
                          (1 - train_y) * np.log(1 - output_all + 1e-10))
            logger.info(f"Iteration {k+1}/{max_iter}, Loss: {loss:.6f}")

    return weights


def test_logistic_regression(
    weights: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray
) -> float:
    """
    Test trained logistic regression model and calculate accuracy.

    Args:
        weights: Trained weight vector, shape (num_features, 1)
        test_x: Test features, shape (num_samples, num_features)
        test_y: Test labels, shape (num_samples, 1)

    Returns:
        accuracy: Proportion of correct predictions (0 to 1)
    """
    num_samples = test_x.shape[0]
    match_count = 0

    for i in range(num_samples):
        # Get prediction probability
        x_i = test_x[i:i+1, :]
        prob = sigmoid((x_i @ weights)[0, 0])

        # Convert to binary prediction (threshold at 0.5)
        predict = prob > 0.5

        # Check if correct
        if predict == bool(test_y[i, 0]):
            match_count += 1

    accuracy = match_count / num_samples

    logger.info(f"Test Results:")
    logger.info(f"  Correct: {match_count}/{num_samples}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return accuracy


def visualize_logistic_regression(
    weights: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray
) -> None:
    """
    Visualize the trained logistic regression decision boundary.

    Note: Only works with 2D data (3 features including bias).

    Args:
        weights: Trained weight vector
        train_x: Training features including bias column
        train_y: Training labels

    Raises:
        ValueError: If data is not 2D
    """
    num_samples, num_features = train_x.shape

    if num_features != 3:
        raise ValueError(
            f"Visualization only supports 2D data (3 features with bias). "
            f"Got {num_features} features."
        )

    logger.info("Generating visualization...")

    # Plot samples
    for i in range(num_samples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or', label='Class 0' if i == 0 else '')
        else:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob', label='Class 1' if i == 0 else '')

    # Draw decision boundary
    # Line equation: w0 + w1*x1 + w2*x2 = 0
    # Solve for x2: x2 = -(w0 + w1*x1) / w2
    min_x = np.min(train_x[:, 1])
    max_x = np.max(train_x[:, 1])

    w = weights.flatten()
    y_min = -(w[0] + w[1] * min_x) / w[2]
    y_max = -(w[0] + w[1] * max_x) / w[2]

    plt.plot([min_x, max_x], [y_min, y_max], '-g', linewidth=2, label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example usage
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)

    # Create synthetic 2D dataset
    num_samples = 100

    # Class 0: centered at (2, 2)
    class0 = np.random.randn(num_samples // 2, 2) + np.array([2, 2])

    # Class 1: centered at (5, 5)
    class1 = np.random.randn(num_samples // 2, 2) + np.array([5, 5])

    # Combine data
    X = np.vstack([class0, class1])
    y = np.vstack([np.zeros((num_samples // 2, 1)), np.ones((num_samples // 2, 1))])

    # Add bias term
    X_with_bias = np.hstack([np.ones((num_samples, 1)), X])

    # Training options
    options = {
        'alpha': 0.01,
        'maxIter': 500,
        'optimizeType': 'smoothStocGradDescent'
    }

    # Train model
    trained_weights = train_logistic_regression(X_with_bias, y, options)

    # Test model
    accuracy = test_logistic_regression(trained_weights, X_with_bias, y)

    # Visualize (optional - uncomment to show plot)
    # visualize_logistic_regression(trained_weights, X_with_bias, y)
