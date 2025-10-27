"""
Logistic Regression with Stochastic Gradient Descent (SGD) for CTR Prediction

This module implements a memory-efficient logistic regression model using:
- Hash trick for feature engineering (2^27 dimensional space)
- Adaptive learning rate for SGD optimization
- Bounded numerical operations for stability

Modernized from original py_lh4.py with:
- Python 3 compatibility
- Error handling and logging
- Input validation
- Configurable parameters
- Type hints
"""

import logging
import sys
from csv import DictReader
from math import exp, log, sqrt
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class CTRPredictor:
    """Click-Through Rate predictor using Logistic Regression with SGD."""

    def __init__(
        self,
        dimension: int = 2**27,
        learning_rate: float = 0.145,
        log_interval: int = 1_000_000
    ):
        """
        Initialize the CTR predictor.

        Args:
            dimension: Number of features for hash trick (default: 2^27 = ~134M)
            learning_rate: Alpha parameter for SGD (default: 0.145)
            log_interval: How often to log progress during training
        """
        self.D = dimension
        self.alpha = learning_rate
        self.log_interval = log_interval

        # Initialize model weights and feature counters
        self.w: List[float] = [0.0] * self.D
        self.n: List[float] = [0.0] * self.D

        logger.info("Initialized CTR Predictor:")
        logger.info(f"  Dimension: {self.D:,}")
        logger.info(f"  Learning rate: {self.alpha}")
        logger.info(f"  Memory usage: ~{(self.D * 16) / (1024**3):.2f} GB")

    @staticmethod
    def logloss(p: float, y: float) -> float:
        """
        Calculate bounded logarithmic loss.

        Args:
            p: Predicted probability (0 to 1)
            y: True label (0 or 1)

        Returns:
            Logarithmic loss value
        """
        # Bound prediction to prevent log(0)
        epsilon = 1e-12
        p = max(min(p, 1.0 - epsilon), epsilon)

        if y == 1.0:
            return -log(p)
        else:
            return -log(1.0 - p)

    def get_features(self, csv_row: Dict[str, str]) -> List[int]:
        """
        Apply hash trick to convert CSV row to feature indices.

        Treats both integer and categorical features as categorical,
        using a simple hash function to map to feature space.

        Args:
            csv_row: Dictionary from CSV DictReader

        Returns:
            List of feature indices where value is 1
        """
        x = [0]  # Index 0 is the bias term

        for key, value in csv_row.items():
            if not value:  # Skip empty values
                continue

            try:
                # Simple hash: concatenate value and feature name, convert to hex
                # This is intentionally simple for speed (though weak as a hash)
                hash_input = value + key[1:]
                index = int(hash_input, 16) % self.D
                x.append(index)
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid feature {key}={value}: {e}")
                continue

        return x

    def predict_probability(self, x: List[int]) -> float:
        """
        Calculate probability P(y=1|x) using logistic sigmoid.

        Args:
            x: List of feature indices

        Returns:
            Predicted probability between 0 and 1
        """
        # Calculate w^T * x
        wTx = sum(self.w[i] for i in x)

        # Apply bounded sigmoid to prevent overflow
        # Bound to [-20, 20] before exp
        wTx_bounded = max(min(wTx, 20.0), -20.0)

        return 1.0 / (1.0 + exp(-wTx_bounded))

    def update_weights(
        self,
        x: List[int],
        p: float,
        y: float
    ) -> None:
        """
        Update model weights using SGD with adaptive learning rate.

        Args:
            x: Feature indices
            p: Predicted probability
            y: True label (0 or 1)
        """
        for i in x:
            # Adaptive learning rate: alpha / (sqrt(n) + 1)
            # This decreases learning rate for frequently seen features
            adaptive_lr = self.alpha / (sqrt(self.n[i]) + 1.0)

            # Gradient: (p - y) * x[i], where x[i] = 1 for indices in x
            gradient = p - y

            # Update weight
            self.w[i] -= gradient * adaptive_lr

            # Increment feature counter
            self.n[i] += 1.0

    def train(self, train_path: Path) -> None:
        """
        Train the model on training data using online SGD.

        Args:
            train_path: Path to training CSV file

        Raises:
            FileNotFoundError: If training file doesn't exist
            ValueError: If file format is invalid
        """
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")

        logger.info(f"Starting training from: {train_path}")
        logger.info("=" * 80)

        cumulative_loss = 0.0
        sample_count = 0

        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                reader = DictReader(f)

                # Validate required columns
                if reader.fieldnames and 'Label' not in reader.fieldnames:
                    raise ValueError("Training file must contain 'Label' column")

                for t, row in enumerate(reader, start=1):
                    # Parse label
                    try:
                        y = 1.0 if row['Label'] == '1' else 0.0
                    except KeyError:
                        logger.error(f"Row {t} missing 'Label' column")
                        continue

                    # Remove label and ID from features
                    row.pop('Label', None)
                    row.pop('Id', None)

                    # Get hashed features
                    x = self.get_features(row)

                    # Get prediction
                    p = self.predict_probability(x)

                    # Calculate loss for monitoring
                    loss = self.logloss(p, y)
                    cumulative_loss += loss
                    sample_count = t

                    # Log progress
                    if t % self.log_interval == 0:
                        avg_loss = cumulative_loss / t
                        logger.info(
                            f"Processed: {t:,} samples | "
                            f"Avg logloss: {avg_loss:.6f} | "
                            f"Current loss: {loss:.6f}"
                        )

                    # Update model
                    self.update_weights(x, p, y)

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info(f"  Total samples: {sample_count:,}")
        logger.info(f"  Final avg logloss: {cumulative_loss / sample_count:.6f}")

    def predict(self, test_path: Path, output_path: Path) -> None:
        """
        Generate predictions for test data and save to CSV.

        Args:
            test_path: Path to test CSV file
            output_path: Path to save predictions

        Raises:
            FileNotFoundError: If test file doesn't exist
        """
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        logger.info(f"Generating predictions from: {test_path}")
        logger.info(f"Saving to: {output_path}")

        prediction_count = 0

        try:
            with open(test_path, 'r', encoding='utf-8') as f_in, \
                 open(output_path, 'w', encoding='utf-8') as f_out:

                # Write header
                f_out.write('Id,Predicted\n')

                reader = DictReader(f_in)

                for t, row in enumerate(reader, start=1):
                    # Get ID
                    row_id = row.get('Id', str(t))
                    row.pop('Id', None)

                    # Get features and predict
                    x = self.get_features(row)
                    p = self.predict_probability(x)

                    # Write prediction
                    f_out.write(f'{row_id},{p:.10f}\n')
                    prediction_count = t

                    # Log progress
                    if t % self.log_interval == 0:
                        logger.info(f"Generated {t:,} predictions")

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

        logger.info(f"Prediction completed! Generated {prediction_count:,} predictions")


def main():
    """Main execution function."""
    # Configuration
    TRAIN_FILE = Path('train.csv')
    TEST_FILE = Path('test.csv')
    OUTPUT_FILE = Path('submission.csv')

    # Model hyperparameters
    DIMENSION = 2 ** 27  # ~134M features
    LEARNING_RATE = 0.145

    try:
        # Initialize model
        model = CTRPredictor(
            dimension=DIMENSION,
            learning_rate=LEARNING_RATE
        )

        # Train model
        model.train(TRAIN_FILE)

        # Generate predictions
        model.predict(TEST_FILE, OUTPUT_FILE)

        logger.info("All tasks completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        logger.error("Please ensure train.csv and test.csv are in the current directory")
        logger.error("Download from: https://www.kaggle.com/c/criteo-display-ad-challenge/data")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
