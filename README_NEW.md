# Predict Click-Through Rates on Display Ads

A machine learning project for predicting Click-Through Rates (CTR) in display advertising using the Criteo dataset. This repository implements multiple algorithms including Logistic Regression with SGD, Gradient Boosting Machines, and Vowpal Wabbit models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-blue.svg)

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Models](#models)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Display advertising is a billion-dollar industry and one of the central applications of machine learning on the Internet. This project was developed for the **Criteo Display Advertising Challenge**, where the goal is to predict the probability that a user will click on a given ad (CTR).

### The Challenge

Given:
- User information
- Page context
- Ad features (39 anonymized features: 13 numerical, 26 categorical)

Predict:
- Probability of click (binary classification)

### Evaluation Metric

- **Log Loss (Logarithmic Loss)**: Lower is better

---

## Project Status

**Version:** 2.0 (Modernized)

**Status:**
- ✅ Python 3 migration complete
- ✅ Core algorithms refactored with modern best practices
- ✅ Documentation updated
- ✅ Error handling and logging added
- 🚧 Unit tests in progress
- 🚧 CI/CD pipeline in progress

**Legacy Code:**
- Original Python 2 implementations preserved with `_legacy` suffix
- See migration guide below for differences

---

## Features

### Algorithms Implemented

1. **Logistic Regression with SGD** (`py_lh4_modernized.py`)
   - Hash trick for feature engineering (2^27 dimensional space)
   - Adaptive learning rate
   - Memory-efficient online learning
   - Bounded numerical operations for stability

2. **Gradient Boosting Machine** (`gbm_modernized.R`)
   - Tree-based ensemble method
   - Cross-validation with ROC optimization
   - Configurable hyperparameters

3. **Vowpal Wabbit** (`csv_to_vw_modernized.py`)
   - Fast linear learner
   - CSV to VW format converter
   - Scalable to massive datasets

### Modern Features

- ✨ Python 3.8+ compatibility
- 🔒 Input validation and error handling
- 📊 Comprehensive logging
- ⚙️ Configurable parameters
- 📝 Type hints and documentation
- 🧪 Unit test infrastructure
- 🐳 Docker support (coming soon)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- R 4.0 or higher (for R models)
- Vowpal Wabbit (for VW models)
- Git

### Clone Repository

```bash
git clone https://github.com/yourusername/Predict-click-through-rates-on-display-ads.git
cd Predict-click-through-rates-on-display-ads
```

### Python Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### R Setup

```bash
# Install R packages
Rscript -e "install.packages(c('data.table', 'caret', 'gbm'), dependencies=TRUE)"
```

### Vowpal Wabbit Setup

```bash
# macOS
brew install vowpal-wabbit

# Ubuntu/Debian
sudo apt-get install vowpal-wabbit

# From source (all platforms)
git clone https://github.com/VowpalWabbit/vowpal_wabbit.git
cd vowpal_wabbit
make
```

---

## Quick Start

### 1. Download Data

```bash
# Create data directory
mkdir -p data

# Download from Kaggle
# Visit: https://www.kaggle.com/c/criteo-display-ad-challenge/data
# Place train.csv and test.csv in data/ directory
```

Or use the download script:

```bash
python scripts/download_data.py
```

### 2. Train a Model

**Logistic Regression (Python):**

```bash
python py_lh4_modernized.py
```

**Gradient Boosting Machine (R):**

```bash
DATA_DIR=./data OUTPUT_DIR=./output Rscript gbm_modernized.R
```

**Vowpal Wabbit:**

```bash
# Convert CSV to VW format
python csv_to_vw_modernized.py data/train.csv data/train.vw
python csv_to_vw_modernized.py data/test.csv data/test.vw --test

# Train model
vw data/train.vw -f models/click.model --passes 3 --cache_file data/train.cache

# Generate predictions
vw data/test.vw -i models/click.model -t -p predictions.txt
```

### 3. Generate Submission

```bash
# Predictions are automatically saved to submission.csv (Python)
# or specified output directory (R)
```

---

## Data

### Dataset Information

- **Source:** [Criteo Labs](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
- **Size:**
  - Training: ~45 million samples
  - Test: ~6 million samples
- **Features:** 39 anonymized features
  - 13 numerical features (`I1-I13`)
  - 26 categorical features (`C1-C26`)
- **Target:** Binary (0 = no click, 1 = click)

### Data Format

**CSV Format:**

```
Id,Label,I1,I2,...,I13,C1,C2,...,C26
1,0,1,5,...,45,68fd1e,80e26c,...,a458ea
2,1,2,,,7,68fd1e,80e26c,...,b458ea
```

**Vowpal Wabbit Format:**

```
1 'id1 |i I1:1 I2:5 I13:45 |c 68fd1e 80e26c a458ea
-1 'id2 |i I1:2 I13:7 |c 68fd1e 80e26c b458ea
```

### Data Preprocessing

Missing values are common in this dataset:

```python
# Option 1: Use models that handle missing values (GBM, Random Forest)
# Option 2: Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
```

---

## Models

### 1. Logistic Regression with SGD

**File:** `py_lh4_modernized.py`

**Features:**
- Hash trick for feature engineering
- Adaptive learning rate: α/(√n + 1)
- Bounded sigmoid and log loss for numerical stability
- Memory-efficient: processes one sample at a time

**Hyperparameters:**
```python
DIMENSION = 2**27      # ~134M features
LEARNING_RATE = 0.145  # Alpha for SGD
```

**Usage:**
```python
from py_lh4_modernized import CTRPredictor

model = CTRPredictor(dimension=2**27, learning_rate=0.145)
model.train(Path('data/train.csv'))
model.predict(Path('data/test.csv'), Path('submission.csv'))
```

**Expected Performance:**
- Training time: 30-60 minutes (45M samples)
- Memory: ~2GB
- Log loss: ~0.44-0.46

### 2. Gradient Boosting Machine (GBM)

**File:** `gbm_modernized.R`

**Features:**
- Tree-based ensemble method
- Cross-validation with ROC optimization
- Handles missing values automatically

**Hyperparameters:**
```r
N_TREES <- 500
INTERACTION_DEPTH <- 22
SHRINKAGE <- 0.1
```

**Usage:**
```bash
DATA_DIR=./data OUTPUT_DIR=./output Rscript gbm_modernized.R
```

**Expected Performance:**
- Training time: 2-4 hours
- Memory: ~8-16GB
- Log loss: ~0.43-0.45

### 3. Vowpal Wabbit

**Files:** `csv_to_vw_modernized.py`, VW commands

**Features:**
- Extremely fast linear learner
- Scalable to billions of samples
- Online learning capable

**Usage:**
```bash
# Convert format
python csv_to_vw_modernized.py data/train.csv data/train.vw

# Train with multiple passes
vw data/train.vw \
   --loss_function logistic \
   --passes 3 \
   --cache_file data/train.cache \
   -f models/click.model

# Predict
vw data/test.vw \
   -i models/click.model \
   -t \
   -p predictions.txt
```

**Expected Performance:**
- Training time: 10-20 minutes
- Memory: ~2-4GB
- Log loss: ~0.44-0.46

---

## Project Structure

```
Predict-click-through-rates-on-display-ads/
│
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── requirements-r.txt             # R dependencies
├── .gitignore                     # Git ignore rules
│
├── data/                          # Data directory (not in repo)
│   ├── train.csv                  # Training data (download separately)
│   ├── test.csv                   # Test data (download separately)
│   ├── train.vw                   # VW format training data
│   └── test.vw                    # VW format test data
│
├── models/                        # Trained models
│   ├── click.model.vw             # Vowpal Wabbit model
│   └── gbm_model.rds              # R GBM model
│
├── output/                        # Output directory
│   ├── submission.csv             # Predictions for submission
│   └── training.log               # Training logs
│
├── Modern Implementations:
│   ├── py_lh4_modernized.py       # LR with SGD (Python 3)
│   ├── logReg_modernized.py       # LR module (Python 3)
│   ├── csv_to_vw_modernized.py    # CSV to VW converter (Python 3)
│   └── gbm_modernized.R           # GBM (R, modern)
│
├── Legacy Code (Original):
│   ├── py_lh4.py                  # Original LR implementation
│   ├── logReg.py                  # Original LR module
│   ├── gbm.R                      # Original GBM script
│   └── [other legacy files]
│
├── tests/                         # Unit tests
│   ├── test_lr_model.py
│   ├── test_data_loading.py
│   └── test_vw_converter.py
│
├── scripts/                       # Utility scripts
│   ├── download_data.py           # Data download helper
│   └── evaluate_model.py          # Model evaluation
│
└── docs/                          # Additional documentation
    ├── MIGRATION_GUIDE.md         # Python 2 to 3 migration notes
    ├── PERFORMANCE.md             # Performance benchmarks
    └── API.md                     # API documentation
```

---

## Usage

### Configuration

**Python Models:**

Edit configuration at the top of each script:

```python
# In py_lh4_modernized.py
TRAIN_FILE = Path('data/train.csv')
TEST_FILE = Path('data/test.csv')
OUTPUT_FILE = Path('submission.csv')
DIMENSION = 2 ** 27
LEARNING_RATE = 0.145
```

**R Models:**

Use environment variables:

```bash
export DATA_DIR=/path/to/data
export OUTPUT_DIR=/path/to/output
Rscript gbm_modernized.R
```

### Training Options

**Full Training:**
```bash
python py_lh4_modernized.py
```

**Sample Training (first 1M rows):**
```bash
head -n 1000001 data/train.csv > data/train_sample.csv
# Update script to use train_sample.csv
python py_lh4_modernized.py
```

### Monitoring Training

Training progress is logged to both console and `training.log`:

```bash
# Watch log in real-time
tail -f training.log
```

### Model Evaluation

```bash
python scripts/evaluate_model.py \
    --predictions submission.csv \
    --ground_truth data/train_with_labels.csv
```

---

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (coming soon)
pre-commit install

# Run code formatters
black .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_lr_model.py
```

### Code Style

This project follows:
- **Python:** PEP 8, enforced with `black` and `flake8`
- **R:** Tidyverse style guide
- **Documentation:** Google-style docstrings

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Deep learning models (Neural Networks, LSTM)
- [ ] Feature engineering improvements
- [ ] Hyperparameter optimization
- [ ] Ensemble methods
- [ ] Docker containerization
- [ ] Web API for predictions
- [ ] Performance optimizations

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2019-2025 Tianxiang Liu

---

## Acknowledgments

### Dataset
- **Criteo Labs** for providing the dataset
- Kaggle for hosting the competition

### References
- Original Vowpal Wabbit converter by Triskelion and Zygmunt Zając
- Criteo Display Advertising Challenge: https://www.kaggle.com/c/criteo-display-ad-challenge

### Papers
- Rendle, S. (2010). "Factorization Machines"
- McMahan, H. B. et al. (2013). "Ad Click Prediction: a View from the Trenches"
- He, X. et al. (2014). "Practical Lessons from Predicting Clicks on Ads at Facebook"

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## Changelog

### Version 2.0 (2025)
- ✨ Migrated to Python 3.8+
- ✨ Added comprehensive error handling and logging
- ✨ Refactored code with type hints and documentation
- ✨ Added configuration management
- ✨ Improved README with detailed instructions
- ✨ Added unit test infrastructure

### Version 1.0 (2014)
- Initial implementation
- Python 2 codebase
- Multiple algorithm implementations

---

**Happy Machine Learning! 🚀**
