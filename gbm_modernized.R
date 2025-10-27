# Gradient Boosting Machine (GBM) for CTR Prediction
# Modernized version with configurable paths and better structure

# ============================================================================
# Configuration
# ============================================================================

# Use environment variable or default to current directory
DATA_DIR <- Sys.getenv("DATA_DIR", default = "./data")
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR", default = "./output")

# File paths (relative to DATA_DIR)
TRAIN_FILE <- file.path(DATA_DIR, "train_num_na_yesno.csv")
TEST_FILE <- file.path(DATA_DIR, "test_num_impute.csv")
OUTPUT_FILE <- file.path(OUTPUT_DIR, "submit_gbm_num_nona_impute.csv")

# Model hyperparameters
N_TREES <- 500
INTERACTION_DEPTH <- 22
SHRINKAGE <- 0.1
N_CV_FOLDS <- 2
SEED <- 888

# ============================================================================
# Setup
# ============================================================================

cat("==========================================================\n")
cat("GBM CTR Prediction Model\n")
cat("==========================================================\n")
cat(sprintf("Data directory: %s\n", DATA_DIR))
cat(sprintf("Output directory: %s\n", OUTPUT_DIR))
cat(sprintf("Training file: %s\n", TRAIN_FILE))
cat(sprintf("Test file: %s\n", TEST_FILE))
cat("==========================================================\n\n")

# Load required libraries
required_packages <- c("data.table", "caret", "gbm")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing package: %s\n", pkg))
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
  cat(sprintf("Created output directory: %s\n", OUTPUT_DIR))
}

# ============================================================================
# Data Loading and Preparation
# ============================================================================

cat("\n[1/5] Loading training data...\n")

if (!file.exists(TRAIN_FILE)) {
  stop(sprintf("Training file not found: %s\nPlease prepare the data first.", TRAIN_FILE))
}

train <- read.csv(TRAIN_FILE, stringsAsFactors = TRUE)

# Remove ID column if present
if ("Id" %in% colnames(train) || "id" %in% colnames(train)) {
  train <- train[, !(colnames(train) %in% c("Id", "id"))]
}

cat(sprintf("  Loaded %d samples with %d features\n", nrow(train), ncol(train) - 1))
cat(sprintf("  Target variable: Label\n"))
cat(sprintf("  Class distribution:\n"))
print(table(train$Label))

# ============================================================================
# Model Training
# ============================================================================

cat("\n[2/5] Training GBM model...\n")
cat(sprintf("  Trees: %d\n", N_TREES))
cat(sprintf("  Interaction depth: %d\n", INTERACTION_DEPTH))
cat(sprintf("  Shrinkage: %.3f\n", SHRINKAGE))
cat(sprintf("  CV folds: %d\n", N_CV_FOLDS))

# Set seed for reproducibility
set.seed(SEED)

# Define training grid
gbm_grid <- expand.grid(
  n.trees = N_TREES,
  interaction.depth = INTERACTION_DEPTH,
  shrinkage = SHRINKAGE,
  n.minobsinnode = 10
)

# Define training control
fit_control <- trainControl(
  method = "cv",
  number = N_CV_FOLDS,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# Train model
start_time <- Sys.time()

gbm_fit <- train(
  Label ~ .,
  data = train,
  method = "gbm",
  trControl = fit_control,
  tuneGrid = gbm_grid,
  metric = "ROC",
  verbose = TRUE
)

end_time <- Sys.time()
training_time <- difftime(end_time, start_time, units = "mins")

cat(sprintf("\nTraining completed in %.2f minutes\n", training_time))

# ============================================================================
# Model Evaluation
# ============================================================================

cat("\n[3/5] Evaluating model on training data...\n")

train_pred <- predict(gbm_fit, train)
conf_matrix <- confusionMatrix(train_pred, train$Label)

print(conf_matrix)

# ============================================================================
# Load Test Data and Generate Predictions
# ============================================================================

cat("\n[4/5] Loading test data and generating predictions...\n")

if (!file.exists(TEST_FILE)) {
  warning(sprintf("Test file not found: %s\nSkipping predictions.", TEST_FILE))
} else {
  test <- read.csv(TEST_FILE, stringsAsFactors = TRUE)

  # Keep ID column for submission
  test_id <- test$Id

  cat(sprintf("  Loaded %d test samples\n", nrow(test)))

  # Generate predictions (probabilities)
  test_pred_prob <- predict(gbm_fit, test, type = "prob")

  # Create submission dataframe
  submission <- data.frame(
    Id = test_id,
    Predicted = test_pred_prob[, "Yes"]  # Probability of positive class
  )

  # ============================================================================
  # Save Predictions
  # ============================================================================

  cat("\n[5/5] Saving predictions...\n")

  write.csv(
    submission,
    OUTPUT_FILE,
    row.names = FALSE,
    quote = FALSE
  )

  cat(sprintf("  Predictions saved to: %s\n", OUTPUT_FILE))
  cat(sprintf("  Submission format: %d rows, 2 columns (Id, Predicted)\n", nrow(submission)))
}

# ============================================================================
# Summary
# ============================================================================

cat("\n==========================================================\n")
cat("Model Training Summary\n")
cat("==========================================================\n")
cat(sprintf("Training samples: %d\n", nrow(train)))
cat(sprintf("Features: %d\n", ncol(train) - 1))
cat(sprintf("Training time: %.2f minutes\n", training_time))
cat(sprintf("Training accuracy: %.4f\n", conf_matrix$overall['Accuracy']))
cat(sprintf("Model saved: gbm_fit\n"))
cat("==========================================================\n")
cat("\nDone!\n")

# ============================================================================
# Optional: Save model object
# ============================================================================

# Uncomment to save the trained model for later use
# saveRDS(gbm_fit, file.path(OUTPUT_DIR, "gbm_model.rds"))
# cat(sprintf("Model object saved to: %s\n", file.path(OUTPUT_DIR, "gbm_model.rds")))

# To load the model later:
# gbm_fit <- readRDS(file.path(OUTPUT_DIR, "gbm_model.rds"))
