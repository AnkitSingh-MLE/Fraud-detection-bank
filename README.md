# Fraud detection bank


## Project Structure

- `Fraud_detection_bank.py`: Main script for data loading, exploration, feature engineering, model training, and evaluation.
- `requirements.txt`: List of required Python packages.

## Data Exploration

The script performs the following data exploration steps:
- Displays the first few rows and basic information about the dataset.
- Checks for missing values and duplicates.
- Explores categorical features and visualizes the distributions of numerical features.

## Feature Engineering

The script creates initial fraud labels based on the following rules:
- High transaction amount (top 1% of all transactions).
- High number of login attempts (greater than 5).
- Frequent location changes for the same account.
- Fast consecutive transactions (time difference < 10 minutes for the same account).

## Model Training and Evaluation

The script uses a RandomForestClassifier to train a model on the dataset. It performs the following steps:
- Splits the data into training and testing sets.
- Trains the model on the training set.
- Evaluates the model on the testing set using accuracy, precision, recall, F1-score, and confusion matrix.

## Results

The script prints the evaluation metrics and displays the confusion matrix and classification report.

