

# Credit Fraud Detection

This project aims to build a machine learning model to detect fraudulent credit card transactions using various algorithms and techniques. The dataset includes transactions made in Europe in September 2013, with a high class imbalance (only 0.172% of the transactions are fraudulent). This project explores data preprocessing, model training, and performance evaluation on imbalanced data.

## Project Overview

In this project, we:
1. **Preprocessed the Data**:
   - Scaled features using a Robust Scaler.
   - Split the dataset into an 80/20 train-test ratio.
   - Handled class imbalance using oversampling and undersampling techniques.

2. **Trained Multiple Models**:
   - Logistic Regression
   - XGBoost
   - Artificial Neural Network (ANN)

3. **Evaluated Model Performance**:
   - Tested on both imbalanced and balanced datasets.
   - Used metrics such as precision, recall, F1-score, and AUC-ROC to measure model effectiveness.

## Requirements

- Python 3.8 or higher
- Jupyter Notebook
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow` (for ANN), `matplotlib`, `seaborn`

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yasminefathi/credit-fraud-detection-.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Credit_Fraud_Detection.ipynb
   ```

## Project Structure

- `Credit_Fraud_Detection.ipynb`: Main notebook containing code for preprocessing, model training, and evaluation.

## Results

The project successfully highlights the challenges and solutions for detecting fraud in highly imbalanced datasets. Models were fine-tuned and assessed for their predictive power, and the ANN performed the best on balanced data, achieving high accuracy in identifying fraudulent transactions.

