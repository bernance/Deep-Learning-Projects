# Bank Customer Churn Prediction — Handling Class Imbalance with ANNs

A deep learning project that explores how different class imbalance handling techniques affect a neural network’s ability to predict bank customer churn. The dataset has a natural ~4:1 imbalance (80% stayed, 20% churned), making this a realistic and common challenge in binary classification.

-----

## Problem Statement

Predicting customer churn is valuable for banks — retaining an existing customer is significantly cheaper than acquiring a new one. The challenge is that churned customers are the minority class, and a naive model can achieve ~80% accuracy by simply predicting everyone stays. The goal is to build a model that actually identifies churners correctly.

-----

## Dataset

**Churn Modelling dataset** — 10,000 bank customers.

|Feature        |Description                         |
|---------------|------------------------------------|
|CreditScore    |Customer credit score               |
|Geography      |Country (France, Germany, Spain)    |
|Gender         |Male / Female                       |
|Age            |Customer age                        |
|Tenure         |Years with the bank                 |
|Balance        |Account balance                     |
|NumOfProducts  |Number of bank products used        |
|HasCrCard      |Whether customer has a credit card  |
|IsActiveMember |Whether customer is active          |
|EstimatedSalary|Estimated annual salary             |
|**Exited**     |**Target — 1 = churned, 0 = stayed**|

**Class distribution:** 7,963 stayed (79.6%) vs 2,037 churned (20.4%)

Dataset source: [Kaggle — Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

-----

## Preprocessing

- Dropped non-informative columns: `RowNumber`, `CustomerId`, `Surname`
- Label encoded `Gender` (Female → 0, Male → 1)
- One-hot encoded `Geography` using `pd.get_dummies`
- Applied `MinMaxScaler` to high-range numerical features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`
- Scaler was fit on `X_train` only and used to transform `X_test`, avoiding data leakage

-----

## Model Architecture

The same ANN architecture was used across all experiments for a fair comparison:

```
Input (12 features)
→ Dense(30, relu)
→ Dense(20, relu)
→ Dense(10, relu)
→ Dense(1, sigmoid)
```

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 100
- Seed: `tf.random.set_seed(42)` for reproducibility

-----

## Imbalance Handling Techniques

### 1. Baseline — No resampling

Train directly on the imbalanced dataset. Establishes the performance ceiling for the majority class and a floor for the minority class. High accuracy but poor churn recall.

### 2. Undersampling

Randomly samples the majority class down to match the minority class size (2,037 each). Simple and fast but discards ~75% of available training data.

**Pipeline:** Resample full dataset → split → scale

### 3. Random Oversampling

Randomly duplicates minority class samples (with replacement) up to the majority class size (7,963 each). Preserves all data but test set includes duplicated rows.

**Pipeline:** Resample full dataset → split → scale

### 4. SMOTE

Generates synthetic minority samples by interpolating between existing minority examples in feature space. Applied only to the training set after splitting to prevent synthetic samples from leaking into evaluation.

**Pipeline:** Split → scale → SMOTE on `X_train` only

### 5. Ensemble Batching

Splits the majority class into 4 batches, each paired with the full minority class. Trains a separate ANN on each batch, then combines predictions using majority voting. Uses all majority class data without any resampling.

**Pipeline:** Split → scale → batch training → majority vote

-----

## Key Takeaways

- **Accuracy is misleading on imbalanced data** — the baseline achieves ~86% accuracy but barely identifies churners.
- **Undersampling** improved minority class recall significantly but at the cost of overall accuracy, since a large portion of training data was discarded.
- **Oversampling** performed better than undersampling, with improved scores across both classes.
- **SMOTE** requires a stricter pipeline — resampling must happen after the train-test split and only on the training set. When implemented correctly, SMOTE’s test set reflects real-world class distribution (uneven support), which makes evaluation more honest and it performs slighlty better on the  minority class compared to the base model .
- **Ensemble batching** showed improved minority class recall compared to the baseline without requiring any synthetic data generation.
- **Recall on the minority class** is the most important metric in this context — missing a churner is more costly than a false alarm.

-----

## Project Structure

```
Bank Churn/
│
├── notebook/
│   └── Bank_Customer_Churn.ipynb   # Main notebook
├── datasets/
│   └── Churn_Modelling.csv         # Download from Kaggle (link above)
└── README.md                       # This file
```

-----

## Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
tensorflow
```

Install with:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow
```

-----

## How to Run

1. Clone the repo
1. Download the dataset from Kaggle and place it in a `datasets/` folder
1. Open `Bank_Customer_Churn.ipynb` in Jupyter or VS Code
1. Run all cells in order

-----

## Author

**Bernard Worthy Izuchukwu**  
[GitHub](https://github.com/bernance/Deep-Learning-Projects)
