# Deep Learning Projects

A collection of deep learning projects built with **TensorFlow** and **Keras**, covering foundational theory and applied problem-solving. Each project goes beyond surface-level accuracy reporting. The focus is on honest evaluation, handling real-world challenges like class imbalance, and understanding what's actually happening under the hood.

---

## Projects

### 🏦 Bank Customer Churn Prediction (ANN)
**`Bank Churn/`**

Predicts which bank customers are likely to churn using an Artificial Neural Network. The core challenge here is class imbalance. Churned customers are the minority, and a model that ignores that will look good on paper while failing in practice.

Five strategies are compared head-to-head:

| Strategy | Description |
|---|---|
| Baseline | No imbalance handling |
| Undersampling | Randomly reduce the majority class |
| Oversampling | Randomly duplicate the minority class |
| SMOTE | Synthesize new minority samples |
| Ensemble Batching | Balance within each training batch |

Each approach is evaluated on precision, recall, F1-score, and ROC-AUC  with special attention to how well the model catches actual churners, not just overall accuracy.

**Stack:** TensorFlow · Keras · scikit-learn · imbalanced-learn · pandas · matplotlib

---

### 🔢 Digit Recognition with CNN
**`Digits prediction with CNN.ipynb`**

Classifies handwritten digits from the MNIST dataset using a Convolutional Neural Network. Covers the full pipeline: data preprocessing, building and training a CNN from scratch, and evaluating performance with a confusion matrix.

**Stack:** TensorFlow · Keras · NumPy · matplotlib

---

### 👗 Fashion MNIST Classification with CNN
**`Fashion Mnist prediction using CNN.ipynb`**

Extends digit recognition to a harder problem, classifying 10 categories of clothing items (shirts, shoes, bags, etc.) from the Fashion-MNIST dataset. Explores how CNNs handle more visually complex classes and where they tend to struggle.

**Stack:** TensorFlow · Keras · NumPy · matplotlib

---

### 🧠 Neural Network from Scratch
**`Implementing a neural network from scratch.ipynb`**

Builds a fully connected neural network using only NumPy(no Keras, no TensorFlow). Implements forward propagation, backpropagation, and gradient descent by hand. The goal is to build intuition for what deep learning frameworks are actually doing behind the scenes.

**Stack:** NumPy · Python

---

## Setup

```bash
git clone https://github.com/bernance/Deep-Learning-Projects.git
cd Deep-Learning-Projects
pip install tensorflow keras scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

Open any `.ipynb` file in Jupyter Notebook or JupyterLab and run the cells top to bottom.

---

## Stack Overview

- **Python 3.x**
- **TensorFlow / Keras** — model building and training
- **scikit-learn** — preprocessing and evaluation metrics
- **imbalanced-learn** — SMOTE and resampling strategies
- **pandas / NumPy** — data manipulation
- **matplotlib / seaborn** — visualization

---
