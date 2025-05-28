# Credit Card Fraud Detection System – Machine Learning Based

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Fraud%20Detection-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Project Overview

This project implements a **Credit Card Fraud Detection System** using machine learning techniques. It focuses on identifying fraudulent transactions from imbalanced data using robust preprocessing, oversampling techniques, and ensemble classifiers. Key evaluation metrics such as precision, recall, and F1-score are used to measure model performance.

## 🎯 Objectives

* Preprocess imbalanced datasets using **SMOTE** and **undersampling**.
* Train and evaluate machine learning models for fraud detection.
* Build a **command-line interface** to test transactions in real time.
* Visualize and analyze key metrics for fraud identification.

## 🚀 Models Implemented

### 1. **Random Forest Classifier**

* Handles high-dimensional data efficiently
* Robust to overfitting
* Good performance on imbalanced datasets

### 2. **Gradient Boosting Classifier**

* Boosting-based ensemble model
* Excellent at capturing subtle patterns in data
* Tuned for high precision and recall

## 📊 Dataset

**Credit Card Fraud Detection Dataset**

* **Source**: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Samples**: 284,807 transactions
* **Features**: 30 (V1–V28 anonymized features, Time, Amount)
* **Target**: `Class` (0 = legitimate, 1 = fraud)

### Key Characteristics:

* Highly **imbalanced** (only 0.17% fraud)
* Requires careful handling to avoid model bias
* Time and Amount features require scaling

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or VSCode (optional)
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Clone Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

## 📁 Project Structure

```
credit-card-fraud-detection/
├── README.md
├── data/
│   └── creditcard.csv              # Original dataset
├── models/
│   └── random_forest_model.pkl     # Saved model
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── scripts/
│   └── cli_test.py                 # Command-line testing script
└── requirements.txt
```

## 🏃‍♂️ How to Run

### Option 1: Run with Jupyter Notebook

1. Launch the notebook:

   ```bash
   jupyter notebook
   ```
2. Open:

   ```
   notebooks/fraud_detection_analysis.ipynb
   ```
3. Execute cells step-by-step

### Option 2: Run Command-Line Testing Interface

```bash
python scripts/cli_test.py
```

You’ll be prompted to enter feature values for prediction.

### Important: Pretrained Model

Make sure `random_forest_model.pkl` is present in the `models/` folder before running `cli_test.py`.

## 📈 Results

### Model Performance (on test set)

| Model             | Precision | Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
| Random Forest     | 0.92      | 0.88   | 0.90     |
| Gradient Boosting | 0.94      | 0.89   | 0.91     |

### Key Findings

1. **Oversampling (SMOTE)** improved model recall significantly.
2. **Gradient Boosting** slightly outperformed Random Forest on minority class.
3. **High precision** ensures fewer false alarms in fraud detection.
4. Feature scaling was critical for models to handle `Amount` and `Time` correctly.

### Confusion Matrix

* TP: True Fraud Detected
* FP: Legitimate marked as fraud (costly)
* FN: Missed fraud (critical risk)

## 📊 Visualizations

* Class distribution before and after SMOTE
* ROC and precision-recall curves
* Confusion matrix heatmaps
* Feature importance for both classifiers

## 🔍 Technical Implementation Details

### Data Preprocessing

* Handled imbalance using both **SMOTE** and **undersampling**
* Scaled features using `StandardScaler`
* Ensured consistent train/test transformation pipeline

### Model Training & Evaluation

* Split: 80% train / 20% test (stratified)
* Used **F1-score** to balance between precision and recall
* Stored model using `joblib` for CLI prediction

### CLI System

* Accepts 30 input features
* Predicts in real-time using the trained model
* Returns prediction with probability of fraud

## 🚧 Limitations & Future Work

### Limitations

* Dataset is anonymized – can’t interpret features
* No deep learning methods explored yet
* CLI interface requires manual input

### Future Improvements

* Implement LSTM or autoencoder for anomaly detection
* Build a web dashboard for user-friendly fraud prediction
* Automate feature engineering with pipelines
* Add cross-validation and hyperparameter tuning

## 🤝 Contributing

Contributions are welcome! You can improve:

* Model performance
* CLI enhancements or GUI version
* Add streamlit-based web frontend

Feel free to fork the repo and create a pull request.

## 📧 Contact

**Your Name**

* GitHub: [@https://github.com/MuhammadTalha549](https://github.com/MuhammadTalha549)
* Email: [@talhamuahammad549@gmail.com](talhamuahammad549@gmail.com)

## 🙏 Acknowledgments

* Dataset from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* `scikit-learn`, `imbalanced-learn` for modeling and preprocessing
* Community support and open-source ML ecosystem

---

⭐ **Found this project useful? Star the repository to support it!** ⭐

