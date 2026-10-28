# 💳 Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent credit card transactions using **Logistic Regression** and **Random Forest**, with an interactive Streamlit dashboard for transaction predictions and model performance comparison.

## 🚀 Project Overview

Credit card fraud detection is a binary classification problem with a highly imbalanced dataset. This project explores transaction data, trains two machine learning models, compares their performance, and provides an interactive web application for testing predictions.

### Key Features

- Exploratory Data Analysis (EDA) and visualizations
- Data cleaning and duplicate removal
- Stratified train-test split
- Logistic Regression and Random Forest model comparison
- Precision, recall, F1-score, Average Precision, and ROC-AUC evaluation
- Saved model and metadata using Joblib
- Interactive Streamlit dashboard
- Real dataset transaction testing
- Model performance comparison chart

## 🛠️ Technology Stack

- **Language:** Python 3.13.5
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Streamlit
- **Model persistence:** Joblib
- **Development:** Visual Studio Code

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

**Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The original dataset contains 284,807 transactions and 31 columns:

- `Time`: Seconds elapsed between a transaction and the first transaction in the dataset
- `V1`–`V28`: Anonymized features obtained through Principal Component Analysis (PCA)
- `Amount`: Transaction amount
- `Class`: Target label (`0` = legitimate, `1` = fraud)

After removing 1,081 exact duplicate rows, the cleaned dataset contains 283,726 transactions, including 473 fraud cases.

### Class Distribution After Cleaning

| Transaction type | Count |
|---|---:|
| Legitimate | 283,253 |
| Fraud | 473 |
| Total | 283,726 |

Fraudulent transactions represent approximately 0.167% of the cleaned dataset, making class imbalance an important consideration during model training and evaluation.

## 🧠 Machine Learning Models

### 1. Logistic Regression
A classification baseline trained using feature scaling and class balancing.

### 2. Random Forest
An ensemble classifier trained using 150 decision trees and balanced class weights.

The dataset was split into training and test sets using an 80:20 stratified split with `random_state=42`.

## 📈 Model Evaluation Results

The following results were obtained on the held-out test set using a classification threshold of 0.5.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 5.64% | 98.53% |
| Recall | 87.37% | 70.53% |
| F1-score | 10.59% | 82.21% |
| Average Precision | 67.19% | 80.05% |
| ROC-AUC | 96.57% | 92.92% |
| False Positives | 1,389 | 1 |
| False Negatives | 12 | 28 |

Random Forest achieved higher precision and F1-score, while Logistic Regression achieved higher recall and ROC-AUC. Random Forest was selected by the training script based on Average Precision.

These results illustrate the trade-off between catching fraudulent transactions and minimizing false alarms.

## 🖥️ Streamlit Application

The application provides:

- Dataset summary metrics
- A slider to select a transaction from the cleaned dataset
- A fraud prediction and estimated fraud probability
- Comparison with the actual dataset label after prediction
- A model performance comparison table and chart

The actual-label comparison is intended for demonstration. It is not an independent evaluation of model performance.

## 📁 Project Structure

```text
credit-card-fraud-detection/
├── app.py
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_detection_model.joblib
│   └── model_metadata.json
├── reports/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── amount_by_class.png
│   ├── transaction_time.png
│   ├── correlation_heatmap.png
│   ├── eda_summary.csv
│   ├── missing_values.csv
│   └── model_comparison.csv
├── src/
│   ├── eda.py
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated files may vary depending on which scripts have been run.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.13.5
- Git
- Visual Studio Code (recommended)

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd credit-card-fraud-detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from the Kaggle dataset linked above and place it in the `data/` directory.

### 5. Run the Streamlit application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

### 6. Reproduce the analysis and training

To regenerate EDA outputs:

```powershell
python src/eda.py
```

To train and evaluate the models:

```powershell
python src/train_model.py
```

Training regenerates the saved model and model evaluation reports.

## ⚠️ Limitations

- The dataset contains historical transactions and anonymized features.
- Individual transaction predictions are not guarantees of fraud or legitimacy.
- Model performance may change with different datasets and transaction patterns.
- The displayed probability should not be interpreted as a calibrated real-world risk estimate without additional validation.
- The model has not been established as suitable for production financial decisions.
- The reported model comparison uses the held-out test set; model selection and threshold tuning should be validated separately for a more rigorous evaluation.

## 🔮 Future Improvements

- Tune the classification threshold using validation data
- Evaluate additional imbalance-handling techniques
- Add confusion matrix and precision-recall visualizations
- Add automated model evaluation reports
- Improve model explainability
- Deploy the Streamlit application online
- Add automated tests and continuous integration

## 👨‍💻 Author

Developed as a machine learning project to practice data analysis, imbalanced classification, model evaluation, and deployment with Streamlit.
