# Credit_Card_Fraud_Detection
Credit Card Fraud Detection using ML: Built a real-time fraud detection system using Python and Streamlit, achieving 97%+ accuracy with Random Forest and SVM classifiers. Features include manual/batch prediction, model evaluation metrics, and interactive deployment for end-users.




# 🛡️ Credit Card Fraud Detection using Machine Learning

This project uses Machine Learning algorithms to detect fraudulent transactions from credit card data. It involves data cleaning, model building, evaluation, and deployment via Streamlit.

## 📌 Problem Statement
Credit card fraud detection is a major challenge. The dataset is highly imbalanced and contains sensitive financial data. The aim is to predict fraud with high precision and recall.

## 🧠 Algorithms Used
- Random Forest Classifier
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP Neural Net)

## 📊 Model Performance (based on `model_metrics.pkl`)
- **Accuracy**: 97%
- **Precision**: 95%
- **F1-Score**: 94%
- Evaluated using cross-validation and test dataset.

## 💻 Streamlit App
An interactive app was created to:
- Upload test transactions
- Predict manually or in batch
- Show fraud probability & metrics

## 🧾 Files
| File | Description |
|------|-------------|
| `fraud_detection.ipynb` | Full analysis and ML pipeline |
| `fraud_detection_model.pkl` | Trained ML model |
| `model_metrics.pkl` | Evaluation metrics (accuracy, etc.) |
| `streamlit_app.py` | Streamlit interface |
| `requirements.txt` | Python dependencies |
| `data/test_sample.csv` | Sample input data |

## 🚀 Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/Credit_Card_Fraud_Detection.git
   cd Credit_Card_Fraud_Detection

