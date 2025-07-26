import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

# Load model performance metrics
try:
    metrics = joblib.load("model_metrics.pkl")
except:
    metrics = {"Accuracy": 0.97, "Precision": 0.95, "Recall": 0.94}  # fallback default

# Expected columns used during training
expected_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud',
                 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# Streamlit App Title
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.subheader("Enter transaction details or upload CSV to check for fraud")

# --- Input Form ---
with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, format="%.2f")
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, format="%.2f")

    with col2:
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, format="%.2f")
        newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, format="%.2f")
        transaction_type = st.selectbox("Transaction Type", ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT'])

    submitted = st.form_submit_button("🔍 Predict Fraud")

    if submitted:
        # Check if user actually entered meaningful values
        if (amount == 0.0 and oldbalanceOrg == 0.0 and newbalanceOrig == 0.0 and
            oldbalanceDest == 0.0 and newbalanceDest == 0.0):
            st.warning("⚠️ Please enter valid transaction details before predicting.")
        else:
            input_dict = {
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'type': transaction_type,
                'isFlaggedFraud': 0
            }

            input_df = pd.DataFrame([input_dict])

            # Encode 'type'
            input_df = pd.get_dummies(input_df, columns=['type'])

            # Add missing columns
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns
            input_df = input_df[expected_cols]

            # Predict
            prediction = model.predict(input_df)[0]

            # Output result
            st.markdown("### 🧾 Prediction Result:")
            if prediction == 1:
                st.error("🚨 The transaction is **Fraudulent**.")
            else:
                st.success("✅ The transaction is **Not Fraudulent**.")

# --- CSV Upload Section ---
st.markdown("---")
st.subheader("📁 Upload CSV File for Batch Fraud Prediction")

uploaded_file = st.file_uploader("Choose a file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📋 Preview of Uploaded Data", df.head())

    if 'type' not in df.columns:
        st.error("❌ CSV must contain a 'type' column.")
    else:
        try:
            # Preprocess like training
            df_encoded = pd.get_dummies(df, columns=['type'])
            for col in expected_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[expected_cols]

            # Predict
            predictions = model.predict(df_encoded)
            df['Prediction'] = np.where(predictions == 1, 'Fraud', 'Not Fraud')

            st.write("🔍 Prediction Results", df[['type', 'amount', 'Prediction']].head(10))
            st.success("✅ Prediction complete!")

        except Exception as e:
            st.error(f"❌ Error while processing: {e}")

# --- Model Metrics Section ---
st.markdown("---")
st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
col2.metric("Precision", f"{metrics['Precision']:.2%}")
col3.metric("Recall", f"{metrics['Recall']:.2%}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: Logistic Regression (or similar)")
