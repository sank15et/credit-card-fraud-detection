
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)


@st.cache_data
def load_comparison():
    return pd.read_csv(REPORT_PATH)


# Only these files are required to run the online app.
for path in [MODEL_PATH, REPORT_PATH]:
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()

saved_object = load_model()

if not isinstance(saved_object, dict) or "model" not in saved_object:
    st.error("The saved model file has an unexpected format.")
    st.stop()

model = saved_object["model"]
threshold = float(saved_object.get("threshold", 0.5))
comparison = load_comparison()

# The dataset is optional. It is excluded from GitHub.
data = load_dataset() if DATA_PATH.exists() else None

st.title("💳 Credit Card Fraud Detection")
st.write(
    "A machine learning application that estimates the probability "
    "of credit card fraud using a trained Random Forest model."
)

st.warning(
    "Educational demonstration only. Predictions are estimates, "
    "not guarantees, and must not be used for real financial decisions."
)

if data is not None:
    total = len(data)
    fraud_count = int(data["Class"].sum())
    legitimate_count = total - fraud_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Legitimate Transactions", f"{legitimate_count:,}")
    c3.metric("Fraud Transactions", f"{fraud_count:,}")
else:
    st.info(
        "Online demo mode: the original CSV dataset is not included "
        "in this deployment. Enter sample feature values below."
    )

st.divider()
st.subheader("🔎 Test a Transaction")

if data is not None:
    st.write(
        "Choose a transaction from the local dataset. "
        "Its actual label is used only for comparison after prediction."
    )

    row_number = st.slider(
        "Dataset transaction row",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
    )

    selected = data.iloc[row_number]
    transaction = selected[FEATURES].to_frame().T

    st.caption(
        f"Selected row: {row_number:,} | "
        f"Amount: {selected['Amount']:.2f}"
    )

    analyze = st.button("Analyze Selected Transaction", type="primary")

else:
    st.write(
        "Enter the Time, Amount, and anonymized V1–V28 features. "
        "Zero values are provided as demo defaults, not real transaction data."
    )

    with st.form("transaction_form"):
        time_value = st.number_input(
            "Time (seconds from dataset reference)",
            min_value=0.0,
            value=0.0,
        )
        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=0.0,
        )

        with st.expander("Anonymized features (V1–V28)"):
            feature_values = {}
            cols = st.columns(4)

            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    feature_values[f"V{i}"] = st.number_input(
                        f"V{i}",
                        value=0.0,
                        format="%.6f",
                        key=f"feature_V{i}",
                    )

        analyze = st.form_submit_button(
            "Analyze Transaction",
            type="primary",
        )

        transaction_values = {
            "Time": time_value,
            **feature_values,
            "Amount": amount_value,
        }
        transaction = pd.DataFrame([transaction_values], columns=FEATURES)

if analyze:
    try:
        if not hasattr(model, "predict_proba"):
            st.error("This model does not support probability predictions.")
            st.stop()

        probabilities = model.predict_proba(transaction)[0]
        class_labels = list(model.classes_)

        if 1 not in class_labels:
            st.error("The model does not contain a fraud class.")
            st.stop()

        probability = float(probabilities[class_labels.index(1)])
        prediction = int(probability >= threshold)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Model Prediction: Potentially Fraudulent")
        else:
            st.success("✅ Model Prediction: Legitimate")

        st.metric("Estimated Fraud Probability", f"{probability:.2%}")
        st.caption(f"Decision threshold: {threshold:.2f}")

        if data is not None:
            actual = int(data.iloc[row_number]["Class"])
            st.subheader("Actual Dataset Label")
            st.write(f"Actual label: **{'Fraud' if actual == 1 else 'Legitimate'}**")

            if prediction == actual:
                st.success("Prediction matches the dataset label.")
            else:
                st.error("Prediction does not match the dataset label.")
        else:
            st.caption(
                "No actual label is available for this manually entered "
                "example. This is not an independent model evaluation."
            )

        with st.expander("View transaction features"):
            st.dataframe(transaction, use_container_width=True)

    except Exception as error:
        st.error(f"Unable to analyze transaction: {error}")

st.divider()
st.subheader("📊 Model Performance Comparison")
st.dataframe(comparison, use_container_width=True, hide_index=True)

if {"model", "precision", "recall", "f1_score"}.issubset(comparison.columns):
    st.bar_chart(
        comparison.set_index("model")[["precision", "recall", "f1_score"]]
    )

st.caption(
    "Metrics are from the saved test-set evaluation. "
    "This application is for educational purposes only."
)
st.divider()
st.caption("Built with Python, pandas, scikit-learn, joblib, and Streamlit.")
