
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# --------------------------------------------------
# 2. Load saved resources
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return (
        pd.read_csv(DATA_PATH)
        .drop_duplicates()
        .reset_index(drop=True)
    )


@st.cache_data
def load_comparison():
    return pd.read_csv(REPORT_PATH)


# Check required files
for path in [MODEL_PATH, DATA_PATH, REPORT_PATH]:
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()

saved_object = load_model()

if not isinstance(saved_object, dict) or "model" not in saved_object:
    st.error("The saved model file does not contain the expected model.")
    st.stop()

model = saved_object["model"]
threshold = float(saved_object.get("threshold", 0.5))

data = load_dataset()

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# --------------------------------------------------
# 3. Dashboard header
# --------------------------------------------------
st.title("💳 Credit Card Fraud Detection")

st.write(
    "A machine learning application that predicts potentially "
    "fraudulent credit card transactions using Random Forest."
)

st.warning(
    "Educational demonstration only. Predictions are not guarantees "
    "and should not be used for real-world financial decisions."
)

# --------------------------------------------------
# 4. Dataset overview
# --------------------------------------------------
total = len(data)
fraud_count = int(data["Class"].sum())
legitimate_count = total - fraud_count

c1, c2, c3 = st.columns(3)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Legitimate Transactions", f"{legitimate_count:,}")
c3.metric("Fraud Transactions", f"{fraud_count:,}")

st.divider()

# --------------------------------------------------
# 5. Select a real dataset transaction
# --------------------------------------------------
st.subheader("🔍 Test a Real Dataset Transaction")

st.write(
    "Select a transaction from the dataset. The model receives only "
    "the transaction features, not the actual Class label."
)

row_number = st.slider(
    "Dataset transaction row",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1,
)

selected = data.iloc[row_number]

st.caption(
    f"Selected row: {row_number:,} | "
    f"Transaction amount: {selected['Amount']:.2f}"
)

if st.button("Analyze Selected Transaction", type="primary"):

    # Class is excluded from the prediction input.
    transaction = selected[FEATURES].to_frame().T

    try:
        # Obtain fraud probability and apply the saved threshold.
        if not hasattr(model, "predict_proba"):
            st.error("This model does not support probability predictions.")
            st.stop()

        probabilities = model.predict_proba(transaction)[0]
        class_labels = list(model.classes_)

        if 1 not in class_labels:
            st.error("The model does not contain a fraud class.")
            st.stop()

        fraud_index = class_labels.index(1)
        probability = float(probabilities[fraud_index])

        # Use the threshold saved during training.
        prediction = int(probability >= threshold)

        # Read the actual label only after predicting.
        actual = int(selected["Class"])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Model Prediction: Potentially Fraudulent")
        else:
            st.success("✅ Model Prediction: Legitimate")

        st.metric(
            "Estimated Fraud Probability",
            f"{probability:.2%}",
        )

        st.caption(f"Decision threshold: {threshold:.2f}")

        st.subheader("Actual Dataset Label")

        if actual == 1:
            st.write("Actual label: **Fraud**")
        else:
            st.write("Actual label: **Legitimate**")

        if prediction == actual:
            st.success("Prediction matches the actual dataset label.")
        else:
            st.error("Prediction does not match the actual dataset label.")

        with st.expander("View transaction features"):
            st.dataframe(transaction, use_container_width=True)

    except Exception as error:
        st.error(f"Unable to analyze transaction: {error}")

st.caption(
    "The actual label is displayed for demonstration after prediction. "
    "This individual example is not an independent model evaluation."
)

# --------------------------------------------------
# 6. Model performance comparison
# --------------------------------------------------
st.divider()
st.subheader("📊 Model Performance Comparison")

comparison = load_comparison()

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(
    comparison.set_index("model")[
        ["precision", "recall", "f1_score"]
    ]
)

st.caption(
    "Metrics are from the saved test-set evaluation. "
    "Threshold tuning should be performed using validation data."
)

# --------------------------------------------------
# 7. Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Built with Python, pandas, scikit-learn, joblib, and Streamlit."
)
