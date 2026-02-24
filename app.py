import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Column index mapping for the scaler
FEATURE_INDEX_MAP = {
    "gender":           0,
    "SeniorCitizen":    1,
    "Partner":          2,
    "Dependents":       3,
    "tenure":           4,
    "PhoneService":     5,
    "MultipleLines":    6,
    "InternetService":  7,
    "OnlineSecurity":   8,
    "OnlineBackup":     9,
    "DeviceProtection": 10,
    "TechSupport":      11,
    "StreamingTV":      12,
    "StreamingMovies":  13,
    "Contract":         14,
    "PaperlessBilling": 15,
    "PaymentMethod":    16,
    "MonthlyCharges":   17,
    "TotalCharges":     18,
}

TOTAL_FEATURES = 19

# Encoding maps matching the label encoding used during training
ENCODINGS = {
    "gender":           {"Male": 0, "Female": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "MultipleLines":    {"No phone service": 0, "No": 1, "Yes": 2},
    "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity":   {"No internet service": 0, "No": 1, "Yes": 2},
    "OnlineBackup":     {"No internet service": 0, "No": 1, "Yes": 2},
    "DeviceProtection": {"No internet service": 0, "No": 1, "Yes": 2},
    "TechSupport":      {"No internet service": 0, "No": 1, "Yes": 2},
    "StreamingTV":      {"No internet service": 0, "No": 1, "Yes": 2},
    "StreamingMovies":  {"No internet service": 0, "No": 1, "Yes": 2},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaymentMethod":    {
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    },
}

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction System")
st.divider()

# Load model and scaler
@st.cache_resource
def load_artifacts():
    try:
        model  = pickle.load(open("churn_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    scaler_n = getattr(scaler, "n_features_in_", None)
    if scaler_n is not None and scaler_n != TOTAL_FEATURES:
        st.error(f"❌ Feature count mismatch: scaler expects {scaler_n}, got {TOTAL_FEATURES}")
        st.stop()

    st.success(f"✅ Model loaded successfully")
    return model, scaler

model, scaler = load_artifacts()

# Session state for prediction results
for key in ["prediction", "churn_prob"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Input form
st.subheader("Enter Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Info**")
        tenure    = st.number_input("Tenure (months)", 0, 72, 6)
        contract  = st.selectbox("Contract", list(ENCODINGS["Contract"].keys()))
        payment   = st.selectbox("Payment Method", list(ENCODINGS["PaymentMethod"].keys()))
        paperless = st.selectbox("Paperless Billing", list(ENCODINGS["PaperlessBilling"].keys()))

    with col2:
        st.markdown("**Demographics & Charges**")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 85.0)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 12000.0, 868.4)
        gender          = st.selectbox("Gender", list(ENCODINGS["gender"].keys()))
        senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner         = st.selectbox("Partner", list(ENCODINGS["Partner"].keys()))
        dependents      = st.selectbox("Dependents", list(ENCODINGS["Dependents"].keys()))

    with col3:
        st.markdown("**Services**")
        phone_service     = st.selectbox("Phone Service",     list(ENCODINGS["PhoneService"].keys()))
        multiple_lines    = st.selectbox("Multiple Lines",    list(ENCODINGS["MultipleLines"].keys()))
        internet_service  = st.selectbox("Internet Service",  list(ENCODINGS["InternetService"].keys()))
        online_security   = st.selectbox("Online Security",   list(ENCODINGS["OnlineSecurity"].keys()))
        online_backup     = st.selectbox("Online Backup",     list(ENCODINGS["OnlineBackup"].keys()))
        device_protection = st.selectbox("Device Protection", list(ENCODINGS["DeviceProtection"].keys()))
        tech_support      = st.selectbox("Tech Support",      list(ENCODINGS["TechSupport"].keys()))
        streaming_tv      = st.selectbox("Streaming TV",      list(ENCODINGS["StreamingTV"].keys()))
        streaming_movies  = st.selectbox("Streaming Movies",  list(ENCODINGS["StreamingMovies"].keys()))

    submit = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

st.divider()

# Build feature vector from user inputs
def build_feature_vector(raw_inputs):
    vector = np.zeros((1, TOTAL_FEATURES), dtype=float)
    for feature, col_idx in FEATURE_INDEX_MAP.items():
        value = raw_inputs[feature]
        if feature in ENCODINGS:
            value = ENCODINGS[feature][value]
        vector[0, col_idx] = value
    return vector

# Run prediction on form submit
if submit:
    raw_inputs = {
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "gender":           gender,
        "Partner":          partner,
        "Dependents":       dependents,
        "PhoneService":     phone_service,
        "PaperlessBilling": paperless,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaymentMethod":    payment,
    }

    try:
        feature_vector = build_feature_vector(raw_inputs)
        scaled_vector  = scaler.transform(feature_vector)
        st.session_state.prediction = int(model.predict(scaled_vector)[0])
        st.session_state.churn_prob = float(model.predict_proba(scaled_vector)[0][1])
        st.rerun()
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# Display results
tab1, tab2 = st.tabs(["Prediction Result", "Model Info"])

with tab1:
    if st.session_state.prediction is not None:
        prob = st.session_state.churn_prob
        pred = st.session_state.prediction

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Will Churn?", "🚨 YES" if pred else "✅ NO")
        with col2:
            st.metric("Churn Probability", f"{prob:.1%}")
        with col3:
            if prob > 0.7:
                st.error("🔴 HIGH RISK")
            elif prob > 0.4:
                st.warning("🟡 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

        st.subheader("📊 Risk Breakdown")
        prob_df = pd.DataFrame({
            "Outcome":     ["Churn", "Stay"],
            "Probability": [prob, 1 - prob]
        })
        st.bar_chart(prob_df.set_index("Outcome"))

        st.subheader("💡 Suggested Actions")
        if prob > 0.7:
            st.error("High churn risk. Recommended: offer a contract upgrade, proactively reach out, or review open service complaints.")
        elif prob > 0.4:
            st.warning("Moderate risk. Consider a satisfaction survey or highlighting underused services.")
        else:
            st.success("Customer appears stable. Standard engagement is sufficient.")
    else:
        st.info("Fill in the form and click **Predict Churn** to see results.")

with tab2:
    st.subheader("🔍 Model Details")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Type", type(model).__name__)
        st.metric("Total Features", TOTAL_FEATURES)
    with col2:
        st.metric("Scaler Features", getattr(scaler, "n_features_in_", "Unknown"))

    # Feature importance chart
    if hasattr(model, "feature_importances_"):
        feature_names = [k for k, _ in sorted(FEATURE_INDEX_MAP.items(), key=lambda x: x[1])]
        if len(feature_names) == len(model.feature_importances_):
            importance_df = pd.DataFrame({
                "Feature":    feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.subheader("📈 Feature Importances")
            st.bar_chart(importance_df.set_index("Feature"))