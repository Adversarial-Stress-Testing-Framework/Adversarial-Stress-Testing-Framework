import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# =========================
# LOAD MODEL + SCALER
# =========================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


# Fit encoders on the same training data/column order main.py used, since
# main.py doesn't persist the LabelEncoders. Deterministic given the same file.
@st.cache_resource
def fit_encoders():
    df = pd.read_csv("KDDTrain+.txt", header=None)
    X = df.iloc[:, :-2]
    cat_cols = X.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X[col])
        encoders[col] = le
    return encoders, list(X.columns), list(cat_cols)


def preprocess(df_raw, encoders, feature_cols, cat_cols):
    X = df_raw.copy()
    X = X.iloc[:, : len(feature_cols)]
    X.columns = feature_cols

    for col in cat_cols:
        le = encoders[col]
        known = set(le.classes_)
        X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
        X[col] = le.transform(X[col])

    return X


def predict_batch(model, scaler, X):
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    labels = np.where(preds == 0, "Normal", "Attack")
    return X_scaled, preds, labels


# =========================
# APP
# =========================
st.title("🛡️ Network Intrusion Detection System")
st.caption("RandomForest classifier trained on the NSL-KDD dataset")

model, scaler = load_model_and_scaler()
encoders, feature_cols, cat_cols = fit_encoders()

tab1, tab2, tab3 = st.tabs(["📁 Batch Prediction", "⚔️ Adversarial Robustness", "ℹ️ Model Info"])

# -------------------------
# TAB 1: BATCH PREDICTION
# -------------------------
with tab1:
    st.subheader("Upload traffic data (CSV, no header, 41 KDD features)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "txt"], key="batch")

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, header=None)
        st.write(f"Loaded **{df_raw.shape[0]}** rows, **{df_raw.shape[1]}** columns.")

        try:
            X = preprocess(df_raw, encoders, feature_cols, cat_cols)
            _, preds, labels = predict_batch(model, scaler, X)

            result_df = df_raw.copy()
            result_df["Prediction"] = labels

            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(result_df, use_container_width=True)
            with col2:
                attack_count = int((preds == 1).sum())
                normal_count = int((preds == 0).sum())
                st.metric("Normal", normal_count)
                st.metric("Attack", attack_count)
                st.bar_chart(pd.Series({"Normal": normal_count, "Attack": attack_count}))

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv_out, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Failed to process file: {e}")
    else:
        st.info("Upload a CSV to get predictions.")

# -------------------------
# TAB 2: ADVERSARIAL ROBUSTNESS
# -------------------------
with tab2:
    st.subheader("Compare predictions on clean vs. noise-perturbed data")
    uploaded_adv = st.file_uploader("Choose a CSV file", type=["csv", "txt"], key="adv")
    noise_std = st.slider("Gaussian noise std-dev", 0.0, 2.0, 0.5, 0.1)

    if uploaded_adv is not None:
        df_raw_adv = pd.read_csv(uploaded_adv, header=None)

        try:
            X = preprocess(df_raw_adv, encoders, feature_cols, cat_cols)
            X_scaled, preds_clean, labels_clean = predict_batch(model, scaler, X)

            rng = np.random.default_rng(42)
            noise = rng.normal(0, noise_std, X_scaled.shape)
            X_scaled_adv = X_scaled + noise
            preds_adv = model.predict(X_scaled_adv)
            labels_adv = np.where(preds_adv == 0, "Normal", "Attack")

            flipped = int((preds_clean != preds_adv).sum())

            comparison_df = pd.DataFrame({
                "Clean Prediction": labels_clean,
                "Adversarial Prediction": labels_adv,
                "Flipped": preds_clean != preds_adv,
            })

            st.dataframe(comparison_df, use_container_width=True)

            colA, colB, colC = st.columns(3)
            colA.metric("Total rows", len(comparison_df))
            colB.metric("Predictions flipped", flipped)
            colC.metric("Flip rate", f"{flipped / len(comparison_df):.2%}")

            st.caption(
                "If the uploaded file includes the true label column (2nd from last, "
                "as in KDDTrain+.txt), accuracy comparison can be computed manually "
                "against 'Clean Prediction' / 'Adversarial Prediction'."
            )

        except Exception as e:
            st.error(f"Failed to process file: {e}")
    else:
        st.info("Upload a CSV to run the adversarial comparison.")

# -------------------------
# TAB 3: MODEL INFO
# -------------------------
with tab3:
    st.subheader("Model details")
    st.write(f"**Algorithm:** {type(model).__name__}")
    st.write(f"**Number of trees:** {getattr(model, 'n_estimators', 'N/A')}")
    st.write(f"**Number of features expected:** {model.n_features_in_}")
    st.write(f"**Classes:** {dict(zip(model.classes_, ['Normal', 'Attack']))}")
