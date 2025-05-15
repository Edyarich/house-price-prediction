import os
import pandas as pd
import requests
import streamlit as st

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:5000")

st.title("🏠 House-price predictor")
st.write(
    "Upload a **CSV** with the same columns the model was trained on "
    "(except *price_doc*). We’ll return a new column **predicted_price**."
)

file = st.file_uploader("Choose CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.write(f"✅ Loaded {len(df)} rows.")
    
    if st.button("Predict"):
        preds = []
        for _, row in df.iterrows():
            features = {k: (v if pd.notna(v) else None) for k, v in row.items()}
            r = requests.post(f"{MODEL_SERVICE_URL}/predict", json={"features": features})
            preds.append(r.json()["prediction"])
            
        df["predicted_price"] = preds
        st.success("Done!")
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode()
        st.download_button("Download predictions", csv, file_name="predictions.csv")
