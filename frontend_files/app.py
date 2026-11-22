import os
import requests
import pandas as pd
import streamlit as st

# Backend address. Inside the Docker network the backend container is reachable by its name.
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
st.title("🛒 SuperKart Sales Forecast")
st.write("Predict total product sales for a store, for one product or a whole CSV file.")

# ---------------- Online (single) prediction ----------------
st.subheader("Online prediction")

col1, col2 = st.columns(2)
with col1:
    product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.66, step=0.01)
    sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=1.0,
                                     value=0.027, step=0.001, format="%.3f")
    mrp = st.number_input("Product MRP", min_value=0.0, max_value=500.0, value=117.08, step=0.01)
    product_id_char = st.selectbox("Product Category Code", ["FD", "NC", "DR"],
                                   help="FD = Food, NC = Non-Consumable, DR = Drinks")
with col2:
    store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
    city_type = st.selectbox("Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"])
    store_type = st.selectbox("Store Type", ["Supermarket Type2", "Supermarket Type1",
                                             "Departmental Store", "Food Mart"])
    store_age = st.number_input("Store Age (years)", min_value=0, max_value=100, value=16, step=1)
    type_category = st.selectbox("Product Type Category", ["Non Perishables", "Perishables"])

if st.button("Predict Sales"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": int(store_age),
        "Product_Type_Category": type_category,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Product Store Sales Total: {result['Predicted Sales']:,.2f}")
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend API: {e}")

# ---------------- Batch prediction ----------------
st.subheader("Batch prediction")
uploaded_file = st.file_uploader("Upload a CSV file with the 10 feature columns", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(input_df.head())

    if st.button("Predict Batch"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": ("batch.csv", uploaded_file.getvalue(), "text/csv")},
                timeout=60,
            )
            if response.status_code == 200:
                preds = response.json()
                output_df = input_df.copy()
                output_df["Predicted_Sales"] = [preds[str(i)] for i in range(len(output_df))]
                st.success(f"Predictions generated for {len(output_df)} records.")
                st.dataframe(output_df)
                st.download_button("Download predictions as CSV",
                                   output_df.to_csv(index=False).encode("utf-8"),
                                   file_name="superkart_predictions.csv", mime="text/csv")
            else:
                st.error(f"API error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend API: {e}")
