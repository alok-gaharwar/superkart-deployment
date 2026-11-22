import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained pipeline (preprocessing + XGBoost) once, at startup
model = joblib.load("superkart_model.joblib")

# The exact 10 features the model was trained on
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category",
]

# Health check: confirms the API is up
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Online inference: one product-store record as JSON
@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
        prediction = float(model.predict(input_df)[0])
        return jsonify({"Predicted Sales": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Batch inference: a CSV file with many records
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Send a CSV under the key 'file'."}), 400

        input_df = pd.read_csv(request.files["file"])

        missing = [f for f in FEATURES if f not in input_df.columns]
        if missing:
            return jsonify({"error": f"CSV is missing columns: {missing}"}), 400

        predictions = model.predict(input_df[FEATURES])
        # {row_index: predicted_sales}
        result = {int(i): round(float(p), 2) for i, p in zip(input_df.index, predictions)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
