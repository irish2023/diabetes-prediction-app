from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Welcome to the Diabetes Prediction App!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)