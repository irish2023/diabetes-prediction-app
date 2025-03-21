from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    
    # Scale the input features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    
    # Return the result
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)