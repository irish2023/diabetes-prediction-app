# Import necessary libraries
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Display basic dataset info
print("Dataset Shape:", data.shape)
print("First 5 rows of the dataset:")
print(data.head())

# Separate features (X) and target (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict using new input data
new_data = np.array([[2, 170, 70, 25, 85, 28.1, 0.627, 25]])  # Example data
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print("\nPrediction for new input data:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")

# Save the trained model and scaler
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")