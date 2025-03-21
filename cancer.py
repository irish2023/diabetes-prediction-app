# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the Breast Cancer dataset
data = load_breast_cancer()
print(data)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display basic dataset information
print("Dataset Shape:", df.shape)
print("First 5 rows of the dataset:")
print(df.head())

# Visualize the target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df['target'], palette="viridis")
plt.title("Target Distribution (0 = Malignant, 1 = Benign)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# Separate features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Malignant", "Benign"], 
            yticklabels=["Malignant", "Benign"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Predict using new input data
new_data = np.array([[13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
                      0.2699, 0.7886, 1.929, 24.54, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
                      15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

# Output the prediction
print("\nPrediction for new input data:", "Benign" if prediction[0] == 1 else "Malignant")
