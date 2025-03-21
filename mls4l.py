# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (e.g., Pima Indians Diabetes Dataset from Kaggle)
# Replace 'diabetes.csv' with your dataset file path
data = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset
print(data.shape)
print(data.head())

# Separate features (X) and target (y)
X = data.drop(columns=['Outcome'])  # Features (e.g., Glucose, BMI, Age, etc.)
y = data['Outcome']  # Target (1 = Diabetes, 0 = No Diabetes)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset sizes
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
