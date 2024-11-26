import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Step 1: Load the dataset using pandas
data = pd.read_csv("iris.csv")  # Replace with your dataset file
print(data.head())

# Step 2: Check for missing values
if data.isnull().sum().any():
    print("Dataset contains missing values.")
else:
    print("No missing values in the dataset.")

# Step 3: Perform descriptive statistics
print(data.describe())

# Step 4: Check for class balance (for classification tasks, if applicable)
def check_class_balance(df, target_column):
    return df["variety"].value_counts()

# Step 5: Convert text classes to enumerated values 
data["variety"] = data["variety"].map({"Setosa": 0, "Versicolor": 1, "Virginica": 2})


# Step 6: Define input features (X) and target (y)
X = data.drop(columns=["petal.width"])  
y = data["petal.width"]

# Step 7: Split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 8: Create a LinearRegression model
model = LinearRegression()

# Step 9: Train the model
model.fit(X_train, y_train)

# Step 10: Perform predictions
y_pred = model.predict(X_test)

# Step 11: Evaluate the model and recover weights/bias
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Model parameters
print(f"Weights: {model.coef_}")
print(f"Bias: {model.intercept_}")

# Step 12: Implement a custom prediction function
def custom_predict(input_features, weights, bias):
    return sum(i * w for i, w in zip(input_features, weights)) + bias

# Test the custom function
sample_input = X_test.iloc[0].values
print(f"Custom Prediction: {custom_predict(sample_input, model.coef_, model.intercept_)}")
print(f"Sklearn Prediction: {model.predict([sample_input])[0]}")
