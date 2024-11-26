from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
# Required Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load the California Housing dataset
housing_data = fetch_california_housing(as_frame=True)
X = housing_data.data
y = housing_data.target

# Create datasets of different sizes
sizes = [0.06, 0.13, 0.25, 0.50, 0.75, 1.00]
results = {}

for size in sizes:
    subset = X.sample(frac=size, random_state=42)
    subset_y = y[subset.index]
    
    X_train, X_test, y_train, y_test = train_test_split(subset, subset_y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    
    results[f"{int(size * 100)}%"] = (mse_train, mse_test)

# Plot the results
train_mse = [v[0] for v in results.values()]
test_mse = [v[1] for v in results.values()]

plt.figure(figsize=(10, 5))
plt.plot(results.keys(), train_mse, label="Training MSE", marker="o")
plt.plot(results.keys(), test_mse, label="Testing MSE", marker="o")
plt.xlabel("Dataset Size")
plt.ylabel("Mean Squared Error")
plt.title("Impact of Dataset Size on Model Performance")
plt.legend()
plt.grid()
plt.show()

# Select a feature to manipulate
feature = "MedInc"  # Median Income feature

# Define removal rates
removal_rates = [0.2, 0.4, 0.6, 0.8]
mse_missing_values = []

for rate in removal_rates:
    X_copy = X.copy()
    
    # Randomly remove values
    missing_indices = X_copy.sample(frac=rate, random_state=42).index
    X_copy.loc[missing_indices, feature] = None
    
    # Separate rows with and without missing values
    complete_data = X_copy.dropna()
    incomplete_data = X_copy[X_copy[feature].isna()]
    
    # Train a model on complete data
    model = LinearRegression()
    model.fit(complete_data.drop(columns=[feature]), y[complete_data.index])
    
    # Predict missing values
    incomplete_data[feature] = model.predict(incomplete_data.drop(columns=[feature]))
    
    # Recombine the datasets
    filled_data = pd.concat([complete_data, incomplete_data]).sort_index()
    
    # Train a new model with the filled dataset
    X_train, X_test, y_train, y_test = train_test_split(filled_data, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    mse = mean_squared_error(y_test, model.predict(X_test))
    mse_missing_values.append(mse)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(removal_rates, mse_missing_values, marker="o")
plt.xlabel("Removal Rate")
plt.ylabel("Mean Squared Error")
plt.title("Impact of Missing Values on Model Performance")
plt.grid()
plt.show()

