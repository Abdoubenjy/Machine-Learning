Part 1: ML Flowers --> Data Preparation and Model Training

This section focuses on preparing the dataset and implementing a Linear Regression model to make predictions. The key tasks include:

1- Dataset Loading
- Importing the dataset using Pandas.
- Checking for missing values and class balance.

2- Feature Engineering:
- Performing basic descriptive statistics.
- Transforming categorical classes into numeric values.

3- Data Splitting:
- Splitting the data into training and testing subsets using train_test_split.

4- Model Training:
- Building a Linear Regression model using scikit-learn.
- Training the model and evaluating it using Mean Squared Error (MSE).

5- Custom Prediction Function:
- Implementing a custom function to predict outputs using the learned weights and bias for validation.

Part 2: ML California --> Impact Analysis of Dataset Size and Missing Values

This section explores the effect of dataset characteristics on the performance of the regression model. The primary analyses include:

1- Impact of Dataset Size:

- Subsetting the original dataset into varying sizes (6%, 13%, 25%, 50%, 75%, and 100%).
- Training and evaluating the model on each subset using MSE as the metric.
- Visualizing the results to understand how dataset size affects model accuracy.

2- Impact of Missing Values:

- Simulating missing values in a specific feature by randomly removing data.
- Using the trained model to predict and fill the missing values.
- Re-training the model with the filled dataset and comparing its performance against the original data.
- Repeating the process for different removal rates (20%, 40%, 60%, 80%).
- Visualizing the relationship between the removal rate and model accuracy.
