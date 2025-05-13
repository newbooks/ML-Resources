#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    # Load the California housing dataset
    data = fetch_california_housing()

    # show the dataset and how it is organized
    print("Feature names:", data.feature_names)
    print("Target names:", data.target_names)
    print("Data shape:", data.data.shape)
    print("Target shape:", data.target.shape)
    print("Data description:", data.DESCR)



    # Pandas is a powerful Python library for data manipulation and analysis. 
    # It provides data structures like DataFrame and Series, which make it easy to handle structured data.
    # Convert the dataset into a DataFrame
    xs = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name=data.target_names[0])

    # Show the first few rows of the DataFrame
    print(xs.head())
    # Show the target name and first few rows of the target Series
    print(y.name)
    print(y.head())


    # Save the dataset, including both features and target to a CSV file and load it back as a DataFrame for demonstration
    xs.to_csv("california_housing.csv", index=False)
    y.to_csv("california_housing_target.csv", index=False)
    # Load the dataset from CSV files
    x = pd.read_csv("california_housing.csv")
    y = pd.read_csv("california_housing_target.csv")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_lin = lin_reg.predict(X_test_scaled)

    # Evaluate the linear regression model
    mse_lin = mean_squared_error(y_test, y_pred_lin)
    r2_lin = r2_score(y_test, y_pred_lin)
    print(f"Linear Regression MSE: {mse_lin:.2f}, R^2: {r2_lin:.2f}")

    # Train a random forest regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_rf = rf_reg.predict(X_test_scaled)

    # Evaluate the random forest regressor
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"Random Forest Regressor MSE: {mse_rf:.2f}, R^2: {r2_rf:.2f}")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred_lin)
    plt.title("Linear Regression Predictions")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred_rf)
    plt.title("Random Forest Predictions")
    plt.xlabel("True Values")
    
    plt.tight_layout()
    plt.show()