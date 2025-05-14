#!/usr/bin/env python

# California Housing Regression: A Complete Example


# import the modules
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib



if __name__ == "__main__":

    # load the California housing dataset
    data = fetch_california_housing()

    # show how data is organized
    print("What is in the dataset?")
    print(data)
    print()

    # breakdown
    print("Dataset summary:")
    print("Feature names:", data.feature_names)
    print("Target names:", data.target_names)
    print("Data shape:", data.data.shape)
    print("Target shape:", data.target.shape)
    print()

    # Convert the dataset to a DataFrame
    xs = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name=data.target_names[0])

    # Show the first few rows of the DataFrame
    print("First few rows of the features (xs):")
    print(xs.head())
    print()

    # Show the first few rows of the target
    print("First few rows of the target (y):")
    print(y.head())
    print()

    # Save to files
    print("Saving the dataset to CSV files...")
    xs.to_csv("california_housing.csv", index=False)
    y.to_csv("california_housing_target.csv", index=False)
    print("Dataset saved to 'california_housing.csv' and 'california_housing_target.csv'.")
    print()

    # Load the dataset from CSV files
    print("Loading the dataset from CSV files...")
    xs = pd.read_csv("california_housing.csv")
    y = pd.read_csv("california_housing_target.csv")
    print("Dataset loaded from 'california_housing.csv' and 'california_housing_target.csv'.")
    print()


    # Visualize the data
    # Histogram of the target
    print("Visualizing the data...Close the plot to continue.")
    sns.histplot(y, bins=50, kde=True)
    plt.title('Distribution of Median House Value')
    plt.show()
    

    # Correlation heatmap
    print("Visualizing the feature correlation...Close the plot to continue.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(xs.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Train-test split. This split assign 80% of the data as traning set and 20% as testing set, 
    # random_state is given the random seed 42 so that the split returns the same sets. 
    # You can use other random seeds.
    X_train, X_test, y_train, y_test = train_test_split(xs, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Predict
    y_pred_linear = lr.predict(X_test_scaled)

    print("Linear regression model coefficients:")
    # Obtain feature names
    feature_names = X_train.columns.to_list()  # e.g., from a DataFrame: X_train.columns

    # Assuming you have a list of feature names
    for name, coef in zip(feature_names, lr.coef_[0]):
        print(f"{name}: {coef:.4f}")
    print()
    print(f"Intercept is: {lr.intercept_[0]:.4f}")
    print()

    # Visiualize the predicted values over actual values
    print("Visualizing the predicted values over actual values...Close the plot to continue.")
    y_values = y_test["MedHouseVal"].values
    y_predicted = y_pred_linear.flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_predicted, label='Prediction by linear regression', alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate
    mse = mean_squared_error(y_test, y_pred_linear)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_linear)

    print(f"Linear Regression MSE (0 means perfect prediction, 1 means as good as guessing mean): {mse:.2f}")
    print(f"Linear Regression RMSE: {rmse:.2f}")
    print(f"Linear Regression R²: {r2:.2f}")
    print()


    # Train a Random Forest Regressor
    print("Training a Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train.values.ravel())
    rf_pred = rf.predict(X_test_scaled)

    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    print(f"Random Forest MSE: {rf_mse:.2f}")
    print(f"Random Forest R²: {rf_r2:.2f}")
    print()

    # Visualize the predicted values over actual values
    print("Visualizing the predicted values over actual values...Close the plot to continue.")
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test.values.ravel(), y=rf_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Predicted Median House Value')
    plt.title('Actual vs Predicted Values')
    plt.show()


    # Report feature importance
    feature_importances = rf.feature_importances_
    feature_names = xs.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print("Feature Importances:")
    print(feature_importance_df)
    print()

    # Plot feature importances
    print("Plotting feature importances...Close the plot to continue.")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


    # Save models
    print("Saving the models...")
    joblib.dump(lr, 'linear_regression_model.pkl')
    joblib.dump(rf, 'random_forest_model.pkl')
    print("Models saved as 'linear_regression_model.pkl' and 'random_forest_model.pkl'.")
    print()

    # Load models
    print("Loading the models...")
    lr_loaded = joblib.load('linear_regression_model.pkl')
    rf_loaded = joblib.load('random_forest_model.pkl')
    print("Models loaded from 'linear_regression_model.pkl' and 'random_forest_model.pkl'.")
    print()

    # Make prediction with loaded models

    # y_pred_linear = lr.predict(X_test_scaled)
    print("Evaluating the loaded linear model...")
    y_pred_linear_loaded = lr_loaded.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_linear_loaded)
    r2 = r2_score(y_test, y_pred_linear_loaded)
    print("Evaluate the result of loaded linear regression:")
    print(f"Linear Regression MSE: {mse:.2f}")
    print(f"Linear Regression R²: {r2:.2f}")
    print()

    # rf_pred = rf.predict(X_test_scaled)
    print("Evaluating the loaded random forest model...")
    rf_pred_loaded = rf_loaded.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_pred_loaded)
    rf_r2 = r2_score(y_test, rf_pred_loaded)
    print("Evaluate the result of loaded random forest regression:")
    print(f"Random Forest MSE: {rf_mse:.2f}")
    print(f"Random Forest R²: {rf_r2:.2f}")
    print()

