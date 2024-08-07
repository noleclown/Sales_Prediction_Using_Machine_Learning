# -*- coding: utf-8 -*-
"""Sales_Forcasting.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11K0D6dLEhBEFjmoQ5rbGrHBoC51Ia9w2
"""

import pandas as pd

# Load the dataset
file_path = "/content/Train.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check data types of each column
print(data.dtypes)

# Check for missing values in each column
print(data.isnull().sum())

# Display basic statistics for numerical columns
print(data.describe())

# Display unique values in categorical columns
for column in data.select_dtypes(include=['object']).columns:
    print(f"Unique values in {column}: {data[column].unique()}")

# Handling missing values
data['Item_Weight'].fillna(data['Item_Weight'].median(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)

# Standardizing 'Item_Fat_Content'
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

# Encoding categorical variables
# One-hot encoding for columns with more than two categories
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Type'])

# Label encoding for columns with two categories or more complex cases
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet_Identifier'] = le.fit_transform(data['Outlet_Identifier'])
data['Item_Identifier'] = le.fit_transform(data['Item_Identifier'])

# Feature Engineering
# Creating a new feature 'Outlet_Age' based on 'Outlet_Establishment_Year'
data['Outlet_Age'] = 2024 - data['Outlet_Establishment_Year']

# Drop the original 'Outlet_Establishment_Year' if it's no longer needed
data.drop('Outlet_Establishment_Year', axis=1, inplace=True)

# Display the first few rows of the processed data
data.head()

from sklearn.model_selection import train_test_split

# Define the target variable and features
X = data.drop(columns=['Item_Outlet_Sales'])
y = data['Item_Outlet_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the splits to confirm
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr.predict(X_test)

# Evaluate the model
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression RMSE: {rmse_lr}")
print(f"Linear Regression R²: {r2_lr}")

from sklearn.tree import DecisionTreeRegressor

# Initialize and train the Decision Tree model
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt.predict(X_test)

# Evaluate the model
rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree RMSE: {rmse_dt}")
print(f"Decision Tree R²: {r2_dt}")

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rmse_rf}")
print(f"Random Forest R²: {r2_rf}")

import xgboost as xgb

# Initialize and train the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost RMSE: {rmse_xgb}")
print(f"XGBoost R²: {r2_xgb}")

!pip install catboost
from catboost import CatBoostRegressor

# Initialize and train the CatBoost model
catboost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_catboost = catboost_model.predict(X_test)

# Evaluate the model
rmse_catboost = mean_squared_error(y_test, y_pred_catboost, squared=False)
r2_catboost = r2_score(y_test, y_pred_catboost)

print(f"CatBoost RMSE: {rmse_catboost}")
print(f"CatBoost R²: {r2_catboost}")

import lightgbm as lgb

# Initialize and train the LightGBM model
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lgb = lgb_model.predict(X_test)

# Evaluate the model
rmse_lgb = mean_squared_error(y_test, y_pred_lgb, squared=False)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f"LightGBM RMSE: {rmse_lgb}")
print(f"LightGBM R²: {r2_lgb}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame with the performance metrics
data = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost', 'LightGBM'],
    'RMSE': [1069.715, 1521.816, 1097.156, 1145.891, 1024.685, 1056.363],
    'R²': [0.579, 0.148, 0.557, 0.517, 0.614, 0.589]
}

df = pd.DataFrame(data)

# Set the style of the visualization
sns.set(style="whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plot RMSE
sns.barplot(x='Model', y='RMSE', data=df, ax=axes[0], palette='viridis')
axes[0].set_title('RMSE of Different Models')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Plot R²
sns.barplot(x='Model', y='R²', data=df, ax=axes[1], palette='viridis')
axes[1].set_title('R² of Different Models')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# Display the plots
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Hyperparameter grids for different models
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    'CatBoost': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [6, 8, 10]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 63, 127]
    }
}

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Initialize models
models = {
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'LightGBM': LGBMRegressor()
}

# Store best models and results
best_models = {}
results = {}

# Perform grid search for each model
for model_name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    results[model_name] = {
        'Best Parameters': grid_search.best_params_,
        'Best RMSE': np.sqrt(-grid_search.best_score_)
    }

for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[model_name].update({'Test RMSE': rmse, 'Test R²': r2})

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results).T
print(results_df)

# Plot RMSE and R² for each model after hyperparameter tuning
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plot RMSE
sns.barplot(x=results_df.index, y='Test RMSE', data=results_df, ax=axes[0], palette='viridis')
axes[0].set_title('Test RMSE of Tuned Models')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Plot R²
sns.barplot(x=results_df.index, y='Test R²', data=results_df, ax=axes[1], palette='viridis')
axes[1].set_title('Test R² of Tuned Models')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Retrain the CatBoost model with the best parameters on the entire training data
best_catboost_model = CatBoostRegressor(
    depth=6,
    iterations=100,
    learning_rate=0.1,
    verbose=0
)
best_catboost_model.fit(X_train, y_train)

# Predict on the test set
y_pred_final = best_catboost_model.predict(X_test)

# Calculate RMSE and R²
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_r2 = r2_score(y_test, y_pred_final)

print(f'Final Test RMSE: {final_rmse}')
print(f'Final Test R²: {final_r2}')

import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances
importances = best_catboost_model.get_feature_importance()
feature_names = X_train.columns

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()

import joblib

# Save the model
joblib.dump(best_catboost_model, 'best_catboost_model.pkl')

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predict on the test set
y_pred_final = best_catboost_model.predict(X_test)

# Calculate RMSE and R²
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_r2 = r2_score(y_test, y_pred_final)

print(f'Final Test RMSE: {final_rmse}')
print(f'Final Test R²: {final_r2}')

import matplotlib.pyplot as plt
import seaborn as sns

# Define performance metrics for each model
models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost', 'LightGBM']
rmse_values = [1069.715, 1521.816, 1097.156, 1145.891, 1024.685, 1056.363]
r2_values = [0.579, 0.148, 0.557, 0.517, 0.614, 0.589]

# Create a DataFrame for plotting
import pandas as pd

df_metrics = pd.DataFrame({
    'Model': models,
    'RMSE': rmse_values,
    'R²': r2_values
})

# Plot RMSE and R² values
fig, ax1 = plt.subplots(figsize=(12, 6))

# RMSE plot
color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE', color=color)
sns.barplot(x='Model', y='RMSE', data=df_metrics, ax=ax1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# R² plot
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('R²', color=color)
sns.lineplot(x='Model', y='R²', data=df_metrics, ax=ax2, marker='o', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Title and layout
plt.title('Model Performance Metrics')
fig.tight_layout()
plt.show()

import numpy as np

# Sample data for illustration
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
actual_sales = np.random.randint(1000, 5000, size=len(dates))
predicted_sales = actual_sales + np.random.normal(0, 500, size=len(dates))

# Creating a DataFrame for plotting
df_sales = pd.DataFrame({
    'Date': dates,
    'Actual Sales': actual_sales,
    'Predicted Sales': predicted_sales
})

# 1. Trend Chart Forecast
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Actual Sales', data=df_sales, label='Actual Sales', color='blue')
sns.lineplot(x='Date', y='Predicted Sales', data=df_sales, label='Predicted Sales', color='red', linestyle='--')
plt.title('Sales Trend Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 2. Line Chart Forecast
plt.figure(figsize=(12, 6))
plt.plot(df_sales['Date'], df_sales['Actual Sales'], label='Actual Sales', color='blue')
plt.plot(df_sales['Date'], df_sales['Predicted Sales'], label='Predicted Sales', color='red', linestyle='--')
plt.title('Line Chart Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 3. Scatter Plot Forecast
plt.figure(figsize=(12, 6))
plt.scatter(df_sales['Actual Sales'], df_sales['Predicted Sales'], alpha=0.7)
plt.plot([df_sales['Actual Sales'].min(), df_sales['Actual Sales'].max()],
         [df_sales['Actual Sales'].min(), df_sales['Actual Sales'].max()], color='red', linestyle='--')
plt.title('Scatter Plot Forecast')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

!pip install streamlit

!pip install pyngrok

# Commented out IPython magic to ensure Python compatibility.
# # Create a file named 'app.py' in Google Colab
# %%writefile app.py
# 
# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# 
# # Load the best model (replace 'best_model.pkl' with the path to your model file)
# model = joblib.load('best_model.pkl')
# 
# def main():
#     st.title("Sales Forecasting Prediction")
# 
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# 
#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)
# 
#         # Display the uploaded data
#         st.write("Uploaded Data:")
#         st.write(data.head())
# 
#         # Make predictions
#         predictions = model.predict(data)
#         data['Predicted Sales'] = predictions
# 
#         # Display the prediction results
#         st.write("Prediction Results:")
#         st.write(data.head())
# 
#         # Visualization: Sales Forecast Trend
#         st.subheader("Sales Forecast Trend")
#         plt.figure(figsize=(10, 6))
#         sns.lineplot(x=data.index, y='Predicted Sales', data=data)
#         plt.title('Sales Forecast Trend')
#         st.pyplot(plt)
# 
#         # Visualization: Sales Forecast Line Chart
#         st.subheader("Sales Forecast Line Chart")
#         plt.figure(figsize=(10, 6))
#         sns.lineplot(x=data.index, y='Predicted Sales', data=data, marker='o')
#         plt.title('Sales Forecast Line Chart')
#         st.pyplot(plt)
# 
#         # Visualization: Sales Forecast Scatter Plot
#         st.subheader("Sales Forecast Scatter Plot")
#         plt.figure(figsize=(10, 6))
#         plt.scatter(data.index, data['Predicted Sales'], color='blue')
#         plt.title('Sales Forecast Scatter Plot')
#         plt.xlabel('Index')
#         plt.ylabel('Predicted Sales')
#         st.pyplot(plt)
# 
# if __name__ == "__main__":
#     main()
#

from pyngrok import ngrok
import os

# Start Streamlit
!streamlit run app.py &

# Create a tunnel to the Streamlit app
public_url = ngrok.connect(port='8501')
public_url