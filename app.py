
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the best model (replace 'best_model.pkl' with the path to your model file)
model = joblib.load('best_model.pkl')

def main():
    st.title("Sales Forecasting Prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data.head())

        # Make predictions
        predictions = model.predict(data)
        data['Predicted Sales'] = predictions

        # Display the prediction results
        st.write("Prediction Results:")
        st.write(data.head())

        # Visualization: Sales Forecast Trend
        st.subheader("Sales Forecast Trend")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=data.index, y='Predicted Sales', data=data)
        plt.title('Sales Forecast Trend')
        st.pyplot(plt)

        # Visualization: Sales Forecast Line Chart
        st.subheader("Sales Forecast Line Chart")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=data.index, y='Predicted Sales', data=data, marker='o')
        plt.title('Sales Forecast Line Chart')
        st.pyplot(plt)

        # Visualization: Sales Forecast Scatter Plot
        st.subheader("Sales Forecast Scatter Plot")
        plt.figure(figsize=(10, 6))
        plt.scatter(data.index, data['Predicted Sales'], color='blue')
        plt.title('Sales Forecast Scatter Plot')
        plt.xlabel('Index')
        plt.ylabel('Predicted Sales')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
