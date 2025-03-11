import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit page configuration
st.set_page_config(page_title="Apple Stock Price Prediction", layout="wide")

# Title and description
st.title("📈 Apple Stock Price Prediction")
st.write("""
Upload your CSV file and see the prediction along with the graph for actual vs predicted prices.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Extract relevant columns
    st.write("### Preview of your dataset")
    st.write(df.head())

    # Automatically identify required columns
    if 'close' in df.columns and 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'volume' in df.columns:
        # Selecting features and target
        X = df[['open', 'high', 'low', 'volume']]
        y = df['close']

        # Scale the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display metrics
        st.write("### Model Evaluation Metrics")
        st.write(f"📊 **Mean Squared Error:** {mse}")
        st.write(f"📈 **R2 Score:** {r2}")

        # Plot Actual vs Predicted Prices
        st.write("### Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label='Actual Price', color='blue')
        ax.plot(y_pred, label='Predicted Price', color='red')
        ax.set_title("Actual vs Predicted Stock Prices")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("❌ Error: Could not find the required columns ('open', 'high', 'low', 'volume', 'close') in your CSV file.")
else:
    st.info("📂 Please upload a CSV file to proceed.")
