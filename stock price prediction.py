#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import pandas as pd

# Path to your ZIP file
zip_path = r"C:\Users\Achuth Kaja\Downloads\archive.zip"
extract_path = r"C:\Users\Achuth Kaja\Downloads\archive"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Get the list of extracted files
files = [f for f in os.listdir(extract_path) if f.endswith('.csv')]

# Loop through each CSV file and clean it
for file in files:
    file_path = os.path.join(extract_path, file)
    df = pd.read_csv(file_path)
    
    # Basic Data Cleaning
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    df.dropna(inplace=True)  # Drop rows with missing values
    
    # Convert column names to lowercase and strip whitespaces
    df.columns = df.columns.str.lower().str.strip()
    
    # Convert date columns to datetime if any
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Save the cleaned file
    cleaned_file_path = os.path.join(extract_path, f"cleaned_{file}")
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned file saved: {cleaned_file_path}")

print("Data cleaning completed.")


# In[3]:


import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os

# Path to your cleaned ZIP file
zip_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL.zip"
extract_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Find the extracted CSV file
files = [f for f in os.listdir(extract_path) if f.endswith('.csv')]
if not files:
    raise FileNotFoundError("No CSV file found in the ZIP archive.")

csv_file_path = os.path.join(extract_path, files[0])

# Load the dataset
df = pd.read_csv(csv_file_path)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Feature selection
data = df[['close']]  # Assuming 'close' is the target variable

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM model
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 60  # Sequence length for time series data
x, y = create_sequences(scaled_data, seq_length)

# Reshape data for LSTM
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build LSTM Model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x, y, batch_size=32, epochs=10)

# Predict future stock prices
predicted_prices = model.predict(x)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(df.index[seq_length:], data[seq_length:], label='Actual Prices')
plt.plot(df.index[seq_length:], predicted_prices, label='Predicted Prices')
plt.title('Stock Price Prediction for AAPL')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[5]:


pip install streamlit


# In[7]:


pip install prophet


# In[9]:


pip install fbprophet


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import zipfile
import os

# Extract ZIP file
zip_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL.zip"
extract_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

files = [f for f in os.listdir(extract_path) if f.endswith('.csv')]
csv_file_path = os.path.join(extract_path, files[0])

# Load data
df = pd.read_csv(csv_file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data = df[['close']]
scaled_data = scaler.fit_transform(data)

# Prepare data for LSTM
seq_length = 60
x, y = [], []
for i in range(len(scaled_data) - seq_length):
    x.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, batch_size=32, epochs=10)

# Predict next 30 days using LSTM
future_input = scaled_data[-seq_length:]
future_input = np.reshape(future_input, (1, seq_length, 1))

predicted_prices = []
for _ in range(30):
    pred = model.predict(future_input)
    pred = np.reshape(pred, (1, 1, 1))
    predicted_prices.append(pred[0][0][0])
    future_input = np.append(future_input[:, 1:, :], pred, axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(df['close'], label='Actual Prices')
plt.plot(pd.date_range(start=df.index[-1], periods=30, freq='B'), predicted_prices, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction (AAPL)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import zipfile
import os

# Extract ZIP file
zip_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL.zip"
extract_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

files = [f for f in os.listdir(extract_path) if f.endswith('.csv')]
csv_file_path = os.path.join(extract_path, files[0])

# Load data
df = pd.read_csv(csv_file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data = df[['close']]
scaled_data = scaler.fit_transform(data)

# Prepare data for LSTM
seq_length = 60
x, y = [], []
for i in range(len(scaled_data) - seq_length):
    x.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, batch_size=32, epochs=10)

# Predict next 2 years using LSTM
future_input = scaled_data[-seq_length:]
future_input = np.reshape(future_input, (1, seq_length, 1))

predicted_prices = []
future_days = 730  # Approx 2 years
for _ in range(future_days):
    pred = model.predict(future_input)
    pred = np.reshape(pred, (1, 1, 1))
    predicted_prices.append(pred[0][0][0])
    future_input = np.append(future_input[:, 1:, :], pred, axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Create a date range for 2 years
last_date = df.index[-1]
date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
predicted_df = pd.DataFrame(predicted_prices, index=date_range, columns=['Predicted'])

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(df['close'], label='Actual Prices')
plt.plot(predicted_df['Predicted'], label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.title('Stock Price Prediction for Next 2 Years (AAPL)')
plt.legend()
plt.show()

# Display future stock prices
display(predicted_df)


# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import zipfile
import os

# Extract ZIP file
zip_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL.zip"
extract_path = r"C:\Users\Achuth Kaja\Downloads\archive\cleaned_AAPL"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

files = [f for f in os.listdir(extract_path) if f.endswith('.csv')]
csv_file_path = os.path.join(extract_path, files[0])

# Load data
df = pd.read_csv(csv_file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data = df[['close']]
scaled_data = scaler.fit_transform(data)

# Prepare data for LSTM
seq_length = 60
x, y = [], []
for i in range(len(scaled_data) - seq_length):
    x.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, batch_size=32, epochs=10)

# Predict next 100 days using LSTM
future_input = scaled_data[-seq_length:]
future_input = np.reshape(future_input, (1, seq_length, 1))

predicted_prices = []
future_days = 100  # Predict for 100 days
for _ in range(future_days):
    pred = model.predict(future_input)
    pred = np.reshape(pred, (1, 1, 1))
    predicted_prices.append(pred[0][0][0])
    future_input = np.append(future_input[:, 1:, :], pred, axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Create a date range for 100 days
last_date = df.index[-1]
date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
predicted_df = pd.DataFrame(predicted_prices, index=date_range, columns=['Predicted'])

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(df['close'], label='Actual Prices')
plt.plot(predicted_df['Predicted'], label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.title('Stock Price Prediction for Next 100 Days (AAPL)')
plt.legend()
plt.show()

# Display future stock prices
display(predicted_df)


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data (assuming df is already loaded and preprocessed)
# df = pd.read_csv('your_data.csv')  # Uncomment and load your data if not already loaded
# df['date'] = pd.to_datetime(df['date'])
# df.set_index('date', inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data = df[['close']]
scaled_data = scaler.fit_transform(data)  # Ensure scaled_data is defined

# Prepare data for LSTM
seq_length = 60  # Ensure seq_length is defined
x, y = [], []
for i in range(len(scaled_data) - seq_length):
    x.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, batch_size=32, epochs=10)  # Train the model

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

# Prepare the test data for LSTM
x_test, y_test = [], []
for i in range(len(test_data) - seq_length):
    x_test.append(test_data[i:i+seq_length])
    y_test.append(test_data[i+seq_length])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict on the test data
predicted_test = model.predict(x_test)  # Now model is defined
predicted_test = scaler.inverse_transform(predicted_test)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the error metrics
mae = mean_absolute_error(y_test, predicted_test)
mse = mean_squared_error(y_test, predicted_test)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the actual vs predicted prices for the test set
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual Prices')
plt.plot(predicted_test, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Close Price (USD)')
plt.title('Actual vs Predicted Prices (Test Set)')
plt.legend()
plt.show()


# In[20]:


import numpy as np

# Define the Mean Absolute Percentage Error (MAPE) function
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, predicted_test)

# Calculate accuracy in percentage
accuracy = 100 - mape

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")


# In[ ]:




