# Import libraries  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  

# Load historical data  
data = {  
    'Year': [2022, 2023, 2024],  
    'Sales': [50.1498, 307.24746, 303.32267],  
    'Operating_Profit': [2.46703, 17.04941, 20.11142]  
}  
df = pd.DataFrame(data).set_index('Year')  

# Normalize data  
scaler = MinMaxScaler(feature_range=(0, 1))  
data_scaled = scaler.fit_transform(df)  

# Prepare sequences for LSTM  
X, y = [], []  
for i in range(len(data_scaled) - 1):  
    X.append(data_scaled[i])  
    y.append(data_scaled[i+1])  
X, y = np.array(X), np.array(y)  

# Reshape input to [samples, time_steps, features]  
X = X.reshape((X.shape[0], 1, X.shape[1]))  

# Build LSTM model  
model = Sequential()  
model.add(LSTM(32, activation='relu', input_shape=(1, 2)))  
model.add(Dense(2))  
model.compile(optimizer='adam', loss='mse')  

# Train model  
model.fit(X, y, epochs=500, verbose=0)  

# Forecast 2025–2028  
future_predictions = []  
last_input = data_scaled[-1].reshape(1, 1, 2)  # Start from 2024  

for _ in range(4):  
    pred = model.predict(last_input)  
    future_predictions.append(pred[0])  
    last_input = pred.reshape(1, 1, 2)  

# Inverse transform predictions  
predicted_values = scaler.inverse_transform(np.array(future_predictions))  

# Create results DataFrame  
forecast_df = pd.DataFrame({  
    'Year': [2025, 2026, 2027, 2028],  
    'Predicted_Sales': predicted_values[:, 0],  
    'Predicted_Profit': predicted_values[:, 1]  
}).set_index('Year')  

# Visualization  
plt.figure(figsize=(12, 6))  
plt.plot(df['Sales'], marker='o', label='Historical Sales')  
plt.plot(df['Operating_Profit'], marker='o', label='Historical Profit')  
plt.plot(forecast_df['Predicted_Sales'], marker='o', linestyle='--', label='Predicted Sales')  
plt.plot(forecast_df['Predicted_Profit'], marker='o', linestyle='--', label='Predicted Profit')  
plt.title('Celine Korea: Sales & Profit Forecast (2022–2028)')  
plt.xlabel('Year'), plt.ylabel('Billion KRW'), plt.legend(), plt.grid(True)  
plt.show()  

print("Forecast Results (2025–2028):")  
print(forecast_df.round(2))  
