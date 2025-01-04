import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scale = sc.fit_transform(training_set)

X_train = []
Y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scale[i - 60:i, 0])
    Y_train.append(training_set_scale[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.api.layers import Dense
from keras.api.layers import LSTM
from keras.api.layers import Dropout
from keras.api.models import Sequential

regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units=1))

# compiling
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, Y_train, epochs=100, batch_size=32)  # trianing

# Getting the real stock price
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualize
plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google stock price prediction")
plt.xlabel("Time")
plt.ylabel("Google stock price")
plt.legend()
plt.show()

# Number of days to predict (e.g., 60 days = 2 months approx)
future_days = 60

# Start with the last 60 days of data from the combined dataset
last_60_days = inputs[-60:]
predicted_future_prices = []

for _ in range(future_days):
    # Prepare the input as a 3D array (same shape as X_test)
    input_data = last_60_days.reshape(1, 60, 1)

    # Predict the next price
    predicted_price = regressor.predict(input_data)

    # Inverse transform to get the actual price
    predicted_price_actual = sc.inverse_transform(predicted_price)
    predicted_future_prices.append(predicted_price_actual[0, 0])

    # Update the input window: Remove the oldest value, add the predicted value
    next_price_scaled = sc.transform(predicted_price)
    last_60_days = np.append(last_60_days[1:], next_price_scaled)

# Convert predictions to a NumPy array for visualization
predicted_future_prices = np.array(predicted_future_prices)

# Visualize the predictions
plt.plot(predicted_future_prices, color="blue", label="Predicted Future Prices")
plt.title("Future Stock Price Predictions")
plt.xlabel("Time (days)")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
