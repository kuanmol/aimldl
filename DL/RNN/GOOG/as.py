import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.api.layers import Dense, LSTM, Dropout
from keras.api.models import Sequential

# ✅ Force TensorFlow to Use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU is being used by TensorFlow!")
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU found, using CPU instead.")

# ✅ Load Dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# ✅ Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scale = sc.fit_transform(training_set)

X_train, Y_train = [], []
for i in range(60, 1258):
    X_train.append(training_set_scale[i - 60:i, 0])
    Y_train.append(training_set_scale[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# ✅ Build LSTM Model
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

# ✅ Compile Model
regressor.compile(optimizer='adam', loss='mean_squared_error')

# ✅ Train Model on GPU
with tf.device('/GPU:0'):  # Force execution on GPU
    regressor.fit(X_train, Y_train, epochs=100, batch_size=32)

# ✅ Load Test Data
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# ✅ Prepare Data for Prediction
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = [inputs[i - 60:i, 0] for i in range(60, 80)]
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ✅ Predict Stock Prices
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# ✅ Visualize Results
plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
