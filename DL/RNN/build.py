import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scale = sc.fit_transform(training_set)
print(training_set_scale)

X_train = []
Y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scale[i - 60:i, 0])
    Y_train.append(training_set_scale[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
