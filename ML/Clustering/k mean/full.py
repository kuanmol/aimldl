import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

print(X)