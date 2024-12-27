import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# line-regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(x, y, color="red")
plt.plot(x,lin_reg.predict(x), color="blue")
plt.title("truth or bluff(Linear Regreesion)")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

plt.scatter(x, y, color="red")
plt.plot(x,lin_reg_2.predict(X_poly), color="blue")
plt.title("truth or bluff(polynomial Regreesion)")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()






















