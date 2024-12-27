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

poly_reg = PolynomialFeatures(degree=4)
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

#higher resolution
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#new result with linear
lin_reg.predict([[6.5]])

#new result with polynomial
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))




















