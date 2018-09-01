import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()
lin_regressor.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures
pol_regressor = PolynomialFeatures(degree = 4)
X_poly = pol_regressor.fit_transform(X)
pol_regressor.fit(X_poly, y)
lin_regressor2 = LinearRegression()
lin_regressor2.fit(X_poly, y)

import matplotlib.pyplot as plt

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regressor2.predict(pol_regressor.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
