import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, [2]].values


from sklearn.preprocessing import StandardScaler
sX = StandardScaler()
sy = StandardScaler()
X = sX.fit_transform(X)
y = sy.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

y_predict = regressor.predict(6.5)
y_predict = sy.inverse_transform(y_predict)

import matplotlib.pyplot as plt

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
