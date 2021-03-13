import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] 

dataset = pd.read_csv('housing.csv', header=None, delim_whitespace=True, names=labels)

RMSE, RSquared = [], []

poly_regression = LinearRegression()

for i in range(len(labels)-1):
    X = dataset.iloc[:, i:i+1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X_train)
    poly_regression.fit(X_poly, y_train)
    y_pred = poly_regression.predict(poly_reg.fit_transform(X_test))
    RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    RSquared.append(metrics.r2_score(y_test, y_pred))

print('Feature with the best fit line - ' + labels[RMSE.index(min(RMSE))] + ": percentage lower status of the population")

print(f"{labels[RMSE.index(min(RMSE))]} RMSE value: {min(RMSE)}")

print(f"{labels[RSquared.index(max(RSquared))]} RSquared value: {max(RSquared)}")


X = dataset.iloc[:, 12:13].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_regression.fit(X_poly, y_train)
y_pred = poly_regression.predict(poly_reg.fit_transform(X_test))

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Percentage lower status of the population vs House Price (Degree=2)')
plt.xlabel('Percentage lower status of the population')
plt.ylabel('House Price')
plt.savefig('Polynomial_Regression_Best_Fit_Line_Degree_2.png', dpi=400)
plt.show()

X = dataset.iloc[:, 12:13].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
poly_reg = PolynomialFeatures(degree = 20)
X_poly = poly_reg.fit_transform(X_train)
poly_regression.fit(X_poly, y_train)
y_pred = poly_regression.predict(poly_reg.fit_transform(X_test))

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Percentage lower status of the population vs House Price (Degree=20)')
plt.xlabel('Percentage lower status of the population')
plt.ylabel('House Price')
plt.savefig('Polynomial_Regression_Best_Fit_Line_Degree_20.png', dpi=400)
plt.show()