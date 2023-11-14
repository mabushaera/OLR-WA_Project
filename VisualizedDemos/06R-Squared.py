import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Models.BatchRegression import BatchRegression
from Utils import Predictions

def r2(y_actual, y_predicted):
    y_bar = np.mean(y_actual)
    numerator = sum((y_actual - y_predicted)**2)
    dominator = sum((y_actual - y_bar)**2)
    r2 = 1 - (numerator/dominator)
    return r2

n_samples = 100
n_features = 1
noise = 0
random_state = 42
X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
plt.scatter(X,y, color='blue')

w = BatchRegression.linear_regression(X,y)
y_predicted = Predictions._compute_predictions_(X,w)

r2 = r2(y, y_predicted)
print(r2)
plt.plot(X, y_predicted, color='blue')
plt.show()

