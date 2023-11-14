"""
This python script represents the implementation of the Batch Gradient Descent
Please note that the batch gradient descent performance yield to the exact same
performance of Batch Regression (Pseudo-Inverse) which is used as the benchmark
for other models performance in our experiments.
"""

import numpy as np


def batch_gradient_descent(X, y_true, epochs, learning_rate):
    total_samples = X.shape[0]
    number_of_features = X.shape[1]
    w = np.zeros(number_of_features)
    b = 0

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        y_predicted = np.dot(w, X.T) + b

        w_grad = -(2 / total_samples) * (X.T.dot(y_true - y_predicted))
        b_grad = -(2 / total_samples) * np.sum(y_true - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.mean(np.square(y_true - y_predicted))  # MSE (Mean Squared Error)
        if i % 50 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, b, cost_list, epoch_list


