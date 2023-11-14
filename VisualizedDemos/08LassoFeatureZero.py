import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Generate some synthetic data with two features
np.random.seed(0)
n_samples = 100
X = np.random.randn(n_samples, 2)
# Let's make the second feature irrelevant by setting it to constant values
X[:, 1] = 1.0
y = 2 * X[:, 0] + 1 + np.random.randn(n_samples) * 0.5

# Fit a Lasso regression model with different alpha values
alphas = [0.1, 1.0, 10.0]
coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)

# Plot the coefficients of the two features for different alpha values
plt.figure(figsize=(8, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficients')
plt.title('Lasso Regularization - Coefficients vs. Alpha')
plt.legend(['Feature 1', 'Feature 2'])
plt.grid()
plt.show()
