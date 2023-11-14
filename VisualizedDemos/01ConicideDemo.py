import numpy as np
import matplotlib.pyplot as plt
from HyperPlanesUtil import PlanesIntersection
from matplotlib import style
style.use('ggplot')

# x + 2y + 3 = 0
# x = 1, y = 2, d=3
# y = (-x -3)/2
n1 = [1,2]
d1 = 3

# 2x + 4y + 6 = 0
# x = 2, y = 4 d = 6
# y = (-2x -6)/4
n2 = [2,4,6]
d2 = 6
res = PlanesIntersection.isCoincident(n1,n2, d1, d2)
print(res)

# plotting
def base_line_equation(x):
    return (-1*x - 3)/2

def incremental_line_equation(x):
    return (-2*x - 6)/4

# Generate x values for plotting
x_values = np.linspace(-10, 10, 100)
# Plot the two lines
plt.plot(x_values, base_line_equation(x_values), label='base model: x + 2y + 3 = 0', color='blue')
plt.plot(x_values, incremental_line_equation(x_values), label='incremental model: 2x + 4y + 6 = 0', color='orange')


# Set labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Base Model and Incremental Model are Conicide')
plt.legend()
plt.show()