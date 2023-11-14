import numpy as np
import matplotlib.pyplot as plt
from HyperPlanesUtil import PlanesIntersection
from matplotlib import style
style.use('ggplot')

# 2x -y -1 = 0
# y = 2x - 1
n1 = [2,-1,-1]

# 2x -2y -1 = 0
# y = (2x - 1)/2
n2 = [2,-2,-1]

w_base = .5
w_inc = .5
intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1[:-1],n2[:-1],n1[-1], n2[-1], w_base, w_inc)

# plotting

def base_line_equation(x):
    return 2*x-1

def incremental_line_equation(x):
    return (2*x - 1)/2

# Plot the two lines
x_values = np.linspace(-10, 10, 100)
plt.plot(x_values, base_line_equation(x_values), label='Base Model: y = 2x - 1')
plt.plot(x_values, incremental_line_equation(x_values), label='Incremental Model: y = x - .5 ', color='blue')
plt.scatter(intersection_point[0], intersection_point[1], label='Intersection Point', color='red',zorder=5)
# Set labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Base Model and Incremental Model are Intersecting')
plt.legend()
plt.show()

