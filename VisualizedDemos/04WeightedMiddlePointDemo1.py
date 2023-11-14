import numpy as np
from HyperPlanesUtil import PlanesIntersection

# Line 1: y = 2x + 3 -> 2x - y + 3 = 0
# Line 2: y = 2x - 4 -> 2x -  y - 4 = 0
n1 = [2, -1]
n2 = [2, -1]
d1 = 3
d2 = -4

w_base = .5
w_inc = .5

intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1, n2, d1, d2, w_base, w_inc)
print(intersection_point)

import matplotlib.pyplot as plt


# Define the equations of the two lines
# y = 2x + 3 -> 2x - y + 3 = 0
def line1(x):
    return 2 * x + 3


# Line 2: y = 2x - 4 -> 2x -  y - 4 = 0
def line2(x):
    return 2 * x - 4

# Generate x values for plotting
x_values = np.linspace(-10, 10, 100)

# Plot the two lines
plt.plot(x_values, line1(x_values), label='Base Model: 2x - y + 3 = 0')
plt.plot(x_values, line2(x_values), label='Incremental Model: 2x - y - 4 = 0')
#
# Plot the intersection point
intersection_x = intersection_point[0]
intersection_y = intersection_point[1]
plt.plot(intersection_x, intersection_y, 'ro', label='weighted middle point', color='green')

# Set labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Weighted Middle Point $W_{base}=0.1$, $W_{inc}=0.9$')
plt.legend()

# Show the plot
plt.grid()
plt.show()
