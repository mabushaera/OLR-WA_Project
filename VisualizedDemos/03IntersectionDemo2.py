import numpy as np
import matplotlib.pyplot as plt
from HyperPlanesUtil import PlanesIntersection
from matplotlib.lines import Line2D

# Define the coefficients of the two planes
n1 = [1, 1, 1]  # Coefficients of the first plane
d1 = 1  # Constant for the first plane
n2 = [-1, 3, 1]  # Coefficients of the second plane
d2 = 1  # Constant for the second plane

w_base = 0.5
w_inc = 0.5

# Find the intersection point
intersection = PlanesIntersection.find_intersection_hyperplaneND(n1, n2, d1, d2, w_base, w_inc)

# Create a figure and 3D axis with a white background
fig = plt.figure(facecolor='white')  # Set the background color to white
ax = fig.add_subplot(111, projection='3d')

# Define a grid of points
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Calculate the corresponding Z values for the two planes
Z1 = (-n1[0] * X - n1[1] * Y - d1) / n1[2]
Z2 = (-n2[0] * X - n2[1] * Y - d2) / n2[2]

# Plot the two planes with labels
surf1 = ax.plot_surface(X, Y, Z1, alpha=0.5, color='blue')
surf2 = ax.plot_surface(X, Y, Z2, alpha=0.5, color='orange')

# Plot the intersection point as a red dot with label
ax.scatter([intersection[0]], [intersection[1]], [intersection[2]], color='red', label='Intersection Point')

'''
# Define the coefficients of the two planes
n1 = [1, 1, 1]  # Coefficients of the first plane
d1 = 1  # Constant for the first plane
n2 = [-1, 3, 1]  # Coefficients of the second plane
d2 = 1  # Constant for the second plane
'''

# Create a custom legend
legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Base Model: y = -x -z -1'),
                   Line2D([0], [0], color='orange', lw=2, label='Incremental Model: y = (x - z - 1) / 3'),
                   Line2D([0], [0], marker='o', color='red', label='Intersection Point', markersize=5, markerfacecolor='red', linestyle='None')]

# Add the legend
ax.legend(handles=legend_elements, loc='upper left')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Base Model and Incremental Model are intersecting')

plt.show()
