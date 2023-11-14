from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from HyperPlanesUtil import PlanesIntersection

# Plane 1: 8x + 12y - 4z + 5 = 0
# Plane 2: 2x + 3y - z - 14 = 0
n1 = [8, 12, -4]
n2 = [2, 3, -1]
d1 = 5
d2 = -14

w_base = .5
w_inc = .5

intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1, n2, d1, d2, w_base, w_inc)
print(intersection_point)


# Plot the planes and the intersection point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Define the equations of the planes
def plane1(x, y):
    return (d1 + n1[0] * x + n1[1] * y) / n1[2]


def plane2(x, y):
    return (d2 + n2[0] * x + n2[1] * y) / n2[2]


# Generate x, y values for plotting
x_values = np.linspace(-5, 5, 100)
y_values = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_values, y_values)

# Calculate the corresponding z values for each plane
Z1 = plane1(X, Y)
Z2 = plane2(X, Y)

# Plot the planes and add labels for the legend
plane1_plot = ax.plot_surface(X, Y, Z1, alpha=0.5, color='blue')
plane2_plot = ax.plot_surface(X, Y, Z2, alpha=0.5, color='red')

# Plot the intersection point
intersection_x = intersection_point[0]
intersection_y = intersection_point[1]
intersection_z = intersection_point[2]
ax.scatter(intersection_x, intersection_y, intersection_z, color='green', s=100, label='weighted middle point')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(r'Weighted Middle Point $W_{base}=0.5$, $W_{inc}=0.5$', fontsize=9)

# # Create proxy artists for the planes to be used in the legend
# # Plane 1: 8x + 12y - 4z + 5 = 0
# # Plane 2: 2x + 3y - z - 14 = 0
# plane1_proxy = Line3DCollection([], color='blue',alpha=0.5, label=r'Base Model: $8x + 12y - 4z + 5 = 0$')
# plane2_proxy = Line3DCollection([], color='orange', alpha=0.5, label=r'Incremental Model: $2x + 3y - z - 14 = 0$')
#
# # Add the plots to the proxy artists
# plane1_proxy.set_array(np.array([]))
# plane2_proxy.set_array(np.array([]))

legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Base Model: 8x + 12y - 4z + 5 = 0'),
                   Line2D([0], [0], color='orange', lw=2, label='Incremental Model: 2x + 3y - z - 14 = 0'),
                   Line2D([0], [0], marker='o', color='green', label='weighted middle point', markersize=3,
                          markerfacecolor='green', linestyle='None')]

# Add the legend
ax.legend(handles=legend_elements)

# Show the plot
plt.show()
