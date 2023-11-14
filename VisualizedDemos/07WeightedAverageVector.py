# line1 = 2x -y +3
# line2 = -x +3y +2
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
import numpy as np

norm_vector1 = np.array([2, -1])
norm_vector2 = np.array([-1, 3])


weighted_average1 = (np.dot(.5 ,norm_vector1) + np.dot(.5 , norm_vector2)) / (.5+.5)
weighted_average2 =  -1*(np.dot(.5 , norm_vector1) + np.dot(.5 , norm_vector2)) / (.5+.5)

# Plot the vectors
ax.quiver(0, 0, norm_vector1[0], norm_vector1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Norm Vector 1')
ax.quiver(0, 0, norm_vector2[0], norm_vector2[1], angles='xy', scale_units='xy', scale=1, color='red', label='Norm Vector 2')
ax.quiver(0, 0, weighted_average1[0], weighted_average1[1], angles='xy', scale_units='xy', scale=1, color='green', label='Weighted Average1')
ax.quiver(0, 0, weighted_average2[0], weighted_average2[1], angles='xy', scale_units='xy', scale=1, color='green', label='Weighted Average2')

# Set axis limits
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 4)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Add a legend
ax.legend()

# Set grid
ax.grid(True)
plt.show()