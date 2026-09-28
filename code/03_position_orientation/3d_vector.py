import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Original vector
vector = np.array([1, 1, 1])

# Rotation matrix around Z-axis by 45 degrees
angle = np.radians(45)
rotation_matrix_z = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
])

# Rotated vector
rotated_vector = rotation_matrix_z @ vector

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot original vector
ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='r', label='Original Vector')

# Plot rotated vector
ax.quiver(0, 0, 0, rotated_vector[0], rotated_vector[1], rotated_vector[2], color='b', label='Rotated Vector')

# Set plot limits
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_zlim([0, 2])

# Labeling
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
