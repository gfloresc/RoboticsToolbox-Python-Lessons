import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the vectors
a = np.array([4, -2, 1])
b = np.array([1, -1, 3])
c = a + b

# Define the start and end points for the vectors
starts = np.zeros((3, 3))  # All vectors start at the origin
ends = np.array([a, b, c])

# Create a 3D plot
fig = plt.figure()  # Initialize a new figure for the plot
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot to the figure

# Plot the vectors using quiver
ax.quiver(
    starts[:, 0], starts[:, 1], starts[:, 2],  # Starting points of the vectors (all at the origin)
    ends[:, 0], ends[:, 1], ends[:, 2],        # Endpoints of the vectors (defined by vectors a, b, and c)
    color=['r', 'g', 'b'],                     # Colors for each vector (red, green, blue)
    arrow_length_ratio=0.1                     # Adjust the arrow size relative to the vector length
)

# Set equal aspect ratio for better visualization
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Vector Plot")
ax.grid()
plt.show()
