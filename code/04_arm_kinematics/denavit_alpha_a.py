import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define origins for the frames again
origin_A = np.array([0, 0, 0])
origin_B = np.array([2, 0, 0])  # Correct: origin_B along X-axis of frame A (a_j distance)

# Redefine axis directions for the two frames (tilted Frame B to reflect alpha_j)
x_axis_A = np.array([1, 0, 0])
y_axis_A = np.array([0, 1, 0])
z_axis_A = np.array([0, 0, 1])

# Frame B now has a rotation around the X-axis to represent alpha_j
alpha_j = np.radians(30)  # Example angle of 30 degrees for alpha_j
x_axis_B = np.array([1, 0, 0])
y_axis_B = np.array([0, np.cos(alpha_j), np.sin(alpha_j)])
z_axis_B = np.array([0, -np.sin(alpha_j), np.cos(alpha_j)])

# Plot Frame A
ax.quiver(*origin_A, *x_axis_A, color='r', length=1)
ax.quiver(*origin_A, *y_axis_A, color='g', length=1)
ax.quiver(*origin_A, *z_axis_A, color='b', length=1)

# Plot Frame B
ax.quiver(*origin_B, *x_axis_B, color='r', length=1)
ax.quiver(*origin_B, *y_axis_B, color='g', length=1)
ax.quiver(*origin_B, *z_axis_B, color='b', length=1)

# Draw the line representing distance a_j (purely along X-axis)
ax.plot([origin_A[0], origin_B[0]], [origin_A[1], origin_B[1]], [origin_A[2], origin_B[2]], 'k--', label=r'$a_j$')

# Add a representation of alpha_j as an arc between Z_A and Z_B
arc_angle = np.linspace(0, alpha_j, 100)
arc_x = np.zeros_like(arc_angle)
arc_y = np.cos(arc_angle)
arc_z = np.sin(arc_angle)
ax.plot(arc_x, arc_y, arc_z, 'm', label=r'$\alpha_j$', linewidth=2)

# Labels for the coordinate systems
ax.text(*origin_A, r'$X_{j-1}$', color='red', fontsize=12)
ax.text(origin_A[0], origin_A[1] + 1, origin_A[2], r'$Y_{j-1}$', color='green', fontsize=12)
ax.text(origin_A[0], origin_A[1], origin_A[2] + 1, r'$Z_{j-1}$', color='blue', fontsize=12)

ax.text(*origin_B, r'$X_j$', color='red', fontsize=12)
ax.text(origin_B[0], origin_B[1] + y_axis_B[1], origin_B[2] + y_axis_B[2], r'$Y_j$', color='green', fontsize=12)
ax.text(origin_B[0], origin_B[1] + z_axis_B[1], origin_B[2] + z_axis_B[2], r'$Z_j$', color='blue', fontsize=12)

# Set plot limits and labels
ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
ax.set_zlim([-1, 3])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Title
ax.set_title('Denavit-Hartenberg Convention: Corrected Representation of $a_j$ and $\\alpha_j$')

plt.legend()
plt.show()
