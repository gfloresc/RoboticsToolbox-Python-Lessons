import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define Denavit-Hartenberg parameters
a_j = 2  # Distance along the x-axis (a_j)
alpha_j = np.radians(30)  # Rotation around x-axis (alpha_j)
d_j = 1  # Distance along the z_{j-1} axis (d_j)

# Define origins for the frames
origin_A = np.array([0, 0, 0])  # Origin of frame A

# Define axis directions for Frame A
x_axis_A = np.array([1, 0, 0])
y_axis_A = np.array([0, 1, 0])
z_axis_A = np.array([0, 0, 1])

# Define the origin of Frame B based on DH parameters
# First move d_j along the z_{j-1} axis
origin_B_temp = origin_A + d_j * z_axis_A
# Then move a_j along the x-axis of Frame A
origin_B = origin_B_temp + a_j * x_axis_A

# Frame B now has a rotation around the X-axis to represent alpha_j
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

# Plot the line representing distance a_j (along X-axis of Frame A)
ax.plot([origin_A[0], origin_B_temp[0]], [origin_A[1], origin_B_temp[1]], [origin_A[2], origin_B_temp[2]], 'k--', label=r'$a_j$')

# Plot the line representing distance d_j (along Z_{j-1} axis)
ax.plot([origin_A[0], origin_B_temp[0]], [origin_A[1], origin_B_temp[1]], [origin_A[2], origin_B_temp[2]], 'r--', label=r'$d_j$')

# Extend x_j for visualization to show intersection
extended_x_j_pos = origin_B + 2 * x_axis_B  # Positive x_j extension
extended_x_j_neg = origin_B - 2 * x_axis_B  # Negative x_j extension

# Plotting the extended lines in both positive and negative x_j directions
ax.plot([origin_B[0], extended_x_j_pos[0]], [origin_B[1], extended_x_j_pos[1]], [origin_B[2], extended_x_j_pos[2]], 'r:')
ax.plot([origin_B[0], extended_x_j_neg[0]], [origin_B[1], extended_x_j_neg[1]], [origin_B[2], extended_x_j_neg[2]], 'r:')

# Extend z_{j-1} to show intersection with -x_j
extended_z_j_1 = origin_A + 2 * z_axis_A
ax.plot([origin_A[0], extended_z_j_1[0]], [origin_A[1], extended_z_j_1[1]], [origin_A[2], extended_z_j_1[2]], 'b:')

# Find intersection of extended -x_j and extended z_{j-1}
# The intersection occurs at origin_B but projected backwards along -x_j
intersection_point = origin_B - 2 * x_axis_B
ax.scatter(*intersection_point, color='black', s=50, label='Intersection Point (on $-X_j$ and $Z_{j-1}$)')

# Labels for the coordinate systems
ax.text(*origin_A, r'$X_{j-1}$', color='red', fontsize=12)
ax.text(origin_A[0], origin_A[1] + 1, origin_A[2], r'$Y_{j-1}$', color='green', fontsize=12)
ax.text(origin_A[0], origin_A[1], origin_A[2] + 1, r'$Z_{j-1}$', color='blue', fontsize=12)

ax.text(*origin_B, r'$X_j$', color='red', fontsize=12)
ax.text(origin_B[0], origin_B[1] + y_axis_B[1], origin_B[2] + y_axis_B[2], r'$Y_j$', color='green', fontsize=12)
ax.text(origin_B[0], origin_B[1] + z_axis_B[1], origin_B[2] + z_axis_B[2], r'$Z_j$', color='blue', fontsize=12)

# Set plot limits and labels
ax.set_xlim([-3, 4])
ax.set_ylim([-1, 3])
ax.set_zlim([-1, 3])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Title
ax.set_title('Denavit-Hartenberg Convention')

plt.legend()
plt.show()
