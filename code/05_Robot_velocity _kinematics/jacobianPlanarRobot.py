import numpy as np

# Given parameters
a1 = 1.0  # Length of the first link
a2 = 1.0  # Length of the second link
theta1 = np.deg2rad(20)  # Angle of the first joint in radians
theta2 = np.deg2rad(20)  # Angle of the second joint in radians

# Define the positions of the points
o0 = np.array([0, 0, 0])
o1 = np.array([a1 * np.cos(theta1), a1 * np.sin(theta1), 0])
o2 = np.array([
    a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2),
    a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2),
    0
])

# Rotation axes of the joints
z0 = np.array([0, 0, 1])
z1 = np.array([0, 0, 1])

# Compute the Jacobian columns for linear velocity
J_v1 = np.cross(z0, o2 - o0)  # Linear velocity with respect to theta1
J_v2 = np.cross(z1, o2 - o1)  # Linear velocity with respect to theta2

# Build the angular velocity part of the Jacobian
J_w1 = z0  # Angular velocity with respect to theta1
J_w2 = z1  # Angular velocity with respect to theta2

# Construct the complete Jacobian (6x2 matrix)
J = np.array([
    [J_v1[0], J_v2[0]],
    [J_v1[1], J_v2[1]],
    [J_v1[2], J_v2[2]],
    [J_w1[0], J_w2[0]],
    [J_w1[1], J_w2[1]],
    [J_w1[2], J_w2[2]]
])

# Display the Jacobian
print("Jacobian J:")
print(J)