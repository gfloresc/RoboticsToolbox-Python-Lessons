import numpy as np

# Given parameters
a1 = 1.0  # Length of the first link
a2 = 1.0  # Length of the second link
theta1 = np.deg2rad(45)  # Angle of the first joint in radians
theta2 = np.deg2rad(45)  # Angle of the second joint in radians

# Define the positions of the points
o0 = np.array([0, 0, 0])
o1 = np.array([a1 * np.cos(theta1), a1 * np.sin(theta1), 0])
o2 = np.array([a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2),
               a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2), 0])

z0 = np.array([0, 0, 1])  # Rotation axis of the first joint
z1 = np.array([0, 0, 1])  # Rotation axis of the second joint

# Compute the Jacobian columns for linear velocity
J_v1 = np.cross(z0, o2 - o0)  # Linear velocity with respect to theta1
J_v2 = np.cross(z1, o2 - o1)  # Linear velocity with respect to theta2

# Build the angular velocity part of the Jacobian
J_w1 = z0  # Angular velocity with respect to theta1
J_w2 = z1  # Angular velocity with respect to theta2

# Construct the complete Jacobian (6x2 matrix)
J = np.hstack([  # Stack rows horizontally
    np.array([J_v1.T, J_v2.T]),  # Linear velocity part (3x2)
    np.array([J_w1.T, J_w2.T])   # Angular velocity part (3x2)
])

# Display the corrected Jacobian
print("Corrected Jacobian:")
print(J_v1, J_v2)  # Transpose to get the 6x2 matrix
