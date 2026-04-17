import numpy as np

# Given parameters
a1 = 3.0  # Length of the first link
a2 = 2.0  # Length of the second link
theta1 = np.deg2rad(30)  # Angle of the first joint in radians
theta2 = np.deg2rad(45)  # Angle of the second joint in radians

# Define the sine and cosine terms for readability
c1 = np.cos(theta1)
s1 = np.sin(theta1)
c12 = np.cos(theta1 + theta2)
s12 = np.sin(theta1 + theta2)

# Define the positions of the points as numpy arrays
o0 = np.array([0, 0, 0])  # Origin
o1 = np.array([a1 * c1, a1 * s1, 0])  # First joint
o4 = np.array([a1 * c1 + a2 * c12, a1 * s1 + a2 * s12, -1])  # End-effector position (z component d3 - d4 = -1)

# Define the rotation axes as specified
z0 = np.array([0, 0, 1])  # Rotation axis for the first joint
z1 = np.array([0, 0, 1])  # Rotation axis for the second joint
z2 = np.array([0, 0, -1])  # Rotation axis for the third component
z3 = np.array([0, 0, -1])  # Rotation axis for the fourth component

# Compute the Jacobian columns for linear velocity
J_v1 = np.cross(z0, o4 - o0)  # Linear velocity contribution from the first joint
J_v2 = np.cross(z1, o4 - o1)  # Linear velocity contribution from the second joint
J_v3 = z2  # Contribution from the third axis
J_v4 = np.array([0, 0, 0])  # No linear contribution for the fourth axis

# Build the angular velocity part of the Jacobian
J_w1 = z0  # Angular velocity contribution from the first joint
J_w2 = z1  # Angular velocity contribution from the second joint
J_w3 = np.array([0, 0, 0])  # No angular contribution from the third axis
J_w4 = z3  # Angular velocity contribution from the fourth axis

# Construct the complete Jacobian (6x4 matrix)
J = np.hstack([
    np.array([J_v1, J_v2, J_v3, J_v4]),  # Linear velocity part (3x4)
    np.array([J_w1, J_w2, J_w3, J_w4])   # Angular velocity part (3x4)
])

# Display the calculated Jacobian
print("Calculated Jacobian:")
print(J.T)
