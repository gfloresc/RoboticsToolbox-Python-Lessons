import numpy as np

# Define the values for theta2, theta4, theta5
theta2 = np.pi / 2  # pi/2 radians
theta4 = np.pi / 2  # pi/2 radians
theta5 = 0          # 0 radians

# Expression to check
left_side = -np.sin(theta2) * np.cos(theta4) * np.sin(theta5) + np.cos(theta2) * np.cos(theta5)

# Check if the solution is equal to 0
solution = np.isclose(left_side, 0)

# Print the result and the error
print(f"Computed value: {left_side}")
print(f"Is the solution equal to 0?: {solution}")
