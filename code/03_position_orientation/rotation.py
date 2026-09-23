import numpy as np
import matplotlib.pyplot as plt

def rotate_vector(vector, theta):
    """Rotates a 2D vector by angle theta (in radians)."""
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])  # Rotation Matrix
    return R @ vector  # Matrix-vector multiplication

# Define the original vector
v = np.array([-2, -1])  # Example: unit vector along x-axis

# Rotation angle in degrees and radians
theta_deg = -45  # Change this to test different angles
theta_rad = np.radians(theta_deg)

# Rotate the vector
v_rotated = rotate_vector(v, theta_rad)

# Plot the original and rotated vectors
fig, ax = plt.subplots()
ax.quiver(0, 0, v[0], v[1], color='r', angles='xy', scale_units='xy', scale=1, label="Original Vector")
ax.quiver(0, 0, v_rotated[0], v_rotated[1], color='b', angles='xy', scale_units='xy', scale=1, label="Rotated Vector")

# Formatting plot
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks(np.arange(-5, 5, 0.5))
ax.set_yticks(np.arange(-5, 5, 0.5))
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_title(f"2D Vector Rotation by {theta_deg}°")
ax.legend()
plt.show()
