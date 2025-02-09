import numpy as np
import matplotlib.pyplot as plt

# Define the vector v
v = np.array([3, 4])

# Compute the magnitude (norm) of v
norm_v = np.linalg.norm(v)

# Compute the unit vector of v
unit_v = v / norm_v

# Create a 2D plot
plt.figure()
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector v')
plt.quiver(0, 0, unit_v[0], unit_v[1], angles='xy', scale_units='xy', scale=1, color='green', label='Unit Vector v̂')

# Set axis limits
plt.xlim(0, 4)
plt.ylim(0, 4.5)

# Add grid, labels, and legend
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Vector and Unit Vector')

# Show the plot
plt.show()