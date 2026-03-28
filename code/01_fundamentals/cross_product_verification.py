import numpy as np

# Define vectors u and v
u = np.array([3, -1, 1])
v = np.array([2, -3, 1])

# Compute the cross product u × v
cross_u_v = np.cross(u, v)

# Compute the cross product v × u
cross_v_u = np.cross(v, u)

# Verify orthogonality: dot product should be 0
# If u × v is orthogonal to both u and v, then their dot product should be zero
dot_u_cross = np.dot(u, cross_u_v)
dot_v_cross = np.dot(v, cross_u_v)

# Print results
print("Cross product u × v:", cross_u_v)
print("Cross product v × u:", cross_v_u)
print("Dot product u · (u × v):", dot_u_cross)
print("Dot product v · (u × v):", dot_v_cross)
