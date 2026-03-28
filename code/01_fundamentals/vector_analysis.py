import numpy as np

# Define two vectors
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Compute the dot (scalar) product
dot_product = np.dot(x, y)
print(f"Dot product (x, y): {dot_product}")

# Compute the norms of x and y
norm_x = np.linalg.norm(x)
norm_y = np.linalg.norm(y)
print(f"||x||: {norm_x:.4f}")
print(f"||y||: {norm_y:.4f}")

# Verify Cauchy–Schwarz inequality
cauchy_schwarz = abs(dot_product) <= norm_x * norm_y
print(f"Cauchy–Schwarz inequality holds: {cauchy_schwarz}")

# Verify Triangle inequality
triangle_inequality = np.linalg.norm(x + y) <= norm_x + norm_y
print(f"Triangle inequality holds: {triangle_inequality}")

# Geometric interpretation (cosine of the angle)
cos_theta = dot_product / (norm_x * norm_y)
theta = np.arccos(cos_theta)  # Angle in radians
print(f"cos(θ): {cos_theta:.4f}")
print(f"Angle θ (in degrees): {np.degrees(theta):.2f}")
