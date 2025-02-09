import numpy as np
from sympy import Matrix

# Define the matrix A
A = np.array([
    [0, 1, 1, 2],
    [1, 2, 3, 4],
    [2, 0, 2, 0]
])

# Compute the rank
rank = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank}")

# Compute the null space using SymPy
sympy_A = Matrix(A)
null_space = sympy_A.nullspace()

# Display the basis of the null space
print("Basis of Null Space:")
for vector in null_space:
    # Convert each basis vector to a NumPy array, ensure it is float, and flatten for better display
    print(np.array(vector).astype(float).flatten())

# Display the basis of the range space (column space)
# Perform Singular Value Decomposition (SVD) to get U, S, and V^T
U, _, _ = np.linalg.svd(A, full_matrices=False)

# Identify indices of linearly independent columns using the rank of the matrix
independent_cols = np.where(np.abs(U[:, :rank]) > 1e-10)[0]

# Extract the linearly independent columns to form the basis of the range space
basis_range_space = A[:, independent_cols]

print("Basis of Range Space:")
for vector in basis_range_space.T:
    # Transpose the basis matrix and print each vector (as columns)
    print(vector)
