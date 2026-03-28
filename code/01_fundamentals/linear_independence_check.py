from sympy import Matrix

# Part (a): Define the matrix with vectors as columns
A = Matrix([
    [1, 1, -2],
    [-1, 1, 3],
    [2, -2, 1]
])

# Compute the rank of the matrix
rank_a = A.rank()

# Check linear independence for Part (a)
if rank_a == A.shape[1]: # A.shape[1] te da el número de columnas; A.shape[0] te da el número de filas
    result_a = "Linearly Independent"
else:
    result_a = "Linearly Dependent"

# Part (b): Define the matrix with vectors as columns
B = Matrix([
    [1, -1, -5],
    [3, 1, -7],
    [3, 3, 3]
])

# Compute the rank of the matrix
rank_b = B.rank()

# Check linear independence for Part (b)
if rank_b == B.shape[1]:
    result_b = "Linearly Independent"
else:
    result_b = "Linearly Dependent"

# Print results
print("Part (a):", result_a)
print("Part (b):", result_b)
