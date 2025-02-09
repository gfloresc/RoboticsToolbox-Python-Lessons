import sympy as sp  # Import SymPy for symbolic computation

# Define the symbolic variable
x = sp.Symbol('x')

# Define the vector as a column matrix
X = sp.Matrix([
    sp.sin(x),        # First component: sin(x)
    x**2,             # Second component: x^2
    sp.cos(x) - 3*x   # Third component: cos(x) - 3x
])

# Compute the derivative of the vector with respect to x
dX_dx = X.diff(x)

# Display the result in a readable format
sp.pprint(dX_dx)
