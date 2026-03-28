import sympy as sp  # Import SymPy for symbolic computation
import numpy as np 

# Define the symbolic variable
x = sp.Symbol('x')

# Define the vector as a column matrix
X = sp.Matrix([ sp.sin(x),    x**2,     sp.cos(x) - 3*x   ])

# Compute the derivative of the vector with respect to x
dX_dx = X.diff(x)
Y = sp.Matrix([x])
dY_dx = Y.diff(x)

# Display the result in a readable format
sp.pprint(dX_dx)
sp.pprint(dY_dx)
