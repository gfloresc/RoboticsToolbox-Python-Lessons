import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define matrix A with negative eigenvalues
#A = np.array([[-2, -1],
#              [1, -3]])

# Define matrix A with positive eigenvalues
#A = np.array([[3, 1],
#              [0, 2]])

# Define matrix A with complex conjugate eigenvalues [purely imaginary]
#A = np.array([[0, 1],
#              [-2, 0]])

# Define matrix A with eigenvalues that have a negative real part and an imaginary component
#A = np.array([[-1, -2],
#              [2, -1]])

# Define matrix A with eigenvalues that have a positive real part and an imaginary component
A = np.array([[1, -2],
              [2, 1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Define the system of differential equations dx/dt = Ax
def system(t, x):
    return A @ x

# Initial conditions for different trajectories
initial_conditions = [
    np.array([2, 0]),
    np.array([-2, -1]),
    np.array([1, 3]),
    np.array([-3, 2]),
    np.array([0, -2]),
]

# Time span for the simulation
t_span = (0, 5)  # Simulate from t=0 to t=5
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time points for evaluation

# Plot the solutions versus time
plt.figure(figsize=(8, 6))

for x0 in initial_conditions:
    # Solve the system for the given initial condition
    sol = solve_ivp(system, t_span, x0, t_eval=t_eval)
    
    # Plot both components x1(t) and x2(t)
    plt.plot(sol.t, sol.y[0], label=f"x1, x0={x0}")
    plt.plot(sol.t, sol.y[1], linestyle="dashed", label=f"x2, x0={x0}")

# Configure the plot
plt.xlabel("Time (t)")
plt.ylabel("State variables x1 and x2")
plt.title("Solution of the System $\dot{x} = Ax$")
plt.legend()
plt.grid()
plt.show()