import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
a = 10
b = -240
k = 10
x_d = -100
x0 = 6
T = 10.0

# Define the system dynamics
def dynamics(t, x):
    # P control
    #u =  - k * (x - x_d)
    # P + FF control
    u =  (1/b)*( - a*x - k * (x - x_d) )
    #xdot = a * x + b * u  
    xdot = - a * x 
    return xdot # \dot{x} = a*x + u --> closed loop system [RHS]

# Solve using SciPy's integrator
sol = solve_ivp(dynamics, [0, T], [x0], t_eval=np.linspace(0, T, 1000))

# Plot
plt.plot(sol.t, sol.y[0], label='x(t)')
plt.axhline(y=x_d, color='r', linestyle='--', label='x desired')
plt.xlabel('Time (s)')
plt.ylabel('x')
plt.title('Proportional Control: ẋ = -k(x - x_d)')
plt.grid()
plt.legend()
plt.show()

# Plot
plt.plot(u, label='u')
plt.axhline(y=x_d, color='r', linestyle='--', label='x desired')
plt.xlabel('Time (s)')
plt.ylabel('x')
plt.title('Proportional Control: u')
plt.grid()
plt.legend()
plt.show()
