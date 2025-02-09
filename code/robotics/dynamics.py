import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
m = 1.0  # mass (kg)
g = 9.81  # gravitational acceleration (m/s^2)
f = 0  # external force (N) -m*g - 10*y[1]

# Define the differential equation
def system(t, y):
    # y[0] = position (y)
    # y[1] = velocity (v = dy/dt)
    # control: f = -m*g - 10*y[0] - 7*y[1]
    dydt = [y[1], (f - m * g) / m]  # velocity and acceleration
    return dydt

# Initial conditions
y0 = [5, 8]  # initial position and initial velocity
t_span = (0, 10)  # time interval (from 0 to 10 seconds)
t_eval = np.linspace(t_span[0], t_span[1], 100)  # points for evaluation

# Solve the differential equation
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Extract results
t = sol.t        # time values
y = sol.y[0]     # position values
v = sol.y[1]     # velocity values

# Plot the results
plt.figure(figsize=(10, 5))

# Position plot
plt.subplot(2, 1, 1)
plt.plot(t, y, label='Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()

# Velocity plot
plt.subplot(2, 1, 2)
plt.plot(t, v, label='Velocity', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()