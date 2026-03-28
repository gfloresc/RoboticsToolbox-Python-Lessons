import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------
# Parameters / settings
# -----------------------
T = 12.0
x0 = np.array([0.5, -1.0])   # [position, velocity]

# Tracking gains (choose kp>0, kd>0)
kp = 10.0
kd = 6.0
# Integral gains ==========>
I = 0.0
t_prev = None
ki = 8.0

# Desired trajectory: x_d(t) = offset + A sin(w t)
A = 1.0
w = 1
offset = 0.0
d = 10# disturbance

def reference(t):
    """Return desired (x_d, v_d, a_d)."""
    xd = offset + A * np.sin(w * t) #xd = offset + A * np.sin(w * t)
    vd = A * w * np.cos(w * t) #vd = A * w * np.cos(w * t)
    ad = -A * (w**2) * np.sin(w * t) # ad = -A * (w**2) * np.sin(w * t)
    return xd, vd, ad

def control_law(t, x):
    """PD + feedforward for double integrator: u = a_d - kp e - kd e_dot."""
    pos, vel = x
    xd, vd, ad = reference(t)
    e = pos - xd
    edot = vel - vd

    # Integral part
    global I, t_prev
    if t_prev is None:
        dt_local = 0.0
    else:
        dt_local = t - t_prev
    t_prev = t

    I += e * dt_local
    #u = - kp * e - kd * edot - ki * I + ad

    #PD control PLUS ff
    u = - kp * e - kd * edot
    #P control
    #u = - kp * e
    #u = np.clip(u, -2, 2) # Saturation in control
    return u

# Double integrator dynamics:
# xdot = v
# vdot = u
def dynamics(t, x):
    u = control_law(t, x)
    pos, vel = x
    #return [vel, -5*pos - 4*vel + 0*u] #
    return [vel, u]

# Time grid
t_eval = np.linspace(0, T, 1500)

# Integrate
sol = solve_ivp(dynamics, [0, T], x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

# Reconstruct reference and control along the solution
x = sol.y[0]
v = sol.y[1]
xd = np.zeros_like(sol.t)
vd = np.zeros_like(sol.t)
ad = np.zeros_like(sol.t)
u  = np.zeros_like(sol.t)

for i, t in enumerate(sol.t):
    xd[i], vd[i], ad[i] = reference(t)
    u[i] = control_law(t, [x[i], v[i]])

# -----------------------
# Combined plots (subplots)
# -----------------------
fig, axs = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

# Position
axs[0].plot(sol.t, x, label='x(t) position')
axs[0].plot(sol.t, xd, '--', label='x_d(t) desired')
axs[0].set_ylabel('Position')
axs[0].set_title('Double Integrator Tracking')
axs[0].grid(True)
axs[0].legend()

# Velocity
axs[1].plot(sol.t, v, label='v(t) velocity')
axs[1].plot(sol.t, vd, '--', label='v_d(t) desired')
axs[1].set_ylabel('Velocity')
axs[1].grid(True)
axs[1].legend()

# Control
axs[2].plot(sol.t, u, label='u(t) control')
axs[2].axhline(0.0, linestyle='--', linewidth=1)
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Control u')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()