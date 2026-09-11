"""
Robotics and Automation
2ndOrderControl.py

Double integrator with PD, feedforward, and an optional integral term.

    xdot = v
    vdot = u + d          d is a constant disturbance (force offset, gravity)

Reference:  x_d(t) = offset + A sin(w t),  with v_d and a_d obtained by
differentiating it analytically. You need all three.

Switch MODE to compare:
    "P"       u = -kp e                       fails: no damping
    "PD"      u = -kp e - kd edot             stable, but lags a moving target
    "PD+FF"   u = ad - kp e - kd edot         exact when d = 0
    "PID+FF"  u = ad - kp e - kd edot - ki I  also removes a constant d
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
MODE = "PD+FF"          # "P", "PD", "PD+FF", "PID+FF"
U_MAX = None            # set to a number, e.g. 15.0, to saturate the actuator

T = 12.0
x0 = 0.5                # initial position
v0 = -1.0               # initial velocity

kp, kd, ki = 10.0, 6.0, 8.0

A, w, offset = 1.0, 1.0, 0.0     # reference x_d = offset + A sin(w t)
d = 0.0                          # constant disturbance; try d = 10.0


# ----------------------------------------------------------------------
# Reference trajectory: position, velocity, acceleration
# ----------------------------------------------------------------------
def reference(t):
    xd = offset + A * np.sin(w * t)
    vd = A * w * np.cos(w * t)
    ad = -A * w**2 * np.sin(w * t)
    return xd, vd, ad


# ----------------------------------------------------------------------
# Control law
#
# The integral is carried as a THIRD STATE, not as a global variable.
# solve_ivp uses an adaptive step: it evaluates the right-hand side at
# trial times, rejects steps, and goes backwards. A hand-rolled
# "I += e * (t - t_prev)" accumulates garbage under those conditions.
# Letting the integrator integrate it is both correct and simpler.
# ----------------------------------------------------------------------
def control_law(t, pos, vel, integral):
    xd, vd, ad = reference(t)
    e = pos - xd
    edot = vel - vd

    if MODE == "P":
        u = -kp * e
    elif MODE == "PD":
        u = -kp * e - kd * edot
    elif MODE == "PD+FF":
        u = ad - kp * e - kd * edot
    elif MODE == "PID+FF":
        u = ad - kp * e - kd * edot - ki * integral
    else:
        raise ValueError(f"unknown MODE: {MODE}")

    if U_MAX is not None:
        u = np.clip(u, -U_MAX, U_MAX)
    return u, e


def dynamics(t, state):
    pos, vel, integral = state
    u, e = control_law(t, pos, vel, integral)
    return [vel,            # xdot = v
            u + d,          # vdot = u + disturbance
            e]              # Idot = e   (the integral of the error)


# ----------------------------------------------------------------------
# Integrate
# ----------------------------------------------------------------------
t_eval = np.linspace(0, T, 1500)
sol = solve_ivp(dynamics, [0, T], [x0, v0, 0.0], t_eval=t_eval,
                rtol=1e-8, atol=1e-10)

x, v, I = sol.y
xd, vd, ad = reference(sol.t)

# Rebuild the control signal after integrating. control_law is a pure
# function of the state, so this reproduces exactly what was applied.
u = np.array([control_law(t, xi, vi, Ii)[0]
              for t, xi, vi, Ii in zip(sol.t, x, v, I)])

e = x - xd
tail = sol.t > T / 2
print(f"MODE = {MODE},  disturbance d = {d}")
print(f"  steady-state error amplitude : {np.max(np.abs(e[tail])):.4f}")
print(f"  peak control effort          : {np.max(np.abs(u)):.2f}")

# Theory check for PD without feedforward, no disturbance
if MODE == "PD" and d == 0.0:
    pred = A * w**2 / np.sqrt((kp - w**2)**2 + (kd * w)**2)
    print(f"  predicted by the formula     : {pred:.4f}")
# Theory check for PD with feedforward under a constant disturbance
if MODE == "PD+FF" and d != 0.0:
    print(f"  predicted offset d/kp        : {d/kp:.4f}")


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

axs[0].plot(sol.t, x, linewidth=2, label="x(t)")
axs[0].plot(sol.t, xd, "--", linewidth=1.4, label="$x_d(t)$")
axs[0].set_ylabel("position")
axs[0].set_title(f"Double integrator tracking, MODE = {MODE}")
axs[0].grid(alpha=0.3)
axs[0].legend(loc="upper right")

axs[1].plot(sol.t, v, linewidth=2, label="v(t)")
axs[1].plot(sol.t, vd, "--", linewidth=1.4, label="$v_d(t)$")
axs[1].set_ylabel("velocity")
axs[1].grid(alpha=0.3)
axs[1].legend(loc="upper right")

axs[2].plot(sol.t, u, linewidth=2, color="tab:red", label="u(t)")
axs[2].axhline(0.0, color="k", linewidth=1)
if U_MAX is not None:
    axs[2].axhline(U_MAX, color="k", linestyle=":", linewidth=1.2)
    axs[2].axhline(-U_MAX, color="k", linestyle=":", linewidth=1.2)
axs[2].set_xlabel("time [s]")
axs[2].set_ylabel("control u")
axs[2].grid(alpha=0.3)
axs[2].legend(loc="upper right")

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# Questions for the students
#
# 1. MODE = "P". The position still tracks the sine roughly. Why does the
#    velocity plot look so much worse than the position plot?
# 2. MODE = "PD". Measure the error amplitude and check it against
#    A w^2 / sqrt((kp - w^2)^2 + (kd w)^2). Then raise w to 3 and repeat.
# 3. Set d = 10.0 with MODE = "PD+FF". The error no longer goes to zero.
#    Predict the offset before you run it. (Hint: at equilibrium the
#    control must supply -d, and only the -kp*e term can do that.)
# 4. Keep d = 10.0 and switch to "PID+FF". The offset disappears. What
#    happened to the peak control effort?
# 5. Set U_MAX = 5.0 with w = 3. Which mode survives, and which does not?
# ----------------------------------------------------------------------