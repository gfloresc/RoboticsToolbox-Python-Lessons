"""
Robotics and Automation
control.py

First-order plant, constant target, three control laws.

    xdot = a*x + b*u          with a > 0, so the OPEN-LOOP plant is unstable
    goal: drive x(t) to a constant reference x_d

Three control laws are compared on the same plant:

  "p_low"  : u = -k (x - x_d)   with k too small
             -> the closed loop is STILL UNSTABLE. Feedback alone is not enough.
  "p_high" : u = -k (x - x_d)   with k large enough
             -> stable, but x converges to the WRONG value.
  "p_ff"   : u = (1/b) ( -a*x - k (x - x_d) )
             -> stable AND exact, because it forces edot = -k*e.

Closed-loop analysis for the pure P law:
    xdot = (a - b*k) x + b*k*x_d
    stability      requires   k > a/b
    steady state   is         x* = b*k*x_d / (b*k - a)   which is NOT x_d

For a target that moves, see controlTracking.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------
# Plant and reference
# ----------------------------------------------------------------------
a = 3.0          # open-loop coefficient: a > 0 means the plant grows on its own
b = 2.0          # input gain
x_d = 8.0        # constant target
x0 = 10.0        # initial condition
T = 3.0          # simulation horizon [s]

print(f"Pure P control is stable only if k > a/b = {a/b:.2f}\n")


# ----------------------------------------------------------------------
# Control laws
#
# Defining the law as its own function is what lets us plot u(t) later.
# If u is computed inside the integrator it does not exist outside it,
# and plt.plot(u) raises a NameError.
# ----------------------------------------------------------------------
def control_law(x, law, k):
    e = x - x_d                                  # tracking error

    if law in ("p_low", "p_high"):
        return -k * e                            # proportional feedback only

    elif law == "p_ff":
        # feedback + feedforward. Derived in class: from edot = a*e + b*u + a*x_d
        # we impose edot = -k*e and solve for u.
        return (1.0 / b) * (-a * x - k * e)

    else:
        raise ValueError(f"unknown control law: {law}")


def dynamics(t, state, law, k):
    """Right-hand side of the closed-loop system: xdot = a*x + b*u."""
    x = state[0]
    u = control_law(x, law, k)
    return [a * x + b * u]                       # note the b


# ----------------------------------------------------------------------
# Run the three cases
# ----------------------------------------------------------------------
cases = [
    ("P control, k = 1.0  (unstable)", "p_low", 1.0, "tab:red"),
    ("P control, k = 5.0  (offset)", "p_high", 5.0, "tab:orange"),
    ("P + feedforward, k = 5.0 (exact)", "p_ff", 5.0, "tab:blue"),
]

t_eval = np.linspace(0.0, T, 800)
results = []

for label, law, k, color in cases:
    sol = solve_ivp(dynamics, [0.0, T], [x0], t_eval=t_eval, args=(law, k),
                    rtol=1e-8, atol=1e-10)
    x = sol.y[0]

    # Rebuild the control signal AFTER integrating, using the same law.
    u = control_law(x, law, k)

    results.append((label, sol.t, x, u, color))

    # Predicted steady state, so the simulation can be checked against theory
    if law in ("p_low", "p_high"):
        if b * k - a > 0:
            print(f"{label}: predicted x* = {b*k*x_d/(b*k-a):.3f}")
        else:
            print(f"{label}: unstable, no steady state (k < a/b)")
    else:
        print(f"{label}: predicted x* = {x_d:.3f}  (exact)")


# ----------------------------------------------------------------------
# Plots: state on top, control effort below
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

for label, t, x, u, color in results:
    ax1.plot(t, x, color=color, linewidth=2, label=label)
    ax2.plot(t, u, color=color, linewidth=2)

ax1.axhline(y=x_d, color="k", linestyle="--", linewidth=1.2, label="reference $x_d$")
ax1.set_ylabel("state $x(t)$")
ax1.set_ylim(0, 30)          # the unstable case leaves the frame on purpose
ax1.set_title("Same plant, three controllers")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left", fontsize=9)

ax2.axhline(y=0.0, color="k", linewidth=1)
ax2.set_xlabel("time [s]")
ax2.set_ylabel("control input $u(t)$")
ax2.set_ylim(-60, 10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# Questions for the students
#
# 1. Set k = 1.4 and then k = 1.6 in the "p_low" case. Compute the sign of
#    a - b*k first. Watch the long run, not the first two seconds.
# 2. For "p_high", raise k to 20 and to 100. Does the steady-state error
#    ever become exactly zero? Compare with x* = b*k*x_d/(b*k - a).
# 3. Set a = 0 and b = 1. You are now controlling a pure integrator.
#    Check that "p_high" alone reaches x_d with no offset. Why?
# ----------------------------------------------------------------------
