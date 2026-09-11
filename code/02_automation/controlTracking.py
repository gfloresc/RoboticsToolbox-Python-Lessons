"""
Robotics and Automation
controlTracking.py

Same plant as control.py, but now the target moves.

    xdot = a*x + b*u          with a > 0, so the OPEN-LOOP plant is unstable
    goal: make x(t) follow  x_d(t) = A sin(w t)

The only structural change is that the reference now has a derivative:

    edot = xdot - x_d_dot = a*x + b*u - x_d_dot

In control.py the target was constant, so x_d_dot was zero and we could
ignore it. Here it is not, and that single term is the whole lesson.

Two control laws are compared on the same plant:

  "no_ffd" : u = (1/b) ( -a*x - k*e )
             -> edot = -k*e - x_d_dot.  Stable, but driven by cos(w t), so the
                error never reaches zero. The robot follows the right shape
                and arrives late.
  "ffd"    : u = (1/b) ( -a*x - k*e + x_d_dot )
             -> edot = -k*e exactly, so e(t) -> 0 and tracking is exact.

Steady-state error for the "no_ffd" case with A = 1 and w = 1:
    edot + k*e = -cos(t)   =>   amplitude |e| = 1 / sqrt(1 + k^2)
It shrinks like 1/k, and never reaches zero for finite gain. Same lesson as
the constant-target case, now in moving form.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------
# Plant and reference
# ----------------------------------------------------------------------
a = 3.0          # same plant as control.py
b = 2.0
A = 1.0          # amplitude of the target
w = 1.0          # frequency of the target [rad/s]
x0 = 1.5         # initial condition
T = 10.0         # simulation horizon [s]

k = 5.0          # control gain, the same for both laws


def reference(t):
    """Return the target AND its time derivative. You need both."""
    x_d = A * np.sin(w * t)
    x_d_dot = A * w * np.cos(w * t)
    return x_d, x_d_dot


# ----------------------------------------------------------------------
# Control laws
#
# As in control.py, the law is its own function so that u(t) can be
# rebuilt after integrating. The only difference between the two laws
# is the last term.
# ----------------------------------------------------------------------
def control_law(t, x, law):
    x_d, x_d_dot = reference(t)
    e = x - x_d                                   # tracking error

    if law == "no_ffd":
        return (1.0 / b) * (-a * x - k * e)

    elif law == "ffd":
        return (1.0 / b) * (-a * x - k * e + x_d_dot)

    else:
        raise ValueError(f"unknown control law: {law}")


def dynamics(t, state, law):
    """Right-hand side of the closed-loop system: xdot = a*x + b*u."""
    x = state[0]
    u = control_law(t, x, law)
    return [a * x + b * u]


# ----------------------------------------------------------------------
# Run the two cases
# ----------------------------------------------------------------------
cases = [
    ("without the x_d_dot term  (lags)", "no_ffd", "tab:orange"),
    ("with the x_d_dot term  (exact)", "ffd", "tab:blue"),
]

t_eval = np.linspace(0.0, T, 2000)
results = []

for label, law, color in cases:
    sol = solve_ivp(dynamics, [0.0, T], [x0], t_eval=t_eval, args=(law,),
                    rtol=1e-9, atol=1e-11)
    x = sol.y[0]

    # Rebuild the control signal AFTER integrating, using the same law.
    u = np.array([control_law(t, xi, law) for t, xi in zip(sol.t, x)])

    x_d = np.array([reference(t)[0] for t in sol.t])
    e = x - x_d

    results.append((label, sol.t, x, u, e, color))

    # Measure the leftover error once the transient is gone
    tail = sol.t > T / 2
    print(f"{label}: error amplitude = {np.max(np.abs(e[tail])):.4f}")

print(f"\npredicted for the lagging case: 1/sqrt(1 + k^2) = "
      f"{1.0/np.sqrt(1.0 + k**2):.4f}")


# ----------------------------------------------------------------------
# Plots: state on top, tracking error below
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

ax1.plot(t_eval, A * np.sin(w * t_eval), "k--", linewidth=1.4,
         label="reference $x_d(t) = \\sin t$")
for label, t, x, u, e, color in results:
    ax1.plot(t, x, color=color, linewidth=2, label=label)
    ax2.plot(t, e, color=color, linewidth=2, label=label)

ax1.set_ylabel("state $x(t)$")
ax1.set_ylim(-1.8, 1.8)
ax1.set_title("Same plant, same gain, one extra term")
ax1.grid(alpha=0.3)
ax1.legend(loc="lower left", fontsize=9)

pred = 1.0 / np.sqrt(1.0 + k**2)
ax2.axhline(y=0.0, color="k", linewidth=1)
ax2.axhline(y=pred, color="tab:orange", linestyle=":", linewidth=1.2)
ax2.axhline(y=-pred, color="tab:orange", linestyle=":", linewidth=1.2)
ax2.set_xlabel("time [s]")
ax2.set_ylabel("tracking error $e(t)$")
ax2.set_ylim(-0.3, 0.3)
ax2.grid(alpha=0.3)
ax2.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# Questions for the students
#
# 1. Set k = 1, then k = 20, in the "no_ffd" case. Check the amplitude
#    against 1/sqrt(1 + k^2) before you run it. Does the lag ever vanish?
# 2. Change the target to sin(3*t). Remember that x_d_dot changes too.
#    Which case degrades, and why? (Hint: the driving term is now 3 cos 3t.)
# 3. Pretend the plant coefficient is a = 2.5 inside control_law while the
#    true plant still uses a = 3. This is the price of feedforward: it
#    needs the model.
# ----------------------------------------------------------------------
