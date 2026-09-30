import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.grid": True, "legend.edgecolor": "k"})

# ============================================================
# Parameters
# ============================================================
dt, T = 0.02, 30.0
t = np.arange(0, T, dt)
N = len(t)

r, w0 = 3.0, 0.25                 # circle radius and angular speed
kx, ky, kth = 1.2, 2.0, 2.2       # controller gains

# ============================================================
# Reference trajectory (virtual unicycle)
# ============================================================
pd   = r * np.column_stack((np.cos(w0 * t), np.sin(w0 * t)))   # [x_d, y_d]
th_d = w0 * t + np.pi / 2                                       # tangent heading
v_d, w_d = r * w0, w0                                           # nominal inputs

wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))
Rot  = lambda th: np.array([[np.cos(th), -np.sin(th)],
                            [np.sin(th),  np.cos(th)]])          # BODY -> WORLD

def disturbance(tk):
    d_xy = np.array([0.04 * np.sin(0.7 * tk), 0.03 * np.cos(0.5 * tk)])
    d_th = 0.015 * np.sin(0.9 * tk)
    return d_xy, d_th

# ============================================================
# CONTROLLERS
# ============================================================
def control(case, p, th, k):
    eW   = pd[k] - p                     # position error, WORLD frame
    e_th = wrap(th_d[k] - th)            # heading error

    # ########################################################
    # CASE 1: CORRECT CONTROL  ->  e^B = R^T(theta) e^W
    # ########################################################
    if case == 0:
        e = Rot(th).T @ eW               # [forward, lateral] error

    # ########################################################
    # CASE 2: WRONG CONTROL  ->  uses e^W as if it were e^B
    #         (world x,y errors are NOT forward/lateral errors)
    # ########################################################
    elif case == 1:
        e = eW

    # ########################################################
    # CASE 3: NO FEEDBACK  ->  v = v_d,  w = w_d
    # ########################################################
    else:
        return v_d, w_d

    # ---------------- Tracking control law (cases 1 and 2) ---
    #   v = v_d cos(e_th) + kx * e_x
    #   w = w_d + ky * e_y + kth * sin(e_th)
    v = v_d * np.cos(e_th) + kx * e[0]
    w = w_d + ky * e[1] + kth * np.sin(e_th)
    return v, w

# ============================================================
# Simulation (same model and disturbances for all cases)
# ============================================================
labels = ["Control + rotation", "Control without rotation", "No feedback"]
styles = [("k", "-", "o"), ("r", "--", "s"), ("g", ":", "^")]
M = len(labels)

P  = np.zeros((M, N, 2)); P[:, 0] = [4.0, -1.0]
TH = np.zeros((M, N));    TH[:, 0] = np.deg2rad(120)
D  = np.zeros((N, 3))

for k in range(N - 1):
    d_xy, d_th = disturbance(t[k])
    D[k] = [*d_xy, d_th]
    for i in range(M):
        v, w = control(i, P[i, k], TH[i, k], k)
        # Unicycle model: x' = v cos(th) + dx, y' = v sin(th) + dy, th' = w + dth
        P[i, k + 1]  = P[i, k] + dt * (v * np.array([np.cos(TH[i, k]), np.sin(TH[i, k])]) + d_xy)
        TH[i, k + 1] = wrap(TH[i, k] + dt * (w + d_th))
D[-1] = [*disturbance(t[-1])[0], disturbance(t[-1])[1]]

err = np.linalg.norm(pd - P, axis=2)     # ||e^W|| for each case

# ============================================================
# Plots
# ============================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(*pd.T, "b-.", lw=2, label="Reference")
for i in range(M):
    c, ls, _ = styles[i]
    ax.plot(*P[i].T, color=c, ls=ls, lw=1.5, label=labels[i])
ax.set(xlabel="$x$ [m]", ylabel="$y$ [m]", aspect="equal")
ax.legend()

fig, ax = plt.subplots(figsize=(8, 3.5))
for i in range(M):
    c, ls, _ = styles[i]
    ax.plot(t, err[i], color=c, ls=ls, lw=1.5, label=labels[i])
ax.set(xlabel="Time [s]", ylabel=r"$\|e^W\|$ [m]", xlim=(0, T))
ax.legend()

fig, ax = plt.subplots(figsize=(8, 3.5))
for j, (c, ls, lab) in enumerate([("k", "-", "$d_x$"), ("b", "-.", "$d_y$"), ("r", "--", r"$d_\theta$")]):
    ax.plot(t, D[:, j], color=c, ls=ls, label=lab)
ax.set(xlabel="Time [s]", ylabel="Disturbance", xlim=(0, T))
ax.legend()

# ============================================================
# Animation
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(*pd.T, "b-.", lw=1.5, label="Reference")
ax.set(xlim=(-6, 7), ylim=(-6, 7), aspect="equal", xlabel="$x$ [m]", ylabel="$y$ [m]")

trails = [ax.plot([], [], color=c, ls=ls, label=l)[0] for (c, ls, _), l in zip(styles, labels)]
bodies = [ax.plot([], [], color=c, marker=m, ms=8, ls="")[0] for c, _, m in styles]
heads  = [ax.plot([], [], color=c, lw=2.5)[0] for c, _, _ in styles]
target, = ax.plot([], [], "bx", ms=10, mew=2)
ax.legend(loc="upper right")
L = 0.5

def update(k):
    for i in range(M):
        x, y, th = *P[i, k], TH[i, k]
        trails[i].set_data(P[i, :k + 1, 0], P[i, :k + 1, 1])
        bodies[i].set_data([x], [y])
        heads[i].set_data([x, x + L * np.cos(th)], [y, y + L * np.sin(th)])
    target.set_data([pd[k, 0]], [pd[k, 1]])
    return (*trails, *bodies, *heads, target)

ani = FuncAnimation(fig, update, frames=N, interval=20, blit=True)
plt.show()