"""
Inverse kinematics of a 2-DOF planar arm
using the Levenberg-Marquardt algorithm and ONLY the forward kinematics.

Idea: guess the joint angles, check how far the hand is from the target,
move each joint a little to see where the hand goes, and correct.
Repeat until the hand reaches the target.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================================
# >>>>>>>>>>>>>>  CHANGE THE DESIRED POSITION HERE  <<<<<<<<<<<<<<<<<<<
# Target position of the hand [x, y] in meters.
# Reachable targets satisfy  |L1 - L2| <= r <= L1 + L2,  with r = sqrt(x^2 + y^2).
# With L1 = 1.0 and L2 = 0.8:  0.2 m <= r <= 1.8 m  (a ring around the base).
DESIRED_POSITION = np.array([1.2, 0.7])

# Examples to try (uncomment ONE line at a time):
# DESIRED_POSITION = np.array([-1.0, 0.5])   # r = 1.12  -> reachable
# DESIRED_POSITION = np.array([0.0, -1.5])   # r = 1.50  -> reachable
# DESIRED_POSITION = np.array([1.8, 0.0])    # r = 1.80  -> on the outer edge (arm fully
#                                            #              stretched, slower convergence)
# DESIRED_POSITION = np.array([1.5, 1.5])    # r = 2.12  -> NOT reachable (too far)
# DESIRED_POSITION = np.array([0.1, 0.1])    # r = 0.14  -> NOT reachable (too close
#                                            #              to the base)

# Initial guess for the joint angles [theta1, theta2] in radians.
INITIAL_GUESS = np.array([0.1, 0.1])
# =====================================================================

# ---------------- Robot ----------------
L1 = 1.0   # length of link 1 [m]
L2 = 0.8   # length of link 2 [m]


def forward_kinematics(q):
    """Given the joint angles q = [theta1, theta2], return the hand position [x, y]."""
    q1, q2 = q
    x = L1*np.cos(q1) + L2*np.cos(q1 + q2)
    y = L1*np.sin(q1) + L2*np.sin(q1 + q2)
    return np.array([x, y])


# ---------------- Algorithm ----------------
def effect_of_each_joint(q, h=1e-3):
    """Move each joint a little (h) and see how the hand moves.
    Each column of A tells where the hand goes if we rotate that joint."""
    p = forward_kinematics(q)
    A = np.zeros((2, 2))
    for i in range(2):
        q_moved = q.copy()
        q_moved[i] += h
        A[:, i] = (forward_kinematics(q_moved) - p) / h
    return A


def inverse_kinematics(p_desired, q_initial, lam_min=1e-3, tolerance=1e-6, max_iter=300):
    q = np.array(q_initial, dtype=float)
    history = [q.copy()]

    for k in range(max_iter):
        # Step 1: where am I?
        p = forward_kinematics(q)

        # Step 2: how far am I from the target?
        e = p_desired - p
        print(f"Iteration {k:3d}:  theta1 = {np.degrees(q[0]):+8.2f} deg,"
              f"  theta2 = {np.degrees(q[1]):+8.2f} deg,"
              f"  error = {np.linalg.norm(e):.2e} m")
        if np.linalg.norm(e) < tolerance:
            break

        # Step 3: what does each joint do?
        A = effect_of_each_joint(q)

        # Step 4: how much do I move each joint?
        # lam = "take short steps". It grows with the error:
        #   far from the target -> large lam -> short, safe steps
        #   close to the target -> small lam -> long, fast steps
        # (A fixed value, e.g. lam = 0.1, can make the arm oscillate
        #  when the target is far from the initial guess.)
        lam = 0.5*np.dot(e, e) + lam_min
        dq = np.linalg.solve(A.T @ A + lam*np.eye(2), A.T @ e)

        # Step 5: update and repeat
        q = q + dq
        history.append(q.copy())

    return q, history


# ---------------- Test ----------------
if __name__ == "__main__":
    q, history = inverse_kinematics(DESIRED_POSITION, INITIAL_GUESS)

    # OUTPUT of the algorithm: two numbers, the joint angles for this target
    print(f"\nRESULT for target {DESIRED_POSITION}:")
    print(f"   theta1 = {np.degrees(q[0]):.2f} deg  ({q[0]:.4f} rad)")
    print(f"   theta2 = {np.degrees(q[1]):.2f} deg  ({q[1]:.4f} rad)")
    print(f"Check with forward kinematics: {forward_kinematics(q)}")
    r = np.linalg.norm(DESIRED_POSITION)
    if r > L1 + L2 or r < abs(L1 - L2):
        print("Note: the target is out of reach; the arm gets as close as possible.")

    # ---------------- Animation ----------------
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 12})
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    reach = L1 + L2 + 0.2
    ax.set_xlim(-reach, reach); ax.set_ylim(-reach, reach)   # whole workspace
    ax.set_aspect("equal"); ax.grid(True, ls=":")
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$y$ [m]")

    # Workspace boundaries: outer (max reach) and inner (min reach) circles
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot((L1 + L2)*np.cos(t), (L1 + L2)*np.sin(t), color="0.7", lw=1,
            label="Workspace limits")
    ax.plot(abs(L1 - L2)*np.cos(t), abs(L1 - L2)*np.sin(t), color="0.7", lw=1)

    ax.plot(*DESIRED_POSITION, "x", color="tab:blue", ms=12, mew=2.5, label="Target")
    trail, = ax.plot([], [], "k:", lw=1, label="Hand path")
    arm, = ax.plot([], [], "-o", color="k", lw=3, ms=7, label="Arm")
    text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top",
                   bbox=dict(facecolor="white", edgecolor="k", boxstyle="square"))
    ax.legend(loc="lower right", frameon=True, edgecolor="k", fancybox=False,
              fontsize=10)

    hands = np.array([forward_kinematics(qk) for qk in history])

    def draw(k):
        q1, q2 = history[k]
        elbow = np.array([L1*np.cos(q1), L1*np.sin(q1)])
        hand = hands[k]
        arm.set_data([0, elbow[0], hand[0]], [0, elbow[1], hand[1]])
        trail.set_data(hands[:k+1, 0], hands[:k+1, 1])
        error = np.linalg.norm(DESIRED_POSITION - hand)
        # Outputs of the algorithm at this iteration: the joint angles
        text.set_text(f"Iteration {k}\n"
                      rf"$\theta_1$ = {np.degrees(q1):7.2f}$^\circ$" "\n"
                      rf"$\theta_2$ = {np.degrees(q2):7.2f}$^\circ$" "\n"
                      f"Error = {error:.4f} m")
        return arm, trail, text

    # Frames to show: all of them if there are few; otherwise the first 20
    # and then a sample of the rest (so the animation does not last minutes)
    n = len(history)
    frames = list(range(n)) if n <= 40 else \
        sorted(set(list(range(20)) + list(np.linspace(20, n - 1, 20, dtype=int))))

    anim = FuncAnimation(fig, draw, frames=frames, interval=600,
                         repeat_delay=1500)
    plt.tight_layout(pad=0.3)
    anim.save("ik_lm_2dof_simple.gif", writer="pillow", fps=2)
    plt.show()