import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# Homogeneous transformation in 2D
# -----------------------------
def homogeneous_transform_2d(x, y, theta_deg):
    theta = np.radians(theta_deg)
    T = np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0,              0,             1]
    ])
    return T

# -----------------------------
# Draw a reference frame
# -----------------------------
def draw_frame(ax, T, name="Frame", scale=1.0):
    origin = T @ np.array([0, 0, 1])
    x_axis = T @ np.array([scale, 0, 1])
    y_axis = T @ np.array([0, scale, 1])

    # Draw x-axis
    ax.arrow(origin[0], origin[1],
             x_axis[0] - origin[0], x_axis[1] - origin[1],
             head_width=0.08, length_includes_head=True)

    # Draw y-axis
    ax.arrow(origin[0], origin[1],
             y_axis[0] - origin[0], y_axis[1] - origin[1],
             head_width=0.08, length_includes_head=True)

    ax.text(origin[0] + 0.05, origin[1] + 0.05, name, fontsize=10)

# -----------------------------
# Initial point in local coordinates
# -----------------------------
p_local = np.array([1, 1, 1])  # homogeneous coordinates

# -----------------------------
# Create figure
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.3)

# Initial slider values
x0 = 1.0
y0 = 1.0
theta0 = 30.0

# -----------------------------
# Update function
# -----------------------------
def update(val):
    ax.clear()

    # Current slider values
    x = slider_x.val
    y = slider_y.val
    theta = slider_theta.val

    # Base frame
    T_base = np.eye(3)

    # Moving frame
    T = homogeneous_transform_2d(x, y, theta)

    # Transform local point to global point
    p_global = T @ p_local

    # Draw base frame
    draw_frame(ax, T_base, name="Base", scale=1.0)

    # Draw moving frame
    draw_frame(ax, T, name="Moved", scale=1.0)

    # Draw local point in moved frame
    origin_moved = T @ np.array([0, 0, 1])
    ax.plot(p_global[0], p_global[1], 'o', label="Transformed point")

    # Dashed line from moved frame origin to transformed point
    ax.plot([origin_moved[0], p_global[0]],
            [origin_moved[1], p_global[1]],
            '--')

    # Plot settings
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("2D Homogeneous Transformation")

    # Show matrix on screen
    matrix_text = (
        f"T =\n"
        f"[[{T[0,0]: .2f}, {T[0,1]: .2f}, {T[0,2]: .2f}],\n"
        f" [{T[1,0]: .2f}, {T[1,1]: .2f}, {T[1,2]: .2f}],\n"
        f" [{T[2,0]: .2f}, {T[2,1]: .2f}, {T[2,2]: .2f}]]\n\n"
        f"Local point = [{p_local[0]}, {p_local[1]}, {p_local[2]}]\n"
        f"Global point = [{p_global[0]:.2f}, {p_global[1]:.2f}, {p_global[2]:.2f}]"
    )

    ax.text(-4.8, 4.5, matrix_text, fontsize=10, va='top',
            bbox=dict(boxstyle="round", facecolor="white"))

    ax.legend()
    fig.canvas.draw_idle()

# -----------------------------
# Sliders
# -----------------------------
ax_x = plt.axes([0.2, 0.18, 0.6, 0.03])
ax_y = plt.axes([0.2, 0.13, 0.6, 0.03])
ax_theta = plt.axes([0.2, 0.08, 0.6, 0.03])

slider_x = Slider(ax_x, 'x', -4.0, 4.0, valinit=x0)
slider_y = Slider(ax_y, 'y', -4.0, 4.0, valinit=y0)
slider_theta = Slider(ax_theta, 'theta', -180.0, 180.0, valinit=theta0)

slider_x.on_changed(update)
slider_y.on_changed(update)
slider_theta.on_changed(update)

# Initial draw
update(None)

plt.show()