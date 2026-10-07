import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------
# Rotation matrices
# ---------------------------------
def rot_x(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])

def rot_y(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [0,          1, 0],
        [-np.sin(a), 0, np.cos(a)]
    ])

def rot_z(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1]
    ])

# ---------------------------------
# Homogeneous transformation in 3D
# ---------------------------------
def homogeneous_transform_3d(x, y, z, roll, pitch, yaw):
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]

    return T

# ---------------------------------
# Draw a reference frame in 3D
# ---------------------------------
def draw_frame(ax, T, name="Frame", scale=1.0):
    origin = T @ np.array([0, 0, 0, 1])
    x_axis = T @ np.array([scale, 0, 0, 1])
    y_axis = T @ np.array([0, scale, 0, 1])
    z_axis = T @ np.array([0, 0, scale, 1])

    # x-axis
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0] - origin[0], x_axis[1] - origin[1], x_axis[2] - origin[2],
              arrow_length_ratio=0.1)

    # y-axis
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0] - origin[0], y_axis[1] - origin[1], y_axis[2] - origin[2],
              arrow_length_ratio=0.1)

    # z-axis
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0] - origin[0], z_axis[1] - origin[1], z_axis[2] - origin[2],
              arrow_length_ratio=0.1)

    ax.text(origin[0], origin[1], origin[2], name)

# ---------------------------------
# Local point in homogeneous coordinates
# ---------------------------------
p_local = np.array([1, 1, 1, 1])

# ---------------------------------
# Create figure and 3D axis
# ---------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.35)

# Initial values
x0 = 1.0
y0 = 1.0
z0 = 1.0
roll0 = 20.0
pitch0 = 20.0
yaw0 = 30.0

# ---------------------------------
# Update function
# ---------------------------------
def update(val):
    ax.cla()

    x = slider_x.val
    y = slider_y.val
    z = slider_z.val
    roll = slider_roll.val
    pitch = slider_pitch.val
    yaw = slider_yaw.val

    T_base = np.eye(4)
    T = homogeneous_transform_3d(x, y, z, roll, pitch, yaw)

    p_global = T @ p_local
    origin_moved = T @ np.array([0, 0, 0, 1])

    # Draw frames
    draw_frame(ax, T_base, name="Base", scale=1.0)
    draw_frame(ax, T, name="Moved", scale=1.0)

    # Draw transformed point
    ax.scatter(p_global[0], p_global[1], p_global[2], s=50, label="Transformed point")

    # Dashed line from moved frame origin to point
    ax.plot([origin_moved[0], p_global[0]],
            [origin_moved[1], p_global[1]],
            [origin_moved[2], p_global[2]], '--')

    # Axes settings
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Homogeneous Transformation")

    # Matrix text
    matrix_text = (
        f"T =\n"
        f"[{T[0,0]: .2f} {T[0,1]: .2f} {T[0,2]: .2f} {T[0,3]: .2f}]\n"
        f"[{T[1,0]: .2f} {T[1,1]: .2f} {T[1,2]: .2f} {T[1,3]: .2f}]\n"
        f"[{T[2,0]: .2f} {T[2,1]: .2f} {T[2,2]: .2f} {T[2,3]: .2f}]\n"
        f"[{T[3,0]: .2f} {T[3,1]: .2f} {T[3,2]: .2f} {T[3,3]: .2f}]\n\n"
        f"Local point = [{p_local[0]}, {p_local[1]}, {p_local[2]}, {p_local[3]}]\n"
        f"Global point = [{p_global[0]:.2f}, {p_global[1]:.2f}, {p_global[2]:.2f}, {p_global[3]:.2f}]"
    )

    fig.text(0.02, 0.95, matrix_text, fontsize=10, va='top',
             bbox=dict(boxstyle="round", facecolor="white"))

    ax.legend()
    fig.canvas.draw_idle()

# ---------------------------------
# Sliders
# ---------------------------------
ax_x = plt.axes([0.2, 0.25, 0.6, 0.02])
ax_y = plt.axes([0.2, 0.22, 0.6, 0.02])
ax_z = plt.axes([0.2, 0.19, 0.6, 0.02])

ax_roll = plt.axes([0.2, 0.13, 0.6, 0.02])
ax_pitch = plt.axes([0.2, 0.10, 0.6, 0.02])
ax_yaw = plt.axes([0.2, 0.07, 0.6, 0.02])

slider_x = Slider(ax_x, 'x', -4.0, 4.0, valinit=x0)
slider_y = Slider(ax_y, 'y', -4.0, 4.0, valinit=y0)
slider_z = Slider(ax_z, 'z', -4.0, 4.0, valinit=z0)

slider_roll = Slider(ax_roll, 'roll', -180.0, 180.0, valinit=roll0)
slider_pitch = Slider(ax_pitch, 'pitch', -180.0, 180.0, valinit=pitch0)
slider_yaw = Slider(ax_yaw, 'yaw', -180.0, 180.0, valinit=yaw0)

slider_x.on_changed(update)
slider_y.on_changed(update)
slider_z.on_changed(update)
slider_roll.on_changed(update)
slider_pitch.on_changed(update)
slider_yaw.on_changed(update)

# Initial draw
update(None)

plt.show()