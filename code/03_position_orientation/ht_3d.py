# -*- coding: utf-8 -*-
"""HT_3D.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from mpl_toolkits.mplot3d import Axes3D

# Rotation matrices
def rot_x(angle):
    a = np.radians(angle)
    return np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])

def rot_y(angle):
    a = np.radians(angle)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [0,          1, 0],
        [-np.sin(a), 0, np.cos(a)]
    ])

def rot_z(angle):
    a = np.radians(angle)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1]
    ])

# Homogeneous transformation
def T3D(x, y, z, roll, pitch, yaw):
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

# Draw frame
def draw_frame(ax, T, name="Frame", scale=1.0):
    origin = T @ np.array([0, 0, 0, 1])
    x_axis = T @ np.array([scale, 0, 0, 1])
    y_axis = T @ np.array([0, scale, 0, 1])
    z_axis = T @ np.array([0, 0, scale, 1])

    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0] - origin[0],
              x_axis[1] - origin[1],
              x_axis[2] - origin[2])

    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0] - origin[0],
              y_axis[1] - origin[1],
              y_axis[2] - origin[2])

    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0] - origin[0],
              z_axis[1] - origin[1],
              z_axis[2] - origin[2])

    ax.text(origin[0], origin[1], origin[2], name)

# Local point in homogeneous coordinates
p_local = np.array([1, 1, 1, 1])

# Interactive function
def demo(x=1, y=1, z=1, roll=0, pitch=0, yaw=0):

    T = T3D(x, y, z, roll, pitch, yaw)
    p_global = T @ p_local
    translation = T[:3, 3]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw base and moved frames
    draw_frame(ax, np.eye(4), name="Base", scale=1.0)
    draw_frame(ax, T, name="Moved", scale=1.0)

    # Draw transformed point
    ax.scatter(p_global[0], p_global[1], p_global[2])

    # Draw line from moved origin to transformed point
    origin_moved = T @ np.array([0, 0, 0, 1])
    ax.plot([origin_moved[0], p_global[0]],
            [origin_moved[1], p_global[1]],
            [origin_moved[2], p_global[2]], '--')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Homogeneous Transformation")

    plt.show()

    # Print numeric results
    print("Homogeneous transformation matrix T:")
    print(np.round(T, 2))

    print("\nLocal point:")
    print(np.round(p_local, 2))

    print("\nGlobal point:")
    print(np.round(p_global, 2))

    print("\nTranslation vector:")
    print(np.round(translation, 2))

interact(
    demo,
    x=(-3, 3, 0.1),
    y=(-3, 3, 0.1),
    z=(-3, 3, 0.1),
    roll=(-180, 180, 5),
    pitch=(-180, 180, 5),
    yaw=(-180, 180, 5)
);

