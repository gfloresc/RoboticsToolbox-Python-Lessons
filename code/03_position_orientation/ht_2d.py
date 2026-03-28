# -*- coding: utf-8 -*-
"""HT_2D.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

# Homogeneous transformation 2D
def T2D(x, y, theta):

    theta = np.radians(theta)

    T = np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0, 0, 1]
    ])

    return T


# Draw frame function
def draw_frame(ax, T, name="Frame", scale=1):

    origin = T @ np.array([0,0,1])
    x_axis = T @ np.array([scale,0,1])
    y_axis = T @ np.array([0,scale,1])

    ax.arrow(origin[0], origin[1],
             x_axis[0]-origin[0],
             x_axis[1]-origin[1],
             head_width=0.1)

    ax.arrow(origin[0], origin[1],
             y_axis[0]-origin[0],
             y_axis[1]-origin[1],
             head_width=0.1)

    ax.text(origin[0]+0.05, origin[1]+0.05, name)


# Local point
p_local = np.array([1,1,1])


# Interactive demo
def demo(x=1, y=1, theta=30):

    T = T2D(x,y,theta)

    p_global = T @ p_local

    fig, ax = plt.subplots(figsize=(6,6))

    # Base frame
    draw_frame(ax, np.eye(3), "Base")

    # Moving frame
    draw_frame(ax, T, "Moved")

    # Transformed point
    ax.scatter(p_global[0], p_global[1])

    origin_moved = T @ np.array([0,0,1])

    ax.plot([origin_moved[0], p_global[0]],
            [origin_moved[1], p_global[1]],
            '--')

    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.title("2D Homogeneous Transformation")

    plt.show()


    # Print matrices and vectors
    print("Homogeneous matrix T:")
    print(np.round(T,2))

    print("\nLocal point:")
    print(p_local)

    print("\nGlobal point:")
    print(np.round(p_global,2))


interact(
    demo,
    x=(-3,3,0.1),
    y=(-3,3,0.1),
    theta=(-180,180,5)
);