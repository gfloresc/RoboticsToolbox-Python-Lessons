import matplotlib
matplotlib.use('TkAgg')   # útil en macOS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Arc

# =========================================
# Figure
# =========================================
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-2.4, 2.4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.25)
ax.set_title("Line-Follower Robot on a Closed Oval Path", fontsize=17)

# =========================================
# World frame FW
# =========================================
ax.arrow(0, 0, 1.1, 0,
         head_width=0.06, head_length=0.08,
         fc='black', ec='black', linewidth=2,
         length_includes_head=True)
ax.arrow(0, 0, 0, 1.1,
         head_width=0.06, head_length=0.08,
         fc='black', ec='black', linewidth=2,
         length_includes_head=True)

ax.text(1.20, -0.06, r'$x_W$', fontsize=14)
ax.text(-0.08, 1.20, r'$y_W$', fontsize=14)
ax.text(0.10, 0.10, r'$F_W$', fontsize=15)

# =========================================
# Oval path (closed line)
# =========================================
a = 2.4   # semi-axis in x
b = 1.3   # semi-axis in y

s_path = np.linspace(0, 2*np.pi, 500)
x_path = a * np.cos(s_path)
y_path = b * np.sin(s_path)

# draw the line to follow
ax.plot(x_path, y_path, color='dimgray', linewidth=3, label='Line to follow')

# optional dashed centerline effect / guide
ax.plot(x_path, y_path, '--', color='lightgray', linewidth=1)

# =========================================
# Robot geometry in local frame
# simple line-follower style body with nose
# =========================================
body_pts = np.array([
    [-0.28, -0.18],
    [ 0.10, -0.18],
    [ 0.28,  0.00],
    [ 0.10,  0.18],
    [-0.28,  0.18]
])

# small sensor bar under front
sensor_bar_local = np.array([
    [0.18, -0.10],
    [0.24, -0.10],
    [0.24,  0.10],
    [0.18,  0.10]
])

# =========================================
# Utilities
# =========================================
def R(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def transform_points(pts, theta, p):
    return (R(theta) @ pts.T).T + p

# =========================================
# Dynamic artists
# =========================================
robot_patch = None
sensor_patch = None
xR_arrow = None
yR_arrow = None
heading_arrow = None
angle_arc = None
trail_line, = ax.plot([], [], linewidth=2)

theta_text = ax.text(-3.0, 2.05, '', fontsize=14)
pos_text = ax.text(-3.0, 1.75, '', fontsize=13)
theta_arc_label = ax.text(0, 0, r'$\theta$', fontsize=13, color='tab:blue')
frameR_label = ax.text(0, 0, '', fontsize=15, color='purple')
xR_label = ax.text(0, 0, '', fontsize=13, color='tab:orange')
yR_label = ax.text(0, 0, '', fontsize=13, color='tab:green')
heading_label = ax.text(0, 0, '', fontsize=12, color='tab:red')

arc_radius = 0.45

# trail history
trail_x = []
trail_y = []

# pause state
paused = False

def on_key(event):
    global paused
    if event.key == ' ':
        if paused:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        paused = not paused

fig.canvas.mpl_connect('key_press_event', on_key)

# =========================================
# Update function
# =========================================
def update(t):
    global robot_patch, sensor_patch, xR_arrow, yR_arrow, heading_arrow, angle_arc
    global trail_x, trail_y

    # remove previous dynamic patches/arrows
    for artist in [robot_patch, sensor_patch, xR_arrow, yR_arrow, heading_arrow, angle_arc]:
        if artist is not None:
            artist.remove()

    # -------------------------------------
    # Position on oval
    # p(t) = [a cos t, b sin t]
    # tangent = dp/dt = [-a sin t, b cos t]
    # orientation = angle of tangent
    # -------------------------------------
    p = np.array([a * np.cos(t), b * np.sin(t)])
    dp = np.array([-a * np.sin(t), b * np.cos(t)])

    theta = np.arctan2(dp[1], dp[0])   # robot points along tangent direction

    # transformed robot body
    robot_world = transform_points(body_pts, theta, p)
    robot_patch = Polygon(
        robot_world, closed=True,
        facecolor='lightsteelblue',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(robot_patch)

    # sensor bar
    sensor_world = transform_points(sensor_bar_local, theta, p)
    sensor_patch = Polygon(
        sensor_world, closed=True,
        facecolor='black',
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(sensor_patch)

    # robot frame axes
    ex = R(theta) @ np.array([1.0, 0.0])
    ey = R(theta) @ np.array([0.0, 1.0])

    xR_arrow = ax.arrow(
        p[0], p[1], 0.55 * ex[0], 0.55 * ex[1],
        head_width=0.05, head_length=0.07,
        fc='tab:orange', ec='tab:orange',
        linewidth=2.5, length_includes_head=True
    )

    yR_arrow = ax.arrow(
        p[0], p[1], 0.45 * ey[0], 0.45 * ey[1],
        head_width=0.05, head_length=0.07,
        fc='tab:green', ec='tab:green',
        linewidth=2.5, length_includes_head=True
    )

    # heading arrow
    heading_arrow = ax.arrow(
        p[0], p[1], 0.85 * ex[0], 0.85 * ex[1],
        head_width=0.06, head_length=0.08,
        fc='tab:red', ec='tab:red',
        linewidth=2, alpha=0.45, length_includes_head=True
    )

    # labels near robot frame
    xR_label.set_position((p[0] + 0.68 * ex[0], p[1] + 0.68 * ex[1]))
    xR_label.set_text(r'$x_R$')

    yR_label.set_position((p[0] + 0.58 * ey[0], p[1] + 0.58 * ey[1]))
    yR_label.set_text(r'$y_R$')

    frameR_pos = p + 0.17 * (ex + ey)
    frameR_label.set_position((frameR_pos[0], frameR_pos[1]))
    frameR_label.set_text(r'$F_R$')

    heading_label.set_position((p[0] + 0.95 * ex[0], p[1] + 0.95 * ex[1]))
    heading_label.set_text('heading')

    # orientation text
    theta_deg = np.degrees(theta) % 360
    theta_text.set_text(r'Orientation: $\theta = {:.1f}^\circ$'.format(theta_deg))
    pos_text.set_text(r'Position: $(x,y)=({:.2f},{:.2f})$'.format(p[0], p[1]))

    # angle arc around world origin
    angle_arc = Arc(
        (0, 0),
        2 * arc_radius, 2 * arc_radius,
        angle=0,
        theta1=0,
        theta2=theta_deg,
        linewidth=2,
        color='tab:blue'
    )
    ax.add_patch(angle_arc)

    # theta label on arc
    mid_angle = np.deg2rad(theta_deg / 2) if theta_deg > 0 else 0.0
    theta_arc_label.set_position((
        (arc_radius + 0.08) * np.cos(mid_angle),
        (arc_radius + 0.08) * np.sin(mid_angle)
    ))

    # trail
    trail_x.append(p[0])
    trail_y.append(p[1])

    # keep only recent points to avoid infinite clutter
    max_trail = 220
    if len(trail_x) > max_trail:
        trail_x = trail_x[-max_trail:]
        trail_y = trail_y[-max_trail:]

    trail_line.set_data(trail_x, trail_y)

    return (
        trail_line, theta_text, pos_text, theta_arc_label,
        frameR_label, xR_label, yR_label, heading_label
    )

# =========================================
# Animation
# =========================================
frames = np.linspace(0, 2*np.pi, 320)

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,
    blit=False,
    repeat=True
)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()