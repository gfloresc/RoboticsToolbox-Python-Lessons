import asyncio
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg #=====================>
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up Swift 3D viewer
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
env = swift.Swift()
env.launch(realtime=True)

# Load Panda robot and set to ready position
panda = rtb.models.Panda()
panda.q = panda.qr
env.add(panda)

# Time step
dt = 0.05

# Define trajectory: square in XY at Z=0.40
Tstart = panda.fkine(panda.q)
waypoints = [
    Tstart * sm.SE3.Trans(-0.1, -0.1, 0.40),
    Tstart * sm.SE3.Trans(-0.1,  0.1, 0.40),
    Tstart * sm.SE3.Trans( 0.1,  0.1, 0.40),
    Tstart * sm.SE3.Trans( 0.1, -0.1, 0.40),
    Tstart * sm.SE3.Trans(-0.1, -0.1, 0.40)
]

# ---- ADD MARKERS TO SWIFT ----
for wp in waypoints:
    marker = sg.Sphere(
        radius=0.01,
        pose=sm.SE3(wp.t[0], wp.t[1], wp.t[2]),
        color=[1.0, 0.0, 0.0, 1.0]   # red RGBA
    )
    env.add(marker)
env.step(0)

# Prepare lists to store actual trajectory
x_actual, y_actual, z_actual = [], [], []

# Set up live 3D plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line_actual, = ax.plot([], [], [], 'b-', label='Actual Trajectory')
for wp in waypoints:
    ax.scatter(wp.t[0], wp.t[1], wp.t[2], color='red', s=80)  # desired points
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 0.6)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D End-Effector Trajectory')
ax.grid(True)
ax.legend()

# Control loop: follow the trajectory
for Tep in waypoints:
    arrived = False
    t_local = 0
    while not arrived and t_local < 5.0:  # allow up to 5 seconds per target
        v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, gain=1, threshold=1e-3)
        panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
        env.step(dt)

        # Log actual end-effector position
        pos = panda.fkine(panda.q).t
        x_actual.append(pos[0])
        y_actual.append(pos[1])
        z_actual.append(pos[2])

        # Update 3D plot
        line_actual.set_data(x_actual, y_actual)
        line_actual.set_3d_properties(z_actual)
        plt.pause(0.001)

        t_local += dt


    # Stop the robot at the target
    panda.qd = np.zeros(panda.n)
    env.step(dt)

# Hold final plot
plt.ioff()
plt.show()
