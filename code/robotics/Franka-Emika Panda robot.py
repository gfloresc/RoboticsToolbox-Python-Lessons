import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np

# Load the UR5 robot model
robot = rtb.models.UR5()  # Robot model: UR5
#print(robot)
print("Current qr (rest configuration):", robot.qr)

#qini_custom = [0, -1.57, 1.57, 0, 1.57, 0]  # Angles in radians


# Print forward kinematics
Te = robot.fkine(robot.qr)  # forward kinematics
#print(Te)

# Define the desired end-effector pose (translation and orientation) --> Desired Homogenous transformtion
Tep = SE3.Trans(-0.5, -0.5, 0.5) * SE3.RPY(15, 90, 0, unit='deg')  # Position and orientation
print(Tep)

# Print inverse kinematics
# Solve inverse kinematics to find the joint angles for the desired pose
sol = robot.ik_LM(Tep)  # Inverse kinematics solution
print(sol)

# Extract the joint configuration that achieves the desired pose
q_pickup = sol[0]  # Joint configuration for the end-effector position

# Generate a joint trajectory from the current pose to the desired pose
qt = rtb.jtraj(robot.qr, q_pickup, 10)  # Trajectory between the start and target pose
#qt = rtb.jtraj(qini_custom, q_pickup, 10)  # Trajectory between the start and target pose


# Plot the robot following the trajectory using the PyPlot backend
robot.plot(qt.q, backend='PyPlot', block=True)  # Visualize the robot moving along the trajectory

# Interactive tool to visualize the robot's joint states and DH parameters
robot.teach(jointlabels=1)  # Interactive mode for teaching and joint control





