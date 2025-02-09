import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt

# Define the robot's DH parameters (theta, d, a, alpha)
# Example of a robot with 3 links
L1 = rtb.RevoluteDH(d=0.5, a=1, alpha=0)     # Link 1
L2 = rtb.RevoluteDH(d=0, a=1, alpha=0)       # Link 2
L3 = rtb.RevoluteDH(d=0, a=1, alpha=0)       # Link 3

# CREATE the robot from the links
robot = rtb.DHRobot([L1, L2, L3], name='3-DOF Robot')
# Display the DH table (this prints the DH parameters)
print(robot)
# Joint configuration (angles in radians for each link)
q = [0, 0, 0]  # You can modify the values to change the robot's posture
# Compute the forward kinematics (the final transformation of the end-effector)
T = robot.fkine(q)
print(f"End-effector position: \n{T}")
# Plot the robot in configuration q
robot.plot(q, block=True)
# If you want a more detailed visualization of the axes
robot.teach(jointlabels=1)  # This opens an interactive window with DH axes
