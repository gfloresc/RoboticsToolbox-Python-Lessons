from roboticstoolbox import DHRobot, RevoluteDH
import numpy as np
import matplotlib.pyplot as plt

# Define the robot links using DH parameters
L1 = RevoluteDH(a=1, alpha=0, d=0)
L2 = RevoluteDH(a=1, alpha=0, d=0)

# Create the robot with two links
robot = DHRobot([L1, L2], name='2-DOF Robot')

# Define the joint angles
q = [np.pi/4, np.pi/4]  # Joint angles in radians

# Calculate forward kinematics
T = robot.fkine(q)

# Display the transformation matrix
print("Transformation T:")
print(T)

