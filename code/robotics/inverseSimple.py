import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt

# Define the robot's DH parameters (theta, d, a, alpha)
# Example of a robot with 3 links
L1 = rtb.RevoluteDH(d=0.5, a=1, alpha=0)     # Link 1
L2 = rtb.RevoluteDH(d=0, a=1, alpha=0)       # Link 2
L3 = rtb.RevoluteDH(d=0, a=1, alpha=0)       # Link 3

# Create the robot from the links
robot = rtb.DHRobot([L1, L2, L3], name='3-DOF Robot')

# Display the DH table (this prints the DH parameters)
print(robot)

# Joint configuration (angles in radians for each link)
q = [0, 0, 0]  # You can modify the values to change the robot's posture

# Calculate forward kinematics (the final transformation of the end-effector)
T = robot.fkine(q)
print(f"End-effector position (forward kinematics): \n{T}")

# Inverse kinematics
# Define the desired pose for the end-effector
Tep = SE3.Trans(0, 0, 3.5)  # Target pose (x=1, y=0.8, z=0.5)

# Solve inverse kinematics to find the joint angles for the desired pose
q_inv = robot.ikine_LM(Tep)  # Levenberg-Marquardt IK solver

# Extract the joint configuration that achieves the desired pose
q_solution = q_inv.q
print(f"Joint angles (inverse kinematics solution): {q_solution}")

# Plot the robot in the configuration obtained from inverse kinematics
robot.plot(q_solution, block=True)

# Interactive tool to visualize the robot's joint states and DH parameters
robot.teach(jointlabels=1)  # Opens an interactive window with DH axes

