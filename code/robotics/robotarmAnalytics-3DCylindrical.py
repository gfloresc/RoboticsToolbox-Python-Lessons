import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3

# Define symbolic variables for the DH parameters
theta1, theta2, theta3 = sp.symbols('theta1 theta2 theta3')
a1, a2 = sp.symbols('a1 a2')
d1, d2, d3 = sp.symbols('d1 d2 d3')
alpha1, alpha2 = sp.symbols('alpha1 alpha2')

# Create a class for the symbolic robot using Peter Corke's toolbox
class SymbolicRobot(DHRobot):
    def __init__(self):
        # Define the robot's links using RevoluteDH and PrismaticDH
        L1 = RevoluteDH(d=d1)      # First revolute link
        L2 = PrismaticDH(alpha= -sp.pi/2)  # Second prismatic link
        L3 = PrismaticDH()         # Third prismatic link

        # Initialize the robot with the list of links
        super().__init__([L1, L2, L3], name="Cylindrical 3-DOF Robot")

# Create an instance of the symbolic robot
robot = SymbolicRobot()

# Obtain the symbolic forward kinematics matrix
q = [theta1, d2, d3]
T_symbolic = robot.fkine(q)

# Display the symbolic matrix in pretty format
sp.init_printing()  # Enable LaTeX-style printing
print("Forward Kinematics Matrix (DH Homogeneous Matrix):")
sp.pprint(T_symbolic)
