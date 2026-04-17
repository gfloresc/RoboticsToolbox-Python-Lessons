import numpy as np

# -----------------------------
# Geometric parameters
# -----------------------------
a1 = 3.0
a2 = 2.0
d4 = 1.0   # constant tool offset

# -----------------------------
# Joint variables
# SCARA = R R P R
# -----------------------------
theta1 = np.deg2rad(30.0)
theta2 = np.deg2rad(45.0)
d3 = 1.0
theta4 = np.deg2rad(0.0)

# -----------------------------
# Trigonometric shortcuts
# -----------------------------
c1, s1 = np.cos(theta1), np.sin(theta1)
c2, s2 = np.cos(theta2), np.sin(theta2)
c4, s4 = np.cos(theta4), np.sin(theta4)

# -----------------------------
# Individual DH matrices
# -----------------------------
A1 = np.array([
    [c1, -s1,  0.0, a1 * c1],
    [s1,  c1,  0.0, a1 * s1],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

A2 = np.array([
    [c2,  s2,  0.0, a2 * c2],
    [s2, -c2,  0.0, a2 * s2],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0,  0.0, 1.0]
])

A3 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, d3],
    [0.0, 0.0, 0.0, 1.0]
])

A4 = np.array([
    [c4, -s4, 0.0, 0.0],
    [s4,  c4, 0.0, 0.0],
    [0.0, 0.0, 1.0, d4],
    [0.0, 0.0, 0.0, 1.0]
])

# -----------------------------
# Successive forward kinematics
# -----------------------------
T01 = A1
T02 = A1 @ A2
T03 = A1 @ A2 @ A3
T04 = A1 @ A2 @ A3 @ A4

# -----------------------------
# Origins expressed in base frame
# -----------------------------
o0 = np.array([0.0, 0.0, 0.0])
o1 = T01[:3, 3]
o2 = T02[:3, 3]
o3 = T03[:3, 3]
o4 = T04[:3, 3]

# -----------------------------
# z axes expressed in base frame
# -----------------------------
z0 = np.array([0.0, 0.0, 1.0])
z1 = T01[:3, 2]
z2 = T02[:3, 2]
z3 = T03[:3, 2]

# -----------------------------
# Geometric Jacobian for SCARA (R R P R)
# -----------------------------
# Joint 1: revolute
Jv1 = np.cross(z0, o4 - o0)
Jw1 = z0

# Joint 2: revolute
Jv2 = np.cross(z1, o4 - o1)
Jw2 = z1

# Joint 3: prismatic
Jv3 = z2
Jw3 = np.array([0.0, 0.0, 0.0])

# Joint 4: revolute
Jv4 = np.cross(z3, o4 - o3)
Jw4 = z3

# Complete 6x4 Jacobian
J = np.vstack([
    np.column_stack([Jv1, Jv2, Jv3, Jv4]),
    np.column_stack([Jw1, Jw2, Jw3, Jw4])
])

# -----------------------------
# Print results
# -----------------------------
np.set_printoptions(precision=6, suppress=True)

print("T01 =\n", T01)
print("\nT02 =\n", T02)
print("\nT03 =\n", T03)
print("\nT04 =\n", T04)

print("\no0 =", o0)
print("o1 =", o1)
print("o2 =", o2)
print("o3 =", o3)
print("o4 =", o4)

print("\nz0 =", z0)
print("z1 =", z1)
print("z2 =", z2)
print("z3 =", z3)

print("\nGeometric Jacobian J =\n", J)