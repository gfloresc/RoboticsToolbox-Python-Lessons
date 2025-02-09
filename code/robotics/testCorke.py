import roboticstoolbox as rtb
from spatialmath import SE3

robot = rtb.models.Panda()

Te = robot.fkine(robot.qr)  # forward kinematics
Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tep)         # solve IK

q_pickup = sol[0]

#print(sol)
#print(robot.fkine(q_pickup))    # FK shows that desired end-effector pose was achieved

qt = rtb.jtraj(robot.qr, q_pickup, 50)
robot.plot(qt.q, backend='pyplot', movie='panda1.gif')
