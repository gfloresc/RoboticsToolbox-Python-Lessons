import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

panda = rtb.models.Panda()
panda.q = panda.qr

Tep = panda.fkine(panda.q) * sm.SE3.Trans(-0.2, -0.2, 0.45)

arrived = False
env.add(panda)

dt = 0.05

while not arrived:

    v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
    panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
    env.step(dt)

# Uncomment to stop the browser tab from closing
# env.hold()