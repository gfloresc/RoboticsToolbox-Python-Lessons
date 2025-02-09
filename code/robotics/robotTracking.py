import roboticstoolbox as rtb
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

# Definir el robot
L1 = rtb.RevoluteDH(a=1)  # Primer eslabón
L2 = rtb.RevoluteDH(a=1)  # Segundo eslabón
L3 = rtb.RevoluteDH(a=1)  # Tercer eslabón
robot = rtb.DHRobot([L1, L2, L3], name="3-DOF Robot")

# Configuración deseada
q_pickup = [1.5, -2, np.pi/2]  # Ejemplo de configuración de ángulos

# Configuracion inicial
q_inicial = [0, 0, np.pi/2] # alternativamente robot.q

# Generar trayectoria
qt = rtb.jtraj(q_inicial, q_pickup, 100)

# Graficar el robot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# Usar el backend de PyPlot para dibujar el robot
robot.plot(qt.q, backend='pyplot')

# Configurar la animación y guardar como GIF
def save_animation(robot, qt, filename):
    images = []
    for q in qt.q:
        #fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        robot.plot(q, ax=ax, block=False)
        plt.pause(0.1)
        # Capturar la imagen
        #fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    
    # Guardar el GIF
    images[0].save(filename, save_all=True, append_images=images[1:], duration=100, loop=0)

# Llamar a la función para guardar la animación
save_animation(robot, qt, 'panda1.gif')

plt.show()
