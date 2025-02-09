import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

# Definir el robot
L1 = rtb.RevoluteDH(a=1)  # Primer eslabón
L2 = rtb.RevoluteDH(a=1)  # Segundo eslabón
L3 = rtb.RevoluteDH(a=1)  # Tercer eslabón
robot = rtb.DHRobot([L1, L2, L3], name="3-DOF Robot")

# Configuración deseada
q_pickup = [1.5, -2, 1]  # Ejemplo de configuración de ángulos

# Generar trayectoria
qt = rtb.jtraj(robot.q, q_pickup, 50)

# Graficar el robot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_robot_with_frames(robot, q, ax):
    # Graficar el robot
    robot.plot(q, ax=ax, block=False, color='blue')

    # Obtener las matrices de transformación
    T = robot.fkine(q)

    # Graficar los marcos coordenados
    for i in range(len(T)):
        frame = T[i]
        ax.quiver(frame[0, 3], frame[1, 3], frame[2, 3], 
                  frame[0, 0], frame[1, 0], frame[2, 0], color='red', length=0.5, normalize=True)  # X-axis
        ax.quiver(frame[0, 3], frame[1, 3], frame[2, 3], 
                  frame[0, 1], frame[1, 1], frame[2, 1], color='green', length=0.5, normalize=True)  # Y-axis
        ax.quiver(frame[0, 3], frame[1, 3], frame[2, 3], 
                  frame[0, 2], frame[1, 2], frame[2, 2], color='blue', length=0.5, normalize=True)  # Z-axis

    # Configurar etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot en Configuración q con Marcos Coordenados')

# Graficar el robot con los marcos coordenados
plot_robot_with_frames(robot, q_pickup, ax)

# Configurar la animación y guardar como GIF
def save_animation(robot, qt, filename):
    images = []
    for q in qt.q:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_robot_with_frames(robot, q, ax)
        plt.pause(0.1)
        # Capturar la imagen
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    
    # Guardar el GIF
    images[0].save(filename, save_all=True, append_images=images[1:], duration=1000, loop=0)

# Llamar a la función para guardar la animación
save_animation(robot, qt, 'panda1.gif')

plt.show()
