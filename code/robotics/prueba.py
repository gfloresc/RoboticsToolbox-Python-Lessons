import swift  # Entorno gráfico de Swift para visualizar el robot
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

# Crear el entorno de visualización de Swift
env = swift.Swift()
env.launch(realtime=True)

# Crear el modelo del robot (Panda) y establecer la configuración inicial
panda = rtb.models.Panda()
panda.q = panda.qr  # Configuración de reposo

# Definir la posición deseada del efector final en el espacio cartesiano
# Usamos una transformación homogénea SE3 para especificar la posición objetivo
Tep = panda.fkine(panda.q) * sm.SE3.Trans(0.2, 0.2, 0.45)  # Posición deseada

# Añadir el robot al entorno de Swift
env.add(panda)

# Parámetros de control
dt = 0.05  # Paso de tiempo en segundos
arrived = False  # Indicador de llegada al objetivo

# Bucle de control para mover el efector final hacia la posición deseada
while not arrived:
    # Calcular el error de posición usando servo visual
    v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 2.0)
    
    # Calcular las velocidades articulares usando el jacobiano inverso
    panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v #qdot = (J^+)⋅(v)
    
    # Actualizar la visualización y los valores del robot
    env.step(dt)

# Mantener la ventana de visualización abierta al final de la simulación
env.hold()
