import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================
params = {
    "m": 1.0,       # mass [kg]
    "g": 9.81,      # gravity [m/s^2]
    "kp": 4.0,      # proportional gain
    "kd": 3.0       # derivative gain
}

# ============================================================
# ALTITUDE MODEL (z, vz)
# ============================================================
def quad_altitude_dynamics(state, thrust, t, params):
    """
    Simple 1D altitude dynamics of a quadrotor.
    state: [z, vz]
    thrust: T (in Newtons)
    """
    z, vz = state
    m = params["m"]
    g = params["g"]

    # --- DISTURBANCE / NOISE HERE ---
    noise = 20*0.5 * (2*np.random.rand() - 1.0)   # uniform in [-0.5, 0.5]
    # ---------------------------------

    # Time-dependent disturbance
    A = 1.0      # amplitude
    w = 2.0      # frequency [rad/s]
    disturbance = A * np.sin(w * t)

    dz = vz
    dvz = (thrust - m*g) / m + 0*disturbance + noise

    return np.array([dz, dvz])


# ============================================================
# ALTITUDE TRAJECTORY GENERATION
#   Given time t, return:
#   - desired altitude z_ref(t)
#   - desired vertical velocity vz_ref(t)
# ============================================================
def reference_trajectory(t):
    """
    Generates a reference altitude trajectory and its derivative.
    Students can modify this to try different trajectories.
    
    Example:
        Constant altitude of 1 meter.
        z_ref(t) = 1
        vz_ref(t) = 0
    """
    #z_ref = 1.0            # constant altitude
    #vz_ref = 0.0           # derivative of constant = 0

    # Example alternatives for class discussion:
    z_ref = 1 + 0.5*np.sin(0.5*t)     # sinusoidal
    vz_ref = 0.5*0.5*np.cos(0.5*t)

    return np.array([z_ref, vz_ref])


# ============================================================
# PD ALTITUDE CONTROLLER
# ============================================================
def altitude_pd_controller(state, ref, params):
    """
    PD control law for altitude tracking.
    state: [z, vz]
    ref: [z_ref, vz_ref]
    """
    z, vz = state
    z_ref, vz_ref = ref

    kp = params["kp"]
    kd = params["kd"]
    m = params["m"]
    g = params["g"]

    # tracking errors
    e_z = z - z_ref
    e_vz = vz - vz_ref

    # Time-dependent disturbance [estimator]
    A = 1.0      # amplitude
    w = 2.0      # frequency [rad/s]
    disturbance_estimated = A * np.sin(w * t)

    # P and PD Control {desired acceleration}
    #a_des_PD = - kp * e_z - kd * e_vz
    a_des_P = - kd * e_z
    a_des_cubic = - 10 * e_z**3

    # Sliding Mode Control
    lam = 10.0
    s =  e_vz + lam * e_z
    # Control acceleration (SMC law)
    a_des_SMC = - kp * np.sign(s) - lam * e_vz

    # Controller [required thrust]
    thrust = m * (g + a_des_cubic )
    #thrust = a_des
    return thrust


# ============================================================
# SIMULATION SETTINGS
# ============================================================
dt = 0.01
T_end = 10.0
N = int(T_end / dt)

state = np.array([0.0, 0.0])   # initial altitude and velocity

time_hist = []
z_hist = []
thrust_hist = []
zref_hist = []

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================
for k in range(N):
    t = k * dt

    # Generate trajectory reference
    ref = reference_trajectory(t)

    # Compute control using reference
    thrust = altitude_pd_controller(state, ref, params)

    # Propagate dynamics
    dx = quad_altitude_dynamics(state, thrust, t, params)
    state = state + dt * dx

    # Store
    time_hist.append(t)
    z_hist.append(state[0])
    thrust_hist.append(thrust)
    zref_hist.append(ref[0])


# ============================================================
# PLOT RESULTS
# ============================================================
plt.figure(figsize=(8, 6))

# Altitude tracking
plt.subplot(2, 1, 1)
plt.plot(time_hist, z_hist, label="z(t)")
plt.plot(time_hist, zref_hist, '--', label="z_ref(t)")
plt.ylabel("Altitude [m]")
plt.title("Altitude Tracking with PD Control")
plt.legend()
plt.grid(True)

# Thrust input
plt.subplot(2, 1, 2)
plt.plot(time_hist, thrust_hist, label="Thrust T(t)")
plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
