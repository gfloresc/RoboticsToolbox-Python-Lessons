"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      JACOBIAN-BASED KINEMATIC CONTROL — SE(2) matrix log implementation    ║
║                         Following the slides step by step                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Full pipeline (slides p. 12):
  1. T      = FK(q)                        -> current homogeneous transform
  2. T_err  = T_d · T⁻¹  ∈ SE(2)          -> pose error matrix (slide 7)
  3. xi_err = vee(log(T_err))  ∈ R³        -> twist error via matrix log (slide 8)
  4. J(q)   = geometric Jacobian, 3x3
  5. J†(q)  = pinv(J)
  6. q_dot  = +k · J†(q) · xi_err         -> control law (slide 10)*

  * SIGN NOTE: the slide writes q_dot = -k J† xi_error with T_err = T_d · T⁻¹.
    With that definition, xi_err ≈ xi_d - xi points TOWARD the goal, so the
    correct sign for convergence is +k.
    Equivalently: using T_err = T · T_d⁻¹ with -k gives the same result.

    Lyapunov verification (slide 11):
      V  = 1/2 * ||xi_err||²  > 0
      xi_err_dot ≈ -J(q) q_dot = -J(q)(+k J⁻¹ xi_err) = -k xi_err
      V_dot = xi_err^T xi_err_dot = -k||xi_err||² < 0  ✓  exponential convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROBOT  (3-DOF planar RRR)
# ─────────────────────────────────────────────────────────────────────────────
L = [1.2, 1.0, 0.7]          # link lengths [m]
LINK_COLORS = ['#2563EB', '#7C3AED', '#DB2777']

def fk(q):
    """Forward kinematics: returns end-effector pose [x, y, theta]."""
    q1, q2, q3 = q
    x = L[0]*np.cos(q1) + L[1]*np.cos(q1+q2) + L[2]*np.cos(q1+q2+q3)
    y = L[0]*np.sin(q1) + L[1]*np.sin(q1+q2) + L[2]*np.sin(q1+q2+q3)
    return np.array([x, y, q1+q2+q3])

def joint_positions(q):
    """Returns (x, y) coordinates of all joints for plotting."""
    q1, q2, q3 = q
    p0 = np.zeros(2)
    p1 = p0 + L[0]*np.array([np.cos(q1), np.sin(q1)])
    p2 = p1 + L[1]*np.array([np.cos(q1+q2), np.sin(q1+q2)])
    p3 = p2 + L[2]*np.array([np.cos(q1+q2+q3), np.sin(q1+q2+q3)])
    return np.array([p0, p1, p2, p3])

def jacobian(q):
    """Geometric Jacobian J(q) ∈ R^{3x3} for the planar RRR robot."""
    q1, q2, q3 = q
    s1, c1    = np.sin(q1),        np.cos(q1)
    s12, c12  = np.sin(q1+q2),     np.cos(q1+q2)
    s123,c123 = np.sin(q1+q2+q3),  np.cos(q1+q2+q3)
    return np.array([
        [-L[0]*s1-L[1]*s12-L[2]*s123, -L[1]*s12-L[2]*s123, -L[2]*s123],
        [ L[0]*c1+L[1]*c12+L[2]*c123,  L[1]*c12+L[2]*c123,  L[2]*c123],
        [1., 1., 1.]
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 2. SE(2) MATRICES  <- exact implementation of the slides' approach
# ─────────────────────────────────────────────────────────────────────────────
def se2_matrix(q):
    """
    Builds the homogeneous transformation matrix T ∈ SE(2):
        T = [[R, p], [0, 1]]    R ∈ SO(2), p ∈ R²
    This is the planar analog of the SE(3) matrices used in the slides.
    """
    x, y, theta = fk(q)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])

def se2_inverse(T):
    """
    Inverse of T ∈ SE(2):  T⁻¹ = [[R^T, -R^T p], [0, 1]]
    (slide 7, T⁻¹ formula)
    """
    R = T[:2, :2]
    p = T[:2,  2]
    T_inv = np.eye(3)
    T_inv[:2, :2] = R.T
    T_inv[:2,  2] = -R.T @ p
    return T_inv

def se2_log(T_err):
    """
    Matrix logarithm + vee operator:  log: SE(2) -> se(2) -> R³
    Implements the mapping described in slides 8 and 9.

    Given T_err = [[R_err, p_err], [0, 1]] ∈ SE(2):
      omega = arctan2(R_err[1,0], R_err[0,0])      <- rotation angle (slide 9)
      v     = A(omega)⁻¹ · p_err                   <- translational part
      xi_err = [v_x, v_y, omega] ∈ R³

    where A(omega)⁻¹ = [omega/(2(1-cos(omega)))] * [[sin, 1-cos], [-(1-cos), sin]]
    (inverse Rodrigues formula, slide 9)
    """
    R_err = T_err[:2, :2]
    p_err = T_err[:2,  2]
    omega = np.arctan2(R_err[1, 0], R_err[0, 0])   # theta_err wrapped to (-pi, pi]

    if abs(omega) < 1e-8:
        # Special case theta ≈ 0: A(omega) ≈ I  ->  v ≈ p_err
        v = p_err.copy()
    else:
        sin_w, cos_w = np.sin(omega), np.cos(omega)
        coeff = omega / (2.0 * (1.0 - cos_w))
        A_inv = coeff * np.array([[sin_w,       1 - cos_w],
                                   [-(1 - cos_w), sin_w   ]])
        v = A_inv @ p_err

    return np.array([v[0], v[1], omega])

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONTROL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
k_gain    = 2.0    # proportional gain
dt        = 0.02   # integration time step [s]
q_dot_max = 3.0    # joint velocity saturation limit [rad/s]
N_max     = 600    # maximum number of integration steps

# Initial and desired configurations — same elbow-up branch (q2 > 0 for both)
q0    = np.array([ np.pi/6,  np.pi/3,  np.pi/5])
q_des = np.array([ np.pi/2,  np.pi/6,  -np.pi/8])
T_d   = se2_matrix(q_des)    # desired transform: constant throughout simulation

print("="*60)
print("Kinematic control pipeline (SE(2) approach from slides)")
print("="*60)
print(f"q0    [deg]: {np.degrees(q0).round(2)}")
print(f"q_des [deg]: {np.degrees(q_des).round(2)}")
xi_d = fk(q_des)
print(f"xi_d       : x={xi_d[0]:.4f}, y={xi_d[1]:.4f}, theta={np.degrees(xi_d[2]):.2f} deg")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SIMULATION  — exact pipeline from the slides
# ─────────────────────────────────────────────────────────────────────────────
q = q0.copy()
q_hist    = [q.copy()]
xi_hist   = [fk(q)]
err_hist  = []              # xi_error sequence via matrix log
qdot_hist = [np.zeros(3)]
det_hist  = [abs(np.linalg.det(jacobian(q)))]

for step in range(N_max):
    # -- Step 1: T = FK(q)  -> SE(2) matrix
    T = se2_matrix(q)

    # -- Step 2: T_error = T_d · T⁻¹  ∈ SE(2)   (slide 7)
    T_err = T_d @ se2_inverse(T)

    # -- Step 3: xi_error = vee(log(T_error))  ∈ R³  (slides 8-9)
    xi_err = se2_log(T_err)
    err_hist.append(xi_err.copy())

    if np.linalg.norm(xi_err) < 5e-4:
        print(f"\n✓ Converged at t = {step*dt:.3f} s")
        print(f"  ||xi_err|| = {np.linalg.norm(xi_err):.6f}")
        q_hist.append(q.copy()); xi_hist.append(fk(q))
        qdot_hist.append(np.zeros(3)); det_hist.append(abs(np.linalg.det(jacobian(q))))
        break

    # -- Step 4: J(q)
    J = jacobian(q)

    # -- Step 5: J†(q) — exact inverse away from singularities; DLS near them
    det = abs(np.linalg.det(J))
    if det > 0.1:
        J_inv = np.linalg.inv(J)        # exact inverse (non-singular)
    else:
        lam   = 0.05                    # damping coefficient
        J_inv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))  # DLS

    # -- Step 6: q_dot = +k · J†(q) · xi_error   (positive sign with T_d · T⁻¹)
    #    Equivalent to the slide's -k if T_err = T · T_d⁻¹ is used instead
    q_dot = k_gain * J_inv @ xi_err

    # Joint velocity saturation
    speed = np.linalg.norm(q_dot)
    if speed > q_dot_max:
        q_dot *= q_dot_max / speed

    # Euler integration
    q += dt * q_dot

    q_hist.append(q.copy())
    xi_hist.append(fk(q))
    qdot_hist.append(q_dot.copy())
    det_hist.append(abs(np.linalg.det(jacobian(q))))

q_hist    = np.array(q_hist)
xi_hist   = np.array(xi_hist)
err_hist  = np.array(err_hist)
qdot_hist = np.array(qdot_hist)
det_hist  = np.array(det_hist)
err_norm  = np.linalg.norm(err_hist, axis=1)
time_vec  = np.arange(len(err_hist)) * dt
N_actual  = len(time_vec) - 1

print(f"Steps: {N_actual}  |  t_final = {time_vec[-1]:.3f} s")
print(f"Final error ||xi_err|| = {err_norm[-1]:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'axes.facecolor'  : '#0F172A',
    'figure.facecolor': '#0F172A',
    'text.color'      : '#E2E8F0',
    'axes.labelcolor' : '#94A3B8',
    'xtick.color'     : '#475569',
    'ytick.color'     : '#475569',
    'axes.edgecolor'  : '#334155',
    'grid.color'      : '#1E293B',
    'grid.linewidth'  : 0.8,
})

fig = plt.figure(figsize=(15, 9), facecolor='#0F172A')
fig.suptitle(
    r'Kinematic Control — Full SE(2) pipeline (following the slides)'
    '\n'
    r'$T_{\rm err}=T_d\cdot T^{-1}$,  '
    r'$\xi_{\rm err}=\mathrm{vee}(\log(T_{\rm err}))$,  '
    r'$\dot{q}=+k\,J^{\dagger}(q)\,\xi_{\rm err}$',
    fontsize=12, color='#F1F5F9', fontweight='bold', y=0.97)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       left=0.05, right=0.97, top=0.90, bottom=0.07,
                       wspace=0.38, hspace=0.65)

ax_r = fig.add_subplot(gs[:, 0])   # robot workspace
ax_e = fig.add_subplot(gs[0, 1:])  # error norm
ax_c = fig.add_subplot(gs[1, 1:])  # error components
ax_d = fig.add_subplot(gs[2, 1:])  # |det J|
ax_t = fig.add_subplot(gs[3, 1:])  # joint velocities

for ax in [ax_r, ax_e, ax_c, ax_d, ax_t]:
    ax.set_facecolor('#0F172A')
    ax.grid(True, linestyle='--', alpha=0.35)
    for sp in ['top', 'right']:
        ax.spines[sp].set_color('#1E293B')

# ── Robot workspace ───────────────────────────────────────────────────────────
R = sum(L) + 0.3
ax_r.set(xlim=(-R, R), ylim=(-R, R), aspect='equal',
         xlabel='x [m]', ylabel='y [m]')
ax_r.set_title('Workspace — SE(2) matrices', color='#94A3B8', fontsize=10)
ax_r.add_patch(plt.Circle((0, 0), sum(L), color='#334155',
                           fill=False, ls='--', lw=0.8, alpha=0.5))
ax_r.plot(0, 0, 'o', color='#475569', ms=10, zorder=5)   # base joint

# Desired pose (faded ghost)
pts_d = joint_positions(q_des)
for i in range(3):
    ax_r.plot([pts_d[i,0], pts_d[i+1,0]], [pts_d[i,1], pts_d[i+1,1]],
              '-', color='#22C55E', lw=1.5, alpha=0.3)
ax_r.plot(*pts_d[-1], '*', color='#22C55E', ms=22, zorder=6,
          label='Desired pose $T_d$')
ax_r.annotate('', xy=(pts_d[-1,0]+0.25*np.cos(xi_d[2]),
                       pts_d[-1,1]+0.25*np.sin(xi_d[2])),
              xytext=pts_d[-1],
              arrowprops=dict(arrowstyle='->', color='#22C55E', lw=2))

# Animated elements
traj_ln, = ax_r.plot([], [], '-', color='#38BDF8', lw=1.2,
                      alpha=0.55, zorder=3, label='EE trajectory')
link_lns = [ax_r.plot([], [], '-', color=c, lw=7,
                       solid_capstyle='round', zorder=4)[0]
            for c in LINK_COLORS]
joint_ds = [ax_r.plot([], [], 'o', color='#F59E0B', ms=11, zorder=7)[0]
            for _ in range(3)]
ee_d,   = ax_r.plot([], [], 'o', color='#F43F5E', ms=14, zorder=8,
                     label='End-effector')
oa = ax_r.annotate('', xy=(0, 0), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='#F43F5E', lw=2.5))
info = ax_r.text(0.02, 0.98, '', transform=ax_r.transAxes,
                  fontsize=8, va='top', color='#CBD5E1',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='#1E293B', alpha=0.88))
ax_r.text(0.02, 0.02,
    r'$T_{\rm err}=T_d\cdot T^{-1}$'
    + '\n' + r'$\xi_{\rm err}=\mathrm{vee}(\log(T_{\rm err}))$'
    + '\n' + r'$\dot{q}=+k\,J^{-1}\xi_{\rm err}$'
    + f'\n$k={k_gain}$',
    transform=ax_r.transAxes, fontsize=9, va='bottom',
    color='#FCD34D', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E293B',
              edgecolor='#F59E0B', alpha=0.92))
ax_r.legend(loc='lower right', fontsize=8, facecolor='#1E293B',
            labelcolor='#CBD5E1', edgecolor='#334155')

# ── Error norm (log scale) ────────────────────────────────────────────────────
ax_e.set_title(r'$\|\xi_{\rm err}\|=\|\mathrm{vee}(\log(T_d\cdot T^{-1}))\|$ — log scale',
               color='#94A3B8', fontsize=10)
ax_e.set_ylabel(r'$\|\xi_{\rm err}\|$', fontsize=9)
ax_e.semilogy(time_vec, err_norm+1e-9, color='#475569', lw=1, alpha=0.2)
t_th = np.linspace(0, time_vec[-1], 400)
ax_e.semilogy(t_th, err_norm[0]*np.exp(-k_gain*t_th),
              '--', color='#34D399', lw=1.5, alpha=0.6,
              label=r'Theory: $e^{-kt}$')
err_ln, = ax_e.semilogy([], [], '-', color='#F43F5E', lw=2,
                         label='Simulation (matrix log)')
err_dt, = ax_e.plot([], [], 'o', color='#FBBF24', ms=7, zorder=5)
ax_e.set_xlim(0, time_vec[-1])
ax_e.set_ylim(max(1e-5, err_norm.min()*0.5), err_norm[0]*2)
ax_e.legend(fontsize=8, facecolor='#1E293B', labelcolor='#CBD5E1',
            edgecolor='#334155')

# ── Error components ──────────────────────────────────────────────────────────
ax_c.set_title(r'Error components $\xi_{\rm err}=[v_x,\,v_y,\,\omega]$ via SE(2) log',
               color='#94A3B8', fontsize=10)
clrs = ['#38BDF8', '#A78BFA', '#F472B6']
lbls = [r'$v_x$ [m]', r'$v_y$ [m]', r'$\omega$ [rad]']
for i, (c, l) in enumerate(zip(clrs, lbls)):
    ax_c.plot(time_vec, err_hist[:, i], color=c, lw=1, alpha=0.18)
comp_lns = [ax_c.plot([], [], '-', color=c, lw=2, label=l)[0]
            for c, l in zip(clrs, lbls)]
comp_ds  = [ax_c.plot([], [], 'o', color=c, ms=5)[0] for c in clrs]
ax_c.axhline(0, color='#475569', lw=0.8, ls='--')
ax_c.set_xlim(0, time_vec[-1])
ax_c.set_ylim(err_hist.min()*1.3, err_hist.max()*1.3)
ax_c.legend(fontsize=8, facecolor='#1E293B', labelcolor='#CBD5E1',
            edgecolor='#334155', loc='upper right', ncol=3)

# ── |det J| — singularity indicator ──────────────────────────────────────────
ax_d.set_title(r'$|\det J(q)|$ — approaches zero at singularities',
               color='#94A3B8', fontsize=10)
ax_d.plot(time_vec, det_hist[:-1], color='#475569', lw=1, alpha=0.2)
det_ln, = ax_d.plot([], [], '-', color='#34D399', lw=2)
det_dt, = ax_d.plot([], [], 'o', color='#FBBF24', ms=6)
ax_d.axhline(0.1, color='#F59E0B', lw=1, ls=':', alpha=0.8)
ax_d.text(time_vec[-1]*0.02, 0.12, 'DLS active below this line',
          color='#F59E0B', fontsize=8)
ax_d.set_xlim(0, time_vec[-1])
ax_d.set_ylim(-0.02, max(det_hist)*1.15)

# ── Joint velocities ──────────────────────────────────────────────────────────
ax_t.set_title(r'Joint velocities $\dot{q}$ [rad/s] — saturation limit $\pm$'
               + f'{q_dot_max} (dotted line)', color='#94A3B8', fontsize=10)
ax_t.set_xlabel('Time [s]', fontsize=9)
qclrs = ['#3B82F6', '#8B5CF6', '#EC4899']
qlbls = [r'$\dot{q}_1$', r'$\dot{q}_2$', r'$\dot{q}_3$']
for i, (c, l) in enumerate(zip(qclrs, qlbls)):
    ax_t.plot(time_vec, qdot_hist[:-1, i], color=c, lw=1, alpha=0.18)
qdot_lns = [ax_t.plot([], [], '-', color=c, lw=2, label=l)[0]
            for c, l in zip(qclrs, qlbls)]
ax_t.axhline( q_dot_max, color='#94A3B8', lw=1, ls=':', alpha=0.7)
ax_t.axhline(-q_dot_max, color='#94A3B8', lw=1, ls=':', alpha=0.7)
ax_t.set_xlim(0, time_vec[-1])
ymax = max(abs(qdot_hist).max()*1.2, q_dot_max*1.2)
ax_t.set_ylim(-ymax, ymax)
ax_t.legend(fontsize=8, facecolor='#1E293B', labelcolor='#CBD5E1',
            edgecolor='#334155', loc='upper right', ncol=3)

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANIMATION
# ─────────────────────────────────────────────────────────────────────────────
SKIP = 3   # frames skipped per animation step (playback speed)

def init():
    for ln in link_lns: ln.set_data([], [])
    for jd in joint_ds: jd.set_data([], [])
    ee_d.set_data([], []); traj_ln.set_data([], [])
    err_ln.set_data([], []); err_dt.set_data([], [])
    for c in comp_lns: c.set_data([], [])
    for d in comp_ds:  d.set_data([], [])
    det_ln.set_data([], []); det_dt.set_data([], [])
    for ql in qdot_lns: ql.set_data([], [])
    return (link_lns + joint_ds + [ee_d, traj_ln, err_ln, err_dt]
            + comp_lns + comp_ds + [det_ln, det_dt] + qdot_lns)

def update(frame):
    i = min(frame*SKIP, N_actual-1)
    qi = q_hist[i]; xii = xi_hist[i]; p = joint_positions(qi)
    t  = time_vec[:i+1]

    # Update robot links and joints
    for ki, ln in enumerate(link_lns):
        ln.set_data([p[ki,0], p[ki+1,0]], [p[ki,1], p[ki+1,1]])
    for ki, jd in enumerate(joint_ds):
        jd.set_data([p[ki+1,0]], [p[ki+1,1]])
    ee_d.set_data([xii[0]], [xii[1]])
    oa.set_position(p[-1])
    oa.xy = (p[-1,0]+0.30*np.cos(xii[2]), p[-1,1]+0.30*np.sin(xii[2]))
    traj_ln.set_data(xi_hist[:i+1, 0], xi_hist[:i+1, 1])

    # Update charts
    err_ln.set_data(t, err_norm[:i+1]+1e-9)
    err_dt.set_data([t[-1]], [err_norm[i]+1e-9])
    for ki, (cl, cd) in enumerate(zip(comp_lns, comp_ds)):
        cl.set_data(t, err_hist[:i+1, ki])
        cd.set_data([t[-1]], [err_hist[i, ki]])
    det_ln.set_data(t, det_hist[:i+1])
    det_dt.set_data([t[-1]], [det_hist[i]])
    for ki, ql in enumerate(qdot_lns):
        ql.set_data(t, qdot_hist[:i+1, ki])

    # Recompute xi_err at current step for the info box
    T_curr    = se2_matrix(qi)
    T_err_now = T_d @ se2_inverse(T_curr)
    xi_e_now  = se2_log(T_err_now)
    mode = 'exact J⁻¹' if det_hist[i] > 0.1 else 'DLS active'
    info.set_text(
        f"  t = {time_vec[i]:.3f} s\n"
        f"  ||xi_err|| = {err_norm[i]:.5f}\n"
        f"  q1 = {np.degrees(qi[0]):+7.2f} deg\n"
        f"  q2 = {np.degrees(qi[1]):+7.2f} deg\n"
        f"  q3 = {np.degrees(qi[2]):+7.2f} deg\n"
        f"  |det J| = {det_hist[i]:.4f}\n"
        f"  Mode: {mode}\n"
        f"  vx={xi_e_now[0]:.3f}  vy={xi_e_now[1]:.3f}\n"
        f"  omega={np.degrees(xi_e_now[2]):.2f} deg"
    )
    return (link_lns + joint_ds + [ee_d, traj_ln, err_ln, err_dt, info]
            + comp_lns + comp_ds + [det_ln, det_dt] + qdot_lns)

anim = FuncAnimation(fig, update, frames=N_actual//SKIP+1,
                     init_func=init, interval=20, blit=True)

# ─────────────────────────────────────────────────────────────────────────────
# Display: HTML in Jupyter/Colab, interactive window otherwise
# ─────────────────────────────────────────────────────────────────────────────
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        from IPython.display import HTML, display
        display(HTML(anim.to_jshtml(fps=30)))
        print("Animation ready (Jupyter/Colab).")
    else:
        raise RuntimeError
except Exception:
    plt.tight_layout()
    plt.show()