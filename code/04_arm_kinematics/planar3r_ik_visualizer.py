# ══════════════════════════════════════════════════════════════
#  3-DOF Planar Robot Arm (3R) — Inverse Kinematics Visualizer
#
#  Google Colab:
#   Cell 1:  !pip install ipympl -q
#            then Runtime → Restart session
#   Cell 2:  from google.colab import output
#            output.enable_custom_widget_manager()
#            %matplotlib widget
#   Cell 3:  this script
#
#  Local (Jupyter): %matplotlib widget   |   Script: python file.py
# ══════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec

# ─── Robot parameters ─────────────────────────────────────────
A1, A2, A3 = 1.0, 1.0, 1.0      # link lengths a_i (m)
D1 = 0.5                        # base height d_1 (m): only shifts z, the motion is planar
DH = [
    dict(d=D1,  a=A1, alpha=0.0),
    dict(d=0.0, a=A2, alpha=0.0),
    dict(d=0.0, a=A3, alpha=0.0),
]
ELBOW_DOWN, ELBOW_UP = +1, -1   # sign of sin(q2)


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a, dtype=float) + np.pi) % (2*np.pi) - np.pi


# ─── Forward kinematics (standard DH) ─────────────────────────
def dh_matrix(theta, d, a, alpha):
    """A_i = Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1],
    ])


def fk(q):
    """Returns T_3^0 (4x4) and the planar positions of base, J2, J3, EE."""
    T = np.eye(4)
    pts = [np.zeros(2)]
    for qi, p in zip(q, DH):
        T = T @ dh_matrix(qi, p['d'], p['a'], p['alpha'])
        pts.append(T[:2, 3].copy())
    return T, np.array(pts)


# ─── Inverse kinematics (closed form, wrist decoupling) ───────
def ik(tx, ty, phi=0.0, elbow=ELBOW_DOWN):
    """Analytical IK of the planar 3R arm.

    1) Wrist point:   w = p* - a3 [cos phi, sin phi]
    2) 2R problem:    cos q2 = (|w|^2 - a1^2 - a2^2) / (2 a1 a2)
                      q2 = atan2(+-sqrt(1 - c2^2), c2)
                      q1 = atan2(wy, wx) - atan2(a2 s2, a1 + a2 c2)
    3) Orientation:   q3 = phi - q1 - q2

    Returns (q, w). q is None if the target is unreachable.
    """
    w = np.array([tx - A3*np.cos(phi), ty - A3*np.sin(phi)])
    c2 = (w @ w - A1**2 - A2**2) / (2*A1*A2)
    if abs(c2) > 1.0 + 1e-9:
        return None, w
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = elbow*np.sqrt(1.0 - c2**2)
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(w[1], w[0]) - np.arctan2(A2*s2, A1 + A2*c2)
    q3 = phi - q1 - q2
    return wrap([q1, q2, q3]), w


def format_matrix(T):
    rows = ['[ ' + '  '.join(f'{v:+.3f}' for v in r) + ' ]' for r in T]
    return '\n'.join(rows)


# ─── Trajectory ───────────────────────────────────────────────
def eased(qa, qb, n=70):
    """Joint-space smoothstep (3t^2 - 2t^3), shortest angular path."""
    t = np.linspace(0, 1, n)
    s = 3*t**2 - 2*t**3
    return qa + np.outer(s, wrap(qb - qa))


def hold(q, n):
    return np.repeat(np.asarray(q)[None, :], n, axis=0)


q_home = np.zeros(3)
q_A = np.array([0.52, 1.31, -0.82])
q_B = np.array([-0.28, 0.85, 1.18])


def build_sequence(tx, ty, phi, elbow):
    q_sol, w = ik(tx, ty, phi, elbow)
    segs = [
        (eased(q_home, q_A, 70), '1 - Forward kinematics',
         'Interpolating home -> pose A (smoothstep)'),
        (eased(q_A, q_B, 70), '2 - Workspace exploration',
         'Interpolating pose A -> pose B'),
    ]
    if q_sol is not None:
        err = np.linalg.norm(fk(q_sol)[1][-1] - [tx, ty])
        segs += [
            (eased(q_B, q_sol, 90), '3 - Inverse kinematics',
             f'Closed-form IK -> p* = ({tx:.2f}, {ty:.2f}), phi* = {np.degrees(phi):.1f} deg'),
            (hold(q_sol, 60), '3 - Inverse kinematics',
             f'Target reached. Check: ||FK(q*) - p*|| = {err:.1e} m'),
        ]
    else:
        segs.append((hold(q_B, 120), '3 - Target UNREACHABLE',
                     f'||w|| = {np.linalg.norm(w):.2f} m > a1 + a2 = {A1 + A2:.1f} m\n'
                     f'Move the target or change phi'))
    traj = np.vstack([s[0] for s in segs])
    phases = sum([[s[1]]*len(s[0]) for s in segs], [])
    msgs = sum([[s[2]]*len(s[0]) for s in segs], [])
    return traj, phases, msgs


# ─── State ────────────────────────────────────────────────────
state = dict(frame=0, manual=False, paused=False,
             tx=1.5, ty=1.0, phi=0.0, elbow=ELBOW_DOWN)
traj, phases, msgs = build_sequence(state['tx'], state['ty'], state['phi'], state['elbow'])

# ─── Figure layout ────────────────────────────────────────────
plt.style.use('dark_background')
TEAL = '#1D9E75'; AMBER = '#EF9F27'; CORAL = '#D85A30'; MINT = '#5DCAA5'
TXTC = '#d0cec6'; DIM = '#77776f'
BG = '#1a1a18'; PAN = '#141412'; BOX = '#1c1c1a'; EDGE = '#3a3a35'

fig = plt.figure(figsize=(16, 9), facecolor=BG)
gs = GridSpec(4, 3, figure=fig, left=0.03, right=0.985, top=0.93, bottom=0.22,
              hspace=0.50, wspace=0.28)
ax_robot = fig.add_subplot(gs[:, :2])
ax_mat = fig.add_subplot(gs[0, 2])
ax_info = fig.add_subplot(gs[1:3, 2])
ax_dh = fig.add_subplot(gs[3, 2])

# ── Robot canvas ──────────────────────────────────────────────
ax_robot.set_facecolor(PAN)
ax_robot.set_xlim(-3.5, 3.5); ax_robot.set_ylim(-2.8, 2.8)
ax_robot.set_aspect('equal')
ax_robot.grid(True, color='#252522', lw=0.6)
ax_robot.axhline(0, color='#333330', lw=1.0); ax_robot.axvline(0, color='#333330', lw=1.0)
ax_robot.set_xlabel(r'$x$ (m)', color=TXTC, fontsize=9)
ax_robot.set_ylabel(r'$y$ (m)', color=TXTC, fontsize=9)
ax_robot.tick_params(colors=DIM, labelsize=8)
for sp in ax_robot.spines.values():
    sp.set_color('#333330')

th_c = np.linspace(0, 2*np.pi, 300)
R_MAX = A1 + A2 + A3
ax_robot.plot(R_MAX*np.cos(th_c), R_MAX*np.sin(th_c), '--', color='#3a3a35', lw=0.8,
              zorder=1, label='max reach  $a_1+a_2+a_3$')
# Positions reachable WITH orientation phi: circle of radius a1+a2 centred at a3*[cos phi, sin phi]
reach_phi, = ax_robot.plot([], [], ':', color=MINT, lw=1.2, zorder=1,
                           label=r'reachable with $\phi^*$')

# Target, wrist point and alternative IK solution (ghost)
tgt_marker, = ax_robot.plot([], [], 'o', color=CORAL, ms=6, zorder=8)
tgt_arrow = ax_robot.annotate('', xy=(0, 0), xytext=(0, 0), zorder=8,
                              arrowprops=dict(arrowstyle='->', color=CORAL, lw=2.0))
tgt_label = ax_robot.text(0, 0, '', fontsize=8, color=CORAL, zorder=8)
wrist_mk, = ax_robot.plot([], [], 'x', color=MINT, ms=9, mew=2, zorder=8,
                          label=r'wrist $w = p^* - a_3[\cos\phi^*, \sin\phi^*]$')
wrist_seg, = ax_robot.plot([], [], '--', color=MINT, lw=1.0, zorder=7)
ghost, = ax_robot.plot([], [], '--o', color='#8a8a80', lw=1.5, ms=4, alpha=0.5, zorder=2,
                       label='other IK solution')

trace_line, = ax_robot.plot([], [], '-', color=TEAL + '55', lw=1.2, zorder=2)
trace_x, trace_y = [], []

lc = [TEAL, '#0F6E56', '#085041']
jc = [AMBER, '#BA7517', '#854F0B']
links = [ax_robot.plot([], [], '-', color=c, lw=6 - i, solid_capstyle='round', zorder=3)[0]
         for i, c in enumerate(lc)]
llbls = [ax_robot.text(0, 0, f'$L_{i+1}$', fontsize=8, color=TXTC, zorder=4) for i in range(3)]
jdots = [ax_robot.plot([], [], 'o', color=c, ms=9 - 2*i, zorder=5)[0] for i, c in enumerate(jc)]
jlbls = [ax_robot.text(0, 0, f'$J_{i+1}$', fontsize=8, color=jc[i], zorder=6) for i in range(3)]
ee_dot, = ax_robot.plot([], [], 'o', color=CORAL, ms=7, zorder=7)
ee_arr = ax_robot.annotate('', xy=(0, 0), xytext=(0, 0), zorder=6,
                           arrowprops=dict(arrowstyle='->', color=TXTC, lw=1.5))

ax_robot.fill_between([-0.22, 0.22], [-0.08, -0.08], [0, 0], color='#555548', zorder=4)
for bx in np.linspace(-0.18, 0.18, 5):
    ax_robot.plot([bx, bx - 0.06], [-0.08, -0.16], color='#666658', lw=1.5, zorder=3)

box = dict(boxstyle='round,pad=0.4', fc=BOX, ec=EDGE, alpha=0.92)
st_title = ax_robot.text(0.02, 0.985, '', transform=ax_robot.transAxes, fontsize=9,
                         color=AMBER, va='top', fontweight='bold', bbox=box)
st_msg = ax_robot.text(0.02, 0.928, '', transform=ax_robot.transAxes, fontsize=7.5,
                       color=TXTC, va='top', family='monospace', bbox=box)
st_num = ax_robot.text(0.02, 0.80, '', transform=ax_robot.transAxes, fontsize=7.5,
                       color=TXTC, va='top', family='monospace', bbox=box)
pause_overlay = ax_robot.text(0.5, 0.5, '', transform=ax_robot.transAxes, fontsize=24,
                              color='#ffffffaa', va='center', ha='center', fontweight='bold')
ax_robot.legend(loc='lower right', fontsize=7.5, facecolor=BOX, edgecolor=EDGE,
                labelcolor=TXTC, framealpha=0.92)
ax_robot.set_title('3-DOF Planar Robot Arm - Inverse Kinematics', color=TXTC, fontsize=10, pad=8)

# ── Side panels ───────────────────────────────────────────────
for ax in (ax_mat, ax_info, ax_dh):
    ax.set_facecolor(PAN); ax.axis('off')

ax_mat.set_title(r'${}^{0}T_{3}$  (homogeneous transform)', color=TXTC, fontsize=9, pad=4)
mat_text = ax_mat.text(0.04, 0.90, '', transform=ax_mat.transAxes, fontsize=8.5,
                       color=TEAL, va='top', family='monospace')

ax_info.set_title('End-effector, joints and IK', color=TXTC, fontsize=9, pad=4)
info_text = ax_info.text(0.04, 0.97, '', transform=ax_info.transAxes, fontsize=8.5,
                         color=TXTC, va='top', family='monospace')

ax_dh.set_title('DH parameters', color=TXTC, fontsize=9, pad=4)
dh_rows = [[str(i + 1), f"{p['a']:.1f}", f"{p['d']:.1f}", '0', f'$q_{i+1}$']
           for i, p in enumerate(DH)]
tbl = ax_dh.table(cellText=dh_rows, colLabels=['i', r'$a_i$ (m)', r'$d_i$ (m)',
                  r'$\alpha_i$', r'$\theta_i$'], loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1e1e1c' if r % 2 == 0 else '#232320')
    cell.set_edgecolor('#333330')
    cell.set_text_props(color=AMBER if (c == 0 and r > 0) else TXTC)
    if r == 0:
        cell.set_facecolor('#2a2a26')
        cell.set_text_props(color='#9a9a92', fontweight='bold')

# ─── Widgets ──────────────────────────────────────────────────
SC = '#252522'


def make_slider(rect, label, vmin, vmax, v0, col):
    s = Slider(fig.add_axes(rect, facecolor=SC), label, vmin, vmax,
               valinit=v0, color=col, initcolor='none')
    s.label.set_color(col); s.label.set_fontsize(9)
    s.valtext.set_color(TXTC); s.valtext.set_fontsize(8)
    return s


sl1 = make_slider([0.05, 0.125, 0.21, 0.018], r'$q_1$', -np.pi, np.pi, 0, AMBER)
sl2 = make_slider([0.05, 0.095, 0.21, 0.018], r'$q_2$', -np.pi, np.pi, 0, AMBER)
sl3 = make_slider([0.05, 0.065, 0.21, 0.018], r'$q_3$', -np.pi, np.pi, 0, AMBER)
sl_tx = make_slider([0.34, 0.125, 0.21, 0.018], r'$x^*$', -2.9, 2.9, state['tx'], CORAL)
sl_ty = make_slider([0.34, 0.095, 0.21, 0.018], r'$y^*$', -2.9, 2.9, state['ty'], CORAL)
sl_phi = make_slider([0.34, 0.065, 0.21, 0.018], r'$\phi^*$ (rad)', -np.pi, np.pi, state['phi'], MINT)

fig.text(0.05, 0.152, 'Joint angles (FK) - drag to override the animation', color=DIM, fontsize=7.5)
fig.text(0.34, 0.152, 'IK target pose (rebuilds and restarts the sequence)', color=DIM, fontsize=7.5)


def make_button(rect, label):
    b = Button(fig.add_axes(rect, facecolor=SC), label, color=SC, hovercolor='#3a3a35')
    b.label.set_color(TXTC); b.label.set_fontsize(8.5)
    return b


btn_play = make_button([0.615, 0.062, 0.07, 0.058], 'Play')
btn_pause = make_button([0.695, 0.062, 0.07, 0.058], 'Pause')
btn_home = make_button([0.775, 0.062, 0.07, 0.058], 'Home')

ax_rad = fig.add_axes([0.86, 0.045, 0.125, 0.09], facecolor=SC)
rad_elbow = RadioButtons(ax_rad, ('elbow down  (q2 > 0)', 'elbow up  (q2 < 0)'),
                         active=0, activecolor=CORAL)
for lb in rad_elbow.labels:
    lb.set_color(TXTC); lb.set_fontsize(8)

fig.text(0.05, 0.025, 'Key: P = pause / resume', color=DIM, fontsize=7.5)


# ─── Drawing ──────────────────────────────────────────────────
def draw_target():
    tx, ty, phi, elbow = state['tx'], state['ty'], state['phi'], state['elbow']
    tgt_marker.set_data([tx], [ty])
    tgt_label.set_text(f'p* = ({tx:.2f}, {ty:.2f}),  phi* = {np.degrees(phi):.1f} deg')
    tgt_label.set_position((tx + 0.1, ty + 0.12))
    al = 0.45
    tgt_arrow.xy = (tx + al*np.cos(phi), ty + al*np.sin(phi))
    tgt_arrow.xyann = (tx, ty)

    c = A3*np.array([np.cos(phi), np.sin(phi)])
    reach_phi.set_data(c[0] + (A1 + A2)*np.cos(th_c), c[1] + (A1 + A2)*np.sin(th_c))

    q, w = ik(tx, ty, phi, elbow)
    wrist_mk.set_data([w[0]], [w[1]])
    wrist_seg.set_data([w[0], tx], [w[1], ty])
    q_alt, _ = ik(tx, ty, phi, -elbow)
    if q_alt is not None:
        _, p_alt = fk(q_alt)
        ghost.set_data(p_alt[:, 0], p_alt[:, 1])
    else:
        ghost.set_data([], [])
    tgt_marker.set_color(CORAL if q is not None else '#ff4040')


def update_robot(q):
    T, pts = fk(q)
    phi = float(wrap(np.sum(q)))
    ee = pts[-1]

    trace_x.append(ee[0]); trace_y.append(ee[1])
    trace_line.set_data(trace_x, trace_y)
    for i in range(3):
        links[i].set_data(pts[i:i + 2, 0], pts[i:i + 2, 1])
        m = 0.5*(pts[i] + pts[i + 1]) + 0.05
        llbls[i].set_position(m)
        jdots[i].set_data([pts[i, 0]], [pts[i, 1]])
        jlbls[i].set_position(pts[i] + 0.08)
    ee_dot.set_data([ee[0]], [ee[1]])
    ee_arr.xy = (ee[0] + 0.35*np.cos(phi), ee[1] + 0.35*np.sin(phi))
    ee_arr.xyann = (ee[0], ee[1])

    mat_text.set_text(format_matrix(T))

    err_p = np.hypot(ee[0] - state['tx'], ee[1] - state['ty'])
    err_phi = np.degrees(wrap(phi - state['phi']))
    q_sol, w = ik(state['tx'], state['ty'], state['phi'], state['elbow'])
    sol = ('[' + ', '.join(f'{v:+.3f}' for v in q_sol) + ']') if q_sol is not None else 'unreachable'
    info_text.set_text(
        f'x   = {ee[0]:+.4f} m\n'
        f'y   = {ee[1]:+.4f} m\n'
        f'phi = {np.degrees(phi):+.2f} deg\n'
        f'-----------------------------\n'
        f'q1  = {q[0]:+.4f} rad\n'
        f'q2  = {q[1]:+.4f} rad\n'
        f'q3  = {q[2]:+.4f} rad\n'
        f'-----------------------------\n'
        f'wrist w   = ({w[0]:+.3f}, {w[1]:+.3f})\n'
        f'||w||     = {np.linalg.norm(w):.3f}  (<= {A1 + A2:.1f})\n'
        f'q* (IK)   = {sol}\n'
        f'|p - p*|  = {err_p:.4f} m\n'
        f'phi - phi* = {err_phi:+.2f} deg'
    )
    st_num.set_text(f'q  = [{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}] rad\n'
                    f'EE = ({ee[0]:.3f}, {ee[1]:.3f}) m   phi = {np.degrees(phi):.1f} deg')


def sync_sliders(q):
    for sl, v in zip((sl1, sl2, sl3), q):
        sl.eventson = False
        sl.set_val(float(wrap(v)))
        sl.eventson = True


def clear_trace():
    trace_x.clear(); trace_y.clear()
    trace_line.set_data([], [])


def restart_sequence():
    global traj, phases, msgs
    traj, phases, msgs = build_sequence(state['tx'], state['ty'], state['phi'], state['elbow'])
    clear_trace()
    state.update(frame=0, manual=False, paused=False)
    pause_overlay.set_text('')
    btn_pause.label.set_text('Pause')
    draw_target()
    fig.canvas.draw_idle()


# ─── Callbacks ────────────────────────────────────────────────
def on_joint_slider(_):
    state['manual'] = True
    q = np.array([sl1.val, sl2.val, sl3.val])
    update_robot(q)
    st_title.set_text('Manual control (FK)')
    st_msg.set_text('Drag the joint sliders: FK updates in real time.\n'
                    'Try to reach the target by hand, then press Play.')
    fig.canvas.draw_idle()


def on_target_slider(_):
    state.update(tx=sl_tx.val, ty=sl_ty.val, phi=sl_phi.val)
    restart_sequence()


def on_elbow(label):
    state['elbow'] = ELBOW_DOWN if label.startswith('elbow down') else ELBOW_UP
    restart_sequence()


def on_pause(_):
    state['paused'] = not state['paused']
    pause_overlay.set_text('PAUSED' if state['paused'] else '')
    btn_pause.label.set_text('Resume' if state['paused'] else 'Pause')
    fig.canvas.draw_idle()


def on_home(_):
    state.update(manual=True, paused=False)
    pause_overlay.set_text('')
    btn_pause.label.set_text('Pause')
    clear_trace()
    sync_sliders(q_home)
    on_joint_slider(None)


def on_key(event):
    if event.key in ('p', 'P'):
        on_pause(None)


for s in (sl1, sl2, sl3):
    s.on_changed(on_joint_slider)
for s in (sl_tx, sl_ty, sl_phi):
    s.on_changed(on_target_slider)
rad_elbow.on_clicked(on_elbow)
btn_play.on_clicked(lambda _: restart_sequence())
btn_pause.on_clicked(on_pause)
btn_home.on_clicked(on_home)
fig.canvas.mpl_connect('key_press_event', on_key)


# ─── Animation loop ───────────────────────────────────────────
def animate(_):
    if state['paused'] or state['manual']:
        return []
    f = state['frame'] % len(traj)
    if f == 0:
        clear_trace()
    state['frame'] += 1
    q = traj[f]
    update_robot(q)
    sync_sliders(q)
    st_title.set_text(phases[f])
    st_msg.set_text(msgs[f])
    return []


draw_target()
update_robot(traj[0])
ani = animation.FuncAnimation(fig, animate, interval=30, blit=False, cache_frame_data=False)
fig.suptitle('Robotics  |  3R Planar Arm  |  Closed-form IK (wrist decoupling)  |  NumPy / Matplotlib',
             color='#555550', fontsize=8, y=0.975)
plt.show()
