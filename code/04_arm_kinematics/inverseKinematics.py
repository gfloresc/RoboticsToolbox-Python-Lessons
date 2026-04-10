# ══════════════════════════════════════════════════════════════
#  3-DOF Planar Robot Arm — Kinematics Visualizer
#  Google Colab: run these two lines first in a separate cell:
#   !pip install ipympl -q
#
#.   Runtime → Restart session with Ctrl + M .
#    %matplotlib widget
#.   from google.colab import output
#    output.enable_custom_widget_manager()
#    GoogleColab version: https://colab.research.google.com/drive/1LPXK3oISiJZRYOKpaapc7MDxERXzn5aK#scrollTo=1hZP8kOU29Ly
# ══════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

# ─── DH kinematics (pure NumPy, no extra libraries) ───────────
def dh_matrix(theta, d, a, alpha):
    """Standard DH 4×4 homogeneous transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])

# Robot DH parameters: 3 links, each 1 m, planar (alpha=0)
DH = [
    dict(d=0.5, a=1.0, alpha=0.0),   # Link 1
    dict(d=0.0, a=1.0, alpha=0.0),   # Link 2
    dict(d=0.0, a=1.0, alpha=0.0),   # Link 3
]

def fk(q):
    """Forward kinematics.
    Returns T_04 (4×4) and list of joint positions [(x,y), ...]."""
    T = np.eye(4)
    pts = [(0.0, 0.0)]
    for qi, p in zip(q, DH):
        T = T @ dh_matrix(qi, p['d'], p['a'], p['alpha'])
        pts.append((T[0, 3], T[1, 3]))
    return T, pts

def ik(tx, ty, phi=0.0):
    """Analytical IK for planar 3R robot (elbow-up).
    Returns q* = [q1, q2, q3] or None if target is unreachable."""
    a1, a2, a3 = 1.0, 1.0, 1.0
    wx = tx - a3 * np.cos(phi)
    wy = ty - a3 * np.sin(phi)
    c2 = (wx**2 + wy**2 - a1**2 - a2**2) / (2 * a1 * a2)
    if abs(c2) > 1.0:
        return None
    s2 = np.sqrt(1 - c2**2)
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(wy, wx) - np.arctan2(a2*s2, a1 + a2*c2)
    q3 = phi - q1 - q2
    return np.array([q1, q2, q3])

def format_matrix(T):
    """Format 4×4 matrix as a readable string."""
    rows = [
        '⎡ {:+.3f}  {:+.3f}  {:+.3f}  {:+.3f} ⎤'.format(*T[0]),
        '⎢ {:+.3f}  {:+.3f}  {:+.3f}  {:+.3f} ⎥'.format(*T[1]),
        '⎢ {:+.3f}  {:+.3f}  {:+.3f}  {:+.3f} ⎥'.format(*T[2]),
        '⎣ {:+.3f}  {:+.3f}  {:+.3f}  {:+.3f} ⎦'.format(*T[3]),
    ]
    return '\n'.join(rows)

# ─── Trajectory helpers ───────────────────────────────────────
def eased(qa, qb, n=70):
    """Smooth ease-in-out interpolation between two configs."""
    t = np.linspace(0, 1, n)
    e = np.where(t < 0.5, 2*t**2, -1 + (4 - 2*t)*t)
    return np.outer(e, qb - qa) + qa

q_home = np.zeros(3)
q_A    = np.array([ 0.52,  1.31, -0.82])
q_B    = np.array([-0.28,  0.85,  1.18])

def build_sequence(tx, ty, phi=0.0):
    """Build full animation trajectory for a given IK target."""
    q_sol = ik(tx, ty, phi) if ik(tx, ty, phi) is not None else q_home.copy()
    segs = [
        (eased(q_home, q_A,  70),
         '1 — Forward kinematics',
         'Interpolating home → pose A   (ease-in-out)',
         None),
        (eased(q_A,    q_B,  70),
         '2 — Workspace exploration',
         'Interpolating pose A → pose B',
         None),
        (eased(q_B, q_sol, 90),
        '3 — Inverse kinematics',
        f'Analytical IK → target ({tx:.2f}, {ty:.2f})  φ={np.degrees(phi):.1f}°',
        (tx, ty, phi)),   # ← agrega phi aqui
    ]
    traj   = np.vstack([s[0] for s in segs])
    phases = sum([[s[1]] * len(s[0]) for s in segs], [])
    msgs   = sum([[s[2]] * len(s[0]) for s in segs], [])
    tgts   = sum([[s[3]] * len(s[0]) for s in segs], [])
    return traj, phases, msgs, tgts

traj, phases, msgs, tgts = build_sequence(1.5, 1.0)

# ─── Figure layout ────────────────────────────────────────────
plt.style.use('dark_background')
TEAL = '#1D9E75'; AMBER = '#EF9F27'; CORAL = '#D85A30'
TXTC = '#d0cec6'; DIM   = '#666660'
BG   = '#1a1a18'; PAN   = '#141412'

fig = plt.figure(figsize=(16, 9), facecolor=BG)
gs  = GridSpec(4, 3, figure=fig,
               left=0.03, right=0.985, top=0.93, bottom=0.17,
               hspace=0.50, wspace=0.28)

ax_robot = fig.add_subplot(gs[:4, :2])   # main canvas
ax_mat   = fig.add_subplot(gs[0:2, 2])   # T matrix
ax_info  = fig.add_subplot(gs[2, 2])     # EE + joint info
ax_dh    = fig.add_subplot(gs[3, 2])     # DH table

# ── Robot canvas ──────────────────────────────────────────────
ax_robot.set_facecolor(PAN)
ax_robot.set_xlim(-3.5, 3.5)
ax_robot.set_ylim(-2.8, 2.8)
ax_robot.set_aspect('equal')
ax_robot.grid(True, color='#252522', lw=0.6)
ax_robot.axhline(0, color='#333330', lw=1.0)
ax_robot.axvline(0, color='#333330', lw=1.0)
ax_robot.set_xlabel('x (m)', color=TXTC, fontsize=9)
ax_robot.set_ylabel('y (m)', color=TXTC, fontsize=9)
ax_robot.tick_params(colors=DIM, labelsize=8)
for sp in ax_robot.spines.values():
    sp.set_color('#333330')

# Max-reach circle
th_c = np.linspace(0, 2*np.pi, 200)
ax_robot.plot(3*np.cos(th_c), 3*np.sin(th_c),
              '--', color='#2e2e2a', lw=0.8, zorder=1)

# IK target marker
tgt_marker, = ax_robot.plot([1.5], [1.0], 'o',
                              color=CORAL, ms=5, zorder=5)
tgt_arrow = ax_robot.annotate('', xy=(1.5+0.4, 1.0), xytext=(1.5, 1.0),
    zorder=5, arrowprops=dict(arrowstyle='->', color=CORAL, lw=2.0))
tgt_label   = ax_robot.text(1.6, 1.1, 'target (1.50, 1.00)',
                              fontsize=8, color=CORAL, zorder=5)

# EE trace
trace_line, = ax_robot.plot([], [], '-', color=TEAL+'44', lw=1.2, zorder=2)
trace_x, trace_y = [], []

# Robot geometry objects
lc = [TEAL, '#0F6E56', '#085041']
jc = [AMBER, '#BA7517', '#854F0B']
links  = [ax_robot.plot([], [], '-', color=c, lw=5-i,
                         solid_capstyle='round', zorder=3)[0]
          for i, c in enumerate(lc)]
llbls  = [ax_robot.text(0, 0, f'L{i+1}', fontsize=7,
                         color=lc[i] + 'bb', zorder=4) for i in range(3)]
jdots  = [ax_robot.plot([], [], 'o', color=c, ms=9-2*i, zorder=5)[0]
          for i, c in enumerate(jc)]
jlbls  = [ax_robot.text(0, 0, f'J{i+1}', fontsize=7,
                         color=jc[i], zorder=6) for i in range(3)]
ee_dot,= ax_robot.plot([], [], 'o', color=CORAL, ms=7, zorder=7)
ee_arr = ax_robot.annotate('', xy=(0, 0), xytext=(0, 0), zorder=6,
             arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.5))

# Base plate
ax_robot.fill_between([-0.22, 0.22], [-0.08, -0.08], [0, 0],
                        color='#555548', zorder=4)
for bx in np.linspace(-0.18, 0.18, 5):
    ax_robot.plot([bx, bx-0.06], [-0.08, -0.16],
                  color='#666658', lw=1.5, zorder=3)

# Status text overlays
st_title = ax_robot.text(
    0.02, 0.985, '', transform=ax_robot.transAxes,
    fontsize=9, color=AMBER, va='top', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', fc='#1c1c1a', ec='#3a3a35', alpha=0.93))
st_msg = ax_robot.text(
    0.02, 0.928, '', transform=ax_robot.transAxes,
    fontsize=7.5, color=TXTC, va='top', family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', fc='#1c1c1a', ec='#3a3a35', alpha=0.88))
st_num = ax_robot.text(
    0.02, 0.770, '', transform=ax_robot.transAxes,
    fontsize=7.5, color=TXTC, va='top', family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', fc='#1c1c1a', ec='#3a3a35', alpha=0.88))
pause_overlay = ax_robot.text(
    0.5, 0.5, '', transform=ax_robot.transAxes,
    fontsize=24, color='#ffffffaa', va='center', ha='center', fontweight='bold')

ax_robot.set_title('3-DOF Planar Robot Arm — Kinematics Visualizer',
                    color=TXTC, fontsize=10, pad=8)

# ── Homogeneous transform matrix panel ────────────────────────
ax_mat.set_facecolor(PAN)
ax_mat.axis('off')
ax_mat.set_title('T₀⁴  (homogeneous transform)', color=TXTC, fontsize=8, pad=4)
mat_text = ax_mat.text(0.04, 0.88, '', transform=ax_mat.transAxes,
    fontsize=8.5, color=TEAL, va='top', family='monospace')
for sp in ax_mat.spines.values():
    sp.set_color('#333330')

# ── EE + joint info panel ─────────────────────────────────────
ax_info.set_facecolor(PAN)
ax_info.axis('off')
ax_info.set_title('End-effector & joints', color=TXTC, fontsize=8, pad=4)
info_text = ax_info.text(0.04, 0.92, '', transform=ax_info.transAxes,
    fontsize=8.5, color=TXTC, va='top', family='monospace')
for sp in ax_info.spines.values():
    sp.set_color('#333330')

# ── DH table panel ────────────────────────────────────────────
ax_dh.set_facecolor(PAN)
ax_dh.axis('off')
ax_dh.set_title('DH Parameters', color=TXTC, fontsize=8, pad=4)
dh_rows = [['1', '1.0', '0.5', '0', 'q₁'],
           ['2', '1.0', '0.0', '0', 'q₂'],
           ['3', '1.0', '0.0', '0', 'q₃']]
tbl = ax_dh.table(cellText=dh_rows,
                  colLabels=['i', 'a (m)', 'd (m)', 'α', 'θ'],
                  loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1e1e1c' if r % 2 == 0 else '#232320')
    cell.set_edgecolor('#333330')
    cell.set_text_props(color=AMBER if (c == 0 and r > 0) else TXTC)
    if r == 0:
        cell.set_facecolor('#2a2a26')
        cell.set_text_props(color='#888882', fontweight='bold')

# ─── Sliders ──────────────────────────────────────────────────
sc = '#252522'
# Joint sliders
ax_s1 = fig.add_axes([0.05, 0.115, 0.21, 0.018], facecolor=sc)
ax_s2 = fig.add_axes([0.05, 0.088, 0.21, 0.018], facecolor=sc)
ax_s3 = fig.add_axes([0.05, 0.061, 0.21, 0.018], facecolor=sc)
sl1 = Slider(ax_s1, 'q₁', -np.pi, np.pi, valinit=0, color=TEAL, initcolor='none')
sl2 = Slider(ax_s2, 'q₂', -np.pi, np.pi, valinit=0, color=TEAL, initcolor='none')
sl3 = Slider(ax_s3, 'q₃', -np.pi, np.pi, valinit=0, color=TEAL, initcolor='none')
# Target sliders
ax_tx = fig.add_axes([0.33, 0.115, 0.21, 0.018], facecolor=sc)
ax_ty = fig.add_axes([0.33, 0.088, 0.21, 0.018], facecolor=sc)
sl_tx = Slider(ax_tx, 'target x', -2.8, 2.8, valinit=1.5, color=CORAL, initcolor='none')
sl_ty = Slider(ax_ty, 'target y', -2.8, 2.8, valinit=1.0, color=CORAL, initcolor='none')
# Angular sliders
ax_sp = fig.add_axes([0.33, 0.061, 0.21, 0.018], facecolor=sc)
sl_phi = Slider(ax_sp, 'phi (rad)', -np.pi, np.pi, valinit=0.0, color='#5DCAA5', initcolor='none')
sl_phi.label.set_color('#5DCAA5'); sl_phi.label.set_fontsize(8.5)
sl_phi.valtext.set_color(TXTC); sl_phi.valtext.set_fontsize(8)

for sl, col in [(sl1,AMBER),(sl2,AMBER),(sl3,AMBER),(sl_tx,CORAL),(sl_ty,CORAL)]:
    sl.label.set_color(col); sl.label.set_fontsize(8.5)
    sl.valtext.set_color(TXTC); sl.valtext.set_fontsize(8)

fig.text(0.05, 0.140, 'Joint angles (drag to override animation)',
         color=DIM, fontsize=7.5)
fig.text(0.33, 0.140, 'IK target (rebuilds and restarts sequence)',
         color=DIM, fontsize=7.5)

# Buttons
ax_bp  = fig.add_axes([0.615, 0.058, 0.07, 0.058], facecolor='#252522')
ax_bpa = fig.add_axes([0.695, 0.058, 0.07, 0.058], facecolor='#252522')
ax_bh  = fig.add_axes([0.775, 0.058, 0.07, 0.058], facecolor='#252522')
btn_play  = Button(ax_bp,  '▶  Play',    color='#252522', hovercolor='#3a3a35')
btn_pause = Button(ax_bpa, '⏸  Pause',   color='#252522', hovercolor='#3a3a35')
btn_home  = Button(ax_bh,  '⌂  Home',    color='#252522', hovercolor='#3a3a35')
for btn in [btn_play, btn_pause, btn_home]:
    btn.label.set_color(TXTC); btn.label.set_fontsize(8.5)

fig.text(0.05, 0.030,
         'Keys: P or Space = pause / resume',
         color=DIM, fontsize=7.5)

# ─── State ────────────────────────────────────────────────────
state = {'frame': 0, 'manual': False, 'paused': False}

# ─── Core draw function ───────────────────────────────────────
def update_visuals(q, cur_tgt=None):
    T, pts = fk(q)
    phi = float(q[0] + q[1] + q[2])
    ee  = np.array([pts[-1][0], pts[-1][1]])

    # Trace
    trace_x.append(ee[0]); trace_y.append(ee[1])
    trace_line.set_data(trace_x, trace_y)

    # Links
    for i in range(3):
        links[i].set_data([pts[i][0], pts[i+1][0]],
                           [pts[i][1], pts[i+1][1]])
        mx = (pts[i][0] + pts[i+1][0]) / 2 + 0.04
        my = (pts[i][1] + pts[i+1][1]) / 2 + 0.04
        llbls[i].set_position((mx, my))
        jdots[i].set_data([pts[i][0]], [pts[i][1]])
        jlbls[i].set_position((pts[i][0]+0.07, pts[i][1]+0.07))

    # End-effector dot and orientation arrow
    ee_dot.set_data([ee[0]], [ee[1]])
    al = 0.35
    ee_arr.set_position(ee)
    ee_arr.xy    = (ee[0] + al*np.cos(phi), ee[1] + al*np.sin(phi))
    ee_arr.xyann = (ee[0], ee[1])

    # Target marker
    if cur_tgt is not None:
        tgt_marker.set_data([cur_tgt[0]], [cur_tgt[1]])
        tgt_label.set_position((cur_tgt[0]+0.1, cur_tgt[1]+0.1))
        tgt_label.set_text(f'target ({cur_tgt[0]:.2f}, {cur_tgt[1]:.2f})  φ={np.degrees(cur_tgt[2]):.1f}°')
        # Actualiza la flecha con la orientacion deseada
        al = 0.45
        #phi_tgt = cur_tgt[2]
        phi_tgt = cur_tgt[2] if len(cur_tgt) == 3 else 0.0
        tgt_arrow.set_position((cur_tgt[0], cur_tgt[1]))
        tgt_arrow.xy = (cur_tgt[0] + al*np.cos(phi_tgt),
                        cur_tgt[1] + al*np.sin(phi_tgt))
        tgt_arrow.xyann = (cur_tgt[0], cur_tgt[1])

    # Homogeneous transform matrix
    mat_text.set_text(format_matrix(T))

    # Info panel
    info_text.set_text(
        f'x  = {ee[0]:+.4f} m\n'
        f'y  = {ee[1]:+.4f} m\n'
        f'φ  = {np.degrees(phi):+.2f}°\n'
        f'──────────────\n'
        f'q₁ = {q[0]:+.4f} rad\n'
        f'q₂ = {q[1]:+.4f} rad\n'
        f'q₃ = {q[2]:+.4f} rad'
    )
    return T, ee, phi

def sync_sliders(q):
    """Update slider positions without triggering callbacks."""
    for sl, v in zip([sl1, sl2, sl3], q):
        sl.eventson = False
        sl.set_val(v)
        sl.eventson = True

# ─── Slider / button callbacks ────────────────────────────────
def on_joint_slider(_):
    state['manual'] = True
    q = np.array([sl1.val, sl2.val, sl3.val])
    T, ee, phi = update_visuals(q, (sl_tx.val, sl_ty.val))
    st_title.set_text('Manual control')
    st_msg.set_text('Drag joint sliders — FK updates in real time.')
    st_num.set_text(
        f'q  = [{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}] rad\n'
        f'EE = ({ee[0]:.3f}, {ee[1]:.3f}) m   φ = {np.degrees(phi):.1f}°')
    fig.canvas.draw_idle()

def on_target_slider(_):
    global traj, phases, msgs, tgts
    tx, ty, phi = sl_tx.val, sl_ty.val, sl_phi.val
    traj, phases, msgs, tgts = build_sequence(tx, ty, phi)
    tgt_marker.set_data([tx], [ty])
    tgt_label.set_text(f'target ({tx:.2f}, {ty:.2f})  phi={np.degrees(phi):.1f}')
    tgt_label.set_position((tx+0.1, ty+0.1))
    # ── esto es lo que faltaba ──
    al = 0.45
    tgt_arrow.set_position((tx, ty))
    tgt_arrow.xy    = (tx + al*np.cos(phi), ty + al*np.sin(phi))
    tgt_arrow.xyann = (tx, ty)
    # ───────────────────────────
    trace_x.clear(); trace_y.clear()
    trace_line.set_data([], [])
    state.update(frame=0, manual=False, paused=False)
    pause_overlay.set_text('')
    btn_pause.label.set_text('Pause')
    fig.canvas.draw_idle()

sl1.on_changed(on_joint_slider)
sl2.on_changed(on_joint_slider)
sl3.on_changed(on_joint_slider)
sl_tx.on_changed(on_target_slider)
sl_ty.on_changed(on_target_slider)
sl_phi.on_changed(on_target_slider)

def on_play(_):
    global traj, phases, msgs, tgts
    traj, phases, msgs, tgts = build_sequence(sl_tx.val, sl_ty.val)
    trace_x.clear(); trace_y.clear()
    trace_line.set_data([], [])
    state.update(frame=0, manual=False, paused=False)
    pause_overlay.set_text('')
    btn_pause.label.set_text('⏸  Pause')

def on_pause(_):
    state['paused'] = not state['paused']
    if state['paused']:
        pause_overlay.set_text('⏸  PAUSED')
        btn_pause.label.set_text('▶  Resume')
    else:
        pause_overlay.set_text('')
        btn_pause.label.set_text('⏸  Pause')
    fig.canvas.draw_idle()

def on_home(_):
    state.update(manual=True, paused=False)
    trace_x.clear(); trace_y.clear()
    trace_line.set_data([], [])
    pause_overlay.set_text('')
    sl1.set_val(0); sl2.set_val(0); sl3.set_val(0)

def on_key(event):
    if event.key in ('p', ' '):
        on_pause(None)

btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
btn_home.on_clicked(on_home)
fig.canvas.mpl_connect('key_press_event', on_key)

# ─── Animation loop ───────────────────────────────────────────
def animate(_):
    if state['paused'] or state['manual']:
        return []

    f = state['frame'] % len(traj)
    state['frame'] += 1
    q = traj[f]

    T, ee, phi = update_visuals(q, tgts[f])
    sync_sliders(q)

    st_title.set_text(phases[f])
    st_msg.set_text(msgs[f])
    st_num.set_text(
        f'q  = [{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}] rad\n'
        f'EE = ({ee[0]:.3f}, {ee[1]:.3f}) m   φ = {np.degrees(phi):.1f}°')

    return (trace_line, *links, *jdots, ee_dot,
            st_title, st_msg, st_num, mat_text, info_text)

ani = animation.FuncAnimation(
    fig, animate, interval=30, blit=False, cache_frame_data=False)

fig.suptitle(
    'Robotics  |  3-DOF Planar Arm  |  Pure NumPy / Matplotlib  |  Colab ready',
    color='#555550', fontsize=8, y=0.975)

plt.show()