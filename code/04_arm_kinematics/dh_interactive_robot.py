"""
=============================================================
  VISUALIZADOR INTERACTIVO - PARÁMETROS DENAVIT-HARTENBERG
  Para estudiantes de Robótica  |  RAPTOR Lab – TAMIU
=============================================================
  Controles:
    Sliders      → ajusta θ, d, a, α en tiempo real
    RadioButtons → ve la transformación DH paso a paso
    Reset        → regresa a valores de ejemplo
=============================================================
  Requiere: numpy, matplotlib
=============================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

matplotlib.rcParams['toolbar'] = 'None'

# ═══════════════════════════════════════════════════════════
#  PALETA – fondo blanco
# ═══════════════════════════════════════════════════════════
BG       = "white"
PANEL_BG = "#f0f2f6"
BORDER   = "#cccccc"

C0X = "#d62020";  C0Y = "#1a8c1a";  C0Z = "#1a5cb5"  # frame i-1
C1X = "#ff7722";  C1Y = "#22aa55";  C1Z = "#2299dd"  # frame i

THETA_C = "#cc8800"
D_C     = "#0099bb"
A_C     = "#cc2266"
ALPHA_C = "#7722cc"

LINK_BODY  = "#5577aa"
LINK_HI    = "#ff6600"
JOINT_COL  = "#223366"
JOINT_DIAM = 0.10

STEP_COLS  = [THETA_C, D_C, A_C, ALPHA_C]
STEP_NAMES = ["theta: Rot. Z", "d: Trans. Z", "a: Trans. X", "alpha: Rot. X"]

ARROW_L = 0.40

# ═══════════════════════════════════════════════════════════
#  ALGEBRA DH
# ═══════════════════════════════════════════════════════════
def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def dh_step(theta, d, a, alpha, step=4):
    T = np.eye(4)
    ops = [("Rz", theta), ("Tz", d), ("Tx", a), ("Rx", alpha)]
    for i, (op, val) in enumerate(ops):
        if i >= step:
            break
        M = np.eye(4)
        if   op == "Rz": M[:3, :3] = Rz(val)
        elif op == "Tz": M[2, 3]   = val
        elif op == "Tx": M[0, 3]   = val
        elif op == "Rx": M[:3, :3] = Rx(val)
        T = T @ M
    return T

# ═══════════════════════════════════════════════════════════
#  GEOMETRIA DEL ROBOT
# ═══════════════════════════════════════════════════════════
def robot_context_frames():
    T_base = np.eye(4)
    T_prev = np.eye(4)
    T_prev[2, 3] = 0.55
    return T_base, T_prev

def _ortho_basis(v):
    v = v / (np.linalg.norm(v) + 1e-12)
    u = np.array([1., 0., 0.]) if abs(v[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = np.cross(v, u); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(v, e1)
    return e1, e2

def draw_cylinder(ax, p1, p2, radius=0.07, color=LINK_BODY, alpha=0.75, n=16):
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return
    e1, e2 = _ortho_basis(axis / length)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring1 = [p1 + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles]
    ring2 = [p2 + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([ring1[i], ring1[j], ring2[j], ring2[i]])
    faces.append(ring1)
    faces.append(ring2)
    poly = Poly3DCollection(faces, alpha=alpha, linewidth=0)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    ax.add_collection3d(poly)

def draw_disk(ax, center, normal, radius=JOINT_DIAM, color=JOINT_COL, alpha=0.92, n=32):
    e1, e2 = _ortho_basis(normal / (np.linalg.norm(normal) + 1e-12))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = [center + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles]
    poly = Poly3DCollection([verts], alpha=alpha, linewidth=0.8)
    poly.set_facecolor(color)
    poly.set_edgecolor("white")
    ax.add_collection3d(poly)

def draw_box_link(ax, p1, p2, w=0.055, h=0.038, color=LINK_BODY, alpha=0.78):
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return
    e1, e2 = _ortho_basis(axis / length)
    corners = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            off = s1 * w * e1 + s2 * h * e2
            corners.append(p1 + off)
            corners.append(p2 + off)
    c = corners
    faces = [
        [c[0], c[1], c[3], c[2]],
        [c[4], c[5], c[7], c[6]],
        [c[0], c[1], c[5], c[4]],
        [c[2], c[3], c[7], c[6]],
        [c[0], c[2], c[6], c[4]],
        [c[1], c[3], c[7], c[5]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, linewidth=0.5)
    poly.set_facecolor(color)
    poly.set_edgecolor("#99aabb")
    ax.add_collection3d(poly)

def draw_frame(ax, T, colors, label="", alpha_val=1.0, scale=ARROW_L):
    origin = T[:3, 3]
    for i, (col, lbl) in enumerate(zip(colors, ["x", "y", "z"])):
        direction = T[:3, i] * scale
        ax.quiver(*origin, *direction,
                  color=col, linewidth=2.0, alpha=alpha_val,
                  arrow_length_ratio=0.20)
        tip = origin + direction * 1.22
        if alpha_val > 0.4:
            tag = f"${lbl}_{{{label}}}$" if label else f"${lbl}$"
            ax.text(*tip, tag, color=col, fontsize=8.5,
                    fontweight='bold', ha='left', va='center')

def draw_arc(ax, center, normal, radius, start_vec, angle, color, lw=2.2, n=60):
    if abs(angle) < 1e-4:
        return
    u = start_vec / (np.linalg.norm(start_vec) + 1e-12)
    n_hat = normal / (np.linalg.norm(normal) + 1e-12)
    v = np.cross(n_hat, u); v /= (np.linalg.norm(v) + 1e-12)
    ts = np.linspace(0, angle, n)
    pts = (center[:, None]
           + radius * (np.outer(u, np.cos(ts)) + np.outer(v, np.sin(ts))))
    ax.plot(pts[0], pts[1], pts[2], color=color, lw=lw, alpha=0.90, zorder=10)
    tip = pts[:, -1]
    dtip = pts[:, -1] - pts[:, -2]
    dtip /= (np.linalg.norm(dtip) + 1e-12)
    ax.quiver(*tip, *(dtip * 0.11), color=color, linewidth=2,
              arrow_length_ratio=1.0, alpha=0.90, zorder=10)

def draw_dashed(ax, p1, p2, color, lw=1.8, style='--'):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, lw=lw, linestyle=style, alpha=0.80, zorder=8)

def label_dim(ax, pos, text, color):
    ax.text(*pos, text, color=color, fontsize=8.2,
            fontweight='bold', ha='center', va='center', zorder=12,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor=color, alpha=0.88, linewidth=1.2))

# ═══════════════════════════════════════════════════════════
#  ESTADO GLOBAL
# ═══════════════════════════════════════════════════════════
state = dict(theta=0.7, d=0.45, a=0.55, alpha=0.5, step=4)
_radio_full_mode = [True]

DESCRIPTIONS = {
    1: ("Step 1 — theta: Rotation about Z_(i-1)",
        "Rotates frame {i-1} about its own Z axis.\n"
        "For revolute joints, theta is the\n"
        "joint variable (articulation angle)."),
    2: ("Step 2 — d: Translation along Z_(i-1)",
        "Translates the origin along the Z axis\n"
        "(after theta rotation). For prismatic\n"
        "joints, d is the joint variable."),
    3: ("Step 3 — a: Link length along X_i",
        "Translates the origin along the new X\n"
        "axis. Equals the perpendicular distance\n"
        "between consecutive Z axes."),
    4: ("Step 4 — alpha: Link twist about X_i",
        "Rotates the frame about the new X axis.\n"
        "Angle between Z_(i-1) and Z_i axes,\n"
        "measured around the link."),
}
DESC_FULL = (
    "Full view: T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)",
    "All 4 DH parameters define the complete\n"
    "geometric relationship between consecutive\n"
    "frames {i-1} and {i} in a kinematic chain."
)

# ═══════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15.5, 8.6), facecolor=BG)
try:
    fig.canvas.manager.set_window_title(
        "Denavit-Hartenberg Parameters – RAPTOR Lab TAMIU")
except Exception:
    pass

gs = gridspec.GridSpec(1, 2, width_ratios=[2.3, 1.0],
                       left=0.01, right=0.99,
                       bottom=0.03, top=0.93, wspace=0.04)

ax3d = fig.add_subplot(gs[0], projection='3d')
ax3d.set_facecolor(BG)
ax3d.patch.set_facecolor(BG)

gs_r = gridspec.GridSpecFromSubplotSpec(9, 1, subplot_spec=gs[1], hspace=0.45)

fig.text(0.37, 0.978, "Denavit-Hartenberg Parameters",
         color="#1a1a2e", fontsize=15, fontweight='bold',
         ha='center', va='top', fontfamily='monospace')
fig.text(0.37, 0.952, "RAPTOR Lab  |  Texas A&M International University",
         color="#555577", fontsize=8.5, ha='center', va='top',
         fontfamily='monospace')

# Sliders
slider_specs = [
    ("theta  joint angle [rad]", THETA_C, -np.pi, np.pi, 0.7,  "theta"),
    ("d      joint offset [m]",  D_C,    -1.0,   1.0,   0.45, "d"),
    ("a      link length [m]",   A_C,     0.0,   1.2,   0.55, "a"),
    ("alpha  link twist [rad]",  ALPHA_C, -np.pi, np.pi, 0.5,  "alpha"),
]
sliders = []
for k, (lbl, col, vmin, vmax, vinit, key) in enumerate(slider_specs):
    ax_sl = fig.add_subplot(gs_r[k])
    ax_sl.set_facecolor(PANEL_BG)
    for sp in ax_sl.spines.values():
        sp.set_edgecolor(BORDER)
    try:
        sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vinit,
                    color=col, track_color="#d0d8e8",
                    handle_style={'facecolor': col, 'edgecolor': 'white', 'size': 10})
    except TypeError:
        sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vinit, color=col)
    sl.label.set_color("#222222")
    sl.label.set_fontsize(9)
    sl.label.set_fontfamily('monospace')
    sl.valtext.set_color(col)
    sl.valtext.set_fontsize(9)
    sliders.append((key, sl))

# RadioButtons
ax_radio = fig.add_subplot(gs_r[4])
ax_radio.set_facecolor(PANEL_BG)
for sp in ax_radio.spines.values():
    sp.set_edgecolor(BORDER)
radio = RadioButtons(
    ax_radio,
    labels=["Full view",
            "1: Rot Z (theta)", "2: Trans Z (d)",
            "3: Trans X (a)",   "4: Rot X (alpha)"],
    active=0,
    activecolor=THETA_C)
for lbl_obj, col in zip(radio.labels, ["#222222"] + STEP_COLS):
    lbl_obj.set_color(col)
    lbl_obj.set_fontsize(8.2)
    lbl_obj.set_fontfamily('monospace')
ax_radio.set_title("  DH Step", color="#222222",
                   fontsize=8.5, pad=2, fontfamily='monospace', loc='left')

# Info
ax_info = fig.add_subplot(gs_r[5:8])
ax_info.set_facecolor(PANEL_BG)
ax_info.axis('off')
info_title = ax_info.text(0.05, 0.96, "", color="#1a1a2e",
    fontsize=8.0, fontweight='bold',
    transform=ax_info.transAxes, va='top', fontfamily='monospace')
info_body  = ax_info.text(0.05, 0.72, "", color="#444466",
    fontsize=7.8, transform=ax_info.transAxes,
    va='top', fontfamily='monospace')

legend_items = [
    (THETA_C, "theta  joint angle"),
    (D_C,     "d      joint offset"),
    (A_C,     "a      link length"),
    (ALPHA_C, "alpha  link twist"),
]
for n_li, (col, txt) in enumerate(legend_items):
    ax_info.text(0.06, 0.40 - n_li * 0.092, "●", color=col,
                 fontsize=12, transform=ax_info.transAxes, va='top')
    ax_info.text(0.16, 0.40 - n_li * 0.092, txt, color="#333355",
                 fontsize=7.5, transform=ax_info.transAxes,
                 va='top', fontfamily='monospace')

# Reset
ax_rst = fig.add_subplot(gs_r[8])
ax_rst.set_facecolor(PANEL_BG)
btn_reset = Button(ax_rst, "  Restore defaults", color=PANEL_BG, hovercolor="#dde8ff")
btn_reset.label.set_color("#223388")
btn_reset.label.set_fontsize(9)
btn_reset.label.set_fontfamily('monospace')

fig.text(0.995, 0.005, "RAPTOR Lab · TAMIU", color="#aaaaaa",
         fontsize=7, ha='right', fontfamily='monospace')

# ═══════════════════════════════════════════════════════════
#  DIBUJO PRINCIPAL
# ═══════════════════════════════════════════════════════════
def update_plot(_=None):
    theta = state['theta']
    d     = state['d']
    a     = state['a']
    alpha = state['alpha']
    step  = state['step']
    full  = _radio_full_mode[0]

    ax3d.cla()
    ax3d.set_facecolor(BG)
    ax3d.set_xlim(-0.7, 1.5)
    ax3d.set_ylim(-0.7, 1.5)
    ax3d.set_zlim(-0.5, 1.7)
    ax3d.set_xlabel("X [m]", color="#666688", fontsize=8, labelpad=2)
    ax3d.set_ylabel("Y [m]", color="#666688", fontsize=8, labelpad=2)
    ax3d.set_zlabel("Z [m]", color="#666688", fontsize=8, labelpad=2)
    ax3d.tick_params(colors="#999999", labelsize=6.5)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor("#f7f8fc")
        pane.set_edgecolor("#ddddee")
    ax3d.grid(True, color="#ddddee", linestyle='-', linewidth=0.6)

    T_base, T_prev = robot_context_frames()
    T_dh_base = T_prev
    O_base = T_dh_base[:3, 3]
    O0     = np.array([0., 0., 0.])

    T1 = T_dh_base @ dh_step(theta, d, a, alpha, step=1)
    T2 = T_dh_base @ dh_step(theta, d, a, alpha, step=2)
    T3 = T_dh_base @ dh_step(theta, d, a, alpha, step=3)
    T4 = T_dh_base @ dh_step(theta, d, a, alpha, step=4)
    O2 = T2[:3, 3]
    O3 = T3[:3, 3]
    O4 = T4[:3, 3]

    # ── Pedestal / base ──────────────────────────────────
    draw_cylinder(ax3d, O0 + np.array([0, 0, -0.14]),
                  O0, radius=0.18, color="#aabbcc", alpha=0.50)

    # ── Link 0 (base → joint 1) ──────────────────────────
    draw_box_link(ax3d, O0, O_base, w=0.060, h=0.040,
                  color="#8899bb", alpha=0.55)
    draw_disk(ax3d, O0, np.array([0., 0., 1.]),
              radius=0.12, color="#556688", alpha=0.80)

    # ── Junta {i-1} ──────────────────────────────────────
    joint_z_prev = T_dh_base[:3, 2]
    draw_disk(ax3d, O_base, joint_z_prev,
              radius=JOINT_DIAM + 0.025, color="#334466", alpha=0.88)

    # ── Eslabon activo (naranja) ──────────────────────────
    if full:
        draw_box_link(ax3d, O_base, O4, w=0.065, h=0.045,
                      color=LINK_HI, alpha=0.82)
        draw_disk(ax3d, O4, T4[:3, 2],
                  radius=JOINT_DIAM, color="#994400", alpha=0.88)
        # End-effector
        ee = O4 + T4[:3, 2] * 0.22
        draw_cylinder(ax3d, O4, ee, radius=0.025, color="#cc9944", alpha=0.65)
        # Garra (dos dedos)
        side = T4[:3, 0] * 0.06
        for sign in [-1, 1]:
            tip = ee + T4[:3, 2] * 0.12 + sign * side
            draw_cylinder(ax3d, ee, tip, radius=0.018, color="#cc9944", alpha=0.65)
        ax3d.scatter(*O_base, color="#334466", s=130, zorder=15, depthshade=False)
        ax3d.scatter(*O4,     color="#994400", s=130, zorder=15, depthshade=False)
    else:
        O_end = [T1[:3,3], O2, O3, O4][step - 1]
        draw_box_link(ax3d, O_base, O_end, w=0.058, h=0.038,
                      color=LINK_HI, alpha=0.72)
        ax3d.scatter(*O_base, color="#334466", s=120, zorder=15, depthshade=False)
        ax3d.scatter(*O_end,  color="#994400", s=120, zorder=15, depthshade=False)

    # ── Marcos de coordenadas ────────────────────────────
    draw_frame(ax3d, T_dh_base, [C0X, C0Y, C0Z], label="i{-}1")
    if full:
        draw_frame(ax3d, T4, [C1X, C1Y, C1Z], label="i")
    else:
        T_frame = [T1, T2, T3, T4][step - 1]
        draw_frame(ax3d, T_frame, [C1X, C1Y, C1Z], label="i'")

    # ── Anotaciones DH ───────────────────────────────────
    if full:
        # theta – arco Z_{i-1}
        if abs(theta) > 0.02:
            ref_x = T_dh_base[:3, 0]
            draw_arc(ax3d, O_base, T_dh_base[:3, 2],
                     0.32, ref_x, theta, THETA_C, lw=2.5)
            e1, _ = _ortho_basis(T_dh_base[:3, 2])
            v2 = np.cross(T_dh_base[:3, 2] /
                          (np.linalg.norm(T_dh_base[:3, 2]) + 1e-12), e1)
            v2 /= (np.linalg.norm(v2) + 1e-12)
            ang_m = theta / 2
            lp = O_base + 0.44 * (np.cos(ang_m) * e1 + np.sin(ang_m) * v2)
            label_dim(ax3d, lp,
                      f"theta={np.degrees(theta):.1f}deg", THETA_C)
        # d – linea Z_{i-1}
        if abs(d) > 0.02:
            draw_dashed(ax3d, O_base, O2, D_C, lw=2.2)
            off = T_dh_base[:3, 0] * 0.16
            label_dim(ax3d, (O_base + O2) / 2 + off,
                      f"d={d:.2f}m", D_C)
        # a – linea X_i
        if abs(a) > 0.02:
            draw_dashed(ax3d, O2, O3, A_C, lw=2.2)
            off2 = T2[:3, 1] * 0.16
            label_dim(ax3d, (O2 + O3) / 2 + off2,
                      f"a={a:.2f}m", A_C)
        # alpha – arco X_i
        if abs(alpha) > 0.02:
            draw_arc(ax3d, O3, T3[:3, 0], 0.26,
                     T3[:3, 1], alpha, ALPHA_C, lw=2.5)
            lp2 = O3 + T3[:3, 0] * 0.12 + T3[:3, 1] * 0.38
            label_dim(ax3d, lp2,
                      f"alpha={np.degrees(alpha):.1f}deg", ALPHA_C)
    else:
        if step == 1 and abs(theta) > 0.02:
            draw_arc(ax3d, O_base, T_dh_base[:3, 2],
                     0.40, T_dh_base[:3, 0], theta, THETA_C, lw=2.8)
            e1, _ = _ortho_basis(T_dh_base[:3, 2])
            v2 = np.cross(T_dh_base[:3, 2] /
                          (np.linalg.norm(T_dh_base[:3, 2]) + 1e-12), e1)
            v2 /= (np.linalg.norm(v2) + 1e-12)
            ang_m = theta / 2
            lp = O_base + 0.54 * (np.cos(ang_m) * e1 + np.sin(ang_m) * v2)
            label_dim(ax3d, lp,
                      f"theta={np.degrees(theta):.1f}deg", THETA_C)
        elif step == 2 and abs(d) > 0.02:
            draw_dashed(ax3d, O_base, O2, D_C, lw=2.5)
            off = T_dh_base[:3, 0] * 0.20
            label_dim(ax3d, (O_base + O2) / 2 + off,
                      f"d={d:.2f}m", D_C)
        elif step == 3 and abs(a) > 0.02:
            draw_dashed(ax3d, O2, O3, A_C, lw=2.5)
            off2 = T2[:3, 1] * 0.20
            label_dim(ax3d, (O2 + O3) / 2 + off2,
                      f"a={a:.2f}m", A_C)
        elif step == 4 and abs(alpha) > 0.02:
            draw_arc(ax3d, O3, T3[:3, 0], 0.32,
                     T3[:3, 1], alpha, ALPHA_C, lw=2.8)
            lp2 = O3 + T3[:3, 0] * 0.12 + T3[:3, 1] * 0.44
            label_dim(ax3d, lp2,
                      f"alpha={np.degrees(alpha):.1f}deg", ALPHA_C)

    # ── Matriz T ─────────────────────────────────────────
    if full:
        T_rel = dh_step(theta, d, a, alpha, step=4)
    else:
        T_rel = dh_step(theta, d, a, alpha, step=step)
    mat_str = (
        "T(i-1 -> i) =\n"
        f"[{T_rel[0,0]:+.2f}  {T_rel[0,1]:+.2f}  {T_rel[0,2]:+.2f} | {T_rel[0,3]:+.3f}]\n"
        f"[{T_rel[1,0]:+.2f}  {T_rel[1,1]:+.2f}  {T_rel[1,2]:+.2f} | {T_rel[1,3]:+.3f}]\n"
        f"[{T_rel[2,0]:+.2f}  {T_rel[2,1]:+.2f}  {T_rel[2,2]:+.2f} | {T_rel[2,3]:+.3f}]\n"
        "[ 0.00   0.00   0.00  |  1.000]"
    )
    ax3d.text2D(0.01, 0.01, mat_str, transform=ax3d.transAxes,
                color="#223366", fontsize=7.2, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#aabbcc', alpha=0.93, linewidth=1.0))

    # ── Step label inside 3D plot (avoids overlap with fig.text header) ──
    if full:
        step_lbl = "Full view  -  all 4 DH parameters"
        tc = "#1a1a2e"
    else:
        step_lbl = f"Step {step}: {STEP_NAMES[step-1]}"
        tc = STEP_COLS[step - 1]
    ax3d.text2D(0.50, 0.97, step_lbl,
                transform=ax3d.transAxes,
                color=tc, fontsize=9.5, fontweight='bold',
                fontfamily='monospace', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor=tc,
                          alpha=0.88, linewidth=1.2))

    # ── Panel info ────────────────────────────────────────
    if full:
        info_title.set_text(DESC_FULL[0])
        info_body.set_text(DESC_FULL[1])
    else:
        desc = DESCRIPTIONS[step]
        info_title.set_text(desc[0])
        info_body.set_text(desc[1])

    fig.canvas.draw_idle()

# ═══════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════
def on_slider(_):
    for key, sl in sliders:
        state[key] = sl.val
    update_plot()

def on_radio(label):
    if "full" in label.lower():
        _radio_full_mode[0] = True;  state['step'] = 4
    elif "1:" in label:
        _radio_full_mode[0] = False; state['step'] = 1
    elif "2:" in label:
        _radio_full_mode[0] = False; state['step'] = 2
    elif "3:" in label:
        _radio_full_mode[0] = False; state['step'] = 3
    elif "4:" in label:
        _radio_full_mode[0] = False; state['step'] = 4
    update_plot()

def on_reset(_):
    defaults = dict(theta=0.7, d=0.45, a=0.55, alpha=0.5)
    for key, sl in sliders:
        sl.set_val(defaults[key])
    state.update(defaults)
    _radio_full_mode[0] = True
    radio.set_active(0)
    update_plot()

for _, sl in sliders:
    sl.on_changed(on_slider)
radio.on_clicked(on_radio)
btn_reset.on_clicked(on_reset)

ax3d.view_init(elev=24, azim=-52)
update_plot()
plt.show()
