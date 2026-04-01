"""
=============================================================
  INTERACTIVE VISUALIZER - DENAVIT-HARTENBERG PARAMETERS
  Robotics Course  |  RAPTOR Lab - TAMIU
  Gerardo Flores, Ph.D.
=============================================================
  Controls:
    - Sliders : adjust theta, d, a, alpha in real time
    - Radio   : view each DH step individually
    - Reset   : restore default values
=============================================================
  Requires: numpy, matplotlib
=============================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import matplotlib.gridspec as gridspec

matplotlib.rcParams['toolbar'] = 'None'

# ═══════════════════════════════════════════════════════════
#  COLOR PALETTE  –  white background
# ═══════════════════════════════════════════════════════════
BG       = "white"
PANEL_BG = "#f0f2f6"
BORDER   = "#cccccc"

# Frame {i-1}  — bold, saturated
FRAME0_X = "#cc1111"
FRAME0_Y = "#117711"
FRAME0_Z = "#1144bb"

# Frame {i}  — slightly lighter / distinct
FRAME1_X = "#ee6600"
FRAME1_Y = "#119944"
FRAME1_Z = "#1188cc"

# DH parameter colors
THETA_C  = "#b37700"   # amber
D_C      = "#007799"   # teal
A_C      = "#bb1155"   # magenta
ALPHA_C  = "#6611bb"   # purple

LINK_COL  = "#7788aa"  # simple link line

STEP_COLS  = [THETA_C, D_C, A_C, ALPHA_C]
STEP_NAMES = ["theta — Rot. Z", "d — Trans. Z",
              "a — Trans. X",   "alpha — Rot. X"]

# ═══════════════════════════════════════════════════════════
#  DH ALGEBRA
# ═══════════════════════════════════════════════════════════
def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def dh_step(theta, d, a, alpha, step=4):
    """Accumulated DH transform up to 'step' (1–4)."""
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
#  DRAWING HELPERS  (same as original, re-colored for white bg)
# ═══════════════════════════════════════════════════════════
ARROW_SCALE = 0.55

def draw_frame(ax, T, colors, label="", alpha_val=1.0, scale=ARROW_SCALE):
    origin = T[:3, 3]
    for i, (col, lbl) in enumerate(zip(colors, ["x", "y", "z"])):
        direction = T[:3, i] * scale
        ax.quiver(*origin, *direction,
                  color=col, linewidth=2.2, alpha=alpha_val,
                  arrow_length_ratio=0.18)
        tip = origin + direction * 1.18
        if alpha_val > 0.5:
            tag = f"  {lbl}_{{{label}}}" if label else f"  {lbl}"
            ax.text(*tip, tag, color=col, fontsize=8,
                    fontweight='bold', ha='left')


def draw_arc(ax, center, normal, radius, start_vec, angle,
             color, lw=2.0, n=60):
    if abs(angle) < 1e-6:
        return
    u     = start_vec / (np.linalg.norm(start_vec) + 1e-12)
    n_hat = normal    / (np.linalg.norm(normal)     + 1e-12)
    v = np.cross(n_hat, u); v /= (np.linalg.norm(v) + 1e-12)
    ts  = np.linspace(0, angle, n)
    pts = (center[:, None]
           + radius * (np.outer(u, np.cos(ts)) + np.outer(v, np.sin(ts))))
    ax.plot(pts[0], pts[1], pts[2], color=color, lw=lw, alpha=0.88)
    end  = pts[:, -1]
    darr = pts[:, -1] - pts[:, -2]
    darr /= (np.linalg.norm(darr) + 1e-12)
    ax.quiver(*end, *(darr * 0.12), color=color, linewidth=2,
              arrow_length_ratio=0.9, alpha=0.88)


def draw_dashed_line(ax, p1, p2, color, lw=1.6):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, lw=lw, linestyle='--', alpha=0.75)


def draw_link(ax, p1, p2, color=LINK_COL, lw=6, alpha=0.30):
    """Simple thick line between frame origins (same as original)."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, lw=lw, alpha=alpha, solid_capstyle='round')

# ═══════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════
state      = dict(theta=0.5, d=0.4, a=0.6, alpha=0.4, step=4)
_full_view = [True]

DESCRIPTIONS = {
    1: ("Step 1 — theta:  Rotation about Z_(i-1)",
        "Rotates frame {i-1} about its Z axis.\n"
        "For revolute joints, theta is the joint\n"
        "variable (articulation angle)."),
    2: ("Step 2 — d:  Translation along Z_(i-1)",
        "Slides the origin along the Z axis after\n"
        "the theta rotation. For prismatic joints,\n"
        "d is the joint variable."),
    3: ("Step 3 — a:  Link length along X_i",
        "Translates along the new X axis. Equal to\n"
        "the perpendicular distance between\n"
        "consecutive Z axes."),
    4: ("Step 4 — alpha:  Link twist about X_i",
        "Rotates about the new X axis. Angle\n"
        "between Z_(i-1) and Z_i measured around\n"
        "the link (twist angle)."),
}
DESC_FULL = (
    "Full view:  T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)",
    "All 4 DH parameters completely define the\n"
    "geometric relationship between consecutive\n"
    "frames {i-1} and {i} in a kinematic chain."
)

# ═══════════════════════════════════════════════════════════
#  FIGURE LAYOUT  (same structure as original)
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
try:
    fig.canvas.manager.set_window_title(
        "Denavit-Hartenberg Parameters  |  RAPTOR Lab - TAMIU")
except Exception:
    pass

# Two-column grid: 3-D view | controls
# top=0.89 leaves a clear margin for the header text above
gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1],
                       left=0.02, right=0.98,
                       bottom=0.02, top=0.89,
                       wspace=0.05)

# ── 3-D axes ──────────────────────────────────────────────
ax3d = fig.add_subplot(gs[0], projection='3d')
ax3d.set_facecolor(BG)
ax3d.patch.set_facecolor(BG)

# ── Right panel ───────────────────────────────────────────
gs_right = gridspec.GridSpecFromSubplotSpec(
    8, 1, subplot_spec=gs[1], hspace=0.4)

# ── Header  (sits safely above the GridSpec top=0.89) ─────
#    Title at 0.96,  subtitle at 0.93  → both above 0.89
fig.text(0.37, 0.965,
         "Denavit-Hartenberg Parameters",
         color="#1a1a2e", fontsize=15, fontweight='bold',
         ha='center', va='top', fontfamily='monospace')
fig.text(0.37, 0.938,
         "RAPTOR Lab  |  Texas A&M International University  "
         "|  G. Flores, Ph.D.",
         color="#445566", fontsize=8.5,
         ha='center', va='top', fontfamily='monospace')

# ── Sliders ───────────────────────────────────────────────
slider_specs = [
    ("theta  [rad]", THETA_C, -np.pi, np.pi, 0.5,  "theta"),
    ("d      [m]",   D_C,     -1.0,   1.0,   0.4,  "d"),
    ("a      [m]",   A_C,      0.0,   1.2,   0.6,  "a"),
    ("alpha  [rad]", ALPHA_C, -np.pi, np.pi, 0.4,  "alpha"),
]
sliders = []
for k, (lbl, col, vmin, vmax, vinit, key) in enumerate(slider_specs):
    ax_sl = fig.add_subplot(gs_right[k])
    ax_sl.set_facecolor(PANEL_BG)
    for sp in ax_sl.spines.values():
        sp.set_edgecolor(BORDER)
    try:
        sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vinit,
                    color=col, track_color="#d5dce8",
                    handle_style={'facecolor': col,
                                  'edgecolor': 'white', 'size': 10})
    except TypeError:
        sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vinit, color=col)
    sl.label.set_color("#1a1a2e")
    sl.label.set_fontsize(9)
    sl.label.set_fontfamily('monospace')
    sl.valtext.set_color(col)
    sl.valtext.set_fontsize(9)
    sliders.append((key, sl))

# ── Radio buttons ─────────────────────────────────────────
ax_radio = fig.add_subplot(gs_right[4])
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
for lbl_obj, col in zip(radio.labels, ["#1a1a2e"] + STEP_COLS):
    lbl_obj.set_color(col)
    lbl_obj.set_fontsize(8.5)
    lbl_obj.set_fontfamily('monospace')
ax_radio.set_title("  DH Step", color="#1a1a2e",
                   fontsize=9, pad=2, loc='left',
                   fontfamily='monospace')

# ── Info box ──────────────────────────────────────────────
ax_info = fig.add_subplot(gs_right[5:7])
ax_info.set_facecolor(PANEL_BG)
ax_info.axis('off')
for sp in ax_info.spines.values():
    sp.set_edgecolor(BORDER)
info_title = ax_info.text(0.05, 0.88, "",
    color="#1a1a2e", fontsize=8.5, fontweight='bold',
    transform=ax_info.transAxes, va='top', fontfamily='monospace')
info_body  = ax_info.text(0.05, 0.62, "",
    color="#444466", fontsize=7.8,
    transform=ax_info.transAxes, va='top', fontfamily='monospace')

# ── Reset button ──────────────────────────────────────────
ax_reset = fig.add_subplot(gs_right[7])
ax_reset.set_facecolor(PANEL_BG)
btn_reset = Button(ax_reset, "  Restore defaults",
                   color=PANEL_BG, hovercolor="#ddeeff")
btn_reset.label.set_color("#223388")
btn_reset.label.set_fontsize(9)
btn_reset.label.set_fontfamily('monospace')

# ── Footer ────────────────────────────────────────────────
fig.text(0.985, 0.008,
         "RAPTOR Lab · TAMIU  |  Gerardo Flores, Ph.D.",
         color="#888899", fontsize=7,
         ha='right', fontfamily='monospace')

# ═══════════════════════════════════════════════════════════
#  MAIN DRAW FUNCTION
# ═══════════════════════════════════════════════════════════
def update_plot(_=None):
    theta = state['theta']
    d     = state['d']
    a     = state['a']
    alpha = state['alpha']
    step  = state['step']
    full  = _full_view[0]

    ax3d.cla()
    ax3d.set_facecolor(BG)

    # Fixed axes limits so the view never jumps
    ax3d.set_xlim(-1.2, 1.6)
    ax3d.set_ylim(-1.2, 1.6)
    ax3d.set_zlim(-1.2, 1.6)
    ax3d.set_xlabel("X", color="#666688", fontsize=8)
    ax3d.set_ylabel("Y", color="#666688", fontsize=8)
    ax3d.set_zlabel("Z", color="#666688", fontsize=8)
    ax3d.tick_params(colors="#aaaaaa", labelsize=6)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor("#f5f6fa")
        pane.set_edgecolor("#ddddee")
    ax3d.grid(True, color="#e0e0ee", linestyle='-', linewidth=0.5)

    # ── Fixed frame {i-1} at origin ──────────────────────
    T0 = np.eye(4)
    draw_frame(ax3d, T0, [FRAME0_X, FRAME0_Y, FRAME0_Z], label="i{-}1")
    O0 = np.zeros(3)

    # ── Intermediate transforms ───────────────────────────
    T1 = dh_step(theta, d, a, alpha, step=1);  O1 = T1[:3, 3]
    T2 = dh_step(theta, d, a, alpha, step=2);  O2 = T2[:3, 3]
    T3 = dh_step(theta, d, a, alpha, step=3);  O3 = T3[:3, 3]
    T4 = dh_step(theta, d, a, alpha, step=4);  O4 = T4[:3, 3]

    # ── Full view ─────────────────────────────────────────
    if full:
        # theta arc around Z_0
        if abs(theta) > 1e-3:
            draw_arc(ax3d, O0, np.array([0., 0., 1.]), 0.30,
                     np.array([1., 0., 0.]), theta, THETA_C)
            ax3d.text(0.35 * np.cos(theta / 2),
                      0.35 * np.sin(theta / 2), 0.02,
                      f"theta={np.degrees(theta):.1f}deg",
                      color=THETA_C, fontsize=7.5, fontweight='bold')

        # d  dashed along Z
        if abs(d) > 1e-3:
            draw_dashed_line(ax3d, O0, O2, D_C)
            mid_d = (O0 + O2) / 2 + np.array([0.08, 0., 0.])
            ax3d.text(*mid_d, f"d={d:.2f}",
                      color=D_C, fontsize=7.5, fontweight='bold')

        # a  dashed along X_i
        if abs(a) > 1e-3:
            draw_dashed_line(ax3d, O2, O3, A_C)
            mid_a = (O2 + O3) / 2
            ax3d.text(mid_a[0], mid_a[1] + 0.05, mid_a[2] + 0.05,
                      f"a={a:.2f}",
                      color=A_C, fontsize=7.5, fontweight='bold')

        # alpha arc around X_i
        if abs(alpha) > 1e-3:
            draw_arc(ax3d, O3, T3[:3, 0], 0.25, T3[:3, 1], alpha, ALPHA_C)
            ax3d.text(O3[0] + 0.28, O3[1], O3[2] + 0.10,
                      f"alpha={np.degrees(alpha):.1f}deg",
                      color=ALPHA_C, fontsize=7.5, fontweight='bold')

        # Simple link line  O0 → O4
        draw_link(ax3d, O0, O4)

        # Frame {i}
        draw_frame(ax3d, T4, [FRAME1_X, FRAME1_Y, FRAME1_Z], label="i")
        ax3d.scatter(*O4, color="#994400", s=45, zorder=6, depthshade=False)

    # ── Step 1 : theta ────────────────────────────────────
    elif step == 1:
        if abs(theta) > 1e-3:
            draw_arc(ax3d, O0, np.array([0., 0., 1.]), 0.42,
                     np.array([1., 0., 0.]), theta, THETA_C, lw=2.5)
            ax3d.text(0.48 * np.cos(theta / 2),
                      0.48 * np.sin(theta / 2), 0.03,
                      f"theta = {np.degrees(theta):.1f}deg",
                      color=THETA_C, fontsize=9, fontweight='bold')
        draw_frame(ax3d, T1, [FRAME1_X, FRAME1_Y, FRAME1_Z],
                   label="i{-}1'", alpha_val=0.85)
        ax3d.text2D(0.02, 0.93, "Rot_z(theta)",
                    transform=ax3d.transAxes,
                    color=THETA_C, fontsize=9, fontweight='bold',
                    fontfamily='monospace')

    # ── Step 2 : d ────────────────────────────────────────
    elif step == 2:
        draw_frame(ax3d, T1, [FRAME1_X, FRAME1_Y, FRAME1_Z],
                   label="", alpha_val=0.28)
        if abs(d) > 1e-3:
            draw_dashed_line(ax3d, O0, O2, D_C, lw=2.5)
            mid_d = (O0 + O2) / 2 + np.array([0.10, 0., 0.])
            ax3d.text(*mid_d, f"d = {d:.2f} m",
                      color=D_C, fontsize=9, fontweight='bold')
        draw_frame(ax3d, T2, [FRAME1_X, FRAME1_Y, FRAME1_Z],
                   label="i{-}1''", alpha_val=0.85)
        ax3d.scatter(*O2, color=D_C, s=50, zorder=6, depthshade=False)
        ax3d.text2D(0.02, 0.93, "Trans_z(d)",
                    transform=ax3d.transAxes,
                    color=D_C, fontsize=9, fontweight='bold',
                    fontfamily='monospace')

    # ── Step 3 : a ────────────────────────────────────────
    elif step == 3:
        draw_frame(ax3d, T2, [FRAME1_X, FRAME1_Y, FRAME1_Z],
                   label="", alpha_val=0.28)
        if abs(a) > 1e-3:
            draw_dashed_line(ax3d, O2, O3, A_C, lw=2.5)
            mid_a = (O2 + O3) / 2
            ax3d.text(mid_a[0], mid_a[1] + 0.05, mid_a[2] + 0.08,
                      f"a = {a:.2f} m",
                      color=A_C, fontsize=9, fontweight='bold')
        draw_frame(ax3d, T3, [FRAME1_X, FRAME1_Y, FRAME1_Z],
                   label="i''", alpha_val=0.85)
        ax3d.scatter(*O3, color=A_C, s=50, zorder=6, depthshade=False)
        ax3d.text2D(0.02, 0.93, "Trans_x(a)",
                    transform=ax3d.transAxes,
                    color=A_C, fontsize=9, fontweight='bold',
                    fontfamily='monospace')

    # ── Step 4 : alpha ────────────────────────────────────
    elif step == 4:
        draw_frame(ax3d, T3, [FRAME1_X, FRAME1_Y, FRAME1_Z],
                   label="", alpha_val=0.28)
        if abs(alpha) > 1e-3:
            draw_arc(ax3d, O3, T3[:3, 0], 0.28, T3[:3, 1], alpha,
                     ALPHA_C, lw=2.5)
            ax3d.text(O3[0] + 0.32, O3[1], O3[2] + 0.10,
                      f"alpha = {np.degrees(alpha):.1f}deg",
                      color=ALPHA_C, fontsize=9, fontweight='bold')
        draw_frame(ax3d, T4, [FRAME1_X, FRAME1_Y, FRAME1_Z],
                   label="i", alpha_val=0.85)
        ax3d.text2D(0.02, 0.93, "Rot_x(alpha)",
                    transform=ax3d.transAxes,
                    color=ALPHA_C, fontsize=9, fontweight='bold',
                    fontfamily='monospace')

    # ── Transformation matrix (lower-left, inside 3-D canvas) ──
    T_show = dh_step(theta, d, a, alpha, step=4 if full else step)
    mat_str = (
        "T =\n"
        f"[{T_show[0,0]:+.2f}  {T_show[0,1]:+.2f}  {T_show[0,2]:+.2f}  {T_show[0,3]:+.2f}]\n"
        f"[{T_show[1,0]:+.2f}  {T_show[1,1]:+.2f}  {T_show[1,2]:+.2f}  {T_show[1,3]:+.2f}]\n"
        f"[{T_show[2,0]:+.2f}  {T_show[2,1]:+.2f}  {T_show[2,2]:+.2f}  {T_show[2,3]:+.2f}]\n"
        "[ 0.00   0.00   0.00  +1.00]"
    )
    ax3d.text2D(0.01, 0.01, mat_str,
                transform=ax3d.transAxes,
                color="#223366", fontsize=7, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white', edgecolor='#aabbcc',
                          alpha=0.90, linewidth=1.0))

    # ── Step label  (inside 3-D canvas, top-center) ───────
    #    Uses text2D so it NEVER overlaps the header fig.text above
    if full:
        step_lbl = "Full view  —  all 4 DH parameters"
        tc = "#1a1a2e"
    else:
        step_lbl = f"Step {step}:  {STEP_NAMES[step - 1]}"
        tc = STEP_COLS[step - 1]
    ax3d.text2D(0.50, 0.97, step_lbl,
                transform=ax3d.transAxes,
                color=tc, fontsize=9.5, fontweight='bold',
                fontfamily='monospace', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.28',
                          facecolor='white', edgecolor=tc,
                          alpha=0.88, linewidth=1.2))

    # ── Info panel ────────────────────────────────────────
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
        _full_view[0] = True;  state['step'] = 4
    elif "1:" in label:
        _full_view[0] = False; state['step'] = 1
    elif "2:" in label:
        _full_view[0] = False; state['step'] = 2
    elif "3:" in label:
        _full_view[0] = False; state['step'] = 3
    elif "4:" in label:
        _full_view[0] = False; state['step'] = 4
    update_plot()

def on_reset(_):
    defaults = dict(theta=0.5, d=0.4, a=0.6, alpha=0.4)
    for key, sl in sliders:
        sl.set_val(defaults[key])
    state.update(defaults)
    _full_view[0] = True
    radio.set_active(0)
    update_plot()

for _, sl in sliders:
    sl.on_changed(on_slider)
radio.on_clicked(on_radio)
btn_reset.on_clicked(on_reset)

# ═══════════════════════════════════════════════════════════
#  LAUNCH
# ═══════════════════════════════════════════════════════════
ax3d.view_init(elev=22, azim=-55)
update_plot()
plt.show()
