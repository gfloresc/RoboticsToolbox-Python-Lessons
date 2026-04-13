"""
=============================================================
  Geometric Jacobian — 3D Interactive Visualization
  4-DOF Spatial Robot Arm (RRPR-like configuration)

  Joint layout:
    q1 — base rotation   (Z axis)
    q2 — shoulder pitch  (Y axis)
    q3 — elbow pitch     (Y axis)
    q4 — wrist roll      (Z axis, local)

  Panels:
    LEFT   — 3D robot with joint frames & Jacobian columns
    TOP-R  — Homogeneous transformation T ∈ SE(3)   [live]
    MID-R  — Jacobian matrix Jv ∈ ℝ³ˣ⁴             [live]
    BOT-R  — Concept text (cycles per slide)

  Controls:
    Sliders  θ₁–θ₄   change joint angles
    ← Prev / Next →  cycle lecture slides
    Reset            default configuration
=============================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────────────
P = dict(
    bg      = '#FAFAFA',
    panel   = '#FFFFFF',
    dark    = '#1A202C',
    blue    = '#2B6CB0',
    teal    = '#2C7A7B',
    red     = '#C53030',
    green   = '#276749',
    orange  = '#C05621',
    muted   = '#718096',
    light   = '#E2E8F0',
    blueLt  = '#EBF4FF',
    greenLt = '#F0FFF4',
    amberLt = '#FFFBEB',
    grid    = '#E8EDF2',
    j1      = '#3182CE',
    j2      = '#38A169',
    j3      = '#D69E2E',
    j4      = '#805AD5',
)

# ── Robot geometry (link lengths) ────────────────────────────────
L = [1.6, 1.4, 1.1, 0.6]     # link lengths

# ── Rotation helpers ─────────────────────────────────────────────
def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def make_T(R, p):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = p
    return T

# ── Forward kinematics ────────────────────────────────────────────
def fk_all(q):
    """
    Returns list of 4x4 transforms T0→i for i=0..4 (base to each joint + EE).
    Joint convention:
      Joint 1: rotate Rz(q1) then translate L1 along new X
      Joint 2: rotate Ry(q2) then translate L2 along new X
      Joint 3: rotate Ry(q3) then translate L3 along new X
      Joint 4: rotate Rz(q4) then translate L4 along new X (EE)
    """
    q1, q2, q3, q4 = q

    Ts = [np.eye(4)]   # T0 = identity (base)

    # Joint 1 — base yaw around Z
    R1 = Rz(q1)
    p1 = R1 @ np.array([0, 0, 0.4])   # small vertical offset
    T1 = make_T(R1, p1)
    Ts.append(Ts[-1] @ T1)

    # Joint 2 — shoulder pitch around Y (in frame 1)
    R2 = Ry(q2)
    p2 = np.array([0, 0, 0])           # at joint 1 position, no extra offset
    T2 = make_T(R2, p2)
    # Add link 1 translation
    Tl1 = make_T(np.eye(3), np.array([L[0], 0, 0]))
    Ts.append(Ts[-1] @ T2 @ Tl1)

    # Joint 3 — elbow pitch around Y
    R3 = Ry(q3)
    T3 = make_T(R3, np.zeros(3))
    Tl2 = make_T(np.eye(3), np.array([L[1], 0, 0]))
    Ts.append(Ts[-1] @ T3 @ Tl2)

    # Joint 4 — wrist roll around Z (local)
    R4 = Rz(q4)
    T4 = make_T(R4, np.zeros(3))
    Tl3 = make_T(np.eye(3), np.array([L[2], 0, 0]))
    Ts.append(Ts[-1] @ T4 @ Tl3)

    # End-effector (just translate along local X)
    Tl4 = make_T(np.eye(3), np.array([L[3], 0, 0]))
    Ts.append(Ts[-1] @ Tl4)

    return Ts

def ee_transform(q):
    return fk_all(q)[-1]

# ── Geometric Jacobian (3D, 6×4) ─────────────────────────────────
def jacobian_6n(q):
    """
    Geometric Jacobian J ∈ ℝ⁶ˣ⁴.
    For revolute joint i: Jᵥᵢ = zᵢ × (pₑ − pᵢ), Jωᵢ = zᵢ
    """
    Ts = fk_all(q)
    pe = Ts[-1][:3, 3]

    # z-axes and positions for joints 1-4 (Ts[1] to Ts[4])
    # Ts[0]=base, Ts[1]=after j1, Ts[2]=after j1+link1, Ts[3]=after j2+link2, Ts[4]=after j3+link3
    # Joint frame positions: before each joint
    joint_frames = [Ts[0], Ts[1], Ts[2], Ts[3]]
    joint_axes   = [
        np.array([0,0,1]),   # j1 rotates around Z of frame 0
        np.array([0,1,0]),   # j2 rotates around Y of frame 1
        np.array([0,1,0]),   # j3 rotates around Y of frame 2
        np.array([0,0,1]),   # j4 rotates around Z of frame 3
    ]

    J = np.zeros((6, 4))
    for i in range(4):
        Tf = joint_frames[i]
        zi_local = joint_axes[i]
        zi = Tf[:3,:3] @ zi_local          # z-axis in world frame
        pi = Tf[:3, 3]                     # joint origin in world
        r  = pe - pi                       # vector to EE
        J[:3, i] = np.cross(zi, r)        # linear part
        J[3:, i] = zi                     # angular part
    return J

def Jv(q):
    return jacobian_6n(q)[:3, :]          # linear Jacobian ℝ³ˣ⁴

def manipulability(q):
    Jlin = Jv(q)
    return np.sqrt(max(0, np.linalg.det(Jlin @ Jlin.T)))

# ── Slides ───────────────────────────────────────────────────────
SLIDES = [
    {
        'title':   'Slide 1 — Configuration  q ∈ ℝ⁴',
        'concept': (
            "A 4-DOF spatial robot is described by:\n\n"
            "   q = [θ₁  θ₂  θ₃  θ₄]ᵀ  ∈  ℝ⁴\n\n"
            "Joint types (all revolute):\n"
            "   θ₁ — base yaw    (rotates around Z)\n"
            "   θ₂ — shoulder    (rotates around Y)\n"
            "   θ₃ — elbow       (rotates around Y)\n"
            "   θ₄ — wrist roll  (rotates around Z)\n\n"
            "The colored frames show each joint's\n"
            "local coordinate system."
        ),
        'show_frames': True,
        'show_cols':   False,
        'show_twist':  False,
    },
    {
        'title':   'Slide 2 — Forward Kinematics  T(q) ∈ SE(3)',
        'concept': (
            "Forward kinematics gives the full pose:\n\n"
            "   T(q) ∈ SE(3) — 4×4 matrix\n\n"
            "   T(q) = T₁(q₁)·T₂(q₂)·T₃(q₃)·T₄(q₄)\n\n"
            "The panel (top-right) shows T(q) live.\n\n"
            "   R ∈ SO(3) — end-effector orientation\n"
            "   p ∈ ℝ³   — end-effector position\n\n"
            "The RGB triad at the EE shows the\n"
            "columns of R = [x̂ | ŷ | ẑ]."
        ),
        'show_frames': True,
        'show_cols':   False,
        'show_twist':  False,
    },
    {
        'title':   'Slide 3 — Jacobian Columns  J(:,i) ∈ ℝ⁶',
        'concept': (
            "Each column of J(q) ∈ ℝ⁶ˣ⁴ encodes how\n"
            "one joint's motion affects the EE:\n\n"
            "   Jᵥ(:,i) = zᵢ × (p_EE − pᵢ)   [linear]\n"
            "   Jω(:,i) = zᵢ                   [angular]\n\n"
            "zᵢ is the rotation axis of joint i\n"
            "expressed in the world frame.\n\n"
            "Arrows show Jᵥ(:,i) for each joint.\n"
            "They are always ⊥ to zᵢ and to\n"
            "the vector from pᵢ to p_EE."
        ),
        'show_frames': False,
        'show_cols':   True,
        'show_twist':  False,
    },
    {
        'title':   'Slide 4 — Twist  ξ = J(q) q̇',
        'concept': (
            "The end-effector velocity is:\n\n"
            "   ξ = J(q) q̇ =  [v]\n"
            "                  [ω]\n\n"
            "Setting q̇ = [1,1,1,1]ᵀ shows:\n"
            "   v = Jᵥ q̇  (green arrow = total\n"
            "               linear velocity)\n\n"
            "The full twist ξ ∈ ℝ⁶ also includes\n"
            "angular velocity ω = Jω q̇ ∈ ℝ³,\n"
            "shown as the orange arrow."
        ),
        'show_frames': False,
        'show_cols':   True,
        'show_twist':  True,
    },
    {
        'title':   'Slide 5 — Singularities  μ = √det(JᵥJᵥᵀ)',
        'concept': (
            "Singularities = configurations where the\n"
            "robot loses one or more motion directions.\n\n"
            "Geometrically: columns of J become\n"
            "linearly dependent (parallel/coplanar).\n\n"
            "Algebraically:  μ = √det(JᵥJᵥᵀ) = 0\n\n"
            "μ is the manipulability ellipsoid volume.\n\n"
            "Try θ₂=0, θ₃=0  →  fully extended arm\n"
            "(boundary singularity, μ ≈ 0)"
        ),
        'show_frames': False,
        'show_cols':   True,
        'show_twist':  False,
    },
]

current_slide = [0]

# ── Figure ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 9.0), facecolor=P['bg'])
fig.canvas.manager.set_window_title('Geometric Jacobian 3D — Interactive Demo')

gs = GridSpec(5, 3,
              figure=fig,
              left=0.02, right=0.99, top=0.95, bottom=0.13,
              hspace=0.12, wspace=0.22,
              height_ratios=[0.055, 1.0, 0.72, 0.62, 0.05],
              width_ratios=[1.35, 0.85, 0.78])

ax_title   = fig.add_subplot(gs[0, :])
ax_3d      = fig.add_subplot(gs[1:4, 0], projection='3d')
ax_Tmat    = fig.add_subplot(gs[1, 1:])
ax_Jmat    = fig.add_subplot(gs[2, 1:])
ax_concept = fig.add_subplot(gs[3, 1:])

for ax in [ax_title, ax_Tmat, ax_Jmat, ax_concept]:
    ax.set_axis_off()

# Slider axes
sl_axs = [
    fig.add_axes([0.04, 0.095, 0.34, 0.022]),
    fig.add_axes([0.04, 0.068, 0.34, 0.022]),
    fig.add_axes([0.04, 0.041, 0.34, 0.022]),
    fig.add_axes([0.04, 0.014, 0.34, 0.022]),
]
btn_ax_prev  = fig.add_axes([0.43, 0.016, 0.09, 0.055])
btn_ax_next  = fig.add_axes([0.535, 0.016, 0.09, 0.055])
btn_ax_reset = fig.add_axes([0.64, 0.016, 0.09, 0.055])

# ── Sliders ──────────────────────────────────────────────────────
sl_colors = [P['j1'], P['j2'], P['j3'], P['j4']]
sl_inits  = [np.pi/5, np.pi/4, -np.pi/5, np.pi/6]
sl_labels = ['θ₁  (yaw)', 'θ₂  (shoulder)', 'θ₃  (elbow)', 'θ₄  (wrist)']
sliders   = []
for i, (ax_s, col, val, lbl) in enumerate(zip(sl_axs, sl_colors, sl_inits, sl_labels)):
    sl = Slider(ax_s, lbl, -np.pi, np.pi, valinit=val,
                color=col, track_color=P['light'])
    sl.label.set_fontsize(9.5); sl.label.set_color(P['dark'])
    sl.valtext.set_fontsize(9); sl.valtext.set_color(P['muted'])
    sliders.append(sl)

# ── Buttons ──────────────────────────────────────────────────────
btn_prev  = Button(btn_ax_prev,  '← Prev',  color=P['panel'], hovercolor=P['blueLt'])
btn_next  = Button(btn_ax_next,  'Next →',  color=P['blue'],  hovercolor='#2C5282')
btn_reset = Button(btn_ax_reset, 'Reset',   color=P['panel'], hovercolor=P['light'])
btn_next.label.set_color(P['panel']); btn_next.label.set_fontweight('bold')
btn_prev.label.set_color(P['dark']);  btn_reset.label.set_color(P['dark'])

# ── 3D drawing helpers ───────────────────────────────────────────
def draw_frame(ax, T, scale=0.35, lw=2.5, alpha=1.0):
    """Draw RGB coordinate frame axes at transform T."""
    o = T[:3, 3]
    colors = ['#E53E3E', '#38A169', '#3182CE']   # X=red, Y=green, Z=blue
    for ci, (col, axis_idx) in enumerate(zip(colors, range(3))):
        d = T[:3, axis_idx] * scale
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                  color=col, lw=lw, arrow_length_ratio=0.25,
                  alpha=alpha)

def draw_cylinder(ax, p1, p2, radius=0.055, color='#3182CE', alpha=0.85, n=16):
    """Draw a cylindrical link from p1 to p2."""
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-6:
        return
    v_n = v / length

    # Build orthonormal basis
    arbitrary = np.array([0,0,1]) if abs(v_n[0]) < 0.9 else np.array([0,1,0])
    u = np.cross(v_n, arbitrary)
    u /= np.linalg.norm(u)
    w = np.cross(v_n, u)

    theta = np.linspace(0, 2*np.pi, n)
    # Caps and side
    for t_start, t_end in [(0, 1)]:
        pts1 = np.array([p1 + radius*(np.cos(t)*u + np.sin(t)*w) for t in theta])
        pts2 = np.array([p2 + radius*(np.cos(t)*u + np.sin(t)*w) for t in theta])
        for k in range(n-1):
            verts = [pts1[k], pts1[k+1], pts2[k+1], pts2[k]]
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(color)
            poly.set_edgecolor('none')
            ax.add_collection3d(poly)

def draw_sphere(ax, center, radius=0.14, color='#1A202C', alpha=1.0, n=12):
    """Draw a sphere at joint location."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True,
                    linewidth=0, antialiased=True)

# ── Draw T matrix panel ───────────────────────────────────────────
def draw_Tmat(ax, T):
    ax.clear(); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    # Header
    ax.add_patch(FancyBboxPatch((0.1,8.5),9.8,1.4,
        boxstyle='round,pad=0.1', fc=P['dark'], ec='none'))
    ax.text(5.0, 9.25, 'Homogeneous Transformation  T(q) ∈ SE(3)',
            ha='center', va='center', fontsize=11, color='white',
            fontweight='bold')
    ax.text(0.5, 8.65, '4×4  matrix = rotation R + translation p + projective row',
            ha='left', va='center', fontsize=8.5, color=P['muted'])

    R = T[:3,:3]
    p = T[:3, 3]

    # Column headers
    for ci, lbl in enumerate(['col 1', 'col 2', 'col 3', 'col 4']):
        xc = 1.6 + ci*2.0
        col_c = [P['j1'], P['j2'], P['j3'], P['teal']][ci]
        ax.add_patch(FancyBboxPatch((xc-0.85,7.8),1.7,0.58,
            boxstyle='round,pad=0.05', fc=col_c, ec='none'))
        ax.text(xc, 8.1, lbl, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

    row_labels = ['row 1', 'row 2', 'row 3', 'row 4']
    row_desc   = ['', '', '', '']
    for ri in range(4):
        ax.add_patch(FancyBboxPatch((0.05,6.0+ri*(-1.72)+ri*0.0-0.5*(4-ri)),
            0.9, 0.62, boxstyle='round,pad=0.05', fc='#F0F4FF', ec='none'))

    # Matrix values
    for ri in range(4):
        ry = 7.2 - ri*1.72
        for ci in range(4):
            xc = 1.6 + ci*2.0
            val = T[ri, ci]

            # Background coloring: R block vs p column vs bottom row
            if ri < 3 and ci < 3:
                # Rotation block — color by column
                col_c = [P['j1'], P['j2'], P['j3']][ci]
                bg = col_c + '22'
            elif ri < 3 and ci == 3:
                bg = P['teal'] + '22'
                col_c = P['teal']
            else:
                bg = '#F0F0F0'
                col_c = P['muted']

            ax.add_patch(FancyBboxPatch((xc-0.85, ry-0.38), 1.7, 0.76,
                boxstyle='round,pad=0.04', fc=bg, ec=col_c+'55', lw=0.8))
            ax.text(xc, ry, f'{val:+.4f}', ha='center', va='center',
                    fontsize=9.5, color=P['dark'],
                    fontfamily='monospace', fontweight='bold')

    # Bracket annotations
    ax.text(0.38, 5.55, '⎡\n⎢\n⎢\n⎣', ha='center', va='center',
            fontsize=28, color=P['dark'], linespacing=1.05)
    ax.text(9.72, 5.55, '⎤\n⎥\n⎥\n⎦', ha='center', va='center',
            fontsize=28, color=P['dark'], linespacing=1.05)

    # Semantic labels
    ax.add_patch(FancyBboxPatch((0.05, 3.42), 2.2, 1.0,
        boxstyle='round,pad=0.1', fc='#EBF4FF', ec=P['blue'], lw=1))
    ax.text(1.15, 4.0, 'R ∈ SO(3)', ha='center', fontsize=9,
            color=P['blue'], fontweight='bold')
    ax.text(1.15, 3.62, 'orientation', ha='center', fontsize=8, color=P['muted'])

    ax.add_patch(FancyBboxPatch((7.8, 3.42), 2.1, 1.0,
        boxstyle='round,pad=0.1', fc='#E6FFFA', ec=P['teal'], lw=1))
    ax.text(8.85, 4.0, 'p ∈ ℝ³', ha='center', fontsize=9,
            color=P['teal'], fontweight='bold')
    ax.text(8.85, 3.62, 'position', ha='center', fontsize=8, color=P['muted'])

    # Position vector values
    ax.text(5.0, 3.15,
            f'p = ({p[0]:+.3f},  {p[1]:+.3f},  {p[2]:+.3f})  m',
            ha='center', fontsize=10, color=P['teal'],
            fontfamily='monospace', fontweight='bold')

# ── Draw Jacobian panel ───────────────────────────────────────────
def draw_Jmat_panel(ax, J6, q):
    ax.clear(); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    Jlin = J6[:3, :]   # linear part ℝ³ˣ⁴
    Jang = J6[3:, :]   # angular part

    ax.add_patch(FancyBboxPatch((0.1,8.3),9.8,1.6,
        boxstyle='round,pad=0.1', fc=P['dark'], ec='none'))
    ax.text(5.0, 9.15, 'Geometric Jacobian  J(q) ∈ ℝ⁶ˣ⁴',
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.text(0.5, 8.48, 'Upper 3 rows: linear  Jᵥ     Lower 3 rows: angular  Jω',
            ha='left', fontsize=8.5, color=P['muted'])

    col_colors = [P['j1'], P['j2'], P['j3'], P['j4']]
    cxs = [2.1, 4.0, 5.9, 7.8]

    # Column headers
    for ci, (cx, cc) in enumerate(zip(cxs, col_colors)):
        ax.add_patch(FancyBboxPatch((cx-0.83,7.5),1.66,0.68,
            boxstyle='round,pad=0.04', fc=cc, ec='none'))
        ax.text(cx, 7.85, f'j{ci+1}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    row_labels = ['vₓ', 'vy', 'vz', 'ωₓ', 'ωy', 'ωz']
    row_bg     = ['#EBF4FF']*3 + ['#F0FFF4']*3
    rys        = [6.8, 5.95, 5.1, 4.05, 3.2, 2.35]

    for ri, (rl, rbg, ry) in enumerate(zip(row_labels, row_bg, rys)):
        ax.add_patch(FancyBboxPatch((0.1, ry-0.42), 1.1, 0.84,
            boxstyle='round,pad=0.04', fc=P['dark'], ec='none'))
        ax.text(0.65, ry, rl, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

        full_row = np.concatenate([Jlin[ri] if ri<3 else Jang[ri-3]])
        max_abs  = max(np.max(np.abs(full_row)), 1e-6)

        for ci, (cx, cc) in enumerate(zip(cxs, col_colors)):
            val = J6[ri, ci]
            mag = abs(val)/max_abs
            import matplotlib.colors as mc
            rgba = mc.to_rgba(cc, alpha=0.12 + 0.30*mag)
            ax.add_patch(FancyBboxPatch((cx-0.83, ry-0.42), 1.66, 0.84,
                boxstyle='round,pad=0.03', fc=rgba, ec=cc+'66', lw=0.7))
            ax.text(cx, ry, f'{val:+.3f}', ha='center', va='center',
                    fontsize=8.8, color=P['dark'],
                    fontfamily='monospace', fontweight='bold')

    # Separator between Jv and Jw
    ax.plot([0.1, 9.9], [4.62, 4.62], color=P['light'], lw=1.5)
    ax.text(9.65, 4.58, 'Jᵥ', ha='right', fontsize=9, color=P['blue'], fontweight='bold')
    ax.text(9.65, 4.68, 'Jω', ha='right', fontsize=9, color=P['green'], va='bottom', fontweight='bold')

    # Manipulability gauge
    mu = manipulability(q)
    mu_max = 4.0
    frac = np.clip(mu/mu_max, 0, 1)
    bar_col = P['red'] if frac<0.08 else P['orange'] if frac<0.35 else P['green']
    ax.add_patch(FancyBboxPatch((1.4, 1.25), 7.2, 0.55,
        boxstyle='round,pad=0.05', fc=P['light'], ec=P['muted'], lw=0.7))
    if frac > 0.01:
        ax.add_patch(FancyBboxPatch((1.4, 1.25), 7.2*frac, 0.55,
            boxstyle='round,pad=0.05', fc=bar_col, ec='none'))
    ax.text(5.0, 0.78,
            f'μ = {mu:.4f}   '
            + ('⚠ NEAR SINGULARITY' if frac<0.08 else '✓ full rank' if frac>0.45 else ''),
            ha='center', fontsize=9.5, color=bar_col,
            fontweight='bold', fontfamily='monospace')
    ax.text(0.9, 1.52, '✗', ha='center', fontsize=9, color=P['red'])
    ax.text(8.95, 1.52, '✓', ha='center', fontsize=9, color=P['green'])
    ax.text(5.0, 2.05, 'Manipulability  μ = √det(JᵥJᵥᵀ)',
            ha='center', fontsize=9, color=P['muted'])

# ── Draw concept panel ────────────────────────────────────────────
def draw_concept(ax, slide, q, Ts):
    ax.clear(); ax.set_axis_off()
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.add_patch(FancyBboxPatch((0.1,0.2),9.8,9.6,
        boxstyle='round,pad=0.1', fc=P['panel'], ec=P['light'], lw=1.0))
    ax.add_patch(FancyBboxPatch((0.1,8.5),9.8,1.3,
        boxstyle='round,pad=0.1', fc=P['blue']+'22', ec=P['blue'], lw=1.2))
    ax.text(5.0,9.2,'Concept', ha='center', fontsize=10,
            color=P['blue'], fontweight='bold')
    ax.text(0.7, 8.4, slide['concept'],
            ha='left', va='top', fontsize=9.8,
            color=P['dark'], linespacing=1.55, fontfamily='monospace')

    # Live q
    ax.add_patch(FancyBboxPatch((0.3,0.25),9.4,1.95,
        boxstyle='round,pad=0.1', fc=P['blueLt'], ec=P['blue'], lw=1.0))
    ax.text(5.0,2.0,'q  (current)', ha='center', fontsize=9,
            color=P['blue'], fontweight='bold')
    jnames = ['θ₁','θ₂','θ₃','θ₄']
    txt = '   '.join(f'{jn}={np.degrees(q[i]):+.0f}°' for i,jn in enumerate(jnames))
    ax.text(5.0, 1.35, txt, ha='center', fontsize=9.5,
            color=P['dark'], fontfamily='monospace', fontweight='bold')
    p = Ts[-1][:3,3]
    ax.text(5.0, 0.68,
            f'p_EE = ({p[0]:+.3f},  {p[1]:+.3f},  {p[2]:+.3f})',
            ha='center', fontsize=9, color=P['teal'], fontfamily='monospace')

# ── Main draw ────────────────────────────────────────────────────
def draw(val=None):
    q = np.array([sl.val for sl in sliders])
    Ts = fk_all(q)
    T_ee = Ts[-1]
    J6 = jacobian_6n(q)
    slide = SLIDES[current_slide[0]]

    # ── Title ────────────────────────────────────────────────────
    ax_title.clear(); ax_title.set_axis_off()
    ax_title.set_xlim(0,1); ax_title.set_ylim(0,1)
    ax_title.add_patch(mpatches.FancyBboxPatch((0,0),1,1,
        boxstyle='square', fc=P['dark'], ec='none',
        transform=ax_title.transAxes))
    ax_title.add_patch(mpatches.FancyBboxPatch((0,0),0.012,1,
        boxstyle='square', fc=P['blue'], ec='none',
        transform=ax_title.transAxes))
    ax_title.text(0.015, 0.5, slide['title'],
        transform=ax_title.transAxes,
        ha='left', va='center', fontsize=13, fontweight='bold', color='white')
    ax_title.text(0.988, 0.5, f'{current_slide[0]+1} / {len(SLIDES)}',
        transform=ax_title.transAxes,
        ha='right', va='center', fontsize=11, color=P['muted'])

    # ── 3D Robot ─────────────────────────────────────────────────
    ax_3d.clear()
    ax_3d.set_facecolor(P['panel'])
    lim = 3.5
    ax_3d.set_xlim(-lim, lim); ax_3d.set_ylim(-lim, lim); ax_3d.set_zlim(-0.5, lim*1.5)
    ax_3d.set_xlabel('X [m]', fontsize=8, color=P['muted'])
    ax_3d.set_ylabel('Y [m]', fontsize=8, color=P['muted'])
    ax_3d.set_zlabel('Z [m]', fontsize=8, color=P['muted'])
    ax_3d.tick_params(labelsize=7, colors=P['muted'])
    ax_3d.set_title('4R Spatial Robot', fontsize=10,
                    color=P['dark'], pad=4, fontweight='bold')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor(P['light'])
    ax_3d.yaxis.pane.set_edgecolor(P['light'])
    ax_3d.zaxis.pane.set_edgecolor(P['light'])
    ax_3d.grid(True, color=P['grid'], lw=0.5, alpha=0.5)

    # Floor shadow
    pts_x = [Ts[i][:3,3][0] for i in range(len(Ts))]
    pts_y = [Ts[i][:3,3][1] for i in range(len(Ts))]
    ax_3d.plot(pts_x, pts_y, [-0.45]*len(pts_x),
               color=P['muted'], lw=1.0, alpha=0.25, zorder=0)

    # World frame
    draw_frame(ax_3d, np.eye(4), scale=0.45, lw=1.5, alpha=0.4)

    # Links
    link_cols = [P['j1'], P['j2'], P['j3'], P['j4']]
    joint_Ts = [Ts[0], Ts[1], Ts[2], Ts[3], Ts[4]]   # 5 positions (base + 4 after joints)

    for i in range(4):
        p1 = joint_Ts[i][:3,3]
        p2 = joint_Ts[i+1][:3,3]
        draw_cylinder(ax_3d, p1, p2, radius=0.07, color=link_cols[i], alpha=0.88)

    # EE extension
    p_last = Ts[4][:3,3]
    p_ee   = Ts[5][:3,3]
    draw_cylinder(ax_3d, p_last, p_ee, radius=0.045, color='#A0AEC0', alpha=0.7)

    # Joints (spheres)
    joint_colors = ['#1A202C', P['j1'], P['j2'], P['j3'], P['j4']]
    joint_radii  = [0.16, 0.14, 0.12, 0.11, 0.10]
    for i, (Ti, col, rad) in enumerate(zip(joint_Ts[:5], joint_colors, joint_radii)):
        draw_sphere(ax_3d, Ti[:3,3], radius=rad, color=col, alpha=0.95)

    # Joint frames
    if slide['show_frames']:
        for i, Ti in enumerate(joint_Ts[:5]):
            draw_frame(ax_3d, Ti, scale=0.32, lw=1.8, alpha=0.75)

    # EE frame (always shown)
    draw_frame(ax_3d, T_ee, scale=0.55, lw=2.5, alpha=1.0)

    # EE dot
    pe = T_ee[:3,3]
    ax_3d.scatter(*pe, color=P['red'], s=90, zorder=10, depthshade=False)

    # ── Jacobian column arrows ────────────────────────────────────
    if slide['show_cols']:
        for i in range(4):
            col_vec = J6[:3, i]   # linear part
            mag = np.linalg.norm(col_vec) + 1e-8
            col_n = col_vec / mag * min(mag, 1.5)
            orig = pe.copy()
            ax_3d.quiver(*orig, *col_n,
                         color=link_cols[i], lw=2.0,
                         arrow_length_ratio=0.22, alpha=0.9,
                         normalize=False, length=1.0)
            # Label at tip
            tip = orig + col_n*1.05
            ax_3d.text(tip[0], tip[1], tip[2], f'J[:, {i+1}]',
                       fontsize=7.5, color=link_cols[i],
                       fontweight='bold', ha='center')

    # ── Twist arrows ─────────────────────────────────────────────
    if slide['show_twist']:
        qdot = np.ones(4)
        v = J6[:3] @ qdot
        w = J6[3:] @ qdot
        vm = np.linalg.norm(v)+1e-8; wm = np.linalg.norm(w)+1e-8
        vn = v/vm * min(vm,1.8)
        wn = w/wm * min(wm,1.4)
        ax_3d.quiver(*pe, *vn, color=P['green'], lw=3.5,
                     arrow_length_ratio=0.22, alpha=0.95, normalize=False, length=1.0)
        ax_3d.text(*(pe+vn*1.1), 'v = Jᵥq̇', fontsize=8.5,
                   color=P['green'], fontweight='bold')
        ax_3d.quiver(*pe, *wn, color=P['orange'], lw=3.0,
                     arrow_length_ratio=0.22, alpha=0.85, normalize=False, length=1.0)
        ax_3d.text(*(pe+wn*1.12), 'ω = Jωq̇', fontsize=8.5,
                   color=P['orange'], fontweight='bold')

    # Elevation for nice view
    ax_3d.view_init(elev=22, azim=-55)

    # ── Right panels ──────────────────────────────────────────────
    draw_Tmat(ax_Tmat, T_ee)
    draw_Jmat_panel(ax_Jmat, J6, q)
    draw_concept(ax_concept, slide, q, Ts)

    fig.canvas.draw_idle()

# ── Callbacks ────────────────────────────────────────────────────
def next_slide(ev):
    current_slide[0] = (current_slide[0]+1) % len(SLIDES); draw()
def prev_slide(ev):
    current_slide[0] = (current_slide[0]-1) % len(SLIDES); draw()
def reset(ev):
    for sl, v in zip(sliders, sl_inits): sl.set_val(v)

btn_next.on_clicked(next_slide)
btn_prev.on_clicked(prev_slide)
btn_reset.on_clicked(reset)
for sl in sliders:
    sl.on_changed(draw)

# Footer
fig.text(0.04, 0.107, 'Joint angles (drag to explore):', fontsize=9,
         color=P['muted'], va='center')
fig.text(0.43, 0.005,
         '← Prev / Next →  cycle slides     |     RGB triad = end-effector frame axes  x̂ ŷ ẑ',
         fontsize=8.5, color=P['muted'])

draw()
plt.show()
