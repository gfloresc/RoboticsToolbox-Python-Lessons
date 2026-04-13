"""
=============================================================
  Geometric Jacobian — Interactive Visualization
  3-DOF Planar Robot Arm (3R manipulator)
=============================================================
  Slides covered:
    1. Robot configuration  q = [θ₁, θ₂, θ₃]ᵀ
    2. Forward kinematics   p = p(q)
    3. Linear Jacobian      v = Jᵥ(q) q̇
    4. The twist            ξ = J(q) q̇
    5. Singularities        det(Jᵥ) = 0

  Controls:
    • Sliders  — change joint angles
    • Buttons  — cycle through lecture slides
    • Drag end-effector (coming: IK demo)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────────────
P = dict(
    bg       = '#FAFAFA',
    panel    = '#FFFFFF',
    dark     = '#1A202C',
    blue     = '#2B6CB0',
    teal     = '#2C7A7B',
    red      = '#C53030',
    green    = '#276749',
    orange   = '#C05621',
    muted    = '#718096',
    light    = '#E2E8F0',
    blueLt   = '#EBF4FF',
    greenLt  = '#F0FFF4',
    amberLt  = '#FFFBEB',
    link1    = '#3182CE',
    link2    = '#38A169',
    link3    = '#D69E2E',
    joint    = '#1A202C',
    ee       = '#C53030',
    grid     = '#E8EDF2',
)

# ── Robot parameters ─────────────────────────────────────────────
L = np.array([1.8, 1.4, 1.0])   # link lengths

def fk(q):
    """Forward kinematics: returns joint positions + end-effector."""
    x, y = [0.0], [0.0]
    ang = 0.0
    for i, (li, qi) in enumerate(zip(L, q)):
        ang += qi
        x.append(x[-1] + li * np.cos(ang))
        y.append(y[-1] + li * np.sin(ang))
    return np.array(x), np.array(y)

def jacobian(q):
    """Geometric Jacobian Jᵥ ∈ ℝ²ˣ³ for planar 3R robot."""
    n = len(q)
    Jv = np.zeros((2, n))
    xN, yN = fk(q)
    pe = np.array([xN[-1], yN[-1]])      # end-effector
    xj, yj = [0.0], [0.0]
    ang = 0.0
    for i in range(n):
        ang += q[i]
        xj.append(xj[-1] + L[i]*np.cos(ang))
        yj.append(yj[-1] + L[i]*np.sin(ang))
    # Jᵥᵢ = zᵢ × (pₑ − pᵢ)  for revolute (planar: z = ẑ)
    for i in range(n):
        pi = np.array([xj[i], yj[i]])
        r  = pe - pi
        Jv[0, i] = -r[1]   # −(pₑy − pᵢy)
        Jv[1, i] =  r[0]   #  (pₑx − pᵢx)
    return Jv

def singularity_measure(q):
    Jv = jacobian(q)
    return abs(np.linalg.det(Jv @ Jv.T))**0.5   # sqrt(det(JJᵀ))

# ── Slide content ────────────────────────────────────────────────
SLIDES = [
    {
        'title':   'Slide 1 — Configuration Vector  q',
        'concept': (
            "The robot's configuration is fully described by\n"
            "the joint angle vector:\n\n"
            "         q = [θ₁  θ₂  θ₃]ᵀ  ∈  ℝ³\n\n"
            "Each angle θᵢ is measured relative to the\n"
            "previous link.  Drag the sliders to explore\n"
            "how q maps to different arm shapes."
        ),
        'highlight': 'joints',
    },
    {
        'title':   'Slide 2 — Forward Kinematics  p = p(q)',
        'concept': (
            "Forward kinematics computes the end-effector\n"
            "position from joint angles:\n\n"
            "         p(q) ∈ ℝ²  (planar case)\n\n"
            "  x_EE = Σᵢ Lᵢ cos(θ₁+…+θᵢ)\n"
            "  y_EE = Σᵢ Lᵢ sin(θ₁+…+θᵢ)\n\n"
            "The red dot tracks the end-effector\n"
            "as you change q."
        ),
        'highlight': 'ee',
    },
    {
        'title':   'Slide 3 — Jacobian Columns  Jᵥ(:,i)',
        'concept': (
            "Each column of Jᵥ(q) ∈ ℝ²ˣ³ answers:\n\n"
            "  'If only joint i moves at rate q̇ᵢ=1,\n"
            "   how fast does the end-effector move?'\n\n"
            "  Jᵥ(:,i) = ẑ × (p_EE − pᵢ)\n\n"
            "Arrows show the contribution of each\n"
            "joint.  Note they are always tangent to\n"
            "the circle centered at joint i."
        ),
        'highlight': 'columns',
    },
    {
        'title':   'Slide 4 — Twist  ξ = Jᵥ(q) q̇',
        'concept': (
            "The end-effector velocity is the sum of\n"
            "all column contributions:\n\n"
            "  v = Jᵥ(q) q̇ = Σᵢ Jᵥ(:,i) · q̇ᵢ\n\n"
            "Set q̇ = [1, 1, 1]ᵀ to see v = Jᵥ 1.\n"
            "The green arrow is the total velocity.\n\n"
            "(For a 3R planar robot, the full Jacobian\n"
            "J ∈ ℝ³ˣ³ also includes angular velocity ω.)"
        ),
        'highlight': 'twist',
    },
    {
        'title':   'Slide 5 — Singularities  det(Jᵥ JᵥT) = 0',
        'concept': (
            "A singularity occurs when the robot loses\n"
            "the ability to move in some direction.\n\n"
            "Geometrically: two or more Jacobian columns\n"
            "become parallel (linearly dependent).\n\n"
            "Algebraically:  det(Jᵥ JᵥT) = 0\n\n"
            "The gauge bar shows proximity to a\n"
            "singularity.  Try fully extending the arm!"
        ),
        'highlight': 'singular',
    },
]

current_slide = [0]

# ── Figure layout ────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8.5), facecolor=P['bg'])
fig.canvas.manager.set_window_title('Geometric Jacobian — Interactive Demo')

gs = GridSpec(
    3, 3,
    figure=fig,
    left=0.04, right=0.98, top=0.93, bottom=0.18,
    hspace=0.35, wspace=0.3,
    height_ratios=[0.06, 1.0, 0.3],
    width_ratios=[1.05, 0.9, 0.85],
)

ax_robot   = fig.add_subplot(gs[1, 0])   # main robot view
ax_concept = fig.add_subplot(gs[1, 1])   # concept text / equations
ax_Jmat    = fig.add_subplot(gs[1, 2])   # Jacobian matrix visual
ax_title   = fig.add_subplot(gs[0, :])   # title bar

for ax in [ax_title, ax_concept, ax_Jmat]:
    ax.set_axis_off()

# Slider axes
ax_s1 = fig.add_axes([0.06, 0.12, 0.26, 0.025])
ax_s2 = fig.add_axes([0.06, 0.08, 0.26, 0.025])
ax_s3 = fig.add_axes([0.06, 0.04, 0.26, 0.025])

# Button axes
ax_prev = fig.add_axes([0.37, 0.04, 0.10, 0.06])
ax_next = fig.add_axes([0.49, 0.04, 0.10, 0.06])
ax_reset= fig.add_axes([0.62, 0.04, 0.10, 0.06])

# ── Sliders ──────────────────────────────────────────────────────
sl1 = Slider(ax_s1, 'θ₁', -np.pi, np.pi, valinit=np.pi/4,
             color=P['link1'], track_color=P['light'])
sl2 = Slider(ax_s2, 'θ₂', -np.pi, np.pi, valinit=np.pi/4,
             color=P['link2'], track_color=P['light'])
sl3 = Slider(ax_s3, 'θ₃', -np.pi, np.pi, valinit=-np.pi/6,
             color=P['link3'], track_color=P['light'])

for sl in [sl1, sl2, sl3]:
    sl.label.set_fontsize(11)
    sl.label.set_color(P['dark'])
    sl.valtext.set_fontsize(10)
    sl.valtext.set_color(P['muted'])

# ── Buttons ──────────────────────────────────────────────────────
btn_prev  = Button(ax_prev,  '← Prev',  color=P['panel'], hovercolor=P['blueLt'])
btn_next  = Button(ax_next,  'Next →',  color=P['blue'],  hovercolor='#2C5282')
btn_reset = Button(ax_reset, 'Reset',   color=P['panel'], hovercolor=P['light'])

btn_next.label.set_color(P['panel'])
btn_next.label.set_fontweight('bold')
btn_prev.label.set_color(P['dark'])
btn_reset.label.set_color(P['dark'])

# ── Helper: draw colored arc for joint angle ──────────────────────
def draw_angle_arc(ax, cx, cy, r, start_deg, span_deg, color, lw=1.5):
    arc = Arc((cx, cy), 2*r, 2*r,
              angle=0, theta1=min(start_deg, start_deg+span_deg),
              theta2=max(start_deg, start_deg+span_deg),
              color=color, lw=lw, zorder=6)
    ax.add_patch(arc)

# ── Draw Jacobian matrix ──────────────────────────────────────────
def draw_Jmat(ax, Jv, qdot=None):
    ax.clear(); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    ax.text(5, 9.5, 'Jacobian  Jᵥ(q)  ∈  ℝ²ˣ³',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color=P['dark'], fontfamily='monospace')

    # Column headers
    cols = ['Col 1\n(joint 1)', 'Col 2\n(joint 2)', 'Col 3\n(joint 3)']
    col_colors = [P['link1'], P['link2'], P['link3']]
    cxs = [2.0, 5.0, 8.0]

    for ci, (cx, col, cc) in enumerate(zip(cxs, cols, col_colors)):
        ax.add_patch(FancyBboxPatch((cx-1.25, 7.8), 2.5, 0.85,
                     boxstyle='round,pad=0.05', fc=cc, ec='none', zorder=2))
        ax.text(cx, 8.22, col, ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold', zorder=3)

    # Row labels
    rows = ['v_x', 'v_y']
    ry   = [6.5, 5.1]
    for r, y in zip(rows, ry):
        ax.add_patch(FancyBboxPatch((0.0, y-0.5), 1.1, 0.95,
                     boxstyle='round,pad=0.05', fc=P['dark'], ec='none', zorder=2))
        ax.text(0.55, y, r, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=3)

    # Values
    for ri, ry_val in enumerate(ry):
        for ci, cx in enumerate(cxs):
            val = Jv[ri, ci]
            mag = abs(val) / (np.max(np.abs(Jv)) + 1e-6)
            bg = (*[int(235 - 60*mag)]*3,)
            bgf = '#{:02x}{:02x}{:02x}'.format(*[int(235 - 60*mag)]*3)

            # color-code by column
            alpha = 0.15 + 0.35*mag
            import matplotlib.colors as mc
            rgba = mc.to_rgba(col_colors[ci], alpha=alpha)
            ax.add_patch(FancyBboxPatch((cx-1.22, ry_val-0.52), 2.44, 0.94,
                         boxstyle='round,pad=0.03', fc=rgba, ec=col_colors[ci],
                         linewidth=1.0, zorder=2))
            ax.text(cx, ry_val, f'{val:+.3f}', ha='center', va='center',
                    fontsize=10.5, color=P['dark'],
                    fontfamily='monospace', fontweight='bold', zorder=3)

    # Determinant / manipulability
    mu = singularity_measure(np.array([sl1.val, sl2.val, sl3.val]))
    mu_max = np.prod(L)
    frac = np.clip(mu / (mu_max*0.5), 0, 1)

    ax.text(5, 3.9, 'Manipulability  μ = √det(JᵥJᵥᵀ)', ha='center',
            fontsize=9.5, color=P['dark'])
    ax.text(5, 3.3, f'{mu:.4f}', ha='center', fontsize=13,
            color=P['red'] if frac < 0.15 else P['green'] if frac > 0.55 else P['orange'],
            fontweight='bold', fontfamily='monospace')

    # Gauge bar
    bar_x, bar_y, bar_w, bar_h = 1.5, 2.5, 7.0, 0.55
    ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w, bar_h,
                 boxstyle='round,pad=0.05', fc=P['light'], ec=P['muted'], lw=0.8))
    bar_color = P['red'] if frac < 0.15 else P['orange'] if frac < 0.45 else P['green']
    if frac > 0.01:
        ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w*frac, bar_h,
                     boxstyle='round,pad=0.05', fc=bar_color, ec='none'))
    ax.text(bar_x-0.2, bar_y+bar_h/2, '✗', ha='right', va='center',
            fontsize=10, color=P['red'])
    ax.text(bar_x+bar_w+0.2, bar_y+bar_h/2, '✓', ha='left', va='center',
            fontsize=10, color=P['green'])

    status = 'NEAR SINGULARITY !' if frac < 0.12 else 'full rank' if frac > 0.5 else 'reduced rank'
    sc = P['red'] if frac < 0.12 else P['muted']
    ax.text(5, 1.9, status, ha='center', fontsize=9.5, color=sc,
            fontweight='bold' if frac < 0.12 else 'normal')

    # q̇ and velocity if twist slide
    if qdot is not None:
        v = Jv @ qdot
        ax.text(5, 1.2, f'q̇ = [1, 1, 1]ᵀ  →  v = [{v[0]:+.2f},  {v[1]:+.2f}]ᵀ',
                ha='center', fontsize=9, color=P['teal'],
                fontfamily='monospace')

# ── Main draw function ────────────────────────────────────────────
def draw(val=None):
    q   = np.array([sl1.val, sl2.val, sl3.val])
    xs, ys = fk(q)
    Jv  = jacobian(q)
    slide = SLIDES[current_slide[0]]
    hl    = slide['highlight']

    # ── Title bar ─────────────────────────────────────────────────
    ax_title.clear(); ax_title.set_axis_off()
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle='square', fc=P['dark'], ec='none',
        transform=ax_title.transAxes, zorder=0))
    ax_title.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 0.016, 1, boxstyle='square', fc=P['blue'], ec='none',
        transform=ax_title.transAxes, zorder=1))
    ax_title.text(0.02, 0.5, slide['title'],
                  transform=ax_title.transAxes,
                  ha='left', va='center', fontsize=13, fontweight='bold',
                  color='white')
    pcount = f'{current_slide[0]+1} / {len(SLIDES)}'
    ax_title.text(0.985, 0.5, pcount,
                  transform=ax_title.transAxes,
                  ha='right', va='center', fontsize=11, color=P['muted'])

    # ── Robot view ────────────────────────────────────────────────
    ax_robot.clear()
    ax_robot.set_facecolor(P['panel'])
    ax_robot.set_xlim(-4.8, 4.8)
    ax_robot.set_ylim(-4.8, 4.8)
    ax_robot.set_aspect('equal')
    ax_robot.set_title('3R Planar Robot', fontsize=11,
                        color=P['dark'], pad=6, fontweight='bold')

    # Grid
    for g in np.arange(-4, 5, 1):
        ax_robot.axhline(g, color=P['grid'], lw=0.6, zorder=0)
        ax_robot.axvline(g, color=P['grid'], lw=0.6, zorder=0)
    ax_robot.axhline(0, color=P['muted'], lw=0.8, zorder=0)
    ax_robot.axvline(0, color=P['muted'], lw=0.8, zorder=0)

    # Reachability circle
    total_reach = np.sum(L)
    circ = plt.Circle((0,0), total_reach, color=P['blueLt'],
                       fill=True, ec=P['light'], lw=0.8, zorder=1, alpha=0.4)
    ax_robot.add_patch(circ)

    # ── Draw links ────────────────────────────────────────────────
    link_colors = [P['link1'], P['link2'], P['link3']]
    for i in range(3):
        lw = 9 - i*1.5
        ax_robot.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                      color=link_colors[i], lw=lw,
                      solid_capstyle='round', zorder=3)
        # Shadow
        ax_robot.plot([xs[i]+0.04, xs[i+1]+0.04],
                      [ys[i]-0.04, ys[i+1]-0.04],
                      color='#00000015', lw=lw+2,
                      solid_capstyle='round', zorder=2)

    # ── Jacobian columns as arrows (slide 3) ──────────────────────
    if hl in ('columns', 'twist', 'singular'):
        arrow_scale = 1.2
        for i in range(3):
            col = Jv[:, i]
            mag = np.linalg.norm(col) + 1e-8
            col_n = col / mag * min(mag, 1.5) * arrow_scale
            ax_robot.annotate('',
                xy=(xs[-1] + col_n[0], ys[-1] + col_n[1]),
                xytext=(xs[-1], ys[-1]),
                arrowprops=dict(
                    arrowstyle='->', color=link_colors[i],
                    lw=2.2, mutation_scale=16),
                zorder=7)
            ax_robot.text(xs[-1]+col_n[0]*1.12, ys[-1]+col_n[1]*1.12,
                          f'J[:,{i+1}]', fontsize=8.5, color=link_colors[i],
                          fontweight='bold', ha='center', va='center', zorder=8)

    # ── Total velocity (slide 4) ──────────────────────────────────
    if hl == 'twist':
        qdot = np.ones(3)
        v = Jv @ qdot
        vmag = np.linalg.norm(v) + 1e-8
        vn = v / vmag * min(vmag, 2.0) * 1.2
        ax_robot.annotate('',
            xy=(xs[-1]+vn[0], ys[-1]+vn[1]),
            xytext=(xs[-1], ys[-1]),
            arrowprops=dict(arrowstyle='->', color=P['green'],
                            lw=3.0, mutation_scale=22), zorder=9)
        ax_robot.text(xs[-1]+vn[0]*1.14, ys[-1]+vn[1]*1.14,
                      'v = Jᵥ q̇', fontsize=10, color=P['green'],
                      fontweight='bold', ha='center', zorder=10,
                      bbox=dict(fc=P['greenLt'], ec=P['green'], boxstyle='round,pad=0.3', lw=1))

    # ── Joints ────────────────────────────────────────────────────
    joint_ang = 0.0
    for i in range(3):
        r_joint = 0.22 - i*0.03
        col_j = link_colors[i]
        ax_robot.add_patch(plt.Circle((xs[i], ys[i]), r_joint+0.06,
                           color='white', zorder=4))
        ax_robot.add_patch(plt.Circle((xs[i], ys[i]), r_joint,
                           color=col_j, zorder=5))
        ax_robot.add_patch(plt.Circle((xs[i], ys[i]), r_joint*0.45,
                           color='white', zorder=6))

        # Angle label on slide 1
        if hl == 'joints':
            theta_deg = np.degrees(q[i])
            if hl == 'joints':
                prev_ang = joint_ang
                joint_ang += q[i]
                draw_angle_arc(ax_robot, xs[i], ys[i], 0.48,
                               np.degrees(prev_ang), np.degrees(q[i]),
                               col_j, lw=2.0)
                mid_ang = prev_ang + q[i]/2
                rx, ry = 0.72*np.cos(mid_ang), 0.72*np.sin(mid_ang)
                ax_robot.text(xs[i]+rx, ys[i]+ry,
                              f'θ{i+1}={theta_deg:.0f}°',
                              fontsize=8.5, color=col_j,
                              fontweight='bold', ha='center', va='center', zorder=8,
                              bbox=dict(fc='white', ec=col_j, boxstyle='round,pad=0.2',
                                        lw=0.8, alpha=0.9))

    # ── End-effector ─────────────────────────────────────────────
    ee_size = 0.22
    ax_robot.add_patch(plt.Circle((xs[-1], ys[-1]), ee_size+0.06,
                       color='white', zorder=7))
    ax_robot.add_patch(plt.Circle((xs[-1], ys[-1]), ee_size,
                       color=P['ee'], zorder=8))
    ax_robot.add_patch(plt.Circle((xs[-1], ys[-1]), ee_size*0.45,
                       color='white', zorder=9))

    # EE label (slide 2+)
    if hl in ('ee', 'columns', 'twist', 'singular'):
        ax_robot.text(xs[-1]+0.28, ys[-1]+0.28,
                      f'p = ({xs[-1]:.2f}, {ys[-1]:.2f})',
                      fontsize=8.5, color=P['ee'], fontweight='bold',
                      zorder=10,
                      bbox=dict(fc='white', ec=P['ee'],
                                boxstyle='round,pad=0.25', lw=1.0))

    # Base
    ax_robot.add_patch(plt.Polygon(
        [[-.30, 0.0],[.30, 0.0],[.20,-.30],[-.20,-.30]],
        closed=True, fc=P['dark'], ec='none', zorder=3))
    for xi in np.linspace(-.28, .28, 7):
        ax_robot.plot([xi, xi-0.1], [-.30, -.46],
                      color=P['dark'], lw=1.5, zorder=2)

    # Axes labels
    ax_robot.set_xlabel('x  [m]', fontsize=9, color=P['muted'])
    ax_robot.set_ylabel('y  [m]', fontsize=9, color=P['muted'])
    ax_robot.tick_params(labelsize=8, colors=P['muted'])
    for spine in ax_robot.spines.values():
        spine.set_edgecolor(P['light'])

    # ── Concept panel ─────────────────────────────────────────────
    ax_concept.clear(); ax_concept.set_axis_off()
    ax_concept.set_xlim(0, 10); ax_concept.set_ylim(0, 10)

    ax_concept.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                         boxstyle='round,pad=0.2', fc=P['panel'],
                         ec=P['light'], lw=1.0))
    ax_concept.add_patch(FancyBboxPatch((0.2, 8.8), 9.6, 1.0,
                         boxstyle='round,pad=0.2', fc=P['dark'],
                         ec='none'))
    ax_concept.text(5.0, 9.32, 'Concept', ha='center', va='center',
                    fontsize=11, color='white', fontweight='bold')

    # Concept body
    ax_concept.text(0.7, 8.35, slide['concept'],
                    ha='left', va='top', fontsize=10.5,
                    color=P['dark'], linespacing=1.6,
                    fontfamily='monospace')

    # ── Live q display ────────────────────────────────────────────
    ax_concept.add_patch(FancyBboxPatch((0.4, 0.4), 9.2, 1.85,
                         boxstyle='round,pad=0.15', fc=P['blueLt'],
                         ec=P['blue'], lw=1.2))
    ax_concept.text(5.0, 2.05, 'Current configuration  q', ha='center',
                    fontsize=9.5, color=P['blue'], fontweight='bold')
    ax_concept.text(5.0, 1.4,
                    f'θ₁ = {np.degrees(q[0]):+6.1f}°    '
                    f'θ₂ = {np.degrees(q[1]):+6.1f}°    '
                    f'θ₃ = {np.degrees(q[2]):+6.1f}°',
                    ha='center', fontsize=11, color=P['dark'],
                    fontfamily='monospace', fontweight='bold')
    ax_concept.text(5.0, 0.72,
                    f'p_EE = ({xs[-1]:+.3f},  {ys[-1]:+.3f})  m',
                    ha='center', fontsize=10.5, color=P['teal'],
                    fontfamily='monospace')

    # ── Jacobian matrix panel ─────────────────────────────────────
    qdot = np.ones(3) if hl == 'twist' else None
    draw_Jmat(ax_Jmat, Jv, qdot)

    fig.canvas.draw_idle()

# ── Button callbacks ─────────────────────────────────────────────
def next_slide(event):
    current_slide[0] = (current_slide[0] + 1) % len(SLIDES)
    draw()

def prev_slide(event):
    current_slide[0] = (current_slide[0] - 1) % len(SLIDES)
    draw()

def reset(event):
    sl1.set_val(np.pi/4)
    sl2.set_val(np.pi/4)
    sl3.set_val(-np.pi/6)

btn_next.on_clicked(next_slide)
btn_prev.on_clicked(prev_slide)
btn_reset.on_clicked(reset)
sl1.on_changed(draw)
sl2.on_changed(draw)
sl3.on_changed(draw)

# ── Footer labels ────────────────────────────────────────────────
fig.text(0.06, 0.155, 'Joint angles  (drag to explore):', fontsize=9.5,
         color=P['muted'], va='center')
fig.text(0.38, 0.01, '← Prev / Next →  to cycle through lecture slides',
         fontsize=9, color=P['muted'], ha='left', va='bottom')

# ── Initial draw ─────────────────────────────────────────────────
draw()
plt.show()
