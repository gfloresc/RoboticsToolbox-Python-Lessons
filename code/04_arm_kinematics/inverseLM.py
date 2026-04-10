import numpy as np

# ─── DH matrix ────────────────────────────────────────────────
def dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])

# ─── Stanford Manipulator FK ──────────────────────────────────
# Joint variables: q = [theta1, theta2, d3, theta4, theta5, theta6]
# Fixed params: d2=0.154, d6=0.263
D2 = 0.154
D6 = 0.263

def fk_stanford(q):
    t1, t2, d3, t4, t5, t6 = q
    T = np.eye(4)
    T = T @ dh(t1,  0,   0, -np.pi/2)   # Link 1
    T = T @ dh(t2,  D2,  0, +np.pi/2)   # Link 2
    T = T @ dh(0,   d3,  0,  0)          # Link 3 (prismatic)
    T = T @ dh(t4,  0,   0, -np.pi/2)   # Link 4
    T = T @ dh(t5,  0,   0, +np.pi/2)   # Link 5
    T = T @ dh(t6,  D6,  0,  0)          # Link 6
    return T

# ─── Target pose H ────────────────────────────────────────────
H = np.array([
    [0, 1, 0, -0.154],
    [0, 0, 1,  0.763],
    [1, 0, 0,  0.000],
    [0, 0, 0,  1.000]
])

# ─── Error function: 12 residuals (rotation 9 + position 3) ──
def error(q):
    T = fk_stanford(q)
    dR = (H[:3, :3] - T[:3, :3]).flatten()   # 9 rotation errors
    dp = H[:3,  3] - T[:3,  3]               # 3 position errors
    return np.concatenate([dR, dp])

# ─── Numerical Jacobian ───────────────────────────────────────
def jacobian(q, eps=1e-7):
    e0 = error(q)
    J  = np.zeros((len(e0), len(q)))
    for i in range(len(q)):
        dq    = np.zeros(len(q))
        dq[i] = eps
        J[:, i] = (error(q + dq) - e0) / eps
    return J

# ─── Levenberg-Marquardt ──────────────────────────────────────
def levenberg_marquardt(q0, max_iter=500, tol=1e-10):
    q   = q0.copy()
    lam = 1e-3          # damping factor

    print(f"\n{'Iter':>5}  {'||error||':>12}  {'lambda':>12}")
    print("-" * 38)

    for k in range(max_iter):
        e = error(q)
        norm_e = np.linalg.norm(e)

        if k % 20 == 0:
            print(f"{k:>5}  {norm_e:>12.6e}  {lam:>12.6e}")

        if norm_e < tol:
            print(f"{k:>5}  {norm_e:>12.6e}  converged")
            break

        J  = jacobian(q)
        JtJ = J.T @ J
        Jte = J.T @ e

        # LM update: (JᵀJ + λI) Δq = Jᵀe
        delta = np.linalg.solve(JtJ + lam * np.eye(len(q)), Jte)
        q_new = q + delta

        # Accept or reject step
        if np.linalg.norm(error(q_new)) < norm_e:
            q   = q_new
            lam = max(lam / 10, 1e-12)   # reduce damping (more Newton-like)
        else:
            lam = min(lam * 10, 1e+12)   # increase damping (more gradient-like)

    return q, norm_e

# ─── Initial guess (slightly off from true solution) ──────────
q0 = np.array([
    np.pi/2 + 0.1,   # theta1
    np.pi/2 + 0.1,   # theta2
    0.5    + 0.05,   # d3
    np.pi/2 + 0.1,   # theta4
    0.0    + 0.05,   # theta5
    np.pi/2 + 0.1,   # theta6
])

# ─── Solve ────────────────────────────────────────────────────
print("=" * 50)
print("Stanford Manipulator — IK via Levenberg-Marquardt")
print("=" * 50)

q_sol, final_error = levenberg_marquardt(q0)

# ─── Results ──────────────────────────────────────────────────
print(f"\nFinal ||error|| = {final_error:.2e}")
print("\nSolution found:")
print(f"  theta1 = {q_sol[0]:.6f} rad  ({np.degrees(q_sol[0]):.3f} deg)  | expected pi/2 = {np.pi/2:.6f}")
print(f"  theta2 = {q_sol[1]:.6f} rad  ({np.degrees(q_sol[1]):.3f} deg)  | expected pi/2 = {np.pi/2:.6f}")
print(f"  d3     = {q_sol[2]:.6f} m                     | expected 0.5")
print(f"  theta4 = {q_sol[3]:.6f} rad  ({np.degrees(q_sol[3]):.3f} deg)  | expected pi/2 = {np.pi/2:.6f}")
print(f"  theta5 = {q_sol[4]:.6f} rad  ({np.degrees(q_sol[4]):.3f} deg)  | expected 0.0")
print(f"  theta6 = {q_sol[5]:.6f} rad  ({np.degrees(q_sol[5]):.3f} deg)  | expected pi/2 = {np.pi/2:.6f}")

# ─── Verification: FK with solution ───────────────────────────
T_sol = fk_stanford(q_sol)
print("\nFK(q_solution):")
print(np.round(T_sol, 4))
print("\nTarget H:")
print(H)
print(f"\nMax element-wise error: {np.max(np.abs(T_sol - H)):.2e}")

# ─── Verification: FK with exact known solution ───────────────
q_exact = np.array([np.pi/2, np.pi/2, 0.5, np.pi/2, 0.0, np.pi/2])
T_exact = fk_stanford(q_exact)
print("\nFK(q_exact) — book solution:")
print(np.round(T_exact, 4))
print(f"Max element-wise error: {np.max(np.abs(T_exact - H)):.2e}")