"""Matrix Lab backend — step-by-step decomposition calculator.

For 2×2 matrices decompositions are computed analytically with full
step-by-step math (as in the theory tab).  For larger matrices the
black-box solvers from NumPy (LAPACK) are used because a manual
iterative calculation would produce thousands of steps.

Usage:
    python3 server.py
    # → serves http://localhost:5000
    # → POST /api/decompose  { matrix: [[a,b],[c,d]], iters: 10 }
"""

import math

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════════════════════
#  formatting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _lM(M):
    rows = []
    for i in range(M.shape[0]):
        rows.append(" & ".join(_lS(M[i, j]) for j in range(M.shape[1])))
    return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"

def _lV(v):
    return "\\begin{bmatrix} " + " \\\\ ".join(_lS(x) for x in v) + " \\end{bmatrix}"

def _lVs(v):
    return "\\begin{bmatrix} " + " & ".join(_lS(x) for x in v) + " \\end{bmatrix}"

def _lS(x):
    s = round(float(x), 4)
    if s == int(s):
        return str(int(s))
    return f"{s:.4f}".rstrip("0").rstrip(".")

def _step(tp, content):
    return {"type": tp, "content": content}

def _txt(s):   return _step("text", s)
def _math(s):  return _step("math", s)
def _sub(s):   return _step("sub", s)

def _norm(v):
    return math.sqrt(np.dot(v, v))


# ═══════════════════════════════════════════════════════════════════════════
#  detailed 2×2 helpers
# ═══════════════════════════════════════════════════════════════════════════

def _detail_mul_2x2(A, B, label, steps):
    """Matrix mult for 2×2 — shows formula matrix then intermediate then result."""
    steps.append(_math(f"{label} = {_lM(A)} \\cdot {_lM(B)}"))
    rows_f, rows_i = [], []
    for i in range(2):
        cells_f, cells_i = [], []
        for j in range(2):
            terms = [f"{_lS(A[i,k])}\\cdot{_lS(B[k,j])}" for k in range(2)]
            vals  = [ _lS(A[i,k]*B[k,j])               for k in range(2)]
            cells_f.append(terms[0] + "+" + terms[1])
            cells_i.append(vals[0]  + "+" + vals[1])
        rows_f.append(" & ".join(cells_f))
        rows_i.append(" & ".join(cells_i))
    steps.append(_math(f"{label} = \\begin{{bmatrix}} {rows_f[0]} \\\\ {rows_f[1]} \\end{{bmatrix}}"))
    steps.append(_math(f"{label} = \\begin{{bmatrix}} {rows_i[0]} \\\\ {rows_i[1]} \\end{{bmatrix}} = {_lM(A @ B)}"))


def _eigvec_2x2_steps(M, lam, idx, steps, mat_label="C"):
    """Analytical eigenvector of 2×2 with steps matching theory tab."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    mat = M - lam * np.eye(2)
    steps.append(_math(f"({mat_label} - {_lS(lam)}I) = {_lM(mat)}"))
    ratio = None
    if abs(b) > 1e-12:
        ratio = -(a - lam) / b
        steps.append(_math(f"{_lS(mat[0,0])}x + {_lS(mat[0,1])}y = 0"))
        steps.append(_txt(f"Из первой строки: {_lS(mat[0,1])}y = {_lS(-mat[0,0])}x, "
                          f"значит y = {_lS(ratio)}x."))
        v_raw = np.array([1.0, float(ratio)])
    elif abs(c) > 1e-12:
        ratio = -(d - lam) / c
        steps.append(_math(f"{_lS(mat[1,0])}x + {_lS(mat[1,1])}y = 0"))
        steps.append(_txt(f"Из второй строки: {_lS(mat[1,0])}x = {_lS(-mat[1,1])}y, "
                          f"значит x = {_lS(ratio)}y."))
        v_raw = np.array([float(ratio), 1.0])
    else:
        v_raw = np.array([1.0, 0.0])
    steps.append(_txt("Нормируем (сумма квадратов компонент должна равняться единице):"))
    if ratio is not None:
        r2 = 1.0 + float(ratio)**2
        ratio_sq = float(ratio)**2
        steps.append(_math(f"x^2 + y^2 = 1"))
        steps.append(_math(f"x^2 + ({_lS(ratio)} \\cdot x)^2 = 1"))
        steps.append(_math(f"x^2 + {_lS(ratio_sq)} x^2 = 1"))
        steps.append(_math(f"{_lS(r2)} x^2 = 1"))
        x_sq = 1.0 / r2
        x_val = math.sqrt(x_sq)
        y_val = float(ratio) * x_val
        steps.append(_math(f"x^2 = \\frac{{1}}{{{_lS(r2)}}} \\approx {_lS(x_sq)}, "
                           f"\\quad x \\approx {_lS(x_val)}"))
        steps.append(_math(f"y = {_lS(ratio)} \\cdot {_lS(x_val)} \\approx {_lS(y_val)}"))
        v = np.array([x_val, y_val])
    else:
        v = np.array([1.0, 0.0])
    if v[0] < 0:
        v = -v
    steps.append(_math(f"v_{idx} = {_lV(v)}"))
    return v


def _detail_Av_2x2(A, v, idx, steps):
    """A·v for 2×2 — shows formula, then intermediate, then result (matches theory tab)."""
    steps.append(_txt(f"Сначала умножаем A на v{idx}:"))
    rows_f, rows_i = [], []
    for i in range(2):
        t0 = f"{_lS(A[i,0])}\\cdot{_lS(v[0])}"
        t1 = f"{_lS(A[i,1])}\\cdot{_lS(v[1])}"
        v0 = _lS(A[i,0]*v[0])
        v1 = _lS(A[i,1]*v[1])
        total = _lS(A[i,0]*v[0] + A[i,1]*v[1])
        rows_f.append(t0 + "+" + t1)
        rows_i.append(v0 + "+" + v1)
    Av = A @ v
    steps.append(_math(f"A v_{idx} = \\begin{{bmatrix}} {rows_f[0]} \\\\ {rows_f[1]} \\end{{bmatrix}}"))
    steps.append(_math(f"A v_{idx} = \\begin{{bmatrix}} {rows_i[0]} \\\\ {rows_i[1]} \\end{{bmatrix}} = {_lV(Av)}"))
    return Av


# ═══════════════════════════════════════════════════════════════════════════
#  SVD
# ═══════════════════════════════════════════════════════════════════════════

def _eigvec_2x2(M, lam, steps, idx):
    """Analytical eigenvector of 2×2 matrix M for eigenvalue lam, with steps."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    mat = M - lam * np.eye(2)
    steps.append(_math(f"(C - {_lS(lam)}I) = {_lM(mat)}"))

    if abs(b) > 1e-12:
        ratio = -(a - lam) / b
        steps.append(_txt(f"Из первой строки: {_lS(mat[0,0])}·x + {_lS(mat[0,1])}·y = 0"))
        steps.append(_txt(f"  → {_lS(mat[0,1])}·y = {_lS(-mat[0,0])}·x"))
        steps.append(_txt(f"  → y = {_lS(ratio)}·x"))
        v_raw = np.array([1.0, float(ratio)])
    elif abs(c) > 1e-12:
        ratio = -(d - lam) / c
        steps.append(_txt(f"Из второй строки: {_lS(mat[1,0])}·x + {_lS(mat[1,1])}·y = 0"))
        steps.append(_txt(f"  → {_lS(mat[1,0])}·x = {_lS(-mat[1,1])}·y"))
        steps.append(_txt(f"  → x = {_lS(ratio)}·y"))
        v_raw = np.array([float(ratio), 1.0])
    else:
        v_raw = np.array([1.0, 0.0])

    v = v_raw / _norm(v_raw)
    if v[0] < 0:
        v = -v

    steps.append(_txt("Нормируем (сумма квадратов = 1):"))
    r2 = float(v_raw[0]**2 + v_raw[1]**2)
    steps.append(_math(f"x^2 + y^2 = {_lS(r2)} \\quad \\Rightarrow \\quad "
                       f"x = {_lS(v[0])}, \\; y = {_lS(v[1])}"))
    steps.append(_math(f"v_{idx} = {_lV(v)}"))
    return v


def _detail_Av(A, v, idx, steps):
    """Matrix-vector multiply A·v with step-by-step arithmetic."""
    steps.append(_txt(f"Сначала умножаем A на v{idx}:"))
    for i in range(2):
        t0 = f"{_lS(A[i,0])}\\cdot{_lS(v[0])}"
        t1 = f"{_lS(A[i,1])}\\cdot{_lS(v[1])}"
        v0 = A[i,0] * v[0]
        v1 = A[i,1] * v[1]
        total = v0 + v1
        steps.append(_txt(f"  ({i+1}): {t0} + {t1} = {_lS(v0)} + {_lS(v1)} = {_lS(total)}"))
    Av = A @ v
    steps.append(_math(f"A v_{idx} = {_lV(Av)}"))
    return Av


_EXPLANATION = (
    "Для матриц больше 2×2 используется библиотека NumPy (LAPACK), "
    "так как ручной пошаговый расчёт дал бы тысячи шагов."
)


def svd_steps(A):
    steps = []
    steps.append(_sub("SVD — сингулярное разложение"))
    m, n = A.shape

    if m == 2 and n == 2:
        # ── analytical 2×2 (matches theory tab exactly) ──
        At = A.T
        steps.append(_txt("Шаг 1: транспонируем A:"))
        steps.append(_math(f"A^T = {_lM(At)}"))

        steps.append(_txt("Шаг 2: вычисляем AᵀA:"))
        _detail_mul_2x2(At, A, "A^T A", steps)

        AtA = At @ A
        a, b, c, d = AtA[0, 0], AtA[0, 1], AtA[1, 0], AtA[1, 1]

        steps.append(_txt("Шаг 3: ищем собственные значения AᵀA:"))
        steps.append(_txt("Решаем характеристическое уравнение det(AᵀA − λI) = 0:"))
        steps.append(_math(
            f"\\det \\begin{{bmatrix}} {_lS(a)}-\\lambda & {_lS(b)} \\\\"
            f" {_lS(c)} & {_lS(d)}-\\lambda \\end{{bmatrix}} = 0"
        ))
        steps.append(_math(f"({_lS(a)}-\\lambda)({_lS(d)}-\\lambda) - {_lS(b)}\\cdot{_lS(c)} = 0"))
        ad = a * d
        bc = b * c
        steps.append(_txt("Раскрываем скобки:"))
        steps.append(_math(
            f"{_lS(ad)} - {_lS(a)}\\lambda - {_lS(d)}\\lambda"
            f" + \\lambda^2 - {_lS(bc)} = 0"
        ))
        cross1 = a + d
        const = ad - bc
        steps.append(_math(f"\\lambda^2 - {_lS(cross1)}\\lambda + {_lS(const)} = 0"))
        disc = cross1 ** 2 - 4 * const
        sqrt_disc = math.sqrt(max(disc, 0))
        steps.append(_txt("Дискриминант:"))
        steps.append(_math(
            f"D = ({_lS(-cross1)})^2 - 4 \\cdot 1 \\cdot {_lS(const)}"
            f" = {_lS(disc)}"
        ))
        lam1 = max((cross1 + sqrt_disc) / 2, (cross1 - sqrt_disc) / 2)
        lam2 = min((cross1 + sqrt_disc) / 2, (cross1 - sqrt_disc) / 2)
        steps.append(_txt("Корни:"))
        steps.append(_math(
            f"\\lambda_1 = \\frac{{{_lS(cross1)} + \\sqrt{{{_lS(disc)}}}}}{{2}}"
            f" \\approx {_lS(lam1)}"
        ))
        steps.append(_math(
            f"\\lambda_2 = \\frac{{{_lS(cross1)} - \\sqrt{{{_lS(disc)}}}}}{{2}}"
            f" \\approx {_lS(lam2)}"
        ))

        sig1 = math.sqrt(max(lam1, 0))
        sig2 = math.sqrt(max(lam2, 0))
        steps.append(_txt("Шаг 4: сингулярные числа σᵢ = √λᵢ:"))
        steps.append(_math(f"\\sigma_1 = \\sqrt{{{_lS(lam1)}}} \\approx {_lS(sig1)}"))
        steps.append(_math(f"\\sigma_2 = \\sqrt{{{_lS(lam2)}}} \\approx {_lS(sig2)}"))

        for idx, lam, sig in [(1, lam1, sig1), (2, lam2, sig2)]:
            if sig < 1e-12:
                steps.append(_txt(f"σ{idx} ≈ 0 → пропускаем."))
                continue
            steps.append(_txt(f"Шаг 5.{idx}: находим правый сингулярный вектор v{idx}:"))
            steps.append(_txt(f"Решаем (AᵀA − λ{idx}I)·v{idx} = 0:"))
            v = _eigvec_2x2_steps(AtA, lam, idx, steps)
            steps.append(_txt(f"Шаг 6.{idx}: находим левый сингулярный вектор u{idx}:"))
            steps.append(_math(f"u_{idx} = \\dfrac{{A v_{idx}}}{{\\sigma_{idx}}}"))
            Av = _detail_Av_2x2(A, v, idx, steps)
            u = Av / sig
            steps.append(_txt(f"Делим на σ{idx}:"))
            for i in range(2):
                steps.append(_txt(f"  u{i+1} = {_lS(Av[i])} / {_lS(sig)} = {_lS(u[i])}"))
            steps.append(_math(f"u_{idx} = {_lV(u)}"))

        steps.append(_txt("Шаг 7: строим усечённую матрицу ранга 1 A₁ = σ₁·u₁·v₁ᵀ:"))
        v1 = _eigvec_2x2_steps(AtA, lam1, 1, [])  # compute silently
        u1 = (A @ v1) / sig1
        uv = np.outer(u1, v1)
        steps.append(_math(f"u_1 v_1^T = {_lV(u1)} \\cdot {_lVs(v1)} = {_lM(uv)}"))
        A1 = sig1 * uv
        steps.append(_math(f"A_1 = {_lS(sig1)} \\cdot {_lM(uv)} = {_lM(A1)}"))
        steps.append(_math(f"A = {_lM(A)}, \\quad A_1 = {_lM(A1)}"))

    else:
        steps.append(_txt(_EXPLANATION))
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        steps.append(_math(f"U = {_lM(U)}"))
        steps.append(_math(f"\\Sigma = {_lVs(S)}"))
        steps.append(_math(f"V^T = {_lM(Vt)}"))
        Ahat = U @ np.diag(S) @ Vt
        steps.append(_math(f"\\tilde A = U \\Sigma V^T = {_lM(Ahat)}"))

    return {"available": True, "steps": steps}


# ═══════════════════════════════════════════════════════════════════════════
#  PCA
# ═══════════════════════════════════════════════════════════════════════════

def pca_steps(A):
    steps = []
    steps.append(_sub("PCA — метод главных компонент"))
    m, n = A.shape

    mean = A.mean(axis=0)
    steps.append(_txt("Шаг 1: средние по столбцам:"))
    for j in range(n):
        terms = " + ".join(_lS(A[i, j]) for i in range(m))
        steps.append(_math(f"\\mu_{j+1} = \\frac{{{terms}}}{{{m}}} = {_lS(mean[j])}"))

    X = A - mean
    steps.append(_txt("Шаг 2: центрирование X = A − μ:"))
    steps.append(_math(f"X = {_lM(X)}"))

    if m == 2 and n == 2:
        steps.append(_txt("Шаг 3: строим ковариационную матрицу C = Xᵀ·X:"))
        Xt = X.T
        steps.append(_math(f"X^T = {_lM(Xt)}"))
        _detail_mul_2x2(Xt, X, "C", steps)

        C = X.T @ X
        a, b, c, d = C[0, 0], C[0, 1], C[1, 0], C[1, 1]

        steps.append(_txt("Шаг 4: собственные значения:"))
        steps.append(_txt("Решаем det(C − λI) = 0:"))
        steps.append(_math(
            f"\\det \\begin{{bmatrix}} {_lS(a)}-\\lambda & {_lS(b)} \\\\"
            f" {_lS(c)} & {_lS(d)}-\\lambda \\end{{bmatrix}} = 0"
        ))
        steps.append(_math(f"({_lS(a)}-\\lambda)({_lS(d)}-\\lambda) - {_lS(b)}\\cdot{_lS(c)} = 0"))
        ad = a * d
        bc = b * c
        steps.append(_txt("Раскрываем скобки:"))
        steps.append(_math(
            f"{_lS(ad)} - {_lS(a)}\\lambda - {_lS(d)}\\lambda"
            f" + \\lambda^2 - {_lS(bc)} = 0"
        ))
        cross1 = a + d
        const = ad - bc
        steps.append(_math(f"\\lambda^2 - {_lS(cross1)}\\lambda + {_lS(const)} = 0"))
        disc = cross1 ** 2 - 4 * const
        sqrt_disc = math.sqrt(max(disc, 0))
        steps.append(_txt("Дискриминант:"))
        steps.append(_math(f"D = ({_lS(-cross1)})^2 - 4 \\cdot 1 \\cdot {_lS(const)} = {_lS(disc)}"))
        lam1 = max((cross1 + sqrt_disc) / 2, (cross1 - sqrt_disc) / 2)
        lam2 = min((cross1 + sqrt_disc) / 2, (cross1 - sqrt_disc) / 2)
        steps.append(_txt("Корни:"))
        steps.append(_math(f"\\lambda_1 = {_lS(lam1)}, \\quad \\lambda_2 = {_lS(lam2)}"))

        steps.append(_txt("Шаг 5: первый собственный вектор v₁:"))
        v = _eigvec_2x2_steps(C, lam1, 1, steps)

        steps.append(_txt("Шаг 6: проекция Z = X·v₁:"))
        rows_f, rows_i = [], []
        for i in range(m):
            t0 = f"{_lS(X[i,0])}\\cdot{_lS(v[0])}"
            t1 = f"{_lS(X[i,1])}\\cdot{_lS(v[1])}"
            v0 = _lS(X[i,0]*v[0])
            v1 = _lS(X[i,1]*v[1])
            rows_f.append(t0 + "+" + t1)
            rows_i.append(v0 + "+" + v1)
        steps.append(_math(f"Z = \\begin{{bmatrix}} {rows_f[0]} \\\\ {rows_f[1]} \\end{{bmatrix}}"))
        Z = X @ v
        steps.append(_math(f"Z = \\begin{{bmatrix}} {rows_i[0]} \\\\ {rows_i[1]} \\end{{bmatrix}} = {_lV(Z)}"))

        total_var = float(np.trace(C))
        exp_var = float(lam1 / total_var * 100) if total_var > 0 else 100
        steps.append(_txt(f"Шаг 7: объяснённая дисперсия = {exp_var:.1f}%"))
        steps.append(_math(f"\\text{{Var}}_Z = {_lS(lam1)}, \\quad "
                           f"\\text{{Var}}_{{\\text{{tot}}}} = {_lS(total_var)}"))
        steps.append(_txt(f"λ₁ = Var_Z — первая компонента сохраняет "
                          f"{exp_var:.1f}% дисперсии."))

        steps.append(_txt("Проверка восстановления:"))
        Z_outer = np.outer(Z, v)
        steps.append(_math(f"Z \\cdot v_1^T = {_lV(Z)} \\cdot {_lVs(v)} = {_lM(Z_outer)}"))
        rows = []
        for i in range(2):
            cells = []
            for j in range(2):
                val = _lS(Z_outer[i, j] + mean[j])
                c = f"{_lS(Z_outer[i,j])}+{_lS(mean[j])}={val}"
                cells.append(c)
            rows.append(" & ".join(cells))
        steps.append(_math(
            f"A_{{\\text{{восст}}}} = "
            f"\\begin{{bmatrix}} {rows[0]} \\\\ {rows[1]} \\end{{bmatrix}}"
        ))
        A_recon = Z_outer + mean
        steps.append(_math(f"A_{{\\text{{восст}}}} = {_lM(A_recon)}"))
        if abs(lam2) < 1e-10:
            steps.append(_txt(
                "Восстановили идеально, потому что данные изначально "
                "лежали на прямой (λ₂ ≈ 0)."
            ))

    else:
        C = X.T @ X
        steps.append(_txt(f"Матрица {m}×{n}. Ковариация C = Xᵀ·X:"))
        steps.append(_math(f"C = {_lM(C)}"))
        steps.append(_txt(_EXPLANATION))
        eigvals, eigvecs = np.linalg.eigvalsh(C), np.linalg.eig(C)[1]
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        steps.append(_math(f"\\lambda_1 = {_lS(eigvals[0])}, \\; "
                           f"\\lambda_2 = {_lS(eigvals[1])}, \\; \\dots"))
        v1 = eigvecs[:, 0]
        steps.append(_math(f"v_1 = {_lV(v1)}"))

        Z = X @ v1
        steps.append(_txt("Проекция Z = X·v₁:"))
        steps.append(_math(f"Z = {_lV(Z)}"))

        total_var = float(np.trace(C))
        exp_var = float(eigvals[0] / total_var * 100) if total_var > 0 else 100
        steps.append(_txt(f"Объяснённая дисперсия первой компоненты: {exp_var:.1f}%"))
        steps.append(_math(f"\\text{{Var}}_Z = {_lS(eigvals[0])}, \\quad "
                           f"\\text{{Var}}_{{\\text{{tot}}}} = {_lS(total_var)}"))

        steps.append(_txt("Проверка восстановления (ранг 1):"))
        Z_outer = np.outer(Z, v1)
        steps.append(_math(f"Z \\cdot v_1^T = {_lV(Z)} \\cdot {_lVs(v1)} = {_lM(Z_outer)}"))

        rows = []
        for i in range(m):
            cells = []
            for j in range(n):
                val = _lS(Z_outer[i, j] + mean[j])
                c = f"{_lS(Z_outer[i,j])}+{_lS(mean[j])}={val}"
                cells.append(c)
            rows.append(" & ".join(cells))
        steps.append(_math(
            f"A_{{\\text{{восст}}}} = "
            f"\\begin{{bmatrix}} {' \\\\ '.join(rows)} \\end{{bmatrix}}"
        ))

        A_recon = Z_outer + mean
        steps.append(_math(f"A_{{\\text{{восст}}}} = {_lM(A_recon)}"))

        diff = A - A_recon
        steps.append(_math(f"A - A_{{\\text{{восст}}}} = {_lM(diff)}"))

    return {"available": True, "steps": steps}


# ═══════════════════════════════════════════════════════════════════════════
#  NMF
# ═══════════════════════════════════════════════════════════════════════════

def nmf_steps(A, iters=10):
    steps = []
    steps.append(_sub("NMF — неотрицательное матричное разложение"))
    m, n = A.shape

    shift = max(-A.min(), 0)
    X = A.copy()
    if shift > 0:
        X = A + shift
        steps.append(_txt(f"Сдвиг на {_lS(shift)} для неотрицательности:"))
        steps.append(_math(f"X = A + {_lS(shift)} = {_lM(X)}"))
    else:
        steps.append(_math(f"A = {_lM(A)}"))

    k = 1
    np.random.seed(42)
    W = np.random.rand(m, k) + 0.5
    H = np.random.rand(k, n) + 0.5
    steps.append(_txt("Шаг 1: инициализация W (m×1) и H (1×n):"))
    steps.append(_math(f"W = {_lM(W)}, \\qquad H = {_lM(H)}"))
    steps.append(_math(f"WH = {_lM(W)} \\cdot {_lM(H)} = {_lM(W @ H)}"))

    eps = 1e-12
    for it in range(iters):
        steps.append(_txt(f"——— Итерация {it+1} ———"))
        Wt = W.T
        WtX = Wt @ X
        WtW = float((Wt @ W).item())
        WtWH = WtW * H
        H_new = H * WtX / (WtWH + eps)
        steps.append(_math(f"H_{{\\text{{нов}}}} = H \\odot "
                           f"\\frac{{W^T X}}{{W^T W \\cdot H}}"
                           f" = {_lM(H_new)}"))
        H = H_new

        Ht = H.T
        XHt = X @ Ht
        HHt = float((H @ Ht).item())
        WHHt = W * HHt
        W_new = W * XHt / (WHHt + eps)
        steps.append(_math(f"W_{{\\text{{нов}}}} = W \\odot "
                           f"\\frac{{X H^T}}{{W \\cdot H H^T}}"
                           f" = {_lM(W_new)}"))
        W = W_new

    Ahat = W @ H - shift
    steps.append(_txt("Результат:"))
    steps.append(_math(f"\\tilde A = WH - {_lS(shift)} = {_lM(Ahat)}"))
    steps.append(_math(f"A = {_lM(A)}, \\quad \\tilde A = {_lM(Ahat)}"))

    return {"available": True, "steps": steps}


# ═══════════════════════════════════════════════════════════════════════════
#  CUR
# ═══════════════════════════════════════════════════════════════════════════

def _norm_sq_cols(A):
    return np.array([sum(A[i, j] ** 2 for i in range(A.shape[0])) for j in range(A.shape[1])])

def _norm_sq_rows(A):
    return np.array([sum(A[i, j] ** 2 for j in range(A.shape[1])) for i in range(A.shape[0])])

def cur_steps(A):
    steps = []
    steps.append(_sub("CUR — разложение по строкам и столбцам"))
    m, n = A.shape
    r = 1

    col_norms = _norm_sq_cols(A)
    row_norms = _norm_sq_rows(A)

    steps.append(_txt("Квадраты длин столбцов:"))
    for j in range(n):
        terms = " + ".join(f"{_lS(A[i,j])}²" for i in range(m))
        steps.append(_math(f"\\|\\text{{col}}_{j+1}\\|^2 = {terms} = {_lS(col_norms[j])}"))

    steps.append(_txt("Квадраты длин строк:"))
    for i in range(m):
        terms = " + ".join(f"{_lS(A[i,j])}²" for j in range(n))
        steps.append(_math(f"\\|\\text{{row}}_{i+1}\\|^2 = {terms} = {_lS(row_norms[i])}"))

    top_cols = np.argsort(col_norms)[-r:][::-1]
    top_rows = np.argsort(row_norms)[-r:][::-1]

    Cmat = A[:, top_cols]
    Rmat = A[top_rows, :]
    Wcore = A[np.ix_(top_rows, top_cols)]

    steps.append(_math(f"C = {_lM(Cmat)}"))
    steps.append(_math(f"R = {_lM(Rmat)}"))
    steps.append(_math(f"W = {_lM(Wcore)}"))

    steps.append(_txt("Псевдообратная C⁺ = (CᵀC)⁻¹·Cᵀ:"))
    Ct = Cmat.T
    CtC = float((Ct @ Cmat).item())
    steps.append(_math(f"C^T C = {_lS(CtC)} \\quad \\Rightarrow \\quad "
                       f"(C^T C)^{{-1}} = \\frac{{1}}{{{_lS(CtC)}}} = {_lS(1/CtC)}"))
    Cp = Ct / CtC
    steps.append(_math(f"C^+ = {_lM(Cp)}"))

    steps.append(_txt("Псевдообратная R⁺ = Rᵀ·(R·Rᵀ)⁻¹:"))
    Rt = Rmat.T
    RRt = float((Rmat @ Rt).item())
    steps.append(_math(f"R R^T = {_lS(RRt)} \\quad \\Rightarrow \\quad "
                       f"(R R^T)^{{-1}} = \\frac{{1}}{{{_lS(RRt)}}} = {_lS(1/RRt)}"))
    Rp = Rt / RRt
    steps.append(_math(f"R^+ = {_lM(Rp)}"))

    steps.append(_txt("Связующая U = C⁺·A·R⁺:"))
    A_Rp = A @ Rp
    steps.append(_math(f"A \\cdot R^+ = {_lM(A)} \\cdot {_lM(Rp)} = {_lM(A_Rp)}"))
    Umat = Cp @ A_Rp
    steps.append(_math(f"U = C^+ \\cdot (A \\cdot R^+) = {_lM(Umat)}"))

    Ahat = Cmat @ Umat @ Rmat
    steps.append(_math(f"\\tilde A = C \\cdot U \\cdot R = {_lM(Ahat)}"))
    steps.append(_math(f"A = {_lM(A)}, \\quad \\tilde A = {_lM(Ahat)}"))

    return {"available": True, "steps": steps}


# ═══════════════════════════════════════════════════════════════════════════
#  ALS
# ═══════════════════════════════════════════════════════════════════════════

def als_steps(A, iters=5):
    steps = []
    steps.append(_sub("ALS — попеременные наименьшие квадраты"))
    m, n = A.shape
    r = 1
    np.random.seed(42)
    X = np.random.rand(m, r) + 0.5
    Y = np.random.rand(n, r) + 0.5

    steps.append(_txt("Шаг 1: начальные X и Y:"))
    steps.append(_math(f"X = {_lM(X)}, \\qquad Y = {_lM(Y)}"))
    steps.append(_math(f"X Y^T = {_lM(X @ Y.T)}"))

    reg = 1e-2

    for it in range(iters):
        steps.append(_txt(f"——— Итерация {it+1} ———"))

        Yt = Y.T
        YtY = float((Yt @ Y).item())
        YtY_reg = float(YtY + reg)
        steps.append(_math(f"Y^T Y + \\lambda = {_lS(YtY)} + {_lS(reg)} = {_lS(YtY_reg)}"))

        for i in range(m):
            AiY = float((A[i:i+1, :] @ Y).item())
            Xi = AiY / YtY_reg
            X[i, 0] = Xi
            terms = " + ".join(f"{_lS(A[i,j])}·{_lS(Y[j,0])}" for j in range(n))
            steps.append(_math(f"x_{i+1} = \\frac{{{terms}}}{{{_lS(YtY_reg)}}} = "
                               f"\\frac{{{_lS(AiY)}}}{{{_lS(YtY_reg)}}} = {_lS(Xi)}"))

        steps.append(_math(f"X_{{\\text{{нов}}}} = {_lM(X)}"))

        Xt = X.T
        XtX = float((Xt @ X).item())
        XtX_reg = float(XtX + reg)
        steps.append(_math(f"X^T X + \\lambda = {_lS(XtX)} + {_lS(reg)} = {_lS(XtX_reg)}"))

        for j in range(n):
            AtjX = float((A[:, j:j+1].T @ X).item())
            Yj = AtjX / XtX_reg
            Y[j, 0] = Yj
            terms = " + ".join(f"{_lS(A[i,j])}·{_lS(X[i,0])}" for i in range(m))
            col_label = f"A_{{\\text{{:}}{j+1}}}^T"
            steps.append(_math(f"y_{j+1} = \\frac{{{col_label} \\cdot X}}{{{_lS(XtX_reg)}}}"
                               f" = \\frac{{{_lS(AtjX)}}}{{{_lS(XtX_reg)}}} = {_lS(Yj)}"))

        steps.append(_math(f"Y_{{\\text{{нов}}}} = {_lM(Y)}"))

    Ahat = X @ Y.T
    steps.append(_math(f"X Y^T = {_lM(X)} \\cdot {_lM(Y.T)} = {_lM(Ahat)}"))
    steps.append(_math(f"A = {_lM(A)}, \\quad \\tilde A = {_lM(Ahat)}"))

    return {"available": True, "steps": steps}


# ═══════════════════════════════════════════════════════════════════════════
#  Flask
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/decompose", methods=["POST"])
def decompose():
    try:
        data = request.get_json()
        raw = data["matrix"]
        A = np.array(raw, dtype=float)
        m, n = A.shape
        if m < 2 or n < 2:
            return jsonify({"status": "error", "message": "Матрица должна быть 2×2 или больше"}), 400
        iters = min(data.get("iters", 10), 20)
        result = {
            "status": "ok",
            "shape": [m, n],
            "matrix_latex": _lM(A),
            "decompositions": {
                "svd":  svd_steps(A),
                "pca":  pca_steps(A),
                "nmf":  nmf_steps(A, iters),
                "cur":  cur_steps(A),
                "als":  als_steps(A, min(iters, 5)),
            },
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Matrix Lab backend at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
