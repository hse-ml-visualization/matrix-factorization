const codeBlocks = {};

codeBlocks.svd = {
  title: "1. SVD — сингулярное разложение",
  badge: "усечение ранга k",
  desc: `Раскладывает матрицу <code>A</code> на три сомножителя:
    <code>A = U · Σ · V<sup>T</sup></code>. Усечение до ранга k оставляет
    только k наибольших сингулярных чисел — это даёт наилучшее приближение.
    <strong>Где используется:</strong> сжатие изображений, шумоподавление, рекомендации.`,
  example: "Фотография 1000×1000 пикселей через SVD с k=100: вместо 1 000 000 значений храним 100·(1000+1000) = 200 000. Сжатие в 5 раз без заметной потери качества.",
  js: `// Усечённое SVD через numeric.js
// Усечённое SVD для страницы «Код» (автономная копия)
export function svdTruncated(A, k) {
  // Вызываем полное SVD из библиотеки numeric
  const { U, S, V } = numeric.svd(A);
  const { m, n } = dims(A);
  // Реальный ранг — минимум из k и доступных размеров
  const r = Math.max(1, Math.min(k, S.length,
    U[0].length, V.length, m, n));
  // Берём первые r столбцов U (левые сингулярные векторы)
  const Uk = takeUk(U, r);
  // Берём первые r сингулярных чисел
  const Sk = S.slice(0, r);
  // Берём первые r строк V (правые сингулярные векторы)
  const Vk = takeVk(V, r);
  // U·Σ — взвешенные левые векторы
  const US = dot(Uk, diag(Sk));
  // (U·Σ)·Vᵀ = Ã — приближение ранга r
  const Ahat = dot(US, transpose(Vk));
  return { Ahat, U, S, V, r, Uk, Sk, Vk, US };
}`,
  py: `# Усечённое SVD через numpy.linalg.svd
import numpy as np

def svd_truncated(A, k):
    # Полное SVD: A = U·S·V†
    # full_matrices=False — экономный формат
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Реальный ранг — минимум из k и размера S
    r = max(1, min(k, len(S)))
    # Берём первые r столбцов U и строк V†
    Uk = U[:, :r]
    Sk = S[:r]
    Vkt = Vt[:r, :]
    # Σ как диагональная матрица r×r
    Sigma = np.diag(Sk)
    # U·Σ
    US = Uk @ Sigma
    # (U·Σ)·V† = Ã
    Ahat = US @ Vkt
    return {'Ahat': Ahat, 'r': r,
            'Uk': Uk, 'Sk': Sk, 'Vkt': Vkt}`,
};

codeBlocks.pca = {
  title: "2. PCA — метод главных компонент",
  badge: "центрирование + SVD",
  desc: `Центрирует данные (вычитает среднее по каждому столбцу), затем применяет
    SVD к центрированной матрице. Первые k главных компонент описывают наибольшую
    дисперсию данных. <strong>Где используется:</strong> уменьшение размерности,
    визуализация многомерных данных, сжатие признаков в ML.`,
  example: "50 характеристик товара → PCA выделяет 3 главные компоненты, объясняющие 90% вариации. Сокращение данных в 16 раз с сохранением структуры.",
  js: `// PCA через центрирование + SVD
export function pcaReconstructWithSteps(A, k) {
  const { m, n } = dims(A);
  // Шаг 1: среднее каждого столбца
  const mean = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) mean[j] += A[i][j];
    mean[j] /= m;
  }
  // Шаг 2: центрируем — вычитаем среднее
  const X = zeros(m, n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) X[i][j] = A[i][j] - mean[j];
  // Шаг 3: SVD центрированной матрицы (усечённое)
  const svdX = svdTruncated(X, k);
  const { Ahat: Xk } = svdX;
  // Шаг 4: прибавляем среднее обратно → Ã
  const Ahat = zeros(m, n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) Ahat[i][j] = Xk[i][j] + mean[j];
  return { Ahat, mean, X, ...svdX };
}`,
  py: `# PCA: центрирование + SVD (или sklearn)
import numpy as np
from sklearn.decomposition import PCA

def pca_reconstruct(A, k):
    # Шаг 1: среднее по столбцам
    mean = A.mean(axis=0)
    # Шаг 2: центрируем
    X = A - mean
    # Шаг 3: SVD центрированной матрицы
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(k, len(S))
    # Усечённое приближение X̂ = U·Σ·V†
    Xk = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
    # Шаг 4: прибавляем среднее обратно
    Ahat = Xk + mean
    return Ahat

# Или через sklearn (одна строка):
def pca_sklearn(A, k):
    pca = PCA(n_components=k)
    Z = pca.fit_transform(A)       # проекция на k компонент
    Ahat = pca.inverse_transform(Z)  # восстановление
    return Ahat`,
};

codeBlocks.nmf = {
  title: "3. NMF — неотрицательное матричное разложение",
  badge: "мультипликативное обновление",
  desc: `Ищет две неотрицательные матрицы <code>W</code> и <code>H</code> такие, что
    <code>A ≈ W·H</code>. Обновление по правилу:
    <code>H ← H · (W<sup>T</sup>A) / (W<sup>T</sup>WH)</code>,
    <code>W ← W · (AH<sup>T</sup>) / (WHH<sup>T</sup>)</code>.
    <strong>Где используется:</strong> тематическое моделирование, спектры, разделение аудио.`,
  example: "1000 документов × 5000 слов. NMF с k=10 находит 10 «тем»: W — вес темы в документе, H — слова темы. Человек читает 10 списков слов вместо 1000 текстов.",
  js: `// NMF — мультипликативное обновление (MU)
export function nmfReconstructHistory(A, k, totalIters) {
  const { m, n } = dims(A);
  // Если есть отрицательные — сдвигаем вверх
  const shift = minMax(A).mn < 0 ? -minMax(A).mn : 0;
  const X = A.map(r => r.map(v => v + shift));

  // Случайная инициализация W (m×k) и H (k×n)
  let W = zeros(m, k), H = zeros(k, n);
  for (let i = 0; i < m; i++)
    for (let t = 0; t < k; t++) W[i][t] = rnd();
  for (let t = 0; t < k; t++)
    for (let j = 0; j < n; j++) H[t][j] = rnd();

  for (let it = 0; it < totalIters; it++) {
    // Обновление H: H · (WᵀX) / (WᵀW·H)
    const WT_X = dot(transpose(W), X);
    const WT_W_H = dot(dot(transpose(W), W), H);
    for (let t = 0; t < k; t++)
      for (let j = 0; j < n; j++)
        H[t][j] *= WT_X[t][j] / (WT_W_H[t][j] + 1e-9);

    // Обновление W: W · (X·Hᵀ) / (W·H·Hᵀ)
    const X_HT = dot(X, transpose(H));
    const W_H_HT = dot(W, dot(H, transpose(H)));
    for (let i = 0; i < m; i++)
      for (let t = 0; t < k; t++)
        W[i][t] *= X_HT[i][t] / (W_H_HT[i][t] + 1e-9);
  }
  // Ã = W·H, отнимаем сдвиг
  const Ahat = dot(W, H).map(r => r.map(v => v - shift));
  return { result: { Ahat, W, H }, history };
}`,
  py: `# NMF — мультипликативное обновление
import numpy as np
from sklearn.decomposition import NMF

def nmf_reconstruct(A, k, total_iters=80):
    # Сдвиг для неотрицательности
    shift = -A.min() if A.min() < 0 else 0
    X = A + shift
    # Случайная инициализация
    W = np.random.rand(A.shape[0], k) + 0.5
    H = np.random.rand(k, A.shape[1]) + 0.5
    eps = 1e-9
    for _ in range(total_iters):
        # H ← H · (WᵀX) / (WᵀW·H)
        H *= (W.T @ X) / (W.T @ W @ H + eps)
        # W ← W · (X·Hᵀ) / (W·H·Hᵀ)
        W *= (X @ H.T) / (W @ H @ H.T + eps)
    # Ã = W·H, убираем сдвиг
    Ahat = W @ H - shift
    return Ahat, W, H

# Или через sklearn:
def nmf_sklearn(A, k):
    model = NMF(n_components=k, init='random',
                max_iter=200, random_state=0)
    W = model.fit_transform(A)   # веса тем
    H = model.components_        # сами темы
    return W @ H, W, H`,
};

codeBlocks.cur = {
  title: "4. CUR — разложение по строкам и столбцам",
  badge: "выбор + псевдообратная",
  desc: `Выбирает k самых «важных» столбцов (C) и k самых «важных» строк (R)
    по евклидовой норме, вычисляет связующую матрицу <code>U = C<sup>+</sup>·A·R<sup>+</sup></code>
    (через псевдообратные C<sup>+</sup> и R<sup>+</sup>). Итог: <code>A ≈ C·U·R</code>.
    <strong>Где используется:</strong> разреженные данные, интерпретируемые признаки.`,
  example: "Матрица 10⁶×10⁵ покупок. CUR выбирает 20 реальных товаров и 20 реальных покупателей — интерпретируемое приближение (в отличие от SVD, где U/V — абстракции).",
  js: `// CUR — выбор столбцов и строк по норме
export function curReconstruct(A, k) {
  const { m, n } = dims(A);
  const r = Math.max(1, Math.min(k, m, n));

  // Квадраты длин столбцов и строк
  const colNorms = new Array(n).fill(0);
  const rowNorms = new Array(m).fill(0);
  for (let j = 0; j < n; j++)
    for (let i = 0; i < m; i++) colNorms[j] += A[i][j] ** 2;
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) rowNorms[i] += A[i][j] ** 2;

  // Топ-r столбцов и строк по убыванию нормы
  const topCols = [...Array(n).keys()]
    .sort((a, b) => colNorms[b] - colNorms[a]).slice(0, r);
  const topRows = [...Array(m).keys()]
    .sort((a, b) => rowNorms[b] - rowNorms[a]).slice(0, r);

  // C: выбранные столбцы (m×r), R: выбранные строки (r×n)
  const C = zeros(m, r), R = zeros(r, n);
  const Wcore = zeros(r, r);  // пересечение C ∩ R
  for (let i = 0; i < m; i++)
    for (let t = 0; t < r; t++) C[i][t] = A[i][topCols[t]];
  for (let t = 0; t < r; t++)
    for (let j = 0; j < n; j++) R[t][j] = A[topRows[t]][j];
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r; j++) Wcore[i][j] = A[topRows[i]][topCols[j]];

  // C⁺ и R⁺ — псевдообратные через SVD
  const Cp = pinv(C);
  const Rp = pinv(R);
  // U = C⁺ · A · R⁺ — связующая матрица
  const Uc = dot(dot(Cp, A), Rp);

  // Ã = C · U · R
  const Ahat = dot(dot(C, Uc), R);
  return { Ahat, C, U: Uc, R, Wcore, topRows, topCols, r, Cp, Rp };
}`,
  py: `# CUR — выбор по норме + псевдообратная
import numpy as np

def cur_reconstruct(A, k):
    m, n = A.shape
    r = max(1, min(k, m, n))

    # Квадраты длин столбцов и строк
    col_norms = np.linalg.norm(A, axis=0) ** 2
    row_norms = np.linalg.norm(A, axis=1) ** 2

    # Индексы топ-r столбцов и строк
    top_cols = np.argsort(col_norms)[-r:][::-1]
    top_rows = np.argsort(row_norms)[-r:][::-1]

    # C, R и W (пересечение)
    C = A[:, top_cols]               # m×r
    R = A[top_rows, :]               # r×n
    Wcore = A[top_rows[:, None], top_cols]  # r×r

    # C⁺ и R⁺ — псевдообратные
    Cp = np.linalg.pinv(C)           # r×m
    Rp = np.linalg.pinv(R)           # n×r
    # U = C⁺ · A · R⁺ — связующая матрица
    Uc = Cp @ A @ Rp                 # r×r

    # Ã = C · U · R
    Ahat = C @ Uc @ R
    return {'Ahat': Ahat, 'C': C,
            'U': Uc, 'R': R,
            'top_rows': top_rows,
            'top_cols': top_cols}`,
};

codeBlocks.als = {
  title: "5. ALS — попеременные наименьшие квадраты",
  badge: "итеративное уточнение",
  desc: `Ищет <code>A ≈ X·Y<sup>T</sup></code>, попеременно фиксируя одну матрицу
    и уточняя другую через МНК с регуляризацией.
    <code>X ← argmin ||A − X·Y<sup>T</sup>||²</code> при фиксированном Y.
    <strong>Где используется:</strong> рекомендательные системы (Netflix Prize),
    коллаборативная фильтрация, заполнение пропусков.`,
  example: "Netflix: 500K пользователей × 18K фильмов (редкая матрица рейтингов). ALS находит 50-мерные векторы пользователей (X) и фильмов (Y), предсказывая любой рейтинг как xᵤ·yᵢ.",
  js: `// ALS — попеременные наименьшие квадраты
export function alsReconstructHistory(A, k, totalIters, reg = 1e-2) {
  const { m, n } = dims(A);
  const r = Math.max(1, Math.min(k, m, n));

  // Случайные начальные X (m×r) и Y (n×r)
  let X = zeros(m, r), Y = zeros(n, r);
  for (let i = 0; i < m; i++)
    for (let t = 0; t < r; t++) X[i][t] = rnd();
  for (let j = 0; j < n; j++)
    for (let t = 0; t < r; t++) Y[j][t] = rnd();

  for (let it = 0; it < totalIters; it++) {
    // --- Фиксируем Y, уточняем X ---
    // Yᵀ·Y + λ·I (маленькая r×r — решаем быстро)
    const YTY = dot(transpose(Y), Y);
    for (let i = 0; i < r; i++) YTY[i][i] += reg;
    // Для каждой строки i: xᵢ = (YᵀY)⁻¹ · Yᵀ·aᵢ
    for (let i = 0; i < m; i++) {
      const rhs = new Array(r).fill(0);
      for (let t = 0; t < r; t++)
        for (let j = 0; j < n; j++) rhs[t] += Y[j][t] * A[i][j];
      const sol = numeric.solve(YTY, rhs);
      for (let t = 0; t < r; t++) X[i][t] = sol[t];
    }
    // --- Фиксируем X, уточняем Y ---
    const XTX = dot(transpose(X), X);
    for (let i = 0; i < r; i++) XTX[i][i] += reg;
    for (let j = 0; j < n; j++) {
      const rhs = new Array(r).fill(0);
      for (let t = 0; t < r; t++)
        for (let i = 0; i < m; i++) rhs[t] += X[i][t] * A[i][j];
      const sol = numeric.solve(XTX, rhs);
      for (let t = 0; t < r; t++) Y[j][t] = sol[t];
    }
  }
  // Ã = X·Yᵀ
  const Ahat = dot(X, transpose(Y));
  return { result: { Ahat, X, Y, r, reg }, history };
}`,
  py: `# ALS — попеременные наименьшие квадраты
import numpy as np

def als_reconstruct(A, k, total_iters=80, reg=1e-2):
    m, n = A.shape
    r = min(k, m, n)

    # Случайная инициализация
    X = np.random.randn(m, r) * 0.5
    Y = np.random.randn(n, r) * 0.5

    I = np.eye(r)

    for _ in range(total_iters):
        # --- Шаг X: X = A·Y · (YᵀY + λ·I)⁻¹ ---
        YTY = Y.T @ Y + reg * I
        X = (A @ Y) @ np.linalg.inv(YTY)

        # --- Шаг Y: Y = Aᵀ·X · (XᵀX + λ·I)⁻¹ ---
        XTX = X.T @ X + reg * I
        Y = (A.T @ X) @ np.linalg.inv(XTX)

    # Ã = X·Yᵀ
    Ahat = X @ Y.T
    return Ahat, X, Y

# Компактнее через solve (численно устойчивее):
def als_solve(A, k, total_iters=80, reg=1e-2):
    m, n = A.shape
    r = min(k, m, n)
    X = np.random.randn(m, r) * 0.5
    Y = np.random.randn(n, r) * 0.5
    I = np.eye(r)
    for _ in range(total_iters):
        X = np.linalg.solve(Y.T @ Y + reg * I, (A @ Y).T).T
        Y = np.linalg.solve(X.T @ X + reg * I, (A.T @ X).T).T
    return X @ Y.T, X, Y`,
};

function renderCodeBlock(id) {
  const b = codeBlocks[id];
  const html = `
    <div class="algo-block">
      <div class="algo-header">
        <h3>${b.title}</h3>
        <span class="badge">${b.badge}</span>
      </div>
      <div class="desc">${b.desc}</div>
      <div class="example">
        <span style="color:var(--good)">▸</span> ${b.example}
      </div>
      <div class="code-grid">
        <div class="code-col lang-js">
          <div class="code-col-head">
            <span class="lang-label">JavaScript</span>
            <button class="copy-btn" data-copy-id="${id}-js">Копировать</button>
          </div>
          <pre>${escapeHtml(b.js)}</pre>
        </div>
        <div class="code-col lang-py">
          <div class="code-col-head">
            <span class="lang-label">Python</span>
            <button class="copy-btn" data-copy-id="${id}-py">Копировать</button>
          </div>
          <pre>${escapeHtml(b.py)}</pre>
        </div>
      </div>
    </div>
  `;
  return html;
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Рендерит страницу «Код» с алгоритмами, примерами и интерактивным редактором
export function renderCodePage(container) {
  let html = `
    <style>
      .code-tab .algo-block {
        background: linear-gradient(135deg, var(--panel2), var(--panel));
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.5rem;
      }
      .code-tab .algo-header {
        display: flex; align-items: baseline; gap: 0.8rem;
        margin-bottom: 0.6rem; flex-wrap: wrap;
      }
      .code-tab .algo-header h3 {
        font-size: 1.05rem; font-weight: 700; color: var(--accent);
        margin: 0;
      }
      .code-tab .badge {
        font-size: 0.68rem; font-weight: 600;
        background: rgba(91,156,246,0.2); color: var(--accent2);
        padding: 0.15rem 0.55rem; border-radius: 999px;
        white-space: nowrap;
      }
      .code-tab .desc {
        font-size: 0.85rem; line-height: 1.55;
        margin-bottom: 0.6rem; color: var(--text);
      }
      .code-tab .desc strong { color: var(--accent2); }
      .code-tab .desc code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.8rem;
        background: rgba(0,0,0,0.25); padding: 0.05rem 0.3rem;
        border-radius: 4px;
      }
      .code-tab .example {
        background: rgba(0,0,0,0.2);
        border-left: 3px solid var(--good);
        padding: 0.55rem 0.85rem;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-bottom: 1rem;
        color: var(--muted);
        line-height: 1.5;
      }
      .code-tab .code-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
      }
      @media (max-width: 900px) {
        .code-tab .code-grid { grid-template-columns: 1fr; }
      }
      .code-tab .code-col { min-width: 0; }
      .code-tab .code-col-head {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.4rem;
      }
      .code-tab .copy-btn {
        font-size: 0.68rem; font-weight: 600;
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--border);
        color: var(--muted);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        cursor: pointer;
        transition: var(--transition);
        font-family: inherit;
        line-height: 1.4;
      }
      .code-tab .copy-btn:hover {
        background: rgba(255,255,255,0.12);
        color: var(--text);
      }
      .code-tab .copy-btn.copied {
        background: rgba(74,222,128,0.15);
        border-color: var(--good);
        color: var(--good);
      }
      .code-tab .lang-label {
        display: inline-block;
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 0.2rem 0.6rem;
        border-radius: 6px;
      }
      .code-tab .lang-js .lang-label {
        background: rgba(247,223,30,0.15); color: #f7df1e;
      }
      .code-tab .lang-py .lang-label {
        background: rgba(69,133,214,0.15); color: #4585d6;
      }
      .code-tab pre {
        background: rgba(0,0,0,0.35);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.8rem 0.5rem;
        overflow-x: auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.75rem;
        line-height: 1.55;
        tab-size: 2;
        color: var(--text);
      }
    </style>
    <div class="code-tab">
      <div class="code-tab-intro" style="margin-bottom:1.2rem">
        <h2 style="font-size:1.2rem;font-weight:700;margin-bottom:0.2rem">Код матричных разложений</h2>
        <p class="muted" style="font-size:0.85rem">
          JavaScript и Python — каждая строка с комментарием, описание и пример из жизни
        </p>
      </div>
  `;

  for (const key of ["svd", "pca", "nmf", "cur", "als"]) {
    html += renderCodeBlock(key);
  }

  html += `<p class="muted" style="text-align:center;margin-top:0.5rem;font-size:0.8rem">Matrix Lab — код разложений</p></div>`;
  container.innerHTML = html;

  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".copy-btn");
    if (!btn) return;

    const pre = btn.closest(".code-col").querySelector("pre");
    const text = pre.textContent;

    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.textContent;
      btn.textContent = "Скопировано!";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = orig;
        btn.classList.remove("copied");
      }, 1800);
    }).catch(() => {
      btn.textContent = "Ошибка";
      setTimeout(() => { btn.textContent = "Копировать"; }, 1500);
    });
  });
}
