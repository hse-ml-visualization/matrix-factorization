/* global numeric */

import { renderMatrixBlock } from "./heatmap.js";
import { dims, clone, zeros, dot, transpose, diff, frobNorm, minMax, randomMatrix } from "./matrix.js";
import { svdTruncated, pcaReconstructWithSteps, nmfReconstructHistory, curReconstruct, alsReconstructHistory } from "./decompositions.js";
import { clear, latexToHtml } from "./dom.js";
import { viridis, rgb as viridisRgb } from "./palettes.js";

const OP_COLORS = {
  multiply: "#4ade80",
  transpose: "#5b9cf6",
  normalize: "#fbbf24",
  eigen: "#a78bfa",
  init: "#f87171",
  default: "#94a3b8",
};

const OP_TITLES = {
  multiply: "Умножение матриц",
  transpose: "Транспонирование",
  normalize: "Нормализация",
  eigen: "Собственные значения и векторы",
  init: "Инициализация",
};

function opColor(op) {
  return OP_COLORS[op] || OP_COLORS.default;
}

const SUB = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];
function subNum(n) {
  return String(n).split("").map((d) => SUB[+d] || d).join("");
}

// ── Pipeline generators ──

export function generateSvdPipeline(A, k) {
  const { m, n } = dims(A);
  const result = svdTruncated(A, k);
  const { Ahat, Uk, Sk, Vk, r } = result;
  const Vt = transpose(Vk);
  const At = transpose(A);
  const AtA = dot(At, A);
  const sigmaDiag = zeros(r, r);
  for (let i = 0; i < Sk.length && i < r; i++) sigmaDiag[i][i] = Sk[i];
  const scaleA = minMax(A);
  const scaleAtA = minMax(AtA);

  const lambdaVals = Sk.map((s) => s * s);

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  steps.push({ type: "arrow", label: "Транспонируем и умножаем: Aᵀ·A", op: "transpose", data: { type: "matrix_product", leftLabel: "Aᵀ", left: At, rightLabel: "A", right: A, resultLabel: "Aᵀ·A", result: AtA } });
  steps.push({ type: "matrix", id: "AtA", title: "Aᵀ·A", subtitle: "Произведение", data: AtA, scale: scaleAtA });
  steps.push({ type: "arrow", label: "Собственные значения λ = σ²", op: "eigen", data: { type: "eigenvalues", values: lambdaVals.map((v, i) => ({ value: v, label: `λ${subNum(i + 1)} = σ${subNum(i + 1)}²` })) } });
  steps.push({ type: "eigenvalues", id: "eigen", values: Sk.map((s, i) => ({ value: s * s, label: `λ${subNum(i + 1)} = ${(s * s).toFixed(4)}` })), total: Sk.length });
  steps.push({ type: "arrow", label: "Извлекаем корень: σ = √λ", op: "normalize", data: { type: "values_map", fromLabel: "λ = σ²", from: lambdaVals, toLabel: "σ = √λ", to: Sk } });
  steps.push({ type: "matrix", id: "sigma", title: "Σ", subtitle: "Сингулярные числа", data: sigmaDiag, scale: minMax(sigmaDiag) });
  steps.push({ type: "arrow", label: "Находим собственные векторы U и V", op: "normalize", data: { type: "iteration_note", text: "Собственные векторы Aᵀ·A образуют столбцы V, а A·Aᵀ — столбцы U. Берём первые k.", matrices: [{ label: "U", matrix: Uk }, { label: "Vᵀ", matrix: Vt }] } });
  steps.push({ type: "matrices_row", id: "UV", matrices: [
    { title: "U", data: Uk, subtitle: `размер ${m}×${r}` },
    { title: "Vᵀ", data: Vt, subtitle: `размер ${r}×${n}` },
  ]});
  steps.push({ type: "arrow", label: "Перемножаем: A ≈ U·Σ·Vᵀ", op: "multiply", data: { type: "triple_product", aLabel: "U", a: Uk, bLabel: "Σ", b: sigmaDiag, cLabel: "Vᵀ", c: Vt, resultLabel: "Ã", result: Ahat } });
  steps.push({ type: "matrix", id: "Ahat", title: "Ã ≈ U·Σ·Vᵀ", subtitle: "Низкоранговое приближение", data: Ahat, scale: scaleA });
  steps.push({ type: "arrow", label: "Анализ ошибки", op: "normalize", data: { type: "error_detail", A, Ahat, scale: minMax(A) } });
  steps.push({ type: "error", id: "error", A, Ahat });
  return steps;
}

export function generatePcaPipeline(A, k) {
  const { m, n } = dims(A);
  const result = pcaReconstructWithSteps(A, k);
  const { mean, X, Uk, Sk, Vk, US, Xk, Ahat } = result;
  const Vt = transpose(Vk);
  const Xt = transpose(X);
  const scaleA = minMax(A);
  const scaleX = minMax(X);
  const sigmaDiag = zeros(Math.min(m, n), Math.min(m, n));
  for (let i = 0; i < Sk.length && i < sigmaDiag.length; i++) sigmaDiag[i][i] = Sk[i];
  const meanRow = [mean.slice()];
  const cov = dot(Xt, X);

  const lambdaVals = Sk.map((s) => s * s);

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  steps.push({ type: "arrow", label: "Вычисляем средние μ по столбцам", op: "normalize", data: { type: "means", values: mean } });
  steps.push({ type: "matrix", id: "mean", title: "μ (средние)", subtitle: "По столбцам", data: meanRow, scale: minMax(meanRow) });
  steps.push({ type: "arrow", label: "Центрируем: X = A − μ", op: "transpose", data: { type: "subtraction", leftLabel: "A", left: A, rightLabel: "μ", right: meanRow, resultLabel: "X = A − μ", result: X } });
  steps.push({ type: "matrix", id: "X", title: "X = A − μ", subtitle: "Центрированные данные", data: X, scale: scaleX });
  steps.push({ type: "arrow", label: "Ковариация: Xᵀ·X", op: "multiply", data: { type: "matrix_product", leftLabel: "Xᵀ", left: Xt, rightLabel: "X", right: X, resultLabel: "Xᵀ·X", result: cov } });
  steps.push({ type: "matrix", id: "cov", title: "Xᵀ·X", subtitle: "Ковариационная матрица", data: cov, scale: null });
  steps.push({ type: "arrow", label: "Собственные значения и главные компоненты", op: "eigen", data: { type: "eigenvalues", values: lambdaVals.map((v, i) => ({ value: v, label: `λ${subNum(i + 1)}` })) } });
  steps.push({ type: "matrices_row", id: "comp", matrices: [
    { title: "U (на X)", data: Uk, subtitle: `размер ${m}×${result.r}` },
    { title: "Σ", data: sigmaDiag, subtitle: `размер ${result.r}×${result.r}` },
    { title: "Vᵀ", data: Vt, subtitle: `размер ${result.r}×${n}` },
  ]});
  steps.push({ type: "arrow", label: "Восстановление X̂ = U·Σ·Vᵀ", op: "multiply", data: { type: "triple_product", aLabel: "U", a: Uk, bLabel: "Σ", b: sigmaDiag, cLabel: "Vᵀ", c: Vt, resultLabel: "X̂", result: Xk } });
  steps.push({ type: "matrix", id: "Xk", title: "X̂ = U·Σ·Vᵀ", subtitle: "Низкоранговое приближение X", data: Xk, scale: scaleX });
  steps.push({ type: "arrow", label: "Добавление μ: Ã = X̂ + μ", op: "normalize", data: { type: "addition", leftLabel: "X̂", left: Xk, rightLabel: "μ", right: meanRow, resultLabel: "Ã", result: Ahat } });
  steps.push({ type: "matrix", id: "Ahat", title: "Ã = X̂ + μ", subtitle: "Восстановленная матрица", data: Ahat, scale: scaleA });
  steps.push({ type: "arrow", label: "Анализ ошибки", op: "normalize", data: { type: "error_detail", A, Ahat, scale: minMax(A) } });
  steps.push({ type: "error", id: "error", A, Ahat });
  return steps;
}

export function generateNmfPipeline(A, k, totalIters) {
  const { m, n } = dims(A);
  const { mn } = minMax(A);
  const shift = mn < 0 ? -mn : 0;
  const X = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) X[i][j] = A[i][j] + shift;
  const scaleA = minMax(A);
  const cappedIters = Math.min(totalIters || 20, 50);
  const { result, history } = nmfReconstructHistory(A, k, cappedIters);
  const { Ahat, W, H } = result;
  const WHinit = dot(W, H);
  const midIdx = Math.min(Math.floor(history.length / 2), history.length - 1);
  const midHist = history.length > 2 ? history[midIdx] : null;

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  if (shift > 0) {
    steps.push({ type: "arrow", label: `Сдвиг на ${shift.toFixed(2)} для неотрицательности`, op: "normalize", data: { type: "iteration_note", text: `Все элементы A сдвигаются на ${shift.toFixed(2)}, чтобы минимальный элемент стал 0 (неотрицательность для NMF).` } });
    steps.push({ type: "matrix", id: "X", title: `A + ${shift.toFixed(2)}`, subtitle: "Неотрицательная", data: X, scale: minMax(X) });
  }
  steps.push({ type: "arrow", label: "Инициализируем W и H случайно", op: "init", data: { type: "init_matrices", matrices: [{ label: "W", matrix: W }, { label: "H", matrix: H }] } });
  steps.push({ type: "matrices_row", id: "init", matrices: [
    { title: "W", data: W, subtitle: `${m}×${k}` },
    { title: "H", data: H, subtitle: `${k}×${n}` },
  ]});
  steps.push({ type: "arrow", label: "Первое приближение: W·H", op: "multiply", data: { type: "matrix_product", leftLabel: "W", left: W, rightLabel: "H", right: H, resultLabel: "W·H", result: WHinit } });
  steps.push({ type: "matrix", id: "WHinit", title: "W·H (начало)", data: WHinit, scale: scaleA });
  if (history.length > 1) {
    steps.push({ type: "arrow", label: "Обновление W и H по итерациям", op: "multiply", data: { type: "iteration_history", history, scale: scaleA, label: "Итерация" } });
  }
  steps.push({ type: "arrow", label: `Финальное W·H после ${cappedIters} итераций`, op: "multiply", data: { type: "matrix_product", leftLabel: "W", left: W, rightLabel: "H", right: H, resultLabel: "W·H", result: dot(W, H) } });
  steps.push({ type: "matrices_row", id: "finalFactors", matrices: [
    { title: "W (финал)", data: W, subtitle: `${m}×${k}` },
    { title: "H (финал)", data: H, subtitle: `${k}×${n}` },
  ]});
  steps.push({ type: "arrow", label: "Восстановление: A ≈ W·H", op: "multiply", data: { type: "matrix_product", leftLabel: "W", left: W, rightLabel: "H", right: H, resultLabel: "Ã ≈ W·H", result: Ahat } });
  steps.push({ type: "matrix", id: "Ahat", title: "Ã ≈ W·H", subtitle: "Итоговое приближение", data: Ahat, scale: scaleA });
  steps.push({ type: "arrow", label: "Анализ ошибки", op: "normalize", data: { type: "error_detail", A, Ahat, scale: minMax(A) } });
  steps.push({ type: "error", id: "error", A, Ahat });
  return steps;
}

export function generateCurPipeline(A, k) {
  const { m, n } = dims(A);
  const result = curReconstruct(A, k);
  const { Ahat, C, U, R, Wcore, topRows, topCols, r, Cp, Rp } = result;
  const scaleA = minMax(A);
  const colNorms = new Array(n).fill(0);
  const rowNorms = new Array(m).fill(0);
  for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) colNorms[j] += A[i][j] * A[i][j];
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) rowNorms[i] += A[i][j] * A[i][j];

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  steps.push({ type: "arrow", label: "Считаем нормы строк и столбцов", op: "normalize", data: { type: "norm_computation", colNorms, rowNorms, topCols, topRows } });
  steps.push({ type: "norms", id: "norms", colNorms, rowNorms, topRows, topCols, r });
  steps.push({ type: "arrow", label: `Выбираем топ-${r} столбцов и строк`, op: "normalize", data: { type: "selection", topRows, topCols, C, R, W: Wcore } });
  steps.push({ type: "matrices_row", id: "CR", matrices: [
    { title: "C (столбцы)", data: C, subtitle: `${m}×${r}` },
    { title: "R (строки)", data: R, subtitle: `${r}×${n}` },
    { title: "W (пересечение)", data: Wcore, subtitle: `${r}×${r}` },
  ]});
  steps.push({ type: "arrow", label: "Псевдообратная: C⁺, R⁺, U = C⁺·A·R⁺", op: "transpose", data: { type: "pseudoinverse", C, R, Cp, Rp, A, U, W: Wcore } });
  steps.push({ type: "matrix", id: "U", title: "U = C⁺·A·R⁺", subtitle: "Связующая матрица", data: U, scale: minMax(U) });
  steps.push({ type: "arrow", label: "Восстанавливаем: C·U·R", op: "multiply", data: { type: "triple_product", aLabel: "C", a: C, bLabel: "U", b: U, cLabel: "R", c: R, resultLabel: "Ã", result: Ahat } });
  steps.push({ type: "matrix", id: "Ahat", title: "Ã ≈ C·U·R", subtitle: "CUR-приближение", data: Ahat, scale: scaleA });
  steps.push({ type: "arrow", label: "Анализ ошибки", op: "normalize", data: { type: "error_detail", A, Ahat, scale: minMax(A) } });
  steps.push({ type: "error", id: "error", A, Ahat });
  return steps;
}

export function generateAlsPipeline(A, k, totalIters) {
  const { m, n } = dims(A);
  const scaleA = minMax(A);
  const cappedIters = Math.min(totalIters || 20, 50);
  const { result, history } = alsReconstructHistory(A, k, cappedIters, 1e-2);
  const { Ahat, X, Y, r } = result;
  const Yt = transpose(Y);
  const XYtInit = dot(X, Yt);
  const midIdx = Math.min(Math.floor(history.length / 2), history.length - 1);
  const midHist = history.length > 2 ? history[midIdx] : null;

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  steps.push({ type: "arrow", label: "Инициализируем X и Y случайно", op: "init", data: { type: "init_matrices", matrices: [{ label: "X", matrix: X }, { label: "Y", matrix: Y }] } });
  steps.push({ type: "matrices_row", id: "init", matrices: [
    { title: "X", data: X, subtitle: `${m}×${r}` },
    { title: "Y", data: Y, subtitle: `${n}×${r}` },
  ]});
  steps.push({ type: "arrow", label: "Начальное произведение: X·Yᵀ", op: "multiply", data: { type: "matrix_product", leftLabel: "X", left: X, rightLabel: "Yᵀ", right: Yt, resultLabel: "X·Yᵀ", result: XYtInit } });
  steps.push({ type: "matrix", id: "XYtInit", title: "X·Yᵀ (начало)", data: XYtInit, scale: scaleA });
  if (history.length > 1) {
    steps.push({ type: "arrow", label: "Итерации ALS", op: "multiply", data: { type: "iteration_history", history, scale: scaleA, label: "Итерация ALS" } });
  }
  steps.push({ type: "arrow", label: `Финальные X и Y после ${cappedIters} итераций`, op: "multiply", data: { type: "iteration_note", text: `После ${cappedIters} итераций попеременных наименьших квадратов X и Y сходятся.`, matrices: [{ label: "X", matrix: X }, { label: "Yᵀ", matrix: transpose(Y) }] } });
  steps.push({ type: "matrices_row", id: "finalFactors", matrices: [
    { title: "X (финал)", data: X, subtitle: `${m}×${r}` },
    { title: "Yᵀ (финал)", data: transpose(Y), subtitle: `${r}×${n}` },
  ]});
  steps.push({ type: "arrow", label: "Восстановление: A ≈ X·Yᵀ", op: "multiply", data: { type: "matrix_product", leftLabel: "X", left: X, rightLabel: "Yᵀ", right: transpose(Y), resultLabel: "Ã ≈ X·Yᵀ", result: Ahat } });
  steps.push({ type: "matrix", id: "Ahat", title: "Ã ≈ X·Yᵀ", subtitle: "Итоговое приближение", data: Ahat, scale: scaleA });
  steps.push({ type: "arrow", label: "Анализ ошибки", op: "normalize", data: { type: "error_detail", A, Ahat, scale: minMax(A) } });
  steps.push({ type: "error", id: "error", A, Ahat });
  return steps;
}

export function generatePipeline(algo, A, k, iters) {
  switch (algo) {
    case "svd": return generateSvdPipeline(A, k);
    case "pca": return generatePcaPipeline(A, k);
    case "nmf": return generateNmfPipeline(A, k, iters);
    case "cur": return generateCurPipeline(A, k);
    case "als": return generateAlsPipeline(A, k, iters);
    default: return [];
  }
}

// ── Step rendering ──

function renderMatrixStep(step) {
  const wrap = document.createElement("div");
  wrap.className = "vis-step__content";
  const scale = step.scale || minMax(step.data);
  renderMatrixBlock(wrap, step.title, step.data, { scale, subtitle: step.subtitle });
  return wrap;
}

function renderMatricesRowStep(step) {
  const wrap = document.createElement("div");
  wrap.className = "vis-matrices-row";
  for (const m of step.matrices) {
    const block = document.createElement("div");
    block.className = "vis-matrix-inline";
    const scale = m.scale || minMax(m.data);
    renderMatrixBlock(block, m.title, m.data, { scale, subtitle: m.subtitle });
    wrap.appendChild(block);
  }
  return wrap;
}

function renderEigenvaluesStep(step) {
  const wrap = document.createElement("div");
  wrap.className = "vis-eigenvalues";
  const maxVal = Math.max(...step.values.map((v) => v.value), 1e-12);
  for (const ev of step.values) {
    const item = document.createElement("div");
    item.className = "vis-eigenvalue";
    const size = 0.7 + 0.8 * (ev.value / maxVal);
    item.style.fontSize = `${size}rem`;
    item.style.fontWeight = "700";
    item.style.color = opColor("eigen");
    item.textContent = ev.label || ev.value.toFixed(4);
    wrap.appendChild(item);
  }
  return wrap;
}

function renderNormsStep(step) {
  const wrap = document.createElement("div");
  wrap.className = "vis-norms";

  const colDiv = document.createElement("div");
  colDiv.className = "vis-norms__group";
  const colTitle = document.createElement("div");
  colTitle.className = "vis-norms__title";
  colTitle.textContent = "Нормы столбцов";
  colDiv.appendChild(colTitle);
  const maxCol = Math.max(...step.colNorms, 1e-12);
  for (let j = 0; j < step.colNorms.length; j++) {
    const bar = document.createElement("div");
    bar.className = "vis-norms__bar";
    const fill = document.createElement("div");
    fill.className = "vis-norms__fill";
    fill.style.width = `${(step.colNorms[j] / maxCol) * 100}%`;
    const label = document.createElement("span");
    label.className = "vis-norms__label";
    label.textContent = `col ${j}: ${step.colNorms[j].toFixed(2)}`;
    if (step.topCols && step.topCols.includes(j)) {
      fill.style.background = "var(--accent)";
      label.style.color = "var(--accent)";
    }
    bar.appendChild(fill);
    bar.appendChild(label);
    colDiv.appendChild(bar);
  }
  wrap.appendChild(colDiv);

  const rowDiv = document.createElement("div");
  rowDiv.className = "vis-norms__group";
  const rowTitle = document.createElement("div");
  rowTitle.className = "vis-norms__title";
  rowTitle.textContent = "Нормы строк";
  rowDiv.appendChild(rowTitle);
  const maxRow = Math.max(...step.rowNorms, 1e-12);
  for (let i = 0; i < step.rowNorms.length; i++) {
    const bar = document.createElement("div");
    bar.className = "vis-norms__bar";
    const fill = document.createElement("div");
    fill.className = "vis-norms__fill";
    fill.style.width = `${(step.rowNorms[i] / maxRow) * 100}%`;
    const label = document.createElement("span");
    label.className = "vis-norms__label";
    label.textContent = `row ${i}: ${step.rowNorms[i].toFixed(2)}`;
    if (step.topRows && step.topRows.includes(i)) {
      fill.style.background = "var(--accent)";
      label.style.color = "var(--accent)";
    }
    bar.appendChild(fill);
    bar.appendChild(label);
    rowDiv.appendChild(bar);
  }
  wrap.appendChild(rowDiv);

  return wrap;
}

function renderErrorStep(step) {
  const wrap = document.createElement("div");
  wrap.className = "vis-error";
  const err = diff(step.A, step.Ahat);
  const fn = frobNorm(err);
  const pct = fn / (frobNorm(step.A) + 1e-12) * 100;
  wrap.innerHTML = `
    <div class="vis-error__metric">
      <span class="vis-error__key">Относительная ошибка:</span>
      <span class="vis-error__val">${pct.toFixed(2)}%</span>
    </div>
    <div class="vis-error__metric">
      <span class="vis-error__key">Frobenius норма:</span>
      <span class="vis-error__val">${fn.toFixed(4)}</span>
    </div>
  `;

  // Visual scale bar
  const scaleWrap = document.createElement("div");
  scaleWrap.className = "vis-error__scale";
  const scaleBar = document.createElement("div");
  scaleBar.style.cssText = "height:10px;border-radius:5px;background:linear-gradient(to right, #4ade80, #a3e635, #facc15, #fb923c, #ef4444);position:relative;overflow:hidden";
  const marker = document.createElement("div");
  const pos = Math.min(1, Math.max(0, pct / 50));
  marker.style.cssText = `position:absolute;top:-3px;left:${pos * 100}%;width:4px;height:16px;background:#fff;border-radius:2px;transform:translateX(-50%);box-shadow:0 0 6px rgba(255,255,255,0.8);transition:left 0.4s ease`;
  scaleBar.appendChild(marker);
  scaleWrap.appendChild(scaleBar);

  const labelsRow = document.createElement("div");
  labelsRow.style.cssText = "display:flex;justify-content:space-between;font-size:0.6rem;color:var(--muted);margin-top:0.15rem";
  labelsRow.innerHTML = `<span>отлично<br><span style="font-size:0.5rem;opacity:0.6">0–1%</span></span><span>хорошо<br><span style="font-size:0.5rem;opacity:0.6">1–5%</span></span><span>удовл.<br><span style="font-size:0.5rem;opacity:0.6">5–15%</span></span><span>плохо<br><span style="font-size:0.5rem;opacity:0.6">15–30%</span></span><span>критично<br><span style="font-size:0.5rem;opacity:0.6">&gt;30%</span></span>`;
  scaleWrap.appendChild(labelsRow);
  wrap.appendChild(scaleWrap);
  return wrap;
}

function renderStep(container, step, index) {
  const stepEl = document.createElement("div");
  stepEl.className = "vis-step";
  stepEl.style.animationDelay = `${index * 0.15}s`;

  let content;
  switch (step.type) {
    case "matrix":
      content = renderMatrixStep(step);
      break;
    case "matrices_row":
      content = renderMatricesRowStep(step);
      break;
    case "eigenvalues":
      content = renderEigenvaluesStep(step);
      break;
    case "norms":
      content = renderNormsStep(step);
      break;
    case "error":
      content = renderErrorStep(step);
      break;
    default:
      content = document.createElement("div");
      content.textContent = "Unknown step type";
  }

  stepEl.appendChild(content);
  container.appendChild(stepEl);
  return stepEl;
}

function renderArrow(container, steps, arrowIndex) {
  const arrowStep = steps[arrowIndex];
  const arrowEl = document.createElement("div");
  arrowEl.className = "vis-arrow";
  arrowEl.style.animationDelay = `${arrowIndex * 0.15 + 0.1}s`;

  const color = opColor(arrowStep.op);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "vis-arrow__svg");
  svg.setAttribute("viewBox", "0 0 60 50");
  svg.setAttribute("preserveAspectRatio", "none");

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", "6");
  line.setAttribute("y1", "25");
  line.setAttribute("x2", "46");
  line.setAttribute("y2", "25");
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", "2.5");
  line.setAttribute("stroke-linecap", "round");
  line.setAttribute("class", "vis-arrow__line");

  const head = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  head.setAttribute("points", "42,20 52,25 42,30");
  head.setAttribute("fill", color);
  head.setAttribute("class", "vis-arrow__head");

  svg.appendChild(line);
  svg.appendChild(head);

  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("cx", "6");
  dot.setAttribute("cy", "25");
  dot.setAttribute("r", "3.5");
  dot.setAttribute("fill", color);
  dot.setAttribute("class", "vis-arrow__dot");
  svg.appendChild(dot);

  arrowEl.appendChild(svg);

  const label = document.createElement("div");
  label.className = "vis-arrow__label";
  label.style.setProperty("--op-color", color);
  label.textContent = arrowStep.label;
  label.title = "Нажмите для подробностей";
  label.addEventListener("click", (e) => {
    e.stopPropagation();
    showArrowDetail(arrowStep, steps, arrowIndex);
  });
  arrowEl.appendChild(label);

  container.appendChild(arrowEl);
  return arrowEl;
}

// ── Arrow detail — live animated sandbox ──

function renderSandboxMatrix(container, label, matrix, scale) {
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lbl = document.createElement("div");
  lbl.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lbl.textContent = label;
  wrap.appendChild(lbl);
  const host = document.createElement("div");
  renderMatrixBlock(host, "", matrix, { scale: scale || minMax(matrix) });
  wrap.appendChild(host);
  container.appendChild(wrap);
  return host;
}

function makeOpSign(text) {
  const el = document.createElement("span");
  el.style.cssText = "font-size:1.4rem;font-weight:700;color:var(--muted);padding:0 0.3rem";
  el.textContent = text;
  return el;
}

function makeLabel(text) {
  const el = document.createElement("div");
  el.style.cssText = "font-size:0.82rem;color:var(--accent);font-weight:600;text-align:center;margin:0.5rem 0 0.25rem";
  el.textContent = text;
  return el;
}

function animateCellFill(host, finalM, rows, cols, onDone) {
  const total = rows * cols;
  const working = zeros(rows, cols);
  const s = minMax(finalM);
  let idx = 0;
  let iv;

  function tick() {
    if (idx >= total) { clearInterval(iv); if (onDone) onDone(); return; }
    const ci = Math.floor(idx / cols);
    const cj = idx % cols;
    working[ci][cj] = finalM[ci][cj];
    idx++;
    host.innerHTML = "";
    renderMatrixBlock(host, "", working, { scale: s });
    const cells = host.querySelectorAll(".cell");
    const justFilled = idx - 1;
    if (cells[justFilled]) {
      cells[justFilled].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
      cells[justFilled].style.transition = "box-shadow 0.25s, transform 0.15s";
      cells[justFilled].style.transform = "scale(1.12)";
      setTimeout(() => { if (cells[justFilled]) cells[justFilled].style.transform = "scale(1)"; }, 180);
    }
  }

  function start() {
    idx = 0;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) working[i][j] = 0;
    clearInterval(iv);
    host.innerHTML = "";
    renderMatrixBlock(host, "", working, { scale: s });
    iv = setInterval(tick, 420);
  }

  start();
  return { restart: start };
}

function addRepeatBtn(container, onRestart) {
  const btn = document.createElement("button");
  btn.className = "live-repeat";
  btn.textContent = "⟳ Повторить";
  btn.style.cssText = "margin-top:0.6rem;padding:0.3rem 0.9rem;font-size:0.78rem;font-weight:600;color:var(--accent);background:rgba(91,156,246,0.12);border:1px solid var(--accent);border-radius:8px;cursor:pointer;transition:background 0.15s";
  btn.addEventListener("mouseenter", () => btn.style.background = "rgba(91,156,246,0.25)");
  btn.addEventListener("mouseleave", () => btn.style.background = "rgba(91,156,246,0.12)");
  btn.addEventListener("click", () => { if (onRestart) onRestart(); });
  container.appendChild(btn);
}

function liveProduct(body, data) {
  const A = data.left, B = data.right, R = data.result;
  const mA = A.length, p = A[0].length, nB = B[0].length;
  const sA = minMax(A), sB = minMax(B), sR = minMax(R);

  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.5rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.5rem";
  const wrapA = document.createElement("div");
  wrapA.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblA = document.createElement("div");
  lblA.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblA.textContent = data.leftLabel;
  wrapA.appendChild(lblA);
  const hostA = document.createElement("div");
  renderMatrixBlock(hostA, "", A, { scale: sA });
  wrapA.appendChild(hostA);
  row.appendChild(wrapA);
  row.appendChild(makeOpSign("·"));
  const wrapB = document.createElement("div");
  wrapB.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblB = document.createElement("div");
  lblB.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblB.textContent = data.rightLabel;
  wrapB.appendChild(lblB);
  const hostB = document.createElement("div");
  renderMatrixBlock(hostB, "", B, { scale: sB });
  wrapB.appendChild(hostB);
  row.appendChild(wrapB);
  row.appendChild(makeOpSign("="));
  const hostR = document.createElement("div");
  hostR.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblR = document.createElement("div");
  lblR.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblR.textContent = data.resultLabel;
  hostR.appendChild(lblR);
  const rHost = document.createElement("div");
  const working = zeros(mA, nB);
  renderMatrixBlock(rHost, "", working, { scale: sR });
  hostR.appendChild(rHost);
  row.appendChild(hostR);
  body.appendChild(row);

  const trace = document.createElement("div");
  trace.style.cssText = "font-size:0.85rem;color:var(--text);text-align:center;font-family:monospace;min-height:2rem;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin:0.3rem 0";
  body.appendChild(trace);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center";
  body.appendChild(info);

  let abort = false, paused = false, resumeFn = null, timeoutId = null;
  let ciState = 0, cjState = 0;
  let cellK = 0, cellSum = 0;

  function updateCell(rowIdx, colIdx, val) {
    const cells = rHost.querySelectorAll(".cell");
    const idx = rowIdx * nB + colIdx;
    if (!cells[idx]) return;
    const t = sR.mx > sR.mn ? (val - sR.mn) / (sR.mx - sR.mn) : 0.5;
    const c = viridis(t);
    cells[idx].style.background = viridisRgb(c);
    cells[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
    cells[idx].textContent = val.toFixed(2);
  }

  function schedule(fn, delay, ...args) {
    timeoutId = setTimeout(() => {
      if (abort) return;
      if (paused) { resumeFn = () => schedule(fn, 1, ...args); return; }
      fn(...args);
    }, delay);
  }

  function computeCell(i, j, cb) {
    if (abort) { cb(); return; }
    let sum = 0;
    let k = 0;
    const cellsA = hostA.querySelectorAll(".cell");
    const cellsB = hostB.querySelectorAll(".cell");
    const rIdx = i * nB + j;

    function unhighlight() {
      for (let kk = 0; kk < p; kk++) {
        const aIdx = i * p + kk;
        const bIdx = kk * nB + j;
        if (cellsA[aIdx]) { cellsA[aIdx].style.boxShadow = ""; cellsA[aIdx].style.outline = ""; }
        if (cellsB[bIdx]) { cellsB[bIdx].style.boxShadow = ""; cellsB[bIdx].style.outline = ""; }
      }
    }

    function stepK() {
      if (abort) { cb(); return; }
      if (k >= p) {
        unhighlight();
        const cellsR = rHost.querySelectorAll(".cell");
        if (cellsR[rIdx]) {
          cellsR[rIdx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
          setTimeout(() => { if (cellsR[rIdx]) cellsR[rIdx].style.boxShadow = ""; }, 600);
        }
        trace.innerHTML = `${data.resultLabel}<sub>${i}${j}</sub> = ${trace.dataset.expr || ""} = <b>${sum.toFixed(2)}</b>`;
        cellSum = sum; cellK = k;
        cb();
        return;
      }
      const aIdx = i * p + k;
      const bIdx = k * nB + j;
      if (cellsA[aIdx]) { cellsA[aIdx].style.boxShadow = "inset 0 0 0 3px var(--good), 0 0 14px rgba(74,222,128,0.5)"; }
      if (cellsB[bIdx]) { cellsB[bIdx].style.boxShadow = "inset 0 0 0 3px var(--good), 0 0 14px rgba(74,222,128,0.5)"; }
      const term = A[i][k] * B[k][j];
      sum += term;
      const expr = trace.dataset.expr || "";
      const newExpr = expr + (k === 0 ? "" : " + ") + `${A[i][k].toFixed(2)}<sub>[${i}][${k}]</sub>·${B[k][j].toFixed(2)}<sub>[${k}][${j}]</sub>`;
      trace.dataset.expr = newExpr;
      trace.innerHTML = `${data.resultLabel}<sub>${i}${j}</sub> = ${newExpr}`;
      info.textContent = `Шаг ${k+1}/${p}: ${data.leftLabel}[${i}][${k}]×${data.rightLabel}[${k}][${j}] = ${A[i][k].toFixed(2)} × ${B[k][j].toFixed(2)} = ${term.toFixed(2)} (сумма: ${sum.toFixed(2)})`;
      cellK = ++k; cellSum = sum;
      schedule(stepK, 500);
    }
    trace.innerHTML = "";
    trace.dataset.expr = "";
    cellK = 0; cellSum = 0;
    stepK();
  }

  function fillNext() {
    if (abort) return;
    if (ciState >= mA) { info.textContent = `Готово: ${data.resultLabel}`; return; }
    const i = ciState, j = cjState;
    computeCell(i, j, () => {
      updateCell(i, j, cellSum);
      working[i][j] = cellSum;
      cjState++;
      if (cjState >= nB) { cjState = 0; ciState++; }
      schedule(fillNext, 300);
    });
  }

  function start() {
    ciState = 0; cjState = 0; abort = false; paused = false; resumeFn = null;
    if (timeoutId) clearTimeout(timeoutId);
    for (let i = 0; i < mA; i++) for (let j = 0; j < nB; j++) working[i][j] = 0;
    rHost.innerHTML = "";
    renderMatrixBlock(rHost, "", working, { scale: sR });
    trace.innerHTML = "";
    info.textContent = "";
    fillNext();
  }

  const pauseBtn = document.createElement("button");
  pauseBtn.textContent = "⏸ Пауза";
  pauseBtn.style.cssText = "margin:0.5rem 0.3rem 0;padding:0.3rem 0.9rem;font-size:0.78rem;font-weight:600;color:var(--accent);background:rgba(91,156,246,0.12);border:1px solid var(--accent);border-radius:8px;cursor:pointer";
  pauseBtn.addEventListener("click", () => {
    if (!paused) {
      paused = true;
      pauseBtn.textContent = "▶ Продолжить";
    } else {
      paused = false;
      pauseBtn.textContent = "⏸ Пауза";
      const fn = resumeFn; resumeFn = null;
      if (fn) fn();
    }
  });

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:0.3rem;justify-content:center;flex-wrap:wrap";
  btnRow.appendChild(pauseBtn);
  addRepeatBtn(btnRow, () => { paused = false; pauseBtn.textContent = "⏸ Пауза"; start(); });
  body.appendChild(btnRow);

  start();
}

function liveTriple(body, data) {
  const A = data.a, B = data.b, C = data.c, R = data.result;
  const mA = A.length, p = A[0].length, q = B[0].length, nC = C[0].length;
  const sA = minMax(A), sB = minMax(B), sC = minMax(C), sR = minMax(R);
  const temp = dot(A, B);
  const sT = minMax(temp);

  let abort = false, paused = false, resumeFn = null, timeoutId = null;

  function updateCell(host, idx, val, scale) {
    const cells = host.querySelectorAll(".cell");
    if (!cells[idx]) return;
    const t = scale.mx > scale.mn ? (val - scale.mn) / (scale.mx - scale.mn) : 0.5;
    const c = viridis(t);
    cells[idx].style.background = viridisRgb(c);
    cells[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
    cells[idx].textContent = val.toFixed(2);
  }

  function makeProduct(hostA, hostB, hostR, hostLabel, leftM, rightM, rows, mid, cols, phaseLabel, scaleR, working, onComplete) {
    const phase = document.createElement("div");
    phase.style.cssText = "font-size:0.82rem;color:var(--accent);font-weight:600;text-align:center;margin:0.3rem 0";
    phase.textContent = phaseLabel;
    body.appendChild(phase);

    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:0.5rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.3rem";
    row.appendChild(hostA);
    row.appendChild(makeOpSign("·"));
    row.appendChild(hostB);
    row.appendChild(makeOpSign("="));
    const wrapR = document.createElement("div");
    wrapR.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
    const lblR = document.createElement("div");
    lblR.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
    lblR.textContent = hostLabel;
    wrapR.appendChild(lblR);
    wrapR.appendChild(hostR);
    row.appendChild(wrapR);
    body.appendChild(row);

    const trace = document.createElement("div");
    trace.style.cssText = "font-size:0.85rem;color:var(--text);text-align:center;font-family:monospace;min-height:2rem;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin:0.3rem 0";
    body.appendChild(trace);

    const info = document.createElement("div");
    info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
    body.appendChild(info);

    let cellSum = 0, cellK = 0;
    let ci = 0, cj = 0;

    function schedule(fn, delay, ...args) {
      timeoutId = setTimeout(() => {
        if (abort) return;
        if (paused) { resumeFn = () => schedule(fn, 1, ...args); return; }
        fn(...args);
      }, delay);
    }

    function computeCell(i, j, cb) {
      if (abort) { cb(); return; }
      let sum = 0; let k = 0;
      const cellsA = hostA.querySelectorAll(".cell");
      const cellsB = hostB.querySelectorAll(".cell");
      const rIdx = i * cols + j;

      function unhighlight() {
        for (let kk = 0; kk < mid; kk++) {
          const aIdx = i * mid + kk, bIdx = kk * cols + j;
          if (cellsA[aIdx]) cellsA[aIdx].style.boxShadow = "";
          if (cellsB[bIdx]) cellsB[bIdx].style.boxShadow = "";
        }
      }

      function stepK() {
        if (abort) { cb(); return; }
        if (k >= mid) {
          unhighlight();
          const cellsR = hostR.querySelectorAll(".cell");
          if (cellsR[rIdx]) { cellsR[rIdx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 14px rgba(91,156,246,0.6)"; setTimeout(() => { if (cellsR[rIdx]) cellsR[rIdx].style.boxShadow = ""; }, 600); }
          trace.innerHTML = `${hostLabel}<sub>${i}${j}</sub> = ${trace.dataset.expr || ""} = <b>${sum.toFixed(2)}</b>`;
          cellSum = sum; cellK = k;
          cb();
          return;
        }
        const aIdx = i * mid + k, bIdx = k * cols + j;
        if (cellsA[aIdx]) cellsA[aIdx].style.boxShadow = "inset 0 0 0 3px var(--good), 0 0 14px rgba(74,222,128,0.5)";
        if (cellsB[bIdx]) cellsB[bIdx].style.boxShadow = "inset 0 0 0 3px var(--good), 0 0 14px rgba(74,222,128,0.5)";
        const term = leftM[i][k] * rightM[k][j];
        sum += term;
        const expr = trace.dataset.expr || "";
        const newExpr = expr + (k === 0 ? "" : " + ") + `${leftM[i][k].toFixed(2)}<sub>[${i}][${k}]</sub>·${rightM[k][j].toFixed(2)}<sub>[${k}][${j}]</sub>`;
        trace.dataset.expr = newExpr;
        trace.innerHTML = `${hostLabel}<sub>${i}${j}</sub> = ${newExpr}`;
        info.textContent = `Шаг ${k+1}/${mid}: [${i}][${k}]×[${k}][${j}] = ${leftM[i][k].toFixed(2)} × ${rightM[k][j].toFixed(2)} = ${term.toFixed(2)} (сумма: ${sum.toFixed(2)})`;
        cellK = ++k; cellSum = sum;
        schedule(stepK, 500);
      }
      trace.innerHTML = ""; trace.dataset.expr = ""; cellK = 0; cellSum = 0;
      stepK();
    }

    function fillNext() {
      if (abort) return;
      if (ci >= rows) { info.textContent = "Готово"; if (onComplete) onComplete(); return; }
      const i = ci, j = cj;
      computeCell(i, j, () => {
        updateCell(hostR, i * cols + j, cellSum, scaleR);
        if (working) working[i][j] = cellSum;
        cj++;
        if (cj >= cols) { cj = 0; ci++; }
        schedule(fillNext, 300);
      });
    }
    fillNext();

    function resetState() { ci = 0; cj = 0; }
    return { resetState };
  }

  function buildAll() {
    const wrapA1 = document.createElement("div");
    wrapA1.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
    const lblA1 = document.createElement("div");
    lblA1.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
    lblA1.textContent = data.aLabel;
    wrapA1.appendChild(lblA1);
    const hostA1 = document.createElement("div");
    renderMatrixBlock(hostA1, "", A, { scale: sA });
    wrapA1.appendChild(hostA1);

    const wrapB1 = document.createElement("div");
    wrapB1.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
    const lblB1 = document.createElement("div");
    lblB1.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
    lblB1.textContent = data.bLabel;
    wrapB1.appendChild(lblB1);
    const hostB1 = document.createElement("div");
    renderMatrixBlock(hostB1, "", B, { scale: sB });
    wrapB1.appendChild(hostB1);

    const working1 = zeros(mA, q);
    const hostT = document.createElement("div");
    renderMatrixBlock(hostT, "", working1, { scale: sT });

    makeProduct(wrapA1, wrapB1, hostT, "U·Σ", A, B, mA, p, q, "Фаза 1: U · Σ → промежуточная", sT, working1, () => {
      timeoutId = setTimeout(() => {
        const wrapA2 = document.createElement("div");
        wrapA2.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
        const lblA2 = document.createElement("div");
        lblA2.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
        lblA2.textContent = "U·Σ";
        wrapA2.appendChild(lblA2);
        const hostA2 = document.createElement("div");
        renderMatrixBlock(hostA2, "", temp, { scale: sT });
        wrapA2.appendChild(hostA2);

        const wrapB2 = document.createElement("div");
        wrapB2.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
        const lblB2 = document.createElement("div");
        lblB2.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
        lblB2.textContent = data.cLabel;
        wrapB2.appendChild(lblB2);
        const hostB2 = document.createElement("div");
        renderMatrixBlock(hostB2, "", C, { scale: sC });
        wrapB2.appendChild(hostB2);

        const working2 = zeros(mA, nC);
        const hostR2 = document.createElement("div");
        renderMatrixBlock(hostR2, "", working2, { scale: sR });

        makeProduct(wrapA2, wrapB2, hostR2, data.resultLabel, temp, C, mA, q, nC, "Фаза 2: (U·Σ) · Vᵀ → результат", sR, working2, null);
      }, 400);
    });
  }

  buildAll();

  const pauseBtn = document.createElement("button");
  pauseBtn.textContent = "⏸ Пауза";
  pauseBtn.style.cssText = "margin:0.5rem 0.3rem 0;padding:0.3rem 0.9rem;font-size:0.78rem;font-weight:600;color:var(--accent);background:rgba(91,156,246,0.12);border:1px solid var(--accent);border-radius:8px;cursor:pointer";
  pauseBtn.addEventListener("click", () => {
    if (!paused) {
      paused = true;
      pauseBtn.textContent = "▶ Продолжить";
    } else {
      paused = false;
      pauseBtn.textContent = "⏸ Пауза";
      const fn = resumeFn; resumeFn = null;
      if (fn) fn();
    }
  });

  function rebuild() {
    abort = true; paused = false; resumeFn = null;
    if (timeoutId) clearTimeout(timeoutId);
    setTimeout(() => {
      abort = false; timeoutId = null;
      const toRemove = body.querySelectorAll("div, button");
      for (const el of toRemove) {
        if (el.classList.contains("live-repeat")) continue;
        if (el.parentNode === body) el.remove();
      }
      const existingRepeats = body.querySelectorAll(".live-repeat");
      for (const b of existingRepeats) b.remove();
      buildAll();
      btnRow.appendChild(pauseBtn);
      pauseBtn.textContent = "⏸ Пауза";
      addRepeatBtn(btnRow, rebuild);
    }, 100);
  }

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:0.3rem;justify-content:center;flex-wrap:wrap";
  btnRow.appendChild(pauseBtn);
  addRepeatBtn(btnRow, rebuild);
  body.appendChild(btnRow);
}

function liveAddition(body, data) {
  const A = data.left, B = data.right, R = data.result;
  const m = A.length, n = A[0].length;
  const sA = minMax(A), sR = minMax(R);
  const working = zeros(m, n);

  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.5rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.5rem";
  const wrapA = document.createElement("div");
  wrapA.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblA = document.createElement("div");
  lblA.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblA.textContent = data.leftLabel;
  wrapA.appendChild(lblA);
  const hostA = document.createElement("div");
  renderMatrixBlock(hostA, "", A, { scale: sA });
  wrapA.appendChild(hostA);
  row.appendChild(wrapA);
  row.appendChild(makeOpSign("+"));
  const wrapB = document.createElement("div");
  wrapB.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblB = document.createElement("div");
  lblB.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblB.textContent = data.rightLabel;
  wrapB.appendChild(lblB);
  const hostB = document.createElement("div");
  renderMatrixBlock(hostB, "", B, { scale: sA });
  wrapB.appendChild(hostB);
  row.appendChild(wrapB);
  row.appendChild(makeOpSign("="));
  const hostR = document.createElement("div");
  hostR.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblR = document.createElement("div");
  lblR.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblR.textContent = data.resultLabel;
  hostR.appendChild(lblR);
  const rHost = document.createElement("div");
  renderMatrixBlock(rHost, "", working, { scale: sR });
  hostR.appendChild(rHost);
  row.appendChild(hostR);
  body.appendChild(row);

  const trace = document.createElement("div");
  trace.style.cssText = "font-size:0.85rem;color:var(--text);text-align:center;font-family:monospace;min-height:2rem;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin:0.3rem 0";
  body.appendChild(trace);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center";
  body.appendChild(info);

  const cellsA = hostA.querySelectorAll(".cell");
  const cellsB = hostB.querySelectorAll(".cell");
  const cellsR = rHost.querySelectorAll(".cell");

  let ci = 0, cj = 0, paused = false, resumeFn = null;

  function tick() {
    if (ci >= m) { info.textContent = "Результат готов"; return; }
    const aIdx = ci * n + cj;
    const vB = B.length === 1 ? (B[0] ? B[0][cj] || 0 : 0) : (B[ci] ? B[ci][cj] || 0 : 0);
    const vR = A[ci][cj] + vB;
    if (cellsA[aIdx]) cellsA[aIdx].style.boxShadow = "inset 0 0 0 3px var(--good)";
    if (cellsB[aIdx]) cellsB[aIdx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
    working[ci][cj] = vR;
    const t = sR.mx > sR.mn ? (vR - sR.mn) / (sR.mx - sR.mn) : 0.5;
    const c = viridis(Math.max(0, Math.min(1, t)));
    cellsR[aIdx].style.background = viridisRgb(c);
    cellsR[aIdx].style.color = t > 0.6 ? "#080c14" : "#fff";
    cellsR[aIdx].textContent = vR.toFixed(2);
    cellsR[aIdx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
    trace.innerHTML = `${data.leftLabel}<sub>${ci}${cj}</sub> + ${data.rightLabel}<sub>${ci}${cj}</sub> = ${A[ci][cj].toFixed(2)} + ${vB.toFixed(2)} = <b>${vR.toFixed(2)}</b>`;
    info.textContent = `(${ci},${cj}): ${A[ci][cj].toFixed(2)} + ${vB.toFixed(2)} = ${vR.toFixed(2)}`;
    setTimeout(() => {
      if (cellsA[aIdx]) cellsA[aIdx].style.boxShadow = "";
      if (cellsB[aIdx]) cellsB[aIdx].style.boxShadow = "";
      if (cellsR[aIdx]) cellsR[aIdx].style.boxShadow = "";
    }, 300);
    cj++;
    if (cj >= n) { cj = 0; ci++; }
    scheduleNext();
  }

  function scheduleNext() {
    if (paused) { resumeFn = tick; return; }
    setTimeout(() => { if (!paused) tick(); else resumeFn = tick; }, 400);
  }

  scheduleNext();

  function restartAdd() {
    paused = false; resumeFn = null;
    ci = 0; cj = 0;
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) working[i][j] = 0;
    rHost.innerHTML = "";
    renderMatrixBlock(rHost, "", working, { scale: sR });
    trace.textContent = "";
    info.textContent = "";
    scheduleNext();
  }

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:0.3rem;justify-content:center;flex-wrap:wrap;margin-top:0.5rem";
  const pauseBtn2 = document.createElement("button");
  pauseBtn2.textContent = "⏸ Пауза";
  pauseBtn2.style.cssText = "padding:0.25rem 0.7rem;font-size:0.75rem;cursor:pointer;border:1px solid var(--border);border-radius:4px;background:var(--bg-card);color:var(--text)";
  pauseBtn2.addEventListener("click", () => {
    if (!paused) { paused = true; pauseBtn2.textContent = "▶ Продолжить"; }
    else { paused = false; pauseBtn2.textContent = "⏸ Пауза"; const fn = resumeFn; resumeFn = null; if (fn) fn(); }
  });
  btnRow.appendChild(pauseBtn2);
  addRepeatBtn(btnRow, restartAdd);
  body.appendChild(btnRow);
}

function liveSubtraction(body, data) {
  const A = data.left, B = data.right, R = data.result;
  const m = A.length, n = A[0].length;
  const sA = minMax(A), sR = minMax(R);
  const working = zeros(m, n);

  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.5rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.5rem";
  const wrapA = document.createElement("div");
  wrapA.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblA = document.createElement("div");
  lblA.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblA.textContent = data.leftLabel;
  wrapA.appendChild(lblA);
  const hostA = document.createElement("div");
  renderMatrixBlock(hostA, "", A, { scale: sA });
  wrapA.appendChild(hostA);
  row.appendChild(wrapA);
  row.appendChild(makeOpSign("−"));
  const wrapB = document.createElement("div");
  wrapB.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblB = document.createElement("div");
  lblB.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblB.textContent = data.rightLabel;
  wrapB.appendChild(lblB);
  const hostB = document.createElement("div");
  renderMatrixBlock(hostB, "", B, { scale: sA });
  wrapB.appendChild(hostB);
  row.appendChild(wrapB);
  row.appendChild(makeOpSign("="));
  const hostR = document.createElement("div");
  hostR.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblR = document.createElement("div");
  lblR.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblR.textContent = data.resultLabel;
  hostR.appendChild(lblR);
  const rHost = document.createElement("div");
  renderMatrixBlock(rHost, "", working, { scale: sR });
  hostR.appendChild(rHost);
  row.appendChild(hostR);
  body.appendChild(row);

  const trace = document.createElement("div");
  trace.style.cssText = "font-size:0.85rem;color:var(--text);text-align:center;font-family:monospace;min-height:2rem;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin:0.3rem 0";
  body.appendChild(trace);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center";
  body.appendChild(info);

  const cellsA = hostA.querySelectorAll(".cell");
  const cellsB = hostB.querySelectorAll(".cell");
  const cellsR = rHost.querySelectorAll(".cell");

  let ci = 0, cj = 0, paused = false, resumeFn = null;

  function tick() {
    if (ci >= m) { info.textContent = "Результат готов"; return; }
    const aIdx = ci * n + cj;
    const vA = A[ci][cj], vB = B.length === 1 ? (B[0] ? B[0][cj] || 0 : 0) : (B[ci] ? B[ci][cj] || 0 : 0), vR = vA - vB;
    if (cellsA[aIdx]) cellsA[aIdx].style.boxShadow = "inset 0 0 0 3px var(--good)";
    if (cellsB[aIdx]) cellsB[aIdx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
    working[ci][cj] = vR;
    const t = sR.mx > sR.mn ? (vR - sR.mn) / (sR.mx - sR.mn) : 0.5;
    const c = viridis(Math.max(0, Math.min(1, t)));
    cellsR[aIdx].style.background = viridisRgb(c);
    cellsR[aIdx].style.color = t > 0.6 ? "#080c14" : "#fff";
    cellsR[aIdx].textContent = vR.toFixed(2);
    cellsR[aIdx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
    trace.innerHTML = `${data.leftLabel}<sub>${ci}${cj}</sub> − ${data.rightLabel}<sub>${ci}${cj}</sub> = ${vA.toFixed(2)} − ${vB.toFixed(2)} = <b>${vR.toFixed(2)}</b>`;
    info.textContent = `(${ci},${cj}): ${vA.toFixed(2)} − ${vB.toFixed(2)} = ${vR.toFixed(2)}`;
    setTimeout(() => {
      if (cellsA[aIdx]) cellsA[aIdx].style.boxShadow = "";
      if (cellsB[aIdx]) cellsB[aIdx].style.boxShadow = "";
      if (cellsR[aIdx]) cellsR[aIdx].style.boxShadow = "";
    }, 300);
    cj++;
    if (cj >= n) { cj = 0; ci++; }
    scheduleNext();
  }

  function scheduleNext() {
    if (paused) { resumeFn = tick; return; }
    setTimeout(() => { if (!paused) tick(); else resumeFn = tick; }, 400);
  }

  scheduleNext();

  function restartSub() {
    paused = false; resumeFn = null;
    ci = 0; cj = 0;
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) working[i][j] = 0;
    rHost.innerHTML = "";
    renderMatrixBlock(rHost, "", working, { scale: sR });
    trace.textContent = "";
    info.textContent = "";
    for (const c of cellsA) c.style.boxShadow = "";
    for (const c of cellsB) c.style.boxShadow = "";
    scheduleNext();
  }

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:0.3rem;justify-content:center;flex-wrap:wrap;margin-top:0.5rem";
  const pauseBtn2 = document.createElement("button");
  pauseBtn2.textContent = "⏸ Пауза";
  pauseBtn2.style.cssText = "padding:0.25rem 0.7rem;font-size:0.75rem;cursor:pointer;border:1px solid var(--border);border-radius:4px;background:var(--bg-card);color:var(--text)";
  pauseBtn2.addEventListener("click", () => {
    if (!paused) { paused = true; pauseBtn2.textContent = "▶ Продолжить"; }
    else { paused = false; pauseBtn2.textContent = "⏸ Пауза"; const fn = resumeFn; resumeFn = null; if (fn) fn(); }
  });
  btnRow.appendChild(pauseBtn2);
  addRepeatBtn(btnRow, restartSub);
  body.appendChild(btnRow);
}

function liveMeans(body, data) {
  const vals = data.values;
  const formula = document.createElement("div");
  formula.style.cssText = "font-size:0.85rem;color:var(--accent);text-align:center;font-family:monospace;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin-bottom:0.3rem";
  formula.innerHTML = "μ<sub>j</sub> = (1/m) · Σ<sub>i</sub> A<sub>ij</sub>";
  body.appendChild(formula);

  const d = document.createElement("div");
  d.style.cssText = "display:flex;flex-wrap:wrap;gap:0.6rem;justify-content:center;padding:0.5rem 0";
  const cards = [];
  for (let j = 0; j < vals.length; j++) {
    const card = document.createElement("div");
    card.style.cssText = "background:rgba(0,0,0,0.25);border:1px solid var(--border);border-radius:8px;padding:0.5rem 0.8rem;text-align:center;min-width:70px;opacity:0;transform:translateY(8px);transition:opacity 0.5s,transform 0.5s";
    const valStr = Number.isInteger(vals[j]) ? vals[j] : vals[j].toFixed(4);
    card.innerHTML = `<div style="font-size:0.65rem;color:var(--muted)">μ<sub>${j+1}</sub></div><div style="font-size:1rem;font-weight:700;color:var(--accent);margin-top:0.15rem">${valStr}</div><div style="font-size:0.6rem;color:var(--muted);margin-top:0.1rem">среднее столбца ${j+1}</div>`;
    d.appendChild(card);
    cards.push(card);
  }
  body.appendChild(d);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
  body.appendChild(info);

  let ci = 0;
  function reveal(cb) {
    ci = 0;
    const iv = setInterval(() => {
      if (ci >= cards.length) { clearInterval(iv); if (cb) cb(); return; }
      cards[ci].style.opacity = "1";
      cards[ci].style.transform = "translateY(0)";
      info.textContent = `μ${ci+1} = ${Number.isInteger(vals[ci]) ? vals[ci] : vals[ci].toFixed(4)}`;
      ci++;
    }, 400);
    return iv;
  }
  reveal();

  addRepeatBtn(body, () => {
    for (const c of cards) { c.style.opacity = "0"; c.style.transform = "translateY(8px)"; }
    info.textContent = "";
    reveal();
  });
}

function liveEigenvalues(body, data) {
  const vals = data.values;
  const formula = document.createElement("div");
  formula.style.cssText = "font-size:0.85rem;color:var(--accent);text-align:center;font-family:monospace;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin-bottom:0.3rem";
  formula.innerHTML = "λ<sub>i</sub> = σ<sub>i</sub>²";
  body.appendChild(formula);

  const d = document.createElement("div");
  d.style.cssText = "display:flex;flex-wrap:wrap;gap:0.6rem;justify-content:center;padding:0.5rem 0";
  const cards = [];
  for (const ev of vals) {
    const card = document.createElement("div");
    card.style.cssText = "background:rgba(0,0,0,0.25);border:1px solid var(--border);border-radius:8px;padding:0.5rem 0.8rem;text-align:center;opacity:0;transform:scale(0.8);transition:opacity 0.5s,transform 0.5s";
    const v = ev.value;
    card.innerHTML = `<div style="font-size:0.7rem;color:var(--muted)">${ev.label}</div><div style="font-size:1.1rem;font-weight:700;color:var(--accent2)">${Number.isInteger(v) ? v : v.toFixed(4)}</div>`;
    d.appendChild(card);
    cards.push(card);
  }
  body.appendChild(d);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
  body.appendChild(info);

  let ci = 0;
  function reveal() {
    ci = 0;
    for (const c of cards) { c.style.opacity = "0"; c.style.transform = "scale(0.8)"; }
    const iv = setInterval(() => {
      if (ci >= cards.length) { clearInterval(iv); return; }
      cards[ci].style.opacity = "1";
      cards[ci].style.transform = "scale(1)";
      info.textContent = `${vals[ci].label} = ${Number.isInteger(vals[ci].value) ? vals[ci].value : vals[ci].value.toFixed(4)}`;
      const glow = cards[ci].querySelector("div:last-child");
      if (glow) { glow.style.transition = "text-shadow 0.4s"; glow.style.textShadow = "0 0 12px rgba(167,139,250,0.6)"; setTimeout(() => { glow.style.textShadow = "none"; }, 400); }
      ci++;
    }, 400);
    return iv;
  }
  let iv = reveal();

  addRepeatBtn(body, () => {
    clearInterval(iv);
    iv = reveal();
  });
}

function liveValueMap(body, data) {
  const from = data.from, to = data.to;
  const formula = document.createElement("div");
  formula.style.cssText = "font-size:0.85rem;color:var(--accent);text-align:center;font-family:monospace;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin-bottom:0.3rem";
  formula.innerHTML = "σ<sub>i</sub> = √λ<sub>i</sub>";
  body.appendChild(formula);

  const d = document.createElement("div");
  d.style.cssText = "display:flex;flex-wrap:wrap;gap:0.6rem;justify-content:center;padding:0.5rem 0";
  const cards = [];
  for (let i = 0; i < from.length && i < to.length; i++) {
    const card = document.createElement("div");
    card.style.cssText = "background:rgba(0,0,0,0.25);border:1px solid var(--border);border-radius:8px;padding:0.5rem 0.8rem;text-align:center;opacity:0;transition:opacity 0.5s";
    card.innerHTML = `<div style="font-size:0.7rem;color:var(--muted)">${data.fromLabel}<sub>${i+1}</sub></div><div style="font-size:1rem;font-weight:700;color:var(--text)">${from[i].toFixed(4)}</div><div style="font-size:1.1rem;color:var(--accent);margin:0.15rem 0;font-weight:700">→ √ →</div><div style="font-size:1rem;font-weight:700;color:var(--good)">${to[i].toFixed(4)}</div><div style="font-size:0.6rem;color:var(--muted);margin-top:0.1rem">√${from[i].toFixed(2)} = ${to[i].toFixed(4)}</div>`;
    d.appendChild(card);
    cards.push(card);
  }
  body.appendChild(d);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
  body.appendChild(info);

  let ci = 0;
  function reveal() {
    ci = 0;
    for (const c of cards) c.style.opacity = "0";
    const iv = setInterval(() => {
      if (ci >= cards.length) { clearInterval(iv); return; }
      cards[ci].style.opacity = "1";
      info.textContent = `σ${ci+1} = √${from[ci].toFixed(4)} = ${to[ci].toFixed(4)}`;
      ci++;
    }, 450);
    return iv;
  }
  let iv = reveal();

  addRepeatBtn(body, () => {
    clearInterval(iv);
    iv = reveal();
  });
}

function liveNorms(body, data) {
  const formula = document.createElement("div");
  formula.style.cssText = "font-size:0.85rem;color:var(--accent);text-align:center;font-family:monospace;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin-bottom:0.3rem";
  formula.innerHTML = "||col<sub>j</sub>||² = Σ<sub>i</sub> A<sub>ij</sub>² &nbsp;&nbsp; ||row<sub>i</sub>||² = Σ<sub>j</sub> A<sub>ij</sub>²";
  body.appendChild(formula);

  const fills = [];
  const labels = [];
  function renderBars(arr, label, top, prefix) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "margin:0.25rem 0";
    const title = document.createElement("div");
    title.style.cssText = "font-size:0.72rem;color:var(--muted);margin-bottom:0.2rem";
    title.textContent = label;
    wrap.appendChild(title);
    const max = Math.max(...arr, 1e-12);
    for (let i = 0; i < arr.length; i++) {
      const bar = document.createElement("div");
      bar.style.cssText = "display:flex;align-items:center;gap:0.3rem;margin:0.1rem 0";
      const fill = document.createElement("div");
      const finalPct = (arr[i]/max)*100;
      fill.style.cssText = `height:14px;width:0%;background:${top.includes(i)?"var(--accent)":"var(--border)"};border-radius:4px;transition:width 0.15s linear;min-width:0px`;
      const lbl = document.createElement("span");
      lbl.style.cssText = "font-size:0.65rem;color:var(--muted);white-space:nowrap;min-width:3.5rem";
      lbl.textContent = `${i}: ${arr[i].toFixed(2)}`;
      bar.appendChild(fill);
      bar.appendChild(lbl);
      wrap.appendChild(bar);
      fills.push({ el: fill, target: finalPct, idx: i, val: arr[i], prefix });
    }
    return wrap;
  }
  const d = document.createElement("div");
  d.style.cssText = "display:flex;gap:1rem;flex-wrap:wrap;justify-content:center";
  d.appendChild(renderBars(data.colNorms, "||colⱼ||²", data.topCols || [], "col"));
  d.appendChild(renderBars(data.rowNorms, "||rowᵢ||²", data.topRows || [], "row"));
  body.appendChild(d);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
  body.appendChild(info);

  let fi = 0;
  function startBars() {
    fi = 0;
    for (const f of fills) f.el.style.width = "0%";
    const iv = setInterval(() => {
      if (fi >= fills.length) { clearInterval(iv); info.textContent = "Готово"; return; }
      const f = fills[fi];
      f.el.style.width = f.target + "%";
      f.el.style.transition = "width 0.35s ease-out, box-shadow 0.3s";
      f.el.style.boxShadow = "0 0 8px rgba(91,156,246,0.5)";
      setTimeout(() => { f.el.style.boxShadow = "none"; }, 400);
      info.textContent = `${f.prefix}[${f.idx}]² = ${f.val.toFixed(2)}`;
      fi++;
    }, 300);
    return iv;
  }
  let iv = startBars();

  addRepeatBtn(body, () => {
    clearInterval(iv);
    iv = startBars();
  });
}

function liveSelection(body, data) {
  const info = document.createElement("div");
  info.style.cssText = "font-size:0.82rem;color:var(--text);text-align:center;margin-bottom:0.4rem";
  info.innerHTML = `Строки: [${data.topRows.join(", ")}] &nbsp; Столбцы: [${data.topCols.join(", ")}]`;
  body.appendChild(info);
  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.8rem;align-items:center;justify-content:center;flex-wrap:wrap";
  const hosts = [];
  for (const { label, M, s } of [{ label: "C", M: data.C, s: minMax(data.C) }, { label: "R", M: data.R, s: minMax(data.R) }, { label: "W", M: data.W, s: minMax(data.W) }]) {
    const host = renderSandboxMatrix(row, label, zeros(M.length, M[0].length), s);
    hosts.push({ host, M, rows: M.length, cols: M[0].length });
  }
  body.appendChild(row);
  const info2 = document.createElement("div");
  info2.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;margin-top:0.3rem";
  info2.textContent = "Заполняем матрицы...";
  body.appendChild(info2);
  const restarts = [];
  let done = 0;
  for (const h of hosts) {
    const anim = animateCellFill(h.host, h.M, h.rows, h.cols, () => { done++; if (done === hosts.length) info2.textContent = "Готово"; });
    restarts.push(anim.restart);
  }
  addRepeatBtn(body, () => {
    info2.textContent = "Заполняем матрицы...";
    done = 0;
    for (const r of restarts) r();
  });
}

function liveInit(body, data) {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.8rem;align-items:center;justify-content:center;flex-wrap:wrap";
  const hosts = [];
  for (const m of data.matrices) {
    const s = minMax(m.matrix);
    const host = renderSandboxMatrix(row, m.label, zeros(m.matrix.length, m.matrix[0].length), s);
    hosts.push({ host, M: m.matrix, rows: m.matrix.length, cols: m.matrix[0].length });
  }
  body.appendChild(row);
  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;margin-top:0.3rem";
  info.textContent = "Инициализируем случайные матрицы...";
  body.appendChild(info);
  const restarts = [];
  let done = 0;
  for (const h of hosts) {
    const anim = animateCellFill(h.host, h.M, h.rows, h.cols, () => { done++; if (done === hosts.length) info.textContent = "Готово"; });
    restarts.push(anim.restart);
  }
  addRepeatBtn(body, () => {
    info.textContent = "Инициализируем случайные матрицы...";
    done = 0;
    for (const r of restarts) r();
  });
}

function liveIterationHistory(body, data) {
  const history = data.history;
  const scaleA = data.scale || minMax(data.history[0] ? data.history[0].Ahat : [[0]]);
  const maxFrob = Math.max(...history.map(h => h.frob), 1e-12);
  const label = data.label || "Итерация";

  let idx = 0, paused = false, resumeFn = null, timeoutId = null, abort = false;

  function updateFrob(frob) {
    const pct = Math.min(frob / maxFrob * 100, 100);
    info.innerHTML = `${label} <b>${history[idx] ? history[idx].i : 0}</b> &nbsp;|&nbsp; Ошибка: ${frob.toFixed(4)}`;
    barFill.style.width = pct + "%";
    barFill.style.background = `hsl(${120 - pct * 1.2}, 80%, 50%)`;
  }

  function pauseCheck(fn) {
    if (paused) { resumeFn = fn; return true; }
    return false;
  }

  function showIter() {
    if (abort) return;
    if (pauseCheck(showIter)) return;
    if (idx >= history.length) { info.innerHTML += " &nbsp;✅ Готово"; return; }
    const entry = history[idx];
    updateFrob(entry.frob);
    matrixHost.innerHTML = "";
    renderMatrixBlock(matrixHost, "", entry.Ahat, { scale: scaleA });
    const cells = matrixHost.querySelectorAll(".cell");
    for (let ci = 0; ci < cells.length; ci++) {
      cells[ci].style.animation = "none";
      cells[ci].offsetHeight;
      cells[ci].style.animation = `cellPulse 0.4s ease ${ci * 0.02}s both`;
    }
    idx++;
    timeoutId = setTimeout(showIter, 500);
  }

  const formula = document.createElement("div");
  formula.style.cssText = "font-size:0.85rem;color:var(--accent);text-align:center;font-family:monospace;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin-bottom:0.3rem";
  formula.innerHTML = `Отслеживание сходимости: ${history.length} шагов`;
  body.appendChild(formula);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.88rem;color:var(--text);text-align:center;font-family:monospace;padding:0.3rem;margin-bottom:0.3rem;min-height:1.5rem";
  body.appendChild(info);

  const barWrap = document.createElement("div");
  barWrap.style.cssText = "height:8px;background:rgba(255,255,255,0.1);border-radius:4px;margin:0.2rem 0 0.5rem;overflow:hidden";
  const barFill = document.createElement("div");
  barFill.style.cssText = "height:100%;width:100%;border-radius:4px;transition:width 0.3s";
  barWrap.appendChild(barFill);
  body.appendChild(barWrap);

  const matrixHost = document.createElement("div");
  matrixHost.style.cssText = "display:flex;justify-content:center";
  body.appendChild(matrixHost);

  showIter();

  const pauseBtn = document.createElement("button");
  pauseBtn.textContent = "⏸ Пауза";
  pauseBtn.style.cssText = "margin:0.5rem 0.3rem 0;padding:0.3rem 0.9rem;font-size:0.78rem;font-weight:600;color:var(--accent);background:rgba(91,156,246,0.12);border:1px solid var(--accent);border-radius:8px;cursor:pointer";
  pauseBtn.addEventListener("click", () => {
    if (!paused) {
      paused = true; if (timeoutId) clearTimeout(timeoutId);
      pauseBtn.textContent = "▶ Продолжить";
    } else {
      paused = false;
      pauseBtn.textContent = "⏸ Пауза";
      const fn = resumeFn; resumeFn = null;
      if (fn) timeoutId = setTimeout(fn, 50);
    }
  });

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:0.3rem;justify-content:center;flex-wrap:wrap";
  btnRow.appendChild(pauseBtn);
  addRepeatBtn(btnRow, () => {
    abort = true; if (timeoutId) clearTimeout(timeoutId);
    setTimeout(() => {
      abort = false; paused = false; resumeFn = null; timeoutId = null;
      idx = 0;
      info.innerHTML = "";
      barFill.style.width = "100%";
      matrixHost.innerHTML = "";
      showIter();
    }, 50);
  });
  body.appendChild(btnRow);
}

function liveNote(body, data) {
  if (data.matrices) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:1rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.5rem";
    const anims = [];
    for (const m of data.matrices) {
      const host = renderSandboxMatrix(row, m.label, zeros(m.matrix.length, m.matrix[0].length), minMax(m.matrix));
      anims.push({ host, M: m.matrix, rows: m.matrix.length, cols: m.matrix[0].length });
    }
    body.appendChild(row);
    const info = document.createElement("div");
    info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;margin-bottom:0.3rem";
    info.textContent = "Показываем матрицы...";
    body.appendChild(info);
    let done = 0;
    let restarts = [];
    for (const a of anims) {
      const anim = animateCellFill(a.host, a.M, a.rows, a.cols, () => { done++; if (done === anims.length) info.textContent = "Готово"; });
      restarts.push(anim.restart);
    }
    const btn = document.createElement("button");
    btn.textContent = "⟳ Повторить";
    btn.style.cssText = "margin-top:0.3rem;margin-bottom:0.5rem;padding:0.3rem 0.9rem;font-size:0.78rem;font-weight:600;color:var(--accent);background:rgba(91,156,246,0.12);border:1px solid var(--accent);border-radius:8px;cursor:pointer;transition:background 0.15s";
    btn.addEventListener("mouseenter", () => btn.style.background = "rgba(91,156,246,0.25)");
    btn.addEventListener("mouseleave", () => btn.style.background = "rgba(91,156,246,0.12)");
    btn.addEventListener("click", () => { info.textContent = "Показываем матрицы..."; info.style.color = ""; done = 0; for (const r of restarts) r(); });
    body.appendChild(btn);
  }
  const note = document.createElement("div");
  note.style.cssText = "font-size:0.88rem;color:var(--text);line-height:1.5;padding:0.5rem;background:rgba(0,0,0,0.15);border-radius:8px;border:1px solid var(--border);min-height:1.5em";
  body.appendChild(note);
  const txt = data.text;
  let pos = 0;
  let cursor = null;
  function typeChar() {
    if (pos >= txt.length) { if (cursor) cursor.remove(); return; }
    note.textContent = txt.slice(0, pos + 1);
    pos++;
    setTimeout(typeChar, 25);
  }
  typeChar();
  addRepeatBtn(body, () => {
    pos = 0;
    note.textContent = "";
    typeChar();
  });
}

function livePseudoinverse(body, data) {
  const { C, R, Cp, Rp, A, U } = data;
  const r = C[0].length;
  const m = C.length;
  const n = R[0].length;

  function matStr(M) {
    return M.map(r => r.map(v => v.toFixed(3)).join(" ")).join(" | ");
  }

  function showMatrixTbl(parent, mat, label, highlight) {
    const block = document.createElement("div");
    block.style.cssText = "text-align:center;opacity:0;transition:opacity 0.5s";
    const lbl = document.createElement("div");
    lbl.style.cssText = "font-weight:600;font-size:0.75rem;margin-bottom:0.15rem;color:var(--accent)";
    lbl.textContent = label;
    block.appendChild(lbl);
    const tbl = document.createElement("table");
    tbl.style.cssText = "border-collapse:collapse;margin:0 auto;font-size:0.68rem;font-family:monospace";
    for (let i = 0; i < mat.length; i++) {
      const tr = document.createElement("tr");
      for (let j = 0; j < mat[0].length; j++) {
        const td = document.createElement("td");
        const isHi = highlight && highlight[0] === i && highlight[1] === j;
        td.style.cssText = "padding:1px 5px;border:1px solid var(--border);text-align:right" + (isHi ? ";background:rgba(74,222,128,0.15)" : "");
        td.textContent = mat[i][j].toFixed(3);
        tr.appendChild(td);
      }
      tbl.appendChild(tr);
    }
    block.appendChild(tbl);
    parent.appendChild(block);
    requestAnimationFrame(() => block.style.opacity = "1");
  }

  function showFormula(parent, text) {
    const el = document.createElement("div");
    el.style.cssText = "text-align:center;font-size:0.72rem;font-family:monospace;opacity:0;transition:opacity 0.5s;margin:0.2rem 0";
    el.textContent = text;
    parent.appendChild(el);
    requestAnimationFrame(() => el.style.opacity = "1");
  }

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:0.6rem;font-size:0.78rem";

  const phaseDiv = document.createElement("div");
  phaseDiv.style.cssText = "font-size:0.8rem;color:var(--accent);font-weight:600;margin-bottom:0.3rem";
  wrap.appendChild(phaseDiv);

  const contentDiv = document.createElement("div");
  contentDiv.style.cssText = "display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:center;min-height:60px";
  wrap.appendChild(contentDiv);

  const statusDiv = document.createElement("div");
  statusDiv.style.cssText = "font-size:0.75rem;color:var(--text-dim);text-align:center;min-height:1.2em";
  wrap.appendChild(statusDiv);

  let paused = false;
  let resumeFn = null;
  let abort = false;

  const pauseBtn = document.createElement("button");
  pauseBtn.textContent = "⏸ Пауза";
  pauseBtn.style.cssText = "padding:0.25rem 0.6rem;font-size:0.72rem;cursor:pointer;border:1px solid var(--border);border-radius:4px;background:var(--bg-card);color:var(--text)";
  pauseBtn.addEventListener("click", () => {
    if (!paused) {
      paused = true;
      pauseBtn.textContent = "▶ Продолжить";
    } else {
      paused = false;
      pauseBtn.textContent = "⏸ Пауза";
      const fn = resumeFn; resumeFn = null;
      if (fn) fn();
    }
  });

  function restartPinv() {
    abort = true;
    paused = false;
    resumeFn = null;
    body.removeChild(wrap);
    livePseudoinverse(body, data);
  }

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:0.4rem;justify-content:center;margin-top:0.3rem";
  btnRow.appendChild(pauseBtn);
  addRepeatBtn(btnRow, restartPinv);
  wrap.appendChild(btnRow);

  body.appendChild(wrap);

  function schedule(fn, delay) {
    if (abort) return;
    const id = setTimeout(() => {
      if (abort) return;
      if (paused) {
        resumeFn = fn;
        return;
      }
      fn();
    }, delay);
    return id;
  }

  function renderCPinvPhase() {
    contentDiv.innerHTML = "";
    phaseDiv.textContent = `Шаг 1: C⁺ = (CᵀC)⁻¹·Cᵀ (псевдообратная к C)`;
    statusDiv.textContent = "Вычисляем псевдообразную C⁺...";

    const Ct = transpose(C);
    const CtC = dot(Ct, C);
    const scalarCase = r === 1;

    schedule(() => {
      showMatrixTbl(contentDiv, C, "C");
      statusDiv.textContent = "Матрица C (выбранные столбцы)";
    }, 100);

    schedule(() => {
      statusDiv.textContent = "Вычисляем CᵀC ...";
      showFormula(contentDiv, `CᵀC = ${matStr(CtC)}`);
    }, 800);

    schedule(() => {
      if (scalarCase) {
        const val = CtC[0][0];
        const inv = 1 / val;
        showFormula(contentDiv, `(CᵀC)⁻¹ = 1 / ${val.toFixed(3)} = ${inv.toFixed(3)}`);
        statusDiv.textContent = `CᵀC = ${val.toFixed(3)} → (CᵀC)⁻¹ = ${inv.toFixed(3)}`;
      } else {
        showFormula(contentDiv, "(CᵀC)⁻¹ — обратная матрица");
        statusDiv.textContent = "(CᵀC)⁻¹ вычислена";
      }
      schedule(() => renderCRegistration(), 1200);
    }, 1600);

    function renderCRegistration() {
      showMatrixTbl(contentDiv, Cp, "C⁺ = (CᵀC)⁻¹·Cᵀ", [0, 0]);
      statusDiv.textContent = "C⁺ найдена!";
      schedule(() => renderRPinvPhase(), 800);
    }
  }

  function renderRPinvPhase() {
    contentDiv.innerHTML = "";
    phaseDiv.textContent = `Шаг 2: R⁺ = Rᵀ·(R·Rᵀ)⁻¹ (псевдообратная к R)`;
    statusDiv.textContent = "Вычисляем псевдообразную R⁺...";

    const Rt = transpose(R);
    const RRt = dot(R, Rt);
    const scalarCase = r === 1;

    schedule(() => {
      showMatrixTbl(contentDiv, R, "R");
      statusDiv.textContent = "Матрица R (выбранные строки)";
    }, 100);

    schedule(() => {
      statusDiv.textContent = "Вычисляем R·Rᵀ ...";
      showFormula(contentDiv, `R·Rᵀ = ${matStr(RRt)}`);
    }, 800);

    schedule(() => {
      if (scalarCase) {
        const val = RRt[0][0];
        const inv = 1 / val;
        showFormula(contentDiv, `(R·Rᵀ)⁻¹ = 1 / ${val.toFixed(3)} = ${inv.toFixed(3)}`);
        statusDiv.textContent = `R·Rᵀ = ${val.toFixed(3)} → (R·Rᵀ)⁻¹ = ${inv.toFixed(3)}`;
      } else {
        showFormula(contentDiv, "(R·Rᵀ)⁻¹ — обратная матрица");
        statusDiv.textContent = "(R·Rᵀ)⁻¹ вычислена";
      }
      schedule(() => renderRRegistration(), 1200);
    }, 1600);

    function renderRRegistration() {
      showMatrixTbl(contentDiv, Rp, "R⁺ = Rᵀ·(R·Rᵀ)⁻¹", [0, 0]);
      statusDiv.textContent = "R⁺ найдена!";
      schedule(() => renderUPhase(), 800);
    }
  }

  function renderUPhase() {
    contentDiv.innerHTML = "";
    phaseDiv.textContent = `Шаг 3: U = C⁺·A·R⁺ (связующая матрица)`;
    statusDiv.textContent = "Перемножаем C⁺ · A · R⁺ ...";

    schedule(() => {
      showFormula(contentDiv, "U = C⁺ · A · R⁺");
      statusDiv.textContent = "Промежуточный результат: C⁺ · A";
    }, 300);

    schedule(() => {
      showFormula(contentDiv, "Финальное умножение: (C⁺·A) · R⁺");
      statusDiv.textContent = "Вычисляем (C⁺·A) · R⁺ ...";
    }, 1200);

    schedule(() => {
      const result = document.createElement("div");
      result.style.cssText = "text-align:center;opacity:0;transition:opacity 0.5s;margin-top:0.4rem";
      const rlbl = document.createElement("div");
      rlbl.style.cssText = "font-weight:600;font-size:0.78rem;margin-bottom:0.2rem;color:var(--good)";
      rlbl.textContent = "U = C⁺·A·R⁺ = ";
      result.appendChild(rlbl);

      const tbl = document.createElement("table");
      tbl.style.cssText = "border-collapse:collapse;margin:0 auto;font-size:0.68rem;font-family:monospace";
      for (let i = 0; i < U.length; i++) {
        const tr = document.createElement("tr");
        for (let j = 0; j < U[0].length; j++) {
          const td = document.createElement("td");
          td.style.cssText = "padding:1px 5px;border:1px solid var(--border);text-align:right;background:rgba(74,222,128,0.08)";
          td.textContent = U[i][j].toFixed(3);
          tr.appendChild(td);
        }
        tbl.appendChild(tr);
      }
      result.appendChild(tbl);
      contentDiv.appendChild(result);
      requestAnimationFrame(() => result.style.opacity = "1");
      statusDiv.textContent = "Связующая матрица U = C⁺·A·R⁺ найдена!";
    }, 2000);
  }

  renderCPinvPhase();
}

function liveErrorDetail(body, data) {
  const { A, Ahat, scale } = data;
  const m = A.length, n = A[0].length;
  const sA = scale || minMax(A);

  let paused = false, resumeFn = null, abort = false;
  let ci = 0, cj = 0;
  let frobSum = 0;

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:0.5rem;font-size:0.78rem";

  const phaseDiv = document.createElement("div");
  phaseDiv.style.cssText = "font-size:0.82rem;color:var(--accent);font-weight:600;margin-bottom:0.2rem";
  wrap.appendChild(phaseDiv);

  const matricesRow = document.createElement("div");
  matricesRow.style.cssText = "display:flex;flex-wrap:wrap;gap:0.6rem;justify-content:center;align-items:center";
  wrap.appendChild(matricesRow);

  const trace = document.createElement("div");
  trace.style.cssText = "font-size:0.85rem;color:var(--text);text-align:center;font-family:monospace;min-height:2rem;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin:0.3rem 0";
  wrap.appendChild(trace);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
  wrap.appendChild(info);

  function renderMiniMat(container, mat, label, colorKey) {
    const block = document.createElement("div");
    block.style.cssText = "text-align:center";
    const lbl = document.createElement("div");
    lbl.style.cssText = "font-weight:600;font-size:0.72rem;margin-bottom:0.1rem;color:" + colorKey;
    lbl.textContent = label;
    block.appendChild(lbl);
    const host = document.createElement("div");
    renderMatrixBlock(host, "", mat, { scale: sA });
    block.appendChild(host);
    container.innerHTML = "";
    container.appendChild(block);
  }

  function schedule(fn, delay) {
    if (abort) return;
    setTimeout(() => {
      if (abort) return;
      if (paused) { resumeFn = fn; return; }
      fn();
    }, delay);
  }

  // Phase 1: show A and Ahat side by side, then compute error cell-by-cell
  function phase1() {
    phaseDiv.textContent = "Шаг 1: Матрица ошибки E = A − Ã";
    info.textContent = "Вычисляем поэлементно...";

    const wrapA = document.createElement("div");
    wrapA.style.cssText = "text-align:center";
    const lblA = document.createElement("div");
    lblA.style.cssText = "font-weight:600;font-size:0.72rem;margin-bottom:0.1rem;color:var(--accent)";
    lblA.textContent = "A";
    wrapA.appendChild(lblA);
    const hostA = document.createElement("div");
    renderMatrixBlock(hostA, "", A, { scale: sA });
    wrapA.appendChild(hostA);
    matricesRow.appendChild(wrapA);

    const opSign = document.createElement("div");
    opSign.style.cssText = "font-size:1.2rem;font-weight:700;color:var(--text);padding:0 0.2rem";
    opSign.textContent = "−";
    matricesRow.appendChild(opSign);

    const wrapAhat = document.createElement("div");
    wrapAhat.style.cssText = "text-align:center";
    const lblAh = document.createElement("div");
    lblAh.style.cssText = "font-weight:600;font-size:0.72rem;margin-bottom:0.1rem;color:var(--good)";
    lblAh.textContent = "Ã";
    wrapAhat.appendChild(lblAh);
    const hostAhat = document.createElement("div");
    renderMatrixBlock(hostAhat, "", Ahat, { scale: sA });
    wrapAhat.appendChild(hostAhat);
    matricesRow.appendChild(wrapAhat);

    const eqSign = document.createElement("div");
    eqSign.style.cssText = "font-size:1.2rem;font-weight:700;color:var(--text);padding:0 0.2rem";
    eqSign.textContent = "=";
    matricesRow.appendChild(eqSign);

    const hostE = document.createElement("div");
    const errWork = zeros(m, n);
    renderMatrixBlock(hostE, "", errWork, { scale: sA });
    const wrapE = document.createElement("div");
    wrapE.style.cssText = "text-align:center";
    const lblE = document.createElement("div");
    lblE.style.cssText = "font-weight:600;font-size:0.72rem;margin-bottom:0.1rem;color:var(--bad)";
    lblE.textContent = "E = A − Ã";
    wrapE.appendChild(lblE);
    wrapE.appendChild(hostE);
    matricesRow.appendChild(wrapE);

    const cellsA = hostA.querySelectorAll(".cell");
    const cellsAh = hostAhat.querySelectorAll(".cell");
    const cellsE = hostE.querySelectorAll(".cell");

    function fillErrCell() {
      if (abort) return;
      if (ci >= m) {
        info.textContent = "Матрица ошибки E готова!";
        schedule(phase2, 800);
        return;
      }
      const idx = ci * n + cj;
      const val = A[ci][cj] - Ahat[ci][cj];
      if (cellsA[idx]) cellsA[idx].style.boxShadow = "inset 0 0 0 3px var(--accent)";
      if (cellsAh[idx]) cellsAh[idx].style.boxShadow = "inset 0 0 0 3px var(--good)";
      if (cellsE[idx]) {
        errWork[ci][cj] = val;
        const t = sA.mx > sA.mn ? (val - sA.mn) / (sA.mx - sA.mn) : 0.5;
        const c = viridis(Math.max(0, Math.min(1, t)));
        cellsE[idx].style.background = viridisRgb(c);
        cellsE[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
        cellsE[idx].textContent = val.toFixed(2);
        cellsE[idx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
      }
      trace.innerHTML = `E<sub>${ci}${cj}</sub> = ${A[ci][cj].toFixed(2)} − ${Ahat[ci][cj].toFixed(2)} = <b>${val.toFixed(2)}</b>`;
      info.textContent = `(${ci},${cj}) разница = ${val.toFixed(2)}`;

      setTimeout(() => {
        if (cellsA[idx]) cellsA[idx].style.boxShadow = "";
        if (cellsAh[idx]) cellsAh[idx].style.boxShadow = "";
        if (cellsE[idx]) cellsE[idx].style.boxShadow = "";
      }, 400);

      cj++;
      if (cj >= n) { cj = 0; ci++; }
      schedule(fillErrCell, 350);
    }
    fillErrCell();
  }

  // Phase 2: compute squared terms for Frobenius norm
  function phase2() {
    ci = 0; cj = 0; frobSum = 0;
    phaseDiv.textContent = "Шаг 2: Сумма квадратов Σᵢⱼ Eᵢⱼ² → ||E||_F";

    renderMiniMat(matricesRow, zeros(m, n), "E² (поквадратно)", "var(--bad)");

    const hostSq = matricesRow.querySelector("div:last-child div:last-child");
    const cellsSq = hostSq ? hostSq.querySelectorAll(".cell") : [];
    const cellsE = matricesRow.querySelectorAll(".cell");

    const sumRow = document.createElement("div");
    sumRow.style.cssText = "text-align:center;font-size:0.75rem;font-family:monospace;margin-top:0.3rem;min-height:1.4rem";
    wrap.insertBefore(sumRow, trace);

    function fillSqCell() {
      if (abort) return;
      if (ci >= m) {
        const frob = Math.sqrt(frobSum);
        const normA = Math.sqrt(A.reduce((s, row) => s + row.reduce((ss, v) => ss + v * v, 0), 0));
        const relErr = frob / (normA + 1e-12);
        trace.innerHTML = `ΣEᵢⱼ² = <b>${frobSum.toFixed(4)}</b> → ||E||_F = √(${frobSum.toFixed(4)}) = <b>${frob.toFixed(4)}</b>`;
        info.textContent = `Frobenius норма: ${frob.toFixed(4)}`;
        sumRow.textContent = `Относительная ошибка: ||E||_F / ||A||_F = ${(relErr * 100).toFixed(2)}%`;
        sumRow.style.fontWeight = "600";
        sumRow.style.color = "var(--good)";
        return;
      }
      const idx = ci * n + cj;
      const val = A[ci][cj] - Ahat[ci][cj];
      const sq = val * val;
      frobSum += sq;
      if (cellsSq && cellsSq[idx]) {
        const t = (sq - 0) / (Math.max(...A.flat().map(v => v * v), 1));
        const c = viridis(Math.max(0, Math.min(1, t)));
        cellsSq[idx].style.background = viridisRgb(c);
        cellsSq[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
        cellsSq[idx].textContent = sq.toFixed(4);
        cellsSq[idx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
      }
      if (cellsE && cellsE[idx]) cellsE[idx].style.boxShadow = "inset 0 0 0 3px var(--good)";
      trace.innerHTML = `E<sub>${ci}${cj}</sub>² = (${val.toFixed(2)})² = ${sq.toFixed(4)}`;
      sumRow.textContent = `Σ = ${frobSum.toFixed(4)} (из ${m * n})`;
      info.textContent = `Текущая сумма: ${frobSum.toFixed(4)}`;

      setTimeout(() => {
        if (cellsSq && cellsSq[idx]) cellsSq[idx].style.boxShadow = "";
        if (cellsE && cellsE[idx]) cellsE[idx].style.boxShadow = "";
      }, 400);

      cj++;
      if (cj >= n) { cj = 0; ci++; }
      schedule(fillSqCell, 300);
    }
    fillSqCell();
  }

  const pauseBtn = document.createElement("button");
  pauseBtn.textContent = "⏸ Пауза";
  pauseBtn.style.cssText = "padding:0.25rem 0.6rem;font-size:0.72rem;cursor:pointer;border:1px solid var(--border);border-radius:4px;background:var(--bg-card);color:var(--text)";
  pauseBtn.addEventListener("click", () => {
    if (!paused) { paused = true; pauseBtn.textContent = "▶ Продолжить"; }
    else { paused = false; pauseBtn.textContent = "⏸ Пауза"; const fn = resumeFn; resumeFn = null; if (fn) fn(); }
  });

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:0.4rem;justify-content:center;margin-top:0.3rem";
  btnRow.appendChild(pauseBtn);
  addRepeatBtn(btnRow, () => {
    abort = true; paused = false; resumeFn = null;
    ci = m; // stop loops
    body.removeChild(wrap);
    liveErrorDetail(body, data);
  });
  wrap.appendChild(btnRow);
  body.appendChild(wrap);

  phase1();
}

function showArrowDetail(arrowStep, steps, arrowIndex) {
  const { op, label: arrowLabel, data } = arrowStep;
  const title = OP_TITLES[op] || "Матричная операция";
  const color = opColor(op);

  const overlay = document.createElement("div");
  overlay.className = "vis-modal-overlay";

  const modal = document.createElement("div");
  modal.className = "vis-modal vis-modal--wide";

  const head = document.createElement("div");
  head.className = "vis-modal__head";

  const dot = document.createElement("span");
  dot.className = "vis-modal__dot";
  dot.style.background = color;

  const titleEl = document.createElement("span");
  titleEl.className = "vis-modal__title";
  titleEl.textContent = `${title}: ${arrowLabel}`;

  const close = document.createElement("button");
  close.className = "vis-modal__close";
  close.textContent = "×";
  close.addEventListener("click", () => overlay.remove());

  head.appendChild(dot);
  head.appendChild(titleEl);
  head.appendChild(close);

  const body = document.createElement("div");
  body.className = "vis-modal__body";

  if (data) {
    try {
      switch (data.type) {
        case "matrix_product": liveProduct(body, data); break;
        case "triple_product": liveTriple(body, data); break;
        case "addition": liveAddition(body, data); break;
        case "subtraction": liveSubtraction(body, data); break;
        case "means": liveMeans(body, data); break;
        case "eigenvalues": liveEigenvalues(body, data); break;
        case "values_map": liveValueMap(body, data); break;
        case "norm_computation": liveNorms(body, data); break;
        case "selection": liveSelection(body, data); break;
        case "init_matrices": liveInit(body, data); break;
        case "iteration_note": liveNote(body, data); break;
        case "iteration_history": liveIterationHistory(body, data); break;
        case "pseudoinverse": livePseudoinverse(body, data); break;
        case "error_detail": liveErrorDetail(body, data); break;
        default: break;
      }
    } catch (e) {
      const err = document.createElement("div");
      err.style.cssText = "font-size:0.82rem;color:var(--bad);padding:0.5rem;text-align:center";
      err.textContent = `Ошибка: ${e.message}`;
      body.appendChild(err);
    }
  }

  if (steps) {
    const nav = document.createElement("div");
    nav.style.cssText = "display:flex;gap:0.5rem;justify-content:center;margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid var(--border)";

    function findPrevArrow(idx) {
      for (let i = idx - 1; i >= 0; i--) if (steps[i].type === "arrow") return i;
      return -1;
    }
    function findNextArrow(idx) {
      for (let i = idx + 1; i < steps.length; i++) if (steps[i].type === "arrow") return i;
      return -1;
    }

    const prevIdx = findPrevArrow(arrowIndex);
    const nextIdx = findNextArrow(arrowIndex);

    if (prevIdx >= 0) {
      const prevBtn = document.createElement("button");
      prevBtn.textContent = "◀ " + steps[prevIdx].label;
      prevBtn.style.cssText = "padding:0.3rem 0.7rem;font-size:0.75rem;font-weight:600;color:var(--accent);background:rgba(91,156,246,0.1);border:1px solid var(--border);border-radius:6px;cursor:pointer;max-width:40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      prevBtn.addEventListener("click", () => { overlay.remove(); showArrowDetail(steps[prevIdx], steps, prevIdx); });
      nav.appendChild(prevBtn);
    }

    if (nextIdx >= 0) {
      const nextBtn = document.createElement("button");
      nextBtn.textContent = steps[nextIdx].label + " ▶";
      nextBtn.style.cssText = "padding:0.3rem 0.7rem;font-size:0.75rem;font-weight:600;color:var(--accent);background:rgba(91,156,246,0.1);border:1px solid var(--border);border-radius:6px;cursor:pointer;max-width:40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
      nextBtn.addEventListener("click", () => { overlay.remove(); showArrowDetail(steps[nextIdx], steps, nextIdx); });
      nav.appendChild(nextBtn);
    }

    if (prevIdx >= 0 || nextIdx >= 0) body.appendChild(nav);
  }

  modal.appendChild(head);
  modal.appendChild(body);
  overlay.appendChild(modal);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add("vis-modal--open"));
}

// ── Interactive legend — click dot to see op demo ──

const OP_DEMO = {
  multiply: {
    title: "Умножение матриц",
    latex: "C_{ij} = \\sum_{k} A_{ik}B_{kj}",
    steps: [
      { desc: "Пример: две матрицы 2×2", latex: "" },
      { desc: "A =", latex: "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}" },
      { desc: "B =", latex: "\\begin{bmatrix} 5 & 6 \\\\ 7 & 8 \\end{bmatrix}" },
      { desc: "C = A·B, элемент C₁₁ = 1·5 + 2·7 = 5 + 14 = 19:", latex: "C = \\begin{bmatrix} 1\\cdot5+2\\cdot7 & 1\\cdot6+2\\cdot8 \\\\ 3\\cdot5+4\\cdot7 & 3\\cdot6+4\\cdot8 \\end{bmatrix}" },
      { desc: "Промежуточные произведения:", latex: "C = \\begin{bmatrix} 5+14 & 6+16 \\\\ 15+28 & 18+32 \\end{bmatrix}" },
      { desc: "Результат:", latex: "C = \\begin{bmatrix} 19 & 22 \\\\ 43 & 50 \\end{bmatrix}" },
    ],
  },
  transpose: {
    title: "Транспонирование",
    latex: "(A^T)_{ij} = A_{ji}",
    steps: [
      { desc: "Строки становятся столбцами:", latex: "" },
      { desc: "A =", latex: "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}" },
      { desc: "Aᵀ =", latex: "\\begin{bmatrix} 1 & 3 \\\\ 2 & 4 \\end{bmatrix}" },
    ],
  },
  normalize: {
    title: "Нормализация / центрирование",
    latex: "\\hat{x}_{ij} = x_{ij} - \\mu_j",
    steps: [
      { desc: "Пример: вычитание среднего по столбцам:", latex: "" },
      { desc: "A =", latex: "\\begin{bmatrix} 2 & 3 \\\\ 4 & 5 \\end{bmatrix}" },
      { desc: "μ₁ = (2+4)/2 = 3, μ₂ = (3+5)/2 = 4", latex: "" },
      { desc: "X = A − μ =", latex: "\\begin{bmatrix} 2-3 & 3-4 \\\\ 4-3 & 5-4 \\end{bmatrix} = \\begin{bmatrix} -1 & -1 \\\\ 1 & 1 \\end{bmatrix}" },
    ],
  },
  eigen: {
    title: "Собственные значения и векторы",
    latex: "A \\cdot v = \\lambda \\cdot v",
    steps: [
      { desc: "Собственный вектор v не меняет направление при умножении на A, только масштабируется на λ.", latex: "" },
      { desc: "A =", latex: "\\begin{bmatrix} 4 & 1 \\\\ 2 & 3 \\end{bmatrix}" },
      { desc: "det(A − λI) = 0 → (4−λ)(3−λ) − 2 = 0 → λ² − 7λ + 10 = 0", latex: "" },
      { desc: "λ₁ = 5, λ₂ = 2", latex: "" },
    ],
  },
  init: {
    title: "Инициализация",
    latex: "W_{ij} \\sim U(0, 1)",
    steps: [
      { desc: "Начальные значения факторных матриц выбираются случайно из равномерного распределения.", latex: "" },
      { desc: "Пример: W 3×2:", latex: "\\begin{bmatrix} 0.54 & 0.91 \\\\ 0.12 & 0.67 \\\\ 0.83 & 0.34 \\end{bmatrix}" },
      { desc: "Пример: H 2×3:", latex: "\\begin{bmatrix} 0.45 & 0.72 & 0.19 \\\\ 0.88 & 0.31 & 0.56 \\end{bmatrix}" },
    ],
  },
};

function renderTheorySteps(container, steps) {
  for (const s of steps) {
    const block = document.createElement("div");
    block.style.cssText = "margin:0.3rem 0;padding:0.3rem 0.5rem;border-left:3px solid var(--accent);background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0";
    const desc = document.createElement("div");
    desc.style.cssText = "font-size:0.8rem;color:var(--text);margin-bottom:0.15rem";
    desc.textContent = s.desc;
    block.appendChild(desc);
    if (s.latex) {
      const fw = document.createElement("div");
      fw.style.cssText = "font-size:0.85rem;text-align:center;padding:0.15rem 0";
      try {
        fw.innerHTML = window.katex.renderToString(s.latex, { displayMode: true, throwOnError: false, trust: true });
      } catch {
        fw.textContent = s.latex;
      }
      block.appendChild(fw);
    }
    container.appendChild(block);
  }
}

function showOpDetailModal(op) {
  const demo = OP_DEMO[op];
  if (!demo) return;

  const color = opColor(op);
  const overlay = document.createElement("div");
  overlay.className = "vis-modal-overlay";

  const modal = document.createElement("div");
  modal.className = "vis-modal vis-modal--wide";

  const head = document.createElement("div");
  head.className = "vis-modal__head";

  const dot = document.createElement("span");
  dot.className = "vis-modal__dot";
  dot.style.background = color;

  const titleEl = document.createElement("span");
  titleEl.className = "vis-modal__title";
  titleEl.textContent = demo.title;

  const close = document.createElement("button");
  close.className = "vis-modal__close";
  close.textContent = "×";
  close.addEventListener("click", () => overlay.remove());
  head.appendChild(dot);
  head.appendChild(titleEl);
  head.appendChild(close);

  const body = document.createElement("div");
  body.className = "vis-modal__body";

  const fw = document.createElement("div");
  fw.className = "vis-modal__formula";
  try {
    fw.innerHTML = window.katex.renderToString(demo.latex, { displayMode: true, throwOnError: false, trust: true });
  } catch {
    fw.textContent = demo.latex;
  }
  body.appendChild(fw);

  const wrap = document.createElement("div");
  wrap.className = "vis-modal__comp";
  renderTheorySteps(wrap, demo.steps);
  body.appendChild(wrap);

  modal.appendChild(head);
  modal.appendChild(body);
  overlay.appendChild(modal);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add("vis-modal--open"));
}

// ── Matrix editor with mouse wheel ──

function renderMatrixEditor(container, A, onChange) {
  const { m, n } = dims(A);
  const editor = document.createElement("div");
  editor.className = "vis-editor";

  const grid = document.createElement("div");
  grid.className = "matrix-grid";
  grid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;

  let activeCell = null;
  const { mn, mx } = minMax(A);

  function cellColor(v) {
    const t = (v - mn) / (mx - mn + 1e-12);
    return viridis(t);
  }

  function applyColor(cell, v) {
    const c = cellColor(v);
    cell.style.background = viridisRgb(c);
    const t = (v - mn) / (mx - mn + 1e-12);
    cell.style.color = t > 0.6 ? "#080c14" : "#fff";
  }

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const v = A[i][j];
      const cell = document.createElement("div");
      cell.className = "cell cell-editable";
      cell.dataset.i = i;
      cell.dataset.j = j;

      applyColor(cell, v);
      cell.textContent = v.toFixed(2);

      cell.addEventListener("click", () => {
        if (activeCell) activeCell.classList.remove("cell-active");
        cell.classList.add("cell-active");
        activeCell = cell;
      });

      cell.addEventListener("wheel", (e) => {
        e.preventDefault();
        if (cell !== activeCell) {
          if (activeCell) activeCell.classList.remove("cell-active");
          cell.classList.add("cell-active");
          activeCell = cell;
        }
        const delta = e.deltaY < 0 ? 1 : -1;
        const step = e.shiftKey ? 1 : 0.1;
        const newVal = Math.round((A[i][j] + delta * step) * 100) / 100;
        A[i][j] = newVal;
        applyColor(cell, newVal);
        cell.textContent = newVal.toFixed(2);
        if (onChange) onChange(clone(A));
      }, { passive: false });

      grid.appendChild(cell);
    }
  }

  editor.appendChild(grid);
  container.appendChild(editor);
  return editor;
}

// ── Norms display for the control bar ──

function renderRowColNorms(container, A) {
  const { m, n } = dims(A);
  const wrap = document.createElement("div");
  wrap.className = "vis-norms-mini";

  const colNorms = new Array(n).fill(0);
  const rowNorms = new Array(m).fill(0);
  for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) colNorms[j] += A[i][j] * A[i][j];
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) rowNorms[i] += A[i][j] * A[i][j];

  const maxCol = Math.max(...colNorms, 1e-12);
  const maxRow = Math.max(...rowNorms, 1e-12);

  const colDiv = document.createElement("div");
  colDiv.className = "vis-norms-mini__col";
  const colTitle = document.createElement("div");
  colTitle.className = "vis-norms-mini__title";
  colTitle.innerHTML = "||col||²";
  colDiv.appendChild(colTitle);
  for (let j = 0; j < n; j++) {
    const colWrap = document.createElement("div");
    colWrap.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:2px";
    const bar = document.createElement("div");
    bar.className = "vis-norms-mini__bar";
    const fill = document.createElement("div");
    fill.className = "vis-norms-mini__fill";
    fill.style.height = `${(colNorms[j] / maxCol) * 100}%`;
    bar.appendChild(fill);
    colWrap.appendChild(bar);
    const val = document.createElement("div");
    val.style.cssText = "font-size:0.55rem;color:var(--muted);text-align:center;line-height:1";
    val.textContent = colNorms[j].toFixed(1);
    colWrap.appendChild(val);
    colDiv.appendChild(colWrap);
  }
  wrap.appendChild(colDiv);

  const rowDiv = document.createElement("div");
  rowDiv.className = "vis-norms-mini__row";
  const rowTitle = document.createElement("div");
  rowTitle.className = "vis-norms-mini__title";
  rowTitle.innerHTML = "||row||²";
  rowDiv.appendChild(rowTitle);
  for (let i = 0; i < m; i++) {
    const rowWrap = document.createElement("div");
    rowWrap.style.cssText = "display:flex;align-items:center;gap:3px";
    const bar = document.createElement("div");
    bar.className = "vis-norms-mini__bar";
    const fill = document.createElement("div");
    fill.className = "vis-norms-mini__fill";
    fill.style.width = `${(rowNorms[i] / maxRow) * 100}%`;
    bar.appendChild(fill);
    rowWrap.appendChild(bar);
    const val = document.createElement("div");
    val.style.cssText = "font-size:0.55rem;color:var(--muted);white-space:nowrap";
    val.textContent = rowNorms[i].toFixed(1);
    rowWrap.appendChild(val);
    rowDiv.appendChild(rowWrap);
  }
  wrap.appendChild(rowDiv);

  container.appendChild(wrap);
}

// ── Presets ──

function applyPreset(name, rows, cols, seed = 42) {
  switch (name) {
    case "identity": {
      const M = zeros(rows, cols);
      for (let i = 0; i < Math.min(rows, cols); i++) M[i][i] = 1;
      return M;
    }
    case "zeros": return zeros(rows, cols);
    case "random": return randomMatrix(rows, cols, 0, 10, seed);
    case "example": return [
      [4, 3],
      [2, 1],
    ];
    default: return randomMatrix(rows, cols, 0, 10, seed);
  }
}

// ── Main page render ──

export function renderVisualizerPage(container, state) {
  try {
  clear(container);
  const algo = state.visAlgo || "svd";
  const A = state.visA || clone(state.A) || [[4, 3], [2, 1]];
  const k = state.visK || Math.min(state.k || 2, A[0].length, A.length);
  const iters = state.visIters || state.iters || 20;
  const visSeed = state.visSeed || 42;

  const layout = document.createElement("div");
  layout.className = "vis-layout";

  // ── Arrow color legend ──
  const legendBar = document.createElement("div");
  legendBar.className = "vis-legend-bar";
  const legendItems = [
    { color: OP_COLORS.multiply, label: "умножение", op: "multiply" },
    { color: OP_COLORS.transpose, label: "транспонирование", op: "transpose" },
    { color: OP_COLORS.normalize, label: "нормализация", op: "normalize" },
    { color: OP_COLORS.eigen, label: "собственные значения", op: "eigen" },
    { color: OP_COLORS.init, label: "инициализация", op: "init" },
  ];
  for (const item of legendItems) {
    const dot = document.createElement("span");
    dot.className = "vis-legend-dot";
    dot.style.background = item.color;
    dot.title = `Нажмите для демонстрации операции «${item.label}»`;
    dot.style.cursor = "pointer";
    dot.addEventListener("click", () => showOpDetailModal(item.op));
    const lbl = document.createElement("span");
    lbl.className = "vis-legend-text";
    lbl.textContent = item.label;
    const wrap = document.createElement("span");
    wrap.className = "vis-legend-item";
    wrap.style.cursor = "pointer";
    wrap.appendChild(dot);
    wrap.appendChild(lbl);
    wrap.addEventListener("click", () => showOpDetailModal(item.op));
    legendBar.appendChild(wrap);
  }
  layout.appendChild(legendBar);

  // ── Matrix editor + Pipeline ──
  const body = document.createElement("div");
  body.className = "vis-body";

  // Editor card
  const editorCard = document.createElement("div");
  editorCard.className = "card vis-editor-card";
  const editorHead = document.createElement("div");
  editorHead.className = "card__head";
  editorHead.innerHTML = `<h2>Матрица A</h2><div class="sub">клик → выбор, колёсико ±0.1 (Shift ±1)</div>`;
  const normsMini = document.createElement("div");
  normsMini.style.cssText = "margin-left:auto;display:flex;align-self:center";
  renderRowColNorms(normsMini, A);
  editorHead.appendChild(normsMini);
  editorCard.appendChild(editorHead);

  // Viridis color scale bar
  const scaleBar = document.createElement("div");
  scaleBar.className = "vis-scale-bar";
  const scaleGrad = document.createElement("div");
  scaleGrad.className = "vis-scale-gradient";
  const scaleLabels = document.createElement("div");
  scaleLabels.className = "vis-scale-labels";
  scaleLabels.innerHTML = `<span>мин</span><span>макс</span>`;
  scaleBar.appendChild(scaleGrad);
  scaleBar.appendChild(scaleLabels);
  editorCard.appendChild(scaleBar);

  const editorHost = document.createElement("div");
  editorHost.className = "vis-editor-host";
  renderMatrixEditor(editorHost, A, (newA) => {
    state.visA = newA;
    renderVisualizerPage(container, state);
  });
  editorCard.appendChild(editorHost);

  // Size controls
  const sizeRow = document.createElement("div");
  sizeRow.className = "vis-size-row";
  const rowInput = document.createElement("input");
  rowInput.type = "number";
  rowInput.min = 2;
  rowInput.max = 10;
  rowInput.value = A.length;
  rowInput.style.width = "60px";
  const colInput = document.createElement("input");
  colInput.type = "number";
  colInput.min = 2;
  colInput.max = 10;
  colInput.value = A[0].length;
  colInput.style.width = "60px";
  const sizeLabel = document.createElement("span");
  sizeLabel.className = "vis-size-label";
  sizeLabel.textContent = "Размер: ";
  const applySize = document.createElement("button");
  applySize.textContent = "Применить";
  applySize.addEventListener("click", () => {
    const newM = Math.max(2, Math.min(10, Number(rowInput.value) || 2));
    const newN = Math.max(2, Math.min(10, Number(colInput.value) || 2));
    const newA = zeros(newM, newN);
    for (let i = 0; i < Math.min(newM, A.length); i++)
      for (let j = 0; j < Math.min(newN, A[0].length); j++)
        newA[i][j] = A[i][j];
    state.visA = newA;
    renderVisualizerPage(container, state);
  });
  sizeRow.appendChild(sizeLabel);
  sizeRow.appendChild(document.createTextNode(" m: "));
  sizeRow.appendChild(rowInput);
  sizeRow.appendChild(document.createTextNode(" n: "));
  sizeRow.appendChild(colInput);
  sizeRow.appendChild(applySize);
  editorCard.appendChild(sizeRow);

  body.appendChild(editorCard);

  // Pipeline
  const pipelineWrap = document.createElement("div");
  pipelineWrap.className = "vis-pipeline-wrap";

  const pipeHead = document.createElement("div");
  pipeHead.className = "card__head";
  pipeHead.innerHTML = `<h2>Визуальный процесс: ${algo.toUpperCase()}</h2><div class="sub">пошаговое разложение с анимацией</div>`;
  pipelineWrap.appendChild(pipeHead);

  // Pipeline inline bar: methods | presets | settings
  const pipeBar = document.createElement("div");
  pipeBar.className = "vis-pipeline-bar";

  // Section 1: Methods
  const pipeSection1 = document.createElement("div");
  pipeSection1.className = "vis-pipe-section";
  const algoGroup = document.createElement("div");
  algoGroup.className = "vis-algo-group";
  const algos = [
    { id: "svd", label: "SVD" },
    { id: "pca", label: "PCA" },
    { id: "nmf", label: "NMF" },
    { id: "cur", label: "CUR" },
    { id: "als", label: "ALS" },
  ];
  for (const a of algos) {
    const btn = document.createElement("button");
    btn.className = `vis-algo-btn${algo === a.id ? " active" : ""}`;
    btn.textContent = a.label;
    btn.addEventListener("click", () => {
      state.visAlgo = a.id;
      renderVisualizerPage(container, state);
    });
    algoGroup.appendChild(btn);
  }
  pipeSection1.appendChild(algoGroup);
  pipeBar.appendChild(pipeSection1);

  // Section 2: Presets
  const pipeSection2 = document.createElement("div");
  pipeSection2.className = "vis-pipe-section";
  const presetGroup = document.createElement("div");
  presetGroup.className = "vis-preset-group";
  const presets = [
    { id: "identity", label: "Единичная" },
    { id: "zeros", label: "Нулевая" },
    { id: "random", label: "Случайная" },
    { id: "example", label: "Пример" },
  ];
  for (const p of presets) {
    const btn = document.createElement("button");
    btn.className = "vis-preset-btn";
    btn.textContent = p.label;
    btn.addEventListener("click", () => {
      const { m, n } = dims(state.visA || state.A);
      state.visA = applyPreset(p.id, m, n, visSeed);
      renderVisualizerPage(container, state);
    });
    presetGroup.appendChild(btn);
  }
  pipeSection2.appendChild(presetGroup);
  pipeBar.appendChild(pipeSection2);

  // Section 3: Settings (rank, iters, seed)
  const pipeSection3 = document.createElement("div");
  pipeSection3.className = "vis-pipe-section vis-pipe-settings";

  const rankGroup = document.createElement("div");
  rankGroup.className = "vis-inline-group";
  const rankLabel = document.createElement("span");
  rankLabel.textContent = "Ранг k:";
  const rankInput = document.createElement("input");
  rankInput.type = "number";
  rankInput.min = 1;
  rankInput.max = Math.min(A.length, A[0].length, 10);
  rankInput.value = k;
  rankInput.className = "vis-rank-input";
  rankInput.addEventListener("change", () => {
    state.visK = Math.max(1, Math.min(Number(rankInput.value) || 1, Math.min(A.length, A[0].length, 10)));
    renderVisualizerPage(container, state);
  });
  rankGroup.appendChild(rankLabel);
  rankGroup.appendChild(rankInput);
  pipeSection3.appendChild(rankGroup);

  const iterGroup = document.createElement("div");
  iterGroup.className = "vis-inline-group";
  const iterLabel = document.createElement("span");
  iterLabel.textContent = "Итерации:";
  const iterInput = document.createElement("input");
  iterInput.type = "number";
  iterInput.min = 1;
  iterInput.max = 50;
  iterInput.value = iters;
  iterInput.className = "vis-rank-input";
  iterInput.style.width = "60px";
  iterInput.addEventListener("change", () => {
    state.visIters = Math.max(1, Math.min(50, Number(iterInput.value) || 20));
    renderVisualizerPage(container, state);
  });
  iterGroup.appendChild(iterLabel);
  iterGroup.appendChild(iterInput);
  if (algo !== "nmf" && algo !== "als") iterGroup.style.display = "none";
  pipeSection3.appendChild(iterGroup);

  const seedGroup = document.createElement("div");
  seedGroup.className = "vis-inline-group";
  const seedLabel = document.createElement("span");
  seedLabel.textContent = "Seed:";
  const seedInput = document.createElement("input");
  seedInput.type = "number";
  seedInput.min = 0;
  seedInput.max = 999999;
  seedInput.value = visSeed;
  seedInput.className = "vis-rank-input";
  seedInput.style.width = "75px";
  seedInput.addEventListener("change", () => {
    state.visSeed = Math.max(0, Math.floor(Number(seedInput.value) || 0));
    renderVisualizerPage(container, state);
  });
  seedGroup.appendChild(seedLabel);
  seedGroup.appendChild(seedInput);
  pipeSection3.appendChild(seedGroup);

  pipeBar.appendChild(pipeSection3);

  pipelineWrap.appendChild(pipeBar);

  const pipeContainer = document.createElement("div");
  pipeContainer.className = "vis-pipeline";

  const steps = generatePipeline(algo, A, k, iters);
  for (let i = 0; i < steps.length; i++) {
    if (steps[i].type === "arrow") {
      renderArrow(pipeContainer, steps, i);
    } else {
      renderStep(pipeContainer, steps[i], i);
    }
  }

  pipelineWrap.appendChild(pipeContainer);
  body.appendChild(pipelineWrap);

  layout.appendChild(body);
  container.appendChild(layout);
  } catch (e) {
    container.innerHTML = `<div style="padding:1rem;color:#f87171;font-family:monospace;font-size:0.85rem">
      <strong>Ошибка рендера:</strong> ${e?.message || e}
      <pre style="margin-top:0.5rem;font-size:0.75rem;color:var(--muted)">${e?.stack || ""}</pre>
    </div>`;
    console.error("Visualizer error:", e);
  }
}
