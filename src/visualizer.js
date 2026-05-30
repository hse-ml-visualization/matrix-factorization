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
  const sigmaDiag = zeros(result.r, result.r);
  for (let i = 0; i < Sk.length && i < result.r; i++) sigmaDiag[i][i] = Sk[i];
  const meanRow = [mean.slice()];
  const cov = dot(Xt, X);

  const lambdaVals = Sk.map((s) => s * s);

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  steps.push({ type: "arrow", label: "Вычисляем средние μ по столбцам", op: "normalize", data: { type: "means", values: mean } });
  steps.push({ type: "matrix", id: "mean", title: "μ (средние)", subtitle: "По столбцам", data: meanRow, scale: minMax(meanRow) });
  steps.push({ type: "arrow", label: "Центрируем: X = A − μ", op: "transpose", data: { type: "subtraction", leftLabel: "A", left: A, rightLabel: "μ", right: meanRow, resultLabel: "X = A − μ", result: X } });
  steps.push({ type: "matrix", id: "X", title: "X = A − μ", subtitle: "Центрированные данные", data: X, scale: scaleX });
  steps.push({ type: "arrow", label: "Xᵀ·X (матрица рассеивания)", op: "multiply", data: { type: "matrix_product", leftLabel: "Xᵀ", left: Xt, rightLabel: "X", right: X, resultLabel: "Xᵀ·X", result: cov } });
  steps.push({ type: "matrix", id: "cov", title: "Xᵀ·X", subtitle: "Матрица рассеивания", data: cov, scale: null });
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
  const W0 = history.length > 0 ? history[0].W : W;
  const H0 = history.length > 0 ? history[0].H : H;
  const WHinit = dot(W0, H0);
  const midIdx = Math.min(Math.floor(history.length / 2), history.length - 1);
  const midHist = history.length > 2 ? history[midIdx] : null;

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  if (shift > 0) {
    steps.push({ type: "arrow", label: `Сдвиг на ${shift.toFixed(2)} для неотрицательности`, op: "normalize", data: { type: "iteration_note", text: `Все элементы A сдвигаются на ${shift.toFixed(2)}, чтобы минимальный элемент стал 0 (неотрицательность для NMF).` } });
    steps.push({ type: "matrix", id: "X", title: `A + ${shift.toFixed(2)}`, subtitle: "Неотрицательная", data: X, scale: minMax(X) });
  }
  steps.push({ type: "arrow", label: "Инициализируем W и H случайно", op: "init", data: { type: "init_matrices", matrices: [{ label: "W", matrix: W0 }, { label: "H", matrix: H0 }] } });
  steps.push({ type: "matrices_row", id: "init", matrices: [
    { title: "W", data: W0, subtitle: `${m}×${k}` },
    { title: "H", data: H0, subtitle: `${k}×${n}` },
  ]});
  steps.push({ type: "arrow", label: "Первое приближение: W·H", op: "multiply", data: { type: "matrix_product", leftLabel: "W", left: W0, rightLabel: "H", right: H0, resultLabel: "W·H", result: WHinit } });
  steps.push({ type: "matrix", id: "WHinit", title: "W·H (начало)", data: WHinit, scale: scaleA });
  if (history.length > 1) {
    steps.push({ type: "iteration_viewer", id: "iters", label: "Обновление W и H по итерациям", history, matrixLabels: ["W", "H"] });
  }
  steps.push({ type: "arrow", label: `Финальное W·H после ${cappedIters} итераций`, op: "multiply", data: { type: "matrix_product", leftLabel: "W", left: W, rightLabel: "H", right: H, resultLabel: "Ã ≈ W·H", result: Ahat } });
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
  steps.push({ type: "arrow", label: "Считаем квадраты норм строк и столбцов", op: "normalize", data: { type: "norm_computation", colNorms, rowNorms, topCols, topRows } });
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
  const X0 = history.length > 0 ? history[0].X : X;
  const Y0 = history.length > 0 ? history[0].Y : Y;
  const Y0t = transpose(Y0);
  const XYtInit = dot(X0, Y0t);
  const midIdx = Math.min(Math.floor(history.length / 2), history.length - 1);
  const midHist = history.length > 2 ? history[midIdx] : null;

  const steps = [];
  steps.push({ type: "matrix", id: "A", title: "A", subtitle: "Исходная матрица", data: A, scale: scaleA });
  steps.push({ type: "arrow", label: "Инициализируем X и Y случайно", op: "init", data: { type: "init_matrices", matrices: [{ label: "X", matrix: X0 }, { label: "Y", matrix: Y0 }] } });
  steps.push({ type: "matrices_row", id: "init", matrices: [
    { title: "X", data: X0, subtitle: `${m}×${r}` },
    { title: "Y", data: Y0, subtitle: `${n}×${r}` },
  ]});
  steps.push({ type: "arrow", label: "Начальное произведение: X·Yᵀ", op: "multiply", data: { type: "matrix_product", leftLabel: "X", left: X0, rightLabel: "Yᵀ", right: Y0t, resultLabel: "X·Yᵀ", result: XYtInit } });
  steps.push({ type: "matrix", id: "XYtInit", title: "X·Yᵀ (начало)", data: XYtInit, scale: scaleA });
  if (history.length > 1) {
    steps.push({ type: "iteration_viewer", id: "iters", label: "Итерации ALS", history, matrixLabels: ["X", "Y"] });
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
  const scale = step.scale || minMax(step.data);
  renderMatrixBlock(wrap, step.title, step.data, { scale, subtitle: step.subtitle });
  return wrap;
}

function renderMatricesRowStep(step) {
  const wrap = document.createElement("div");
  const inner = document.createElement("div");
  inner.className = "vis-matrices-row";
  for (const m of step.matrices) {
    const block = document.createElement("div");
    block.className = "vis-matrix-inline";
    const scale = m.scale || minMax(m.data);
    renderMatrixBlock(block, m.title, m.data, { scale, subtitle: m.subtitle });
    inner.appendChild(block);
  }
  wrap.appendChild(inner);
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
  colTitle.textContent = "Квадраты норм столбцов";
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
  rowTitle.textContent = "Квадраты норм строк";
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
  const fnA = frobNorm(step.A);
  const pct = fn / (fnA + 1e-12) * 100;

  const metric1 = document.createElement("div");
  metric1.className = "vis-error__metric";
  metric1.innerHTML = `<span class="vis-error__key">Относительная ошибка:</span><span class="vis-error__value">${pct.toFixed(2)}%</span>`;
  wrap.appendChild(metric1);

  const barWrap = document.createElement("div");
  barWrap.className = "vis-error__bar";
  const barFill = document.createElement("div");
  barFill.className = "vis-error__bar-fill";
  const hue = Math.max(0, Math.min(120, 120 - pct * 1.2));
  barFill.style.width = Math.min(pct, 100) + "%";
  barFill.style.background = `hsl(${hue}, 80%, 50%)`;
  barWrap.appendChild(barFill);
  wrap.appendChild(barWrap);

  const metric2 = document.createElement("div");
  metric2.className = "vis-error__metric";
  metric2.innerHTML = `<span class="vis-error__key">Frobenius:</span><span class="vis-error__value">${fn.toFixed(4)}</span>`;
  wrap.appendChild(metric2);

  return wrap;
}

function renderEditorStep(step) {
  const wrap = document.createElement("div");
  const A = step.A || step.data;
  const onChange = step.onChange;

  const head = document.createElement("div");
  head.style.cssText = "display:flex;align-items:center;gap:0.35rem;margin-bottom:0.2rem;flex-wrap:wrap;width:100%";
  const title = document.createElement("h3");
  title.style.cssText = "margin:0;font-size:0.82rem;color:var(--text)";
  title.textContent = step.title || "A";
  head.appendChild(title);
  const subtitle = document.createElement("span");
  subtitle.style.cssText = "font-size:0.62rem;color:var(--muted)";
  subtitle.textContent = step.subtitle || "Исходная";
  head.appendChild(subtitle);
  const normsMini = document.createElement("div");
  normsMini.style.cssText = "margin-left:auto;display:flex;align-self:center";
  renderRowColNorms(normsMini, A);
  head.appendChild(normsMini);
  wrap.appendChild(head);

  const scaleBar = document.createElement("div");
  scaleBar.className = "vis-scale-bar";
  const scaleGrad = document.createElement("div");
  scaleGrad.className = "vis-scale-gradient";
  const scaleLabels = document.createElement("div");
  scaleLabels.className = "vis-scale-labels";
  scaleLabels.innerHTML = `<span>мин</span><span>макс</span>`;
  scaleBar.appendChild(scaleGrad);
  scaleBar.appendChild(scaleLabels);
  wrap.appendChild(scaleBar);

  const editorHost = document.createElement("div");
  editorHost.className = "vis-editor-host";
  renderMatrixEditor(editorHost, A, onChange);
  wrap.appendChild(editorHost);

  const sizeRow = document.createElement("div");
  sizeRow.className = "vis-size-row";
  const rowInput = document.createElement("input");
  rowInput.type = "number";
  rowInput.min = 2;
  rowInput.max = 10;
  rowInput.value = A.length;
  rowInput.style.cssText = "width:44px;font-size:0.72rem;padding:0.12rem 0.2rem";
  const colInput = document.createElement("input");
  colInput.type = "number";
  colInput.min = 2;
  colInput.max = 10;
  colInput.value = A[0].length;
  colInput.style.cssText = "width:44px;font-size:0.72rem;padding:0.12rem 0.2rem";
  const sizeLabel = document.createElement("span");
  sizeLabel.className = "vis-size-label";
  sizeLabel.textContent = "Размер";
  const applySize = document.createElement("button");
  applySize.className = "vis-size-btn";
  applySize.textContent = "Применить";
  applySize.addEventListener("click", () => {
    const newM = Math.max(2, Math.min(10, Number(rowInput.value) || 2));
    const newN = Math.max(2, Math.min(10, Number(colInput.value) || 2));
    const newA = zeros(newM, newN);
    for (let i = 0; i < Math.min(newM, A.length); i++)
      for (let j = 0; j < Math.min(newN, A[0].length); j++)
        newA[i][j] = A[i][j];
    if (onChange) onChange(newA);
  });
  sizeRow.appendChild(sizeLabel);
  sizeRow.appendChild(rowInput);
  sizeRow.appendChild(document.createTextNode("×"));
  sizeRow.appendChild(colInput);
  sizeRow.appendChild(applySize);
  wrap.appendChild(sizeRow);

  return wrap;
}

function renderIterationViewerStep(step) {
  const history = step.history;
  const labels = step.matrixLabels || ["W", "H"];
  const isNmf = labels[0] === "W";
  const maxFrob = Math.max(...history.map(h => h.frob), 1e-12);
  const totalFrames = history.length;
  const wrap = document.createElement("div");

  // Title
  const title = document.createElement("div");
  title.className = "vis-iterview__title";
  title.textContent = step.label || "Итерации";
  wrap.appendChild(title);

  // Info line
  const info = document.createElement("div");
  info.className = "vis-iterview__info";
  wrap.appendChild(info);

  // Matrix hosts side by side
  const row = document.createElement("div");
  row.className = "vis-iterview__row";
  const host1 = document.createElement("div");
  const host2 = document.createElement("div");
  row.appendChild(host1);
  row.appendChild(host2);
  wrap.appendChild(row);

  // Error bar
  const barWrap = document.createElement("div");
  barWrap.className = "vis-iterview__bar";
  const barFill = document.createElement("div");
  barFill.className = "vis-iterview__bar-fill";
  barWrap.appendChild(barFill);
  wrap.appendChild(barWrap);

  function renderFrame(n) {
    const entry = history[n];
    const frob = entry.frob;
    const pct = Math.min(frob / maxFrob * 100, 100);
    info.innerHTML = `<b>${labels[0]}</b>, <b>${labels[1]}</b> &nbsp;итерация <b>${entry.i}</b> &nbsp;|&nbsp; ошибка: ${frob.toFixed(4)}`;
    barFill.style.width = pct + "%";
    barFill.style.background = `hsl(${120 - pct * 1.2}, 80%, 50%)`;

    host1.innerHTML = "";
    host2.innerHTML = "";
    const mat1 = isNmf ? entry.W : entry.X;
    const mat2 = isNmf ? entry.H : entry.Y;
    renderMatrixBlock(host1, labels[0], mat1, { scale: minMax(mat1) });
    renderMatrixBlock(host2, labels[1], mat2, { scale: minMax(mat2) });
  }

  createFrameSlider(wrap, totalFrames, renderFrame, { speed: 400 });
  return wrap;
}

function renderStep(container, step, index) {
  const stepEl = document.createElement("div");
  stepEl.className = "vis-step";

  let inner;
  if (step.editable && step.type === "matrix") {
    inner = renderEditorStep(step);
  } else {
    switch (step.type) {
      case "matrix":
        inner = renderMatrixStep(step);
        break;
      case "matrices_row":
        inner = renderMatricesRowStep(step);
        break;
      case "eigenvalues":
        inner = renderEigenvaluesStep(step);
        break;
      case "norms":
        inner = renderNormsStep(step);
        break;
      case "error":
        inner = renderErrorStep(step);
        break;
      case "iteration_viewer":
        inner = renderIterationViewerStep(step);
        break;
      case "arrow":
        inner = document.createElement("div");
        inner.className = "vis-arrow-note";
        inner.textContent = step.label || "";
        break;
      default:
        inner = document.createElement("div");
        inner.textContent = "Unknown step type";
    }
  }

  const content = document.createElement("div");
  content.className = "vis-step__content";
  content.appendChild(inner);
  stepEl.appendChild(content);
  container.appendChild(stepEl);
  return stepEl;
}

function renderArrow(container, steps, arrowIndex) {
  const arrowStep = steps[arrowIndex];
  const arrowEl = document.createElement("div");
  arrowEl.className = "vis-arrow";
  arrowEl.style.animationDelay = `0s`;

  const color = opColor(arrowStep.op);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "vis-arrow__svg");
  svg.setAttribute("viewBox", "0 0 60 30");
  svg.setAttribute("preserveAspectRatio", "none");

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", "4");
  line.setAttribute("y1", "15");
  line.setAttribute("x2", "46");
  line.setAttribute("y2", "15");
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", "2");
  line.setAttribute("stroke-linecap", "round");
  line.setAttribute("class", "vis-arrow__line");

  const head = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  head.setAttribute("points", "42,10 52,15 42,20");
  head.setAttribute("fill", color);
  head.setAttribute("class", "vis-arrow__head");

  svg.appendChild(line);
  svg.appendChild(head);

  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("cx", "4");
  dot.setAttribute("cy", "15");
  dot.setAttribute("r", "3");
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

// ── Frame slider for scrubbing animations ──

function createFrameSlider(body, totalFrames, renderFrame, opts = {}) {
  if (totalFrames <= 1) {
    renderFrame(0);
    return null;
  }

  const wrap = document.createElement("div");
  wrap.className = "frame-slider";

  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;gap:0.4rem;padding:0.4rem 0.3rem;flex-wrap:wrap";

  const playBtn = document.createElement("button");
  playBtn.textContent = "▶";
  playBtn.title = "Воспроизвести";
  playBtn.style.cssText = "padding:0.2rem 0.5rem;font-size:0.82rem;font-weight:600;cursor:pointer;border:1px solid var(--accent);border-radius:6px;background:rgba(91,156,246,0.15);color:var(--accent);min-width:2.2rem";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = 0;
  slider.max = totalFrames - 1;
  slider.value = 0;
  slider.style.cssText = "flex:1;min-width:80px;accent-color:var(--accent)";

  const frameLabel = document.createElement("span");
  frameLabel.style.cssText = "font-size:0.72rem;color:var(--muted);font-family:monospace;min-width:5rem;text-align:center";
  frameLabel.textContent = `0 / ${totalFrames - 1}`;

  const resetBtn = document.createElement("button");
  resetBtn.textContent = "⟳";
  resetBtn.title = "Сброс";
  resetBtn.style.cssText = "padding:0.2rem 0.5rem;font-size:0.82rem;font-weight:600;cursor:pointer;border:1px solid var(--border);border-radius:6px;background:rgba(255,255,255,0.05);color:var(--muted);min-width:2.2rem";

  row.appendChild(playBtn);
  row.appendChild(slider);
  row.appendChild(frameLabel);
  row.appendChild(resetBtn);
  wrap.appendChild(row);

  let playing = false;
  let interval = null;
  let currentFrame = 0;

  function setFrame(n) {
    currentFrame = n;
    slider.value = n;
    frameLabel.textContent = `${n} / ${totalFrames - 1}`;
    renderFrame(n);
  }

  slider.addEventListener("input", () => {
    if (playing) {
      playing = false;
      clearInterval(interval);
      playBtn.textContent = "▶";
    }
    setFrame(Number(slider.value));
  });

  playBtn.addEventListener("click", () => {
    if (playing) {
      playing = false;
      clearInterval(interval);
      playBtn.textContent = "▶";
    } else {
      playing = true;
      playBtn.textContent = "⏸";
      const speed = opts.speed || 400;
      interval = setInterval(() => {
        if (currentFrame >= totalFrames - 1) {
          playing = false;
          clearInterval(interval);
          playBtn.textContent = "▶";
          return;
        }
        setFrame(currentFrame + 1);
      }, speed);
    }
  });

  resetBtn.addEventListener("click", () => {
    if (playing) {
      playing = false;
      clearInterval(interval);
      playBtn.textContent = "▶";
    }
    setFrame(0);
  });

  body.appendChild(wrap);
  setFrame(0);

  return { setFrame, currentFrame: () => currentFrame, slider, playBtn };
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



function liveProduct(body, data) {
  const A = data.left, B = data.right, R = data.result;
  const mA = A.length, p = A[0].length, nB = B[0].length;
  const sA = minMax(A), sB = minMax(B), sR = minMax(R);

  const cellData = [];
  for (let i = 0; i < mA; i++) {
    for (let j = 0; j < nB; j++) {
      let sum = 0;
      const parts = [];
      for (let k = 0; k < p; k++) {
        const term = A[i][k] * B[k][j];
        sum += term;
        parts.push(`${A[i][k].toFixed(2)}·${B[k][j].toFixed(2)}`);
      }
      cellData.push({ i, j, value: sum, formula: parts.join(" + ") });
    }
  }
  const totalFrames = cellData.length + 1;

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

  const allCellsA = hostA.querySelectorAll(".cell");
  const allCellsB = hostB.querySelectorAll(".cell");

  function renderFrame(n) {
    const cellsA = allCellsA;
    const cellsB = allCellsB;
    const cellsR = rHost.querySelectorAll(".cell");
    for (let ci = 0; ci < mA; ci++) {
      for (let cj = 0; cj < nB; cj++) {
        const idx = ci * nB + cj;
        if (!cellsR[idx]) continue;
        if (idx < n) {
          const cd = cellData[idx];
          working[ci][cj] = cd.value;
          const t = sR.mx > sR.mn ? (cd.value - sR.mn) / (sR.mx - sR.mn) : 0.5;
          const c = viridis(t);
          cellsR[idx].style.background = viridisRgb(c);
          cellsR[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
          cellsR[idx].textContent = cd.value.toFixed(2);
        }
        cellsR[idx].style.boxShadow = "";
      }
    }
    for (let ci = 0; ci < mA * p; ci++) cellsA[ci] ? cellsA[ci].style.boxShadow = "" : 0;
    for (let ci = 0; ci < p * nB; ci++) cellsB[ci] ? cellsB[ci].style.boxShadow = "" : 0;
    if (n > 0) {
      const last = cellData[n - 1];
      const lastIdx = (n - 1);
      if (cellsR[lastIdx]) {
        cellsR[lastIdx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
      }
      for (let k = 0; k < p; k++) {
        const aIdx = last.i * p + k;
        if (cellsA[aIdx]) cellsA[aIdx].style.boxShadow = "inset 0 0 0 3px var(--good)";
        const bIdx = k * nB + last.j;
        if (cellsB[bIdx]) cellsB[bIdx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
      }
      trace.innerHTML = `${data.resultLabel}<sub>${last.i}${last.j}</sub> = ${last.formula} = <b>${last.value.toFixed(2)}</b>`;
      info.textContent = `(${last.i},${last.j}): Σ = ${last.value.toFixed(2)}`;
    } else {
      trace.innerHTML = "";
      info.textContent = "";
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 300 });
}

function liveTriple(body, data) {
  const A = data.a, B = data.b, C = data.c, R = data.result;
  const mA = A.length, p = A[0].length, q = B[0].length, nC = C[0].length;
  const sA = minMax(A), sB = minMax(B), sC = minMax(C), sR = minMax(R);
  const temp = dot(A, B);
  const sT = minMax(temp);

  const cellData1 = [];
  for (let i = 0; i < mA; i++) {
    for (let j = 0; j < q; j++) {
      let sum = 0;
      const parts = [];
      for (let k = 0; k < p; k++) {
        const term = A[i][k] * B[k][j];
        sum += term;
        parts.push(`${A[i][k].toFixed(2)}·${B[k][j].toFixed(2)}`);
      }
      cellData1.push({ i, j, value: sum, formula: parts.join(" + ") });
    }
  }

  if (C.length < q) throw new Error("Dimension mismatch: C rows < q");
  const cellData2 = [];
  for (let i = 0; i < mA; i++) {
    for (let j = 0; j < nC; j++) {
      let sum = 0;
      const parts = [];
      for (let k = 0; k < q; k++) {
        const term = temp[i][k] * C[k][j];
        sum += term;
        parts.push(`${temp[i][k].toFixed(2)}·${C[k][j].toFixed(2)}`);
      }
      cellData2.push({ i, j, value: sum, formula: parts.join(" + ") });
    }
  }

  const phase1Len = cellData1.length;
  const totalFrames = phase1Len + cellData2.length + 1;

  // Phase 1: U · Σ → temp
  const phase1Label = document.createElement("div");
  phase1Label.style.cssText = "font-size:0.82rem;color:var(--accent);font-weight:600;text-align:center;margin:0.3rem 0";
  phase1Label.textContent = "Фаза 1: U · Σ → промежуточная";
  body.appendChild(phase1Label);

  const row1 = document.createElement("div");
  row1.style.cssText = "display:flex;gap:0.5rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.3rem";

  const wrapA1 = document.createElement("div");
  wrapA1.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblA1 = document.createElement("div");
  lblA1.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblA1.textContent = data.aLabel;
  wrapA1.appendChild(lblA1);
  const hostA1 = document.createElement("div");
  renderMatrixBlock(hostA1, "", A, { scale: sA });
  wrapA1.appendChild(hostA1);
  row1.appendChild(wrapA1);
  row1.appendChild(makeOpSign("·"));

  const wrapB1 = document.createElement("div");
  wrapB1.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblB1 = document.createElement("div");
  lblB1.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblB1.textContent = data.bLabel;
  wrapB1.appendChild(lblB1);
  const hostB1 = document.createElement("div");
  renderMatrixBlock(hostB1, "", B, { scale: sB });
  wrapB1.appendChild(hostB1);
  row1.appendChild(wrapB1);
  row1.appendChild(makeOpSign("="));

  const hostT = document.createElement("div");
  hostT.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblT = document.createElement("div");
  lblT.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblT.textContent = "U·Σ";
  hostT.appendChild(lblT);
  const rHostT = document.createElement("div");
  const working1 = zeros(mA, q);
  renderMatrixBlock(rHostT, "", working1, { scale: sT });
  rHostT.dataset.rows = mA;
  rHostT.dataset.cols = q;
  rHostT.dataset.scale = JSON.stringify(sT);
  hostT.appendChild(rHostT);
  row1.appendChild(hostT);
  body.appendChild(row1);

  // Phase 2: temp · Vᵀ → result
  const phase2Wrap = document.createElement("div");
  phase2Wrap.style.cssText = "display:contents";
  const phase2Label = document.createElement("div");
  phase2Label.style.cssText = "font-size:0.82rem;color:var(--accent);font-weight:600;text-align:center;margin:0.3rem 0";
  phase2Label.textContent = "Фаза 2: (U·Σ) · Vᵀ → результат";
  phase2Wrap.appendChild(phase2Label);

  const row2 = document.createElement("div");
  row2.style.cssText = "display:flex;gap:0.5rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.3rem";

  const hostA2 = document.createElement("div");
  renderMatrixBlock(hostA2, "", temp, { scale: sT });
  row2.appendChild(hostA2);
  row2.appendChild(makeOpSign("·"));

  const wrapB2 = document.createElement("div");
  wrapB2.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblB2 = document.createElement("div");
  lblB2.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblB2.textContent = data.cLabel;
  wrapB2.appendChild(lblB2);
  const hostB2 = document.createElement("div");
  renderMatrixBlock(hostB2, "", C, { scale: sC });
  wrapB2.appendChild(hostB2);
  row2.appendChild(wrapB2);
  row2.appendChild(makeOpSign("="));

  const hostR2 = document.createElement("div");
  hostR2.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:0.25rem";
  const lblR2 = document.createElement("div");
  lblR2.style.cssText = "font-size:0.78rem;font-weight:600;color:var(--text);text-align:center";
  lblR2.textContent = data.resultLabel;
  hostR2.appendChild(lblR2);
  const rHostR2 = document.createElement("div");
  const working2 = zeros(mA, nC);
  renderMatrixBlock(rHostR2, "", working2, { scale: sR });
  rHostR2.dataset.rows = mA;
  rHostR2.dataset.cols = nC;
  rHostR2.dataset.scale = JSON.stringify(sR);
  hostR2.appendChild(rHostR2);
  row2.appendChild(hostR2);
  phase2Wrap.appendChild(row2);

  const trace2 = document.createElement("div");
  trace2.style.cssText = "font-size:0.85rem;color:var(--text);text-align:center;font-family:monospace;min-height:2rem;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin:0.3rem 0";
  phase2Wrap.appendChild(trace2);

  const info2 = document.createElement("div");
  info2.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
  phase2Wrap.appendChild(info2);

  body.appendChild(phase2Wrap);
  phase2Wrap.style.display = "none";

  // Trace/info for both phases
  const trace1 = document.createElement("div");
  trace1.style.cssText = "font-size:0.85rem;color:var(--text);text-align:center;font-family:monospace;min-height:2rem;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin:0.3rem 0";
  body.appendChild(trace1);

  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2rem";
  body.appendChild(info);

  const cellsA1 = hostA1.querySelectorAll(".cell");
  const cellsB1 = hostB1.querySelectorAll(".cell");

  function fillHost(host, cellData, count) {
    const cells = host.querySelectorAll(".cell");
    const rows = parseInt(host.dataset.rows) || (cellData.length > 0 ? Math.max(...cellData.map(c => c.i)) + 1 : 1);
    const cols = parseInt(host.dataset.cols) || (cellData.length > 0 ? Math.max(...cellData.map(c => c.j)) + 1 : 1);
    const scale = host.dataset.scale ? JSON.parse(host.dataset.scale) : null;
    for (let ci = 0; ci < cellData.length; ci++) {
      const cd = cellData[ci];
      const idx = cd.i * cols + cd.j;
      if (!cells[idx]) continue;
      if (ci < count) {
        const t2 = scale && scale.mx > scale.mn ? (cd.value - scale.mn) / (scale.mx - scale.mn) : 0.5;
        const c2 = viridis(t2);
        cells[idx].style.background = viridisRgb(c2);
        cells[idx].style.color = t2 > 0.6 ? "#080c14" : "#fff";
        cells[idx].textContent = cd.value.toFixed(2);
        cells[idx].style.boxShadow = ci === count - 1 ? "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)" : "";
      } else {
        cells[idx].style.background = "";
        cells[idx].style.color = "";
        cells[idx].textContent = "";
        cells[idx].style.boxShadow = "";
      }
    }
  }

  function clearSourceHighlights() {
    for (let ci = 0; ci < mA * p; ci++) if (cellsA1[ci]) cellsA1[ci].style.boxShadow = "";
    for (let ci = 0; ci < p * q; ci++) if (cellsB1[ci]) cellsB1[ci].style.boxShadow = "";
  }

  function highlightPhase1(i, j) {
    for (let k = 0; k < p; k++) {
      const aIdx = i * p + k;
      if (cellsA1[aIdx]) cellsA1[aIdx].style.boxShadow = "inset 0 0 0 3px var(--good)";
      const bIdx = k * q + j;
      if (cellsB1[bIdx]) cellsB1[bIdx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
    }
  }

  const cellsA2 = hostA2.querySelectorAll(".cell");
  const cellsB2 = hostB2.querySelectorAll(".cell");

  function clearPhase2Highlights() {
    for (let ci = 0; ci < mA * q; ci++) if (cellsA2[ci]) cellsA2[ci].style.boxShadow = "";
    for (let ci = 0; ci < q * nC; ci++) if (cellsB2[ci]) cellsB2[ci].style.boxShadow = "";
  }

  function highlightPhase2(i, j) {
    for (let k = 0; k < q; k++) {
      const aIdx = i * q + k;
      if (cellsA2[aIdx]) cellsA2[aIdx].style.boxShadow = "inset 0 0 0 3px var(--good)";
      const bIdx = k * nC + j;
      if (cellsB2[bIdx]) cellsB2[bIdx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
    }
  }

  function renderFrame(n) {
    if (n === 0) {
      phase1Label.style.opacity = "1";
      phase2Label.style.opacity = "0.3";
      phase2Wrap.style.display = "none";
      fillHost(rHostT, cellData1, 0);
      fillHost(rHostR2, cellData2, 0);
      trace1.innerHTML = "";
      info.textContent = "";
      clearSourceHighlights();
      clearPhase2Highlights();
      return;
    }
    if (n <= phase1Len) {
      phase1Label.style.opacity = "1";
      phase2Label.style.opacity = "0.3";
      phase2Wrap.style.display = "none";
      fillHost(rHostT, cellData1, n);
      fillHost(rHostR2, cellData2, 0);
      clearSourceHighlights();
      clearPhase2Highlights();
      const last = cellData1[n - 1];
      highlightPhase1(last.i, last.j);
      trace1.innerHTML = `(U·Σ)<sub>${last.i}${last.j}</sub> = ${last.formula} = <b>${last.value.toFixed(2)}</b>`;
      info.textContent = `Фаза 1 (${n}/${phase1Len}): (${last.i},${last.j}) Σ = ${last.value.toFixed(2)}`;
    } else {
      phase1Label.style.opacity = "0.6";
      phase2Label.style.opacity = "1";
      phase2Wrap.style.display = "";
      fillHost(rHostT, cellData1, phase1Len);
      clearSourceHighlights();
      const phase2n = n - phase1Len;
      fillHost(rHostR2, cellData2, phase2n);
      clearPhase2Highlights();
      if (phase2n > 0) {
        const last = cellData2[phase2n - 1];
        highlightPhase2(last.i, last.j);
        trace2.innerHTML = `${data.resultLabel}<sub>${last.i}${last.j}</sub> = ${last.formula} = <b>${last.value.toFixed(2)}</b>`;
        info2.textContent = `Фаза 2 (${phase2n}/${cellData2.length}): (${last.i},${last.j}) Σ = ${last.value.toFixed(2)}`;
      }
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 250 });
}

function liveAddition(body, data) {
  const A = data.left, B = data.right, R = data.result;
  const m = A.length, n = A[0].length;
  const sA = minMax(A), sR = minMax(R);
  const working = zeros(m, n);

  const cellData = [];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const vB = B.length === 1 ? (B[0] ? B[0][j] || 0 : 0) : (B[i] ? B[i][j] || 0 : 0);
      cellData.push({ i, j, value: A[i][j] + vB, vA: A[i][j], vB });
    }
  }
  const totalFrames = cellData.length + 1;

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

  function renderFrame(frame) {
    for (let ci = 0; ci < m; ci++) {
      for (let cj = 0; cj < n; cj++) {
        const idx = ci * n + cj;
        const cellsR = rHost.querySelectorAll(".cell");
        if (!cellsR[idx]) continue;
        if (idx < frame) {
          const cd = cellData[idx];
          working[ci][cj] = cd.value;
          const t = sR.mx > sR.mn ? (cd.value - sR.mn) / (sR.mx - sR.mn) : 0.5;
          const c = viridis(Math.max(0, Math.min(1, t)));
          cellsR[idx].style.background = viridisRgb(c);
          cellsR[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
          cellsR[idx].textContent = cd.value.toFixed(2);
          cellsR[idx].style.boxShadow = "";
          if (cellsA[idx]) cellsA[idx].style.boxShadow = "";
          if (cellsB[idx]) cellsB[idx].style.boxShadow = "";
        } else {
          working[ci][cj] = 0;
          cellsR[idx].style.background = "";
          cellsR[idx].style.color = "";
          cellsR[idx].textContent = "";
          cellsR[idx].style.boxShadow = "";
          if (cellsA[idx]) cellsA[idx].style.boxShadow = "";
          if (cellsB[idx]) cellsB[idx].style.boxShadow = "";
        }
      }
    }
    if (frame > 0) {
      const last = cellData[frame - 1];
      const cellsR = rHost.querySelectorAll(".cell");
      const lastIdx = (frame - 1);
      if (cellsA[lastIdx]) cellsA[lastIdx].style.boxShadow = "inset 0 0 0 3px var(--good)";
      if (cellsB[lastIdx]) cellsB[lastIdx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
      if (cellsR[lastIdx]) cellsR[lastIdx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
      trace.innerHTML = `${data.leftLabel}<sub>${last.i}${last.j}</sub> + ${data.rightLabel}<sub>${last.i}${last.j}</sub> = ${last.vA.toFixed(2)} + ${last.vB.toFixed(2)} = <b>${last.value.toFixed(2)}</b>`;
      info.textContent = `(${last.i},${last.j}): ${last.vA.toFixed(2)} + ${last.vB.toFixed(2)} = ${last.value.toFixed(2)}`;
    } else {
      trace.innerHTML = "";
      info.textContent = "";
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 300 });
}

function liveSubtraction(body, data) {
  const A = data.left, B = data.right, R = data.result;
  const m = A.length, n = A[0].length;
  const sA = minMax(A), sR = minMax(R);
  const working = zeros(m, n);

  const cellData = [];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const vB = B.length === 1 ? (B[0] ? B[0][j] || 0 : 0) : (B[i] ? B[i][j] || 0 : 0);
      cellData.push({ i, j, value: A[i][j] - vB, vA: A[i][j], vB });
    }
  }
  const totalFrames = cellData.length + 1;

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

  function renderFrame(frame) {
    for (let ci = 0; ci < m; ci++) {
      for (let cj = 0; cj < n; cj++) {
        const idx = ci * n + cj;
        const cellsR = rHost.querySelectorAll(".cell");
        if (!cellsR[idx]) continue;
        if (idx < frame) {
          const cd = cellData[idx];
          working[ci][cj] = cd.value;
          const t = sR.mx > sR.mn ? (cd.value - sR.mn) / (sR.mx - sR.mn) : 0.5;
          const c = viridis(Math.max(0, Math.min(1, t)));
          cellsR[idx].style.background = viridisRgb(c);
          cellsR[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
          cellsR[idx].textContent = cd.value.toFixed(2);
          cellsR[idx].style.boxShadow = "";
          if (cellsA[idx]) cellsA[idx].style.boxShadow = "";
          if (cellsB[idx]) cellsB[idx].style.boxShadow = "";
        } else {
          working[ci][cj] = 0;
          cellsR[idx].style.background = "";
          cellsR[idx].style.color = "";
          cellsR[idx].textContent = "";
          cellsR[idx].style.boxShadow = "";
          if (cellsA[idx]) cellsA[idx].style.boxShadow = "";
          if (cellsB[idx]) cellsB[idx].style.boxShadow = "";
        }
      }
    }
    if (frame > 0) {
      const last = cellData[frame - 1];
      const cellsR = rHost.querySelectorAll(".cell");
      const lastIdx = (frame - 1);
      if (cellsA[lastIdx]) cellsA[lastIdx].style.boxShadow = "inset 0 0 0 3px var(--good)";
      if (cellsB[lastIdx]) cellsB[lastIdx].style.boxShadow = "inset 0 0 0 3px var(--bad)";
      if (cellsR[lastIdx]) cellsR[lastIdx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
      trace.innerHTML = `${data.leftLabel}<sub>${last.i}${last.j}</sub> − ${data.rightLabel}<sub>${last.i}${last.j}</sub> = ${last.vA.toFixed(2)} − ${last.vB.toFixed(2)} = <b>${last.value.toFixed(2)}</b>`;
      info.textContent = `(${last.i},${last.j}): ${last.vA.toFixed(2)} − ${last.vB.toFixed(2)} = ${last.value.toFixed(2)}`;
    } else {
      trace.innerHTML = "";
      info.textContent = "";
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 300 });
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

  const totalFrames = cards.length + 1;
  function renderFrame(n) {
    for (let ci = 0; ci < cards.length; ci++) {
      if (ci < n) {
        cards[ci].style.opacity = "1";
        cards[ci].style.transform = "translateY(0)";
      } else {
        cards[ci].style.opacity = "0";
        cards[ci].style.transform = "translateY(8px)";
      }
    }
    if (n > 0 && n <= vals.length) {
      info.textContent = `μ${n} = ${Number.isInteger(vals[n-1]) ? vals[n-1] : vals[n-1].toFixed(4)}`;
    } else {
      info.textContent = "";
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 350 });
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

  const totalFrames = cards.length + 1;
  function renderFrame(n) {
    for (let ci = 0; ci < cards.length; ci++) {
      if (ci < n) {
        cards[ci].style.opacity = "1";
        cards[ci].style.transform = "scale(1)";
        const glow = cards[ci].querySelector("div:last-child");
        if (glow && ci === n - 1) glow.style.textShadow = "0 0 12px rgba(167,139,250,0.6)";
        else if (glow) glow.style.textShadow = "none";
      } else {
        cards[ci].style.opacity = "0";
        cards[ci].style.transform = "scale(0.8)";
      }
    }
    if (n > 0 && n <= vals.length) {
      info.textContent = `${vals[n-1].label} = ${Number.isInteger(vals[n-1].value) ? vals[n-1].value : vals[n-1].value.toFixed(4)}`;
    } else {
      info.textContent = "";
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 350 });
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

  const totalFrames = cards.length + 1;
  function renderFrame(n) {
    for (let ci = 0; ci < cards.length; ci++) {
      cards[ci].style.opacity = ci < n ? "1" : "0";
    }
    if (n > 0 && n <= cards.length) {
      info.textContent = `σ${n} = √${from[n-1].toFixed(4)} = ${to[n-1].toFixed(4)}`;
    } else {
      info.textContent = "";
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 350 });
}

function liveNorms(body, data) {
  const formula = document.createElement("div");
  formula.style.cssText = "font-size:0.85rem;color:var(--accent);text-align:center;font-family:monospace;padding:0.3rem;background:rgba(0,0,0,0.12);border-radius:6px;margin-bottom:0.3rem";
  formula.innerHTML = "||col<sub>j</sub>||² = Σ<sub>i</sub> A<sub>ij</sub>² &nbsp;&nbsp; ||row<sub>i</sub>||² = Σ<sub>j</sub> A<sub>ij</sub>²";
  body.appendChild(formula);

  const fills = [];
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
      fill.style.cssText = `height:14px;width:0%;background:${top.includes(i)?"var(--accent)":"var(--border)"};border-radius:4px;transition:width 0.2s ease-out;min-width:0px`;
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

  const totalFrames = fills.length + 1;
  function renderFrame(n) {
    for (let fi = 0; fi < fills.length; fi++) {
      if (fi < n) {
        fills[fi].el.style.width = fills[fi].target + "%";
        fills[fi].el.style.boxShadow = fi === n - 1 ? "0 0 8px rgba(91,156,246,0.5)" : "none";
      } else {
        fills[fi].el.style.width = "0%";
        fills[fi].el.style.boxShadow = "none";
      }
    }
    if (n > 0 && n <= fills.length) {
      const f = fills[n - 1];
      info.textContent = `${f.prefix}[${f.idx}]² = ${f.val.toFixed(2)}`;
    } else {
      info.textContent = "";
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 250 });
}

function liveSelection(body, data) {
  const info = document.createElement("div");
  info.style.cssText = "font-size:0.82rem;color:var(--text);text-align:center;margin-bottom:0.4rem";
  info.innerHTML = `Строки: [${data.topRows.join(", ")}] &nbsp; Столбцы: [${data.topCols.join(", ")}]`;
  body.appendChild(info);

  const matDefs = [
    { label: "C", M: data.C, cols: data.C[0].length, s: minMax(data.C) },
    { label: "R", M: data.R, cols: data.R[0].length, s: minMax(data.R) },
    { label: "W", M: data.W, cols: data.W[0].length, s: minMax(data.W) },
  ];
  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.8rem;align-items:center;justify-content:center;flex-wrap:wrap";
  const hosts = [];
  const allCells = [];
  let maxCells = 0;
  for (const def of matDefs) {
    const host = renderSandboxMatrix(row, def.label, zeros(def.M.length, def.M[0].length), def.s);
    const cells = [];
    for (let i = 0; i < def.M.length; i++) {
      for (let j = 0; j < def.M[0].length; j++) {
        cells.push({ i, j, value: def.M[i][j], cols: def.M[0].length, s: def.s });
      }
    }
    allCells.push({ cells, host, cols: def.M[0].length });
    maxCells = Math.max(maxCells, cells.length);
    hosts.push(host);
  }
  body.appendChild(row);
  const info2 = document.createElement("div");
  info2.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;margin-top:0.3rem";
  body.appendChild(info2);

  const totalFrames = maxCells + 1;
  function renderFrame(n) {
    for (let hi = 0; hi < allCells.length; hi++) {
      const { cells, host, cols } = allCells[hi];
      const hostCells = host.querySelectorAll(".cell");
      for (let ci = 0; ci < cells.length; ci++) {
        const cd = cells[ci];
        const idx = cd.i * cols + cd.j;
        if (!hostCells[idx]) continue;
        if (ci < n) {
          const t = cd.s.mx > cd.s.mn ? (cd.value - cd.s.mn) / (cd.s.mx - cd.s.mn) : 0.5;
          const c = viridis(t);
          hostCells[idx].style.background = viridisRgb(c);
          hostCells[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
          hostCells[idx].textContent = cd.value.toFixed(2);
          hostCells[idx].style.boxShadow = ci === n - 1 ? "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)" : "";
        } else {
          hostCells[idx].style.background = "";
          hostCells[idx].style.color = "";
          hostCells[idx].textContent = "";
          hostCells[idx].style.boxShadow = "";
        }
      }
    }
    info2.textContent = n === 0 ? "" : (n >= maxCells ? "Готово" : `Заполнено ${n}/${maxCells}`);
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 200 });
}

function liveInit(body, data) {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.8rem;align-items:center;justify-content:center;flex-wrap:wrap";
  const matDefs = [];
  for (const m of data.matrices) {
    matDefs.push({ label: m.label, M: m.matrix, cols: m.matrix[0].length, s: minMax(m.matrix) });
  }
  const allCells = [];
  let maxCells = 0;
  for (const def of matDefs) {
    const host = renderSandboxMatrix(row, def.label, zeros(def.M.length, def.M[0].length), def.s);
    const cells = [];
    for (let i = 0; i < def.M.length; i++) {
      for (let j = 0; j < def.M[0].length; j++) {
        cells.push({ i, j, value: def.M[i][j], cols: def.M[0].length, s: def.s });
      }
    }
    allCells.push({ cells, host, cols: def.M[0].length });
    maxCells = Math.max(maxCells, cells.length);
  }
  body.appendChild(row);
  const info = document.createElement("div");
  info.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;margin-top:0.3rem";
  body.appendChild(info);

  const totalFrames = maxCells + 1;
  function renderFrame(n) {
    for (let hi = 0; hi < allCells.length; hi++) {
      const { cells, host, cols } = allCells[hi];
      const hostCells = host.querySelectorAll(".cell");
      for (let ci = 0; ci < cells.length; ci++) {
        const cd = cells[ci];
        const idx = cd.i * cols + cd.j;
        if (!hostCells[idx]) continue;
        if (ci < n) {
          const t = cd.s.mx > cd.s.mn ? (cd.value - cd.s.mn) / (cd.s.mx - cd.s.mn) : 0.5;
          const c = viridis(t);
          hostCells[idx].style.background = viridisRgb(c);
          hostCells[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
          hostCells[idx].textContent = cd.value.toFixed(2);
          hostCells[idx].style.boxShadow = ci === n - 1 ? "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)" : "";
        } else {
          hostCells[idx].style.background = "";
          hostCells[idx].style.color = "";
          hostCells[idx].textContent = "";
          hostCells[idx].style.boxShadow = "";
        }
      }
    }
    info.textContent = n === 0 ? "" : (n >= maxCells ? "Готово" : `Заполнено ${n}/${maxCells}`);
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 200 });
}

function liveIterationHistory(body, data) {
  const history = data.history;
  const scaleA = data.scale || minMax(data.history[0] ? data.history[0].Ahat : [[0]]);
  const maxFrob = Math.max(...history.map(h => h.frob), 1e-12);
  const label = data.label || "Итерация";

  const totalFrames = history.length + 1;

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
  barFill.style.cssText = "height:100%;width:0%;border-radius:4px;transition:width 0.3s";
  barWrap.appendChild(barFill);
  body.appendChild(barWrap);

  const matrixHost = document.createElement("div");
  matrixHost.style.cssText = "display:flex;justify-content:center";
  body.appendChild(matrixHost);

  function renderFrame(n) {
    if (n === 0) {
      info.innerHTML = "";
      barFill.style.width = "0%";
      matrixHost.innerHTML = "";
      return;
    }
    const entry = history[n - 1];
    const frob = entry.frob;
    const pct = Math.min(frob / maxFrob * 100, 100);
    info.innerHTML = `${label} <b>${entry.i}</b> &nbsp;|&nbsp; Ошибка: ${frob.toFixed(4)}`;
    barFill.style.width = pct + "%";
    barFill.style.background = `hsl(${120 - pct * 1.2}, 80%, 50%)`;
    matrixHost.innerHTML = "";
    renderMatrixBlock(matrixHost, "", entry.Ahat, { scale: scaleA });
    const cells = matrixHost.querySelectorAll(".cell");
    for (let ci = 0; ci < cells.length; ci++) {
      cells[ci].style.animation = "none";
      cells[ci].offsetHeight;
      cells[ci].style.animation = `cellPulse 0.4s ease ${ci * 0.02}s both`;
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 400 });
}

function liveNote(body, data) {
  const txt = data.text;
  const hasMatrices = data.matrices && data.matrices.length > 0;
  let matrixInfo = null;
  let matCellsArr = [];
  let matMaxCells = 0;
  let totalCells = 0;

  if (hasMatrices) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:1rem;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:0.5rem";
    for (const m of data.matrices) {
      const s = minMax(m.matrix);
      const host = renderSandboxMatrix(row, m.label, zeros(m.matrix.length, m.matrix[0].length), s);
      const cells = [];
      for (let i = 0; i < m.matrix.length; i++) {
        for (let j = 0; j < m.matrix[0].length; j++) {
          cells.push({ i, j, value: m.matrix[i][j], cols: m.matrix[0].length, s });
        }
      }
      matCellsArr.push({ cells, host, cols: m.matrix[0].length });
      matMaxCells = Math.max(matMaxCells, cells.length);
      totalCells += cells.length;
    }
    body.appendChild(row);
    matrixInfo = document.createElement("div");
    matrixInfo.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;margin-bottom:0.3rem";
    body.appendChild(matrixInfo);
  }

  const note = document.createElement("div");
  note.style.cssText = "font-size:0.88rem;color:var(--text);line-height:1.5;padding:0.5rem;background:rgba(0,0,0,0.15);border-radius:8px;border:1px solid var(--border);min-height:1.5em";
  body.appendChild(note);

  const cellFrames = hasMatrices ? totalCells : 0;
  const totalFrames = 1 + cellFrames + 1;

  function fillCell(flatIdx) {
    let accum = 0;
    for (let hi = 0; hi < matCellsArr.length; hi++) {
      const { cells, host, cols } = matCellsArr[hi];
      if (flatIdx < accum + cells.length) {
        const localIdx = flatIdx - accum;
        const cd = cells[localIdx];
        const hostCells = host.querySelectorAll(".cell");
        const idx = cd.i * cols + cd.j;
        if (hostCells[idx]) {
          const t = cd.s.mx > cd.s.mn ? (cd.value - cd.s.mn) / (cd.s.mx - cd.s.mn) : 0.5;
          const c = viridis(t);
          hostCells[idx].style.background = viridisRgb(c);
          hostCells[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
          hostCells[idx].textContent = cd.value.toFixed(2);
          hostCells[idx].style.boxShadow = "inset 0 0 0 3px var(--accent), 0 0 18px rgba(91,156,246,0.6)";
        }
        return;
      }
      accum += cells.length;
    }
  }

  function clearAllCells() {
    for (let hi = 0; hi < matCellsArr.length; hi++) {
      const { cells, host, cols } = matCellsArr[hi];
      const hostCells = host.querySelectorAll(".cell");
      for (let ci = 0; ci < cells.length; ci++) {
        const cd = cells[ci];
        const idx = cd.i * cols + cd.j;
        if (hostCells[idx]) {
          hostCells[idx].style.background = "";
          hostCells[idx].style.color = "";
          hostCells[idx].textContent = "";
          hostCells[idx].style.boxShadow = "";
        }
      }
    }
  }

  function renderFrame(n) {
    if (n === 0) {
      clearAllCells();
      note.textContent = "";
      if (matrixInfo) matrixInfo.textContent = "";
      return;
    }
    note.textContent = txt;
    if (hasMatrices) {
      clearAllCells();
      const fillCount = n - 1;
      const toFill = Math.min(fillCount, totalCells);
      for (let ci = 0; ci < toFill; ci++) {
        fillCell(ci);
      }
      if (matrixInfo) {
        matrixInfo.textContent = toFill >= totalCells ? "Готово" : `Заполнено ${toFill}/${totalCells}`;
      }
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 120 });
}

function livePseudoinverse(body, data) {
  const { C, R, Cp, Rp, A, U } = data;
  const r = C[0].length;

  const Ct = transpose(C);
  const CtC = dot(Ct, C);
  const Rt = transpose(R);
  const RRt = dot(R, Rt);
  const scalarCase = r === 1;

  const frames = [];
  frames.push({ phase: "Шаг 1: C⁺ = (CᵀC)⁻¹·Cᵀ", status: "Начало" });
  frames.push({ phase: "Шаг 1: C⁺ = (CᵀC)⁻¹·Cᵀ", status: "Матрица C (выбранные столбцы)", show: "C" });
  frames.push({ phase: "Шаг 1: C⁺ = (CᵀC)⁻¹·Cᵀ", status: "Вычисляем CᵀC ...", show: "CtC" });
  if (scalarCase) {
    const inv = 1 / CtC[0][0];
    frames.push({ phase: "Шаг 1: C⁺ = (CᵀC)⁻¹·Cᵀ", status: `CᵀC = ${CtC[0][0].toFixed(3)} → (CᵀC)⁻¹ = ${inv.toFixed(3)}`, show: "CtC_inv_scalar" });
  } else {
    frames.push({ phase: "Шаг 1: C⁺ = (CᵀC)⁻¹·Cᵀ", status: "(CᵀC)⁻¹ вычислена", show: "CtC_inv" });
  }
  frames.push({ phase: "Шаг 1: C⁺ = (CᵀC)⁻¹·Cᵀ", status: "C⁺ найдена!", show: "Cp" });

  frames.push({ phase: "Шаг 2: R⁺ = Rᵀ·(R·Rᵀ)⁻¹", status: "Матрица R (выбранные строки)", show: "R" });
  frames.push({ phase: "Шаг 2: R⁺ = Rᵀ·(R·Rᵀ)⁻¹", status: "Вычисляем R·Rᵀ ...", show: "RRt" });
  if (scalarCase) {
    const inv = 1 / RRt[0][0];
    frames.push({ phase: "Шаг 2: R⁺ = Rᵀ·(R·Rᵀ)⁻¹", status: `R·Rᵀ = ${RRt[0][0].toFixed(3)} → (R·Rᵀ)⁻¹ = ${inv.toFixed(3)}`, show: "RRt_inv_scalar" });
  } else {
    frames.push({ phase: "Шаг 2: R⁺ = Rᵀ·(R·Rᵀ)⁻¹", status: "(R·Rᵀ)⁻¹ вычислена", show: "RRt_inv" });
  }
  frames.push({ phase: "Шаг 2: R⁺ = Rᵀ·(R·Rᵀ)⁻¹", status: "R⁺ найдена!", show: "Rp" });

  frames.push({ phase: "Шаг 3: U = C⁺·A·R⁺", status: "Перемножаем C⁺ · A · R⁺ ...", show: "U_formula1" });
  frames.push({ phase: "Шаг 3: U = C⁺·A·R⁺", status: "Вычисляем (C⁺·A) · R⁺ ...", show: "U_formula2" });
  frames.push({ phase: "Шаг 3: U = C⁺·A·R⁺", status: "Связующая матрица U = C⁺·A·R⁺ найдена!", show: "U_result" });

  const totalFrames = frames.length + 1;

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:0.6rem;font-size:0.78rem";

  const phaseDiv = document.createElement("div");
  phaseDiv.style.cssText = "font-size:0.8rem;color:var(--accent);font-weight:600;margin-bottom:0.3rem";
  wrap.appendChild(phaseDiv);

  const contentDiv = document.createElement("div");
  contentDiv.style.cssText = "display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:center;min-height:60px";
  wrap.appendChild(contentDiv);

  const statusDiv = document.createElement("div");
  statusDiv.style.cssText = "font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2em";
  wrap.appendChild(statusDiv);

  body.appendChild(wrap);

  function matStr(M) {
    return M.map(r => r.map(v => v.toFixed(3)).join(" ")).join(" | ");
  }

  function showMatrixTbl(parent, mat, label, highlight) {
    const block = document.createElement("div");
    block.style.cssText = "text-align:center";
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
  }

  function showFormula(parent, text) {
    const el = document.createElement("div");
    el.style.cssText = "text-align:center;font-size:0.72rem;font-family:monospace;margin:0.2rem 0;color:var(--text)";
    el.textContent = text;
    parent.appendChild(el);
  }

  function renderFrame(n) {
    contentDiv.innerHTML = "";
    if (n === 0) {
      phaseDiv.textContent = "";
      statusDiv.textContent = "";
      return;
    }
    const f = frames[n - 1];
    phaseDiv.textContent = f.phase;
    statusDiv.textContent = f.status;

    switch (f.show) {
      case "C": showMatrixTbl(contentDiv, C, "C"); break;
      case "CtC": showMatrixTbl(contentDiv, CtC, "CᵀC"); showFormula(contentDiv, `CᵀC = ${matStr(CtC)}`); break;
      case "CtC_inv_scalar": showFormula(contentDiv, `(CᵀC)⁻¹ = 1 / ${CtC[0][0].toFixed(3)} = ${(1/CtC[0][0]).toFixed(3)}`); break;
      case "CtC_inv": showFormula(contentDiv, "(CᵀC)⁻¹ — обратная матрица"); break;
      case "Cp": showMatrixTbl(contentDiv, Cp, "C⁺ = (CᵀC)⁻¹·Cᵀ"); break;
      case "R": showMatrixTbl(contentDiv, R, "R"); break;
      case "RRt": showMatrixTbl(contentDiv, RRt, "R·Rᵀ"); showFormula(contentDiv, `R·Rᵀ = ${matStr(RRt)}`); break;
      case "RRt_inv_scalar": showFormula(contentDiv, `(R·Rᵀ)⁻¹ = 1 / ${RRt[0][0].toFixed(3)} = ${(1/RRt[0][0]).toFixed(3)}`); break;
      case "RRt_inv": showFormula(contentDiv, "(R·Rᵀ)⁻¹ — обратная матрица"); break;
      case "Rp": showMatrixTbl(contentDiv, Rp, "R⁺ = Rᵀ·(R·Rᵀ)⁻¹"); break;
      case "U_formula1": showFormula(contentDiv, "U = C⁺ · A · R⁺"); break;
      case "U_formula2": showFormula(contentDiv, "Финальное умножение: (C⁺·A) · R⁺"); break;
      case "U_result": {
        const result = document.createElement("div");
        result.style.cssText = "text-align:center;margin-top:0.4rem";
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
        break;
      }
      default: break;
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 600 });
}

function liveErrorDetail(body, data) {
  const { A, Ahat, scale } = data;
  const m = A.length, n = A[0].length;
  const sA = scale || minMax(A);

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

  body.appendChild(wrap);

  // Phase 1: A − Ã = E
  const phase1Cells = m * n;
  const phase2Cells = m * n;
  const totalFrames = phase1Cells + phase2Cells + 2;

  // Precompute all error values
  const errVals = [];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      errVals.push({ i, j, val: A[i][j] - Ahat[i][j] });
    }
  }

  // Precompute all squared values
  const sqVals = errVals.map(e => ({ ...e, sq: e.val * e.val }));

  // Build static layout (A − Ã = E)
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

  const errWork = zeros(m, n);
  const hostE = document.createElement("div");
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

  // Phase 2: squared error matrix (will be cleared and recreated per frame)
  let sumRow = null;
  let hostSq = null;

  function renderFrame(fIdx) {
    if (fIdx === 0) {
      phaseDiv.textContent = "";
      info.textContent = "";
      trace.innerHTML = "";
      for (let ci = 0; ci < m * n; ci++) {
        if (cellsA[ci]) cellsA[ci].style.boxShadow = "";
        if (cellsAh[ci]) cellsAh[ci].style.boxShadow = "";
        if (cellsE[ci]) {
          cellsE[ci].style.background = "";
          cellsE[ci].style.color = "";
          cellsE[ci].textContent = "";
          cellsE[ci].style.boxShadow = "";
        }
        errWork[Math.floor(ci / n)][ci % n] = 0;
      }
      if (sumRow) { sumRow.textContent = ""; }
      return;
    }

    if (fIdx <= phase1Cells) {
      // Phase 1: error cells
      phaseDiv.textContent = "Шаг 1: Матрица ошибки E = A − Ã";
      info.textContent = `Вычисляем поэлементно... (${fIdx}/${phase1Cells})`;
      const count = fIdx - 1;
      for (let ci = 0; ci <= count && ci < errVals.length; ci++) {
        const ev = errVals[ci];
        const idx = ev.i * n + ev.j;
        errWork[ev.i][ev.j] = ev.val;
        if (cellsE[idx]) {
          const t = sA.mx > sA.mn ? (ev.val - sA.mn) / (sA.mx - sA.mn) : 0.5;
          const c = viridis(Math.max(0, Math.min(1, t)));
          cellsE[idx].style.background = viridisRgb(c);
          cellsE[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
          cellsE[idx].textContent = ev.val.toFixed(2);
          cellsE[idx].style.boxShadow = ci === count ? "inset 0 0 0 3px var(--bad)" : "";
        }
        if (cellsA[idx]) cellsA[idx].style.boxShadow = ci === count ? "inset 0 0 0 3px var(--accent)" : "";
        if (cellsAh[idx]) cellsAh[idx].style.boxShadow = ci === count ? "inset 0 0 0 3px var(--good)" : "";
      }
      if (count >= 0 && count < errVals.length) {
        const ev = errVals[count];
        trace.innerHTML = `E<sub>${ev.i}${ev.j}</sub> = ${A[ev.i][ev.j].toFixed(2)} − ${Ahat[ev.i][ev.j].toFixed(2)} = <b>${ev.val.toFixed(2)}</b>`;
      }
      if (sumRow) sumRow.textContent = "";
      return;
    }

    // Phase 2: squared error
    const phase2Idx = fIdx - phase1Cells - 1;
    if (phase2Idx === 0) {
      // First frame of phase 2: setup
      phaseDiv.textContent = "Шаг 2: Сумма квадратов Σᵢⱼ Eᵢⱼ² → ||E||_F";
      info.textContent = "Вычисляем сумму квадратов...";
      trace.innerHTML = "";

      // Remove old E² matrix if exists
      const existingSq = matricesRow.querySelector(".err-sq-wrap");
      if (existingSq) existingSq.remove();

      const sqWrap = document.createElement("div");
      sqWrap.className = "err-sq-wrap";
      sqWrap.style.cssText = "text-align:center";
      const sqLbl = document.createElement("div");
      sqLbl.style.cssText = "font-weight:600;font-size:0.72rem;margin-bottom:0.1rem;color:var(--bad)";
      sqLbl.textContent = "E² (поквадратно)";
      sqWrap.appendChild(sqLbl);
      hostSq = document.createElement("div");
      renderMatrixBlock(hostSq, "", zeros(m, n), { scale: sA });
      sqWrap.appendChild(hostSq);
      matricesRow.appendChild(sqWrap);

      if (!sumRow) {
        sumRow = document.createElement("div");
        sumRow.style.cssText = "text-align:center;font-size:0.75rem;font-family:monospace;margin-top:0.3rem;min-height:1.4rem";
        wrap.insertBefore(sumRow, trace);
      }
    }

    if (phase2Idx > 0) {
      let frobSum = 0;
      const cellsSq = hostSq ? hostSq.querySelectorAll(".cell") : [];
      const count = Math.min(phase2Idx, phase2Cells);

      for (let ci = 0; ci < phase2Cells; ci++) {
        const sv = sqVals[ci];
        const idx = sv.i * n + sv.j;
        if (ci < count) {
          frobSum += sv.sq;
          if (cellsSq && cellsSq[idx]) {
            const t = Math.min(sv.sq / (Math.max(...sqVals.map(x => x.sq), 1)), 1);
            const c = viridis(t);
            cellsSq[idx].style.background = viridisRgb(c);
            cellsSq[idx].style.color = t > 0.6 ? "#080c14" : "#fff";
            cellsSq[idx].textContent = sv.sq.toFixed(4);
            cellsSq[idx].style.boxShadow = ci === count - 1 ? "inset 0 0 0 3px var(--bad)" : "";
          }
          if (cellsE && cellsE[idx]) cellsE[idx].style.boxShadow = ci === count - 1 ? "inset 0 0 0 3px var(--good)" : "";
        }
      }

      if (count <= phase2Cells && count > 0) {
        const sv = sqVals[count - 1];
        trace.innerHTML = `E<sub>${sv.i}${sv.j}</sub>² = (${sv.val.toFixed(2)})² = ${sv.sq.toFixed(4)}`;
        info.textContent = `Сумма квадратов: ${frobSum.toFixed(4)} (${count}/${phase2Cells})`;
        if (sumRow) sumRow.textContent = `Σ = ${frobSum.toFixed(4)}`;
      }

      if (count >= phase2Cells) {
        const frob = Math.sqrt(frobSum);
        const normA = Math.sqrt(A.reduce((s, row) => s + row.reduce((ss, v) => ss + v * v, 0), 0));
        const relErr = frob / (normA + 1e-12);
        trace.innerHTML = `ΣEᵢⱼ² = <b>${frobSum.toFixed(4)}</b> → ||E||_F = √(${frobSum.toFixed(4)}) = <b>${frob.toFixed(4)}</b>`;
        info.textContent = `Frobenius норма: ${frob.toFixed(4)}`;
        if (sumRow) {
          sumRow.textContent = `Относительная ошибка: ||E||_F / ||A||_F = ${(relErr * 100).toFixed(2)}%`;
          sumRow.style.fontWeight = "600";
          sumRow.style.color = "var(--good)";
        }
      }
    }
  }

  createFrameSlider(body, totalFrames, renderFrame, { speed: 250 });
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
  // ── Sidebar (left) ──
  const sidebar = document.createElement("div");
  sidebar.className = "vis-sidebar";

  const sidebarTitle = document.createElement("div");
  sidebarTitle.className = "vis-sidebar__title";
  sidebarTitle.textContent = "Управление";
  sidebar.appendChild(sidebarTitle);

  // Algo group
  const algoLabel = document.createElement("div");
  algoLabel.className = "vis-sidebar__label";
  algoLabel.textContent = "Метод";
  sidebar.appendChild(algoLabel);
  const algoGroup = document.createElement("div");
  algoGroup.className = "vis-sidebar__group";
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
  sidebar.appendChild(algoGroup);

  // Presets
  const presetLabel = document.createElement("div");
  presetLabel.className = "vis-sidebar__label";
  presetLabel.textContent = "Шаблон";
  sidebar.appendChild(presetLabel);
  const presetGroup = document.createElement("div");
  presetGroup.className = "vis-sidebar__group";
  const presets = [
    { id: "identity", label: "Единичная" },
    { id: "zeros", label: "Нулевая" },
    { id: "random", label: "Случайная" },
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
  sidebar.appendChild(presetGroup);

  // Rank
  const rankLabel = document.createElement("div");
  rankLabel.className = "vis-sidebar__label";
  rankLabel.textContent = "Ранг k";
  sidebar.appendChild(rankLabel);
  const rankInput = document.createElement("input");
  rankInput.type = "number";
  rankInput.min = 1;
  rankInput.max = Math.min(A.length, A[0].length, 10);
  rankInput.value = k;
  rankInput.className = "vis-sidebar__input";
  rankInput.addEventListener("change", () => {
    state.visK = Math.max(1, Math.min(Number(rankInput.value) || 1, Math.min(A.length, A[0].length, 10)));
    renderVisualizerPage(container, state);
  });
  sidebar.appendChild(rankInput);

  // Seed
  const seedLabel = document.createElement("div");
  seedLabel.className = "vis-sidebar__label";
  seedLabel.textContent = "Seed";
  sidebar.appendChild(seedLabel);
  const seedInput = document.createElement("input");
  seedInput.type = "number";
  seedInput.min = 0;
  seedInput.max = 999999;
  seedInput.value = visSeed;
  seedInput.className = "vis-sidebar__input";
  seedInput.addEventListener("change", () => {
    state.visSeed = Math.max(0, Math.floor(Number(seedInput.value) || 0));
    renderVisualizerPage(container, state);
  });
  sidebar.appendChild(seedInput);

  // Iterations (only for NMF/ALS)
  const iterBlock = document.createElement("div");
  iterBlock.className = "vis-sidebar__iter-block";
  if (algo !== "nmf" && algo !== "als") iterBlock.style.display = "none";
  const iterLabel = document.createElement("div");
  iterLabel.className = "vis-sidebar__label";
  iterLabel.textContent = algo === "als" ? "Итерации ALS" : "Итерации";
  iterBlock.appendChild(iterLabel);
  const iterInput = document.createElement("input");
  iterInput.type = "number";
  iterInput.min = 1;
  iterInput.max = 50;
  iterInput.value = iters;
  iterInput.className = "vis-sidebar__input";
  iterInput.addEventListener("change", () => {
    state.visIters = Math.max(1, Math.min(50, Number(iterInput.value) || 20));
    renderVisualizerPage(container, state);
  });
  iterBlock.appendChild(iterInput);
  sidebar.appendChild(iterBlock);

  layout.appendChild(sidebar);

  // ── Main body (right) ──
  const body = document.createElement("div");
  body.className = "vis-body";

  body.appendChild(legendBar);

  const pipelineWrap = document.createElement("div");
  pipelineWrap.className = "vis-pipeline-wrap";

  const pipeHead = document.createElement("div");
  pipeHead.className = "card__head";
  pipeHead.innerHTML = `<h2>Визуальный процесс: ${algo.toUpperCase()}</h2><div class="sub">пошаговое разложение с анимацией</div>`;
  pipelineWrap.appendChild(pipeHead);

  const pipeContainer = document.createElement("div");
  pipeContainer.className = "vis-pipeline";

  const steps = generatePipeline(algo, A, k, iters);

  // Make first step editable (embedded matrix editor)
  if (steps.length > 0 && steps[0].type === "matrix") {
    steps[0].editable = true;
    steps[0].A = A;
    steps[0].onChange = (newA) => {
      state.visA = newA;
      renderVisualizerPage(container, state);
    };
  }

  let i = 0;
  while (i < steps.length) {
    if (steps[i].type === "arrow" && i + 1 < steps.length && steps[i + 1].type !== "arrow") {
      const group = document.createElement("div");
      group.className = "vis-step-group";
      renderArrow(group, steps, i);
      renderStep(group, steps[i + 1], i + 1);
      pipeContainer.appendChild(group);
      i += 2;
    } else if (steps[i].type === "arrow") {
      renderArrow(pipeContainer, steps, i);
      i++;
    } else {
      renderStep(pipeContainer, steps[i], i);
      i++;
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
