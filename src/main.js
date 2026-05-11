/* global numeric */

import { clear, readChecked, readNumber, latexToHtml, renderAccordion, renderConvergenceChart, renderSigmaChart, renderIterationSlider } from "./dom.js";
import { dims, diff, minMax, absMean, frobNorm, randomMatrix, transpose, dot as dotM, diag } from "./matrix.js";
import { renderLegend, renderMatrixBlock } from "./heatmap.js";
import { svdTruncated, pcaReconstruct, pcaReconstructWithSteps, nmfReconstruct, nmfReconstructHistory, curReconstruct, alsReconstruct, alsReconstructHistory } from "./decompositions.js";
import { renderCenterEdgeExperiment, renderPerturbation } from "./experiments.js";
import { renderTheoryPage } from "./theory.js";
import { renderCodePage } from "./code.js";
import { renderCalcPage } from "./calculator.js";

const panelMatrix = document.getElementById("panel-matrix");
const panelDecomp = document.getElementById("panel-decomp");
const panelExp = document.getElementById("panel-exp");
const panelTheory = document.getElementById("panel-theory");
const panelCode = document.getElementById("panel-code");
const panelCalc = document.getElementById("panel-calc");

const state = {
  seed: 42,
  range: [0, 10],
  A: randomMatrix(6, 6, 0, 10, 42),
  k: 2,
  iters: 80,
  algos: { svd: true, pca: true, nmf: true, cur: true, als: true },
  lockScale: false,
};

function ensureNumeric() {
  if (typeof numeric === "undefined" || typeof numeric.svd !== "function") {
    document.body.innerHTML = "<div style=\"padding:2rem;color:#f87171\">numeric.js не загрузилась. Запустите через локальный сервер: <code>python3 -m http.server 8080</code></div>";
    return false;
  }
  return true;
}

function parseRange(s) {
  const [a, b] = String(s).split(",").map(Number);
  return [Number.isFinite(a) ? a : 0, Number.isFinite(b) ? b : 10];
}

function syncState() {
  state.seed = readNumber("a-seed", 42);
  state.range = parseRange(document.getElementById("a-range").value);
  state.k = readNumber("rank-k", 2);
  state.iters = Math.min(readNumber("iters", 80), 200);
  state.algos = {
    svd: readChecked("alg-svd"),
    pca: readChecked("alg-pca"),
    nmf: readChecked("alg-nmf"),
    cur: readChecked("alg-cur"),
    als: readChecked("alg-als"),
  };
  state.lockScale = readChecked("a-lock-scale");
}

function getScale() {
  return state.lockScale ? minMax(state.A) : undefined;
}

function mrow(parent, title, A, opts = {}) {
  renderMatrixBlock(parent, title, A, {
    scale: opts.scale !== undefined ? opts.scale : getScale(),
    subtitle: opts.subtitle,
    selected: opts.selected,
    badge: opts.badge,
    diverge: opts.diverge,
    onCellClick: opts.onCellClick,
    onDblClick: opts.onDblClick,
  });
}

// ── TAB 1: MATRIX A ──────────────────────────────────────────────────────────

function renderTabMatrix() {
  const hasControls = !!document.getElementById("a-rows");
  if (hasControls) syncState();

  const showOrig = hasControls ? readChecked("vis-orig") : true;
  const showRecon = hasControls ? readChecked("vis-recon") : true;
  const showErr = hasControls ? readChecked("vis-err") : true;

  clear(panelMatrix);

  const rangeStr = state.range.join(",");

  const layout = document.createElement("div");
  layout.className = "layout-sidebar";

  // Sidebar
  const sidebar = document.createElement("div");
  sidebar.className = "sidebar-sticky";

  // Matrix controls card
  const cardA = document.createElement("div");
  cardA.className = "card";
  cardA.innerHTML = `
    <div class="card__head">
      <h2>Матрица A</h2>
    </div>
    <div class="controls grid2">
      <label><span>m</span><input id="a-rows" type="number" min="2" max="20" value="${state.A.length}"></label>
      <label><span>n</span><input id="a-cols" type="number" min="2" max="20" value="${state.A[0].length}"></label>
      <label><span>Диапазон</span>
        <select id="a-range">
          <option value="0,10" ${rangeStr === "0,10" ? "selected" : ""}>[0; 10]</option>
          <option value="-1,1" ${rangeStr === "-1,1" ? "selected" : ""}>[-1; 1]</option>
          <option value="-10,10" ${rangeStr === "-10,10" ? "selected" : ""}>[-10; 10]</option>
        </select>
      </label>
      <label><span>Seed</span><input id="a-seed" type="number" min="0" value="${state.seed}"></label>
    </div>
    <div class="controls">
      <label class="toggle"><input id="a-lock-scale" type="checkbox" ${state.lockScale ? "checked" : ""}><span>Общая шкала цвета</span></label>
    </div>
    <div class="controls">
      <button id="btn-a-new" class="primary">Новая A</button>
      <button id="btn-a-reset">Сброс</button>
    </div>
  `;
  sidebar.appendChild(cardA);

  // Algorithms card
  const cardAlgo = document.createElement("div");
  cardAlgo.className = "card";
  cardAlgo.innerHTML = `
    <div class="card__head">
      <h2>Алгоритмы</h2>
    </div>
    <div class="controls stack">
      <label class="toggle"><input type="checkbox" id="alg-svd" ${state.algos.svd ? "checked" : ""}><span>SVD — усечение ранга k</span></label>
      <label class="toggle"><input type="checkbox" id="alg-pca" ${state.algos.pca ? "checked" : ""}><span>PCA — центрирование + SVD</span></label>
      <label class="toggle"><input type="checkbox" id="alg-nmf" ${state.algos.nmf ? "checked" : ""}><span>NMF — неотрицательное</span></label>
      <label class="toggle"><input type="checkbox" id="alg-cur" ${state.algos.cur ? "checked" : ""}><span>CUR — строки + столбцы</span></label>
      <label class="toggle"><input type="checkbox" id="alg-als" ${state.algos.als ? "checked" : ""}><span>ALS — итеративное</span></label>
    </div>
    <div class="controls grid2">
      <label><span>k (ранг)</span><input id="rank-k" type="number" min="1" max="10" value="${state.k}"></label>
      <label><span>Итерации</span><input id="iters" type="number" min="1" max="200" value="${state.iters}"></label>
    </div>
  `;
  sidebar.appendChild(cardAlgo);

  // Visibility card
  const cardVis = document.createElement("div");
  cardVis.className = "card";
  cardVis.innerHTML = `
    <div class="card__head">
      <h2>Видимость</h2>
    </div>
    <div class="controls stack">
      <label class="toggle"><input type="checkbox" id="vis-orig" ${showOrig ? "checked" : ""}><span>Показывать оригинал A</span></label>
      <label class="toggle"><input type="checkbox" id="vis-recon" ${showRecon ? "checked" : ""}><span>Показывать Ã (восстановление)</span></label>
      <label class="toggle"><input type="checkbox" id="vis-err" ${showErr ? "checked" : ""}><span>Показывать ошибку A - Ã</span></label>
    </div>
  `;
  sidebar.appendChild(cardVis);

  layout.appendChild(sidebar);

  // Main: matrix editor + quick results
  const main = document.createElement("div");
  main.style.display = "flex";
  main.style.flexDirection = "column";
  main.style.gap = "1rem";

  const editorCard = document.createElement("div");
  editorCard.className = "card";
  editorCard.innerHTML = `<div class="card__head"><h2>Редактор A</h2><div class="sub">dblclick по ячейке — изменить значение</div></div>`;
  const editorHost = document.createElement("div");
  editorCard.appendChild(editorHost);
  main.appendChild(editorCard);

  const resultsCard = document.createElement("div");
  resultsCard.className = "card";
  resultsCard.innerHTML = `<div class="card__head"><h2>Сравнение: A → Ã</h2><div class="sub">восстановление каждым алгоритмом рядом с оригиналом</div></div>`;
  const resultsHost = document.createElement("div");
  resultsCard.appendChild(resultsHost);
  main.appendChild(resultsCard);

  layout.appendChild(main);
  panelMatrix.appendChild(layout);

  // Events
  document.getElementById("btn-a-new").addEventListener("click", () => {
    const m = readNumber("a-rows", 6);
    const n = readNumber("a-cols", 6);
    const [lo, hi] = state.range;
    state.A = randomMatrix(m, n, lo, hi, state.seed + Math.floor(Math.random() * 99999));
    syncState();
    renderTabMatrix();
  });

  document.getElementById("btn-a-reset").addEventListener("click", () => {
    const m = readNumber("a-rows", 6);
    const n = readNumber("a-cols", 6);
    const [lo, hi] = state.range;
    state.A = randomMatrix(m, n, lo, hi, state.seed);
    syncState();
    renderTabMatrix();
  });

  for (const id of ["a-rows", "a-cols", "a-range", "a-seed", "a-lock-scale", "alg-svd", "alg-pca", "alg-nmf", "alg-cur", "alg-als", "rank-k", "iters"]) {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => {
      syncState();
      renderTabMatrix();
    });
  }

  // Render editor
  const scale = getScale();
  renderLegend(editorHost, ["sequential"]);
  mrow(editorHost, "A", state.A, {
    scale,
    onDblClick: (i, j, val) => {
      state.A[i][j] = val;
      renderTabMatrix();
    },
  });

  // Render quick results
  const [lo] = state.range;
  const nmfOk = lo >= 0;

  renderLegend(resultsHost, ["sequential", "diverging"]);

  const algos = [];
  if (state.algos.svd) algos.push({ id: "svd", name: "SVD", run: () => svdTruncated(state.A, state.k).Ahat });
  if (state.algos.pca) algos.push({ id: "pca", name: "PCA", run: () => pcaReconstruct(state.A, state.k).Ahat });
  if (state.algos.nmf) {
    if (nmfOk) algos.push({ id: "nmf", name: "NMF", run: () => nmfReconstruct(state.A, state.k, state.iters).Ahat });
    else algos.push({ id: "nmf_skip", name: "NMF", skip: true });
  }
  if (state.algos.cur) algos.push({ id: "cur", name: "CUR", run: () => curReconstruct(state.A, state.k).Ahat });
  if (state.algos.als) algos.push({ id: "als", name: "ALS", run: () => alsReconstruct(state.A, state.k, state.iters).Ahat });

  if (!algos.length) {
    resultsHost.appendChild(document.createTextNode("Включите хотя бы один алгоритм"));
    return;
  }

  // showOrig, showRecon, showErr — saved at top

  const row = document.createElement("div");
  row.className = "matrix-row";

  for (const algo of algos) {
    if (algo.skip) {
      const note = document.createElement("div");
      note.className = "deviation";
      note.innerHTML = `*NMF* требует неотрицательные значения — пропускается при текущем диапазоне`;
      row.appendChild(note);
      continue;
    }

    const card = document.createElement("div");
    card.className = "card";
    card.style.minWidth = "200px";

    const h = document.createElement("h3");
    h.style.fontSize = "0.9rem";
    h.style.fontWeight = "600";
    h.textContent = algo.name;
    card.appendChild(h);

    try {
      const Ahat = algo.run();
      const err = diff(state.A, Ahat);

      const metrics = document.createElement("div");
      metrics.className = "metrics";
      metrics.innerHTML = `<div class="metric"><span class="metric__key">F = </span><span class="metric__val">${frobNorm(err).toFixed(4)}</span></div>
        <div class="metric"><span class="metric__key">mean = </span><span class="metric__val">${absMean(err).toFixed(4)}</span></div>`;
      card.appendChild(metrics);

      const inner = document.createElement("div");
      inner.className = "matrix-row";
      inner.style.marginTop = "0.4rem";

      if (showOrig) mrow(inner, "A", state.A, { scale });
      if (showRecon) mrow(inner, "Ã", Ahat, { scale });
      if (showErr) mrow(inner, "A - Ã", err, { diverge: true });

      card.appendChild(inner);
    } catch (e) {
      const fail = document.createElement("div");
      fail.className = "muted";
      fail.style.fontSize = "0.8rem";
      fail.textContent = `Ошибка: ${e?.message || e}`;
      card.appendChild(fail);
    }

    row.appendChild(card);
  }

  resultsHost.appendChild(row);
}

// ── TAB 2: DECOMPOSITIONS ───────────────────────────────────────────────────

function renderTabDecomp() {
  clear(panelDecomp);
  syncState();

  const [lo] = state.range;
  const nmfOk = lo >= 0;
  const scale = getScale();

  const accordion = document.createElement("div");
  accordion.className = "accordion";
  panelDecomp.appendChild(accordion);

  // SVD
  const svdResult = svdTruncated(state.A, state.k);
  const { r, Uk, Sk, Vk, US, Ahat } = svdResult;
  const Vt = transpose(Vk);

  renderAccordion(accordion, [{
    name: "SVD",
    badge: `k=${r}`,
    defaultOpen: true,
    render(body) {
      const fm = document.createElement("div");
      fm.className = "accordion__formula";
      fm.innerHTML = `${latexToHtml("A = U \\Sigma V^T", false)} &nbsp;→&nbsp; ${latexToHtml("\\tilde A_k = U_k \\Sigma_k V_k^T", false)}`;
      body.appendChild(fm);

      const step = document.createElement("div");
      step.className = "matrix-row";
      mrow(step, "A", state.A, { scale });
      mrow(step, "U_k", Uk, { subtitle: `${state.A.length}×${r}` });
      mrow(step, "V_k^T", Vt, { subtitle: `${r}×${state.A[0].length}` });
      body.appendChild(step);

      const diagRow = document.createElement("div");
      diagRow.className = "matrix-row";
      mrow(diagRow, `Σ_k diag(σ)`, diag(Sk), { subtitle: `${r}×${r}` });

      const sigmaCard = document.createElement("div");
      sigmaCard.className = "card";
      sigmaCard.innerHTML = `<div class="card__head"><h2>Сингулярные числа</h2><div class="sub">высота пропорциональна ${latexToHtml("\\sigma_i / \\sigma_1")}</div></div>`;
      const sigmaWrap = document.createElement("div");
      renderSigmaChart(sigmaWrap, Sk, r);
      sigmaCard.appendChild(sigmaWrap);
      diagRow.appendChild(sigmaCard);
      body.appendChild(diagRow);

      const usRow = document.createElement("div");
      usRow.className = "matrix-row";
      mrow(usRow, "U_k · Σ_k", US, { subtitle: `${state.A.length}×${r}` });
      mrow(usRow, "Ã", Ahat, { scale });
      body.appendChild(usRow);

      const metrics = document.createElement("div");
      metrics.className = "metrics";
      metrics.innerHTML = `<div class="metric"><span class="metric__key">${latexToHtml("\\|A - \\tilde A\\|_F")} = </span><span class="metric__val">${frobNorm(diff(state.A, Ahat)).toFixed(5)}</span></div>`;
      body.appendChild(metrics);
    },
  }]);

  // PCA
  const pcaResult = pcaReconstructWithSteps(state.A, state.k);
  const { mean, X, Uk: pcaUk, Sk: pcaSk, Vk: pcaVk, US: pcaUS, Xk, Ahat: pcaAhat } = pcaResult;
  const pcaVt = transpose(pcaVk);
  const meanRow = [mean.slice()];

  renderAccordion(accordion, [{
    name: "PCA",
    badge: `k=${pcaResult.r}`,
    defaultOpen: false,
    render(body) {
      const fm = document.createElement("div");
      fm.className = "accordion__formula";
      fm.innerHTML = `${latexToHtml("X = A - \\mu", false)} &nbsp;→&nbsp; ${latexToHtml("\\tilde A = \\hat X + \\mu", false)}`;
      body.appendChild(fm);

      const step = document.createElement("div");
      step.className = "matrix-row";
      mrow(step, "A", state.A, { scale });
      mrow(step, `\u03BC (строка)`, meanRow, { subtitle: `1×${state.A[0].length}` });
      mrow(step, "X = A - \u03BC", X, { subtitle: `центрированные данные` });
      body.appendChild(step);

      const diagRow = document.createElement("div");
      diagRow.className = "matrix-row";
      mrow(diagRow, `U_k \\text{(на X)}`, pcaUk, { subtitle: `${state.A.length}×${pcaResult.r}` });
      mrow(diagRow, "Σ_k", diag(pcaSk), { subtitle: `${pcaResult.r}×${pcaResult.r}` });
      mrow(diagRow, "V_k^T", pcaVt, { subtitle: `${pcaResult.r}×${state.A[0].length}` });
      body.appendChild(diagRow);

      const step2 = document.createElement("div");
      step2.className = "matrix-row";
      mrow(step2, "U_k · Σ_k", pcaUS, { subtitle: "промежуточное" });
      mrow(step2, "X̂", Xk, { subtitle: "низкоранговое приближение X" });
      body.appendChild(step2);

      const finalRow = document.createElement("div");
      finalRow.className = "matrix-row";
      mrow(finalRow, "Ã = X̂ + \u03BC", pcaAhat, { scale });
      body.appendChild(finalRow);

      const metrics = document.createElement("div");
      metrics.className = "metrics";
      metrics.innerHTML = `<div class="metric"><span class="metric__key">${latexToHtml("\\|A - \\tilde A\\|_F")} = </span><span class="metric__val">${frobNorm(diff(state.A, pcaAhat)).toFixed(5)}</span></div>`;
      body.appendChild(metrics);
    },
  }]);

  // NMF with iterations
  if (state.algos.nmf) {
    const cappedIters = Math.min(state.iters, 50);

    if (!nmfOk) {
      renderAccordion(accordion, [{
        name: "NMF",
        badge: "недоступен",
        defaultOpen: false,
        render(body) {
          const dev = document.createElement("div");
          dev.className = "deviation";
          dev.innerHTML = `NMF требует неотрицательную матрицу — текущий диапазон содержит отрицательные числа, выберите [0; 10]`;
          body.appendChild(dev);
        },
      }]);
    } else {
      const { result: nmfRes, history: nmfHist } = nmfReconstructHistory(state.A, state.k, cappedIters);
      const { Ahat: nmfAhat, shift, W, H } = nmfRes;
      const Xpos = state.A.map((r) => r.map((v) => v + shift));
      const WH = dotM(W, H);

      renderAccordion(accordion, [{
        name: "NMF",
        badge: `${cappedIters} итераций`,
        defaultOpen: state.algos.nmf,
        render(body) {
          const fm = document.createElement("div");
          fm.className = "accordion__formula";
          fm.innerHTML = `${latexToHtml("A \\approx W \\cdot H", false)}`;
          body.appendChild(fm);

          renderConvergenceChart(body, nmfHist);

          let currentIdx = Math.floor(nmfHist.length / 2);
          let currentAhat = nmfHist[currentIdx]?.Ahat || nmfAhat;

          const slider = renderIterationSlider(body, nmfHist, (h, idx) => {
            currentIdx = idx;
            currentAhat = h.Ahat;
            refreshAhat();
          });

          function refreshAhat() {
            const wrapper = body.querySelector(".iter-snap-wrap");
            if (!wrapper) return;
            wrapper.innerHTML = "";
            mrow(wrapper, `Ã на итерации ${nmfHist[currentIdx].i}`, currentAhat, {
              subtitle: `${latexToHtml("\\|A - \\tilde A\\|_F")} = ${nmfHist[currentIdx].frob.toFixed(5)}`,
              scale,
            });
            const note2 = document.createElement("div");
            note2.className = "muted";
            note2.style.fontSize = "0.76rem";
            note2.textContent = `F = ${frobNorm(diff(state.A, currentAhat)).toFixed(5)} · mean = ${absMean(diff(state.A, currentAhat)).toFixed(5)}`;
            wrapper.appendChild(note2);
          }

          const snapWrap = document.createElement("div");
          snapWrap.className = "iter-snap-wrap matrix-row";
          snapWrap.style.marginTop = "0.5rem";
          snapWrap.style.display = "flex";
          snapWrap.style.flexWrap = "wrap";
          snapWrap.style.gap = "1rem";
          snapWrap.style.alignItems = "flex-start";
          mrow(snapWrap, `Ã на итерации ${nmfHist[currentIdx].i}`, currentAhat, {
            subtitle: `${latexToHtml("\\|A - \\tilde A\\|_F")} = ${nmfHist[currentIdx].frob.toFixed(5)}`,
            scale,
          });
          body.appendChild(snapWrap);

          const step = document.createElement("div");
          step.className = "matrix-row";
          if (shift > 0) mrow(step, `A + ${shift.toFixed(2)}`, Xpos, { subtitle: `сдвиг для неотрицательности` });
          else mrow(step, "A", Xpos, { scale });
          mrow(step, "W", W, { subtitle: `${state.A.length}×${state.k}` });
          mrow(step, "H", H, { subtitle: `${state.k}×${state.A[0].length}` });
          mrow(step, "W · H", WH, { subtitle: "произведение" });
          body.appendChild(step);

          const finalRow = document.createElement("div");
          finalRow.className = "matrix-row";
          mrow(finalRow, "Ã", nmfAhat, { scale });
          body.appendChild(finalRow);

          const metrics = document.createElement("div");
          metrics.className = "metrics";
           metrics.innerHTML = `<div class="metric"><span class="metric__key">${latexToHtml("\\|A - \\tilde A\\|_F")} = </span><span class="metric__val">${frobNorm(diff(state.A, nmfAhat)).toFixed(5)}</span></div>`;
          body.appendChild(metrics);
        },
      }]);
    }
  }

  // CUR
  if (state.algos.cur) {
    const curRes = curReconstruct(state.A, state.k);
    const { Ahat: curAhat, C, U, R, Wcore, topRows, topCols } = curRes;

    renderAccordion(accordion, [{
      name: "CUR",
      badge: `r=${curRes.r}`,
      defaultOpen: false,
      render(body) {
        const fm = document.createElement("div");
        fm.className = "accordion__formula";
        fm.innerHTML = `${latexToHtml("\\tilde A = C \\cdot U \\cdot R", false)}`;
        body.appendChild(fm);

        const note = document.createElement("div");
        note.className = "muted";
        note.style.fontSize = "0.8rem";
        note.textContent = `Столбцы: ${topCols.join(", ")}  ·  строки: ${topRows.join(", ")}`;
        body.appendChild(note);

        const step = document.createElement("div");
        step.className = "matrix-row";
        mrow(step, "A", state.A, { scale });
        mrow(step, "C", C, { subtitle: `${state.A.length}×${curRes.r}, выбранные столбцы` });
        mrow(step, "W (подматрица)", Wcore, { subtitle: `${curRes.r}×${curRes.r}` });
        body.appendChild(step);

        const step2 = document.createElement("div");
        step2.className = "matrix-row";
        mrow(step2, "U ≈ W⁺", U, { subtitle: "псевдообратная к W" });
        mrow(step2, "R", R, { subtitle: `${curRes.r}×${state.A[0].length}, выбранные строки` });
        body.appendChild(step2);

        const finalRow = document.createElement("div");
        finalRow.className = "matrix-row";
        mrow(finalRow, "Ã", curAhat, { scale });
        body.appendChild(finalRow);

        const metrics = document.createElement("div");
        metrics.className = "metrics";
         metrics.innerHTML = `<div class="metric"><span class="metric__key">${latexToHtml("\\|A - \\tilde A\\|_F")} = </span><span class="metric__val">${frobNorm(diff(state.A, curAhat)).toFixed(5)}</span></div>`;
        body.appendChild(metrics);
      },
    }]);
  }

  // ALS with iterations
  if (state.algos.als) {
    const cappedIters = Math.min(state.iters, 50);
    const { result: alsRes, history: alsHist } = alsReconstructHistory(state.A, state.k, cappedIters, 1e-2);
    const { Ahat: alsAhat, X, Y } = alsRes;
    const Yt = transpose(Y);

    renderAccordion(accordion, [{
      name: "ALS",
      badge: `${cappedIters} итераций`,
      defaultOpen: false,
      render(body) {
        const fm = document.createElement("div");
        fm.className = "accordion__formula";
        fm.innerHTML = `${latexToHtml("\\tilde A = X \\cdot Y^T", false)}`;
        body.appendChild(fm);

        renderConvergenceChart(body, alsHist);

        let currentIdx = Math.floor(alsHist.length / 2);
        let currentAhat = alsHist[currentIdx]?.Ahat || alsAhat;

        renderIterationSlider(body, alsHist, (h, idx) => {
          currentIdx = idx;
          currentAhat = h.Ahat;
          refreshAhat();
        });

        function refreshAhat() {
          const wrapper = body.querySelector(".als-snap-wrap");
          if (!wrapper) return;
          wrapper.innerHTML = "";
          mrow(wrapper, `Ã на итерации ${alsHist[currentIdx].i}`, currentAhat, {
            scale,
          });
          const note2 = document.createElement("div");
          note2.className = "muted";
          note2.style.fontSize = "0.76rem";
          note2.textContent = `F = ${frobNorm(diff(state.A, currentAhat)).toFixed(5)} · mean = ${absMean(diff(state.A, currentAhat)).toFixed(5)}`;
          wrapper.appendChild(note2);
        }

        const snapWrap = document.createElement("div");
        snapWrap.className = "als-snap-wrap matrix-row";
        snapWrap.style.marginTop = "0.5rem";
        snapWrap.style.display = "flex";
        snapWrap.style.flexWrap = "wrap";
        snapWrap.style.gap = "1rem";
        snapWrap.style.alignItems = "flex-start";
        mrow(snapWrap, `Ã на итерации ${alsHist[currentIdx].i}`, currentAhat, {
          scale,
        });
        body.appendChild(snapWrap);

        const step = document.createElement("div");
        step.className = "matrix-row";
        mrow(step, "A", state.A, { scale });
        mrow(step, "X", X, { subtitle: `${state.A.length}×${alsRes.r}` });
        mrow(step, "Y^T", Yt, { subtitle: `${alsRes.r}×${state.A[0].length}` });
        body.appendChild(step);

        const finalRow = document.createElement("div");
        finalRow.className = "matrix-row";
        mrow(finalRow, "Ã", alsAhat, { scale });
        body.appendChild(finalRow);

        const metrics = document.createElement("div");
        metrics.className = "metrics";
         metrics.innerHTML = `<div class="metric"><span class="metric__key">${latexToHtml("\\|A - \\tilde A\\|_F")} = </span><span class="metric__val">${frobNorm(diff(state.A, alsAhat)).toFixed(5)}</span></div>
          <div class="metric"><span class="metric__key">рег. ${latexToHtml("\\lambda")} = </span><span class="metric__val">${alsRes.reg}</span></div>`;
        body.appendChild(metrics);
      },
    }]);
  }
}

// ── TAB 3: EXPERIMENTS ──────────────────────────────────────────────────────

function renderTabExp() {
  clear(panelExp);

  const card1 = document.createElement("div");
  card1.className = "card";
  card1.innerHTML = `<div class="card__head"><h2>Центр vs края</h2><div class="sub">средняя ошибка в центральной области и на периферии матрицы</div></div>`;
  const host1 = document.createElement("div");
  card1.appendChild(host1);
  panelExp.appendChild(card1);

  const card2 = document.createElement("div");
  card2.className = "card";
  card2.innerHTML = `<div class="card__head"><h2>Возмущение Δ</h2><div class="sub">как Δ в одной ячейке A влияет на ΔÃ — радиальный распад</div></div>`;
  const host2 = document.createElement("div");
  card2.appendChild(host2);
  panelExp.appendChild(card2);

  renderCenterEdgeExperiment(host1, state);
  renderPerturbation(host2, state);
}

// ── TAB SWITCHING ───────────────────────────────────────────────────────────

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = {
    matrix: panelMatrix,
    decomp: panelDecomp,
    exp: panelExp,
    theory: panelTheory,
    code: panelCode,
    calc: panelCalc,
  };

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      for (const [key, panel] of Object.entries(panels)) {
        panel.classList.toggle("active", key === tab);
      }

      if (tab === "matrix") renderTabMatrix();
      else if (tab === "decomp") renderTabDecomp();
      else if (tab === "exp") renderTabExp();
      else if (tab === "theory") renderTheoryPage(panelTheory);
      else if (tab === "code") renderCodePage(panelCode);
      else if (tab === "calc") renderCalcPage(panelCalc);
    });
  });
}

// ── INIT ────────────────────────────────────────────────────────────────────

function init() {
  try {
    if (!ensureNumeric()) return;
    setupTabs();
    renderTabMatrix();
  } catch(e) {
    console.error("Matrix Lab init error:", e);
    const errDiv = document.createElement("div");
    errDiv.style.cssText = "padding:2rem;color:#f87171;font-family:monospace;font-size:0.85rem";
    errDiv.innerHTML = `<strong>Ошибка инициализации:</strong><br>${e?.message || e}<br><br><pre>${e?.stack || ""}</pre>`;
    document.querySelector(".app-body")?.appendChild(errDiv);
  }
}

init();