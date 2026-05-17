/* global numeric */

import { clear, readChecked, readNumber, latexToHtml, renderAccordion, renderConvergenceChart, renderSigmaChart, renderIterationSlider } from "./dom.js";
import { dims, diff, minMax, absMean, frobNorm, randomMatrix, transpose, dot as dotM, diag } from "./matrix.js";
import { renderLegend, renderMatrixBlock } from "./heatmap.js";
import { svdTruncated, pcaReconstruct, pcaReconstructWithSteps, nmfReconstruct, nmfReconstructHistory, curReconstruct, alsReconstruct, alsReconstructHistory } from "./decompositions.js";
import { renderCenterEdgeExperiment, renderPerturbation } from "./experiments.js";
import { renderTheoryPage } from "./theory.js";
import { renderCodePage } from "./code.js";
import { renderCalcPage } from "./calculator.js";
import { renderVisualizerPage } from "./visualizer.js";

const panelVisual = document.getElementById("panel-visual");
const panelExp = document.getElementById("panel-exp");
const panelTheory = document.getElementById("panel-theory");
const panelCode = document.getElementById("panel-code");
const panelCalc = document.getElementById("panel-calc");

const state = {
  seed: 42,
  range: [0, 10],
  A: randomMatrix(4, 4, 0, 10, 42),
  k: 2,
  iters: 80,
  algos: { svd: true, pca: true, nmf: true, cur: true, als: true },
  lockScale: false,
  visAlgo: "svd",
  visA: null,
  visK: 2,
  visIters: 20,
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

// ── TAB SWITCHING ───────────────────────────────────────────────────────────

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = {
    visual: panelVisual,
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

      if (tab === "visual") renderVisualizerPage(panelVisual, state);
      else if (tab === "exp") renderTabExp();
      else if (tab === "theory") renderTheoryPage(panelTheory);
      else if (tab === "code") renderCodePage(panelCode);
      else if (tab === "calc") renderCalcPage(panelCalc);
    });
  });
}

// ── TAB: EXPERIMENTS ────────────────────────────────────────────────────────

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

// ── INIT ────────────────────────────────────────────────────────────────────

function init() {
  try {
    if (!ensureNumeric()) return;
    setupTabs();
    renderVisualizerPage(panelVisual, state);
  } catch(e) {
    console.error("Matrix Lab init error:", e);
    const errDiv = document.createElement("div");
    errDiv.style.cssText = "padding:2rem;color:#f87171;font-family:monospace;font-size:0.85rem";
    errDiv.innerHTML = `<strong>Ошибка инициализации:</strong><br>${e?.message || e}<br><br><pre>${e?.stack || ""}</pre>`;
    document.querySelector(".app-body")?.appendChild(errDiv);
  }
}

init();
