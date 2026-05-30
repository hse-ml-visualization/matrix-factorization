import { absMean, addAt, centerMask, clone, diff, dims, frobNorm, meanAbsInMask, randomMatrix } from "./matrix.js";
import { renderLegend, renderMatrixBlock } from "./heatmap.js";
import { el, clear, latexToHtml, renderBarChart, renderRadialDecay } from "./dom.js";
import { alsReconstruct, curReconstruct, nmfReconstruct, pcaReconstruct, svdTruncated } from "./decompositions.js";

function tex(latex) {
  return latexToHtml(latex, false);
}

// Формирует список алгоритмов для эксперимента на основе state
export function algoListFromState(state, rangeLo = null) {
  const lo = rangeLo !== null && Number.isFinite(rangeLo) ? rangeLo : state.range[0];
  const algos = [];
  if (state.algos.svd) algos.push({ id: "svd", name: "SVD", run: (A) => svdTruncated(A, state.k).Ahat });
  if (state.algos.pca) algos.push({ id: "pca", name: "PCA", run: (A) => pcaReconstruct(A, state.k).Ahat });
  if (state.algos.nmf) {
    if (lo >= 0) algos.push({ id: "nmf", name: "NMF", run: (A) => nmfReconstruct(A, state.k, state.iters).Ahat });
    else algos.push({ id: "nmf", name: "NMF", run: null, skip: true });
  }
  if (state.algos.cur) algos.push({ id: "cur", name: "CUR", run: (A) => curReconstruct(A, state.k).Ahat });
  if (state.algos.als) algos.push({ id: "als", name: "ALS", run: (A) => alsReconstruct(A, state.k, state.iters).Ahat });
  return algos;
}

function radialDecay(M, i0, j0) {
  const m = M.length;
  const n = M[0].length;
  const bins = new Map();
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const d = Math.abs(i - i0) + Math.abs(j - j0);
      const cur = bins.get(d) || { sum: 0, count: 0 };
      cur.sum += Math.abs(M[i][j]);
      cur.count += 1;
      bins.set(d, cur);
    }
  }
  const ds = [...bins.keys()].sort((a, b) => a - b).slice(0, 8);
  return ds.map((d) => ({ d, val: bins.get(d).sum / bins.get(d).count }));
}

// Эксперимент: сравнивает качество восстановления центра и краёв матрицы
export function renderCenterEdgeExperiment(container, state) {
  clear(container);

  const desc = el("p", { className: "muted", style: "margin-bottom:0.7rem; max-width:72ch" });
  desc.innerHTML = `Статистика по ${tex("N")} случайным матрицам одного размера с ${tex("A")} — сравнение ошибки в центре и на краях гипотеза: центр аппроксимируется хуже потому что там концентрируется больше информации`;
  container.appendChild(desc);

  const controls = el("div", { className: "controls grid2" }, [
    el("label", {}, [el("span", { text: "Число матриц N" }), el("input", { id: "exp-n", type: "number", min: "10", max: "5000", step: "10", value: "400" })]),
    el("label", {}, [el("span", { text: "Диапазон элементов" }), el("select", { id: "exp-range" }, [
      el("option", { value: "0,10", text: "[0; 10]" }),
      el("option", { value: "-1,1", text: "[-1; 1]" }),
      el("option", { value: "-10,10", text: "[-10; 10]" }),
    ])]),
  ]);
  container.appendChild(controls);

  const btn = el("button", { className: "primary", text: "Запустить эксперимент", onClick: run });
  container.appendChild(el("div", { className: "controls" }, [btn]));

  const out = el("div", {});
  container.appendChild(out);

  function run() {
    const N = Math.max(10, Number(document.getElementById("exp-n").value || 400));
    const [lo, hi] = String(document.getElementById("exp-range").value).split(",").map(Number);
    const { m, n } = dims(state.A);
    const mask = centerMask(m, n);
    const algos = algoListFromState(state, lo);
    const runnable = algos.filter((a) => !a.skip);
    if (!runnable.length) {
      out.innerHTML = `<p class="muted">Выберите хотя бы один алгоритм</p>`;
      return;
    }

    out.innerHTML = `<p class="muted">Считаю ${N} матриц\u2026</p>`;

    const stats = runnable.map((a) => ({ ...a, sumC: 0, sumE: 0, cWorse: 0, eWorse: 0, tie: 0, count: 0 }));
    for (let r = 0; r < N; r++) {
      const A = randomMatrix(m, n, lo, hi, state.seed + r);
      for (const st of stats) {
        try {
          const Ahat = st.run(A);
          const ec = meanAbsInMask(A, Ahat, mask, true);
          const ee = meanAbsInMask(A, Ahat, mask, false);
          st.sumC += ec;
          st.sumE += ee;
          st.count++;
          if (ec > ee + 1e-12) st.cWorse++;
          else if (ee > ec + 1e-12) st.eWorse++;
          else st.tie++;
        } catch (_) { }
      }
    }

    clear(out);

    const total = stats.reduce((acc, s) => acc + s.count, 0);
    const header = el("div", { className: "muted", style: "margin-bottom:0.5rem; font-size:0.8rem" });
    header.innerHTML = `${tex("N=" + N + ", " + m + "\\times" + n)} — ${total} успешных прогонов`;
    out.appendChild(header);

    renderBarChart(out, stats.map((s) => ({
      name: s.name,
      cWorse: s.cWorse,
      eWorse: s.eWorse,
      tie: s.tie,
    })));

    const detailWrap = el("div", { style: "margin-top:0.75rem" });
    out.appendChild(detailWrap);

    for (const s of stats) {
      if (s.count === 0) continue;
      const row = el("div", { style: "font-size:0.78rem; color:var(--muted); margin-bottom:0.2rem" });
      row.textContent = `${s.name}: центр=${(s.sumC / s.count).toFixed(5)}, края=${(s.sumE / s.count).toFixed(5)} — ${s.count} успешных прогонов`;
      detailWrap.appendChild(row);
    }
  }
}

// Эксперимент: возмущает один элемент матрицы и оценивает влияние на Ahat
export function renderPerturbation(container, state) {
  clear(container);

  const desc = el("p", { className: "muted", style: "margin-bottom:0.7rem; max-width:72ch" });
  desc.innerHTML = `Возмущение ${tex("\\Delta")} в ячейке — крутите колёсико на ячейке ${tex("A")}, ${tex("\\Delta")} меняется, эффект виден на ${tex("\\tilde A")}`;
  container.appendChild(desc);

  let p = { baseA: null, sel: [0, 0], delta: 0, size: { m: 6, n: 6 } };
  p.baseA = clone(state.A);

  const sizeControls = el("div", { className: "controls grid2" }, [
    el("label", {}, [el("span", { text: "m" }), el("input", { id: "pert-m", type: "number", min: 2, max: 20, value: String(p.size.m) })]),
    el("label", {}, [el("span", { text: "n" }), el("input", { id: "pert-n", type: "number", min: 2, max: 20, value: String(p.size.n) })]),
  ]);
  container.appendChild(sizeControls);

  const btnRow = el("div", { className: "controls" }, [
    el("button", { text: "\u041D\u043E\u0432\u0430\u044F A", onClick: newBase }),
    el("button", { text: "\u0421\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u0394=0", onClick: () => { p.delta = 0; redraw(); } }),
  ]);
  container.appendChild(btnRow);

  const deltaInfo = el("div", { className: "muted", style: "font-size:0.82rem; margin:0.3rem 0" });
  container.appendChild(deltaInfo);

  const deltaBadge = el("div", { style: "font-size:1.6rem; font-weight:700; font-variant-numeric:tabular-nums; letter-spacing:-0.02em; margin:0.4rem 0 0.2rem; color:var(--accent)" });
  container.appendChild(deltaBadge);

  const view = el("div", {});
  container.appendChild(view);

  for (const id of ["pert-m", "pert-n"]) {
    document.getElementById(id).addEventListener("change", () => {
      p.size.m = Number(document.getElementById("pert-m").value) || 6;
      p.size.n = Number(document.getElementById("pert-n").value) || 6;
      const [lo, hi] = state.range;
      p.baseA = randomMatrix(p.size.m, p.size.n, lo, hi, state.seed);
      p.sel = [0, 0];
      p.delta = 0;
      redraw();
    });
  }

  function newBase() {
    const [lo, hi] = state.range;
    p.baseA = randomMatrix(p.size.m, p.size.n, lo, hi, state.seed);
    p.sel = [0, 0];
    p.delta = 0;
    redraw();
  }

  function makeWheelEditable(cellEl, i, j) {
    cellEl.addEventListener("wheel", (e) => {
      e.preventDefault();
      p.sel = [i, j];
      const dir = e.deltaY < 0 ? 0.1 : -0.1;
      const step = e.shiftKey ? 1 : 0.1;
      p.delta = Math.round((p.delta + dir * step) * 100) / 100;
      redraw();
    }, { passive: false });
  }

  function redraw() {
    clear(view);
    const A = p.baseA;
    const Aprime = addAt(A, p.sel[0], p.sel[1], p.delta);
    const algos = algoListFromState(state);
    const runnable = algos.filter((a) => !a.skip);
    if (!runnable.length) {
      view.innerHTML = `<p class="muted">Выберите хотя бы один алгоритм</p>`;
      return;
    }

    deltaBadge.innerHTML = `${tex("\\Delta")} = ${p.delta > 0 ? "+" : ""}${p.delta.toFixed(2)}`;
    deltaInfo.innerHTML = `${tex("\\Delta")} = ${p.delta.toFixed(2)} в ячейке [${p.sel[0]}][${p.sel[1]}] ${tex("\\to")} A' = A с ${tex("\\Delta")} в этой ячейке`;

    renderLegend(view, ["sequential", "diverging"]);

    const row0 = el("div", { className: "matrix-row" });
    renderMatrixBlock(row0, "A", A, {
      selected: p.sel,
      badge: p.sel,
      onCellClick: (i, j) => {
        p.sel = [i, j];
        redraw();
      },
      onCellCreate: makeWheelEditable,
    });
    renderMatrixBlock(row0, `A' = A`, Aprime, {
      badge: p.sel,
    });
    view.appendChild(row0);

    const algosRow = el("div", { className: "matrix-row" });
    for (const algo of runnable) {
      try {
        const Ah = algo.run(A);
        const AhP = algo.run(Aprime);
        const dAhat = diff(AhP, Ah);
        const decay = radialDecay(dAhat, p.sel[0], p.sel[1]);

        const card = el("div", { style: "display:flex; flex-direction:column; gap:0.3rem" });

        const header = el("div", { style: "font-size:0.82rem; font-weight:600" });
        header.innerHTML = `${algo.name}: ${tex("\\tilde A' - \\tilde A")}`;
        card.appendChild(header);

        const metrics = el("div", { className: "muted", style: "font-size:0.75rem; font-family:ui-monospace,monospace" });
        metrics.innerHTML = `mean|ΔÃ|=${absMean(dAhat).toFixed(5)} \u00B7 ${latexToHtml("\\|\\Delta\\tilde A\\|_F")}=${frobNorm(dAhat).toFixed(5)}`;
        card.appendChild(metrics);

        const gridWrap = el("div", {});
        renderMatrixBlock(gridWrap, "", dAhat, { diverge: true, selected: p.sel });
        card.appendChild(gridWrap);

        const decayTitle = el("div", { className: "muted", style: "font-size:0.75rem; margin-top:0.3rem" });
        decayTitle.textContent = "Среднее |ΔÃ| по расстоянию (d — манхэттенское)";
        card.appendChild(decayTitle);

        const decayWrap = el("div", {});
        renderRadialDecay(decayWrap, decay);
        card.appendChild(decayWrap);

        algosRow.appendChild(card);
      } catch (e) {
        const fail = el("div", { className: "muted", style: "font-size:0.8rem" });
        fail.textContent = `${algo.name}: ошибка — ${e?.message || e}`;
        algosRow.appendChild(fail);
      }
    }
    view.appendChild(algosRow);
  }

  redraw();
}