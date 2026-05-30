const API_URL = "http://localhost:5000/api/decompose";
const calcState = { rows: 2, cols: 2, iters: 10, values: [[4, 3], [2, 1]] };

function buildMatrixInput(rows, cols) {
  const table = document.createElement("table");
  table.className = "calc-matrix-table";
  table.style.cssText = "border-collapse:collapse";
  for (let i = 0; i < rows; i++) {
    const tr = document.createElement("tr");
    for (let j = 0; j < cols; j++) {
      const td = document.createElement("td");
      td.style.padding = "1px";
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = "any";
      const stored = calcState.values[i] && calcState.values[i][j] !== undefined;
      inp.value = stored ? String(calcState.values[i][j]) : "0";
      inp.style.cssText =
        "width:60px;height:36px;text-align:center;background:rgba(0,0,0,0.25);border:1px solid var(--border);color:var(--text);border-radius:6px;font-family:monospace;font-size:0.85rem";
      inp.addEventListener("input", saveValues);
      td.appendChild(inp);
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
  return table;
}

function saveValues() {
  const inputs = document.querySelectorAll("#calc-grid input");
  const r = calcState.rows;
  const c = calcState.cols;
  const vals = [];
  let idx = 0;
  for (let i = 0; i < r; i++) {
    const row = [];
    for (let j = 0; j < c; j++) {
      row.push(parseFloat(inputs[idx]?.value) || 0);
      idx++;
    }
    vals.push(row);
  }
  calcState.values = vals;
}

// ── helpers for colourful heatmaps ──

function parseLatexMatrices(latex) {
  const re = /\\begin\{bmatrix\}([\s\S]*?)\\end\{bmatrix\}/g;
  const results = [];
  let m;
  while ((m = re.exec(latex)) !== null) {
    const rows = m[1].split("\\\\").map(r =>
      r.split("&").map(c => {
        const cleaned = c.trim();
        const num = parseFloat(cleaned);
        return isNaN(num) ? null : num;
      })
    ).filter(r => r.length > 0);
    if (rows.length && rows.every(r => r.every(v => v !== null))) {
      results.push(rows);
    }
  }
  return results.length ? results : null;
}

function renderMatrixHeatmap(values) {
  if (!values || !values.length) return null;
  const flat = values.flat().filter(v => v !== null);
  if (!flat.length) return null;
  const mx = Math.max(...flat.map(Math.abs), 1e-12);
  const wrap = document.createElement("span");
  wrap.style.cssText = "display:inline-grid;vertical-align:middle;margin-left:0.5rem;" +
    `grid-template-columns:repeat(${values[0].length},28px);gap:2px`;
  let cellIdx = 0;
  for (const row of values) {
    for (const v of row) {
      if (v === null) continue;
      const t = v / mx;
      const r = Math.round(255 * Math.max(0, t));
      const b = Math.round(255 * Math.max(0, -t));
      const g = Math.round(255 * (1 - Math.abs(t)));
      const cell = document.createElement("span");
      cell.style.cssText = `display:inline-flex;align-items:center;justify-content:center;` +
        `width:28px;height:26px;border-radius:4px;font-size:0.65rem;font-family:monospace;` +
        `color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.5);` +
        `background:rgb(${r},${g},${b});cursor:default;`;
      cell.textContent = v.toFixed(v % 1 === 0 ? 0 : 2);
      cell.animate([
        { opacity: 0, transform: "scale(0)" },
        { opacity: 1, transform: "scale(1.15)", offset: 0.6 },
        { opacity: 1, transform: "scale(1)" }
      ], { duration: 350, easing: "cubic-bezier(0.34,1.56,0.64,1)", delay: cellIdx * 60, fill: "both" });
      cell.addEventListener("mouseenter", () => { cell.style.transform = "scale(1.25)"; cell.style.boxShadow = "0 0 12px rgba(255,255,255,0.5)"; });
      cell.addEventListener("mouseleave", () => { cell.style.transform = ""; cell.style.boxShadow = ""; });
      wrap.appendChild(cell);
      cellIdx++;
    }
  }
  return wrap;
}

function animateStep(el, delay, type) {
  const dur = type === "math" ? 550 : 450;
  el.animate([
    { opacity: 0, transform: "translateX(-20px) scale(0.95)" },
    { opacity: 1, transform: "translateX(0) scale(1)" }
  ], { duration: dur, easing: "cubic-bezier(0.34,1.56,0.64,1)", delay: delay * 1000, fill: "both" });
  const glow = document.createElement("div");
  glow.style.cssText = "position:absolute;inset:0;border-radius:8px;pointer-events:none;" +
    `background:${type === "math" ? "rgba(99,102,241,0.12)" : "rgba(99,102,241,0.06)"};`;
  glow.animate([
    { opacity: 0 }, { opacity: 1, offset: 0.2 }, { opacity: 0 }
  ], { duration: 700, easing: "ease", delay: delay * 1000, fill: "both" });
  el.style.position = "relative";
  el.appendChild(glow);
}

// Рендерит страницу матричного калькулятора
export function renderCalcPage(container) {
  container.innerHTML = "";

  const style = document.createElement("style");
  style.textContent = `
    .calc-layout { display: flex; flex-direction: column; gap: 1.2rem; }
    .calc-card { background: linear-gradient(135deg, var(--panel2), var(--panel)); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem 1.4rem; }
    .calc-card h3 { font-size: 1rem; font-weight: 700; margin-bottom: 0.8rem; color: var(--accent); }
    .calc-controls { display: flex; align-items: flex-end; gap: 1rem; flex-wrap: wrap; }
    .calc-controls label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.8rem; color: var(--muted); }
    .calc-controls input, .calc-controls select { background: rgba(0,0,0,0.25); border: 1px solid var(--border); color: var(--text); padding: 0.35rem 0.6rem; border-radius: 6px; font-family: inherit; font-size: 0.85rem; width: 80px; }
    .calc-btn { background: var(--accent); color: #fff; border: none; padding: 0.45rem 1.2rem; border-radius: 8px; font-weight: 600; font-size: 0.85rem; cursor: pointer; transition: var(--transition); }
    .calc-btn:hover { filter: brightness(1.15); }
    .calc-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .calc-error { color: var(--bad); font-size: 0.85rem; padding: 0.5rem; background: rgba(248,113,113,0.1); border-radius: 8px; }
    .calc-matrix-table input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
    .calc-step-num { display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;border-radius:50%;background:var(--accent);color:#fff;font-size:0.65rem;font-weight:700;margin-right:0.5rem;flex-shrink:0;animation:cellPop 0.35s cubic-bezier(0.34,1.56,0.64,1) both; }
    .calc-step-num--math { margin:-0.1rem 0.3rem 0 0;align-self:flex-start; }
    .calc-step-text { font-size: 0.85rem; line-height: 1.6; margin-bottom: 0.3rem; color: var(--text); padding: 0.3rem 0.6rem; border-left: 3px solid rgba(99,102,241,0.4); background: rgba(99,102,241,0.04); border-radius: 0 6px 6px 0; display:flex;align-items:baseline;gap:0.2rem; }
    .calc-step-math { overflow-x: auto; padding: 0.5rem 0.7rem; margin: 0.3rem 0; background: linear-gradient(135deg, rgba(0,0,0,0.15), rgba(99,102,241,0.06)); border: 1px solid var(--border); border-radius: 8px; text-align: center; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 0.4rem; }
    .calc-step-math .katex { font-size: 1em; }
    .calc-step-sub { font-size: 0.88rem; font-weight: 600; margin: 0.6rem 0 0.2rem; color: var(--text); padding-left: 0.4rem; border-left: 3px solid var(--accent); }
    .calc-spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes stepSlideIn { from { opacity:0; transform:translateX(-20px) scale(0.95) } to { opacity:1; transform:translateX(0) scale(1) } }
    @keyframes stepGlow { 0% { opacity:0 } 20% { opacity:1 } 100% { opacity:0 } }
    @keyframes cellPop { 0% { opacity:0; transform:scale(0) } 60% { transform:scale(1.15) } 100% { opacity:1; transform:scale(1) } }
    .calc-result-block { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; border-left: 3px solid var(--accent); }
    .calc-result-block:nth-child(2) { border-left-color: #a78bfa; }
    .calc-result-block:nth-child(3) { border-left-color: #34d399; }
    .calc-result-block:nth-child(4) { border-left-color: #fb923c; }
    .calc-result-block:nth-child(5) { border-left-color: #f472b6; }
    .calc-result-head {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0.6rem 1rem; cursor: pointer;
      background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(99,102,241,0.03));
      user-select: none; transition: var(--transition);
    }
    .calc-result-head:hover { background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(99,102,241,0.06)); }
    .calc-result-head h4 { font-size: 0.9rem; font-weight: 600; color: var(--accent); margin: 0; }
    .calc-result-head .step-count { font-size: 0.72rem; color: var(--muted); }
    .calc-result-head .arrow { font-size: 0.7rem; color: var(--muted); transition: transform 0.2s; }
    .calc-result-head .arrow.open { transform: rotate(90deg); }
    .calc-result-body { padding: 0.6rem 1rem 1rem; display: none; }
    .calc-result-body.open { display: block; }
  `;
  container.appendChild(style);

  const layout = document.createElement("div");
  layout.className = "calc-layout";
  container.appendChild(layout);

  // ── input card ──
  const inputCard = document.createElement("div");
  inputCard.className = "calc-card";
  inputCard.innerHTML = `<h3>Введите матрицу</h3>`;
  layout.appendChild(inputCard);

  const controlsRow = document.createElement("div");
  controlsRow.className = "calc-controls";
  controlsRow.innerHTML = `
    <label>Строк <input id="calc-rows" type="number" min="2" max="6" value="${calcState.rows}"></label>
    <label>Столбцов <input id="calc-cols" type="number" min="2" max="6" value="${calcState.cols}"></label>
    <label>Итераций <input id="calc-iters" type="number" min="1" max="20" value="${calcState.iters}"></label>
  `;
  inputCard.appendChild(controlsRow);

  const gridHost = document.createElement("div");
  gridHost.id = "calc-grid";
  gridHost.style.marginTop = "0.8rem";
  gridHost.appendChild(buildMatrixInput(calcState.rows, calcState.cols));
  inputCard.appendChild(gridHost);

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "margin-top:0.8rem;display:flex;align-items:center;gap:1rem";
  const calcBtn = document.createElement("button");
  calcBtn.className = "calc-btn";
  calcBtn.textContent = "Разложить";
  btnRow.appendChild(calcBtn);
  const statusEl = document.createElement("span");
  statusEl.style.fontSize = "0.85rem";
  statusEl.style.color = "var(--muted)";
  btnRow.appendChild(statusEl);
  inputCard.appendChild(btnRow);

  // ── resize ──
  const rowsInp = document.getElementById("calc-rows");
  const colsInp = document.getElementById("calc-cols");

  function rebuildGrid() {
    const r = Math.max(2, Math.min(6, parseInt(rowsInp.value) || 2));
    const c = Math.max(2, Math.min(6, parseInt(colsInp.value) || 2));
    calcState.rows = r;
    calcState.cols = c;
    if (!calcState.values || calcState.values.length !== r || (calcState.values[0] && calcState.values[0].length !== c)) {
      const v = [];
      for (let i = 0; i < r; i++) { const row = []; for (let j = 0; j < c; j++) row.push(0); v.push(row); }
      if (r >= 2 && c >= 2) { v[0][0]=4; v[0][1]=3; v[1][0]=2; v[1][1]=1; }
      calcState.values = v;
    }
    gridHost.innerHTML = "";
    gridHost.appendChild(buildMatrixInput(r, c));
  }

  rowsInp.addEventListener("change", rebuildGrid);
  colsInp.addEventListener("change", rebuildGrid);

  // ── results ──
  const resultsCard = document.createElement("div");
  resultsCard.className = "calc-card";
  resultsCard.id = "calc-results";
  resultsCard.innerHTML = '<h3>Результаты разложений</h3><p class="muted" style="font-size:0.85rem">Нажмите «Разложить» — получите пошаговый расчёт</p>';
  layout.appendChild(resultsCard);

  // ── calc handler ──
  calcBtn.addEventListener("click", async () => {
    const r = Math.max(2, Math.min(6, parseInt(rowsInp.value) || 2));
    const c = Math.max(2, Math.min(6, parseInt(colsInp.value) || 2));
    const iters = Math.max(1, Math.min(20, parseInt(document.getElementById("calc-iters").value) || 10));

    saveValues();
    const matrix = calcState.values;

    calcBtn.disabled = true;
    calcBtn.textContent = "Расчёт…";
    statusEl.innerHTML = '<span class="calc-spinner"></span>';

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matrix, iters }),
      });
      const data = await resp.json();
      if (data.status !== "ok") {
        statusEl.innerHTML = `<span class="calc-error">Ошибка: ${data.message}</span>`;
        return;
      }
      statusEl.textContent = "Готово!";
      renderResults(resultsCard, data);
    } catch {
      statusEl.innerHTML =
        '<span class="calc-error">Сервер недоступен. Запустите: <code style="font-size:0.8rem">python3 server.py</code></span>';
    } finally {
      calcBtn.disabled = false;
      calcBtn.textContent = "Разложить";
    }
  });
}

// ── render results ──

function renderResults(card, data) {
  card.innerHTML = `<h3>Результаты разложений</h3>
    <p class="muted" style="font-size:0.82rem;margin-bottom:0.8rem;display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem">
      <span>Исходная: <span class="math">${data.matrix_latex}</span></span>
      <span style="color:var(--muted)">(${data.shape[0]}×${data.shape[1]})</span>
    </p>
    <div class="calc-results"></div>`;

  const headerP = card.querySelector("p");
  const allOrig = parseLatexMatrices(data.matrix_latex);
  if (allOrig) {
    for (const m of allOrig) {
      const hm = renderMatrixHeatmap(m);
      if (hm) headerP.appendChild(hm);
    }
  }

  const host = card.querySelector(".calc-results");
  const order = ["svd", "pca", "nmf", "cur", "als"];
  const labels = {
    svd: "SVD — сингулярное разложение",
    pca: "PCA — метод главных компонент",
    nmf: "NMF — неотрицательное разложение",
    cur: "CUR — разложение по строкам/столбцам",
    als: "ALS — попеременные квадраты",
  };

  for (const key of order) {
    const dec = data.decompositions[key];
    if (!dec || !dec.available) continue;

    const block = document.createElement("div");
    block.className = "calc-result-block";

    const head = document.createElement("div");
    head.className = "calc-result-head";
    const stepCount = dec.steps.filter(s => s.type === "math" || s.type === "text" || s.type === "sub").length;
    head.innerHTML = `
      <h4>${labels[key]}</h4>
      <div style="display:flex;align-items:center;gap:0.6rem">
        <span class="step-count">${stepCount} шагов</span>
        <span class="arrow">▶</span>
      </div>`;
    block.appendChild(head);

    const body = document.createElement("div");
    body.className = "calc-result-body";
    block.appendChild(body);

    // открываем SVD до анимации — иначе анимация не видна (display:none / вне DOM)
    if (key === "svd") body.classList.add("open");

    let stepIdx = 0;
    let globalStepIdx = 0;
    for (const step of dec.steps) {
      if (step.type === "sub") {
        const el = document.createElement("div");
        el.className = "calc-step-sub";
        el.textContent = step.content;
        body.appendChild(el);
        animateStep(el, stepIdx * 0.04, "sub");
        stepIdx++;
      } else if (step.type === "text") {
        const el = document.createElement("div");
        el.className = "calc-step-text";
        const badge = document.createElement("span");
        badge.className = "calc-step-num";
        badge.textContent = String(++globalStepIdx);
        badge.animate([
          { opacity: 0, transform: "scale(0)" },
          { opacity: 1, transform: "scale(1.15)", offset: 0.6 },
          { opacity: 1, transform: "scale(1)" }
        ], { duration: 350, easing: "cubic-bezier(0.34,1.56,0.64,1)", delay: (stepIdx * 0.04 + 0.1) * 1000, fill: "both" });
        el.appendChild(badge);
        const content = document.createElement("span");
        content.textContent = step.content;
        el.appendChild(content);
        body.appendChild(el);
        animateStep(el, stepIdx * 0.04, "text");
        stepIdx++;
      } else if (step.type === "math") {
        const el = document.createElement("div");
        el.className = "calc-step-math";
        const badge = document.createElement("span");
        badge.className = "calc-step-num calc-step-num--math";
        badge.textContent = String(++globalStepIdx);
        badge.animate([
          { opacity: 0, transform: "scale(0)" },
          { opacity: 1, transform: "scale(1.15)", offset: 0.6 },
          { opacity: 1, transform: "scale(1)" }
        ], { duration: 350, easing: "cubic-bezier(0.34,1.56,0.64,1)", delay: (stepIdx * 0.04 + 0.1) * 1000, fill: "both" });
        el.appendChild(badge);
        try {
          if (window.katex) {
            el.innerHTML = window.katex.renderToString(step.content, {
              displayMode: true,
              throwOnError: false,
            });
          } else {
            el.textContent = step.content;
          }
        } catch {
          el.textContent = step.content;
        }
        
        body.appendChild(el);
        animateStep(el, stepIdx * 0.04, "math");
        stepIdx++;
      }
    }

    host.appendChild(block);

    // toggle
    head.addEventListener("click", () => {
      const isOpen = body.classList.toggle("open");
      head.querySelector(".arrow").classList.toggle("open", isOpen);
    });

    // open first block by default (arrow class only — body уже открыт)
    if (key === "svd") {
      head.querySelector(".arrow").classList.add("open");
    }
  }
}
