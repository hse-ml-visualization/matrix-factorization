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
    .calc-step-text { font-size: 0.85rem; line-height: 1.6; margin-bottom: 0.3rem; color: var(--text); }
    .calc-step-math { overflow-x: auto; padding: 0.5rem 0.7rem; margin: 0.3rem 0; background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 8px; text-align: center; }
    .calc-step-math .katex { font-size: 1em; }
    .calc-step-sub { font-size: 0.88rem; font-weight: 600; margin: 0.6rem 0 0.2rem; color: var(--text); }
    .calc-spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .calc-result-block { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    .calc-result-head {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0.6rem 1rem; cursor: pointer;
      background: rgba(255,255,255,0.03);
      user-select: none; transition: var(--transition);
    }
    .calc-result-head:hover { background: rgba(255,255,255,0.06); }
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
    <p class="muted" style="font-size:0.82rem;margin-bottom:0.8rem">
      Исходная: <span class="math">${data.matrix_latex}</span>
      <span style="color:var(--muted)">(${data.shape[0]}×${data.shape[1]})</span>
    </p>
    <div class="calc-results"></div>`;

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

    for (const step of dec.steps) {
      if (step.type === "sub") {
        const el = document.createElement("div");
        el.className = "calc-step-sub";
        el.textContent = step.content;
        body.appendChild(el);
      } else if (step.type === "text") {
        const el = document.createElement("div");
        el.className = "calc-step-text";
        el.textContent = step.content;
        body.appendChild(el);
      } else if (step.type === "math") {
        const el = document.createElement("div");
        el.className = "calc-step-math";
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
      }
    }

    block.appendChild(body);
    host.appendChild(block);

    // toggle
    head.addEventListener("click", () => {
      const isOpen = body.classList.toggle("open");
      head.querySelector(".arrow").classList.toggle("open", isOpen);
    });

    // open first block by default
    if (key === "svd") {
      body.classList.add("open");
      head.querySelector(".arrow").classList.add("open");
    }
  }
}
