import { viridis, diverging, rgb } from "./palettes.js";
import { minMax } from "./matrix.js";

function renderTitle(title) {
  if (!/[_^\\]/.test(title)) return title;
  const stripped = title.replace(/\\text\{[^}]*\}/g, "");
  if (/[а-яА-Я]/.test(stripped)) return title;
  try {
    return window.katex.renderToString(title, { throwOnError: false, trust: true });
  } catch {
    return title;
  }
}

export function renderLegend(host, kinds) {
  const set = new Set(kinds);
  const wrap = document.createElement("div");
  wrap.className = "legend";
  if (set.has("sequential")) {
    const bar = document.createElement("div");
    bar.className = "bar seq";
    const txt = document.createElement("div");
    txt.className = "text";
    txt.textContent = "последовательная — значения матрицы, мин → макс";
    wrap.appendChild(bar);
    wrap.appendChild(txt);
  }
  if (set.has("diverging")) {
    const bar = document.createElement("div");
    bar.className = "bar div";
    const txt = document.createElement("div");
    txt.className = "text";
    txt.textContent = "diverging — ошибка вокруг 0, синий < 0, красный > 0";
    wrap.appendChild(bar);
    wrap.appendChild(txt);
  }
  host.appendChild(wrap);
}

export function renderMatrixBlock(parent, title, A, options = {}) {
  const { subtitle, selected, badge, onCellClick, diverge = false, scale, onDblClick, onCellCreate } = options;
  const { mn, mx } = scale || minMax(A);
  const m = A.length;
  const n = A[0].length;

  const block = document.createElement("div");
  block.className = "matrix-block";

  const head = document.createElement("div");
  head.className = "matrix-title";
  const h3 = document.createElement("h3");
  h3.innerHTML = renderTitle(title);
  const dimsEl = document.createElement("div");
  dimsEl.className = "dims";
  dimsEl.textContent = `[${m}\u00D7${n}]`;
  head.appendChild(h3);
  head.appendChild(dimsEl);
  block.appendChild(head);

  if (subtitle) {
    const sub = document.createElement("div");
    sub.className = "matrix-sub";
    sub.innerHTML = subtitle;
    block.appendChild(sub);
  }

  const grid = document.createElement("div");
  grid.className = "matrix-grid";
  grid.style.gridTemplateColumns = `repeat(${n}, 1fr)`;

  const vmax = diverge ? Math.max(Math.abs(mn), Math.abs(mx), 1e-12) : 0;

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const v = A[i][j];
      const cell = document.createElement("div");
      cell.className = "cell";

      if (diverge) {
        const c = diverging(v, vmax);
        cell.style.background = rgb(c);
        const sat = Math.abs(v) / vmax;
        cell.style.color = sat > 0.55 ? "#fff" : "#080c14";
      } else {
        const t = (v - mn) / (mx - mn);
        const c = viridis(t);
        cell.style.background = rgb(c);
        cell.style.color = t > 0.6 ? "#080c14" : "#fff";
      }

      if (onCellClick || onDblClick) cell.classList.add("clickable");
      if (selected && selected[0] === i && selected[1] === j) cell.classList.add("selected");
      if (badge && badge[0] === i && badge[1] === j) cell.classList.add("badge");

      cell.textContent = Number.isFinite(v) ? v.toFixed(2) : String(v);

      const originalV = v;

      cell.addEventListener("click", () => {
        if (onCellClick) onCellClick(i, j);
      });

      if (onDblClick) {
        cell.addEventListener("dblclick", (e) => {
          e.stopPropagation();
          startEdit(cell, originalV, (newVal) => onDblClick(i, j, newVal));
        });
      }

      if (onCellCreate) onCellCreate(cell, i, j);

      grid.appendChild(cell);
    }
  }

  block.appendChild(grid);
  parent.appendChild(block);
}

function startEdit(cell, originalVal, onCommit) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "cell-input";
  input.value = typeof originalVal === "number" ? originalVal.toFixed(4) : String(originalVal);

  cell.textContent = "";
  cell.appendChild(input);
  input.focus();
  input.select();

  function commit() {
    const v = Number(input.value);
    if (Number.isFinite(v)) {
      onCommit(v);
    }
    cell.textContent = Number.isFinite(originalVal) ? originalVal.toFixed(2) : String(originalVal);
  }

  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    }
    if (e.key === "Escape") {
      cell.textContent = Number.isFinite(originalVal) ? originalVal.toFixed(2) : String(originalVal);
    }
  });
}