// Универсальный конструктор DOM-элементов с атрибутами и дочерними узлами
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, String(v));
  }
  for (const ch of children) node.appendChild(typeof ch === "string" ? document.createTextNode(ch) : ch);
  return node;
}

// Очищает innerHTML контейнера
export function clear(node) {
  if (node) node.innerHTML = "";
}

// Читает число из input по id, при невалидном числе — fallback
export function readNumber(id, fallback) {
  const v = Number(document.getElementById(id)?.value);
  return Number.isFinite(v) ? v : fallback;
}

// Читает checked из checkbox по id
export function readChecked(id) {
  return Boolean(document.getElementById(id)?.checked);
}

// Рендерит LaTeX-строку через KaTeX (displayMode по умолчанию false)
export function latexToHtml(str, display = false) {
  try {
    return window.katex.renderToString(str, { displayMode: display, throwOnError: false, trust: true });
  } catch {
    return `<span class="mono">${str}</span>`;
  }
}

// Алиас для latexToHtml
export function katex(str, display = false) {
  return latexToHtml(str, display);
}

// Аккордеон с раскрывающимися секциями
export function renderAccordion(container, items) {
  for (const item of items) {
    const itemEl = el("div", { className: "accordion__item" });

    const trigger = el("button", { className: "accordion__trigger" }, [
      el("span", { className: "accordion__trigger-name" }, [
        el("span", { text: item.name }),
        item.badge ? el("span", { className: "accordion__badge", text: item.badge }) : null,
      ].filter(Boolean)),
      el("span", { className: "accordion__arrow", text: "\u25BC" }),
    ]);

    const body = el("div", { className: "accordion__body" });

    trigger.addEventListener("click", () => {
      const open = itemEl.classList.contains("open");
      itemEl.classList.toggle("open", !open);
    });

    itemEl.appendChild(trigger);
    itemEl.appendChild(body);

    if (item.defaultOpen) itemEl.classList.add("open");

    container.appendChild(itemEl);

    if (item.render) item.render(body);
  }
}

// График сходимости frob-ошибки по итерациям
export function renderConvergenceChart(container, history) {
  if (!history || history.length < 2) return;

  const wrap = el("div", { className: "convergence-chart" });

  const title = el("div", { className: "convergence-chart__title" });
  title.innerHTML = `сходимость \u2014 ${latexToHtml("\\|A - \\tilde A\\|_F")} по итерациям \u2014 клик на столбец показывает соответствующее ${latexToHtml("\\tilde A")}`;
  wrap.appendChild(title);

  const bars = el("div", { className: "convergence-chart__bars" });

  const maxFrob = Math.max(...history.map((h) => h.frob), 1e-12);
  const minFrob = Math.min(...history.map((h) => h.frob));
  const logRange = Math.log10(maxFrob / Math.max(minFrob, 1e-12));

  const barEls = [];

  for (const h of history) {
    const barWrap = el("div", { className: "conv-bar-wrap" });

    const logH = Math.log10(Math.max(h.frob, 1e-12));
    const logMin = Math.log10(Math.max(minFrob, 1e-12));
    const heightPct = Math.max(2, ((logH - logMin) / logRange) * 100);

    const bar = el("div", { className: "conv-bar" });
    bar.style.height = heightPct + "%";

    const tooltip = el("div", { className: "conv-tooltip" });
    tooltip.textContent = `i=${h.i}: ${h.frob.toFixed(5)}`;

    barWrap.appendChild(bar);
    barWrap.appendChild(tooltip);
    bars.appendChild(barWrap);
    barEls.push({ wrap: barWrap, h });
  }

  wrap.appendChild(bars);
  container.appendChild(wrap);

  return barEls;
}

// Столбчатая диаграмма сингулярных чисел
export function renderSigmaChart(container, Sk, r) {
  if (!Sk || !Sk.length) return;

  const wrap = el("div", { className: "sigma-chart" });

  const svgNS = "http://www.w3.org/2000/svg";
  const W = 360, H = 120, pad = { top: 8, bottom: 24, left: 8, right: 8 };
  const chartW = W - pad.left - pad.right, chartH = H - pad.top - pad.bottom;
  const maxS = Math.max(...Sk, 1e-12);
  const barW = Math.max(2, chartW / Sk.length - 1);

  for (let i = 0; i < Sk.length; i++) {
    const pct = Sk[i] / maxS;
    const x = pad.left + i * (barW + 1);
    const h = Math.max(1, pct * chartH);
    const y = pad.top + chartH - h;
    const bar = document.createElementNS(svgNS, "rect");
    bar.setAttribute("x", x.toString());
    bar.setAttribute("y", y.toString());
    bar.setAttribute("width", barW.toString());
    bar.setAttribute("height", h.toString());
    bar.setAttribute("fill", i < r ? "var(--accent)" : "var(--panel3)");
    bar.setAttribute("rx", "1");
    wrap.appendChild(bar);
  }

  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  while (wrap.firstChild) svg.appendChild(wrap.firstChild);
  wrap.appendChild(svg);
  container.appendChild(wrap);
}

// Слайдер переключения между итерациями с кнопками play/pause
// Слайдер переключения между итерациями с кнопками play/pause
export function renderIterationSlider(container, history, onIterChange) {
  if (!history || history.length < 2) return;

  const maxIter = history[history.length - 1].i;
  const initialIter = history[Math.floor(history.length / 2)].i;

  const control = el("div", { className: "iter-control" });

  const header = el("div", { className: "iter-control__header" });
  const lbl = el("span", { className: "iter-control__label" });
  lbl.textContent = "Итерация";
  const val = el("span", { className: "iter-control__value" });
  val.textContent = `${initialIter} / ${maxIter}`;
  header.appendChild(lbl);
  header.appendChild(val);

  const slider = el("input", {
    type: "range",
    min: "0",
    max: String(history.length - 1),
    value: String(Math.floor(history.length / 2)),
  });

  slider.addEventListener("input", () => {
    const idx = Number(slider.value);
    const h = history[idx];
    val.textContent = `${h.i} / ${maxIter}`;
    if (onIterChange) onIterChange(h, idx);
  });

  control.appendChild(header);
  control.appendChild(slider);
  container.appendChild(control);

  return { slider, history, val, maxIter };
}

// Столбчатая диаграмма ошибок алгоритмов (frob-норма)
export function renderBarChart(container, algorithms) {
  const wrap = el("div", { className: "bar-chart" });

  for (const algo of algorithms) {
    const total = algo.cWorse + algo.eWorse + algo.tie;
    if (total === 0) continue;

    const row = el("div", { className: "bar-chart__row" });

    const label = el("span", { className: "bar-chart__label" });
    label.textContent = algo.name;
    row.appendChild(label);

    const track = el("div", { className: "bar-chart__track" });

    const seg = (cls, val, label) => {
      if (val === 0) return;
      const s = el("div", { className: `bar-chart__seg ${cls}` });
      const pct = (val / total) * 100;
      s.style.width = pct + "%";
      if (pct > 10) {
        s.textContent = pct.toFixed(0) + "%";
        s.style.color = "#0b1020";
      }
      track.appendChild(s);
    };

    seg("center", algo.cWorse);
    seg("edge", algo.eWorse);
    seg("tie", algo.tie);

    row.appendChild(track);
    wrap.appendChild(row);
  }

  const legend = el("div", { className: "bar-chart__legend" });
  const items = [
    { cls: "center", text: "\u0446\u0435\u043D\u0442\u0440 \u0445\u0443\u0436\u0435" },
    { cls: "edge", text: "\u043A\u0440\u0430\u044F \u0445\u0443\u0436\u0435" },
    { cls: "tie", text: "\u043F\u043E\u0447\u0442\u0438 \u0440\u0430\u0432\u043D\u043E" },
  ];
  for (const it of items) {
    const li = el("div", { className: "bar-chart__legend-item" });
    const dot = el("span", { className: "bar-chart__legend-dot", style: `background: var(--${it.cls === "center" ? "bad" : it.cls === "edge" ? "accent" : "good"})` });
    li.appendChild(dot);
    li.appendChild(document.createTextNode(it.text));
    legend.appendChild(li);
  }

  wrap.appendChild(legend);
  container.appendChild(wrap);
}

// Радиальный график затухания (для эксперимента с возмущением)
export function renderRadialDecay(container, decay) {
  if (!decay || !decay.length) return;

  const wrap = el("div", { className: "radial-decay" });
  const maxVal = Math.max(...decay.map((d) => d.val), 1e-12);

  for (const d of decay) {
    const barWrap = el("div", { className: "radial-decay__bar-wrap" });
    const bar = el("div", { className: "radial-decay__bar" });
    bar.style.height = ((d.val / maxVal) * 100).toFixed(1) + "%";
    const dLabel = el("div", { className: "radial-decay__d" });
    dLabel.textContent = `d${d.d}`;
    const vLabel = el("div", { className: "radial-decay__val" });
    vLabel.textContent = d.val.toFixed(3);
    barWrap.appendChild(bar);
    barWrap.appendChild(dLabel);
    barWrap.appendChild(vLabel);
    wrap.appendChild(barWrap);
  }

  container.appendChild(wrap);
}