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

export function clear(node) {
  if (node) node.innerHTML = "";
}

export function readNumber(id, fallback) {
  const v = Number(document.getElementById(id)?.value);
  return Number.isFinite(v) ? v : fallback;
}

export function readChecked(id) {
  return Boolean(document.getElementById(id)?.checked);
}

export function latexToHtml(str, display = false) {
  try {
    return window.katex.renderToString(str, { displayMode: display, throwOnError: false, trust: true });
  } catch {
    return `<span class="mono">${str}</span>`;
  }
}

export function katex(str, display = false) {
  return latexToHtml(str, display);
}

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

export function renderSigmaChart(container, Sk, r) {
  if (!Sk || !Sk.length) return;

  const wrap = el("div", { className: "sigma-chart" });

  for (let idx = 0; idx < Math.min(Sk.length, r); idx++) {
    const v = Sk[idx];
    const barWrap = el("div", { className: "sigma-row" });

    const label = el("span", { className: "sigma-row__label" });
    label.innerHTML = latexToHtml(`\\sigma_{${idx + 1}}`);

    const barOuter = el("div", { className: "sigma-row__bar-wrap", style: "position:relative" });
    const bar = el("div", { className: "sigma-row__bar" });
    bar.style.width = "0%";
    bar.style.width = ((v / Sk[0]) * 100).toFixed(1) + "%";

    const tooltip = el("span", { className: "sigma-tooltip" });
    tooltip.textContent = v.toFixed(4);
    tooltip.style.position = "absolute";
    tooltip.style.left = "100%";
    tooltip.style.marginLeft = "6px";
    tooltip.style.top = "50%";
    tooltip.style.transform = "translateY(-50%)";

    barOuter.appendChild(bar);
    barOuter.appendChild(tooltip);

    const val = el("span", { className: "sigma-row__val" });
    val.textContent = v.toFixed(4);

    barWrap.appendChild(label);
    barWrap.appendChild(barOuter);
    barWrap.appendChild(val);
    wrap.appendChild(barWrap);
  }

  container.appendChild(wrap);
}

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