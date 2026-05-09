import { latexToHtml } from "./dom.js";

const _A = "A = \\begin{bmatrix} 4 & 3 \\\\ 2 & 1 \\end{bmatrix}";

function tex(s, display = false) {
  return latexToHtml(s, display);
}

function m(expr) {
  return { tag: "math", value: expr };
}

function sub(title) {
  return { tag: "sub", value: title };
}

function p(text) {
  return text;
}

function section(title, content) {
  const sec = document.createElement("div");
  sec.className = "theory-section";
  const h2 = document.createElement("h2");
  h2.className = "theory-h2";
  h2.textContent = title;
  sec.appendChild(h2);
  for (const item of content) {
    if (typeof item === "string") {
      const el = document.createElement("div");
      el.className = "theory-p";
      el.innerHTML = item;
      sec.appendChild(el);
    } else if (item.tag === "math") {
      const el = document.createElement("div");
      el.className = "theory-math";
      el.innerHTML = tex(item.value, true);
      sec.appendChild(el);
    } else if (item.tag === "sub") {
      const h3 = document.createElement("h3");
      h3.className = "theory-h3";
      h3.textContent = item.value;
      sec.appendChild(h3);
    }
  }
  return sec;
}

export function renderTheoryPage(container) {
  container.innerHTML = `
    <div class="theory-intro">
      <h1>Матричные разложения на примере 2×2</h1>
      <p class="muted">Пошаговый разбор SVD, PCA, NMF, CUR, ALS для матрицы ${tex(_A)}</p>
    </div>
  `;

  const content = document.createElement("div");
  content.className = "theory-content";

  // ═══════════════════════════════════════════════════════════════════════════
  // SVD
  // ═══════════════════════════════════════════════════════════════════════════
  content.appendChild(section("1. SVD — сингулярное разложение", [
    sub("Шаг 1: транспонируем A"),
    p(`Меняем строки и столбцы местами:`),
    m("A^T = \\begin{bmatrix} 4 & 2 \\\\ 3 & 1 \\end{bmatrix}"),

    sub("Шаг 2: вычисляем AᵀA"),
    p(`Умножаем: элемент (i,j) это скалярное произведение i-й строки Aᵀ на j-й столбец A:`),
    p(`${tex("A^T A = \\begin{bmatrix} 4 & 2 \\\\ 3 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 4 & 3 \\\\ 2 & 1 \\end{bmatrix}")}`),
    m("A^T A = \\begin{bmatrix} 4\\cdot4+2\\cdot2 & 4\\cdot3+2\\cdot1 \\\\ 3\\cdot4+1\\cdot2 & 3\\cdot3+1\\cdot1 \\end{bmatrix}"),
    m("A^T A = \\begin{bmatrix} 16+4 & 12+2 \\\\ 12+2 & 9+1 \\end{bmatrix} = \\begin{bmatrix} 20 & 14 \\\\ 14 & 10 \\end{bmatrix}"),

    sub("Шаг 3: ищем собственные значения AᵀA"),
    p(`Решаем характеристическое уравнение ${tex("\\det(A^T A - \\lambda I) = 0")}:`),
    m("\\det \\begin{bmatrix} 20-\\lambda & 14 \\\\ 14 & 10-\\lambda \\end{bmatrix} = 0"),
    p(`Определитель матрицы 2×2: ad − bc:`),
    p(`${tex("(20-\\lambda)(10-\\lambda) - 14 \\cdot 14 = 0")}`),
    p(`Раскрываем скобки:`),
    p(`${tex("200 - 20\\lambda - 10\\lambda + \\lambda^2 - 196 = 0")}`),
    p(`${tex("\\lambda^2 - 30\\lambda + 4 = 0")}`),
    p(`Решаем квадратное уравнение. Дискриминант:`),
    m("D = (-30)^2 - 4 \\cdot 1 \\cdot 4 = 900 - 16 = 884"),
    p(`Корни:`),
    m("\\lambda_1 = \\frac{30 + \\sqrt{884}}{2} \\approx \\frac{30 + 29.732}{2} = \\frac{59.732}{2} \\approx 29.866"),
    m("\\lambda_2 = \\frac{30 - \\sqrt{884}}{2} \\approx \\frac{30 - 29.732}{2} = \\frac{0.268}{2} \\approx 0.134"),

    sub("Шаг 4: находим сингулярные числа"),
    p(`Сингулярные числа — это квадратные корни из собственных значений:`),
    m("\\sigma_1 = \\sqrt{\\lambda_1} = \\sqrt{29.866} \\approx 5.465"),
    m("\\sigma_2 = \\sqrt{\\lambda_2} = \\sqrt{0.134} \\approx 0.366"),

    sub("Шаг 5: находим правый сингулярный вектор v₁"),
    p(`Решаем систему ${tex("(A^T A - \\lambda_1 I)v_1 = 0")}:`),
    m("\\begin{bmatrix} 20-29.866 & 14 \\\\ 14 & 10-29.866 \\end{bmatrix} \\cdot \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix}"),
    m("\\begin{bmatrix} -9.866 & 14 \\\\ 14 & -19.866 \\end{bmatrix} \\cdot \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix}"),
    p(`Из первой строки: −9.866x + 14y = 0, значит 14y = 9.866x, откуда ${tex("y = \\frac{9.866}{14}x \\approx 0.705x")}.`),
    p(`Нормируем вектор (сумма квадратов компонент должна равняться единице):`),
    p(`${tex("x^2 + y^2 = 1")}`),
    p(`${tex("x^2 + (0.705x)^2 = 1")}`),
    p(`${tex("x^2 + 0.497x^2 = 1")}`),
    p(`${tex("1.497x^2 = 1")}`),
    p(`${tex("x^2 = 0.668, \\quad x \\approx 0.817")}`),
    p(`${tex("y = 0.705 \\cdot 0.817 \\approx 0.576")}`),
    m("v_1 = \\begin{bmatrix} 0.817 \\\\ 0.576 \\end{bmatrix}"),

    sub("Шаг 6: находим левый сингулярный вектор u₁"),
    p(`Формула: ${tex("u_1 = \\dfrac{A v_1}{\\sigma_1}")}`),
    p(`Сначала умножаем матрицу A на вектор v₁:`),
    m("A v_1 = \\begin{bmatrix} 4 & 3 \\\\ 2 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 0.817 \\\\ 0.576 \\end{bmatrix} = \\begin{bmatrix} 4\\cdot0.817 + 3\\cdot0.576 \\\\ 2\\cdot0.817 + 1\\cdot0.576 \\end{bmatrix}"),
    m("A v_1 = \\begin{bmatrix} 3.268 + 1.728 \\\\ 1.634 + 0.576 \\end{bmatrix} = \\begin{bmatrix} 4.996 \\\\ 2.210 \\end{bmatrix}"),
    p(`Делим на σ₁:`),
    m("u_1 = \\dfrac{1}{5.465} \\cdot \\begin{bmatrix} 4.996 \\\\ 2.210 \\end{bmatrix} = \\begin{bmatrix} 4.996/5.465 \\\\ 2.210/5.465 \\end{bmatrix} \\approx \\begin{bmatrix} 0.915 \\\\ 0.404 \\end{bmatrix}"),

    sub("Шаг 7: строим усечённую матрицу ранга 1"),
    p(`${tex("A_1 = \\sigma_1 \\cdot u_1 \\cdot v_1^T")}`),
    p(`Сначала перемножаем u₁ (столбец) и v₁ᵀ (строка):`),
    m("u_1 v_1^T = \\begin{bmatrix} 0.915 \\\\ 0.404 \\end{bmatrix} \\cdot \\begin{bmatrix} 0.817 & 0.576 \\end{bmatrix} = \\begin{bmatrix} 0.915\\cdot0.817 & 0.915\\cdot0.576 \\\\ 0.404\\cdot0.817 & 0.404\\cdot0.576 \\end{bmatrix}"),
    m("u_1 v_1^T = \\begin{bmatrix} 0.748 & 0.527 \\\\ 0.330 & 0.233 \\end{bmatrix}"),
    p(`Умножаем на σ₁ = 5.465:`),
    m("A_1 = 5.465 \\cdot \\begin{bmatrix} 0.748 & 0.527 \\\\ 0.330 & 0.233 \\end{bmatrix} = \\begin{bmatrix} 5.465\\cdot0.748 & 5.465\\cdot0.527 \\\\ 5.465\\cdot0.330 & 5.465\\cdot0.233 \\end{bmatrix}"),
    m("A_1 = \\begin{bmatrix} 4.088 & 2.880 \\\\ 1.803 & 1.273 \\end{bmatrix}"),
    p(`Сравниваем с исходной матрицей:`),
    p(`${tex(_A)} , ${tex("A_1 = \\begin{bmatrix} 4.09 & 2.88 \\\\ 1.80 & 1.27 \\end{bmatrix}")}`),
  ]));

  // ═══════════════════════════════════════════════════════════════════════════
  // PCA
  // ═══════════════════════════════════════════════════════════════════════════
  content.appendChild(section("2. PCA — метод главных компонент", [
    sub("Шаг 1: считаем средние значения по столбцам"),
    p(`${tex("\\mu_1 = \\frac{4+2}{2} = \\frac{6}{2} = 3, \\qquad \\mu_2 = \\frac{3+1}{2} = \\frac{4}{2} = 2")}`),

    sub("Шаг 2: центрируем данные"),
    p(`Вычитаем из каждого элемента среднее по его столбцу:`),
    m("A_{\\text{центр}} = \\begin{bmatrix} 4-3 & 3-2 \\\\ 2-3 & 1-2 \\end{bmatrix} = \\begin{bmatrix} 1 & 1 \\\\ -1 & -1 \\end{bmatrix}"),

    sub("Шаг 3: строим ковариационную матрицу"),
    p(`${tex("C = A_{\\text{центр}}^T \\cdot A_{\\text{центр}}")}`),
    p(`Транспонируем центрированную матрицу:`),
    m("A_{\\text{центр}}^T = \\begin{bmatrix} 1 & -1 \\\\ 1 & -1 \\end{bmatrix}"),
    p(`Перемножаем:`),
    m("C = \\begin{bmatrix} 1 & -1 \\\\ 1 & -1 \\end{bmatrix} \\cdot \\begin{bmatrix} 1 & 1 \\\\ -1 & -1 \\end{bmatrix} = \\begin{bmatrix} 1\\cdot1+(-1)\\cdot(-1) & 1\\cdot1+(-1)\\cdot(-1) \\\\ 1\\cdot1+(-1)\\cdot(-1) & 1\\cdot1+(-1)\\cdot(-1) \\end{bmatrix}"),
    m("C = \\begin{bmatrix} 1+1 & 1+1 \\\\ 1+1 & 1+1 \\end{bmatrix} = \\begin{bmatrix} 2 & 2 \\\\ 2 & 2 \\end{bmatrix}"),

    sub("Шаг 4: находим собственные значения"),
    p(`Решаем ${tex("\\det(C - \\lambda I) = 0")}:`),
    m("\\det \\begin{bmatrix} 2-\\lambda & 2 \\\\ 2 & 2-\\lambda \\end{bmatrix} = 0"),
    m("(2-\\lambda)(2-\\lambda) - 2 \\cdot 2 = 0"),
    m("(2-\\lambda)^2 - 4 = 0"),
    m("4 - 4\\lambda + \\lambda^2 - 4 = 0"),
    m("\\lambda^2 - 4\\lambda = 0"),
    m("\\lambda(\\lambda - 4) = 0"),
    m("\\lambda_1 = 4, \\quad \\lambda_2 = 0"),

    sub("Шаг 5: находим первый собственный вектор"),
    p(`Для λ₁ = 4 решаем ${tex("(C - 4I)v = 0")}:`),
    m("\\begin{bmatrix} 2-4 & 2 \\\\ 2 & 2-4 \\end{bmatrix} \\cdot \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix}"),
    m("\\begin{bmatrix} -2 & 2 \\\\ 2 & -2 \\end{bmatrix} \\cdot \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix}"),
    p(`Из первой строки: −2x + 2y = 0, значит 2y = 2x, или y = x.`),
    p(`Нормируем вектор:`),
    p(`${tex("x^2 + x^2 = 1, \\quad 2x^2 = 1, \\quad x^2 = \\frac{1}{2}, \\quad x = \\frac{1}{\\sqrt{2}} \\approx 0.707")}`),
    p(`${tex("y = 0.707")}`),
    m("v_1 = \\begin{bmatrix} 0.707 \\\\ 0.707 \\end{bmatrix}"),

    sub("Шаг 6: проецируем данные"),
    p(`Умножаем центрированную матрицу на главный вектор:`),
    m("Z = A_{\\text{центр}} \\cdot v_1 = \\begin{bmatrix} 1 & 1 \\\\ -1 & -1 \\end{bmatrix} \\cdot \\begin{bmatrix} 0.707 \\\\ 0.707 \\end{bmatrix}"),
    p(`Считаем каждую компоненту:`),
    p(`${tex("z_1 = 1 \\cdot 0.707 + 1 \\cdot 0.707 = 1.414")}`),
    p(`${tex("z_2 = (-1) \\cdot 0.707 + (-1) \\cdot 0.707 = -1.414")}`),
    m("Z = \\begin{bmatrix} 1.414 \\\\ -1.414 \\end{bmatrix}"),
    p(`Это координаты двух точек в новом базисе (вдоль главной компоненты).`),

    sub("Шаг 7: проверяем качество сжатия"),
    p(`Дисперсия исходных данных после центрирования:`),
    p(`${tex("\\text{Var} = \\frac{1^2 + (-1)^2 + 1^2 + (-1)^2}{1} = 4")}`),
    p(`Дисперсия проекции:`),
    p(`${tex("\\text{Var}_Z = \\frac{1.414^2 + (-1.414)^2}{1} = \\frac{2+2}{1} = 4")}`),
    p(`Совпадает с λ₁ = 4 — первая компонента сохранила всю дисперсию.`),
    p(`Восстанавливаем исходные данные:`),
    p(`${tex("A_{\\text{восст}} = Z \\cdot v_1^T + \\text{средние} = \\begin{bmatrix} 1.414 \\\\ -1.414 \\end{bmatrix} \\cdot \\begin{bmatrix} 0.707 & 0.707 \\end{bmatrix} + \\begin{bmatrix} 3 & 2 \\\\ 3 & 2 \\end{bmatrix}")}`),
    m("A_{\\text{восст}} = \\begin{bmatrix} 1.0+3 & 1.0+2 \\\\ -1.0+3 & -1.0+2 \\end{bmatrix} = \\begin{bmatrix} 4 & 3 \\\\ 2 & 1 \\end{bmatrix} = A"),
    p(`Восстановили идеально, потому что данные изначально лежали на прямой.`),
  ]));

  // ═══════════════════════════════════════════════════════════════════════════
  // NMF
  // ═══════════════════════════════════════════════════════════════════════════
  content.appendChild(section("3. NMF — неотрицательное матричное разложение", [
    sub("Шаг 1: инициализация"),
    p(`Выбираем ранг r = 1, задаём начальные неотрицательные матрицы:`),
    m("W = \\begin{bmatrix} 2 \\\\ 1 \\end{bmatrix}, \\qquad H = \\begin{bmatrix} 1 & 1.5 \\end{bmatrix}"),
    p(`Проверим произведение WH:`),
    m("WH = \\begin{bmatrix} 2 \\\\ 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 1 & 1.5 \\end{bmatrix} = \\begin{bmatrix} 2\\cdot1 & 2\\cdot1.5 \\\\ 1\\cdot1 & 1\\cdot1.5 \\end{bmatrix} = \\begin{bmatrix} 2 & 3 \\\\ 1 & 1.5 \\end{bmatrix}"),

    sub("Шаг 2: обновляем матрицу H"),
    p(`Правило обновления: ${tex("H \\leftarrow H \\odot \\dfrac{W^T A}{W^T W H}")}, где ⊙ — поэлементное умножение, дробь — поэлементное деление.`),
    p(`Транспонируем W:`),
    m("W^T = \\begin{bmatrix} 2 & 1 \\end{bmatrix}"),
    p(`Вычисляем числитель WᵀA:`),
    m("W^T A = \\begin{bmatrix} 2 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 4 & 3 \\\\ 2 & 1 \\end{bmatrix} = \\begin{bmatrix} 2\\cdot4+1\\cdot2 & 2\\cdot3+1\\cdot1 \\end{bmatrix} = \\begin{bmatrix} 8+2 & 6+1 \\end{bmatrix} = \\begin{bmatrix} 10 & 7 \\end{bmatrix}"),
    p(`Вычисляем WᵀW:`),
    m("W^T W = \\begin{bmatrix} 2 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 2 \\\\ 1 \\end{bmatrix} = 2\\cdot2 + 1\\cdot1 = 4 + 1 = 5"),
    p(`Вычисляем знаменатель WᵀW·H:`),
    m("W^T W H = 5 \\cdot \\begin{bmatrix} 1 & 1.5 \\end{bmatrix} = \\begin{bmatrix} 5 & 7.5 \\end{bmatrix}"),
    p(`Делим поэлементно числитель на знаменатель:`),
    m("\\frac{W^T A}{W^T W H} = \\begin{bmatrix} \\dfrac{10}{5} & \\dfrac{7}{7.5} \\end{bmatrix} = \\begin{bmatrix} 2 & 0.933 \\end{bmatrix}"),
    p(`Умножаем поэлементно на старую H:`),
    m("H_{\\text{нов}} = \\begin{bmatrix} 1\\cdot2 & 1.5\\cdot0.933 \\end{bmatrix} = \\begin{bmatrix} 2 & 1.4 \\end{bmatrix}"),

    sub("Шаг 3: обновляем матрицу W"),
    p(`Правило обновления: ${tex("W \\leftarrow W \\odot \\dfrac{A H^T}{W H H^T}")}`),
    p(`Транспонируем новую H:`),
    m("H^T = \\begin{bmatrix} 2 \\\\ 1.4 \\end{bmatrix}"),
    p(`Вычисляем числитель A·Hᵀ:`),
    m("A H^T = \\begin{bmatrix} 4 & 3 \\\\ 2 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 2 \\\\ 1.4 \\end{bmatrix} = \\begin{bmatrix} 4\\cdot2 + 3\\cdot1.4 \\\\ 2\\cdot2 + 1\\cdot1.4 \\end{bmatrix} = \\begin{bmatrix} 8 + 4.2 \\\\ 4 + 1.4 \\end{bmatrix} = \\begin{bmatrix} 12.2 \\\\ 5.4 \\end{bmatrix}"),
    p(`Вычисляем H·Hᵀ:`),
    m("H H^T = \\begin{bmatrix} 2 & 1.4 \\end{bmatrix} \\cdot \\begin{bmatrix} 2 \\\\ 1.4 \\end{bmatrix} = 2\\cdot2 + 1.4\\cdot1.4 = 4 + 1.96 = 5.96"),
    p(`Вычисляем знаменатель W·H·Hᵀ:`),
    m("W H H^T = W \\cdot (H H^T) = \\begin{bmatrix} 2 \\\\ 1 \\end{bmatrix} \\cdot 5.96 = \\begin{bmatrix} 11.92 \\\\ 5.96 \\end{bmatrix}"),
    p(`Делим поэлементно:`),
    m("\\frac{A H^T}{W H H^T} = \\begin{bmatrix} \\dfrac{12.2}{11.92} \\\\ \\dfrac{5.4}{5.96} \\end{bmatrix} = \\begin{bmatrix} 1.024 \\\\ 0.906 \\end{bmatrix}"),
    p(`Умножаем поэлементно на старую W:`),
    m("W_{\\text{нов}} = \\begin{bmatrix} 2\\cdot1.024 \\\\ 1\\cdot0.906 \\end{bmatrix} = \\begin{bmatrix} 2.048 \\\\ 0.906 \\end{bmatrix}"),
  ]));

  // ═══════════════════════════════════════════════════════════════════════════
  // CUR
  // ═══════════════════════════════════════════════════════════════════════════
  content.appendChild(section("4. CUR — разложение по строкам и столбцам", [
    sub("Шаг 1: вычисляем длины строк"),
    p(`Квадрат длины строки (для вероятностного отбора):`),
    m("\\|\\text{строка}_1\\|^2 = 4^2 + 3^2 = 16 + 9 = 25"),
    m("\\|\\text{строка}_2\\|^2 = 2^2 + 1^2 = 4 + 1 = 5"),
    p(`Строка 1 имеет бóльшую длину, выбираем её.`),

    sub("Шаг 2: вычисляем длины столбцов"),
    p(`Квадрат длины столбца:`),
    m("\\|\\text{столбец}_1\\|^2 = 4^2 + 2^2 = 16 + 4 = 20"),
    m("\\|\\text{столбец}_2\\|^2 = 3^2 + 1^2 = 9 + 1 = 10"),
    p(`Столбец 1 имеет бóльшую длину, выбираем его.`),

    sub("Шаг 3: формируем матрицы C и R"),
    p(`C — выбранный столбец, R — выбранная строка:`),
    m("C = \\begin{bmatrix} 4 \\\\ 2 \\end{bmatrix}, \\qquad R = \\begin{bmatrix} 4 & 3 \\end{bmatrix}"),

    sub("Шаг 4: вычисляем псевдообратную C⁺"),
    p(`Формула: ${tex("C^+ = (C^T C)^{-1} C^T")}`),
    p(`Транспонируем C:`),
    m("C^T = \\begin{bmatrix} 4 & 2 \\end{bmatrix}"),
    p(`Вычисляем CᵀC:`),
    m("C^T C = \\begin{bmatrix} 4 & 2 \\end{bmatrix} \\cdot \\begin{bmatrix} 4 \\\\ 2 \\end{bmatrix} = 4\\cdot4 + 2\\cdot2 = 16 + 4 = 20"),
    p(`Обратное число: ${tex("(C^T C)^{-1} = \\frac{1}{20} = 0.05")}`),
    p(`Умножаем:`),
    m("C^+ = 0.05 \\cdot \\begin{bmatrix} 4 & 2 \\end{bmatrix} = \\begin{bmatrix} 0.2 & 0.1 \\end{bmatrix}"),

    sub("Шаг 5: вычисляем псевдообратную R⁺"),
    p(`Формула: ${tex("R^+ = R^T (R R^T)^{-1}")}`),
    p(`Транспонируем R:`),
    m("R^T = \\begin{bmatrix} 4 \\\\ 3 \\end{bmatrix}"),
    p(`Вычисляем R·Rᵀ:`),
    m("R R^T = \\begin{bmatrix} 4 & 3 \\end{bmatrix} \\cdot \\begin{bmatrix} 4 \\\\ 3 \\end{bmatrix} = 4\\cdot4 + 3\\cdot3 = 16 + 9 = 25"),
    p(`Обратное число: ${tex("(R R^T)^{-1} = \\frac{1}{25} = 0.04")}`),
    p(`Умножаем:`),
    m("R^+ = \\begin{bmatrix} 4 \\\\ 3 \\end{bmatrix} \\cdot 0.04 = \\begin{bmatrix} 0.16 \\\\ 0.12 \\end{bmatrix}"),

    sub("Шаг 6: вычисляем связующую матрицу U"),
    p(`${tex("U = C^+ \\cdot A \\cdot R^+")}`),
    p(`Сначала умножаем A на R⁺:`),
    m("A R^+ = \\begin{bmatrix} 4 & 3 \\\\ 2 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 0.16 \\\\ 0.12 \\end{bmatrix} = \\begin{bmatrix} 4\\cdot0.16 + 3\\cdot0.12 \\\\ 2\\cdot0.16 + 1\\cdot0.12 \\end{bmatrix} = \\begin{bmatrix} 0.64 + 0.36 \\\\ 0.32 + 0.12 \\end{bmatrix} = \\begin{bmatrix} 1.0 \\\\ 0.44 \\end{bmatrix}"),
    p(`Теперь умножаем C⁺ на результат:`),
    m("U = \\begin{bmatrix} 0.2 & 0.1 \\end{bmatrix} \\cdot \\begin{bmatrix} 1.0 \\\\ 0.44 \\end{bmatrix} = 0.2\\cdot1.0 + 0.1\\cdot0.44 = 0.2 + 0.044 = 0.244"),

    sub("Шаг 7: восстанавливаем матрицу"),
    p(`${tex("A \\approx C \\cdot U \\cdot R")}`),
    p(`Сначала умножаем C на U:`),
    m("C \\cdot U = \\begin{bmatrix} 4 \\\\ 2 \\end{bmatrix} \\cdot 0.244 = \\begin{bmatrix} 4\\cdot0.244 \\\\ 2\\cdot0.244 \\end{bmatrix} = \\begin{bmatrix} 0.976 \\\\ 0.488 \\end{bmatrix}"),
    p(`Теперь умножаем результат на R:`),
    m("(CU) \\cdot R = \\begin{bmatrix} 0.976 \\\\ 0.488 \\end{bmatrix} \\cdot \\begin{bmatrix} 4 & 3 \\end{bmatrix} = \\begin{bmatrix} 0.976\\cdot4 & 0.976\\cdot3 \\\\ 0.488\\cdot4 & 0.488\\cdot3 \\end{bmatrix} = \\begin{bmatrix} 3.904 & 2.928 \\\\ 1.952 & 1.464 \\end{bmatrix}"),
    p(`Сравниваем с исходной:`),
    p(`${tex(_A)} , ${tex("CUR = \\begin{bmatrix} 3.90 & 2.93 \\\\ 1.95 & 1.46 \\end{bmatrix}")}`),
  ]));

  // ═══════════════════════════════════════════════════════════════════════════
  // ALS
  // ═══════════════════════════════════════════════════════════════════════════
  content.appendChild(section("5. ALS — попеременные наименьшие квадраты", [
    sub("Шаг 1: инициализация"),
    p(`Выбираем ранг r = 1, задаём начальные матрицы:`),
    m("X = \\begin{bmatrix} 1.5 \\\\ 0.8 \\end{bmatrix}, \\qquad Y = \\begin{bmatrix} 2.0 \\\\ 1.5 \\end{bmatrix}"),
    p(`Проверим XYᵀ:`),
    m("X Y^T = \\begin{bmatrix} 1.5 \\\\ 0.8 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.0 & 1.5 \\end{bmatrix} = \\begin{bmatrix} 1.5\\cdot2.0 & 1.5\\cdot1.5 \\\\ 0.8\\cdot2.0 & 0.8\\cdot1.5 \\end{bmatrix} = \\begin{bmatrix} 3.0 & 2.25 \\\\ 1.6 & 1.2 \\end{bmatrix}"),

    sub("Шаг 2: фиксируем Y, обновляем первую строку X"),
    p(`Формула для строки i: ${tex("x_i = \\dfrac{A_{i:} \\cdot Y}{Y^T Y}")}`),
    p(`Считаем знаменатель YᵀY (одинаков для всех строк):`),
    m("Y^T = \\begin{bmatrix} 2.0 & 1.5 \\end{bmatrix}"),
    m("Y^T Y = \\begin{bmatrix} 2.0 & 1.5 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.0 \\\\ 1.5 \\end{bmatrix} = 2.0\\cdot2.0 + 1.5\\cdot1.5 = 4.0 + 2.25 = 6.25"),
    p(`Считаем числитель для первой строки (${tex("A_{1:} = [4 \\; 3]")}):`),
    m("A_{1:} \\cdot Y = \\begin{bmatrix} 4 & 3 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.0 \\\\ 1.5 \\end{bmatrix} = 4\\cdot2.0 + 3\\cdot1.5 = 8.0 + 4.5 = 12.5"),
    p(`Делим:`),
    m("x_1 = \\frac{12.5}{6.25} = 2.0"),

    sub("Шаг 3: обновляем вторую строку X"),
    p(`Считаем числитель для второй строки (${tex("A_{2:} = [2 \\; 1]")}):`),
    m("A_{2:} \\cdot Y = \\begin{bmatrix} 2 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.0 \\\\ 1.5 \\end{bmatrix} = 2\\cdot2.0 + 1\\cdot1.5 = 4.0 + 1.5 = 5.5"),
    p(`Делим на тот же знаменатель:`),
    m("x_2 = \\frac{5.5}{6.25} = 0.88"),
    p(`Получаем обновлённую матрицу X:`),
    m("X_{\\text{нов}} = \\begin{bmatrix} 2.0 \\\\ 0.88 \\end{bmatrix}"),

    sub("Шаг 4: фиксируем X, обновляем первый столбец Y"),
    p(`Формула для столбца j: ${tex("y_j = \\dfrac{A_{:j}^T \\cdot X}{X^T X}")}`),
    p(`Считаем знаменатель XᵀX (одинаков для всех столбцов):`),
    m("X^T = \\begin{bmatrix} 2.0 & 0.88 \\end{bmatrix}"),
    m("X^T X = \\begin{bmatrix} 2.0 & 0.88 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.0 \\\\ 0.88 \\end{bmatrix} = 2.0\\cdot2.0 + 0.88\\cdot0.88 = 4.0 + 0.7744 = 4.7744"),
    p(`Считаем числитель для первого столбца (${tex("A_{:1} = [4 \\; 2]^T")}):`),
    m("A_{:1}^T = \\begin{bmatrix} 4 & 2 \\end{bmatrix}"),
    m("A_{:1}^T \\cdot X = \\begin{bmatrix} 4 & 2 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.0 \\\\ 0.88 \\end{bmatrix} = 4\\cdot2.0 + 2\\cdot0.88 = 8.0 + 1.76 = 9.76"),
    p(`Делим:`),
    m("y_1 = \\frac{9.76}{4.7744} \\approx 2.044"),

    sub("Шаг 5: обновляем второй столбец Y"),
    p(`Считаем числитель для второго столбца (${tex("A_{:2} = [3 \\; 1]^T")}):`),
    m("A_{:2}^T = \\begin{bmatrix} 3 & 1 \\end{bmatrix}"),
    m("A_{:2}^T \\cdot X = \\begin{bmatrix} 3 & 1 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.0 \\\\ 0.88 \\end{bmatrix} = 3\\cdot2.0 + 1\\cdot0.88 = 6.0 + 0.88 = 6.88"),
    p(`Делим:`),
    m("y_2 = \\frac{6.88}{4.7744} \\approx 1.441"),
    p(`Получаем обновлённую матрицу Y:`),
    m("Y_{\\text{нов}} = \\begin{bmatrix} 2.044 \\\\ 1.441 \\end{bmatrix}"),

    sub("Шаг 6: вычисляем новое приближение"),
    m("X_{\\text{нов}} Y_{\\text{нов}}^T = \\begin{bmatrix} 2.0 \\\\ 0.88 \\end{bmatrix} \\cdot \\begin{bmatrix} 2.044 & 1.441 \\end{bmatrix}"),
    m("X Y^T = \\begin{bmatrix} 2.0\\cdot2.044 & 2.0\\cdot1.441 \\\\ 0.88\\cdot2.044 & 0.88\\cdot1.441 \\end{bmatrix} = \\begin{bmatrix} 4.088 & 2.882 \\\\ 1.799 & 1.268 \\end{bmatrix}"),
    p(`Сравниваем с исходной:`),
    p(`${tex(_A)} , ${tex("X Y^T = \\begin{bmatrix} 4.09 & 2.88 \\\\ 1.80 & 1.27 \\end{bmatrix}")}`),
  ]));

  container.appendChild(content);
}
