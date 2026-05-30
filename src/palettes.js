import { clamp } from "./rng.js";

// Цветовая схема viridis: t∈[0,1] → {r,g,b} (интерполяция 5 опорных точек)
export function viridis(t) {
  const c = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  t = clamp(t, 0, 1);
  const i = Math.min(c.length - 2, Math.floor(t * (c.length - 1)));
  const f = t * (c.length - 1) - i;
  const a = c[i];
  const b = c[i + 1];
  const r = Math.round(a[0] + f * (b[0] - a[0]));
  const g = Math.round(a[1] + f * (b[1] - a[1]));
  const bl = Math.round(a[2] + f * (b[2] - a[2]));
  return { r, g, b: bl };
}

// Diverging-схема: синий (<0) → нейтральный → красный (>0), масштаб vmax
export function diverging(v, vmax) {
  const t = vmax < 1e-12 ? 0 : clamp(v / vmax, -1, 1);
  // blue -> neutral -> red
  const blue = { r: 65, g: 105, b: 225 };
  const neutral = { r: 230, g: 230, b: 230 };
  const red = { r: 220, g: 20, b: 60 };

  if (t === 0) return neutral;
  if (t > 0) {
    const f = t;
    return {
      r: Math.round(neutral.r + f * (red.r - neutral.r)),
      g: Math.round(neutral.g + f * (red.g - neutral.g)),
      b: Math.round(neutral.b + f * (red.b - neutral.b)),
    };
  }
  const f = -t;
  return {
    r: Math.round(neutral.r + f * (blue.r - neutral.r)),
    g: Math.round(neutral.g + f * (blue.g - neutral.g)),
    b: Math.round(neutral.b + f * (blue.b - neutral.b)),
  };
}

// {r,g,b} → "rgb(r,g,b)"
export function rgb({ r, g, b }) {
  return `rgb(${r},${g},${b})`;
}

