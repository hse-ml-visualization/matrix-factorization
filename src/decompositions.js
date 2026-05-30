/* global numeric */

import { clone, dims, dot, transpose, diag, zeros, minMax, diff, frobNorm } from "./matrix.js";
import { mulberry32 } from "./rng.js";

// Берёт первые k столбцов матрицы M
function takeCols(M, k) {
  const m = M.length;
  const out = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(k);
    for (let t = 0; t < k; t++) row[t] = M[i][t];
    out[i] = row;
  }
  return out;
}

// Детерминированный seed из данных матрицы и соли
function seededFromMatrix(A, salt = 0) {
  let h = (2166136261 ^ (salt >>> 0)) >>> 0;
  const m = A.length;
  const n = A[0].length;
  h = Math.imul(h ^ m, 16777619) >>> 0;
  h = Math.imul(h ^ n, 16777619) >>> 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const v = Number.isFinite(A[i][j]) ? A[i][j] : 0;
      const q = Math.round(v * 1e6);
      h = Math.imul(h ^ (q >>> 0), 16777619) >>> 0;
    }
  }
  if (h === 0) h = 123456789;
  return h >>> 0;
}

// Усечённое SVD ранга k; при m<n транспонирует вход для numeric.svd
export function svdTruncated(A, k) {
  const { m, n } = dims(A);
  const needTranspose = m < n;
  const X = needTranspose ? transpose(A) : A;
  const { U, S, V } = numeric.svd(X);
  const r = Math.max(1, Math.min(k, S.length, U[0].length, V.length, m, n));
  let Uk = takeCols(U, r);
  let Sk = S.slice(0, r);
  let Vk = takeCols(V, r);
  if (needTranspose) {
    const tmp = Uk; Uk = Vk; Vk = tmp;
    const tmp2 = U; U = V; V = tmp2;
  }
  const US = dot(Uk, diag(Sk));
  const Ahat = dot(US, transpose(Vk));
  return { Ahat, U, S, V, r, Uk, Sk, Vk, US };
}

// PCA: реконструкция через SVD; возвращает только Ahat, mean, S, r
export function pcaReconstruct(A, k) {
  const { Ahat, mean, S, r } = pcaReconstructWithSteps(A, k);
  return { Ahat, mean, S, r };
}

// PCA с промежуточными шагами (центрирование, SVD, сдвиг обратно)
export function pcaReconstructWithSteps(A, k) {
  const { m, n } = dims(A);
  const mean = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) mean[j] += A[i][j];
    mean[j] /= m;
  }
  const X = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) X[i][j] = A[i][j] - mean[j];
  const svdX = svdTruncated(X, k);
  const { Ahat: Xk, S, r, Uk, Sk, Vk, US } = svdX;
  const Ahat = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) Ahat[i][j] = Xk[i][j] + mean[j];
  return { Ahat, mean, X, S, r, Uk, Sk, Vk, US, Xk };
}

// NMF: реконструкция через мультипликативное обновление; возвращает только результат
export function nmfReconstruct(A, k, iters = 80) {
  return nmfReconstructHistory(A, k, iters, false).result;
}

// NMF с историей итераций; сдвигает матрицу для неотрицательности
export function nmfReconstructHistory(A, k, totalIters = 80, captureHistory = true) {
  const { m, n } = dims(A);
  const { mn } = minMax(A);
  const shift = mn < 0 ? -mn : 0;
  const X = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) X[i][j] = A[i][j] + shift;

  const maxCapture = 50;
  const stride = Math.max(1, Math.floor(totalIters / maxCapture));

  const eps = 1e-9;
  const rndBase = mulberry32(seededFromMatrix(A, (k << 16) ^ totalIters ^ 0x4e4d46));
  const rnd = () => 0.5 + rndBase();

  let W = zeros(m, k);
  let H = zeros(k, n);
  for (let i = 0; i < m; i++) for (let t = 0; t < k; t++) W[i][t] = rnd();
  for (let t = 0; t < k; t++) for (let j = 0; j < n; j++) H[t][j] = rnd();

  const history = [];
  const captureIters = new Set();

  if (captureHistory) {
    for (let i = 0; i <= totalIters; i += stride) {
      captureIters.add(i);
    }
    captureIters.add(totalIters);
  }

  const capture = (iter, WH) => {
    if (!captureHistory) return;
    if (captureIters.has(iter)) {
      const Ahat = zeros(m, n);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) Ahat[i][j] = WH[i][j] - shift;
      const err = diff(A, Ahat);
      const frob = frobNorm(err);
      history.push({ i: iter, frob, Ahat, W: clone(W), H: clone(H) });
    }
  };

  let WH = dot(W, H);
  capture(0, WH);

  for (let it = 0; it < totalIters; it++) {
    const WT = transpose(W);
    const WT_X = dot(WT, X);
    const WT_W = dot(WT, W);
    const WT_W_H = dot(WT_W, H);
    for (let t = 0; t < k; t++) {
      for (let j = 0; j < n; j++) {
        H[t][j] = H[t][j] * (WT_X[t][j] / (WT_W_H[t][j] + eps));
        if (!Number.isFinite(H[t][j]) || H[t][j] < 0) H[t][j] = 0;
      }
    }
    const HT = transpose(H);
    const X_HT = dot(X, HT);
    const H_HT = dot(H, HT);
    const W_H_HT = dot(W, H_HT);
    for (let i = 0; i < m; i++) {
      for (let t = 0; t < k; t++) {
        W[i][t] = W[i][t] * (X_HT[i][t] / (W_H_HT[i][t] + eps));
        if (!Number.isFinite(W[i][t]) || W[i][t] < 0) W[i][t] = 0;
      }
    }

    WH = dot(W, H);
    capture(it + 1, WH);
  }

  const finalAhat = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) finalAhat[i][j] = WH[i][j] - shift;

  return {
    result: { Ahat: finalAhat, shift, W, H },
    history,
  };
}

// Псевдообратная Мура–Пенроуза через SVD; при m<n транспонирует
export function pinv(A) {
  const m = A.length;
  const n = A[0].length;
  if (m < n) return transpose(pinv(transpose(A)));
  const { U, S, V } = numeric.svd(A);
  const r = Math.min(S.length, U[0].length, V.length);
  const tol = 1e-10 * Math.max(...S);
  const Sinv = zeros(r, r);
  for (let i = 0; i < r; i++) Sinv[i][i] = S[i] > tol ? 1 / S[i] : 0;
  const Ur = takeCols(U, r);
  const Vr = takeCols(V, r);
  return dot(dot(Vr, Sinv), transpose(Ur));
}

// CUR-разложение: выбирает r столбцов/строк по квадратам норм
export function curReconstruct(A, k) {
  const { m, n } = dims(A);
  const r = Math.max(1, Math.min(k, m, n));

  const colNorms = new Array(n).fill(0);
  const rowNorms = new Array(m).fill(0);
  for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) colNorms[j] += A[i][j] * A[i][j];
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) rowNorms[i] += A[i][j] * A[i][j];

  const topCols = [...Array(n).keys()].sort((a, b) => colNorms[b] - colNorms[a]).slice(0, r);
  const topRows = [...Array(m).keys()].sort((a, b) => rowNorms[b] - rowNorms[a]).slice(0, r);

  const C = zeros(m, r);
  const R = zeros(r, n);
  const Wcore = zeros(r, r);
  for (let i = 0; i < m; i++) for (let t = 0; t < r; t++) C[i][t] = A[i][topCols[t]];
  for (let t = 0; t < r; t++) for (let j = 0; j < n; j++) R[t][j] = A[topRows[t]][j];
  for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) Wcore[i][j] = A[topRows[i]][topCols[j]];

  const Cp = pinv(C);
  const Rp = pinv(R);
  const Uc = dot(dot(Cp, A), Rp);
  const Ahat = dot(dot(C, Uc), R);
  return { Ahat, C, U: Uc, R, Wcore, topRows, topCols, r, Cp, Rp };
}

// ALS: реконструкция через чередующиеся наименьшие квадраты; возвращает только результат
export function alsReconstruct(A, k, iters = 80, reg = 1e-2) {
  return alsReconstructHistory(A, k, iters, reg, false).result;
}

// ALS с историей итераций; чередует фиксацию X/Y с L2-регуляризацией
export function alsReconstructHistory(A, k, totalIters = 80, reg = 1e-2, captureHistory = true) {
  const { m, n } = dims(A);
  const r = Math.max(1, Math.min(k, m, n));
  const regSalt = Math.round(reg * 1e9) >>> 0;
  const rndBase = mulberry32(seededFromMatrix(A, (k << 16) ^ totalIters ^ regSalt ^ 0x414c53));
  const rnd = () => 0.5 + rndBase();

  let X = zeros(m, r);
  let Y = zeros(n, r);
  for (let i = 0; i < m; i++) for (let t = 0; t < r; t++) X[i][t] = rnd();
  for (let j = 0; j < n; j++) for (let t = 0; t < r; t++) Y[j][t] = rnd();

  const maxCapture = 50;
  const stride = Math.max(1, Math.floor(totalIters / maxCapture));

  const history = [];
  const captureIters = new Set();

  if (captureHistory) {
    for (let i = 0; i <= totalIters; i += stride) captureIters.add(i);
    captureIters.add(totalIters);
  }

  const capture = (iter, Xcur, Ycur) => {
    if (!captureHistory) return;
    if (captureIters.has(iter)) {
      const Ahat = dot(Xcur, transpose(Ycur));
      const err = diff(A, Ahat);
      const frob = frobNorm(err);
      history.push({ i: iter, frob, Ahat, X: clone(Xcur), Y: clone(Ycur) });
    }
  };

  capture(0, X, Y);

  for (let it = 0; it < totalIters; it++) {
    const YT = transpose(Y);
    const YTY = dot(YT, Y);
    const YTYreg = clone(YTY);
    for (let i = 0; i < r; i++) YTYreg[i][i] += reg;

    for (let i = 0; i < m; i++) {
      const rhs = new Array(r).fill(0);
      for (let t = 0; t < r; t++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += Y[j][t] * A[i][j];
        rhs[t] = s;
      }
      const sol = numeric.solve(YTYreg, rhs);
      for (let t = 0; t < r; t++) X[i][t] = sol[t];
    }

    const XT = transpose(X);
    const XTX = dot(XT, X);
    const XTXreg = clone(XTX);
    for (let i = 0; i < r; i++) XTXreg[i][i] += reg;

    for (let j = 0; j < n; j++) {
      const rhs = new Array(r).fill(0);
      for (let t = 0; t < r; t++) {
        let s = 0;
        for (let i = 0; i < m; i++) s += X[i][t] * A[i][j];
        rhs[t] = s;
      }
      const sol = numeric.solve(XTXreg, rhs);
      for (let t = 0; t < r; t++) Y[j][t] = sol[t];
    }

    capture(it + 1, X, Y);
  }

  const finalAhat = dot(X, transpose(Y));
  return {
    result: { Ahat: finalAhat, X, Y, r, reg },
    history,
  };
}
