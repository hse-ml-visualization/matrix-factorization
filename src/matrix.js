/* global numeric */

import { mulberry32 } from "./rng.js";

export function zeros(m, n) {
  const A = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = 0;
    A[i] = row;
  }
  return A;
}

export function clone(A) {
  return A.map((r) => r.slice());
}

export function dims(A) {
  return { m: A.length, n: A[0].length };
}

export function randomMatrix(m, n, lo, hi, seed = 42) {
  const rnd = mulberry32(seed);
  const A = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = lo + rnd() * (hi - lo);
    A[i] = row;
  }
  return A;
}

export function minMax(A) {
  let mn = Infinity;
  let mx = -Infinity;
  for (const row of A) {
    for (const v of row) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) return { mn: 0, mx: 1 };
  if (mn === mx) mx = mn + 1e-9;
  return { mn, mx };
}

export function diff(A, B) {
  const m = A.length;
  const n = A[0].length;
  const D = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = A[i][j] - B[i][j];
    D[i] = row;
  }
  return D;
}

export function addAt(A, i, j, delta) {
  const B = clone(A);
  B[i][j] += delta;
  return B;
}

export function absMean(A) {
  let s = 0;
  let c = 0;
  for (const row of A) for (const v of row) (s += Math.abs(v)), c++;
  return c ? s / c : 0;
}

export function frobNorm(A) {
  let s = 0;
  for (const row of A) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

export function centerMask(m, n) {
  const rs = Math.floor(m / 4);
  const re = Math.floor((3 * m) / 4);
  const cs = Math.floor(n / 4);
  const ce = Math.floor((3 * n) / 4);
  const mask = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = i >= rs && i < re && j >= cs && j < ce;
    mask[i] = row;
  }
  return mask;
}

export function meanAbsInMask(A, B, mask, inside) {
  const m = A.length;
  const n = A[0].length;
  let s = 0;
  let c = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if ((mask[i][j] && inside) || (!mask[i][j] && !inside)) {
        const d = A[i][j] - B[i][j];
        if (Number.isFinite(d)) { s += Math.abs(d); c++; }
      }
    }
  }
  return c ? s / c : 0;
}

export function dot(A, B) {
  return numeric.dot(A, B);
}

export function transpose(A) {
  return numeric.transpose(A);
}

export function diag(v) {
  return numeric.diag(v);
}

export function solve(A, b) {
  return numeric.solve(A, b);
}

