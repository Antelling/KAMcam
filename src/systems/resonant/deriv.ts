import { d, std } from 'typegpu';
import { hash } from '../shared/hash';

export const systemDeriv = (
  th0: number, om0: number, th1: number, om1: number,
  m0: number, m1: number, L0: number, L1: number, a0: number,
) => {
  'use gpu';
  const b0 = L0 - a0;
  const I0 = m0 * a0 * a0 + b0 * b0 * m1;
  const I1 = m1 * L1 * L1;
  const mu0 = m0 * a0 - b0 * m1;
  const mu1 = m1 * L1;
  const M01 = -b0 * mu1;
  const c01 = std.cos(th0 - th1);
  const s01 = std.sin(th0 - th1);
  const A00 = I0;
  const A01 = M01 * c01;
  const A11 = I1;
  const f0 = -9.81 * mu0 * std.sin(th0) - M01 * s01 * om1 * om1;
  const f1 = -9.81 * mu1 * std.sin(th1) + M01 * s01 * om0 * om0;
  const det = A00 * A11 - A01 * A01;
  return {
    da_th: om0,
    da_om: (f0 * A11 - A01 * f1) / det,
    db_th: om1,
    db_om: (A00 * f1 - f0 * A01) / det,
  };
};

export const resonantDivergence = (
  bt0: number, bw0: number,
  bt1: number, bw1: number,
  pt0: number, pw0: number,
  pt1: number, pw1: number,
) => {
  'use gpu';
  const TWO_PI = 2.0 * std.acos(0);
  let da0 = bt0 - pt0;
  da0 = da0 - std.floor(da0 / TWO_PI + 0.5) * TWO_PI;
  let da1 = bt1 - pt1;
  da1 = da1 - std.floor(da1 / TWO_PI + 0.5) * TWO_PI;
  const dw0 = bw0 - pw0;
  const dw1 = bw1 - pw1;
  const TOL_A0 = 0.05;
  const TOL_W0 = 0.1;
  const TOL_A1 = 1.0;
  const TOL_W1 = 2.0;
  return (da0 * da0) / (TOL_A0 * TOL_A0) + (dw0 * dw0) / (TOL_W0 * TOL_W0)
    + (da1 * da1) / (TOL_A1 * TOL_A1) + (dw1 * dw1) / (TOL_W1 * TOL_W1);
};

export const computeResonantTip = (
  t0: number, t1: number, L0: number, L1: number, a0: number,
) => {
  'use gpu';
  const b0 = L0 - a0;
  const ax = -b0 * std.sin(t0);
  const ay = b0 * std.cos(t0);
  return d.vec2f(ax + L1 * std.sin(t1), ay - L1 * std.cos(t1));
};