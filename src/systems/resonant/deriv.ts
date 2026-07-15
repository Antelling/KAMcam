import { std } from 'typegpu';

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

export const computeResonantTip = (
  t0: number, t1: number, L0: number, L1: number, a0: number,
) => {
  'use gpu';
  const b0 = L0 - a0;
  const ax = -b0 * std.sin(t0);
  const ay = b0 * std.cos(t0);
  return { x: ax + L1 * std.sin(t1), y: ay - L1 * std.cos(t1) };
};
