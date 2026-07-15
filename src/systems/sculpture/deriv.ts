import { std } from 'typegpu';

type V4 = { x: number; y: number; z: number; w: number };

export const det4 = (m0: V4, m1: V4, m2: V4, m3: V4) => {
  'use gpu';
  const d00 = m1.y * (m2.z * m3.w - m2.w * m3.z) - m1.z * (m2.y * m3.w - m2.w * m3.y) + m1.w * (m2.y * m3.z - m2.z * m3.y);
  const d01 = m1.x * (m2.z * m3.w - m2.w * m3.z) - m1.z * (m2.x * m3.w - m2.w * m3.x) + m1.w * (m2.x * m3.z - m2.z * m3.x);
  const d02 = m1.x * (m2.y * m3.w - m2.w * m3.y) - m1.y * (m2.x * m3.w - m2.w * m3.x) + m1.w * (m2.x * m3.y - m2.y * m3.x);
  const d03 = m1.x * (m2.y * m3.z - m2.z * m3.y) - m1.y * (m2.x * m3.z - m2.z * m3.x) + m1.z * (m2.x * m3.y - m2.y * m3.x);
  return m0.x * d00 - m0.y * d01 + m0.z * d02 - m0.w * d03;
};

export const scSolve4 = (m0: V4, m1: V4, m2: V4, m3: V4, f0: number, f1: number, f2: number, f3: number) => {
  'use gpu';
  const d = det4(m0, m1, m2, m3);
  const inv = 1.0 / d;
  const M0 = { x: f0, y: m0.y, z: m0.z, w: m0.w };
  const M1 = { x: m1.x, y: f1, z: m1.z, w: m1.w };
  const M2 = { x: m2.x, y: m2.y, z: f2, w: m2.w };
  const M3 = { x: m3.x, y: m3.y, z: m3.z, w: f3 };
  return {
    x: det4(M0, m1, m2, m3) * inv,
    y: det4(m0, M1, m2, m3) * inv,
    z: det4(m0, m1, M2, m3) * inv,
    w: det4(m0, m1, m2, M3) * inv,
  };
};

export const systemDeriv = (
  t0: number, w0: number, t1: number, w1: number,
  t2: number, w2: number, t3: number, w3: number,
  M0: number, L0: number, A0: number, R: number, N: number,
) => {
  'use gpu';
  const r2 = R * R;
  const r3 = r2 * R;
  const dd = L0 - A0;
  const m0 = M0;
  const m1 = N >= 2.0 ? M0 * R : 0.0;
  const m2 = N >= 3.0 ? M0 * r2 : 0.0;
  const m3 = N >= 4.0 ? M0 * r3 : 0.0;
  const a0 = A0;
  const a1 = N >= 2.0 ? A0 * R : 0.0;
  const a2 = N >= 3.0 ? A0 * r2 : 0.0;
  const a3 = N >= 4.0 ? A0 * r3 : 0.0;
  const b0 = dd;
  const b1 = N >= 2.0 ? dd * R : 0.0;
  const b2 = N >= 3.0 ? dd * r2 : 0.0;
  const b3 = N >= 4.0 ? dd * r3 : 0.0;
  const abv0 = m1 + m2 + m3;
  const abv1 = m2 + m3;
  const abv2 = m3;
  const I0 = m0 * a0 * a0 + b0 * b0 * abv0;
  const I1 = N >= 2.0 ? (m1 * a1 * a1 + b1 * b1 * abv1) : 1.0;
  const I2 = N >= 3.0 ? (m2 * a2 * a2 + b2 * b2 * abv2) : 1.0;
  const I3 = N >= 4.0 ? (m3 * a3 * a3 + b3 * b3 * 0.0) : 1.0;
  const mu0 = m0 * a0 - b0 * abv0;
  const mu1 = N >= 2.0 ? (m1 * a1 - b1 * abv1) : 0.0;
  const mu2 = N >= 3.0 ? (m2 * a2 - b2 * abv2) : 0.0;
  const mu3 = N >= 4.0 ? (m3 * a3 - b3 * 0.0) : 0.0;

  const c01 = std.cos(t0 - t1);
  const c02 = std.cos(t0 - t2);
  const c03 = std.cos(t0 - t3);
  const c12 = std.cos(t1 - t2);
  const c13 = std.cos(t1 - t3);
  const c23 = std.cos(t2 - t3);
  const R0 = { x: I0, y: -b0 * mu1 * c01, z: -b0 * mu2 * c02, w: -b0 * mu3 * c03 };
  const R1 = { x: -b0 * mu1 * c01, y: I1, z: -b1 * mu2 * c12, w: -b1 * mu3 * c13 };
  const R2 = { x: -b0 * mu2 * c02, y: -b1 * mu2 * c12, z: I2, w: -b2 * mu3 * c23 };
  const R3 = { x: -b0 * mu3 * c03, y: -b1 * mu3 * c13, z: -b2 * mu3 * c23, w: I3 };

  const G = 9.81;
  const w0s = w0 * w0;
  const w1s = w1 * w1;
  const w2s = w2 * w2;
  const w3s = w3 * w3;
  const s01 = std.sin(t0 - t1);
  const s02 = std.sin(t0 - t2);
  const s03 = std.sin(t0 - t3);
  const s12 = std.sin(t1 - t2);
  const s13 = std.sin(t1 - t3);
  const s23 = std.sin(t2 - t3);
  const f0 = -G * mu0 * std.sin(t0) + b0 * (mu1 * s01 * w1s + mu2 * s02 * w2s + mu3 * s03 * w3s);
  const f1 = -G * mu1 * std.sin(t1) - b0 * mu1 * s01 * w0s + b1 * mu2 * s12 * w2s + b1 * mu3 * s13 * w3s;
  const f2 = -G * mu2 * std.sin(t2) - b0 * mu2 * s02 * w0s - b1 * mu2 * s12 * w1s + b2 * mu3 * s23 * w3s;
  const f3 = -G * mu3 * std.sin(t3) - b0 * mu3 * s03 * w0s - b1 * mu3 * s13 * w1s - b2 * mu3 * s23 * w2s;

  const al = scSolve4(R0, R1, R2, R3, f0, f1, f2, f3);
  return {
    dt0: w0, dw0: al.x,
    dt1: w1, dw1: al.y,
    dt2: w2, dw2: al.z,
    dt3: w3, dw3: al.w,
  };
};

export const computeSculptureTip = (
  t0: number, t1: number, t2: number, t3: number,
  A0: number, L0: number, R: number, N: number,
) => {
  'use gpu';
  const dd = L0 - A0;
  const b0 = dd;
  const b1 = dd * R;
  const b2 = dd * R * R;
  const a0 = A0;
  const a1 = A0 * R;
  const a2 = A0 * R * R;
  const a3 = A0 * R * R * R;
  const ax = (N >= 2.0 ? -b0 * std.sin(t0) : 0.0)
    + (N >= 3.0 ? -b1 * std.sin(t1) : 0.0)
    + (N >= 4.0 ? -b2 * std.sin(t2) : 0.0);
  const ay = (N >= 2.0 ? b0 * std.cos(t0) : 0.0)
    + (N >= 3.0 ? b1 * std.cos(t1) : 0.0)
    + (N >= 4.0 ? b2 * std.cos(t2) : 0.0);
  const la = N >= 4.0 ? a3 : N >= 3.0 ? a2 : N >= 2.0 ? a1 : a0;
  const lt = N >= 4.0 ? t3 : N >= 3.0 ? t2 : N >= 2.0 ? t1 : t0;
  return { x: ax + la * std.sin(lt), y: ay - la * std.cos(lt) };
};
