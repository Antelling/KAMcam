import { d, std } from 'typegpu';

export const det4 = (m0: d.v4f, m1: d.v4f, m2: d.v4f, m3: d.v4f) => {
  'use gpu';
  const d00 = m1.y * (m2.z * m3.w - m2.w * m3.z) - m1.z * (m2.y * m3.w - m2.w * m3.y) + m1.w * (m2.y * m3.z - m2.z * m3.y);
  const d01 = m1.x * (m2.z * m3.w - m2.w * m3.z) - m1.z * (m2.x * m3.w - m2.w * m3.x) + m1.w * (m2.x * m3.z - m2.z * m3.x);
  const d02 = m1.x * (m2.y * m3.w - m2.w * m3.y) - m1.y * (m2.x * m3.w - m2.w * m3.x) + m1.w * (m2.x * m3.y - m2.y * m3.x);
  const d03 = m1.x * (m2.y * m3.z - m2.z * m3.y) - m1.y * (m2.x * m3.z - m2.z * m3.x) + m1.z * (m2.x * m3.y - m2.y * m3.x);
  return m0.x * d00 - m0.y * d01 + m0.z * d02 - m0.w * d03;
};

export const solveCramer4 = (m0: d.v4f, m1: d.v4f, m2: d.v4f, m3: d.v4f, fx: number, fy: number, fz: number, fw: number) => {
  'use gpu';
  const detVal = det4(m0, m1, m2, m3);
  const inv = 1.0 / detVal;
  const M0 = d.vec4f(fx, m0.y, m0.z, m0.w);
  const M1 = d.vec4f(m1.x, fy, m1.z, m1.w);
  const M2 = d.vec4f(m2.x, m2.y, fz, m2.w);
  const M3 = d.vec4f(m3.x, m3.y, m3.z, fw);
  return d.vec4f(det4(M0, m1, m2, m3) * inv, det4(m0, M1, m2, m3) * inv, det4(m0, m1, M2, m3) * inv, det4(m0, m1, m2, M3) * inv);
};

export const systemDeriv = (
  sa_th: number, sa_om: number, sa_r: number, sa_dr: number,
  sb_th: number, sb_om: number, sb_r: number, sb_dr: number,
  m1: number, m2: number, L1: number, L2: number, k1: number, k2: number,
) => {
  'use gpu';
  const TM = m1 + m2;
  const l1 = L1 + sa_r;
  const l2 = L2 + sb_r;
  const delta = sa_th - sb_th;
  const sinD = std.sin(delta);
  const cosD = std.cos(delta);
  const l1sq = l1 * l1;
  const l2sq = l2 * l2;
  const w1sq = sa_om * sa_om;
  const w2sq = sb_om * sb_om;
  const M0 = d.vec4f(TM * l1sq, 0.0, m2 * l1 * l2 * cosD, -m2 * l1 * sinD);
  const M1 = d.vec4f(0.0, TM, m2 * l2 * sinD, m2 * cosD);
  const M2 = d.vec4f(m2 * l1 * l2 * cosD, m2 * l2 * sinD, m2 * l2sq, 0.0);
  const M3 = d.vec4f(-m2 * l1 * sinD, m2 * cosD, 0.0, m2);
  const fx = -2.0 * TM * l1 * sa_dr * sa_om - 2.0 * m2 * l1 * sb_dr * sb_om * cosD - m2 * l1 * l2 * w2sq * sinD - TM * 9.81 * l1 * std.sin(sa_th);
  const fy = TM * l1 * w1sq + m2 * l2 * w2sq * cosD - 2.0 * m2 * sb_dr * sb_om * sinD + TM * 9.81 * std.cos(sa_th) - k1 * sa_r;
  const fz = -2.0 * m2 * l2 * sa_dr * sa_om * cosD - 2.0 * m2 * l2 * sb_dr * sb_om + m2 * l1 * l2 * w1sq * sinD - m2 * 9.81 * l2 * std.sin(sb_th);
  const fw = 2.0 * m2 * sa_dr * sa_om * sinD + m2 * l1 * w1sq * cosD + m2 * l2 * w2sq + m2 * 9.81 * std.cos(sb_th) - k2 * sb_r;
  const a = solveCramer4(M0, M1, M2, M3, fx, fy, fz, fw);
  return {
    da_th: sa_om, da_om: a.x, da_r: sa_dr, da_dr: a.y,
    db_th: sb_om, db_om: a.z, db_r: sb_dr, db_dr: a.w,
  };
};

export const computeBob2 = (th1: number, th2: number, l1: number, l2: number, r1: number, r2: number) => {
  'use gpu';
  const al = l1 + r1;
  const bl = l2 + r2;
  const x1 = al * std.sin(th1);
  const y1 = -al * std.cos(th1);
  return d.vec2f(x1 + bl * std.sin(th2), y1 - bl * std.cos(th2));
};