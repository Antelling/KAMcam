import type { SimulationConfig } from '../config/schema';

export type RigidState = [number, number, number, number];
export type ElasticStateA = [number, number, number, number];
export type ElasticStateB = [number, number, number, number];
export type SculptureStateA = [number, number, number, number];
export type SculptureStateB = [number, number, number, number];

export function stepRigid(
  state: RigidState,
  config: SimulationConfig,
  dt: number,
): RigidState {
  const [t1, w1, t2, w2] = state;
  const { m1, m2, L1, L2 } = config;
  const a0 = rigidAccel(t1, w1, t2, w2, m1, m2, L1, L2);
  const hw1 = w1 + 0.5 * dt * a0[0];
  const hw2 = w2 + 0.5 * dt * a0[1];
  const nt1 = t1 + dt * hw1;
  const nt2 = t2 + dt * hw2;
  const a1 = rigidAccel(nt1, hw1, nt2, hw2, m1, m2, L1, L2);
  return [nt1, hw1 + 0.5 * dt * a1[0], nt2, hw2 + 0.5 * dt * a1[1]];
}

function rigidAccel(
  t1: number, w1: number, t2: number, w2: number,
  m1: number, m2: number, L1: number, L2: number,
): [number, number] {
  const delta = t1 - t2;
  const sd = Math.sin(delta);
  const cd = Math.cos(delta);
  const denom = m1 + m2 * sd * sd;
  const num1 = -m2 * L1 * w1 * w1 * sd * cd
    - m2 * L2 * w2 * w2 * sd
    - (m1 + m2) * 9.81 * Math.sin(t1)
    + m2 * 9.81 * Math.sin(t2) * cd;
  const num2 = (m1 + m2) * L1 * w1 * w1 * sd
    + m2 * L2 * w2 * w2 * sd * cd
    + (m1 + m2) * 9.81 * Math.sin(t1) * cd
    - (m1 + m2) * 9.81 * Math.sin(t2);
  return [num1 / (L1 * denom), num2 / (L2 * denom)];
}

export function stepElastic(
  sa: ElasticStateA,
  sb: ElasticStateB,
  config: SimulationConfig,
  dt: number,
): [ElasticStateA, ElasticStateB] {
  return rk4Elastic(sa, sb, config, dt, false);
}

export function stepNonlinear(
  sa: ElasticStateA,
  sb: ElasticStateB,
  config: SimulationConfig,
  dt: number,
): [ElasticStateA, ElasticStateB] {
  return rk4Elastic(sa, sb, config, dt, true);
}

function rk4Elastic(
  sa: ElasticStateA,
  sb: ElasticStateB,
  config: SimulationConfig,
  dt: number,
  nonlinear: boolean,
): [ElasticStateA, ElasticStateB] {
  const { m1, m2, L1, L2, k1, k2 } = config;
  const hdt = 0.5 * dt;
  const d1 = elasticDeriv(sa, sb, m1, m2, L1, L2, k1, k2, nonlinear);
  const sa2: ElasticStateA = [
    sa[0] + hdt * d1[0], sa[1] + hdt * d1[1], sa[2] + hdt * d1[2], sa[3] + hdt * d1[3],
  ];
  const sb2: ElasticStateB = [
    sb[0] + hdt * d1[4], sb[1] + hdt * d1[5], sb[2] + hdt * d1[6], sb[3] + hdt * d1[7],
  ];
  const d2 = elasticDeriv(sa2, sb2, m1, m2, L1, L2, k1, k2, nonlinear);
  const sa3: ElasticStateA = [
    sa[0] + hdt * d2[0], sa[1] + hdt * d2[1], sa[2] + hdt * d2[2], sa[3] + hdt * d2[3],
  ];
  const sb3: ElasticStateB = [
    sb[0] + hdt * d2[4], sb[1] + hdt * d2[5], sb[2] + hdt * d2[6], sb[3] + hdt * d2[7],
  ];
  const d3 = elasticDeriv(sa3, sb3, m1, m2, L1, L2, k1, k2, nonlinear);
  const sa4: ElasticStateA = [
    sa[0] + dt * d3[0], sa[1] + dt * d3[1], sa[2] + dt * d3[2], sa[3] + dt * d3[3],
  ];
  const sb4: ElasticStateB = [
    sb[0] + dt * d3[4], sb[1] + dt * d3[5], sb[2] + dt * d3[6], sb[3] + dt * d3[7],
  ];
  const d4 = elasticDeriv(sa4, sb4, m1, m2, L1, L2, k1, k2, nonlinear);
  const s6 = dt / 6.0;
  return [
    [
      sa[0] + s6 * (d1[0] + 2*d2[0] + 2*d3[0] + d4[0]),
      sa[1] + s6 * (d1[1] + 2*d2[1] + 2*d3[1] + d4[1]),
      sa[2] + s6 * (d1[2] + 2*d2[2] + 2*d3[2] + d4[2]),
      sa[3] + s6 * (d1[3] + 2*d2[3] + 2*d3[3] + d4[3]),
    ],
    [
      sb[0] + s6 * (d1[4] + 2*d2[4] + 2*d3[4] + d4[4]),
      sb[1] + s6 * (d1[5] + 2*d2[5] + 2*d3[5] + d4[5]),
      sb[2] + s6 * (d1[6] + 2*d2[6] + 2*d3[6] + d4[6]),
      sb[3] + s6 * (d1[7] + 2*d2[7] + 2*d3[7] + d4[7]),
    ],
  ];
}

function elasticDeriv(
  sa: ElasticStateA, sb: ElasticStateB,
  m1: number, m2: number, L1: number, L2: number,
  k1: number, k2: number, nonlinear: boolean,
): number[] {
  const [th1, om1, r1, dr1] = sa;
  const [th2, om2, r2, dr2] = sb;
  const TM = m1 + m2;
  const l1 = L1 + r1;
  const l2 = L2 + r2;
  const delta = th1 - th2;
  const sinD = Math.sin(delta);
  const cosD = Math.cos(delta);
  const l1sq = l1 * l1;
  const l2sq = l2 * l2;
  const w1sq = om1 * om1;
  const w2sq = om2 * om2;
  const M0: V4 = [TM * l1sq, 0.0, m2 * l1 * l2 * cosD, -m2 * l1 * sinD];
  const M1: V4 = [0.0, TM, m2 * l2 * sinD, m2 * cosD];
  const M2: V4 = [m2 * l1 * l2 * cosD, m2 * l2 * sinD, m2 * l2sq, 0.0];
  const M3: V4 = [-m2 * l1 * sinD, m2 * cosD, 0.0, m2];
  let fy = TM * l1 * w1sq + m2 * l2 * w2sq * cosD - 2.0 * m2 * dr2 * om2 * sinD + TM * 9.81 * Math.cos(th1);
  let fw = 2.0 * m2 * dr1 * om1 * sinD + m2 * l1 * w1sq * cosD + m2 * l2 * w2sq + m2 * 9.81 * Math.cos(th2);
  if (nonlinear) {
    fy += -Math.sign(r1) * k1 * (Math.exp(Math.abs(r1) / L1) - 1.0);
    fw += -Math.sign(r2) * k2 * (Math.exp(Math.abs(r2) / L2) - 1.0);
  } else {
    fy += -k1 * r1;
    fw += -k2 * r2;
  }
  const fx = -2.0 * TM * l1 * dr1 * om1 - 2.0 * m2 * l1 * dr2 * om2 * cosD - m2 * l1 * l2 * w2sq * sinD - TM * 9.81 * l1 * Math.sin(th1);
  const fz = -2.0 * m2 * l2 * dr1 * om1 * cosD - 2.0 * m2 * l2 * dr2 * om2 + m2 * l1 * l2 * w1sq * sinD - m2 * 9.81 * l2 * Math.sin(th2);
  const a = solveCramer4(M0, M1, M2, M3, fx, fy, fz, fw);
  return [om1, a[0], dr1, a[1], om2, a[2], dr2, a[3]];
}

type V4 = [number, number, number, number];

function solveCramer4(
  m0: V4, m1: V4, m2: V4, m3: V4,
  fx: number, fy: number, fz: number, fw: number,
): V4 {
  const d = det4(m0, m1, m2, m3);
  const inv = 1.0 / d;
  const M0: V4 = [fx, m0[1], m0[2], m0[3]] as V4;
  const M1: V4 = [m1[0], fy, m1[2], m1[3]] as V4;
  const M2: V4 = [m2[0], m2[1], fz, m2[3]] as V4;
  const M3: V4 = [m3[0], m3[1], m3[2], fw] as V4;
  return [
    det4(M0, m1, m2, m3) * inv,
    det4(m0, M1, m2, m3) * inv,
    det4(m0, m1, M2, m3) * inv,
    det4(m0, m1, m2, M3) * inv,
  ];
}

function det4(m0: V4, m1: V4, m2: V4, m3: V4): number {
  const d00 = m1[1] * (m2[2]*m3[3] - m2[3]*m3[2]) - m1[2] * (m2[1]*m3[3] - m2[3]*m3[1]) + m1[3] * (m2[1]*m3[2] - m2[2]*m3[1]);
  const d01 = m1[0] * (m2[2]*m3[3] - m2[3]*m3[2]) - m1[2] * (m2[0]*m3[3] - m2[3]*m3[0]) + m1[3] * (m2[0]*m3[2] - m2[2]*m3[0]);
  const d02 = m1[0] * (m2[1]*m3[3] - m2[3]*m3[1]) - m1[1] * (m2[0]*m3[3] - m2[3]*m3[0]) + m1[3] * (m2[0]*m3[1] - m2[1]*m3[0]);
  const d03 = m1[0] * (m2[1]*m3[2] - m2[2]*m3[1]) - m1[1] * (m2[0]*m3[2] - m2[2]*m3[0]) + m1[2] * (m2[0]*m3[1] - m2[1]*m3[0]);
  return m0[0] * d00 - m0[1] * d01 + m0[2] * d02 - m0[3] * d03;
}

export function stepSculpture(
  sa: SculptureStateA,
  sb: SculptureStateB,
  config: SimulationConfig,
  dt: number,
): [SculptureStateA, SculptureStateB] {
  const { sculptureWeight: M0, sculptureRod: L0, sculptureAxle: A0, sculptureReduction: R, sculptureN: N } = config;
  const h = 0.5 * dt;
  const d1 = sculptureDeriv(sa, sb, M0, L0, A0, R, N);
  const sa2: SculptureStateA = [sa[0]+h*d1[0], sa[1]+h*d1[1], sa[2]+h*d1[2], sa[3]+h*d1[3]];
  const sb2: SculptureStateB = [sb[0]+h*d1[4], sb[1]+h*d1[5], sb[2]+h*d1[6], sb[3]+h*d1[7]];
  const d2 = sculptureDeriv(sa2, sb2, M0, L0, A0, R, N);
  const sa3: SculptureStateA = [sa[0]+h*d2[0], sa[1]+h*d2[1], sa[2]+h*d2[2], sa[3]+h*d2[3]];
  const sb3: SculptureStateB = [sb[0]+h*d2[4], sb[1]+h*d2[5], sb[2]+h*d2[6], sb[3]+h*d2[7]];
  const d3 = sculptureDeriv(sa3, sb3, M0, L0, A0, R, N);
  const sa4: SculptureStateA = [sa[0]+dt*d3[0], sa[1]+dt*d3[1], sa[2]+dt*d3[2], sa[3]+dt*d3[3]];
  const sb4: SculptureStateB = [sb[0]+dt*d3[4], sb[1]+dt*d3[5], sb[2]+dt*d3[6], sb[3]+dt*d3[7]];
  const d4 = sculptureDeriv(sa4, sb4, M0, L0, A0, R, N);
  const s6 = dt / 6.0;
  return [
    [sa[0]+s6*(d1[0]+2*d2[0]+2*d3[0]+d4[0]), sa[1]+s6*(d1[1]+2*d2[1]+2*d3[1]+d4[1]),
     sa[2]+s6*(d1[2]+2*d2[2]+2*d3[2]+d4[2]), sa[3]+s6*(d1[3]+2*d2[3]+2*d3[3]+d4[3])],
    [sb[0]+s6*(d1[4]+2*d2[4]+2*d3[4]+d4[4]), sb[1]+s6*(d1[5]+2*d2[5]+2*d3[5]+d4[5]),
     sb[2]+s6*(d1[6]+2*d2[6]+2*d3[6]+d4[6]), sb[3]+s6*(d1[7]+2*d2[7]+2*d3[7]+d4[7])],
  ];
}

function sculptureDeriv(
  sa: SculptureStateA, sb: SculptureStateB,
  M0: number, L0: number, A0: number, R: number, N: number,
): number[] {
  const [t0, w0, t1, w1] = sa;
  const [t2, w2, t3, w3] = sb;
  const r2 = R * R;
  const r3 = r2 * R;
  const dd = L0 - A0;
  const m0v = M0;
  const m1v = N >= 2 ? M0 * R : 0;
  const m2v = N >= 3 ? M0 * r2 : 0;
  const m3v = N >= 4 ? M0 * r3 : 0;
  const a0 = A0;
  const a1 = N >= 2 ? A0 * R : 0;
  const a2 = N >= 3 ? A0 * r2 : 0;
  const a3 = N >= 4 ? A0 * r3 : 0;
  const b0 = dd;
  const b1 = N >= 2 ? dd * R : 0;
  const b2 = N >= 3 ? dd * r2 : 0;
  const b3 = N >= 4 ? dd * r3 : 0;
  const abv0 = m1v + m2v + m3v;
  const abv1 = m2v + m3v;
  const abv2 = m3v;
  const I0 = m0v * a0 * a0 + b0 * b0 * abv0;
  const I1 = N >= 2 ? (m1v * a1 * a1 + b1 * b1 * abv1) : 1;
  const I2 = N >= 3 ? (m2v * a2 * a2 + b2 * b2 * abv2) : 1;
  const I3 = N >= 4 ? (m3v * a3 * a3 + b3 * b3 * 0) : 1;
  const mu0 = m0v * a0 - b0 * abv0;
  const mu1 = N >= 2 ? (m1v * a1 - b1 * abv1) : 0;
  const mu2 = N >= 3 ? (m2v * a2 - b2 * abv2) : 0;
  const mu3 = N >= 4 ? (m3v * a3 - b3 * 0) : 0;
  const c01 = Math.cos(t0 - t1);
  const c02 = Math.cos(t0 - t2);
  const c03 = Math.cos(t0 - t3);
  const c12 = Math.cos(t1 - t2);
  const c13 = Math.cos(t1 - t3);
  const c23 = Math.cos(t2 - t3);
  const R0: V4 = [I0, -b0*mu1*c01, -b0*mu2*c02, -b0*mu3*c03];
  const R1: V4 = [-b0*mu1*c01, I1, -b1*mu2*c12, -b1*mu3*c13];
  const R2: V4 = [-b0*mu2*c02, -b1*mu2*c12, I2, -b2*mu3*c23];
  const R3: V4 = [-b0*mu3*c03, -b1*mu3*c13, -b2*mu3*c23, I3];
  const G = 9.81;
  const s01 = Math.sin(t0 - t1);
  const s02 = Math.sin(t0 - t2);
  const s03 = Math.sin(t0 - t3);
  const s12 = Math.sin(t1 - t2);
  const s13 = Math.sin(t1 - t3);
  const s23 = Math.sin(t2 - t3);
  const f0 = -G*mu0*Math.sin(t0) + b0*(mu1*s01*w1*w1 + mu2*s02*w2*w2 + mu3*s03*w3*w3);
  const f1 = -G*mu1*Math.sin(t1) - b0*mu1*s01*w0*w0 + b1*mu2*s12*w2*w2 + b1*mu3*s13*w3*w3;
  const f2 = -G*mu2*Math.sin(t2) - b0*mu2*s02*w0*w0 - b1*mu2*s12*w1*w1 + b2*mu3*s23*w3*w3;
  const f3 = -G*mu3*Math.sin(t3) - b0*mu3*s03*w0*w0 - b1*mu3*s13*w1*w1 - b2*mu3*s23*w2*w2;
  const al = solveCramer4(R0, R1, R2, R3, f0, f1, f2, f3);
  return [w0, al[0], w1, al[1], w2, al[2], w3, al[3]];
}

export function stepResonant(
  sa: SculptureStateA,
  sb: SculptureStateB,
  config: SimulationConfig,
  dt: number,
): [SculptureStateA, SculptureStateB] {
  const { rpM0, rpM1, rpL0, rpL1, rpA0 } = config;
  const hdt = 0.5 * dt;
  const s6 = dt / 6.0;
  const d1 = resonantDeriv(sa, sb, rpM0, rpM1, rpL0, rpL1, rpA0);
  const sa2: SculptureStateA = [sa[0]+hdt*d1[0], sa[1]+hdt*d1[1], sa[2], sa[3]];
  const sb2: SculptureStateB = [sb[0]+hdt*d1[2], sb[1]+hdt*d1[3], sb[2], sb[3]];
  const d2 = resonantDeriv(sa2, sb2, rpM0, rpM1, rpL0, rpL1, rpA0);
  const sa3: SculptureStateA = [sa[0]+hdt*d2[0], sa[1]+hdt*d2[1], sa[2], sa[3]];
  const sb3: SculptureStateB = [sb[0]+hdt*d2[2], sb[1]+hdt*d2[3], sb[2], sb[3]];
  const d3 = resonantDeriv(sa3, sb3, rpM0, rpM1, rpL0, rpL1, rpA0);
  const sa4: SculptureStateA = [sa[0]+dt*d3[0], sa[1]+dt*d3[1], sa[2], sa[3]];
  const sb4: SculptureStateB = [sb[0]+dt*d3[2], sb[1]+dt*d3[3], sb[2], sb[3]];
  const d4 = resonantDeriv(sa4, sb4, rpM0, rpM1, rpL0, rpL1, rpA0);
  return [
    [sa[0]+s6*(d1[0]+2*d2[0]+2*d3[0]+d4[0]), sa[1]+s6*(d1[1]+2*d2[1]+2*d3[1]+d4[1]), sa[2], sa[3]],
    [sb[0]+s6*(d1[2]+2*d2[2]+2*d3[2]+d4[2]), sb[1]+s6*(d1[3]+2*d2[3]+2*d3[3]+d4[3]), sb[2], sb[3]],
  ];
}

function resonantDeriv(
  sa: SculptureStateA, sb: SculptureStateB,
  m0: number, m1: number, L0: number, L1: number, a0: number,
): number[] {
  const th0 = sa[0], om0 = sa[1], th1 = sb[0], om1 = sb[1];
  const b0v = L0 - a0;
  const I0 = m0 * a0 * a0 + b0v * b0v * m1;
  const I1 = m1 * L1 * L1;
  const mu0 = m0 * a0 - b0v * m1;
  const mu1 = m1 * L1;
  const M01 = -b0v * mu1;
  const c01 = Math.cos(th0 - th1);
  const s01 = Math.sin(th0 - th1);
  const A00 = I0;
  const A01 = M01 * c01;
  const A11 = I1;
  const f0 = -9.81 * mu0 * Math.sin(th0) - M01 * s01 * om1 * om1;
  const f1 = -9.81 * mu1 * Math.sin(th1) + M01 * s01 * om0 * om0;
  const det = A00 * A11 - A01 * A01;
  return [om0, (f0*A11 - A01*f1)/det, om1, (A00*f1 - f0*A01)/det];
}

export function computeBob2(
  state: RigidState | ElasticStateA | ElasticStateB | SculptureStateA | SculptureStateB,
  system: string,
  config: SimulationConfig,
): { x: number; y: number } {
  if (system === 'rigid') {
    const [t1, , t2] = state as RigidState;
    const { L1, L2 } = config;
    const x1 = L1 * Math.sin(t1);
    const y1 = -L1 * Math.cos(t1);
    return { x: x1 + L2 * Math.sin(t2), y: y1 - L2 * Math.cos(t2) };
  }
  if (system === 'sculpture' || system === 'resonant') {
    return computeSculptureNodes(state as SculptureStateA | SculptureStateB, config).tip;
  }
  const sa = state as ElasticStateA;
  const sb = state as ElasticStateB;
  const al = config.L1 + sa[2];
  const bl = config.L2 + sb[2];
  const x1 = al * Math.sin(sa[0]);
  const y1 = -al * Math.cos(sa[0]);
  return { x: x1 + bl * Math.sin(sb[0]), y: y1 - bl * Math.cos(sb[0]) };
}

export function computeSculptureNodes(
  sa: SculptureStateA | SculptureStateB,
  config: SimulationConfig,
): { pivot: { x: number; y: number }; mid: { x: number; y: number }; tip: { x: number; y: number } } {
  if (config.system === 'resonant') {
    const t0 = (sa as SculptureStateA)[0];
    const t1 = (sa as SculptureStateB)[0];
    const b0 = config.rpL0 - config.rpA0;
    const ax = -b0 * Math.sin(t0);
    const ay = b0 * Math.cos(t0);
    return { pivot: { x: 0, y: 0 }, mid: { x: ax, y: ay }, tip: { x: ax + config.rpL1 * Math.sin(t1), y: ay - config.rpL1 * Math.cos(t1) } };
  }
  const { sculptureAxle: A0, sculptureRod: L0, sculptureReduction: R, sculptureN: N } = config;
  const dd = L0 - A0;
  const b0 = dd;
  const b1 = dd * R;
  const b2 = dd * R * R;
  const a0 = A0;
  const a1 = A0 * R;
  const a2 = A0 * R * R;
  const a3 = A0 * R * R * R;
  const t0 = (sa as SculptureStateA)[0];
  const t1 = (sa as SculptureStateA)[2];
  const t2 = (sa as SculptureStateB)[0];
  const t3 = (sa as SculptureStateB)[2];
  const ax = (N >= 2 ? -b0 * Math.sin(t0) : 0) + (N >= 3 ? -b1 * Math.sin(t1) : 0) + (N >= 4 ? -b2 * Math.sin(t2) : 0);
  const ay = (N >= 2 ? b0 * Math.cos(t0) : 0) + (N >= 3 ? b1 * Math.cos(t1) : 0) + (N >= 4 ? b2 * Math.cos(t2) : 0);
  const la = N >= 4 ? a3 : N >= 3 ? a2 : N >= 2 ? a1 : a0;
  const lt = N >= 4 ? t3 : N >= 3 ? t2 : N >= 2 ? t1 : t0;
  const mx = (N >= 2 ? -b0 * Math.sin(t0) : 0);
  const my = (N >= 2 ? b0 * Math.cos(t0) : 0);
  return { pivot: { x: 0, y: 0 }, mid: { x: mx, y: my }, tip: { x: ax + la * Math.sin(lt), y: ay - la * Math.cos(lt) } };
}

export function checkDivergence(
  baseState: RigidState | ElasticStateA | SculptureStateA,
  pertState: RigidState | ElasticStateA | SculptureStateA,
  system: string,
  config: SimulationConfig,
): boolean {
  if (system === 'sculpture') {
    return sculptureDivergenceCheck(baseState as SculptureStateA, pertState as SculptureStateA, config.sculptureN);
  }
  if (system === 'resonant') {
    return resonantDivergenceCheck(baseState as SculptureStateA, pertState as SculptureStateA);
  }
  const TWO_PI = Math.PI * 2;
  const circDiff = (a: number) => a - Math.floor(a / TWO_PI + 0.5) * TWO_PI;
  if (system === 'rigid') {
    const b = baseState as RigidState;
    const p = pertState as RigidState;
    const d0 = circDiff(b[0] - p[0]);
    const dw0 = b[1] - p[1];
    const d1 = circDiff(b[2] - p[2]);
    const dw1 = b[3] - p[3];
    return Math.sqrt(d0*d0 + dw0*dw0 + d1*d1 + dw1*dw1) > 0.05;
  }
  const b = baseState as ElasticStateA;
  const p = pertState as ElasticStateA;
  const dt1 = circDiff(b[0] - p[0]);
  const dw1 = b[1] - p[1];
  const ds1 = b[2] - p[2];
  const dd1 = b[3] - p[3];
  return Math.sqrt(dt1*dt1 + dw1*dw1 + ds1*ds1 + dd1*dd1) > 0.05;
}

function sculptureDivergenceCheck(bsa: SculptureStateA, psa: SculptureStateA, N: number): boolean {
  const TWO_PI = Math.PI * 2;
  const TOL_LO = 0.05;
  const TOL_HI = 1.0;
  const OMEGA_SCALE = 2.0;
  const ratio = TOL_HI / TOL_LO;
  const invN = N <= 1 ? 0 : 1 / (N - 1);
  const circDiff = (a: number) => a - Math.floor(a / TWO_PI + 0.5) * TWO_PI;
  let sum = 0;
  const indices = [0, 2, 4, 6];
  for (let i = 0; i < 4; i++) {
    if (N >= i + 1) {
      const g = i * invN;
      const tolA = TOL_LO * Math.exp(g * Math.log(ratio));
      const tolW = tolA * OMEGA_SCALE;
      const da = circDiff(bsa[indices[i]] - psa[indices[i]]);
      const dw = bsa[indices[i] + 1] - psa[indices[i] + 1];
      sum += (da * da) / (tolA * tolA) + (dw * dw) / (tolW * tolW);
    }
  }
  return sum > 1.0;
}

function resonantDivergenceCheck(bsa: SculptureStateA, psa: SculptureStateA): boolean {
  const TWO_PI = Math.PI * 2;
  const circDiff = (a: number) => a - Math.floor(a / TWO_PI + 0.5) * TWO_PI;
  const da0 = circDiff(bsa[0] - psa[0]);
  const dw0 = bsa[1] - psa[1];
  const da1 = circDiff(bsa[2] - psa[2]);
  const dw1 = bsa[3] - psa[3];
  const TOL_A0 = 0.05, TOL_W0 = 0.1, TOL_A1 = 1.0, TOL_W1 = 2.0;
  return (da0*da0)/(TOL_A0*TOL_A0) + (dw0*dw0)/(TOL_W0*TOL_W0) + (da1*da1)/(TOL_A1*TOL_A1) + (dw1*dw1)/(TOL_W1*TOL_W1) > 1.0;
}

export function calculateEnergies(
  state: RigidState | ElasticStateA | SculptureStateA,
  system: string,
  config: SimulationConfig,
): { ke: number; pe: number; ee: number } {
  const G = 9.81;
  if (system === 'rigid') {
    const [t1, w1, t2, w2] = state as RigidState;
    const { m1, m2, L1, L2 } = config;
    const ke = 0.5 * m1 * L1 * L1 * w1 * w1 + 0.5 * m2 * (L1*L1*w1*w1 + L2*L2*w2*w2 + 2*L1*L2*w1*w2*Math.cos(t1-t2));
    const pe = -(m1+m2) * G * L1 * Math.cos(t1) - m2 * G * L2 * Math.cos(t2);
    return { ke, pe, ee: ke + pe };
  }
  if (system === 'elastic' || system === 'nonlinear') {
    const sa = state as ElasticStateA;
    const l1 = config.L1 + sa[2];
    const ke = 0.5 * config.m1 * l1 * l1 * sa[1] * sa[1];
    const ee1 = system === 'nonlinear'
      ? config.k1 * config.L1 * (Math.exp(Math.abs(sa[2]) / config.L1) - 1 - Math.abs(sa[2]) / config.L1)
      : 0.5 * config.k1 * sa[2] * sa[2];
    const peVal = -config.m1 * G * l1 * Math.cos(sa[0]);
    return { ke, pe: peVal, ee: ke + peVal + ee1 };
  }
  return { ke: 0, pe: 0, ee: 0 };
}
