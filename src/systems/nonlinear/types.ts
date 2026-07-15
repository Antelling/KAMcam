import { d } from 'typegpu';

export const NonlinearStateA = d.struct({
  theta1: d.f32,
  omega1: d.f32,
  stretch1: d.f32,
  stretchRate1: d.f32,
});

export const NonlinearStateB = d.struct({
  theta2: d.f32,
  omega2: d.f32,
  stretch2: d.f32,
  stretchRate2: d.f32,
});

export const NonlinearParams = d.struct({
  m1: d.f32,
  m2: d.f32,
  L1: d.f32,
  L2: d.f32,
  k1: d.f32,
  k2: d.f32,
  dt: d.f32,
  resolution: d.f32,
  cA00_th: d.f32,
  cA00_om: d.f32,
  cA00_r: d.f32,
  cA00_dr: d.f32,
  cA10_th: d.f32,
  cA10_om: d.f32,
  cA10_r: d.f32,
  cA10_dr: d.f32,
  cA01_th: d.f32,
  cA01_om: d.f32,
  cA01_r: d.f32,
  cA01_dr: d.f32,
  cA11_th: d.f32,
  cA11_om: d.f32,
  cA11_r: d.f32,
  cA11_dr: d.f32,
  cB00_th: d.f32,
  cB00_om: d.f32,
  cB00_r: d.f32,
  cB00_dr: d.f32,
  cB10_th: d.f32,
  cB10_om: d.f32,
  cB10_r: d.f32,
  cB10_dr: d.f32,
  cB01_th: d.f32,
  cB01_om: d.f32,
  cB01_r: d.f32,
  cB01_dr: d.f32,
  cB11_th: d.f32,
  cB11_om: d.f32,
  cB11_r: d.f32,
  cB11_dr: d.f32,
});

export const DataCell = d.struct({
  r: d.f32,
  g: d.f32,
  b: d.f32,
  a: d.f32,
});
