import { d } from 'typegpu';

export const ResonantStateA = d.struct({
  theta0: d.f32,
  omega0: d.f32,
  dummy1: d.f32,
  dummy2: d.f32,
});

export const ResonantStateB = d.struct({
  theta1: d.f32,
  omega1: d.f32,
  dummy1: d.f32,
  dummy2: d.f32,
});

export const ResonantParams = d.struct({
  rpM0: d.f32,
  rpM1: d.f32,
  rpL0: d.f32,
  rpL1: d.f32,
  rpA0: d.f32,
  dt: d.f32,
  resolution: d.f32,
  cA00_th: d.f32,
  cA00_om: d.f32,
  cA00_d1: d.f32,
  cA00_d2: d.f32,
  cA10_th: d.f32,
  cA10_om: d.f32,
  cA10_d1: d.f32,
  cA10_d2: d.f32,
  cA01_th: d.f32,
  cA01_om: d.f32,
  cA01_d1: d.f32,
  cA01_d2: d.f32,
  cA11_th: d.f32,
  cA11_om: d.f32,
  cA11_d1: d.f32,
  cA11_d2: d.f32,
  cB00_th: d.f32,
  cB00_om: d.f32,
  cB00_d1: d.f32,
  cB00_d2: d.f32,
  cB10_th: d.f32,
  cB10_om: d.f32,
  cB10_d1: d.f32,
  cB10_d2: d.f32,
  cB01_th: d.f32,
  cB01_om: d.f32,
  cB01_d1: d.f32,
  cB01_d2: d.f32,
  cB11_th: d.f32,
  cB11_om: d.f32,
  cB11_d1: d.f32,
  cB11_d2: d.f32,
});

export const DataCell = d.struct({
  r: d.f32,
  g: d.f32,
  b: d.f32,
  a: d.f32,
});
