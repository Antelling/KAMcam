import { d } from 'typegpu';

export const RigidState = d.struct({
  theta1: d.f32,
  omega1: d.f32,
  theta2: d.f32,
  omega2: d.f32,
});

export const RigidParams = d.struct({
  m1: d.f32,
  m2: d.f32,
  L1: d.f32,
  L2: d.f32,
  dt: d.f32,
  resolution: d.f32,
  c00_t1: d.f32,
  c00_w1: d.f32,
  c00_t2: d.f32,
  c00_w2: d.f32,
  c10_t1: d.f32,
  c10_w1: d.f32,
  c10_t2: d.f32,
  c10_w2: d.f32,
  c01_t1: d.f32,
  c01_w1: d.f32,
  c01_t2: d.f32,
  c01_w2: d.f32,
  c11_t1: d.f32,
  c11_w1: d.f32,
  c11_t2: d.f32,
  c11_w2: d.f32,
});

export const DivParams = d.struct({
  m1: d.f32,
  m2: d.f32,
  L1: d.f32,
  L2: d.f32,
  dt: d.f32,
  resolution: d.f32,
  seed: d.f32,
  perturb: d.f32,
  frameCounter: d.f32,
  c00_t1: d.f32,
  c00_w1: d.f32,
  c00_t2: d.f32,
  c00_w2: d.f32,
  c10_t1: d.f32,
  c10_w1: d.f32,
  c10_t2: d.f32,
  c10_w2: d.f32,
  c01_t1: d.f32,
  c01_w1: d.f32,
  c01_t2: d.f32,
  c01_w2: d.f32,
  c11_t1: d.f32,
  c11_w1: d.f32,
  c11_t2: d.f32,
  c11_w2: d.f32,
});

export const DataCell = d.struct({
  r: d.f32,
  g: d.f32,
  b: d.f32,
  a: d.f32,
});
