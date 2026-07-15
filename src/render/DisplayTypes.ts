import { d } from 'typegpu';

export const DisplayUniforms = d.struct({
  maxValue: d.f32,
  colormap: d.f32,
  toneMapping: d.f32,
  vizMode: d.f32,
  resolution: d.f32,
  _pad0: d.f32,
  _pad1: d.f32,
  _pad2: d.f32,
});
