import { d, std } from 'typegpu';

export const applyTonemap = (value: number, maxValue: number, mode: number) => {
  'use gpu';
  if (mode < 0.5) {
    return value / maxValue;
  }
  if (mode < 1.5) {
    return std.log(1.0 + value) / std.log(1.0 + maxValue);
  }
  if (mode < 2.5) {
    return std.sqrt(value / maxValue);
  }
  const x = value / maxValue;
  return x * x * (3.0 - 2.0 * x);
};
