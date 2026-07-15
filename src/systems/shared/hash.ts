import { d, std } from 'typegpu';

export const hash = (px: number, py: number) => {
  'use gpu';
  const dot = px * 127.1 + py * 311.7;
  return std.fract(std.sin(dot) * 43758.5453);
};
