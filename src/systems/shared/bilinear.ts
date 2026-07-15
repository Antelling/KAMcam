export const bilinear = (c00: number, c10: number, c01: number, c11: number, u: number, v: number) => {
  'use gpu';
  return (1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11;
};
