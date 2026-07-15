import { std } from 'typegpu';

export const AccelResult = (ax: number, ay: number) => {
  'use gpu';
  return { x: ax, y: ay };
};

export const computeAccelerations = (
  theta1: number, omega1: number, theta2: number, omega2: number,
  m1: number, m2: number, L1: number, L2: number,
) => {
  'use gpu';
  const delta = theta1 - theta2;
  const sinDelta = std.sin(delta);
  const cosDelta = std.cos(delta);
  const denom = m1 + m2 * sinDelta * sinDelta;
  const num1 = -m2 * L1 * omega1 * omega1 * sinDelta * cosDelta
    - m2 * L2 * omega2 * omega2 * sinDelta
    - (m1 + m2) * 9.81 * std.sin(theta1)
    + m2 * 9.81 * std.sin(theta2) * cosDelta;
  const num2 = (m1 + m2) * L1 * omega1 * omega1 * sinDelta
    + m2 * L2 * omega2 * omega2 * sinDelta * cosDelta
    + (m1 + m2) * 9.81 * std.sin(theta1) * cosDelta
    - (m1 + m2) * 9.81 * std.sin(theta2);
  return AccelResult(num1 / (L1 * denom), num2 / (L2 * denom));
};

export const Bob2Result = (bx: number, by: number) => {
  'use gpu';
  return { x: bx, y: by };
};

export const computeBob2 = (theta1: number, theta2: number, l1: number, l2: number) => {
  'use gpu';
  const x1 = l1 * std.sin(theta1);
  const y1 = -l1 * std.cos(theta1);
  return Bob2Result(x1 + l2 * std.sin(theta2), y1 - l2 * std.cos(theta2));
};
