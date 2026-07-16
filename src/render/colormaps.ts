import { d, std } from 'typegpu';

const mix3 = (a: number, b: number, t: number) => {
  'use gpu';
  return a + (b - a) * t;
};

const viridis = (t: number) => {
  'use gpu';
  const c0r = 68.0 / 255.0;
  const c0g = 1.0 / 255.0;
  const c0b = 84.0 / 255.0;
  const c1r = 33.0 / 255.0;
  const c1g = 145.0 / 255.0;
  const c1b = 140.0 / 255.0;
  const c2r = 253.0 / 255.0;
  const c2g = 231.0 / 255.0;
  const c2b = 37.0 / 255.0;
  if (t < 0.5) {
    const s = t * 2.0;
    return d.vec3f(mix3(c0r, c1r, s), mix3(c0g, c1g, s), mix3(c0b, c1b, s));
  }
  const s = (t - 0.5) * 2.0;
  return d.vec3f(mix3(c1r, c2r, s), mix3(c1g, c2g, s), mix3(c1b, c2b, s));
};

const magma = (t: number) => {
  'use gpu';
  const c0r = 4.0 / 255.0;
  const c0g = 5.0 / 255.0;
  const c0b = 9.0 / 255.0;
  const c1r = 148.0 / 255.0;
  const c1g = 52.0 / 255.0;
  const c1b = 110.0 / 255.0;
  const c2r = 252.0 / 255.0;
  const c2g = 253.0 / 255.0;
  const c2b = 191.0 / 255.0;
  if (t < 0.5) {
    const s = t * 2.0;
    return d.vec3f(mix3(c0r, c1r, s), mix3(c0g, c1g, s), mix3(c0b, c1b, s));
  }
  const s = (t - 0.5) * 2.0;
  return d.vec3f(mix3(c1r, c2r, s), mix3(c1g, c2g, s), mix3(c1b, c2b, s));
};

const plasma = (t: number) => {
  'use gpu';
  const c0r = 13.0 / 255.0;
  const c0g = 8.0 / 255.0;
  const c0b = 135.0 / 255.0;
  const c1r = 156.0 / 255.0;
  const c1g = 23.0 / 255.0;
  const c1b = 158.0 / 255.0;
  const c2r = 240.0 / 255.0;
  const c2g = 249.0 / 255.0;
  const c2b = 33.0 / 255.0;
  if (t < 0.5) {
    const s = t * 2.0;
    return d.vec3f(mix3(c0r, c1r, s), mix3(c0g, c1g, s), mix3(c0b, c1b, s));
  }
  const s = (t - 0.5) * 2.0;
  return d.vec3f(mix3(c1r, c2r, s), mix3(c1g, c2g, s), mix3(c1b, c2b, s));
};

const inferno = (t: number) => {
  'use gpu';
  const c0r = 0.0 / 255.0;
  const c0g = 0.0 / 255.0;
  const c0b = 4.0 / 255.0;
  const c1r = 187.0 / 255.0;
  const c1g = 55.0 / 255.0;
  const c1b = 84.0 / 255.0;
  const c2r = 252.0 / 255.0;
  const c2g = 255.0 / 255.0;
  const c2b = 164.0 / 255.0;
  if (t < 0.5) {
    const s = t * 2.0;
    return d.vec3f(mix3(c0r, c1r, s), mix3(c0g, c1g, s), mix3(c0b, c1b, s));
  }
  const s = (t - 0.5) * 2.0;
  return d.vec3f(mix3(c1r, c2r, s), mix3(c1g, c2g, s), mix3(c1b, c2b, s));
};

const turbo = (t: number) => {
  'use gpu';
  const r = std.clamp((48.0 + 227.0 * std.sin((t - 0.5) * 3.14159265)) / 255.0, 0.0, 1.0);
  let gv: number;
  if (t < 0.5) {
    gv = t * 400.0;
  } else {
    gv = (1.0 - t) * 400.0;
  }
  const g = std.clamp(gv / 255.0, 0.0, 1.0);
  const b = std.clamp((128.0 + 127.0 * std.cos(t * 3.14159265)) / 255.0, 0.0, 1.0);
  return d.vec3f(r, g, b);
};

const jet = (t: number) => {
  'use gpu';
  let rv: number; let gv: number; let bv: number;
  if (t < 0.5) {
    rv = 0.0;
  } else {
    rv = (t - 0.5) * 2.0;
  }
  if (t < 0.25) {
    gv = t * 4.0;
  } else if (t < 0.75) {
    gv = 1.0;
  } else {
    gv = (1.0 - t) * 4.0;
  }
  if (t < 0.5) {
    bv = (0.5 - t) * 2.0;
  } else {
    bv = 0.0;
  }
  return d.vec3f(std.clamp(rv, 0.0, 1.0), std.clamp(gv, 0.0, 1.0), std.clamp(bv, 0.0, 1.0));
};

const rainbow = (t: number) => {
  'use gpu';
  const hue = (1.0 - t) * 0.85;
  const c = hue * 6.0;
  const x = c - std.floor(c);
  if (hue < 1.0 / 6.0) return d.vec3f(x, 1, 0);
  if (hue < 2.0 / 6.0) return d.vec3f(1 - x, 1, 0);
  if (hue < 3.0 / 6.0) return d.vec3f(0, 1, x);
  if (hue < 4.0 / 6.0) return d.vec3f(0, 1 - x, 1);
  if (hue < 5.0 / 6.0) return d.vec3f(x, 0, 1);
  return d.vec3f(1, 0, 1 - x);
};

export const applyColormap = (t: number, mode: number) => {
  'use gpu';
  if (mode < 0.5) return viridis(t);
  if (mode < 1.5) return magma(t);
  if (mode < 2.5) return plasma(t);
  if (mode < 3.5) return inferno(t);
  if (mode < 4.5) return turbo(t);
  if (mode < 5.5) return jet(t);
  return rainbow(t);
};
