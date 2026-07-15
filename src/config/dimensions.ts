import type { SystemType, PhaseSpaceDimension } from './schema';

export const DIM_ORDER: PhaseSpaceDimension[] = [
  'angle1', 'velocity1', 'stretch1', 'stretchRate1',
  'angle2', 'velocity2', 'stretch2', 'stretchRate2',
];

export const RIGID_DIMENSIONS: PhaseSpaceDimension[] = [
  'angle1', 'velocity1', 'angle2', 'velocity2',
];

export const ELASTIC_DIMENSIONS: PhaseSpaceDimension[] = [
  'angle1', 'velocity1', 'stretch1', 'stretchRate1',
  'angle2', 'velocity2', 'stretch2', 'stretchRate2',
];

export const SCULPTURE_DIMENSIONS: PhaseSpaceDimension[] = [
  'angle1', 'velocity1', 'angle2', 'velocity2', 'stretch1', 'stretchRate1',
];

export const RESONANT_DIMENSIONS: PhaseSpaceDimension[] = [
  'angle1', 'velocity1', 'angle2', 'velocity2',
];

export const DIMENSIONS_BY_SYSTEM: Record<SystemType, PhaseSpaceDimension[]> = {
  rigid: RIGID_DIMENSIONS,
  elastic: ELASTIC_DIMENSIONS,
  nonlinear: ELASTIC_DIMENSIONS,
  sculpture: SCULPTURE_DIMENSIONS,
  resonant: RESONANT_DIMENSIONS,
};

export function sculptureDimensions(n: number): PhaseSpaceDimension[] {
  const count = 2 * Math.max(1, Math.min(4, Math.floor(n)));
  return DIM_ORDER.slice(0, count);
}

export function systemDimensions(system: SystemType, sculptureN: number): PhaseSpaceDimension[] {
  if (system === 'sculpture') return sculptureDimensions(sculptureN);
  return DIMENSIONS_BY_SYSTEM[system];
}

export const DIMENSION_DEFAULTS: Record<PhaseSpaceDimension, { min: number; max: number; initial: number }> = {
  angle1: { min: -Math.PI, max: Math.PI, initial: 0 },
  velocity1: { min: -5, max: 5, initial: 0 },
  angle2: { min: -Math.PI, max: Math.PI, initial: 0 },
  velocity2: { min: -5, max: 5, initial: 0 },
  stretch1: { min: -0.5, max: 0.5, initial: 0 },
  stretchRate1: { min: -5, max: 5, initial: 0 },
  stretch2: { min: -0.5, max: 0.5, initial: 0 },
  stretchRate2: { min: -5, max: 5, initial: 0 },
};

export const DIM_SCALE: Record<PhaseSpaceDimension, number> = {
  angle1: Math.PI,
  velocity1: 5,
  angle2: Math.PI,
  velocity2: 5,
  stretch1: 0.5,
  stretchRate1: 5,
  stretch2: 0.5,
  stretchRate2: 5,
};

export function basisVector(dim: PhaseSpaceDimension): number[] {
  const v = new Array(DIM_ORDER.length).fill(0);
  v[DIM_ORDER.indexOf(dim)] = 1;
  return v;
}

export function initialVector(config: { phaseSpace: { initialValues: Record<PhaseSpaceDimension, number> } }): number[] {
  const iv = config.phaseSpace.initialValues;
  return DIM_ORDER.map(d => iv[d]);
}

export function isAngleDim(system: SystemType, dim: PhaseSpaceDimension): boolean {
  if (system === 'sculpture') {
    return dim === 'angle1' || dim === 'stretch1' || dim === 'angle2' || dim === 'stretch2';
  }
  if (system === 'resonant') {
    return dim === 'angle1' || dim === 'angle2';
  }
  return dim.startsWith('angle');
}
