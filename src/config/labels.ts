import type { SystemType, VizMode, Colormap, ToneMapping, PhaseSpaceDimension } from './schema';
import { systemDimensions } from './dimensions';

export const SYSTEM_NAMES: Record<SystemType, string> = {
  rigid: 'Rigid',
  elastic: 'Elastic Double Pendulum',
  nonlinear: 'Nonlinear Elastic Pendulum',
  sculpture: 'Kinetic Sculpture',
  resonant: 'Resonant Pendulum',
};

export const MODE_NAMES: Record<VizMode, string> = {
  distance: 'Bob2 Distance',
  divergence: 'Divergence Time',
  divergenceDistance: 'Divergence Distance',
  position: 'Pendulum Position',
  neighborDistance: 'Neighbor Distance',
  neighborDistanceAccumulated: 'Accumulated Neighbor Distance',
};

export function vizModeLabel(system: SystemType, mode: VizMode): string {
  if ((system === 'sculpture' || system === 'resonant') && mode === 'distance') return 'Last Bob Distance';
  if (mode === 'position') return (system === 'sculpture' || system === 'resonant') ? 'End Effector Position' : 'Bob2 Position';
  return MODE_NAMES[mode];
}

export const COLORMAP_NAMES: Record<Colormap, string> = {
  0: 'Viridis',
  1: 'Magma',
  2: 'Plasma',
  3: 'Inferno',
  4: 'Turbo',
  5: 'Jet',
  6: 'Rainbow',
};

export const TONE_MAPPING_NAMES: Record<ToneMapping, string> = {
  0: 'Linear',
  1: 'Logarithmic',
  2: 'Square Root',
  3: 'S-Curve',
};

export const DIMENSION_LABELS: Record<PhaseSpaceDimension, string> = {
  angle1: 'First Angle \u03B8\u2081',
  velocity1: 'First Angular Velocity \u03C9\u2081',
  angle2: 'Second Angle \u03B8\u2082',
  velocity2: 'Second Angular Velocity \u03C9\u2082',
  stretch1: 'First Arm Stretch r\u2081',
  stretchRate1: 'First Stretch Rate \u1E59\u2081',
  stretch2: 'Second Arm Stretch r\u2082',
  stretchRate2: 'Second Stretch Rate \u1E59\u2082',
};

export const DIM_SYMBOLS: Record<PhaseSpaceDimension, string> = {
  angle1: '\u03B8\u2081',
  velocity1: '\u03C9\u2081',
  angle2: '\u03B8\u2082',
  velocity2: '\u03C9\u2082',
  stretch1: 'r\u2081',
  stretchRate1: '\u1E59\u2081',
  stretch2: 'r\u2082',
  stretchRate2: '\u1E59\u2082',
};

export function dimensionLabel(system: SystemType, dim: PhaseSpaceDimension): string {
  if (system === 'sculpture') {
    switch (dim) {
      case 'angle1': return 'Level 0 Angle \u03B8\u2080';
      case 'velocity1': return 'Level 0 Velocity \u03C9\u2080';
      case 'stretch1': return 'Level 1 Angle \u03B8\u2081';
      case 'stretchRate1': return 'Level 1 Velocity \u03C9\u2081';
      case 'angle2': return 'Level 2 Angle \u03B8\u2082';
      case 'velocity2': return 'Level 2 Velocity \u03C9\u2082';
      case 'stretch2': return 'Level 3 Angle \u03B8\u2083';
      case 'stretchRate2': return 'Level 3 Velocity \u03C9\u2083';
    }
  }
  if (system === 'resonant') {
    switch (dim) {
      case 'angle1': return 'Level 0 Angle \u03B8\u2080';
      case 'velocity1': return 'Level 0 Velocity \u03C9\u2080';
      case 'angle2': return 'Level 1 Angle \u03B8\u2081';
      case 'velocity2': return 'Level 1 Velocity \u03C9\u2081';
    }
  }
  switch (dim) {
    case 'angle1': return 'First Angle \u03B8\u2081';
    case 'velocity1': return 'First Angular Velocity \u03C9\u2081';
    case 'angle2': return 'Second Angle \u03B8\u2082';
    case 'velocity2': return 'Second Angular Velocity \u03C9\u2082';
    case 'stretch1': return 'First Arm Stretch r\u2081';
    case 'stretchRate1': return 'First Stretch Rate \u1E59\u2081';
    case 'stretch2': return 'Second Arm Stretch r\u2082';
    case 'stretchRate2': return 'Second Stretch Rate \u1E59\u2082';
  }
  return dim;
}

export function dimensionOrder(system: SystemType, sculptureN: number, dim: PhaseSpaceDimension): number {
  const list = systemDimensions(system, sculptureN);
  const idx = list.indexOf(dim);
  return idx < 0 ? 999 : idx;
}
