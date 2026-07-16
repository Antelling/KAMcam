import type { SimulationConfig, PhaseSpaceDimension } from '../config/schema';
import { generateTiling } from '../config/corners';
import { systemDimensions, isAngleDim, initialVector } from '../config/dimensions';
import type { Controls } from '../ui/Controls';
import type { ZoomController } from '../ui/ZoomController';

export interface ControlBindingsContext {
  config: SimulationConfig;
  controls: Controls;
  zoomController: ZoomController;
  markStale: () => void;
  syncUI: () => void;
  regenerateTiling: () => void;
  handleSystemChange: () => void;
  applyAxisDefaults: (axis: 'x' | 'y') => void;
  resetZoom: () => void;
}

export function setupAllBindings(ctx: ControlBindingsContext): void {
  const { config: c, controls: ui } = ctx;

  ui.bindButton('playBtn', () => {});
  ui.bindButton('zoomOutBtn', () => ctx.zoomController.zoomOut());
  ui.bindButton('resetBtn', () => {
    ctx.resetZoom();
    ui.updatePhaseSpaceInputs(c);
    ctx.markStale();
  });

  ui.bindControl('systemType', (v) => {
    c.system = v as SimulationConfig['system'];
    ctx.handleSystemChange();
    ctx.regenerateTiling();
    ctx.syncUI();
    ctx.markStale();
  }, 'change');

  ui.bindControl('vizMode', (v) => {
    c.vizMode = v as SimulationConfig['vizMode'];
    ui.updateModeUI(c);
    ctx.markStale();
  }, 'change');

  ui.bindControl('resolution', (v) => {
    c.resolution = parseInt(v) as SimulationConfig['resolution'];
    ctx.markStale();
  }, 'change');

  ui.bindControl('xDimension', (v) => {
    c.phaseSpace.mode = 'manual';
    c.phaseSpace.x.dimension = v as PhaseSpaceDimension;
    ctx.applyAxisDefaults('x');
    ui.updatePhaseSpaceInputs(c);
    ui.ensureDistinctDimensions(c.phaseSpace.x.dimension, c.phaseSpace.y.dimension);
    ui.updateTilingUI(c);
    ctx.resetZoom();
    ctx.markStale();
  }, 'change');

  ui.bindControl('yDimension', (v) => {
    c.phaseSpace.mode = 'manual';
    c.phaseSpace.y.dimension = v as PhaseSpaceDimension;
    ctx.applyAxisDefaults('y');
    ui.updatePhaseSpaceInputs(c);
    ui.ensureDistinctDimensions(c.phaseSpace.x.dimension, c.phaseSpace.y.dimension);
    ui.updateTilingUI(c);
    ctx.resetZoom();
    ctx.markStale();
  }, 'change');

  ['xMin', 'xMax', 'yMin', 'yMax'].forEach(id => {
    ui.bindControl(id, (v) => {
      const val = parseFloat(v);
      if (id === 'xMin') c.phaseSpace.x.min = val;
      else if (id === 'xMax') c.phaseSpace.x.max = val;
      else if (id === 'yMin') c.phaseSpace.y.min = val;
      else if (id === 'yMax') c.phaseSpace.y.max = val;
      ctx.resetZoom();
      ctx.markStale();
    }, 'change');
  });

  const ivMap: Record<string, PhaseSpaceDimension> = {
    initAngle1: 'angle1', initVelocity1: 'velocity1',
    initAngle2: 'angle2', initVelocity2: 'velocity2',
    initStretch1: 'stretch1', initStretchRate1: 'stretchRate1',
    initStretch2: 'stretch2', initStretchRate2: 'stretchRate2',
  };
  Object.entries(ivMap).forEach(([id, dim]) => {
    ui.bindControl(id, (v) => {
      c.phaseSpace.initialValues[dim] = parseFloat(v);
      ctx.markStale();
    }, 'change');
  });

  ui.bindControl('dt', (v) => {
    c.dt = parseFloat(v);
    ui.updateIntegrationInputs(c);
    ctx.markStale();
  }, 'change');

  ui.bindControl('iterations', (v) => {
    c.iterationsPerFrame = parseInt(v);
    ui.setText('iterValue', String(c.iterationsPerFrame));
  });

  ui.bindControl('maxIter', (v) => {
    c.maxIter = parseInt(v);
    ctx.markStale();
  }, 'change');

  ui.bindControl('perturb', (v) => {
    c.perturb = parseFloat(v);
    ui.setText('perturbValue', c.perturb.toFixed(6));
    ctx.markStale();
  });

  ui.bindControl('perturbDistribution', (v) => {
    c.perturbDistribution = v as SimulationConfig['perturbDistribution'];
    ctx.markStale();
  }, 'change');

  ui.bindControl('trials', (v) => {
    c.trials = Math.max(1, parseInt(v) || 1);
    ui.setText('trialsValue', String(c.trials));
    ctx.markStale();
  });

  ui.bindControl('colormap', (v) => {
    c.colormap = parseInt(v) as SimulationConfig['colormap'];
    ui.updateLegend(c.colormap);
  }, 'change');

  ui.bindControl('toneMapping', (v) => {
    c.toneMapping = parseInt(v) as SimulationConfig['toneMapping'];
  }, 'change');

  ui.bindControl('sliceMode', (v) => {
    c.phaseSpace.mode = v as 'manual' | 'tiling';
    if (c.phaseSpace.mode === 'tiling') ctx.regenerateTiling();
    ui.updateTilingUI(c);
    ctx.markStale();
  }, 'change');

  ui.bindControl('tileCols', (v) => {
    c.phaseSpace.tiling.cols = Math.max(1, parseInt(v) || 1);
    ctx.regenerateTiling();
    ui.updateTilingUI(c);
    ctx.markStale();
  }, 'change');

  ui.bindControl('tileRows', (v) => {
    c.phaseSpace.tiling.rows = Math.max(1, parseInt(v) || 1);
    ctx.regenerateTiling();
    ui.updateTilingUI(c);
    ctx.markStale();
  }, 'change');

  ui.bindButton('regenerateTilesBtn', () => {
    ctx.regenerateTiling();
    ctx.markStale();
  });

  bindRangeParam(ui, c, 'm1', 'm1Value', (v) => { c.m1 = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'm2', 'm2Value', (v) => { c.m2 = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'L1', 'L1Value', (v) => { c.L1 = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'L2', 'L2Value', (v) => { c.L2 = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'k1', 'k1Value', (v) => { c.k1 = v; }, (v) => String(v));
  bindRangeParam(ui, c, 'k2', 'k2Value', (v) => { c.k2 = v; }, (v) => String(v));

  bindRangeParam(ui, c, 'scWeight', 'scWeightValue', (v) => { c.sculptureWeight = v; }, (v) => String(v));
  bindRangeParam(ui, c, 'scRod', 'scRodValue', (v) => { c.sculptureRod = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'scAxle', 'scAxleValue', (v) => { c.sculptureAxle = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'scReduction', 'scReductionValue', (v) => { c.sculptureReduction = v; }, (v) => v.toFixed(2));

  ui.bindControl('scN', (v) => {
    c.sculptureN = Math.max(1, Math.min(4, parseInt(v, 10) || 3));
    ctx.handleSystemChange();
    ctx.regenerateTiling();
    ui.updateModeUI(c);
    ui.updatePhaseSpaceInputs(c);
    ctx.markStale();
  }, 'change');

  bindRangeParam(ui, c, 'rpM0', 'rpM0Value', (v) => { c.rpM0 = v; }, (v) => String(v));
  bindRangeParam(ui, c, 'rpM1', 'rpM1Value', (v) => { c.rpM1 = v; }, (v) => String(v));
  bindRangeParam(ui, c, 'rpL0', 'rpL0Value', (v) => { c.rpL0 = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'rpL1', 'rpL1Value', (v) => { c.rpL1 = v; }, (v) => v.toFixed(1));
  bindRangeParam(ui, c, 'rpA0', 'rpA0Value', (v) => { c.rpA0 = v; }, (v) => v.toFixed(1));
}

function bindRangeParam(
  ui: Controls,
  _c: SimulationConfig,
  id: string,
  valueId: string,
  apply: (v: number) => void,
  fmt: (v: number) => string,
): void {
  ui.bindControl(id, (v) => {
    const n = parseFloat(v);
    apply(n);
    ui.setText(valueId, fmt(n));
  });
}
