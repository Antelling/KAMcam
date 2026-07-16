import type { SimulationConfig, Colormap, PhaseSpaceDimension, SystemType } from '../config/schema';
import { SYSTEM_NAMES, vizModeLabel, dimensionLabel, dimensionOrder } from '../config/labels';
import { systemDimensions } from '../config/dimensions';
import { isAngleDim } from '../config/dimensions';

export class Controls {
  private els: Record<string, HTMLElement> = {};

  constructor() {
    const ids = [
      'systemType', 'vizMode', 'resolution', 'colormap', 'toneMapping',
      'xDimension', 'yDimension', 'xMin', 'xMax', 'yMin', 'yMax',
      'initAngle1', 'initVelocity1', 'initAngle2', 'initVelocity2',
      'initStretch1', 'initStretchRate1', 'initStretch2', 'initStretchRate2',
      'dt', 'iterations', 'maxIter', 'perturb',
      'resetBtn', 'zoomOutBtn', 'playBtn', 'previewToggleBtn',
      'sliceMode', 'tileCols', 'tileRows', 'regenerateTilesBtn',
      'tilingControls', 'tilingIndicator',
      'modeIndicator', 'subtitle', 'legendGradient',
      'frameCount', 'maxDistance', 'fps', 'zoomLevel',
      'frameRow', 'maxDistRow',
      'iterValue', 'perturbValue', 'trialsValue',
      'm1Value', 'm2Value', 'L1Value', 'L2Value',
      'k1Value', 'k2Value',
      'elasticControls', 'doublePendulumParams',
      'sculptureControls', 'sculptureParams',
      'resonantControls',
      'perturbControl', 'perturbModeControl', 'trialsControl',
      'm1', 'm2', 'L1', 'L2', 'k1', 'k2',
      'scWeight', 'scRod', 'scAxle', 'scReduction', 'scN',
      'scWeightValue', 'scRodValue', 'scAxleValue', 'scReductionValue', 'scNValue',
      'rpM0', 'rpM1', 'rpL0', 'rpL1', 'rpA0',
      'rpM0Value', 'rpM1Value', 'rpL0Value', 'rpL1Value', 'rpA0Value',
      'perturbDistribution', 'trials',
      'canvas', 'canvasWrapper', 'zoomOverlay',
      'status',
    ];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) this.els[id] = el;
    }
  }

  el(id: string): HTMLElement | null {
    return this.els[id] ?? document.getElementById(id);
  }

  val(id: string): string {
    const e = this.el(id);
    return e instanceof HTMLInputElement || e instanceof HTMLSelectElement ? e.value : '';
  }

  setVal(id: string, v: string | number): void {
    const e = this.el(id);
    if (e instanceof HTMLInputElement || e instanceof HTMLSelectElement) e.value = String(v);
  }

  setText(id: string, t: string): void {
    const e = this.el(id);
    if (e) e.textContent = t;
  }

  show(id: string): void {
    const e = this.el(id);
    if (e) e.classList.remove('hidden');
  }

  hide(id: string): void {
    const e = this.el(id);
    if (e) e.classList.add('hidden');
  }

  setVisible(id: string, visible: boolean): void {
    visible ? this.show(id) : this.hide(id);
  }

  bindControl(id: string, cb: (v: string) => void, eventType = 'input'): void {
    const e = this.el(id);
    if (!e) return;
    if (eventType === 'input' && (e instanceof HTMLInputElement || e instanceof HTMLSelectElement)) {
      e.addEventListener('input', () => cb(e.value));
    } else {
      e.addEventListener('change', () => cb(this.val(id)));
    }
  }

  bindButton(id: string, cb: () => void): void {
    const e = this.el(id);
    if (e) e.addEventListener('click', cb);
  }

  bindRange(id: string, valueId: string, cb: (v: string) => void, format?: (v: number) => string): void {
    this.bindControl(id, (v) => {
      cb(v);
      const fmt = format ?? ((n: number) => String(n));
      this.setText(valueId, fmt(parseFloat(v)));
    });
  }

  updateModeUI(c: SimulationConfig): void {
    const needsPerturb = c.vizMode === 'divergence' || c.vizMode === 'divergenceDistance';
    this.setVisible('perturbControl', needsPerturb);
    this.setVisible('perturbModeControl', needsPerturb);
    this.setVisible('trialsControl', needsPerturb);

    const isSculpture = c.system === 'sculpture';
    const isResonant = c.system === 'resonant';
    const isElastic = c.system === 'elastic' || c.system === 'nonlinear';
    this.setVisible('doublePendulumParams', !isSculpture && !isResonant);
    this.setVisible('elasticControls', isElastic);
    this.setVisible('sculptureControls', isSculpture);
    this.setVisible('resonantControls', isResonant);

    this.applyDimensionUI(c.system, c.sculptureN);
    this.setText('modeIndicator', `${SYSTEM_NAMES[c.system]} -- ${vizModeLabel(c.system, c.vizMode)}`);

    const isSculptOrRes = isSculpture || isResonant;
    const subtitles: Record<string, string> = {
      distance: isSculptOrRes
        ? 'Total distance traveled by the last bob'
        : isElastic
          ? 'Total distance traveled by bob2 (elastic system)'
          : 'Total distance traveled by the second pendulum bob',
      divergence: 'Iterations until perturbed trajectory diverges',
      divergenceDistance: isSculptOrRes
        ? 'Distance traveled by the last bob when trajectories diverge'
        : 'Distance traveled by bob2 when trajectories diverge',
      position: isSculptOrRes
        ? 'Current end-effector position mapped to a rainbow color'
        : 'Current bob2 position mapped to a rainbow color',
      neighborDistance: 'Average distance between neighboring pendulum positions',
      neighborDistanceAccumulated: 'Accumulated average distance between neighboring pendulum positions',
    };
    this.setText('subtitle', subtitles[c.vizMode]);

    const isDiv = c.vizMode === 'divergence';
    this.setVisible('frameRow', !isDiv);
    this.setVisible('maxDistRow', !isDiv && c.vizMode !== 'position');

    this.ensureDistinctDimensions(c.phaseSpace.x.dimension, c.phaseSpace.y.dimension);
  }

  updateLegend(cm: Colormap): void {
    const g = this.el('legendGradient');
    if (g) {
      g.style.background = cm === 6
        ? 'linear-gradient(90deg, hsl(300,80%,50%), hsl(240,80%,50%), hsl(180,80%,50%), hsl(120,80%,50%), hsl(60,80%,50%), hsl(0,80%,50%))'
        : 'linear-gradient(90deg, rgb(68, 1, 84), rgb(33, 145, 140), rgb(253, 231, 37))';
    }
  }

  updatePhaseSpaceInputs(c: SimulationConfig): void {
    this.setVal('xDimension', c.phaseSpace.x.dimension);
    this.setVal('xMin', c.phaseSpace.x.min.toFixed(2));
    this.setVal('xMax', c.phaseSpace.x.max.toFixed(2));
    this.setVal('yDimension', c.phaseSpace.y.dimension);
    this.setVal('yMin', c.phaseSpace.y.min.toFixed(2));
    this.setVal('yMax', c.phaseSpace.y.max.toFixed(2));

    const iv = c.phaseSpace.initialValues;
    this.setVal('initAngle1', iv.angle1.toFixed(2));
    this.setVal('initVelocity1', iv.velocity1.toFixed(2));
    this.setVal('initAngle2', iv.angle2.toFixed(2));
    this.setVal('initVelocity2', iv.velocity2.toFixed(2));
    this.setVal('initStretch1', iv.stretch1.toFixed(2));
    this.setVal('initStretchRate1', iv.stretchRate1.toFixed(2));
    this.setVal('initStretch2', iv.stretch2.toFixed(2));
    this.setVal('initStretchRate2', iv.stretchRate2.toFixed(2));
  }

  updatePendulumParams(c: SimulationConfig): void {
    this.setVal('m1', c.m1);
    this.setText('m1Value', c.m1.toFixed(1));
    this.setVal('m2', c.m2);
    this.setText('m2Value', c.m2.toFixed(1));
    this.setVal('L1', c.L1);
    this.setText('L1Value', c.L1.toFixed(1));
    this.setVal('L2', c.L2);
    this.setText('L2Value', c.L2.toFixed(1));
    this.setVal('k1', c.k1);
    this.setText('k1Value', String(c.k1));
    this.setVal('k2', c.k2);
    this.setText('k2Value', String(c.k2));
  }

  updateSculptureParams(c: SimulationConfig): void {
    this.setVal('scWeight', c.sculptureWeight);
    this.setText('scWeightValue', String(c.sculptureWeight));
    this.setVal('scRod', c.sculptureRod);
    this.setText('scRodValue', c.sculptureRod.toFixed(1));
    this.setVal('scAxle', c.sculptureAxle);
    this.setText('scAxleValue', c.sculptureAxle.toFixed(1));
    this.setVal('scReduction', c.sculptureReduction);
    this.setText('scReductionValue', c.sculptureReduction.toFixed(2));
    this.setVal('scN', c.sculptureN);
    this.setText('scNValue', String(c.sculptureN));
  }

  updateResonantParams(c: SimulationConfig): void {
    this.setVal('rpM0', c.rpM0);
    this.setText('rpM0Value', String(c.rpM0));
    this.setVal('rpM1', c.rpM1);
    this.setText('rpM1Value', String(c.rpM1));
    this.setVal('rpL0', c.rpL0);
    this.setText('rpL0Value', c.rpL0.toFixed(1));
    this.setVal('rpL1', c.rpL1);
    this.setText('rpL1Value', c.rpL1.toFixed(1));
    this.setVal('rpA0', c.rpA0);
    this.setText('rpA0Value', c.rpA0.toFixed(1));
  }

  updateIntegrationInputs(c: SimulationConfig): void {
    this.setVal('dt', c.dt.toFixed(4));
    this.setVal('iterations', c.iterationsPerFrame);
    this.setText('iterValue', String(c.iterationsPerFrame));
    this.setVal('maxIter', c.maxIter);
    this.setVal('perturb', c.perturb);
    this.setText('perturbValue', c.perturb.toFixed(6));
    this.setVal('perturbDistribution', c.perturbDistribution);
    this.setVal('trials', c.trials);
    this.setText('trialsValue', String(c.trials));
  }

  updateTilingUI(c: SimulationConfig): void {
    const isTiling = c.phaseSpace.mode === 'tiling';
    this.setVisible('tilingControls', isTiling);
    this.setVal('sliceMode', c.phaseSpace.mode);
    this.setVal('tileCols', c.phaseSpace.tiling.cols);
    this.setVal('tileRows', c.phaseSpace.tiling.rows);
  }

  updateStats(fc: number, mv: number, fps: number, zl: number): void {
    this.setText('frameCount', String(fc));
    this.setText('maxDistance', mv.toFixed(2));
    this.setText('fps', String(fps));
    this.setText('zoomLevel', String(zl));
  }

  ensureDistinctDimensions(xDim: PhaseSpaceDimension, yDim: PhaseSpaceDimension): void {
    const xSel = this.el('xDimension') as HTMLSelectElement | null;
    const ySel = this.el('yDimension') as HTMLSelectElement | null;
    if (!xSel || !ySel) return;
    for (let i = 0; i < xSel.options.length; i++) {
      xSel.options[i].disabled = xSel.options[i].value === yDim;
    }
    for (let i = 0; i < ySel.options.length; i++) {
      ySel.options[i].disabled = ySel.options[i].value === xDim;
    }
  }

  private applyDimensionUI(system: SystemType, sculptureN: number): void {
    const active = systemDimensions(system, sculptureN);
    const activeSet = new Set<PhaseSpaceDimension>(active);

    for (const selId of ['xDimension', 'yDimension']) {
      const sel = this.el(selId) as HTMLSelectElement | null;
      if (!sel) continue;
      for (let i = 0; i < sel.options.length; i++) {
        const opt = sel.options[i];
        const dim = opt.value as PhaseSpaceDimension;
        opt.style.display = activeSet.has(dim) ? '' : 'none';
        opt.text = dimensionLabel(system, dim);
      }
    }

    const rows = document.querySelectorAll('.initial-values-grid [data-dim]');
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i] as HTMLElement;
      const dim = (row.getAttribute('data-dim') ?? '') as PhaseSpaceDimension;
      const visible = activeSet.has(dim);
      row.style.display = visible ? '' : 'none';
      if (visible) {
        row.style.order = String(dimensionOrder(system, sculptureN, dim));
        const lbl = row.querySelector('label');
        if (lbl) lbl.textContent = dimensionLabel(system, dim);
      }
    }
  }
}
