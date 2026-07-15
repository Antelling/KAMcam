import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig, VizMode } from '../config/schema';
import type { System } from '../systems/System';
import { createSystemFromConfig } from '../systems/registry';

const VIZ_MODE_DISTANCE = 0;
const VIZ_MODE_DIVERGENCE = 2;
const VIZ_MODE_DIVERGENCE_DISTANCE = 3;

function vizModeIndex(vm: VizMode): number {
  switch (vm) {
    case 'distance': return 0;
    case 'divergence': return 2;
    case 'divergenceDistance': return 3;
    case 'position': return 4;
    case 'neighborDistance': return 1;
    case 'neighborDistanceAccumulated': return 1;
  }
}

export class Simulator {
  private root: TgpuRoot;
  private config: SimulationConfig;
  private system: System;
  private cellCount: number;
  private frameCount = 0;
  private isDivergence = false;
  private trialCount = 0;

  constructor(root: TgpuRoot, config: SimulationConfig) {
    this.root = root;
    this.config = config;
    this.cellCount = config.resolution * config.resolution;
    this.system = createSystemFromConfig(config);
    this.system.build(root, config, this.cellCount);
    this.isDivergence = this.config.vizMode === 'divergence' || this.config.vizMode === 'divergenceDistance';
  }

  get data() { return this.system.data; }
  get configValue() { return this.config; }

  getVizModeIndex(): number {
    return vizModeIndex(this.config.vizMode);
  }

  init(): void {
    this.system.updateParams(this.config);
    if (this.isDivergence) {
      this.trialCount = 0;
      this.system.initDivergence(this.config.seed, this.config.perturb);
    } else {
      this.system.initSim();
    }
    this.frameCount = 0;
  }

  step(): void {
    if (this.isDivergence) {
      this.stepDivergence();
    } else {
      this.stepDistance();
    }
    this.frameCount++;
  }

  private stepDistance(): void {
    const batchSize = Math.min(this.config.iterationsPerFrame, this.config.maxIter - this.frameCount);
    for (let i = 0; i < batchSize; i++) {
      this.system.stepSim();
    }
    this.system.accumulateSim();
  }

  private stepDivergence(): void {
    const batchSize = Math.min(20, this.config.maxIter - this.frameCount);
    for (let i = 0; i < batchSize; i++) {
      this.system.divergenceStep();
    }
  }

  getFrameCount(): number {
    return this.frameCount;
  }

  isComplete(): boolean {
    return this.frameCount >= this.config.maxIter;
  }
}
