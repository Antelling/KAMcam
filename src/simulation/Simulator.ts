import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../config/schema';
import type { System } from '../systems/System';
import { createSystemFromConfig } from '../systems/registry';

export class Simulator {
  private root: TgpuRoot;
  private config: SimulationConfig;
  private system: System;
  private cellCount: number;
  private frameCount = 0;

  constructor(root: TgpuRoot, config: SimulationConfig) {
    this.root = root;
    this.config = config;
    this.cellCount = config.resolution * config.resolution;
    this.system = createSystemFromConfig(config);
    this.system.build(root, config, this.cellCount);
  }

  get data() { return this.system.data; }
  get configValue() { return this.config; }

  init(): void {
    this.system.updateParams(this.config);
    this.system.initSim();
    this.frameCount = 0;
  }

  step(): void {
    for (let i = 0; i < this.config.iterationsPerFrame; i++) {
      this.system.stepSim();
    }
    this.system.accumulateSim();
    this.frameCount++;
  }

  getFrameCount(): number {
    return this.frameCount;
  }
}
