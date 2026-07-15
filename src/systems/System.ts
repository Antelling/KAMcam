import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../config/schema';

export interface SystemBuffers {
  stateA: any;
  stateB: any;
  data: any;
}

export interface SystemPipelines {
  initPipeline: any;
  stepPipeline: any;
  accumulatePipeline: any;
  divergenceStepPipeline?: any;
}

export interface SystemBindGroups {
  init: any;
  stepForward: any;
  stepBackward: any;
  accumulateA: any;
  accumulateB: any;
}

export interface System {
  readonly key: 'rigid' | 'elastic' | 'nonlinear' | 'sculpture' | 'resonant';
  readonly stateSize: number;
  build(root: TgpuRoot, config: SimulationConfig, cellCount: number): void;
  updateParams(config: SimulationConfig): void;
  initSim(): void;
  stepSim(): void;
  accumulateSim(): void;
  get data(): any;
  getReadIndex(): 0 | 1;
}
