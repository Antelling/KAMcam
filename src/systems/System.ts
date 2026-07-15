import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../config/schema';

export interface System {
  readonly key: 'rigid' | 'elastic' | 'nonlinear' | 'sculpture' | 'resonant';
  readonly stateSize: number;
  build(root: TgpuRoot, config: SimulationConfig, cellCount: number): void;
  updateParams(config: SimulationConfig): void;
  initSim(): void;
  stepSim(): void;
  accumulateSim(): void;
  initDivergence(seed: number, perturb: number): void;
  divergenceStep(): void;
  get data(): any;
  getReadIndex(): 0 | 1;
}
