import type { System } from './System';
import type { SimulationConfig } from '../config/schema';
import { RigidSystem } from './rigid/RigidSystem';
import type { TgpuRoot } from 'typegpu';

export function createSystem(systemType: 'rigid'): System {
  switch (systemType) {
    case 'rigid': return new RigidSystem();
  }
}

export function createSystemFromConfig(config: SimulationConfig): System {
  return new RigidSystem();
}
