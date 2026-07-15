import type { System } from './System';
import type { SystemType, SimulationConfig } from '../config/schema';
import { RigidSystem } from './rigid/RigidSystem';
import { ElasticSystem } from './elastic/ElasticSystem';
import { NonlinearSystem } from './nonlinear/NonlinearSystem';
import { SculptureSystem } from './sculpture/SculptureSystem';
import { ResonantSystem } from './resonant/ResonantSystem';

export function createSystem(systemType: SystemType): System {
  switch (systemType) {
    case 'rigid': return new RigidSystem();
    case 'elastic': return new ElasticSystem();
    case 'nonlinear': return new NonlinearSystem();
    case 'sculpture': return new SculptureSystem();
    case 'resonant': return new ResonantSystem();
  }
}

export function createSystemFromConfig(config: SimulationConfig): System {
  return createSystem(config.system);
}
