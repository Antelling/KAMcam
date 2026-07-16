import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { ResonantStateA, ResonantStateB, ResonantParams, ResonantDivParams, DataCell } from './types';
import { systemDeriv, computeResonantTip, resonantDivergence } from './deriv';
import { hash } from '../shared/hash';
import { DualStateSystem } from '../shared/DualStateSystem';
import type { GpuPipelinesResult, DivGpuPipelinesResult } from '../shared/DualStateSystem';

export class ResonantSystem extends DualStateSystem {
  readonly key = 'resonant' as const;
  readonly stateSize = 32;

  createStateArrayA(cellCount: number): any { return d.arrayOf(ResonantStateA, cellCount); }
  createStateArrayB(cellCount: number): any { return d.arrayOf(ResonantStateB, cellCount); }
  createDataArray(cellCount: number): any { return d.arrayOf(DataCell, cellCount); }
  paramsStruct(): any { return ResonantParams; }
  divParamsStruct(): any { return ResonantDivParams; }
  zeroStateA(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta0: 0, omega0: 0, dummy1: 0, dummy2: 0 }));
  }
  zeroStateB(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, dummy1: 0, dummy2: 0 }));
  }

  protected buildGpuPipelines(root: TgpuRoot, stateAa: any, stateAb: any, stateBa: any, stateBb: any): GpuPipelinesResult {
    const initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ResonantParams },
    });
    const stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA }, currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' }, nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ResonantParams },
    });
    const accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA }, currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: ResonantParams },
    });
    const il = initLayout; const sl = stepLayout; const al = accumulateLayout;

    const initCell = (ci: number) => {
      'use gpu';
      const p = il.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const omu = 1 - u; const omv = 1 - v;
      il.$.stateA[ci] = ResonantStateA({ theta0: omu*omv*p.cA00_th+u*omv*p.cA10_th+omu*v*p.cA01_th+u*v*p.cA11_th, omega0: omu*omv*p.cA00_om+u*omv*p.cA10_om+omu*v*p.cA01_om+u*v*p.cA11_om, dummy1: 0, dummy2: 0 });
      il.$.stateB[ci] = ResonantStateB({ theta1: omu*omv*p.cB00_th+u*omv*p.cB10_th+omu*v*p.cB01_th+u*v*p.cB11_th, omega1: omu*omv*p.cB00_om+u*omv*p.cB10_om+omu*v*p.cB01_om+u*v*p.cB11_om, dummy1: 0, dummy2: 0 });
    };

    const stepCell = (ci: number) => {
      'use gpu';
      const sa = sl.$.currentA[ci]; const sb = sl.$.currentB[ci]; const p = sl.$.params;
      const t0 = sa.theta0; const w0 = sa.omega0;
      const t1 = sb.theta1; const w1 = sb.omega1;
      const dt = p.dt;
      const a0 = systemDeriv(t0, w0, t1, w1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      const hw0 = w0 + 0.5 * dt * a0.da_om;
      const hw1 = w1 + 0.5 * dt * a0.db_om;
      const nt0 = t0 + dt * hw0;
      const nt1 = t1 + dt * hw1;
      const a1 = systemDeriv(nt0, hw0, nt1, hw1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      sl.$.nextA[ci] = ResonantStateA({ theta0: nt0, omega0: hw0 + 0.5*dt*a1.da_om, dummy1: 0, dummy2: 0 });
      sl.$.nextB[ci] = ResonantStateB({ theta1: nt1, omega1: hw1 + 0.5*dt*a1.db_om, dummy1: 0, dummy2: 0 });
    };

    const accumulateCell = (ci: number) => {
      'use gpu';
      const sa = al.$.currentA[ci]; const sb = al.$.currentB[ci]; const data = al.$.data[ci]; const p = al.$.params;
      const tip = computeResonantTip(sa.theta0, sb.theta1, p.rpL0, p.rpL1, p.rpA0);
      if (data.a > 0.5) { const dx = tip.x - data.r; const dy = tip.y - data.g; const dist = std.sqrt(dx*dx+dy*dy); al.$.data[ci] = DataCell({ r: tip.x, g: tip.y, b: data.b+dist, a: 1 }); }
      else { al.$.data[ci] = DataCell({ r: tip.x, g: tip.y, b: 0, a: 1 }); }
    };

    return {
      initLayout, stepLayout, accumulateLayout,
      initPipeline: root.createGuardedComputePipeline(initCell),
      stepPipeline: root.createGuardedComputePipeline(stepCell),
      accumulatePipeline: root.createGuardedComputePipeline(accumulateCell),
      initBG: root.createBindGroup(initLayout, { stateA: stateAa, stateB: stateBa, params: this.paramsBuffer }),
      stepFwd: root.createBindGroup(stepLayout, { currentA: stateAa, currentB: stateBa, nextA: stateAb, nextB: stateBb, params: this.paramsBuffer }),
      stepBwd: root.createBindGroup(stepLayout, { currentA: stateAb, currentB: stateBb, nextA: stateAa, nextB: stateBa, params: this.paramsBuffer }),
      accA: root.createBindGroup(accumulateLayout, { currentA: stateAa, currentB: stateBa, data: this.dataBuf, params: this.paramsBuffer }),
      accB: root.createBindGroup(accumulateLayout, { currentA: stateAb, currentB: stateBb, data: this.dataBuf, params: this.paramsBuffer }),
    };
  }

  protected buildDivGpuPipelines(root: TgpuRoot, stateAa: any, stateAb: any, stateBa: any, stateBb: any): DivGpuPipelinesResult {
    const divInitLayout = tgpu.bindGroupLayout({
      baseStateA: { storage: this.StateArrayA, access: 'mutable' }, baseStateB: { storage: this.StateArrayB, access: 'mutable' },
      pertStateA: { storage: this.StateArrayA, access: 'mutable' }, pertStateB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' }, params: { uniform: ResonantDivParams },
    });
    const divStepLayout = tgpu.bindGroupLayout({
      baseCurrentA: { storage: this.StateArrayA }, baseCurrentB: { storage: this.StateArrayB },
      pertCurrentA: { storage: this.StateArrayA }, pertCurrentB: { storage: this.StateArrayB },
      baseNextA: { storage: this.StateArrayA, access: 'mutable' }, baseNextB: { storage: this.StateArrayB, access: 'mutable' },
      pertNextA: { storage: this.StateArrayA, access: 'mutable' }, pertNextB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' }, params: { uniform: ResonantDivParams },
    });
    const dil = divInitLayout; const dsl = divStepLayout;

    const verletAdvance = (t0: number, w0: number, t1: number, w1: number, m0: number, m1: number, L0: number, L1: number, a0: number, dt: number) => {
      'use gpu';
      const d0 = systemDeriv(t0, w0, t1, w1, m0, m1, L0, L1, a0);
      const hw0 = w0 + 0.5 * dt * d0.da_om;
      const hw1 = w1 + 0.5 * dt * d0.db_om;
      const nt0 = t0 + dt * hw0;
      const nt1 = t1 + dt * hw1;
      const d1 = systemDeriv(nt0, hw0, nt1, hw1, m0, m1, L0, L1, a0);
      return { th0: nt0, om0: hw0 + 0.5*dt*d1.da_om, th1: nt1, om1: hw1 + 0.5*dt*d1.db_om };
    };

    const divInitCell = (ci: number) => {
      'use gpu';
      const p = dil.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const omu = 1 - u; const omv = 1 - v;
      const th0 = omu*omv*p.cA00_th+u*omv*p.cA10_th+omu*v*p.cA01_th+u*v*p.cA11_th;
      const om0 = omu*omv*p.cA00_om+u*omv*p.cA10_om+omu*v*p.cA01_om+u*v*p.cA11_om;
      const th1 = omu*omv*p.cB00_th+u*omv*p.cB10_th+omu*v*p.cB01_th+u*v*p.cB11_th;
      const om1 = omu*omv*p.cB00_om+u*omv*p.cB10_om+omu*v*p.cB01_om+u*v*p.cB11_om;
      dil.$.baseStateA[ci] = ResonantStateA({ theta0: th0, omega0: om0, dummy1: 0, dummy2: 0 });
      dil.$.baseStateB[ci] = ResonantStateB({ theta1: th1, omega1: om1, dummy1: 0, dummy2: 0 });
      const px = d.f32(x) / p.resolution; const py = d.f32(y) / p.resolution;
      const h0 = (hash(px*1000+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      const h1 = (hash(px*1000+100+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      dil.$.pertStateA[ci] = ResonantStateA({ theta0: th0+h0, omega0: om0, dummy1: 0, dummy2: 0 });
      dil.$.pertStateB[ci] = ResonantStateB({ theta1: th1+h1, omega1: om1, dummy1: 0, dummy2: 0 });
      dil.$.divData[ci] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divStepCell = (ci: number) => {
      'use gpu';
      const p = dsl.$.params;
      const bsa = dsl.$.baseCurrentA[ci]; const bsb = dsl.$.baseCurrentB[ci];
      const psa = dsl.$.pertCurrentA[ci]; const psb = dsl.$.pertCurrentB[ci];
      const bn = verletAdvance(bsa.theta0, bsa.omega0, bsb.theta1, bsb.omega1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0, p.dt);
      dsl.$.baseNextA[ci] = ResonantStateA({ theta0: bn.th0, omega0: bn.om0, dummy1: 0, dummy2: 0 });
      dsl.$.baseNextB[ci] = ResonantStateB({ theta1: bn.th1, omega1: bn.om1, dummy1: 0, dummy2: 0 });
      const pn = verletAdvance(psa.theta0, psa.omega0, psb.theta1, psb.omega1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0, p.dt);
      dsl.$.pertNextA[ci] = ResonantStateA({ theta0: pn.th0, omega0: pn.om0, dummy1: 0, dummy2: 0 });
      dsl.$.pertNextB[ci] = ResonantStateB({ theta1: pn.th1, omega1: pn.om1, dummy1: 0, dummy2: 0 });
      const div = resonantDivergence(bn.th0, bn.om0, bn.th1, bn.om1, pn.th0, pn.om0, pn.th1, pn.om1);
      const data = dsl.$.divData[ci];
      if (div > 1.0 && data.g < 0.5) { dsl.$.divData[ci] = DataCell({ r: p.frameCounter, g: 1, b: 0, a: 1 }); }
    };

    return {
      divInitLayout, divStepLayout,
      divInitPipeline: root.createGuardedComputePipeline(divInitCell),
      divStepPipeline: root.createGuardedComputePipeline(divStepCell),
      divInitBG: root.createBindGroup(divInitLayout, { baseStateA: stateAa, baseStateB: stateBa, pertStateA: this.pertStateAa, pertStateB: this.pertStateBa, divData: this.divDataBuf, params: this.divParamsBuffer }),
      divStepFwd: root.createBindGroup(divStepLayout, { baseCurrentA: stateAa, baseCurrentB: stateBa, pertCurrentA: this.pertStateAa, pertCurrentB: this.pertStateBa, baseNextA: stateAb, baseNextB: stateBb, pertNextA: this.pertStateAb, pertNextB: this.pertStateBb, divData: this.divDataBuf, params: this.divParamsBuffer }),
      divStepBwd: root.createBindGroup(divStepLayout, { baseCurrentA: stateAb, baseCurrentB: stateBb, pertCurrentA: this.pertStateAb, pertCurrentB: this.pertStateBb, baseNextA: stateAa, baseNextB: stateBa, pertNextA: this.pertStateAa, pertNextB: this.pertStateBa, divData: this.divDataBuf, params: this.divParamsBuffer }),
    };
  }

  buildParamsData(config: SimulationConfig): Record<string, number> {
    this.lastConfig = config;
    return packRpParams(config);
  }

  buildDivParamsData(config: SimulationConfig, seed: number, perturb: number): Record<string, number> & { frameCounter: number } {
    return { ...packRpParams(config), seed, perturb, frameCounter: 0 };
  }
}

function packRpParams(config: SimulationConfig): Record<string, number> {
  const corners = computeCorners(config);
  const ca = (i: number) => elasticPackA(corners[i]);
  const cb = (i: number) => elasticPackB(corners[i]);
  return {
    rpM0: config.rpM0, rpM1: config.rpM1, rpL0: config.rpL0, rpL1: config.rpL1, rpA0: config.rpA0,
    dt: config.dt, resolution: config.resolution,
    cA00_th: ca(0)[0], cA00_om: ca(0)[1], cA00_d1: ca(0)[2], cA00_d2: ca(0)[3],
    cA10_th: ca(1)[0], cA10_om: ca(1)[1], cA10_d1: ca(1)[2], cA10_d2: ca(1)[3],
    cA01_th: ca(2)[0], cA01_om: ca(2)[1], cA01_d1: ca(2)[2], cA01_d2: ca(2)[3],
    cA11_th: ca(3)[0], cA11_om: ca(3)[1], cA11_d1: ca(3)[2], cA11_d2: ca(3)[3],
    cB00_th: cb(0)[0], cB00_om: cb(0)[1], cB00_d1: cb(0)[2], cB00_d2: cb(0)[3],
    cB10_th: cb(1)[0], cB10_om: cb(1)[1], cB10_d1: cb(1)[2], cB10_d2: cb(1)[3],
    cB01_th: cb(2)[0], cB01_om: cb(2)[1], cB01_d1: cb(2)[2], cB01_d2: cb(2)[3],
    cB11_th: cb(3)[0], cB11_om: cb(3)[1], cB11_d1: cb(3)[2], cB11_d2: cb(3)[3],
  };
}