import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { SculptureStateA, SculptureStateB, SculptureParams, SculptureDivParams, DataCell } from './types';
import { systemDeriv, computeSculptureTip, sculptureDivergence } from './deriv';
import { hash } from '../shared/hash';
import { DualStateSystem } from '../shared/DualStateSystem';
import type { GpuPipelinesResult, DivGpuPipelinesResult } from '../shared/DualStateSystem';

export class SculptureSystem extends DualStateSystem {
  readonly key = 'sculpture' as const;
  readonly stateSize = 32;

  createStateArrayA(cellCount: number): any { return d.arrayOf(SculptureStateA, cellCount); }
  createStateArrayB(cellCount: number): any { return d.arrayOf(SculptureStateB, cellCount); }
  createDataArray(cellCount: number): any { return d.arrayOf(DataCell, cellCount); }
  paramsStruct(): any { return SculptureParams; }
  divParamsStruct(): any { return SculptureDivParams; }
  zeroStateA(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta0: 0, omega0: 0, theta1: 0, omega1: 0 }));
  }
  zeroStateB(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta2: 0, omega2: 0, theta3: 0, omega3: 0 }));
  }

  protected buildGpuPipelines(root: TgpuRoot, stateAa: any, stateAb: any, stateBa: any, stateBb: any): GpuPipelinesResult {
    const initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: SculptureParams },
    });
    const stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA }, currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' }, nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: SculptureParams },
    });
    const accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA }, currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: SculptureParams },
    });
    const il = initLayout; const sl = stepLayout; const al = accumulateLayout;

    const rk4Step = (t0: number, w0: number, t1: number, w1: number, t2: number, w2: number, t3: number, w3: number, sM: number, sL: number, sA: number, sR: number, sN: number, dt: number) => {
      'use gpu';
      const h = 0.5 * dt;
      const d1 = systemDeriv(t0, w0, t1, w1, t2, w2, t3, w3, sM, sL, sA, sR, sN);
      const d2 = systemDeriv(t0+h*d1.dt0, w0+h*d1.dw0, t1+h*d1.dt1, w1+h*d1.dw1, t2+h*d1.dt2, w2+h*d1.dw2, t3+h*d1.dt3, w3+h*d1.dw3, sM, sL, sA, sR, sN);
      const d3 = systemDeriv(t0+h*d2.dt0, w0+h*d2.dw0, t1+h*d2.dt1, w1+h*d2.dw1, t2+h*d2.dt2, w2+h*d2.dw2, t3+h*d2.dt3, w3+h*d2.dw3, sM, sL, sA, sR, sN);
      const d4 = systemDeriv(t0+dt*d3.dt0, w0+dt*d3.dw0, t1+dt*d3.dt1, w1+dt*d3.dw1, t2+dt*d3.dt2, w2+dt*d3.dw2, t3+dt*d3.dt3, w3+dt*d3.dw3, sM, sL, sA, sR, sN);
      const s6 = dt / 6.0;
      return {
        t0: t0+s6*(d1.dt0+2*d2.dt0+2*d3.dt0+d4.dt0), w0: w0+s6*(d1.dw0+2*d2.dw0+2*d3.dw0+d4.dw0),
        t1: t1+s6*(d1.dt1+2*d2.dt1+2*d3.dt1+d4.dt1), w1: w1+s6*(d1.dw1+2*d2.dw1+2*d3.dw1+d4.dw1),
        t2: t2+s6*(d1.dt2+2*d2.dt2+2*d3.dt2+d4.dt2), w2: w2+s6*(d1.dw2+2*d2.dw2+2*d3.dw2+d4.dw2),
        t3: t3+s6*(d1.dt3+2*d2.dt3+2*d3.dt3+d4.dt3), w3: w3+s6*(d1.dw3+2*d2.dw3+2*d3.dw3+d4.dw3),
      };
    };

    const initCell = (ci: number) => {
      'use gpu';
      const p = il.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const omu = 1 - u; const omv = 1 - v;
      il.$.stateA[ci] = SculptureStateA({ theta0: omu*omv*p.cA00_t0+u*omv*p.cA10_t0+omu*v*p.cA01_t0+u*v*p.cA11_t0, omega0: omu*omv*p.cA00_w0+u*omv*p.cA10_w0+omu*v*p.cA01_w0+u*v*p.cA11_w0, theta1: omu*omv*p.cA00_t1+u*omv*p.cA10_t1+omu*v*p.cA01_t1+u*v*p.cA11_t1, omega1: omu*omv*p.cA00_w1+u*omv*p.cA10_w1+omu*v*p.cA01_w1+u*v*p.cA11_w1 });
      il.$.stateB[ci] = SculptureStateB({ theta2: omu*omv*p.cB00_t2+u*omv*p.cB10_t2+omu*v*p.cB01_t2+u*v*p.cB11_t2, omega2: omu*omv*p.cB00_w2+u*omv*p.cB10_w2+omu*v*p.cB01_w2+u*v*p.cB11_w2, theta3: omu*omv*p.cB00_t3+u*omv*p.cB10_t3+omu*v*p.cB01_t3+u*v*p.cB11_t3, omega3: omu*omv*p.cB00_w3+u*omv*p.cB10_w3+omu*v*p.cB01_w3+u*v*p.cB11_w3 });
    };
    const stepCell = (ci: number) => {
      'use gpu';
      const sa = sl.$.currentA[ci]; const sb = sl.$.currentB[ci]; const p = sl.$.params;
      const n = rk4Step(sa.theta0, sa.omega0, sa.theta1, sa.omega1, sb.theta2, sb.omega2, sb.theta3, sb.omega3, p.scM0, p.scL0, p.scA0, p.scR, p.scN, p.dt);
      sl.$.nextA[ci] = SculptureStateA({ theta0: n.t0, omega0: n.w0, theta1: n.t1, omega1: n.w1 });
      sl.$.nextB[ci] = SculptureStateB({ theta2: n.t2, omega2: n.w2, theta3: n.t3, omega3: n.w3 });
    };
    const accumulateCell = (ci: number) => {
      'use gpu';
      const sa = al.$.currentA[ci]; const sb = al.$.currentB[ci]; const data = al.$.data[ci]; const p = al.$.params;
      const tip = computeSculptureTip(sa.theta0, sa.theta1, sb.theta2, sb.theta3, p.scA0, p.scL0, p.scR, p.scN);
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
      divData: { storage: this.DataArray, access: 'mutable' }, params: { uniform: SculptureDivParams },
    });
    const divStepLayout = tgpu.bindGroupLayout({
      baseCurrentA: { storage: this.StateArrayA }, baseCurrentB: { storage: this.StateArrayB },
      pertCurrentA: { storage: this.StateArrayA }, pertCurrentB: { storage: this.StateArrayB },
      baseNextA: { storage: this.StateArrayA, access: 'mutable' }, baseNextB: { storage: this.StateArrayB, access: 'mutable' },
      pertNextA: { storage: this.StateArrayA, access: 'mutable' }, pertNextB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' }, params: { uniform: SculptureDivParams },
    });
    const dil = divInitLayout; const dsl = divStepLayout;

    const divRk4 = (t0: number, w0: number, t1: number, w1: number, t2: number, w2: number, t3: number, w3: number, sM: number, sL: number, sA: number, sR: number, sN: number, dt: number) => {
      'use gpu';
      const h = 0.5 * dt;
      const d1 = systemDeriv(t0, w0, t1, w1, t2, w2, t3, w3, sM, sL, sA, sR, sN);
      const d2 = systemDeriv(t0+h*d1.dt0, w0+h*d1.dw0, t1+h*d1.dt1, w1+h*d1.dw1, t2+h*d1.dt2, w2+h*d1.dw2, t3+h*d1.dt3, w3+h*d1.dw3, sM, sL, sA, sR, sN);
      const d3 = systemDeriv(t0+h*d2.dt0, w0+h*d2.dw0, t1+h*d2.dt1, w1+h*d2.dw1, t2+h*d2.dt2, w2+h*d2.dw2, t3+h*d2.dt3, w3+h*d2.dw3, sM, sL, sA, sR, sN);
      const d4 = systemDeriv(t0+dt*d3.dt0, w0+dt*d3.dw0, t1+dt*d3.dt1, w1+dt*d3.dw1, t2+dt*d3.dt2, w2+dt*d3.dw2, t3+dt*d3.dt3, w3+dt*d3.dw3, sM, sL, sA, sR, sN);
      const s6 = dt / 6.0;
      return {
        t0: t0+s6*(d1.dt0+2*d2.dt0+2*d3.dt0+d4.dt0), w0: w0+s6*(d1.dw0+2*d2.dw0+2*d3.dw0+d4.dw0),
        t1: t1+s6*(d1.dt1+2*d2.dt1+2*d3.dt1+d4.dt1), w1: w1+s6*(d1.dw1+2*d2.dw1+2*d3.dw1+d4.dw1),
        t2: t2+s6*(d1.dt2+2*d2.dt2+2*d3.dt2+d4.dt2), w2: w2+s6*(d1.dw2+2*d2.dw2+2*d3.dw2+d4.dw2),
        t3: t3+s6*(d1.dt3+2*d2.dt3+2*d3.dt3+d4.dt3), w3: w3+s6*(d1.dw3+2*d2.dw3+2*d3.dw3+d4.dw3),
      };
    };

    const divInitCell = (ci: number) => {
      'use gpu';
      const p = dil.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const omu = 1 - u; const omv = 1 - v;
      const t0 = omu*omv*p.cA00_t0+u*omv*p.cA10_t0+omu*v*p.cA01_t0+u*v*p.cA11_t0;
      const w0 = omu*omv*p.cA00_w0+u*omv*p.cA10_w0+omu*v*p.cA01_w0+u*v*p.cA11_w0;
      const t1 = omu*omv*p.cA00_t1+u*omv*p.cA10_t1+omu*v*p.cA01_t1+u*v*p.cA11_t1;
      const ow1 = omu*omv*p.cA00_w1+u*omv*p.cA10_w1+omu*v*p.cA01_w1+u*v*p.cA11_w1;
      const t2 = omu*omv*p.cB00_t2+u*omv*p.cB10_t2+omu*v*p.cB01_t2+u*v*p.cB11_t2;
      const w2 = omu*omv*p.cB00_w2+u*omv*p.cB10_w2+omu*v*p.cB01_w2+u*v*p.cB11_w2;
      const t3 = omu*omv*p.cB00_t3+u*omv*p.cB10_t3+omu*v*p.cB01_t3+u*v*p.cB11_t3;
      const w3 = omu*omv*p.cB00_w3+u*omv*p.cB10_w3+omu*v*p.cB01_w3+u*v*p.cB11_w3;
      dil.$.baseStateA[ci] = SculptureStateA({ theta0: t0, omega0: w0, theta1: t1, omega1: ow1 });
      dil.$.baseStateB[ci] = SculptureStateB({ theta2: t2, omega2: w2, theta3: t3, omega3: w3 });
      const px = d.f32(x) / p.resolution; const py = d.f32(y) / p.resolution;
      const h0 = (hash(px*1000+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      const h2 = (hash(px*1000+100+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      dil.$.pertStateA[ci] = SculptureStateA({ theta0: t0+h0, omega0: w0, theta1: t1, omega1: ow1 });
      dil.$.pertStateB[ci] = SculptureStateB({ theta2: t2+h2, omega2: w2, theta3: t3, omega3: w3 });
      dil.$.divData[ci] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divStepCell = (ci: number) => {
      'use gpu';
      const p = dsl.$.params;
      const bsa = dsl.$.baseCurrentA[ci]; const bsb = dsl.$.baseCurrentB[ci];
      const psa = dsl.$.pertCurrentA[ci]; const psb = dsl.$.pertCurrentB[ci];
      const bn = divRk4(bsa.theta0, bsa.omega0, bsa.theta1, bsa.omega1, bsb.theta2, bsb.omega2, bsb.theta3, bsb.omega3, p.scM0, p.scL0, p.scA0, p.scR, p.scN, p.dt);
      dsl.$.baseNextA[ci] = SculptureStateA({ theta0: bn.t0, omega0: bn.w0, theta1: bn.t1, omega1: bn.w1 });
      dsl.$.baseNextB[ci] = SculptureStateB({ theta2: bn.t2, omega2: bn.w2, theta3: bn.t3, omega3: bn.w3 });
      const pn = divRk4(psa.theta0, psa.omega0, psa.theta1, psa.omega1, psb.theta2, psb.omega2, psb.theta3, psb.omega3, p.scM0, p.scL0, p.scA0, p.scR, p.scN, p.dt);
      dsl.$.pertNextA[ci] = SculptureStateA({ theta0: pn.t0, omega0: pn.w0, theta1: pn.t1, omega1: pn.w1 });
      dsl.$.pertNextB[ci] = SculptureStateB({ theta2: pn.t2, omega2: pn.w2, theta3: pn.t3, omega3: pn.w3 });
      const div = sculptureDivergence(bn.t0, bn.w0, bn.t1, bn.w1, bn.t2, bn.w2, bn.t3, bn.w3, pn.t0, pn.w0, pn.t1, pn.w1, pn.t2, pn.w2, pn.t3, pn.w3, p.scN);
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
    return packScParams(config);
  }

  buildDivParamsData(config: SimulationConfig, seed: number, perturb: number): Record<string, number> & { frameCounter: number } {
    return { ...packScParams(config), seed, perturb, frameCounter: 0 };
  }
}

function packScParams(config: SimulationConfig): Record<string, number> {
  const corners = computeCorners(config);
  const ca = (i: number) => elasticPackA(corners[i]);
  const cb = (i: number) => elasticPackB(corners[i]);
  const ca00 = ca(0); const ca10 = ca(1); const ca01 = ca(2); const ca11 = ca(3);
  const cb00 = cb(0); const cb10 = cb(1); const cb01 = cb(2); const cb11 = cb(3);
  return {
    scM0: config.sculptureWeight, scL0: config.sculptureRod, scA0: config.sculptureAxle,
    scR: config.sculptureReduction, scN: config.sculptureN,
    dt: config.dt, resolution: config.resolution,
    cA00_t0: ca00[0], cA00_w0: ca00[1], cA00_t1: ca00[2], cA00_w1: ca00[3],
    cA10_t0: ca10[0], cA10_w0: ca10[1], cA10_t1: ca10[2], cA10_w1: ca10[3],
    cA01_t0: ca01[0], cA01_w0: ca01[1], cA01_t1: ca01[2], cA01_w1: ca01[3],
    cA11_t0: ca11[0], cA11_w0: ca11[1], cA11_t1: ca11[2], cA11_w1: ca11[3],
    cB00_t2: cb00[0], cB00_w2: cb00[1], cB00_t3: cb00[2], cB00_w3: cb00[3],
    cB10_t2: cb10[0], cB10_w2: cb10[1], cB10_t3: cb10[2], cB10_w3: cb10[3],
    cB01_t2: cb01[0], cB01_w2: cb01[1], cB01_t3: cb01[2], cB01_w3: cb01[3],
    cB11_t2: cb11[0], cB11_w2: cb11[1], cB11_t3: cb11[2], cB11_w3: cb11[3],
  };
}