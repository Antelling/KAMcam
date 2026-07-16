import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { NonlinearStateA, NonlinearStateB, NonlinearParams, DivParams, DataCell } from './types';
import { systemDeriv, computeBob2 } from './deriv';
import { hash } from '../shared/hash';
import { DualStateSystem } from '../shared/DualStateSystem';
import type { GpuPipelinesResult, DivGpuPipelinesResult } from '../shared/DualStateSystem';

export class NonlinearSystem extends DualStateSystem {
  readonly key = 'nonlinear' as const;
  readonly stateSize = 32;

  createStateArrayA(cellCount: number): any { return d.arrayOf(NonlinearStateA, cellCount); }
  createStateArrayB(cellCount: number): any { return d.arrayOf(NonlinearStateB, cellCount); }
  createDataArray(cellCount: number): any { return d.arrayOf(DataCell, cellCount); }
  paramsStruct(): any { return NonlinearParams; }
  divParamsStruct(): any { return DivParams; }
  zeroStateA(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, stretch1: 0, stretchRate1: 0 }));
  }
  zeroStateB(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta2: 0, omega2: 0, stretch2: 0, stretchRate2: 0 }));
  }

  protected buildGpuPipelines(root: TgpuRoot, stateAa: any, stateAb: any, stateBa: any, stateBb: any): GpuPipelinesResult {
    const initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });
    const stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA }, currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' }, nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });
    const accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA }, currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });

    const il = initLayout; const sl = stepLayout; const al = accumulateLayout;

    const rk4Step = (th1: number, om1: number, r1: number, dr1: number, th2: number, om2: number, r2: number, dr2: number, m1: number, m2: number, L1: number, L2: number, k1: number, k2: number, dt: number) => {
      'use gpu';
      const hdt = 0.5 * dt;
      const e1 = systemDeriv(th1, om1, r1, dr1, th2, om2, r2, dr2, m1, m2, L1, L2, k1, k2);
      const e2 = systemDeriv(th1+hdt*e1.da_th, om1+hdt*e1.da_om, r1+hdt*e1.da_r, dr1+hdt*e1.da_dr, th2+hdt*e1.db_th, om2+hdt*e1.db_om, r2+hdt*e1.db_r, dr2+hdt*e1.db_dr, m1, m2, L1, L2, k1, k2);
      const e3 = systemDeriv(th1+hdt*e2.da_th, om1+hdt*e2.da_om, r1+hdt*e2.da_r, dr1+hdt*e2.da_dr, th2+hdt*e2.db_th, om2+hdt*e2.db_om, r2+hdt*e2.db_r, dr2+hdt*e2.db_dr, m1, m2, L1, L2, k1, k2);
      const e4 = systemDeriv(th1+dt*e3.da_th, om1+dt*e3.da_om, r1+dt*e3.da_r, dr1+dt*e3.da_dr, th2+dt*e3.db_th, om2+dt*e3.db_om, r2+dt*e3.db_r, dr2+dt*e3.db_dr, m1, m2, L1, L2, k1, k2);
      const s6 = dt / 6.0;
      return { th1: th1+s6*(e1.da_th+2*e2.da_th+2*e3.da_th+e4.da_th), om1: om1+s6*(e1.da_om+2*e2.da_om+2*e3.da_om+e4.da_om), r1: r1+s6*(e1.da_r+2*e2.da_r+2*e3.da_r+e4.da_r), dr1: dr1+s6*(e1.da_dr+2*e2.da_dr+2*e3.da_dr+e4.da_dr), th2: th2+s6*(e1.db_th+2*e2.db_th+2*e3.db_th+e4.db_th), om2: om2+s6*(e1.db_om+2*e2.db_om+2*e3.db_om+e4.db_om), r2: r2+s6*(e1.db_r+2*e2.db_r+2*e3.db_r+e4.db_r), dr2: dr2+s6*(e1.db_dr+2*e2.db_dr+2*e3.db_dr+e4.db_dr) };
    };

    const initCell = (ci: number) => {
      'use gpu';
      const p = il.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const omu = 1 - u; const omv = 1 - v;
      il.$.stateA[ci] = NonlinearStateA({ theta1: omu*omv*p.cA00_th+u*omv*p.cA10_th+omu*v*p.cA01_th+u*v*p.cA11_th, omega1: omu*omv*p.cA00_om+u*omv*p.cA10_om+omu*v*p.cA01_om+u*v*p.cA11_om, stretch1: omu*omv*p.cA00_r+u*omv*p.cA10_r+omu*v*p.cA01_r+u*v*p.cA11_r, stretchRate1: omu*omv*p.cA00_dr+u*omv*p.cA10_dr+omu*v*p.cA01_dr+u*v*p.cA11_dr });
      il.$.stateB[ci] = NonlinearStateB({ theta2: omu*omv*p.cB00_th+u*omv*p.cB10_th+omu*v*p.cB01_th+u*v*p.cB11_th, omega2: omu*omv*p.cB00_om+u*omv*p.cB10_om+omu*v*p.cB01_om+u*v*p.cB11_om, stretch2: omu*omv*p.cB00_r+u*omv*p.cB10_r+omu*v*p.cB01_r+u*v*p.cB11_r, stretchRate2: omu*omv*p.cB00_dr+u*omv*p.cB10_dr+omu*v*p.cB01_dr+u*v*p.cB11_dr });
    };
    const stepCell = (ci: number) => {
      'use gpu';
      const sa = sl.$.currentA[ci]; const sb = sl.$.currentB[ci]; const p = sl.$.params;
      const n = rk4Step(sa.theta1, sa.omega1, sa.stretch1, sa.stretchRate1, sb.theta2, sb.omega2, sb.stretch2, sb.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      sl.$.nextA[ci] = NonlinearStateA({ theta1: n.th1, omega1: n.om1, stretch1: n.r1, stretchRate1: n.dr1 });
      sl.$.nextB[ci] = NonlinearStateB({ theta2: n.th2, omega2: n.om2, stretch2: n.r2, stretchRate2: n.dr2 });
    };
    const accumulateCell = (ci: number) => {
      'use gpu';
      const sa = al.$.currentA[ci]; const sb = al.$.currentB[ci]; const data = al.$.data[ci]; const p = al.$.params;
      const bob = computeBob2(sa.theta1, sb.theta2, p.L1, p.L2, sa.stretch1, sb.stretch2);
      if (data.a > 0.5) { const dx = bob.x - data.r; const dy = bob.y - data.g; const dist = std.sqrt(dx*dx+dy*dy); al.$.data[ci] = DataCell({ r: bob.x, g: bob.y, b: data.b+dist, a: 1 }); }
      else { al.$.data[ci] = DataCell({ r: bob.x, g: bob.y, b: 0, a: 1 }); }
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
      divData: { storage: this.DataArray, access: 'mutable' }, params: { uniform: DivParams },
    });
    const divStepLayout = tgpu.bindGroupLayout({
      baseCurrentA: { storage: this.StateArrayA }, baseCurrentB: { storage: this.StateArrayB },
      pertCurrentA: { storage: this.StateArrayA }, pertCurrentB: { storage: this.StateArrayB },
      baseNextA: { storage: this.StateArrayA, access: 'mutable' }, baseNextB: { storage: this.StateArrayB, access: 'mutable' },
      pertNextA: { storage: this.StateArrayA, access: 'mutable' }, pertNextB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' }, params: { uniform: DivParams },
    });
    const dil = divInitLayout; const dsl = divStepLayout;

    const rk4Step = (th1: number, om1: number, r1: number, dr1: number, th2: number, om2: number, r2: number, dr2: number, m1: number, m2: number, L1: number, L2: number, k1: number, k2: number, dt: number) => {
      'use gpu';
      const hdt = 0.5 * dt;
      const e1 = systemDeriv(th1, om1, r1, dr1, th2, om2, r2, dr2, m1, m2, L1, L2, k1, k2);
      const e2 = systemDeriv(th1+hdt*e1.da_th, om1+hdt*e1.da_om, r1+hdt*e1.da_r, dr1+hdt*e1.da_dr, th2+hdt*e1.db_th, om2+hdt*e1.db_om, r2+hdt*e1.db_r, dr2+hdt*e1.db_dr, m1, m2, L1, L2, k1, k2);
      const e3 = systemDeriv(th1+hdt*e2.da_th, om1+hdt*e2.da_om, r1+hdt*e2.da_r, dr1+hdt*e2.da_dr, th2+hdt*e2.db_th, om2+hdt*e2.db_om, r2+hdt*e2.db_r, dr2+hdt*e2.db_dr, m1, m2, L1, L2, k1, k2);
      const e4 = systemDeriv(th1+dt*e3.da_th, om1+dt*e3.da_om, r1+dt*e3.da_r, dr1+dt*e3.da_dr, th2+dt*e3.db_th, om2+dt*e3.db_om, r2+dt*e3.db_r, dr2+dt*e3.db_dr, m1, m2, L1, L2, k1, k2);
      const s6 = dt / 6.0;
      return { th1: th1+s6*(e1.da_th+2*e2.da_th+2*e3.da_th+e4.da_th), om1: om1+s6*(e1.da_om+2*e2.da_om+2*e3.da_om+e4.da_om), r1: r1+s6*(e1.da_r+2*e2.da_r+2*e3.da_r+e4.da_r), dr1: dr1+s6*(e1.da_dr+2*e2.da_dr+2*e3.da_dr+e4.da_dr), th2: th2+s6*(e1.db_th+2*e2.db_th+2*e3.db_th+e4.db_th), om2: om2+s6*(e1.db_om+2*e2.db_om+2*e3.db_om+e4.db_om), r2: r2+s6*(e1.db_r+2*e2.db_r+2*e3.db_r+e4.db_r), dr2: dr2+s6*(e1.db_dr+2*e2.db_dr+2*e3.db_dr+e4.db_dr) };
    };
    const TWO_PI = 2 * std.acos(0);
    const circDiff = (a: number) => { 'use gpu'; return a - std.floor(a / TWO_PI + 0.5) * TWO_PI; };

    const divInitCell = (ci: number) => {
      'use gpu';
      const p = dil.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const omu = 1 - u; const omv = 1 - v;
      const bt1 = omu*omv*p.cA00_th+u*omv*p.cA10_th+omu*v*p.cA01_th+u*v*p.cA11_th;
      const bo1 = omu*omv*p.cA00_om+u*omv*p.cA10_om+omu*v*p.cA01_om+u*v*p.cA11_om;
      const br1 = omu*omv*p.cA00_r+u*omv*p.cA10_r+omu*v*p.cA01_r+u*v*p.cA11_r;
      const bd1 = omu*omv*p.cA00_dr+u*omv*p.cA10_dr+omu*v*p.cA01_dr+u*v*p.cA11_dr;
      dil.$.baseStateA[ci] = NonlinearStateA({ theta1: bt1, omega1: bo1, stretch1: br1, stretchRate1: bd1 });
      const bt2 = omu*omv*p.cB00_th+u*omv*p.cB10_th+omu*v*p.cB01_th+u*v*p.cB11_th;
      const bo2 = omu*omv*p.cB00_om+u*omv*p.cB10_om+omu*v*p.cB01_om+u*v*p.cB11_om;
      const br2 = omu*omv*p.cB00_r+u*omv*p.cB10_r+omu*v*p.cB01_r+u*v*p.cB11_r;
      const bd2 = omu*omv*p.cB00_dr+u*omv*p.cB10_dr+omu*v*p.cB01_dr+u*v*p.cB11_dr;
      dil.$.baseStateB[ci] = NonlinearStateB({ theta2: bt2, omega2: bo2, stretch2: br2, stretchRate2: bd2 });
      const px = d.f32(x) / p.resolution; const py = d.f32(y) / p.resolution;
      const h1 = (hash(px*1000+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      const h2 = (hash(px*1000+100+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      dil.$.pertStateA[ci] = NonlinearStateA({ theta1: bt1+h1, omega1: bo1, stretch1: br1, stretchRate1: bd1 });
      dil.$.pertStateB[ci] = NonlinearStateB({ theta2: bt2+h2, omega2: bo2, stretch2: br2, stretchRate2: bd2 });
      dil.$.divData[ci] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divStepCell = (ci: number) => {
      'use gpu';
      const p = dsl.$.params;
      const bA = dsl.$.baseCurrentA[ci]; const bB = dsl.$.baseCurrentB[ci];
      const pA = dsl.$.pertCurrentA[ci]; const pB = dsl.$.pertCurrentB[ci];
      const bn = rk4Step(bA.theta1, bA.omega1, bA.stretch1, bA.stretchRate1, bB.theta2, bB.omega2, bB.stretch2, bB.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      const pn = rk4Step(pA.theta1, pA.omega1, pA.stretch1, pA.stretchRate1, pB.theta2, pB.omega2, pB.stretch2, pB.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      dsl.$.baseNextA[ci] = NonlinearStateA({ theta1: bn.th1, omega1: bn.om1, stretch1: bn.r1, stretchRate1: bn.dr1 });
      dsl.$.baseNextB[ci] = NonlinearStateB({ theta2: bn.th2, omega2: bn.om2, stretch2: bn.r2, stretchRate2: bn.dr2 });
      dsl.$.pertNextA[ci] = NonlinearStateA({ theta1: pn.th1, omega1: pn.om1, stretch1: pn.r1, stretchRate1: pn.dr1 });
      dsl.$.pertNextB[ci] = NonlinearStateB({ theta2: pn.th2, omega2: pn.om2, stretch2: pn.r2, stretchRate2: pn.dr2 });
      const dist = std.sqrt(circDiff(bn.th1-pn.th1)**2+(bn.om1-pn.om1)**2+(bn.r1-pn.r1)**2+(bn.dr1-pn.dr1)**2+circDiff(bn.th2-pn.th2)**2+(bn.om2-pn.om2)**2+(bn.r2-pn.r2)**2+(bn.dr2-pn.dr2)**2);
      const data = dsl.$.divData[ci];
      if (dist > 0.05 && data.g < 0.5) { dsl.$.divData[ci] = DataCell({ r: p.frameCounter, g: 1, b: 0, a: 1 }); }
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
    return packCornerParams(config);
  }

  buildDivParamsData(config: SimulationConfig, seed: number, perturb: number): Record<string, number> & { frameCounter: number } {
    return { ...packCornerParams(config), seed, perturb, frameCounter: 0 };
  }
}

function packCornerParams(config: SimulationConfig): Record<string, number> {
  const corners = computeCorners(config);
  const ca = (i: number) => elasticPackA(corners[i]);
  const cb = (i: number) => elasticPackB(corners[i]);
  const ca00 = ca(0); const ca10 = ca(1); const ca01 = ca(2); const ca11 = ca(3);
  const cb00 = cb(0); const cb10 = cb(1); const cb01 = cb(2); const cb11 = cb(3);
  return {
    m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
    k1: config.k1, k2: config.k2, dt: config.dt, resolution: config.resolution,
    cA00_th: ca00[0], cA00_om: ca00[1], cA00_r: ca00[2], cA00_dr: ca00[3],
    cA10_th: ca10[0], cA10_om: ca10[1], cA10_r: ca10[2], cA10_dr: ca10[3],
    cA01_th: ca01[0], cA01_om: ca01[1], cA01_r: ca01[2], cA01_dr: ca01[3],
    cA11_th: ca11[0], cA11_om: ca11[1], cA11_r: ca11[2], cA11_dr: ca11[3],
    cB00_th: cb00[0], cB00_om: cb00[1], cB00_r: cb00[2], cB00_dr: cb00[3],
    cB10_th: cb10[0], cB10_om: cb10[1], cB10_r: cb10[2], cB10_dr: cb10[3],
    cB01_th: cb01[0], cB01_om: cb01[1], cB01_r: cb01[2], cB01_dr: cb01[3],
    cB11_th: cb11[0], cB11_om: cb11[1], cB11_r: cb11[2], cB11_dr: cb11[3],
  };
}