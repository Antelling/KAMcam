import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { ElasticStateA, ElasticStateB, ElasticParams, DivParams, DataCell } from './types';
import { systemDeriv, computeBob2 } from './deriv';
import { hash } from '../shared/hash';
import { DualStateSystem } from '../shared/DualStateSystem';
import type { GpuPipelinesResult, DivGpuPipelinesResult } from '../shared/DualStateSystem';

export class ElasticSystem extends DualStateSystem {
  readonly key = 'elastic' as const;
  readonly stateSize = 32;

  createStateArrayA(cellCount: number): any { return d.arrayOf(ElasticStateA, cellCount); }
  createStateArrayB(cellCount: number): any { return d.arrayOf(ElasticStateB, cellCount); }
  createDataArray(cellCount: number): any { return d.arrayOf(DataCell, cellCount); }
  paramsStruct(): any { return ElasticParams; }
  divParamsStruct(): any { return DivParams; }

  zeroStateA(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, stretch1: 0, stretchRate1: 0 }));
  }
  zeroStateB(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta2: 0, omega2: 0, stretch2: 0, stretchRate2: 0 }));
  }

  protected buildGpuPipelines(
    root: TgpuRoot,
    stateAa: any, stateAb: any, stateBa: any, stateBb: any,
  ): GpuPipelinesResult {
    const initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ElasticParams },
    });
    const stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' },
      nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ElasticParams },
    });
    const accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: ElasticParams },
    });

    const il = initLayout;
    const sl = stepLayout;
    const al = accumulateLayout;

    const initCell = (cellIndex: number) => {
      'use gpu';
      const p = il.$.params;
      const res = p.resolution;
      const x = cellIndex % d.u32(res);
      const y = cellIndex / d.u32(res);
      const u = d.f32(x) / res;
      const v = d.f32(y) / res;
      const omu = 1.0 - u;
      const omv = 1.0 - v;
      il.$.stateA[cellIndex] = ElasticStateA({
        theta1: omu * omv * p.cA00_th + u * omv * p.cA10_th + omu * v * p.cA01_th + u * v * p.cA11_th,
        omega1: omu * omv * p.cA00_om + u * omv * p.cA10_om + omu * v * p.cA01_om + u * v * p.cA11_om,
        stretch1: omu * omv * p.cA00_r + u * omv * p.cA10_r + omu * v * p.cA01_r + u * v * p.cA11_r,
        stretchRate1: omu * omv * p.cA00_dr + u * omv * p.cA10_dr + omu * v * p.cA01_dr + u * v * p.cA11_dr,
      });
      il.$.stateB[cellIndex] = ElasticStateB({
        theta2: omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th,
        omega2: omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om,
        stretch2: omu * omv * p.cB00_r + u * omv * p.cB10_r + omu * v * p.cB01_r + u * v * p.cB11_r,
        stretchRate2: omu * omv * p.cB00_dr + u * omv * p.cB10_dr + omu * v * p.cB01_dr + u * v * p.cB11_dr,
      });
    };

    const rk4Step = (
      th1: number, om1: number, r1: number, dr1: number,
      th2: number, om2: number, r2: number, dr2: number,
      m1: number, m2: number, L1: number, L2: number, k1: number, k2: number, dt: number,
    ) => {
      'use gpu';
      const hdt = 0.5 * dt;
      const e1 = systemDeriv(th1, om1, r1, dr1, th2, om2, r2, dr2, m1, m2, L1, L2, k1, k2);
      const e2 = systemDeriv(
        th1 + hdt * e1.da_th, om1 + hdt * e1.da_om, r1 + hdt * e1.da_r, dr1 + hdt * e1.da_dr,
        th2 + hdt * e1.db_th, om2 + hdt * e1.db_om, r2 + hdt * e1.db_r, dr2 + hdt * e1.db_dr,
        m1, m2, L1, L2, k1, k2,
      );
      const e3 = systemDeriv(
        th1 + hdt * e2.da_th, om1 + hdt * e2.da_om, r1 + hdt * e2.da_r, dr1 + hdt * e2.da_dr,
        th2 + hdt * e2.db_th, om2 + hdt * e2.db_om, r2 + hdt * e2.db_r, dr2 + hdt * e2.db_dr,
        m1, m2, L1, L2, k1, k2,
      );
      const e4 = systemDeriv(
        th1 + dt * e3.da_th, om1 + dt * e3.da_om, r1 + dt * e3.da_r, dr1 + dt * e3.da_dr,
        th2 + dt * e3.db_th, om2 + dt * e3.db_om, r2 + dt * e3.db_r, dr2 + dt * e3.db_dr,
        m1, m2, L1, L2, k1, k2,
      );
      const s6 = dt / 6.0;
      return {
        th1: th1 + s6 * (e1.da_th + 2.0 * e2.da_th + 2.0 * e3.da_th + e4.da_th),
        om1: om1 + s6 * (e1.da_om + 2.0 * e2.da_om + 2.0 * e3.da_om + e4.da_om),
        r1: r1 + s6 * (e1.da_r + 2.0 * e2.da_r + 2.0 * e3.da_r + e4.da_r),
        dr1: dr1 + s6 * (e1.da_dr + 2.0 * e2.da_dr + 2.0 * e3.da_dr + e4.da_dr),
        th2: th2 + s6 * (e1.db_th + 2.0 * e2.db_th + 2.0 * e3.db_th + e4.db_th),
        om2: om2 + s6 * (e1.db_om + 2.0 * e2.db_om + 2.0 * e3.db_om + e4.db_om),
        r2: r2 + s6 * (e1.db_r + 2.0 * e2.db_r + 2.0 * e3.db_r + e4.db_r),
        dr2: dr2 + s6 * (e1.db_dr + 2.0 * e2.db_dr + 2.0 * e3.db_dr + e4.db_dr),
      };
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const sa = sl.$.currentA[cellIndex];
      const sb = sl.$.currentB[cellIndex];
      const p = sl.$.params;
      const n = rk4Step(sa.theta1, sa.omega1, sa.stretch1, sa.stretchRate1, sb.theta2, sb.omega2, sb.stretch2, sb.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      sl.$.nextA[cellIndex] = ElasticStateA({ theta1: n.th1, omega1: n.om1, stretch1: n.r1, stretchRate1: n.dr1 });
      sl.$.nextB[cellIndex] = ElasticStateB({ theta2: n.th2, omega2: n.om2, stretch2: n.r2, stretchRate2: n.dr2 });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const sa = al.$.currentA[cellIndex];
      const sb = al.$.currentB[cellIndex];
      const data = al.$.data[cellIndex];
      const p = al.$.params;
      const bob = computeBob2(sa.theta1, sb.theta2, p.L1, p.L2, sa.stretch1, sb.stretch2);
      if (data.a > 0.5) {
        const dx = bob.x - data.r;
        const dy = bob.y - data.g;
        const dist = std.sqrt(dx * dx + dy * dy);
        al.$.data[cellIndex] = DataCell({ r: bob.x, g: bob.y, b: data.b + dist, a: 1.0 });
      } else {
        al.$.data[cellIndex] = DataCell({ r: bob.x, g: bob.y, b: 0.0, a: 1.0 });
      }
    };

    return {
      initLayout,
      stepLayout,
      accumulateLayout,
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

  protected buildDivGpuPipelines(
    root: TgpuRoot,
    stateAa: any, stateAb: any, stateBa: any, stateBb: any,
  ): DivGpuPipelinesResult {
    const divInitLayout = tgpu.bindGroupLayout({
      baseStateA: { storage: this.StateArrayA, access: 'mutable' },
      baseStateB: { storage: this.StateArrayB, access: 'mutable' },
      pertStateA: { storage: this.StateArrayA, access: 'mutable' },
      pertStateB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: DivParams },
    });
    const divStepLayout = tgpu.bindGroupLayout({
      baseCurrentA: { storage: this.StateArrayA },
      baseCurrentB: { storage: this.StateArrayB },
      pertCurrentA: { storage: this.StateArrayA },
      pertCurrentB: { storage: this.StateArrayB },
      baseNextA: { storage: this.StateArrayA, access: 'mutable' },
      baseNextB: { storage: this.StateArrayB, access: 'mutable' },
      pertNextA: { storage: this.StateArrayA, access: 'mutable' },
      pertNextB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: DivParams },
    });

    const dil = divInitLayout;
    const dsl = divStepLayout;

    const rk4Step = (
      th1: number, om1: number, r1: number, dr1: number,
      th2: number, om2: number, r2: number, dr2: number,
      m1: number, m2: number, L1: number, L2: number, k1: number, k2: number, dt: number,
    ) => {
      'use gpu';
      const hdt = 0.5 * dt;
      const e1 = systemDeriv(th1, om1, r1, dr1, th2, om2, r2, dr2, m1, m2, L1, L2, k1, k2);
      const e2 = systemDeriv(
        th1 + hdt * e1.da_th, om1 + hdt * e1.da_om, r1 + hdt * e1.da_r, dr1 + hdt * e1.da_dr,
        th2 + hdt * e1.db_th, om2 + hdt * e1.db_om, r2 + hdt * e1.db_r, dr2 + hdt * e1.db_dr,
        m1, m2, L1, L2, k1, k2,
      );
      const e3 = systemDeriv(
        th1 + hdt * e2.da_th, om1 + hdt * e2.da_om, r1 + hdt * e2.da_r, dr1 + hdt * e2.da_dr,
        th2 + hdt * e2.db_th, om2 + hdt * e2.db_om, r2 + hdt * e2.db_r, dr2 + hdt * e2.db_dr,
        m1, m2, L1, L2, k1, k2,
      );
      const e4 = systemDeriv(
        th1 + dt * e3.da_th, om1 + dt * e3.da_om, r1 + dt * e3.da_r, dr1 + dt * e3.da_dr,
        th2 + dt * e3.db_th, om2 + dt * e3.db_om, r2 + dt * e3.db_r, dr2 + dt * e3.db_dr,
        m1, m2, L1, L2, k1, k2,
      );
      const s6 = dt / 6.0;
      return {
        th1: th1 + s6 * (e1.da_th + 2.0 * e2.da_th + 2.0 * e3.da_th + e4.da_th),
        om1: om1 + s6 * (e1.da_om + 2.0 * e2.da_om + 2.0 * e3.da_om + e4.da_om),
        r1: r1 + s6 * (e1.da_r + 2.0 * e2.da_r + 2.0 * e3.da_r + e4.da_r),
        dr1: dr1 + s6 * (e1.da_dr + 2.0 * e2.da_dr + 2.0 * e3.da_dr + e4.da_dr),
        th2: th2 + s6 * (e1.db_th + 2.0 * e2.db_th + 2.0 * e3.db_th + e4.db_th),
        om2: om2 + s6 * (e1.db_om + 2.0 * e2.db_om + 2.0 * e3.db_om + e4.db_om),
        r2: r2 + s6 * (e1.db_r + 2.0 * e2.db_r + 2.0 * e3.db_r + e4.db_r),
        dr2: dr2 + s6 * (e1.db_dr + 2.0 * e2.db_dr + 2.0 * e3.db_dr + e4.db_dr),
      };
    };

    const TWO_PI = 2 * std.acos(0);
    const circDiff = (a: number) => {
      'use gpu';
      return a - std.floor(a / TWO_PI + 0.5) * TWO_PI;
    };

    const divInitCell = (cellIndex: number) => {
      'use gpu';
      const p = dil.$.params;
      const res = p.resolution;
      const x = cellIndex % d.u32(res);
      const y = cellIndex / d.u32(res);
      const u = d.f32(x) / res;
      const v = d.f32(y) / res;
      const omu = 1.0 - u;
      const omv = 1.0 - v;
      const bth1 = omu * omv * p.cA00_th + u * omv * p.cA10_th + omu * v * p.cA01_th + u * v * p.cA11_th;
      const bom1 = omu * omv * p.cA00_om + u * omv * p.cA10_om + omu * v * p.cA01_om + u * v * p.cA11_om;
      const br1 = omu * omv * p.cA00_r + u * omv * p.cA10_r + omu * v * p.cA01_r + u * v * p.cA11_r;
      const bdr1 = omu * omv * p.cA00_dr + u * omv * p.cA10_dr + omu * v * p.cA01_dr + u * v * p.cA11_dr;
      dil.$.baseStateA[cellIndex] = ElasticStateA({ theta1: bth1, omega1: bom1, stretch1: br1, stretchRate1: bdr1 });
      const bth2 = omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th;
      const bom2 = omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om;
      const br2 = omu * omv * p.cB00_r + u * omv * p.cB10_r + omu * v * p.cB01_r + u * v * p.cB11_r;
      const bdr2 = omu * omv * p.cB00_dr + u * omv * p.cB10_dr + omu * v * p.cB01_dr + u * v * p.cB11_dr;
      dil.$.baseStateB[cellIndex] = ElasticStateB({ theta2: bth2, omega2: bom2, stretch2: br2, stretchRate2: bdr2 });
      const px = d.f32(x) / res;
      const py = d.f32(y) / res;
      const h1 = (hash(px * 1000 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      const h2 = (hash(px * 1000 + 100 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      dil.$.pertStateA[cellIndex] = ElasticStateA({ theta1: bth1 + h1, omega1: bom1, stretch1: br1, stretchRate1: bdr1 });
      dil.$.pertStateB[cellIndex] = ElasticStateB({ theta2: bth2 + h2, omega2: bom2, stretch2: br2, stretchRate2: bdr2 });
      dil.$.divData[cellIndex] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divStepCell = (cellIndex: number) => {
      'use gpu';
      const p = dsl.$.params;
      const bA = dsl.$.baseCurrentA[cellIndex];
      const bB = dsl.$.baseCurrentB[cellIndex];
      const pA = dsl.$.pertCurrentA[cellIndex];
      const pB = dsl.$.pertCurrentB[cellIndex];
      const bn = rk4Step(bA.theta1, bA.omega1, bA.stretch1, bA.stretchRate1, bB.theta2, bB.omega2, bB.stretch2, bB.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      const pn = rk4Step(pA.theta1, pA.omega1, pA.stretch1, pA.stretchRate1, pB.theta2, pB.omega2, pB.stretch2, pB.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      dsl.$.baseNextA[cellIndex] = ElasticStateA({ theta1: bn.th1, omega1: bn.om1, stretch1: bn.r1, stretchRate1: bn.dr1 });
      dsl.$.baseNextB[cellIndex] = ElasticStateB({ theta2: bn.th2, omega2: bn.om2, stretch2: bn.r2, stretchRate2: bn.dr2 });
      dsl.$.pertNextA[cellIndex] = ElasticStateA({ theta1: pn.th1, omega1: pn.om1, stretch1: pn.r1, stretchRate1: pn.dr1 });
      dsl.$.pertNextB[cellIndex] = ElasticStateB({ theta2: pn.th2, omega2: pn.om2, stretch2: pn.r2, stretchRate2: pn.dr2 });
      const dt1 = circDiff(bn.th1 - pn.th1);
      const dw1 = bn.om1 - pn.om1;
      const ds1 = bn.r1 - pn.r1;
      const dd1 = bn.dr1 - pn.dr1;
      const dt2 = circDiff(bn.th2 - pn.th2);
      const dw2 = bn.om2 - pn.om2;
      const ds2 = bn.r2 - pn.r2;
      const dd2 = bn.dr2 - pn.dr2;
      const dist = std.sqrt(dt1*dt1 + dw1*dw1 + ds1*ds1 + dd1*dd1 + dt2*dt2 + dw2*dw2 + ds2*ds2 + dd2*dd2);
      const data = dsl.$.divData[cellIndex];
      if (dist > 0.05 && data.g < 0.5) {
        dsl.$.divData[cellIndex] = DataCell({ r: p.frameCounter, g: 1, b: 0, a: 1 });
      }
    };

    return {
      divInitLayout,
      divStepLayout,
      divInitPipeline: root.createGuardedComputePipeline(divInitCell),
      divStepPipeline: root.createGuardedComputePipeline(divStepCell),
      divInitBG: root.createBindGroup(divInitLayout, {
        baseStateA: stateAa, baseStateB: stateBa,
        pertStateA: this.pertStateAa, pertStateB: this.pertStateBa,
        divData: this.divDataBuf, params: this.divParamsBuffer,
      }),
      divStepFwd: root.createBindGroup(divStepLayout, {
        baseCurrentA: stateAa, baseCurrentB: stateBa,
        pertCurrentA: this.pertStateAa, pertCurrentB: this.pertStateBa,
        baseNextA: stateAb, baseNextB: stateBb,
        pertNextA: this.pertStateAb, pertNextB: this.pertStateBb,
        divData: this.divDataBuf, params: this.divParamsBuffer,
      }),
      divStepBwd: root.createBindGroup(divStepLayout, {
        baseCurrentA: stateAb, baseCurrentB: stateBb,
        pertCurrentA: this.pertStateAb, pertCurrentB: this.pertStateBb,
        baseNextA: stateAa, baseNextB: stateBa,
        pertNextA: this.pertStateAa, pertNextB: this.pertStateBa,
        divData: this.divDataBuf, params: this.divParamsBuffer,
      }),
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
  return {
    m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
    k1: config.k1, k2: config.k2, dt: config.dt, resolution: config.resolution,
    cA00_th: ca(0)[0], cA00_om: ca(0)[1], cA00_r: ca(0)[2], cA00_dr: ca(0)[3],
    cA10_th: ca(1)[0], cA10_om: ca(1)[1], cA10_r: ca(1)[2], cA10_dr: ca(1)[3],
    cA01_th: ca(2)[0], cA01_om: ca(2)[1], cA01_r: ca(2)[2], cA01_dr: ca(2)[3],
    cA11_th: ca(3)[0], cA11_om: ca(3)[1], cA11_r: ca(3)[2], cA11_dr: ca(3)[3],
    cB00_th: cb(0)[0], cB00_om: cb(0)[1], cB00_r: cb(0)[2], cB00_dr: cb(0)[3],
    cB10_th: cb(1)[0], cB10_om: cb(1)[1], cB10_r: cb(1)[2], cB10_dr: cb(1)[3],
    cB01_th: cb(2)[0], cB01_om: cb(2)[1], cB01_r: cb(2)[2], cB01_dr: cb(2)[3],
    cB11_th: cb(3)[0], cB11_om: cb(3)[1], cB11_r: cb(3)[2], cB11_dr: cb(3)[3],
  };
}