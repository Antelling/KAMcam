import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, rigidPack } from '../../config/corners';
import { RigidState, RigidParams, DivParams, DataCell } from './types';
import { computeAccelerations, computeBob2 } from './deriv';
import { hash } from '../shared/hash';
import { SingleStateSystem } from '../shared/SingleStateSystem';
import type { SingleGpuPipelinesResult, SingleDivGpuPipelinesResult } from '../shared/SingleStateSystem';

export class RigidSystem extends SingleStateSystem {
  readonly key = 'rigid' as const;
  readonly stateSize = 16;

  stateArrayType(cellCount: number): any { return d.arrayOf(RigidState, cellCount); }
  dataArrayType(cellCount: number): any { return d.arrayOf(DataCell, cellCount); }
  paramsStruct(): any { return RigidParams; }
  divParamsStruct(): any { return DivParams; }
  zeroState(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, theta2: 0, omega2: 0 }));
  }

  protected buildGpuPipelines(root: TgpuRoot, stateA: any, stateB: any): SingleGpuPipelinesResult {
    const initLayout = tgpu.bindGroupLayout({
      state: { storage: this.StateArray, access: 'mutable' },
      params: { uniform: RigidParams },
    });
    const stepLayout = tgpu.bindGroupLayout({
      currentState: { storage: this.StateArray },
      nextState: { storage: this.StateArray, access: 'mutable' },
      params: { uniform: RigidParams },
    });
    const accumulateLayout = tgpu.bindGroupLayout({
      currentState: { storage: this.StateArray },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: RigidParams },
    });

    const il = initLayout; const sl = stepLayout; const al = accumulateLayout;

    const initCell = (ci: number) => {
      'use gpu';
      const p = il.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const t1 = (1-u)*(1-v)*p.c00_t1+u*(1-v)*p.c10_t1+(1-u)*v*p.c01_t1+u*v*p.c11_t1;
      const w1 = (1-u)*(1-v)*p.c00_w1+u*(1-v)*p.c10_w1+(1-u)*v*p.c01_w1+u*v*p.c11_w1;
      const t2 = (1-u)*(1-v)*p.c00_t2+u*(1-v)*p.c10_t2+(1-u)*v*p.c01_t2+u*v*p.c11_t2;
      const w2 = (1-u)*(1-v)*p.c00_w2+u*(1-v)*p.c10_w2+(1-u)*v*p.c01_w2+u*v*p.c11_w2;
      il.$.state[ci] = RigidState({ theta1: t1, omega1: w1, theta2: t2, omega2: w2 });
    };

    const stepCell = (ci: number) => {
      'use gpu';
      const s = sl.$.currentState[ci]; const p = sl.$.params; const dt = p.dt;
      const a0 = computeAccelerations(s.theta1, s.omega1, s.theta2, s.omega2, p.m1, p.m2, p.L1, p.L2);
      const hw1 = s.omega1 + 0.5*dt*a0.x; const hw2 = s.omega2 + 0.5*dt*a0.y;
      const nt1 = s.theta1 + dt*hw1; const nt2 = s.theta2 + dt*hw2;
      const a1 = computeAccelerations(nt1, hw1, nt2, hw2, p.m1, p.m2, p.L1, p.L2);
      sl.$.nextState[ci] = RigidState({ theta1: nt1, omega1: hw1+0.5*dt*a1.x, theta2: nt2, omega2: hw2+0.5*dt*a1.y });
    };

    const accumulateCell = (ci: number) => {
      'use gpu';
      const s = al.$.currentState[ci]; const data = al.$.data[ci]; const p = al.$.params;
      const bob = computeBob2(s.theta1, s.theta2, p.L1, p.L2);
      if (data.a > 0.5) { const dx = bob.x-data.r; const dy = bob.y-data.g; const dist = std.sqrt(dx*dx+dy*dy); al.$.data[ci] = DataCell({ r: bob.x, g: bob.y, b: data.b+dist, a: 1 }); }
      else { al.$.data[ci] = DataCell({ r: bob.x, g: bob.y, b: 0, a: 1 }); }
    };

    return {
      initLayout, stepLayout, accumulateLayout,
      initPipeline: root.createGuardedComputePipeline(initCell),
      stepPipeline: root.createGuardedComputePipeline(stepCell),
      accumulatePipeline: root.createGuardedComputePipeline(accumulateCell),
      initBG: root.createBindGroup(initLayout, { state: stateA, params: this.paramsBuffer }),
      stepFwd: root.createBindGroup(stepLayout, { currentState: stateA, nextState: stateB, params: this.paramsBuffer }),
      stepBwd: root.createBindGroup(stepLayout, { currentState: stateB, nextState: stateA, params: this.paramsBuffer }),
      accA: root.createBindGroup(accumulateLayout, { currentState: stateA, data: this.dataBuf, params: this.paramsBuffer }),
      accB: root.createBindGroup(accumulateLayout, { currentState: stateB, data: this.dataBuf, params: this.paramsBuffer }),
    };
  }

  protected buildDivGpuPipelines(root: TgpuRoot, stateA: any, stateB: any): SingleDivGpuPipelinesResult {
    const divInitLayout = tgpu.bindGroupLayout({
      baseState: { storage: this.StateArray, access: 'mutable' },
      pertState: { storage: this.StateArray, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: DivParams },
    });
    const divStepLayout = tgpu.bindGroupLayout({
      baseCurrent: { storage: this.StateArray }, pertCurrent: { storage: this.StateArray },
      baseNext: { storage: this.StateArray, access: 'mutable' }, pertNext: { storage: this.StateArray, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' }, params: { uniform: DivParams },
    });
    const dil = divInitLayout; const dsl = divStepLayout;

    const TWO_PI = 2 * std.acos(0);
    const circDiff = (a: number) => { 'use gpu'; return a - std.floor(a / TWO_PI + 0.5) * TWO_PI; };

    const divInitCell = (ci: number) => {
      'use gpu';
      const p = dil.$.params;
      const x = ci % d.u32(p.resolution); const y = ci / d.u32(p.resolution);
      const u = d.f32(x) / p.resolution; const v = d.f32(y) / p.resolution;
      const t1 = (1-u)*(1-v)*p.c00_t1+u*(1-v)*p.c10_t1+(1-u)*v*p.c01_t1+u*v*p.c11_t1;
      const w1 = (1-u)*(1-v)*p.c00_w1+u*(1-v)*p.c10_w1+(1-u)*v*p.c01_w1+u*v*p.c11_w1;
      const t2 = (1-u)*(1-v)*p.c00_t2+u*(1-v)*p.c10_t2+(1-u)*v*p.c01_t2+u*v*p.c11_t2;
      const w2 = (1-u)*(1-v)*p.c00_w2+u*(1-v)*p.c10_w2+(1-u)*v*p.c01_w2+u*v*p.c11_w2;
      dil.$.baseState[ci] = RigidState({ theta1: t1, omega1: w1, theta2: t2, omega2: w2 });
      const px = d.f32(x) / p.resolution; const py = d.f32(y) / p.resolution;
      const h1 = (hash(px*1000+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      const h2 = (hash(px*1000+100+p.seed, py*1000+p.seed)*2-1)*p.perturb;
      dil.$.pertState[ci] = RigidState({ theta1: t1+h1, omega1: w1, theta2: t2+h2, omega2: w2 });
      dil.$.divData[ci] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divStepCell = (ci: number) => {
      'use gpu';
      const p = dsl.$.params;
      const dt = p.dt;
      const base = dsl.$.baseCurrent[ci];
      const pert = dsl.$.pertCurrent[ci];
      const ba0 = computeAccelerations(base.theta1, base.omega1, base.theta2, base.omega2, p.m1, p.m2, p.L1, p.L2);
      let bw1 = base.omega1 + 0.5 * dt * ba0.x;
      let bw2 = base.omega2 + 0.5 * dt * ba0.y;
      let bt1 = base.theta1 + dt * bw1;
      let bt2 = base.theta2 + dt * bw2;
      const ba1 = computeAccelerations(bt1, bw1, bt2, bw2, p.m1, p.m2, p.L1, p.L2);
      bw1 += 0.5 * dt * ba1.x;
      bw2 += 0.5 * dt * ba1.y;
      dsl.$.baseNext[ci] = RigidState({ theta1: bt1, omega1: bw1, theta2: bt2, omega2: bw2 });
      const pa0 = computeAccelerations(pert.theta1, pert.omega1, pert.theta2, pert.omega2, p.m1, p.m2, p.L1, p.L2);
      let pw1 = pert.omega1 + 0.5 * dt * pa0.x;
      let pw2 = pert.omega2 + 0.5 * dt * pa0.y;
      let pt1 = pert.theta1 + dt * pw1;
      let pt2 = pert.theta2 + dt * pw2;
      const pa1 = computeAccelerations(pt1, pw1, pt2, pw2, p.m1, p.m2, p.L1, p.L2);
      pw1 += 0.5 * dt * pa1.x;
      pw2 += 0.5 * dt * pa1.y;
      dsl.$.pertNext[ci] = RigidState({ theta1: pt1, omega1: pw1, theta2: pt2, omega2: pw2 });
      const da1 = circDiff(bt1 - pt1);
      const dw1 = bw1 - pw1;
      const da2 = circDiff(bt2 - pt2);
      const dw2 = bw2 - pw2;
      const dist = std.sqrt(da1*da1+dw1*dw1+da2*da2+dw2*dw2);
      const data = dsl.$.divData[ci];
      if (dist > 0.05 && data.g < 0.5) { dsl.$.divData[ci] = DataCell({ r: p.frameCounter, g: 1, b: 0, a: 1 }); }
    };

    return {
      divInitLayout, divStepLayout,
      divInitPipeline: root.createGuardedComputePipeline(divInitCell),
      divStepPipeline: root.createGuardedComputePipeline(divStepCell),
      divInitBG: root.createBindGroup(divInitLayout, { baseState: stateA, pertState: this.pertStateA, divData: this.divDataBuf, params: this.divParamsBuffer }),
      divStepFwd: root.createBindGroup(divStepLayout, { baseCurrent: stateA, pertCurrent: this.pertStateA, baseNext: stateB, pertNext: this.pertStateB, divData: this.divDataBuf, params: this.divParamsBuffer }),
      divStepBwd: root.createBindGroup(divStepLayout, { baseCurrent: stateB, pertCurrent: this.pertStateB, baseNext: stateA, pertNext: this.pertStateA, divData: this.divDataBuf, params: this.divParamsBuffer }),
    };
  }

  buildParamsData(config: SimulationConfig): Record<string, number> {
    this.lastConfig = config;
    return packRigidParams(config);
  }

  buildDivParamsData(config: SimulationConfig, seed: number, perturb: number): Record<string, number> & { frameCounter: number } {
    return { ...packRigidParams(config), seed, perturb, frameCounter: 0 };
  }
}

function packRigidParams(config: SimulationConfig): Record<string, number> {
  const corners = computeCorners(config);
  const c = (i: number) => rigidPack(corners[i]);
  return {
    m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
    dt: config.dt, resolution: config.resolution,
    c00_t1: c(0)[0], c00_w1: c(0)[1], c00_t2: c(0)[2], c00_w2: c(0)[3],
    c10_t1: c(1)[0], c10_w1: c(1)[1], c10_t2: c(1)[2], c10_w2: c(1)[3],
    c01_t1: c(2)[0], c01_w1: c(2)[1], c01_t2: c(2)[2], c01_w2: c(2)[3],
    c11_t1: c(3)[0], c11_w1: c(3)[1], c11_t2: c(3)[2], c11_w2: c(3)[3],
  };
}