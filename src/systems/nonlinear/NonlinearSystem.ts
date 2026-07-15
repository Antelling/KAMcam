import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { NonlinearStateA, NonlinearStateB, NonlinearParams, DivParams, DataCell } from './types';
import { systemDeriv, computeBob2 } from './deriv';
import { hash } from '../shared/hash';
import type { System } from '../System';

export class NonlinearSystem implements System {
  readonly key = 'nonlinear' as const;
  readonly stateSize = 32;
  private root!: TgpuRoot;
  private cellCount = 0;
  private StateArrayA: any = null;
  private StateArrayB: any = null;
  private DataArray: any = null;
  private initLayout: any = null;
  private stepLayout: any = null;
  private accumulateLayout: any = null;
  private initPipeline: any = null;
  private stepPipeline: any = null;
  private accumulatePipeline: any = null;
  private initBG: any = null;
  private stepFwd: any = null;
  private stepBwd: any = null;
  private accA: any = null;
  private accB: any = null;
  private paramsBuffer: any = null;
  private dataBuf: any = null;
  private _readIndex: 0 | 1 = 0;
  private pertStateAa: any = null;
  private pertStateAb: any = null;
  private pertStateBa: any = null;
  private pertStateBb: any = null;
  private divDataBuf: any = null;
  private divParamsBuffer: any = null;
  private divInitLayout: any = null;
  private divStepLayout: any = null;
  private divInitPipeline: any = null;
  private divStepPipeline: any = null;
  private divInitBG: any = null;
  private divStepFwd: any = null;
  private divStepBwd: any = null;
  private divFrameCounter = 0;
  private lastConfig!: SimulationConfig;

  build(root: TgpuRoot, config: SimulationConfig, cellCount: number): void {
    this.root = root;
    this.cellCount = cellCount;
    this.StateArrayA = d.arrayOf(NonlinearStateA, cellCount);
    this.StateArrayB = d.arrayOf(NonlinearStateB, cellCount);
    this.DataArray = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(NonlinearParams, this.buildParamsData(config)).$usage('uniform');
    const stateAa = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateAb = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateBa = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    const stateBb = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroDataArray()).$usage('storage');

    this.initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });
    this.stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' },
      nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });
    this.accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: NonlinearParams },
    });

    const initLayout = this.initLayout;
    const stepLayout = this.stepLayout;
    const accumulateLayout = this.accumulateLayout;

    const initCell = (cellIndex: number) => {
      'use gpu';
      const p = initLayout.$.params;
      const res = p.resolution;
      const x = cellIndex % d.u32(res);
      const y = cellIndex / d.u32(res);
      const u = d.f32(x) / res;
      const v = d.f32(y) / res;
      const omu = 1.0 - u;
      const omv = 1.0 - v;
      initLayout.$.stateA[cellIndex] = NonlinearStateA({
        theta1: omu * omv * p.cA00_th + u * omv * p.cA10_th + omu * v * p.cA01_th + u * v * p.cA11_th,
        omega1: omu * omv * p.cA00_om + u * omv * p.cA10_om + omu * v * p.cA01_om + u * v * p.cA11_om,
        stretch1: omu * omv * p.cA00_r + u * omv * p.cA10_r + omu * v * p.cA01_r + u * v * p.cA11_r,
        stretchRate1: omu * omv * p.cA00_dr + u * omv * p.cA10_dr + omu * v * p.cA01_dr + u * v * p.cA11_dr,
      });
      initLayout.$.stateB[cellIndex] = NonlinearStateB({
        theta2: omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th,
        omega2: omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om,
        stretch2: omu * omv * p.cB00_r + u * omv * p.cB10_r + omu * v * p.cB01_r + u * v * p.cB11_r,
        stretchRate2: omu * omv * p.cB00_dr + u * omv * p.cB10_dr + omu * v * p.cB01_dr + u * v * p.cB11_dr,
      });
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const sa = stepLayout.$.currentA[cellIndex];
      const sb = stepLayout.$.currentB[cellIndex];
      const p = stepLayout.$.params;
      const dt = p.dt;
      const hdt = 0.5 * dt;
      const d1 = systemDeriv(sa.theta1, sa.omega1, sa.stretch1, sa.stretchRate1, sb.theta2, sb.omega2, sb.stretch2, sb.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2);
      const d2 = systemDeriv(
        sa.theta1 + hdt * d1.da_th, sa.omega1 + hdt * d1.da_om, sa.stretch1 + hdt * d1.da_r, sa.stretchRate1 + hdt * d1.da_dr,
        sb.theta2 + hdt * d1.db_th, sb.omega2 + hdt * d1.db_om, sb.stretch2 + hdt * d1.db_r, sb.stretchRate2 + hdt * d1.db_dr,
        p.m1, p.m2, p.L1, p.L2, p.k1, p.k2
      );
      const d3 = systemDeriv(
        sa.theta1 + hdt * d2.da_th, sa.omega1 + hdt * d2.da_om, sa.stretch1 + hdt * d2.da_r, sa.stretchRate1 + hdt * d2.da_dr,
        sb.theta2 + hdt * d2.db_th, sb.omega2 + hdt * d2.db_om, sb.stretch2 + hdt * d2.db_r, sb.stretchRate2 + hdt * d2.db_dr,
        p.m1, p.m2, p.L1, p.L2, p.k1, p.k2
      );
      const d4 = systemDeriv(
        sa.theta1 + dt * d3.da_th, sa.omega1 + dt * d3.da_om, sa.stretch1 + dt * d3.da_r, sa.stretchRate1 + dt * d3.da_dr,
        sb.theta2 + dt * d3.db_th, sb.omega2 + dt * d3.db_om, sb.stretch2 + dt * d3.db_r, sb.stretchRate2 + dt * d3.db_dr,
        p.m1, p.m2, p.L1, p.L2, p.k1, p.k2
      );
      const s6 = dt / 6.0;
      stepLayout.$.nextA[cellIndex] = NonlinearStateA({
        theta1: sa.theta1 + s6 * (d1.da_th + 2.0 * d2.da_th + 2.0 * d3.da_th + d4.da_th),
        omega1: sa.omega1 + s6 * (d1.da_om + 2.0 * d2.da_om + 2.0 * d3.da_om + d4.da_om),
        stretch1: sa.stretch1 + s6 * (d1.da_r + 2.0 * d2.da_r + 2.0 * d3.da_r + d4.da_r),
        stretchRate1: sa.stretchRate1 + s6 * (d1.da_dr + 2.0 * d2.da_dr + 2.0 * d3.da_dr + d4.da_dr),
      });
      stepLayout.$.nextB[cellIndex] = NonlinearStateB({
        theta2: sb.theta2 + s6 * (d1.db_th + 2.0 * d2.db_th + 2.0 * d3.db_th + d4.db_th),
        omega2: sb.omega2 + s6 * (d1.db_om + 2.0 * d2.db_om + 2.0 * d3.db_om + d4.db_om),
        stretch2: sb.stretch2 + s6 * (d1.db_r + 2.0 * d2.db_r + 2.0 * d3.db_r + d4.db_r),
        stretchRate2: sb.stretchRate2 + s6 * (d1.db_dr + 2.0 * d2.db_dr + 2.0 * d3.db_dr + d4.db_dr),
      });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const sa = accumulateLayout.$.currentA[cellIndex];
      const sb = accumulateLayout.$.currentB[cellIndex];
      const data = accumulateLayout.$.data[cellIndex];
      const p = accumulateLayout.$.params;
      const bob = computeBob2(sa.theta1, sb.theta2, p.L1, p.L2, sa.stretch1, sb.stretch2);
      if (data.a > 0.5) {
        const dx = bob.x - data.r;
        const dy = bob.y - data.g;
        const dist = std.sqrt(dx * dx + dy * dy);
        accumulateLayout.$.data[cellIndex] = DataCell({ r: bob.x, g: bob.y, b: data.b + dist, a: 1.0 });
      } else {
        accumulateLayout.$.data[cellIndex] = DataCell({ r: bob.x, g: bob.y, b: 0.0, a: 1.0 });
      }
    };

    this.initPipeline = root.createGuardedComputePipeline(initCell);
    this.stepPipeline = root.createGuardedComputePipeline(stepCell);
    this.accumulatePipeline = root.createGuardedComputePipeline(accumulateCell);
    this.initBG = root.createBindGroup(this.initLayout, { stateA: stateAa, stateB: stateBa, params: this.paramsBuffer });
    this.stepFwd = root.createBindGroup(this.stepLayout, { currentA: stateAa, currentB: stateBa, nextA: stateAb, nextB: stateBb, params: this.paramsBuffer });
    this.stepBwd = root.createBindGroup(this.stepLayout, { currentA: stateAb, currentB: stateBb, nextA: stateAa, nextB: stateBa, params: this.paramsBuffer });
    this.accA = root.createBindGroup(this.accumulateLayout, { currentA: stateAa, currentB: stateBa, data: this.dataBuf, params: this.paramsBuffer });
    this.accB = root.createBindGroup(this.accumulateLayout, { currentA: stateAb, currentB: stateBb, data: this.dataBuf, params: this.paramsBuffer });
    this._readIndex = 0;

    this.pertStateAa = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    this.pertStateAb = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    this.pertStateBa = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.pertStateBb = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.divDataBuf = root.createBuffer(this.DataArray, this.zeroDivDataArray()).$usage('storage');
    this.divParamsBuffer = root.createBuffer(DivParams, this.buildDivParamsData(config, 0, 0)).$usage('uniform');

    this.divInitLayout = tgpu.bindGroupLayout({
      baseStateA: { storage: this.StateArrayA, access: 'mutable' },
      baseStateB: { storage: this.StateArrayB, access: 'mutable' },
      pertStateA: { storage: this.StateArrayA, access: 'mutable' },
      pertStateB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: DivParams },
    });
    this.divStepLayout = tgpu.bindGroupLayout({
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

    const divInitLayout = this.divInitLayout;
    const divStepLayout = this.divStepLayout;

    const rk4Step = (
      th1: number, om1: number, r1: number, dr1: number,
      th2: number, om2: number, r2: number, dr2: number,
      m1: number, m2: number, L1: number, L2: number, k1: number, k2: number, dt: number
    ) => {
      'use gpu';
      const hdt = 0.5 * dt;
      const e1 = systemDeriv(th1, om1, r1, dr1, th2, om2, r2, dr2, m1, m2, L1, L2, k1, k2);
      const e2 = systemDeriv(
        th1 + hdt * e1.da_th, om1 + hdt * e1.da_om, r1 + hdt * e1.da_r, dr1 + hdt * e1.da_dr,
        th2 + hdt * e1.db_th, om2 + hdt * e1.db_om, r2 + hdt * e1.db_r, dr2 + hdt * e1.db_dr,
        m1, m2, L1, L2, k1, k2
      );
      const e3 = systemDeriv(
        th1 + hdt * e2.da_th, om1 + hdt * e2.da_om, r1 + hdt * e2.da_r, dr1 + hdt * e2.da_dr,
        th2 + hdt * e2.db_th, om2 + hdt * e2.db_om, r2 + hdt * e2.db_r, dr2 + hdt * e2.db_dr,
        m1, m2, L1, L2, k1, k2
      );
      const e4 = systemDeriv(
        th1 + dt * e3.da_th, om1 + dt * e3.da_om, r1 + dt * e3.da_r, dr1 + dt * e3.da_dr,
        th2 + dt * e3.db_th, om2 + dt * e3.db_om, r2 + dt * e3.db_r, dr2 + dt * e3.db_dr,
        m1, m2, L1, L2, k1, k2
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
      const p = divInitLayout.$.params;
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
      divInitLayout.$.baseStateA[cellIndex] = NonlinearStateA({ theta1: bth1, omega1: bom1, stretch1: br1, stretchRate1: bdr1 });
      const bth2 = omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th;
      const bom2 = omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om;
      const br2 = omu * omv * p.cB00_r + u * omv * p.cB10_r + omu * v * p.cB01_r + u * v * p.cB11_r;
      const bdr2 = omu * omv * p.cB00_dr + u * omv * p.cB10_dr + omu * v * p.cB01_dr + u * v * p.cB11_dr;
      divInitLayout.$.baseStateB[cellIndex] = NonlinearStateB({ theta2: bth2, omega2: bom2, stretch2: br2, stretchRate2: bdr2 });
      const px = d.f32(x) / res;
      const py = d.f32(y) / res;
      const h1 = (hash(px * 1000 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      const h2 = (hash(px * 1000 + 100 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      divInitLayout.$.pertStateA[cellIndex] = NonlinearStateA({ theta1: bth1 + h1, omega1: bom1, stretch1: br1, stretchRate1: bdr1 });
      divInitLayout.$.pertStateB[cellIndex] = NonlinearStateB({ theta2: bth2 + h2, omega2: bom2, stretch2: br2, stretchRate2: bdr2 });
      divInitLayout.$.divData[cellIndex] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divStepCell = (cellIndex: number) => {
      'use gpu';
      const p = divStepLayout.$.params;
      const bA = divStepLayout.$.baseCurrentA[cellIndex];
      const bB = divStepLayout.$.baseCurrentB[cellIndex];
      const pA = divStepLayout.$.pertCurrentA[cellIndex];
      const pB = divStepLayout.$.pertCurrentB[cellIndex];
      const bn = rk4Step(bA.theta1, bA.omega1, bA.stretch1, bA.stretchRate1, bB.theta2, bB.omega2, bB.stretch2, bB.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      const pn = rk4Step(pA.theta1, pA.omega1, pA.stretch1, pA.stretchRate1, pB.theta2, pB.omega2, pB.stretch2, pB.stretchRate2, p.m1, p.m2, p.L1, p.L2, p.k1, p.k2, p.dt);
      divStepLayout.$.baseNextA[cellIndex] = NonlinearStateA({ theta1: bn.th1, omega1: bn.om1, stretch1: bn.r1, stretchRate1: bn.dr1 });
      divStepLayout.$.baseNextB[cellIndex] = NonlinearStateB({ theta2: bn.th2, omega2: bn.om2, stretch2: bn.r2, stretchRate2: bn.dr2 });
      divStepLayout.$.pertNextA[cellIndex] = NonlinearStateA({ theta1: pn.th1, omega1: pn.om1, stretch1: pn.r1, stretchRate1: pn.dr1 });
      divStepLayout.$.pertNextB[cellIndex] = NonlinearStateB({ theta2: pn.th2, omega2: pn.om2, stretch2: pn.r2, stretchRate2: pn.dr2 });
      const dt1 = circDiff(bn.th1 - pn.th1);
      const dw1 = bn.om1 - pn.om1;
      const ds1 = bn.r1 - pn.r1;
      const dd1 = bn.dr1 - pn.dr1;
      const dt2 = circDiff(bn.th2 - pn.th2);
      const dw2 = bn.om2 - pn.om2;
      const ds2 = bn.r2 - pn.r2;
      const dd2 = bn.dr2 - pn.dr2;
      const dist = std.sqrt(dt1*dt1 + dw1*dw1 + ds1*ds1 + dd1*dd1 + dt2*dt2 + dw2*dw2 + ds2*ds2 + dd2*dd2);
      const data = divStepLayout.$.divData[cellIndex];
      if (dist > 0.05 && data.g < 0.5) {
        divStepLayout.$.divData[cellIndex] = DataCell({ r: p.frameCounter, g: 1, b: 0, a: 1 });
      }
    };

    this.divInitPipeline = root.createGuardedComputePipeline(divInitCell);
    this.divStepPipeline = root.createGuardedComputePipeline(divStepCell);
    this.divInitBG = root.createBindGroup(this.divInitLayout, {
      baseStateA: stateAa, baseStateB: stateBa,
      pertStateA: this.pertStateAa, pertStateB: this.pertStateBa,
      divData: this.divDataBuf, params: this.divParamsBuffer,
    });
    this.divStepFwd = root.createBindGroup(this.divStepLayout, {
      baseCurrentA: stateAa, baseCurrentB: stateBa,
      pertCurrentA: this.pertStateAa, pertCurrentB: this.pertStateBa,
      baseNextA: stateAb, baseNextB: stateBb,
      pertNextA: this.pertStateAb, pertNextB: this.pertStateBb,
      divData: this.divDataBuf, params: this.divParamsBuffer,
    });
    this.divStepBwd = root.createBindGroup(this.divStepLayout, {
      baseCurrentA: stateAb, baseCurrentB: stateBb,
      pertCurrentA: this.pertStateAb, pertCurrentB: this.pertStateBb,
      baseNextA: stateAa, baseNextB: stateBa,
      pertNextA: this.pertStateAa, pertNextB: this.pertStateBa,
      divData: this.divDataBuf, params: this.divParamsBuffer,
    });
  }

  updateParams(config: SimulationConfig): void {
    this.paramsBuffer.write(this.buildParamsData(config));
  }

  get data() { return this.dataBuf; }
  getReadIndex() { return this._readIndex; }

  initSim(): void {
    this.initPipeline.with(this.initBG).dispatchThreads(this.cellCount);
    this._readIndex = 0;
  }

  stepSim(): void {
    if (this._readIndex === 0) {
      this.stepPipeline.with(this.stepFwd).dispatchThreads(this.cellCount);
    } else {
      this.stepPipeline.with(this.stepBwd).dispatchThreads(this.cellCount);
    }
    this._readIndex = this._readIndex === 0 ? 1 : 0;
  }

  accumulateSim(): void {
    if (this._readIndex === 0) {
      this.accumulatePipeline.with(this.accA).dispatchThreads(this.cellCount);
    } else {
      this.accumulatePipeline.with(this.accB).dispatchThreads(this.cellCount);
    }
  }

  initDivergence(seed: number, perturb: number): void {
    const cfg = this.lastConfig;
    this.divParamsBuffer.write(this.buildDivParamsData(cfg, seed, perturb));
    this.divFrameCounter = 0;
    this.divInitPipeline.with(this.divInitBG).dispatchThreads(this.cellCount);
    this._readIndex = 0;
  }

  divergenceStep(): void {
    this.divFrameCounter++;
    const cfg = this.lastConfig;
    const data = this.buildDivParamsData(cfg, 0, 0);
    data.frameCounter = this.divFrameCounter;
    this.divParamsBuffer.write(data);
    if (this._readIndex === 0) {
      this.divStepPipeline.with(this.divStepFwd).dispatchThreads(this.cellCount);
    } else {
      this.divStepPipeline.with(this.divStepBwd).dispatchThreads(this.cellCount);
    }
    this._readIndex = this._readIndex === 0 ? 1 : 0;
  }

  private buildParamsData(config: SimulationConfig) {
    this.lastConfig = config;
    const corners = computeCorners(config);
    const cA00 = elasticPackA(corners[0]);
    const cA10 = elasticPackA(corners[1]);
    const cA01 = elasticPackA(corners[2]);
    const cA11 = elasticPackA(corners[3]);
    const cB00 = elasticPackB(corners[0]);
    const cB10 = elasticPackB(corners[1]);
    const cB01 = elasticPackB(corners[2]);
    const cB11 = elasticPackB(corners[3]);
    return {
      m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
      k1: config.k1, k2: config.k2, dt: config.dt, resolution: config.resolution,
      cA00_th: cA00[0], cA00_om: cA00[1], cA00_r: cA00[2], cA00_dr: cA00[3],
      cA10_th: cA10[0], cA10_om: cA10[1], cA10_r: cA10[2], cA10_dr: cA10[3],
      cA01_th: cA01[0], cA01_om: cA01[1], cA01_r: cA01[2], cA01_dr: cA01[3],
      cA11_th: cA11[0], cA11_om: cA11[1], cA11_r: cA11[2], cA11_dr: cA11[3],
      cB00_th: cB00[0], cB00_om: cB00[1], cB00_r: cB00[2], cB00_dr: cB00[3],
      cB10_th: cB10[0], cB10_om: cB10[1], cB10_r: cB10[2], cB10_dr: cB10[3],
      cB01_th: cB01[0], cB01_om: cB01[1], cB01_r: cB01[2], cB01_dr: cB01[3],
      cB11_th: cB11[0], cB11_om: cB11[1], cB11_r: cB11[2], cB11_dr: cB11[3],
    };
  }

  private buildDivParamsData(config: SimulationConfig, seed: number, perturb: number) {
    const corners = computeCorners(config);
    const cA00 = elasticPackA(corners[0]);
    const cA10 = elasticPackA(corners[1]);
    const cA01 = elasticPackA(corners[2]);
    const cA11 = elasticPackA(corners[3]);
    const cB00 = elasticPackB(corners[0]);
    const cB10 = elasticPackB(corners[1]);
    const cB01 = elasticPackB(corners[2]);
    const cB11 = elasticPackB(corners[3]);
    return {
      m1: config.m1, m2: config.m2, L1: config.L1, L2: config.L2,
      k1: config.k1, k2: config.k2, dt: config.dt, resolution: config.resolution,
      seed: seed, perturb: perturb, frameCounter: 0,
      cA00_th: cA00[0], cA00_om: cA00[1], cA00_r: cA00[2], cA00_dr: cA00[3],
      cA10_th: cA10[0], cA10_om: cA10[1], cA10_r: cA10[2], cA10_dr: cA10[3],
      cA01_th: cA01[0], cA01_om: cA01[1], cA01_r: cA01[2], cA01_dr: cA01[3],
      cA11_th: cA11[0], cA11_om: cA11[1], cA11_r: cA11[2], cA11_dr: cA11[3],
      cB00_th: cB00[0], cB00_om: cB00[1], cB00_r: cB00[2], cB00_dr: cB00[3],
      cB10_th: cB10[0], cB10_om: cB10[1], cB10_r: cB10[2], cB10_dr: cB10[3],
      cB01_th: cB01[0], cB01_om: cB01[1], cB01_r: cB01[2], cB01_dr: cB01[3],
      cB11_th: cB11[0], cB11_om: cB11[1], cB11_r: cB11[2], cB11_dr: cB11[3],
    };
  }

  private zeroStateA() {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, stretch1: 0, stretchRate1: 0 }));
  }

  private zeroStateB() {
    return Array.from({ length: this.cellCount }, () => ({ theta2: 0, omega2: 0, stretch2: 0, stretchRate2: 0 }));
  }

  private zeroDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }

  private zeroDivDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 1 }));
  }
}
