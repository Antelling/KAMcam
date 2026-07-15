import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { ResonantStateA, ResonantStateB, ResonantParams, ResonantDivParams, DataCell } from './types';
import { systemDeriv, computeResonantTip, resonantDivergence } from './deriv';
import { hash } from '../shared/hash';
import type { System } from '../System';

export class ResonantSystem implements System {
  readonly key = 'resonant' as const;
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
    this.StateArrayA = d.arrayOf(ResonantStateA, cellCount);
    this.StateArrayB = d.arrayOf(ResonantStateB, cellCount);
    this.DataArray = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(ResonantParams, this.buildParamsData(config)).$usage('uniform');
    const stateAa = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateAb = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateBa = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    const stateBb = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroDataArray()).$usage('storage');

    this.initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ResonantParams },
    });
    this.stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' },
      nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: ResonantParams },
    });
    this.accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: ResonantParams },
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
      initLayout.$.stateA[cellIndex] = ResonantStateA({
        theta0: omu * omv * p.cA00_th + u * omv * p.cA10_th + omu * v * p.cA01_th + u * v * p.cA11_th,
        omega0: omu * omv * p.cA00_om + u * omv * p.cA10_om + omu * v * p.cA01_om + u * v * p.cA11_om,
        dummy1: 0.0,
        dummy2: 0.0,
      });
      initLayout.$.stateB[cellIndex] = ResonantStateB({
        theta1: omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th,
        omega1: omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om,
        dummy1: 0.0,
        dummy2: 0.0,
      });
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const sa = stepLayout.$.currentA[cellIndex];
      const sb = stepLayout.$.currentB[cellIndex];
      const p = stepLayout.$.params;
      const dt = p.dt;
      const hdt = 0.5 * dt;
      const s6 = dt / 6.0;
      const d1 = systemDeriv(sa.theta0, sa.omega0, sb.theta1, sb.omega1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      const k2_th0 = sa.theta0 + hdt * d1.da_th;
      const k2_om0 = sa.omega0 + hdt * d1.da_om;
      const k2_th1 = sb.theta1 + hdt * d1.db_th;
      const k2_om1 = sb.omega1 + hdt * d1.db_om;
      const d2 = systemDeriv(k2_th0, k2_om0, k2_th1, k2_om1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      const k3_th0 = sa.theta0 + hdt * d2.da_th;
      const k3_om0 = sa.omega0 + hdt * d2.da_om;
      const k3_th1 = sb.theta1 + hdt * d2.db_th;
      const k3_om1 = sb.omega1 + hdt * d2.db_om;
      const d3 = systemDeriv(k3_th0, k3_om0, k3_th1, k3_om1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      const k4_th0 = sa.theta0 + dt * d3.da_th;
      const k4_om0 = sa.omega0 + dt * d3.da_om;
      const k4_th1 = sb.theta1 + dt * d3.db_th;
      const k4_om1 = sb.omega1 + dt * d3.db_om;
      const d4 = systemDeriv(k4_th0, k4_om0, k4_th1, k4_om1, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      stepLayout.$.nextA[cellIndex] = ResonantStateA({
        theta0: sa.theta0 + s6 * (d1.da_th + 2.0 * d2.da_th + 2.0 * d3.da_th + d4.da_th),
        omega0: sa.omega0 + s6 * (d1.da_om + 2.0 * d2.da_om + 2.0 * d3.da_om + d4.da_om),
        dummy1: 0.0,
        dummy2: 0.0,
      });
      stepLayout.$.nextB[cellIndex] = ResonantStateB({
        theta1: sb.theta1 + s6 * (d1.db_th + 2.0 * d2.db_th + 2.0 * d3.db_th + d4.db_th),
        omega1: sb.omega1 + s6 * (d1.db_om + 2.0 * d2.db_om + 2.0 * d3.db_om + d4.db_om),
        dummy1: 0.0,
        dummy2: 0.0,
      });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const sa = accumulateLayout.$.currentA[cellIndex];
      const sb = accumulateLayout.$.currentB[cellIndex];
      const data = accumulateLayout.$.data[cellIndex];
      const p = accumulateLayout.$.params;
      const tip = computeResonantTip(sa.theta0, sb.theta1, p.rpL0, p.rpL1, p.rpA0);
      if (data.a > 0.5) {
        const dx = tip.x - data.r;
        const dy = tip.y - data.g;
        const dist = std.sqrt(dx * dx + dy * dy);
        accumulateLayout.$.data[cellIndex] = DataCell({ r: tip.x, g: tip.y, b: data.b + dist, a: 1.0 });
      } else {
        accumulateLayout.$.data[cellIndex] = DataCell({ r: tip.x, g: tip.y, b: 0.0, a: 1.0 });
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
    this.divParamsBuffer = root.createBuffer(ResonantDivParams, this.buildDivParamsData(config, 0, 0)).$usage('uniform');

    this.divInitLayout = tgpu.bindGroupLayout({
      baseStateA: { storage: this.StateArrayA, access: 'mutable' },
      baseStateB: { storage: this.StateArrayB, access: 'mutable' },
      pertStateA: { storage: this.StateArrayA, access: 'mutable' },
      pertStateB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: ResonantDivParams },
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
      params: { uniform: ResonantDivParams },
    });

    const divInitLayout = this.divInitLayout;
    const divStepLayout = this.divStepLayout;

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
      const th0 = omu * omv * p.cA00_th + u * omv * p.cA10_th + omu * v * p.cA01_th + u * v * p.cA11_th;
      const om0 = omu * omv * p.cA00_om + u * omv * p.cA10_om + omu * v * p.cA01_om + u * v * p.cA11_om;
      const th1 = omu * omv * p.cB00_th + u * omv * p.cB10_th + omu * v * p.cB01_th + u * v * p.cB11_th;
      const om1 = omu * omv * p.cB00_om + u * omv * p.cB10_om + omu * v * p.cB01_om + u * v * p.cB11_om;
      divInitLayout.$.baseStateA[cellIndex] = ResonantStateA({ theta0: th0, omega0: om0, dummy1: 0.0, dummy2: 0.0 });
      divInitLayout.$.baseStateB[cellIndex] = ResonantStateB({ theta1: th1, omega1: om1, dummy1: 0.0, dummy2: 0.0 });
      const px = d.f32(x) / res;
      const py = d.f32(y) / res;
      const h0 = (hash(px * 1000 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      const h1 = (hash(px * 1000 + 100 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      divInitLayout.$.pertStateA[cellIndex] = ResonantStateA({ theta0: th0 + h0, omega0: om0, dummy1: 0.0, dummy2: 0.0 });
      divInitLayout.$.pertStateB[cellIndex] = ResonantStateB({ theta1: th1 + h1, omega1: om1, dummy1: 0.0, dummy2: 0.0 });
      divInitLayout.$.divData[cellIndex] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divRk4 = (
      th0: number, om0: number, th1: number, om1: number,
      dt: number, m0: number, m1: number, L0: number, L1: number, a0: number,
    ) => {
      'use gpu';
      const hdt = 0.5 * dt;
      const s6 = dt / 6.0;
      const d1 = systemDeriv(th0, om0, th1, om1, m0, m1, L0, L1, a0);
      const k2_th0 = th0 + hdt * d1.da_th;
      const k2_om0 = om0 + hdt * d1.da_om;
      const k2_th1 = th1 + hdt * d1.db_th;
      const k2_om1 = om1 + hdt * d1.db_om;
      const d2 = systemDeriv(k2_th0, k2_om0, k2_th1, k2_om1, m0, m1, L0, L1, a0);
      const k3_th0 = th0 + hdt * d2.da_th;
      const k3_om0 = om0 + hdt * d2.da_om;
      const k3_th1 = th1 + hdt * d2.db_th;
      const k3_om1 = om1 + hdt * d2.db_om;
      const d3 = systemDeriv(k3_th0, k3_om0, k3_th1, k3_om1, m0, m1, L0, L1, a0);
      const k4_th0 = th0 + dt * d3.da_th;
      const k4_om0 = om0 + dt * d3.da_om;
      const k4_th1 = th1 + dt * d3.db_th;
      const k4_om1 = om1 + dt * d3.db_om;
      const d4 = systemDeriv(k4_th0, k4_om0, k4_th1, k4_om1, m0, m1, L0, L1, a0);
      return {
        th0: th0 + s6 * (d1.da_th + 2.0 * d2.da_th + 2.0 * d3.da_th + d4.da_th),
        om0: om0 + s6 * (d1.da_om + 2.0 * d2.da_om + 2.0 * d3.da_om + d4.da_om),
        th1: th1 + s6 * (d1.db_th + 2.0 * d2.db_th + 2.0 * d3.db_th + d4.db_th),
        om1: om1 + s6 * (d1.db_om + 2.0 * d2.db_om + 2.0 * d3.db_om + d4.db_om),
      };
    };

    const divStepCell = (cellIndex: number) => {
      'use gpu';
      const p = divStepLayout.$.params;
      const dt = p.dt;
      const bsa = divStepLayout.$.baseCurrentA[cellIndex];
      const bsb = divStepLayout.$.baseCurrentB[cellIndex];
      const psa = divStepLayout.$.pertCurrentA[cellIndex];
      const psb = divStepLayout.$.pertCurrentB[cellIndex];
      const bn = divRk4(bsa.theta0, bsa.omega0, bsb.theta1, bsb.omega1, dt, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      divStepLayout.$.baseNextA[cellIndex] = ResonantStateA({ theta0: bn.th0, omega0: bn.om0, dummy1: 0.0, dummy2: 0.0 });
      divStepLayout.$.baseNextB[cellIndex] = ResonantStateB({ theta1: bn.th1, omega1: bn.om1, dummy1: 0.0, dummy2: 0.0 });
      const pn = divRk4(psa.theta0, psa.omega0, psb.theta1, psb.omega1, dt, p.rpM0, p.rpM1, p.rpL0, p.rpL1, p.rpA0);
      divStepLayout.$.pertNextA[cellIndex] = ResonantStateA({ theta0: pn.th0, omega0: pn.om0, dummy1: 0.0, dummy2: 0.0 });
      divStepLayout.$.pertNextB[cellIndex] = ResonantStateB({ theta1: pn.th1, omega1: pn.om1, dummy1: 0.0, dummy2: 0.0 });
      const div = resonantDivergence(bn.th0, bn.om0, bn.th1, bn.om1, pn.th0, pn.om0, pn.th1, pn.om1);
      const data = divStepLayout.$.divData[cellIndex];
      if (div > 1.0 && data.g < 0.5) {
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
    this.divParamsBuffer.write(this.buildDivParamsData(this.lastConfig, seed, perturb));
    this.divFrameCounter = 0;
    this.divInitPipeline.with(this.divInitBG).dispatchThreads(this.cellCount);
    this._readIndex = 0;
  }

  divergenceStep(): void {
    this.divFrameCounter++;
    const data = this.buildDivParamsData(this.lastConfig, 0, 0);
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
      rpM0: config.rpM0,
      rpM1: config.rpM1,
      rpL0: config.rpL0,
      rpL1: config.rpL1,
      rpA0: config.rpA0,
      dt: config.dt,
      resolution: config.resolution,
      cA00_th: cA00[0], cA00_om: cA00[1], cA00_d1: cA00[2], cA00_d2: cA00[3],
      cA10_th: cA10[0], cA10_om: cA10[1], cA10_d1: cA10[2], cA10_d2: cA10[3],
      cA01_th: cA01[0], cA01_om: cA01[1], cA01_d1: cA01[2], cA01_d2: cA01[3],
      cA11_th: cA11[0], cA11_om: cA11[1], cA11_d1: cA11[2], cA11_d2: cA11[3],
      cB00_th: cB00[0], cB00_om: cB00[1], cB00_d1: cB00[2], cB00_d2: cB00[3],
      cB10_th: cB10[0], cB10_om: cB10[1], cB10_d1: cB10[2], cB10_d2: cB10[3],
      cB01_th: cB01[0], cB01_om: cB01[1], cB01_d1: cB01[2], cB01_d2: cB01[3],
      cB11_th: cB11[0], cB11_om: cB11[1], cB11_d1: cB11[2], cB11_d2: cB11[3],
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
      rpM0: config.rpM0,
      rpM1: config.rpM1,
      rpL0: config.rpL0,
      rpL1: config.rpL1,
      rpA0: config.rpA0,
      dt: config.dt,
      resolution: config.resolution,
      seed: seed,
      perturb: perturb,
      frameCounter: 0,
      cA00_th: cA00[0], cA00_om: cA00[1], cA00_d1: cA00[2], cA00_d2: cA00[3],
      cA10_th: cA10[0], cA10_om: cA10[1], cA10_d1: cA10[2], cA10_d2: cA10[3],
      cA01_th: cA01[0], cA01_om: cA01[1], cA01_d1: cA01[2], cA01_d2: cA01[3],
      cA11_th: cA11[0], cA11_om: cA11[1], cA11_d1: cA11[2], cA11_d2: cA11[3],
      cB00_th: cB00[0], cB00_om: cB00[1], cB00_d1: cB00[2], cB00_d2: cB00[3],
      cB10_th: cB10[0], cB10_om: cB10[1], cB10_d1: cB10[2], cB10_d2: cB10[3],
      cB01_th: cB01[0], cB01_om: cB01[1], cB01_d1: cB01[2], cB01_d2: cB01[3],
      cB11_th: cB11[0], cB11_om: cB11[1], cB11_d1: cB11[2], cB11_d2: cB11[3],
    };
  }

  private zeroStateA() {
    return Array.from({ length: this.cellCount }, () => ({ theta0: 0, omega0: 0, dummy1: 0, dummy2: 0 }));
  }

  private zeroStateB() {
    return Array.from({ length: this.cellCount }, () => ({ theta1: 0, omega1: 0, dummy1: 0, dummy2: 0 }));
  }

  private zeroDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }

  private zeroDivDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 1 }));
  }
}
