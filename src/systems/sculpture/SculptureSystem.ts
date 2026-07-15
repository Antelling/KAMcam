import { tgpu, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import { computeCorners, elasticPackA, elasticPackB } from '../../config/corners';
import { SculptureStateA, SculptureStateB, SculptureParams, SculptureDivParams, DataCell } from './types';
import { systemDeriv, computeSculptureTip, sculptureDivergence } from './deriv';
import { hash } from '../shared/hash';
import type { System } from '../System';

export class SculptureSystem implements System {
  readonly key = 'sculpture' as const;
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
    this.StateArrayA = d.arrayOf(SculptureStateA, cellCount);
    this.StateArrayB = d.arrayOf(SculptureStateB, cellCount);
    this.DataArray = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(SculptureParams, this.buildParamsData(config)).$usage('uniform');
    const stateAa = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateAb = root.createBuffer(this.StateArrayA, this.zeroStateA()).$usage('storage');
    const stateBa = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    const stateBb = root.createBuffer(this.StateArrayB, this.zeroStateB()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroDataArray()).$usage('storage');

    this.initLayout = tgpu.bindGroupLayout({
      stateA: { storage: this.StateArrayA, access: 'mutable' },
      stateB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: SculptureParams },
    });
    this.stepLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      nextA: { storage: this.StateArrayA, access: 'mutable' },
      nextB: { storage: this.StateArrayB, access: 'mutable' },
      params: { uniform: SculptureParams },
    });
    this.accumulateLayout = tgpu.bindGroupLayout({
      currentA: { storage: this.StateArrayA },
      currentB: { storage: this.StateArrayB },
      data: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: SculptureParams },
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
      initLayout.$.stateA[cellIndex] = SculptureStateA({
        theta0: omu * omv * p.cA00_t0 + u * omv * p.cA10_t0 + omu * v * p.cA01_t0 + u * v * p.cA11_t0,
        omega0: omu * omv * p.cA00_w0 + u * omv * p.cA10_w0 + omu * v * p.cA01_w0 + u * v * p.cA11_w0,
        theta1: omu * omv * p.cA00_t1 + u * omv * p.cA10_t1 + omu * v * p.cA01_t1 + u * v * p.cA11_t1,
        omega1: omu * omv * p.cA00_w1 + u * omv * p.cA10_w1 + omu * v * p.cA01_w1 + u * v * p.cA11_w1,
      });
      initLayout.$.stateB[cellIndex] = SculptureStateB({
        theta2: omu * omv * p.cB00_t2 + u * omv * p.cB10_t2 + omu * v * p.cB01_t2 + u * v * p.cB11_t2,
        omega2: omu * omv * p.cB00_w2 + u * omv * p.cB10_w2 + omu * v * p.cB01_w2 + u * v * p.cB11_w2,
        theta3: omu * omv * p.cB00_t3 + u * omv * p.cB10_t3 + omu * v * p.cB01_t3 + u * v * p.cB11_t3,
        omega3: omu * omv * p.cB00_w3 + u * omv * p.cB10_w3 + omu * v * p.cB01_w3 + u * v * p.cB11_w3,
      });
    };

    const stepCell = (cellIndex: number) => {
      'use gpu';
      const sa = stepLayout.$.currentA[cellIndex];
      const sb = stepLayout.$.currentB[cellIndex];
      const p = stepLayout.$.params;
      const dt = p.dt;
      const h = 0.5 * dt;
      const sM = p.scM0;
      const sL = p.scL0;
      const sA = p.scA0;
      const sR = p.scR;
      const sN = p.scN;
      const d1 = systemDeriv(sa.theta0, sa.omega0, sa.theta1, sa.omega1, sb.theta2, sb.omega2, sb.theta3, sb.omega3, sM, sL, sA, sR, sN);
      const k2t0 = sa.theta0 + h * d1.dt0;
      const k2w0 = sa.omega0 + h * d1.dw0;
      const k2t1 = sa.theta1 + h * d1.dt1;
      const k2w1 = sa.omega1 + h * d1.dw1;
      const k2t2 = sb.theta2 + h * d1.dt2;
      const k2w2 = sb.omega2 + h * d1.dw2;
      const k2t3 = sb.theta3 + h * d1.dt3;
      const k2w3 = sb.omega3 + h * d1.dw3;
      const d2 = systemDeriv(k2t0, k2w0, k2t1, k2w1, k2t2, k2w2, k2t3, k2w3, sM, sL, sA, sR, sN);
      const k3t0 = sa.theta0 + h * d2.dt0;
      const k3w0 = sa.omega0 + h * d2.dw0;
      const k3t1 = sa.theta1 + h * d2.dt1;
      const k3w1 = sa.omega1 + h * d2.dw1;
      const k3t2 = sb.theta2 + h * d2.dt2;
      const k3w2 = sb.omega2 + h * d2.dw2;
      const k3t3 = sb.theta3 + h * d2.dt3;
      const k3w3 = sb.omega3 + h * d2.dw3;
      const d3 = systemDeriv(k3t0, k3w0, k3t1, k3w1, k3t2, k3w2, k3t3, k3w3, sM, sL, sA, sR, sN);
      const k4t0 = sa.theta0 + dt * d3.dt0;
      const k4w0 = sa.omega0 + dt * d3.dw0;
      const k4t1 = sa.theta1 + dt * d3.dt1;
      const k4w1 = sa.omega1 + dt * d3.dw1;
      const k4t2 = sb.theta2 + dt * d3.dt2;
      const k4w2 = sb.omega2 + dt * d3.dw2;
      const k4t3 = sb.theta3 + dt * d3.dt3;
      const k4w3 = sb.omega3 + dt * d3.dw3;
      const d4 = systemDeriv(k4t0, k4w0, k4t1, k4w1, k4t2, k4w2, k4t3, k4w3, sM, sL, sA, sR, sN);
      const s6 = dt / 6.0;
      stepLayout.$.nextA[cellIndex] = SculptureStateA({
        theta0: sa.theta0 + s6 * (d1.dt0 + 2.0 * d2.dt0 + 2.0 * d3.dt0 + d4.dt0),
        omega0: sa.omega0 + s6 * (d1.dw0 + 2.0 * d2.dw0 + 2.0 * d3.dw0 + d4.dw0),
        theta1: sa.theta1 + s6 * (d1.dt1 + 2.0 * d2.dt1 + 2.0 * d3.dt1 + d4.dt1),
        omega1: sa.omega1 + s6 * (d1.dw1 + 2.0 * d2.dw1 + 2.0 * d3.dw1 + d4.dw1),
      });
      stepLayout.$.nextB[cellIndex] = SculptureStateB({
        theta2: sb.theta2 + s6 * (d1.dt2 + 2.0 * d2.dt2 + 2.0 * d3.dt2 + d4.dt2),
        omega2: sb.omega2 + s6 * (d1.dw2 + 2.0 * d2.dw2 + 2.0 * d3.dw2 + d4.dw2),
        theta3: sb.theta3 + s6 * (d1.dt3 + 2.0 * d2.dt3 + 2.0 * d3.dt3 + d4.dt3),
        omega3: sb.omega3 + s6 * (d1.dw3 + 2.0 * d2.dw3 + 2.0 * d3.dw3 + d4.dw3),
      });
    };

    const accumulateCell = (cellIndex: number) => {
      'use gpu';
      const sa = accumulateLayout.$.currentA[cellIndex];
      const sb = accumulateLayout.$.currentB[cellIndex];
      const data = accumulateLayout.$.data[cellIndex];
      const p = accumulateLayout.$.params;
      const tip = computeSculptureTip(
        sa.theta0, sa.theta1, sb.theta2, sb.theta3,
        p.scA0, p.scL0, p.scR, p.scN,
      );
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
    this.divParamsBuffer = root.createBuffer(SculptureDivParams, this.buildDivParamsData(config, 0, 0)).$usage('uniform');

    this.divInitLayout = tgpu.bindGroupLayout({
      baseStateA: { storage: this.StateArrayA, access: 'mutable' },
      baseStateB: { storage: this.StateArrayB, access: 'mutable' },
      pertStateA: { storage: this.StateArrayA, access: 'mutable' },
      pertStateB: { storage: this.StateArrayB, access: 'mutable' },
      divData: { storage: this.DataArray, access: 'mutable' },
      params: { uniform: SculptureDivParams },
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
      params: { uniform: SculptureDivParams },
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
      const t0 = omu * omv * p.cA00_t0 + u * omv * p.cA10_t0 + omu * v * p.cA01_t0 + u * v * p.cA11_t0;
      const w0 = omu * omv * p.cA00_w0 + u * omv * p.cA10_w0 + omu * v * p.cA01_w0 + u * v * p.cA11_w0;
      const t1 = omu * omv * p.cA00_t1 + u * omv * p.cA10_t1 + omu * v * p.cA01_t1 + u * v * p.cA11_t1;
      const ow1 = omu * omv * p.cA00_w1 + u * omv * p.cA10_w1 + omu * v * p.cA01_w1 + u * v * p.cA11_w1;
      const t2 = omu * omv * p.cB00_t2 + u * omv * p.cB10_t2 + omu * v * p.cB01_t2 + u * v * p.cB11_t2;
      const w2 = omu * omv * p.cB00_w2 + u * omv * p.cB10_w2 + omu * v * p.cB01_w2 + u * v * p.cB11_w2;
      const t3 = omu * omv * p.cB00_t3 + u * omv * p.cB10_t3 + omu * v * p.cB01_t3 + u * v * p.cB11_t3;
      const w3 = omu * omv * p.cB00_w3 + u * omv * p.cB10_w3 + omu * v * p.cB01_w3 + u * v * p.cB11_w3;
      divInitLayout.$.baseStateA[cellIndex] = SculptureStateA({ theta0: t0, omega0: w0, theta1: t1, omega1: ow1 });
      divInitLayout.$.baseStateB[cellIndex] = SculptureStateB({ theta2: t2, omega2: w2, theta3: t3, omega3: w3 });
      const px = d.f32(x) / res;
      const py = d.f32(y) / res;
      const h0 = (hash(px * 1000 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      const h2 = (hash(px * 1000 + 100 + p.seed, py * 1000 + p.seed) * 2 - 1) * p.perturb;
      divInitLayout.$.pertStateA[cellIndex] = SculptureStateA({ theta0: t0 + h0, omega0: w0, theta1: t1, omega1: ow1 });
      divInitLayout.$.pertStateB[cellIndex] = SculptureStateB({ theta2: t2 + h2, omega2: w2, theta3: t3, omega3: w3 });
      divInitLayout.$.divData[cellIndex] = DataCell({ r: 0, g: 0, b: 0, a: 1 });
    };

    const divRk4 = (
      t0: number, w0: number, t1: number, w1: number,
      t2: number, w2: number, t3: number, w3: number,
      dt: number, sM: number, sL: number, sA: number, sR: number, sN: number,
    ) => {
      'use gpu';
      const h = 0.5 * dt;
      const d1 = systemDeriv(t0, w0, t1, w1, t2, w2, t3, w3, sM, sL, sA, sR, sN);
      const k2t0 = t0 + h * d1.dt0;
      const k2w0 = w0 + h * d1.dw0;
      const k2t1 = t1 + h * d1.dt1;
      const k2w1 = w1 + h * d1.dw1;
      const k2t2 = t2 + h * d1.dt2;
      const k2w2 = w2 + h * d1.dw2;
      const k2t3 = t3 + h * d1.dt3;
      const k2w3 = w3 + h * d1.dw3;
      const d2 = systemDeriv(k2t0, k2w0, k2t1, k2w1, k2t2, k2w2, k2t3, k2w3, sM, sL, sA, sR, sN);
      const k3t0 = t0 + h * d2.dt0;
      const k3w0 = w0 + h * d2.dw0;
      const k3t1 = t1 + h * d2.dt1;
      const k3w1 = w1 + h * d2.dw1;
      const k3t2 = t2 + h * d2.dt2;
      const k3w2 = w2 + h * d2.dw2;
      const k3t3 = t3 + h * d2.dt3;
      const k3w3 = w3 + h * d2.dw3;
      const d3 = systemDeriv(k3t0, k3w0, k3t1, k3w1, k3t2, k3w2, k3t3, k3w3, sM, sL, sA, sR, sN);
      const k4t0 = t0 + dt * d3.dt0;
      const k4w0 = w0 + dt * d3.dw0;
      const k4t1 = t1 + dt * d3.dt1;
      const k4w1 = w1 + dt * d3.dw1;
      const k4t2 = t2 + dt * d3.dt2;
      const k4w2 = w2 + dt * d3.dw2;
      const k4t3 = t3 + dt * d3.dt3;
      const k4w3 = w3 + dt * d3.dw3;
      const d4 = systemDeriv(k4t0, k4w0, k4t1, k4w1, k4t2, k4w2, k4t3, k4w3, sM, sL, sA, sR, sN);
      const s6 = dt / 6.0;
      return {
        t0: t0 + s6 * (d1.dt0 + 2.0 * d2.dt0 + 2.0 * d3.dt0 + d4.dt0),
        w0: w0 + s6 * (d1.dw0 + 2.0 * d2.dw0 + 2.0 * d3.dw0 + d4.dw0),
        t1: t1 + s6 * (d1.dt1 + 2.0 * d2.dt1 + 2.0 * d3.dt1 + d4.dt1),
        w1: w1 + s6 * (d1.dw1 + 2.0 * d2.dw1 + 2.0 * d3.dw1 + d4.dw1),
        t2: t2 + s6 * (d1.dt2 + 2.0 * d2.dt2 + 2.0 * d3.dt2 + d4.dt2),
        w2: w2 + s6 * (d1.dw2 + 2.0 * d2.dw2 + 2.0 * d3.dw2 + d4.dw2),
        t3: t3 + s6 * (d1.dt3 + 2.0 * d2.dt3 + 2.0 * d3.dt3 + d4.dt3),
        w3: w3 + s6 * (d1.dw3 + 2.0 * d2.dw3 + 2.0 * d3.dw3 + d4.dw3),
      };
    };

    const divStepCell = (cellIndex: number) => {
      'use gpu';
      const p = divStepLayout.$.params;
      const dt = p.dt;
      const sM = p.scM0;
      const sL = p.scL0;
      const sA = p.scA0;
      const sR = p.scR;
      const sN = p.scN;
      const bsa = divStepLayout.$.baseCurrentA[cellIndex];
      const bsb = divStepLayout.$.baseCurrentB[cellIndex];
      const psa = divStepLayout.$.pertCurrentA[cellIndex];
      const psb = divStepLayout.$.pertCurrentB[cellIndex];
      const bn = divRk4(bsa.theta0, bsa.omega0, bsa.theta1, bsa.omega1, bsb.theta2, bsb.omega2, bsb.theta3, bsb.omega3, dt, sM, sL, sA, sR, sN);
      divStepLayout.$.baseNextA[cellIndex] = SculptureStateA({ theta0: bn.t0, omega0: bn.w0, theta1: bn.t1, omega1: bn.w1 });
      divStepLayout.$.baseNextB[cellIndex] = SculptureStateB({ theta2: bn.t2, omega2: bn.w2, theta3: bn.t3, omega3: bn.w3 });
      const pn = divRk4(psa.theta0, psa.omega0, psa.theta1, psa.omega1, psb.theta2, psb.omega2, psb.theta3, psb.omega3, dt, sM, sL, sA, sR, sN);
      divStepLayout.$.pertNextA[cellIndex] = SculptureStateA({ theta0: pn.t0, omega0: pn.w0, theta1: pn.t1, omega1: pn.w1 });
      divStepLayout.$.pertNextB[cellIndex] = SculptureStateB({ theta2: pn.t2, omega2: pn.w2, theta3: pn.t3, omega3: pn.w3 });
      const div = sculptureDivergence(bn.t0, bn.w0, bn.t1, bn.w1, bn.t2, bn.w2, bn.t3, bn.w3, pn.t0, pn.w0, pn.t1, pn.w1, pn.t2, pn.w2, pn.t3, pn.w3, sN);
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
      scM0: config.sculptureWeight, scL0: config.sculptureRod, scA0: config.sculptureAxle,
      scR: config.sculptureReduction, scN: config.sculptureN,
      dt: config.dt, resolution: config.resolution,
      cA00_t0: cA00[0], cA00_w0: cA00[1], cA00_t1: cA00[2], cA00_w1: cA00[3],
      cA10_t0: cA10[0], cA10_w0: cA10[1], cA10_t1: cA10[2], cA10_w1: cA10[3],
      cA01_t0: cA01[0], cA01_w0: cA01[1], cA01_t1: cA01[2], cA01_w1: cA01[3],
      cA11_t0: cA11[0], cA11_w0: cA11[1], cA11_t1: cA11[2], cA11_w1: cA11[3],
      cB00_t2: cB00[0], cB00_w2: cB00[1], cB00_t3: cB00[2], cB00_w3: cB00[3],
      cB10_t2: cB10[0], cB10_w2: cB10[1], cB10_t3: cB10[2], cB10_w3: cB10[3],
      cB01_t2: cB01[0], cB01_w2: cB01[1], cB01_t3: cB01[2], cB01_w3: cB01[3],
      cB11_t2: cB11[0], cB11_w2: cB11[1], cB11_t3: cB11[2], cB11_w3: cB11[3],
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
      scM0: config.sculptureWeight, scL0: config.sculptureRod, scA0: config.sculptureAxle,
      scR: config.sculptureReduction, scN: config.sculptureN,
      dt: config.dt, resolution: config.resolution,
      seed: seed, perturb: perturb, frameCounter: 0,
      cA00_t0: cA00[0], cA00_w0: cA00[1], cA00_t1: cA00[2], cA00_w1: cA00[3],
      cA10_t0: cA10[0], cA10_w0: cA10[1], cA10_t1: cA10[2], cA10_w1: cA10[3],
      cA01_t0: cA01[0], cA01_w0: cA01[1], cA01_t1: cA01[2], cA01_w1: cA01[3],
      cA11_t0: cA11[0], cA11_w0: cA11[1], cA11_t1: cA11[2], cA11_w1: cA11[3],
      cB00_t2: cB00[0], cB00_w2: cB00[1], cB00_t3: cB00[2], cB00_w3: cB00[3],
      cB10_t2: cB10[0], cB10_w2: cB10[1], cB10_t3: cB10[2], cB10_w3: cB10[3],
      cB01_t2: cB01[0], cB01_w2: cB01[1], cB01_t3: cB01[2], cB01_w3: cB01[3],
      cB11_t2: cB11[0], cB11_w2: cB11[1], cB11_t3: cB11[2], cB11_w3: cB11[3],
    };
  }

  private zeroStateA() {
    return Array.from({ length: this.cellCount }, () => ({ theta0: 0, omega0: 0, theta1: 0, omega1: 0 }));
  }

  private zeroStateB() {
    return Array.from({ length: this.cellCount }, () => ({ theta2: 0, omega2: 0, theta3: 0, omega3: 0 }));
  }

  private zeroDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }

  private zeroDivDataArray() {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 1 }));
  }
}
