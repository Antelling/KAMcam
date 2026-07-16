import type { TgpuRoot } from 'typegpu';
import type { SimulationConfig } from '../../config/schema';
import type { System } from '../System';

export abstract class SingleStateSystem implements System {
  abstract readonly key: System['key'];
  abstract readonly stateSize: number;

  protected root!: TgpuRoot;
  protected cellCount = 0;
  protected StateArray: any = null;
  protected DataArray: any = null;

  protected initLayout: any = null;
  protected stepLayout: any = null;
  protected accumulateLayout: any = null;
  protected initPipeline: any = null;
  protected stepPipeline: any = null;
  protected accumulatePipeline: any = null;
  protected initBG: any = null;
  protected stepFwd: any = null;
  protected stepBwd: any = null;
  protected accA: any = null;
  protected accB: any = null;
  protected paramsBuffer: any = null;
  protected dataBuf: any = null;
  protected _readIndex: 0 | 1 = 0;

  protected pertStateA: any = null;
  protected pertStateB: any = null;
  protected divDataBuf: any = null;
  protected divParamsBuffer: any = null;
  protected divInitLayout: any = null;
  protected divStepLayout: any = null;
  protected divInitPipeline: any = null;
  protected divStepPipeline: any = null;
  protected divInitBG: any = null;
  protected divStepFwd: any = null;
  protected divStepBwd: any = null;
  protected divFrameCounter = 0;
  protected lastConfig!: SimulationConfig;

  build(root: TgpuRoot, config: SimulationConfig, cellCount: number): void {
    this.root = root;
    this.cellCount = cellCount;
    this.StateArray = this.stateArrayType(cellCount);
    this.DataArray = this.dataArrayType(cellCount);

    this.paramsBuffer = root.createBuffer(this.paramsStruct(), this.buildParamsData(config)).$usage('uniform');
    const stateA = root.createBuffer(this.StateArray, this.zeroState()).$usage('storage');
    const stateB = root.createBuffer(this.StateArray, this.zeroState()).$usage('storage');
    this.dataBuf = root.createBuffer(this.DataArray, this.zeroData()).$usage('storage');

    const buildResult = this.buildGpuPipelines(root, stateA, stateB);
    this.initLayout = buildResult.initLayout;
    this.stepLayout = buildResult.stepLayout;
    this.accumulateLayout = buildResult.accumulateLayout;
    this.initPipeline = buildResult.initPipeline;
    this.stepPipeline = buildResult.stepPipeline;
    this.accumulatePipeline = buildResult.accumulatePipeline;
    this.initBG = buildResult.initBG;
    this.stepFwd = buildResult.stepFwd;
    this.stepBwd = buildResult.stepBwd;
    this.accA = buildResult.accA;
    this.accB = buildResult.accB;

    this.pertStateA = root.createBuffer(this.StateArray, this.zeroState()).$usage('storage');
    this.pertStateB = root.createBuffer(this.StateArray, this.zeroState()).$usage('storage');
    this.divDataBuf = root.createBuffer(this.DataArray, this.zeroDivData()).$usage('storage');
    this.divParamsBuffer = root.createBuffer(this.divParamsStruct(), this.buildDivParamsData(config, 0, 0)).$usage('uniform');

    const divResult = this.buildDivGpuPipelines(root, stateA, stateB);
    this.divInitLayout = divResult.divInitLayout;
    this.divStepLayout = divResult.divStepLayout;
    this.divInitPipeline = divResult.divInitPipeline;
    this.divStepPipeline = divResult.divStepPipeline;
    this.divInitBG = divResult.divInitBG;
    this.divStepFwd = divResult.divStepFwd;
    this.divStepBwd = divResult.divStepBwd;
    this._readIndex = 0;
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

  protected zeroData(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 0 }));
  }

  protected zeroDivData(): any[] {
    return Array.from({ length: this.cellCount }, () => ({ r: 0, g: 0, b: 0, a: 1 }));
  }

  abstract stateArrayType(cellCount: number): any;
  abstract dataArrayType(cellCount: number): any;
  abstract paramsStruct(): any;
  abstract divParamsStruct(): any;
  abstract zeroState(): any[];

  abstract buildParamsData(config: SimulationConfig): Record<string, number>;
  abstract buildDivParamsData(config: SimulationConfig, seed: number, perturb: number): Record<string, number> & { frameCounter: number };

  protected abstract buildGpuPipelines(root: TgpuRoot, stateA: any, stateB: any): SingleGpuPipelinesResult;
  protected abstract buildDivGpuPipelines(root: TgpuRoot, stateA: any, stateB: any): SingleDivGpuPipelinesResult;
}

export interface SingleGpuPipelinesResult {
  initLayout: any;
  stepLayout: any;
  accumulateLayout: any;
  initPipeline: any;
  stepPipeline: any;
  accumulatePipeline: any;
  initBG: any;
  stepFwd: any;
  stepBwd: any;
  accA: any;
  accB: any;
}

export interface SingleDivGpuPipelinesResult {
  divInitLayout: any;
  divStepLayout: any;
  divInitPipeline: any;
  divStepPipeline: any;
  divInitBG: any;
  divStepFwd: any;
  divStepBwd: any;
}