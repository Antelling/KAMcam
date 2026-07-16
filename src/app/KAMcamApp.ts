import { createContext } from '../gpu/Context';
import { Simulator } from '../simulation/Simulator';
import { DisplayRenderer } from '../render/DisplayRenderer';
import { Reductions } from '../gpu/Reductions';
import type { SimulationConfig, PhaseSpaceDimension } from '../config/schema';
import { DEFAULT_CONFIG } from '../config/defaults';
import { generateTiling } from '../config/corners';
import { systemDimensions, isAngleDim, initialVector } from '../config/dimensions';
import type { PlayState } from './PlayState';
import { AnimationLoop } from './AnimationLoop';
import { Controls } from '../ui/Controls';
import { ZoomController } from '../ui/ZoomController';
import { StatsTracker } from '../ui/StatsTracker';
import { setupAllBindings } from './ControlBindings';
import { downloadBlob } from '../utils/download';
import { PendulumPreview } from '../preview/PendulumPreview';
import { Tiler } from '../simulation/Tiler';

export class KAMcamApp {
  private config: SimulationConfig;
  private controls: Controls;
  private zoomController: ZoomController;
  private stats: StatsTracker;
  private preview: PendulumPreview;
  private loop: AnimationLoop;
  private simulator: Simulator | null = null;
  private tiler: Tiler | null = null;
  private renderer: DisplayRenderer | null = null;
  private reductions: Reductions | null = null;
  private playState: PlayState = 'idle';
  private maxValue = 1;
  private canvasContext: GPUCanvasContext | null = null;
  private root: any = null;
  private isDragging = false;
  private dragStart: { x: number; y: number } | null = null;
  private dragCurrent: { x: number; y: number } | null = null;

  constructor() {
    this.config = { ...DEFAULT_CONFIG };
    this.controls = new Controls();
    this.zoomController = new ZoomController(this.config, () => this.onZoomChange());
    this.stats = new StatsTracker();
    this.loop = new AnimationLoop();
    this.preview = new PendulumPreview(
      document.getElementById('canvas') as HTMLCanvasElement,
      this.config,
    );
  }

  async start(): Promise<void> {
    const canvas = this.controls.el('canvas') as HTMLCanvasElement;
    if (!canvas) throw new Error('Canvas not found');
    const ctx = await createContext(canvas);
    this.root = ctx.root;
    this.canvasContext = ctx.canvasContext;

    this.initSimulation();
    this.setupControls();
    this.setupZoomControls();
    this.syncUI();
    this.renderOnce();
  }

  private initSimulation(): void {
    this.tiler = null;
    if (this.config.phaseSpace.mode === 'tiling') {
      this.tiler = new Tiler(this.root, this.config);
      this.renderer = new DisplayRenderer(this.root, this.tiler.data, this.config.resolution);
    } else {
      this.simulator = new Simulator(this.root, this.config);
      this.simulator.init();
      this.reductions = new Reductions(this.root, this.simulator.data, this.cellCount());
      this.renderer = new DisplayRenderer(this.root, this.simulator.data, this.config.resolution);
    }
  }

  private cellCount(): number { return this.config.resolution * this.config.resolution; }

  private rebuildSimulation(): void {
    this.tiler = null;
    if (this.config.phaseSpace.mode === 'tiling') {
      this.tiler = new Tiler(this.root, this.config);
      this.renderer = new DisplayRenderer(this.root, this.tiler.data, this.config.resolution);
    } else {
      this.simulator = new Simulator(this.root, this.config);
      this.simulator.init();
      this.reductions = new Reductions(this.root, this.simulator.data, this.cellCount());
      this.renderer = new DisplayRenderer(this.root, this.simulator.data, this.config.resolution);
    }
    this.preview.rebuildForConfig(this.config);
    const canvas = this.controls.el('canvas') as HTMLCanvasElement;
    if (canvas) {
      canvas.width = this.config.resolution;
      canvas.height = this.config.resolution;
    }
  }

  private syncUI(): void {
    const ui = this.controls;
    ui.updateModeUI(this.config);
    ui.updateLegend(this.config.colormap);
    ui.updatePendulumParams(this.config);
    ui.updateSculptureParams(this.config);
    ui.updateResonantParams(this.config);
    ui.updatePhaseSpaceInputs(this.config);
    ui.updateIntegrationInputs(this.config);
    ui.updateTilingUI(this.config);
    ui.ensureDistinctDimensions(this.config.phaseSpace.x.dimension, this.config.phaseSpace.y.dimension);
    this.updatePlayButton();
  }

  private markStale(): void {
    if (this.playState === 'idle') return;
    this.playState = 'stale';
    this.updatePlayButton();
  }

  private togglePlay(): void {
    if (this.playState === 'idle' || this.playState === 'stale') {
      this.rebuildSimulation();
      this.playState = 'playing';
      this.startLoop();
    } else if (this.playState === 'playing') {
      this.playState = 'paused';
      this.loop.stop();
    } else if (this.playState === 'paused') {
      this.playState = 'playing';
      this.startLoop();
    } else if (this.playState === 'completed') {
      this.rebuildSimulation();
      this.playState = 'playing';
      this.startLoop();
    }
    this.updatePlayButton();
  }

  private togglePreview(): void {
    const panel = document.getElementById('previewPanel');
    if (!panel) return;
    if (panel.classList.contains('active')) {
      panel.classList.remove('active');
      this.preview.stop();
    } else {
      panel.classList.add('active');
      this.preview.start();
    }
  }

  private startLoop(): void {
    this.loop.start(() => this.tick());
  }

  private tick(): void {
    if (this.tiler) {
      this.tickTiling();
      return;
    }
    if (!this.simulator || !this.renderer) return;
    if (this.playState === 'playing' && !this.simulator.isComplete()) {
      this.simulator.step();
      if (this.simulator.getFrameCount() % 10 === 0) {
        this.reductions!.computeMax().then(max => { this.maxValue = max; });
      }
    }
    this.renderOnce();
    this.stats.update(this.simulator.isComplete());
    this.controls.updateStats(
      this.simulator.getFrameCount(), this.maxValue,
      this.stats.getFps(), this.zoomController.level,
    );
    if (this.playState === 'playing' && this.simulator.isComplete()) {
      this.playState = 'completed';
      this.loop.stop();
      this.updatePlayButton();
    }
  }

  private tickTiling(): void {
    if (!this.tiler || !this.renderer) return;
    if (this.playState === 'playing' && !this.tiler.isComplete()) {
      this.tiler.step();
    }
    this.renderOnce();
    const prog = this.tiler.getProgress();
    this.stats.update(this.tiler.isComplete());
    this.controls.updateStats(prog.current, this.maxValue, this.stats.getFps(), this.zoomController.level);
    if (this.playState === 'playing' && this.tiler.isComplete()) {
      this.playState = 'completed';
      this.loop.stop();
      this.updatePlayButton();
    }
  }

  private renderOnce(): void {
    if (!this.renderer || !this.canvasContext) return;
    const tiler = this.tiler;
    const sim = this.simulator;
    const vmIdx = tiler ? 0 : sim!.getVizModeIndex();
    this.renderer.setParams(
      this.maxValue, this.config.colormap, this.config.toneMapping,
      vmIdx, this.config.resolution,
    );
    this.renderer.render(this.canvasContext);
  }

  private updatePlayButton(): void {
    const btn = this.controls.el('playBtn') as HTMLButtonElement | null;
    if (!btn) return;
    switch (this.playState) {
      case 'idle': btn.textContent = 'Render'; break;
      case 'playing': btn.textContent = 'Pause'; break;
      case 'paused': btn.textContent = 'Resume'; break;
      case 'stale': btn.textContent = 'Rerender'; break;
      case 'completed': btn.textContent = 'Render More'; break;
    }
  }

  private onZoomChange(): void {
    this.controls.updatePhaseSpaceInputs(this.config);
    this.markStale();
  }

  handleSystemChange(): void {
    const avail = systemDimensions(this.config.system, this.config.sculptureN);
    if (!avail.includes(this.config.phaseSpace.x.dimension)) {
      this.config.phaseSpace.x.dimension = 'angle1';
      this.applyAxisDefaults('x');
    }
    if (!avail.includes(this.config.phaseSpace.y.dimension) ||
        this.config.phaseSpace.y.dimension === this.config.phaseSpace.x.dimension) {
      const fallback = avail.find(d => d !== this.config.phaseSpace.x.dimension && isAngleDim(this.config.system, d));
      this.config.phaseSpace.y.dimension = fallback ?? 'angle2';
      this.applyAxisDefaults('y');
    }
  }

  applyAxisDefaults(axis: 'x' | 'y'): void {
    const dim = this.config.phaseSpace[axis].dimension;
    const sys = this.config.system;
    if (isAngleDim(sys, dim)) {
      this.config.phaseSpace[axis].min = -Math.PI;
      this.config.phaseSpace[axis].max = Math.PI;
    } else if (sys !== 'sculpture' && dim.startsWith('stretch') && !dim.includes('Rate')) {
      this.config.phaseSpace[axis].min = -0.5;
      this.config.phaseSpace[axis].max = 0.5;
    } else {
      this.config.phaseSpace[axis].min = -5;
      this.config.phaseSpace[axis].max = 5;
    }
  }

  regenerateTiling(): void {
    const t = this.config.phaseSpace.tiling;
    this.config.phaseSpace.tiling = generateTiling(
      this.config.system, this.config.sculptureN,
      Math.max(1, t.cols), Math.max(1, t.rows),
      initialVector(this.config),
    );
  }

  resetZoomController(): void {
    this.zoomController = new ZoomController(this.config, () => this.onZoomChange());
  }

  private setupControls(): void {
    const ui = this.controls;
    setupAllBindings({
      config: this.config,
      controls: ui,
      zoomController: this.zoomController,
      markStale: () => this.markStale(),
      syncUI: () => this.syncUI(),
      regenerateTiling: () => this.regenerateTiling(),
      handleSystemChange: () => this.handleSystemChange(),
      applyAxisDefaults: (a) => this.applyAxisDefaults(a),
      resetZoom: () => this.resetZoomController(),
    });

    ui.bindButton('playBtn', () => this.togglePlay());
    ui.bindButton('downloadBtn', () => this.download());
    ui.bindButton('previewToggleBtn', () => this.togglePreview());
  }

  private download(): void {
    const canvas = this.controls.el('canvas') as HTMLCanvasElement | null;
    if (!canvas) return;
    const fc = this.tiler ? this.tiler.getProgress().current : (this.simulator?.getFrameCount() ?? 0);
    const filename = `chaos-${this.config.system}-${this.config.vizMode}-frame${fc}.png`;
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, filename);
    }, 'image/png');
  }

  private setupZoomControls(): void {
    const canvas = this.controls.el('canvas') as HTMLCanvasElement;
    const overlay = this.controls.el('zoomOverlay');
    if (!canvas || !overlay) return;

    canvas.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);
      this.isDragging = true;
      this.dragStart = { x, y };
      this.dragCurrent = { x, y };
    });

    canvas.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const rect = canvas.getBoundingClientRect();
      this.dragCurrent = {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height),
      };
      this.updateZoomOverlay(overlay, canvas);
    });

    canvas.addEventListener('mouseup', () => {
      if (!this.isDragging || !this.dragStart || !this.dragCurrent) return;
      const dist = Math.sqrt(
        (this.dragCurrent.x - this.dragStart.x) ** 2 +
        (this.dragCurrent.y - this.dragStart.y) ** 2,
      );
      if (dist > 5) {
        this.zoomController.applyRectangle(
          this.dragStart.x, this.dragStart.y,
          this.dragCurrent.x, this.dragCurrent.y,
          canvas.width, canvas.height,
        );
      }
      this.isDragging = false;
      this.dragStart = null;
      this.dragCurrent = null;
      overlay.style.display = 'none';
    });

    canvas.addEventListener('mouseleave', () => {
      this.isDragging = false;
      this.dragStart = null;
      this.dragCurrent = null;
      overlay.style.display = 'none';
    });

    canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this.zoomController.zoomOut();
    });
  }

  private updateZoomOverlay(overlay: HTMLElement, canvas: HTMLCanvasElement): void {
    if (!this.dragStart || !this.dragCurrent) return;
    const wrapper = this.controls.el('canvasWrapper');
    if (!wrapper) return;
    const wRect = wrapper.getBoundingClientRect();
    const sx = wRect.width / canvas.width;
    const sy = wRect.height / canvas.height;
    const x1 = Math.min(this.dragStart.x, this.dragCurrent.x) * sx;
    const y1 = Math.min(this.dragStart.y, this.dragCurrent.y) * sy;
    const x2 = Math.max(this.dragStart.x, this.dragCurrent.x) * sx;
    const y2 = Math.max(this.dragStart.y, this.dragCurrent.y) * sy;
    overlay.style.display = 'block';
    overlay.style.left = x1 + 'px';
    overlay.style.top = y1 + 'px';
    overlay.style.width = (x2 - x1) + 'px';
    overlay.style.height = (y2 - y1) + 'px';
  }
}
