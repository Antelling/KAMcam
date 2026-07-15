import { createContext } from '../gpu/Context';
import { Simulator } from '../simulation/Simulator';
import { DisplayRenderer } from '../render/DisplayRenderer';
import { Reductions } from '../gpu/Reductions';
import type { SimulationConfig } from '../config/schema';
import { DEFAULT_CONFIG } from '../config/defaults';
import type { PlayState } from './PlayState';

export class KAMcamApp {
  private config: SimulationConfig;
  private simulator: Simulator | null = null;
  private renderer: DisplayRenderer | null = null;
  private reductions: Reductions | null = null;
  private playState: PlayState = 'idle';
  private maxValue = 1;
  private rafId = 0;
  private canvasContext: GPUCanvasContext | null = null;

  constructor() {
    this.config = { ...DEFAULT_CONFIG };
  }

  async start(): Promise<void> {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    if (!canvas) throw new Error('Canvas not found');
    const ctx = await createContext(canvas);
    const root = ctx.root;
    this.canvasContext = ctx.canvasContext;

    this.simulator = new Simulator(root, this.config);
    this.simulator.init();
    this.reductions = new Reductions(root, this.simulator.data, this.config.resolution * this.config.resolution);
    this.renderer = new DisplayRenderer(root, this.simulator.data, this.config.resolution);

    canvas.width = this.config.resolution;
    canvas.height = this.config.resolution;

    const playBtn = document.getElementById('playBtn') as HTMLButtonElement | null;
    if (playBtn) playBtn.addEventListener('click', () => this.togglePlay());

    this.render();
  }

  private togglePlay(): void {
    if (this.playState === 'playing') {
      this.playState = 'paused';
      cancelAnimationFrame(this.rafId);
    } else {
      this.playState = 'playing';
      this.loop();
    }
    this.updateStatus();
  }

  private loop = (): void => {
    if (this.playState !== 'playing') return;
    this.simulator!.step();
    if (this.simulator!.getFrameCount() % 10 === 0) {
      this.reductions!.computeMax().then(max => {
        this.maxValue = max;
        this.render();
      });
    } else {
      this.render();
    }
    if (this.simulator!.getFrameCount() >= this.config.maxIter) {
      this.playState = 'completed';
      this.updateStatus();
      return;
    }
    this.rafId = requestAnimationFrame(this.loop);
  };

  private render(): void {
    this.renderer!.setParams(
      this.maxValue,
      this.config.colormap,
      this.config.toneMapping,
      this.simulator!.getVizModeIndex(),
      this.config.resolution,
    );
    this.renderer!.render(this.canvasContext!);
    this.updateStatus();
  }

  private updateStatus(): void {
    const status = document.getElementById('status');
    if (status) {
      status.textContent = `Frame: ${this.simulator!.getFrameCount()} | Max: ${this.maxValue.toFixed(4)} | ${this.playState}`;
    }
  }
}
