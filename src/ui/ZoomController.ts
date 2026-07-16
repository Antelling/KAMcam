import type { SimulationConfig, PhaseSpaceAxis } from '../config/schema';

export class ZoomController {
  private history: Array<{ x: PhaseSpaceAxis; y: PhaseSpaceAxis }> = [];

  private homeX: { min: number; max: number };
  private homeY: { min: number; max: number };

  constructor(
    private config: SimulationConfig,
    private onChange: () => void,
  ) {
    this.homeX = { min: config.phaseSpace.x.min, max: config.phaseSpace.x.max };
    this.homeY = { min: config.phaseSpace.y.min, max: config.phaseSpace.y.max };
  }

  get level(): number {
    return this.history.length + 1;
  }

  applyRectangle(
    sx: number, sy: number, ex: number, ey: number,
    cw: number, ch: number,
  ): void {
    const nx1 = Math.max(0, Math.min(1, Math.min(sx, ex) / cw));
    const nx2 = Math.max(0, Math.min(1, Math.max(sx, ex) / cw));
    const ny1 = Math.max(0, Math.min(1, 1 - Math.max(sy, ey) / ch));
    const ny2 = Math.max(0, Math.min(1, 1 - Math.min(sy, ey) / ch));

    const xRange = this.config.phaseSpace.x.max - this.config.phaseSpace.x.min;
    const yRange = this.config.phaseSpace.y.max - this.config.phaseSpace.y.min;
    const dx1 = this.config.phaseSpace.x.min + nx1 * xRange;
    const dx2 = this.config.phaseSpace.x.min + nx2 * xRange;
    const dy1 = this.config.phaseSpace.y.min + ny1 * yRange;
    const dy2 = this.config.phaseSpace.y.min + ny2 * yRange;

    this.history.push({
      x: { ...this.config.phaseSpace.x },
      y: { ...this.config.phaseSpace.y },
    });

    this.config.phaseSpace.x = {
      dimension: this.config.phaseSpace.x.dimension,
      min: Math.min(dx1, dx2),
      max: Math.max(dx1, dx2),
    };
    this.config.phaseSpace.y = {
      dimension: this.config.phaseSpace.y.dimension,
      min: Math.min(dy1, dy2),
      max: Math.max(dy1, dy2),
    };

    this.onChange();
  }

  zoomOut(): void {
    if (this.history.length > 0) {
      const prev = this.history.pop()!;
      this.config.phaseSpace.x = prev.x;
      this.config.phaseSpace.y = prev.y;
    } else {
      this.config.phaseSpace.x.min = this.homeX.min;
      this.config.phaseSpace.x.max = this.homeX.max;
      this.config.phaseSpace.y.min = this.homeY.min;
      this.config.phaseSpace.y.max = this.homeY.max;
    }
    this.onChange();
  }

  reset(): void {
    this.history = [];
    this.config.phaseSpace.x.min = this.homeX.min;
    this.config.phaseSpace.x.max = this.homeX.max;
    this.config.phaseSpace.y.min = this.homeY.min;
    this.config.phaseSpace.y.max = this.homeY.max;
    this.onChange();
  }
}
