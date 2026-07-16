export class StatsTracker {
  private lastTime = performance.now();
  private fps = 0;

  update(_isComplete: boolean): void {
    const now = performance.now();
    this.fps = Math.round(1000 / Math.max(1, now - this.lastTime));
    this.lastTime = now;
  }

  getFps(): number {
    return this.fps;
  }
}
