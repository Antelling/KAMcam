export class AnimationLoop {
  private rafId = 0;
  private running = false;

  start(fn: () => void): void {
    this.running = true;
    const tick = (): void => {
      if (!this.running) return;
      fn();
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  isRunning(): boolean {
    return this.running;
  }
}
