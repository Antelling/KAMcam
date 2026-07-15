const WARMUP = 300;
const MAX_POINTS = 10000;

export class PoincareSection {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private points: { x: number; y: number }[] = [];
  private keBuf: number[] = [];
  private prevKe = 0;
  private prevSign = 0;
  private prevT1 = 0;
  private prevW1 = 0;
  private threshold = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.w = canvas.width;
    this.h = canvas.height;
  }

  reset() {
    this.points = [];
    this.keBuf = [];
    this.prevKe = 0;
    this.prevSign = 0;
    this.prevT1 = 0;
    this.prevW1 = 0;
    this.threshold = 0;
  }

  addPoint(t1: number, w1: number, t2: number, w2: number, ke: number) {
    this.keBuf.push(ke);
    if (this.keBuf.length > MAX_POINTS) this.keBuf.shift();

    if (this.keBuf.length < WARMUP) {
      this.prevKe = ke;
      this.prevT1 = t1;
      this.prevW1 = w1;
      return;
    }

    if (this.keBuf.length === WARMUP) {
      let sum = 0;
      for (const v of this.keBuf) sum += v;
      this.threshold = sum / this.keBuf.length;
    } else if (this.keBuf.length % 200 === 0) {
      let sum = 0;
      for (const v of this.keBuf) sum += v;
      this.threshold = sum / this.keBuf.length;
    }

    const sign = ke - this.threshold > 0 ? 1 : -1;
    if (this.prevSign !== 0 && sign !== this.prevSign) {
      const denom = ke - this.prevKe;
      if (Math.abs(denom) > 1e-12) {
        const frac = (this.threshold - this.prevKe) / denom;
        const interpT1 = this.prevT1 + frac * (t1 - this.prevT1);
        const interpW1 = this.prevW1 + frac * (w1 - this.prevW1);
        this.points.push({ x: interpT1, y: interpW1 });
        if (this.points.length > MAX_POINTS) this.points.shift();
      }
    }

    this.prevKe = ke;
    this.prevSign = sign;
    this.prevT1 = t1;
    this.prevW1 = w1;
  }

  draw() {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    const pad = { l: 50, r: 10, t: 10, b: 30 };
    const gw = w - pad.l - pad.r;
    const gh = h - pad.t - pad.b;

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    if (this.points.length === 0) {
      ctx.fillStyle = '#444';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Collecting data...', w / 2, h / 2);
      return;
    }

    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    for (const p of this.points) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
    const xr = xMax - xMin || 1;
    const yr = yMax - yMin || 1;
    xMin -= xr * 0.05;
    xMax += xr * 0.05;
    yMin -= yr * 0.05;
    yMax += yr * 0.05;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    const xToS = (v: number) => pad.l + ((v - xMin) / xRange) * gw;
    const yToS = (v: number) => pad.t + gh - ((v - yMin) / yRange) * gh;

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    const nGrid = 5;
    for (let i = 0; i <= nGrid; i++) {
      const y = pad.t + (gh * i) / nGrid;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + gw, y);
      ctx.stroke();
      const val = yMax - (yRange * i) / nGrid;
      ctx.fillStyle = '#666';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(1), pad.l - 4, y + 3);
    }
    for (let i = 0; i <= nGrid; i++) {
      const x = pad.l + (gw * i) / nGrid;
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + gh);
      ctx.stroke();
      const val = xMin + (xRange * i) / nGrid;
      ctx.fillStyle = '#666';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(val.toFixed(1), x, pad.t + gh + 14);
    }

    ctx.fillStyle = '#00d4aa';
    for (const p of this.points) {
      ctx.beginPath();
      ctx.arc(xToS(p.x), yToS(p.y), 2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('theta1', pad.l + gw / 2, h - 2);
    ctx.save();
    ctx.translate(10, pad.t + gh / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('omega1', 0, 0);
    ctx.restore();

    ctx.fillStyle = '#666';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${this.points.length} crossings`, pad.l + 8, pad.t + 14);
  }

  resize(w: number, h: number) {
    this.w = w;
    this.h = h;
    this.canvas.width = w;
    this.canvas.height = h;
  }
}
