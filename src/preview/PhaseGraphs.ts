const COLORS = ['#00d4aa', '#4488ff', '#e8a030', '#ff4444'];
const LABELS = ['theta1', 'omega1', 'theta2', 'omega2'];
const MAX_POINTS = 1000;

export class PhaseGraphs {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;
  private buf: number[][] = [[], [], [], []];
  private legendX = 0;
  private legendY = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.w = canvas.width;
    this.h = canvas.height;
  }

  reset() {
    for (let i = 0; i < 4; i++) this.buf[i] = [];
  }

  addPoint(t1: number, w1: number, t2: number, w2: number) {
    this.buf[0].push(t1);
    this.buf[1].push(w1);
    this.buf[2].push(t2);
    this.buf[3].push(w2);
    for (let i = 0; i < 4; i++) {
      if (this.buf[i].length > MAX_POINTS) this.buf[i].shift();
    }
  }

  draw() {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    const pad = { l: 50, r: 10, t: 10, b: 20 };
    const gw = w - pad.l - pad.r;
    const gh = h - pad.t - pad.b;

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    if (this.buf[0].length < 2) return;

    let allMin = Infinity;
    let allMax = -Infinity;
    for (let i = 0; i < 4; i++) {
      for (const v of this.buf[i]) {
        if (v < allMin) allMin = v;
        if (v > allMax) allMax = v;
      }
    }
    const range = allMax - allMin || 1;
    const yMin = allMin - range * 0.05;
    const yMax = allMax + range * 0.05;
    const yRange = yMax - yMin;

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

    ctx.lineWidth = 1.2;
    for (let i = 0; i < 4; i++) {
      const data = this.buf[i];
      if (data.length < 2) continue;
      ctx.strokeStyle = COLORS[i];
      ctx.beginPath();
      for (let j = 0; j < data.length; j++) {
        const x = pad.l + (j / (MAX_POINTS - 1)) * gw;
        const y = yToS(data[j]);
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    this.legendX = pad.l + 8;
    this.legendY = pad.t + 14;
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';
    for (let i = 0; i < 4; i++) {
      const last = this.buf[i][this.buf[i].length - 1] ?? 0;
      ctx.fillStyle = COLORS[i];
      ctx.fillText(`${LABELS[i]}: ${last.toFixed(3)}`, this.legendX, this.legendY + i * 16);
    }
  }

  resize(w: number, h: number) {
    this.w = w;
    this.h = h;
    this.canvas.width = w;
    this.canvas.height = h;
  }
}
