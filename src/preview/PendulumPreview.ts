import type { SimulationConfig } from '../config/schema';
import { computeCorners, bilinearSample } from '../config/corners';
import {
  stepRigid, stepElastic, stepNonlinear, stepSculpture, stepResonant,
  computeBob2, computeSculptureNodes, checkDivergence, calculateEnergies,
  type RigidState, type ElasticStateA, type ElasticStateB,
  type SculptureStateA, type SculptureStateB,
} from './PreviewPhysics';
import { PhaseGraphs } from './PhaseGraphs';
import { PoincareSection } from './PoincareSection';

const TRAIL_LEN = 500;
const SUB_STEPS = 15;

export class PendulumPreview {
  private mainCanvas: HTMLCanvasElement;
  private config: SimulationConfig;
  private corners!: ReturnType<typeof computeCorners>;

  private previewCanvas: HTMLCanvasElement | null = null;
  private graphCanvas: HTMLCanvasElement | null = null;
  private poincareCanvas: HTMLCanvasElement | null = null;
  private phaseGraphs: PhaseGraphs | null = null;
  private poincareSection: PoincareSection | null = null;

  private active = false;
  private playing = true;
  private rafId = 0;

  private baseA: RigidState | ElasticStateA | SculptureStateA = [0, 0, 0, 0];
  private baseB: RigidState | ElasticStateB | SculptureStateB = [0, 0, 0, 0];
  private pertA: RigidState | ElasticStateA | SculptureStateA = [0, 0, 0, 0];
  private pertB: RigidState | ElasticStateB | SculptureStateB = [0, 0, 0, 0];
  private hasState = false;
  private diverged = false;
  private trail: { x: number; y: number }[] = [];

  private onClickBound: ((e: MouseEvent) => void) | null = null;
  private toggleBtn: HTMLButtonElement | null = null;
  private playPauseBtn: HTMLButtonElement | null = null;
  private saveBtn: HTMLButtonElement | null = null;
  private descArea: HTMLTextAreaElement | null = null;
  private controlsDiv: HTMLElement | null = null;
  private statusEl: HTMLElement | null = null;

  constructor(mainCanvas: HTMLCanvasElement, config: SimulationConfig) {
    this.mainCanvas = mainCanvas;
    this.config = config;
    this.updateCorners();
  }

  private updateCorners() {
    this.corners = computeCorners(this.config);
  }

  start() {
    this.active = true;
    this.previewCanvas = document.getElementById('previewCanvas') as HTMLCanvasElement | null;
    this.graphCanvas = document.getElementById('graphCanvas') as HTMLCanvasElement | null;
    this.poincareCanvas = document.getElementById('poincareCanvas') as HTMLCanvasElement | null;

    if (this.previewCanvas) {
      this.phaseGraphs = new PhaseGraphs(this.graphCanvas!);
      this.poincareSection = new PoincareSection(this.poincareCanvas!);
    }

    this.controlsDiv = document.getElementById('playbackControls') as HTMLElement | null;
    this.toggleBtn = document.getElementById('previewToggleBtn') as HTMLButtonElement | null;
    this.playPauseBtn = document.getElementById('playPauseBtn') as HTMLButtonElement | null;
    this.saveBtn = document.getElementById('savePeriodBtn') as HTMLButtonElement | null;
    this.descArea = document.getElementById('pendulumDescription') as HTMLTextAreaElement | null;
    this.statusEl = document.getElementById('previewStatus') as HTMLElement | null;

    if (this.controlsDiv) this.controlsDiv.style.display = '';

    this.toggleBtn?.addEventListener('click', () => this.stop());
    this.playPauseBtn?.addEventListener('click', () => this.togglePlayPause());
    this.saveBtn?.addEventListener('click', () => this.copyDescription());

    this.onClickBound = (e: MouseEvent) => this.onMainCanvasClick(e);
    this.mainCanvas.addEventListener('click', this.onClickBound);

    this.loop();
  }

  stop() {
    this.active = false;
    cancelAnimationFrame(this.rafId);
    this.mainCanvas.removeEventListener('click', this.onClickBound!);
    if (this.controlsDiv) this.controlsDiv.style.display = 'none';
    this.hasState = false;
  }

  rebuildForConfig(config: SimulationConfig) {
    this.config = config;
    this.updateCorners();
    this.hasState = false;
    this.trail = [];
    this.phaseGraphs?.reset();
    this.poincareSection?.reset();
    this.diverged = false;
    this.playing = true;
    if (this.playPauseBtn) this.playPauseBtn.textContent = 'Pause';
    if (this.statusEl) this.statusEl.textContent = 'Click map to sample IC';
  }

  private togglePlayPause() {
    this.playing = !this.playing;
    if (this.playPauseBtn) this.playPauseBtn.textContent = this.playing ? 'Pause' : 'Play';
    if (this.playing && this.active) this.loop();
  }

  private onMainCanvasClick(e: MouseEvent) {
    const rect = this.mainCanvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const vals = bilinearSample(this.corners, nx, ny);
    this.initFromValues(vals);
  }

  private initFromValues(vals: number[]) {
    this.updateCorners();
    this.trail = [];
    this.diverged = false;
    this.hasState = true;
    this.phaseGraphs?.reset();
    this.poincareSection?.reset();
    this.playing = true;
    if (this.playPauseBtn) this.playPauseBtn.textContent = 'Pause';

    const sys = this.config.system;
    const pert = this.config.perturb;

    if (sys === 'rigid') {
      this.baseA = [vals[0], vals[1], vals[4], vals[5]];
      this.pertA = [vals[0] + pert, vals[1], vals[4] + pert, vals[5]];
      this.baseB = this.baseA as any;
      this.pertB = this.pertA as any;
    } else if (sys === 'elastic' || sys === 'nonlinear') {
      this.baseA = [vals[0], vals[1], vals[2], vals[3]];
      this.baseB = [vals[4], vals[5], vals[6], vals[7]];
      this.pertA = [vals[0] + pert, vals[1], vals[2], vals[3]];
      this.pertB = [vals[4] + pert, vals[5], vals[6], vals[7]];
    } else if (sys === 'sculpture') {
      this.baseA = [vals[0], vals[1], vals[2], vals[3]];
      this.baseB = [vals[4], vals[5], vals[6], vals[7]];
      this.pertA = [vals[0] + pert, vals[1], vals[2], vals[3]];
      this.pertB = [vals[4] + pert, vals[5], vals[6], vals[7]];
    } else {
      this.baseA = [vals[0], vals[1], 0, 0];
      this.baseB = [vals[4], vals[5], 0, 0];
      this.pertA = [vals[0] + pert, vals[1], 0, 0];
      this.pertB = [vals[4] + pert, vals[5], 0, 0];
    }

    if (this.descArea) {
      this.descArea.value = JSON.stringify({
        version: 1,
        system: sys,
        params: this.extractParams(),
        initialState: { angle1: vals[0], velocity1: vals[1], angle2: vals[4], velocity2: vals[5] },
      }, null, 2);
    }
  }

  private extractParams() {
    const c = this.config;
    return { m1: c.m1, m2: c.m2, L1: c.L1, L2: c.L2, k1: c.k1, k2: c.k2, dt: c.dt };
  }

  private copyDescription() {
    if (this.descArea) {
      navigator.clipboard.writeText(this.descArea.value);
    }
  }

  private loop = () => {
    if (!this.active || !this.playing) return;
    this.stepFrame();
    this.drawFrame();
    this.rafId = requestAnimationFrame(this.loop);
  };

  private stepFrame() {
    if (!this.hasState) return;
    const dt = this.config.dt;
    const sys = this.config.system;

    for (let i = 0; i < SUB_STEPS; i++) {
      if (sys === 'rigid') {
        this.baseA = stepRigid(this.baseA as RigidState, this.config, dt);
        this.pertA = stepRigid(this.pertA as RigidState, this.config, dt);
      } else if (sys === 'elastic') {
        const [na, nb] = stepElastic(this.baseA as ElasticStateA, this.baseB as ElasticStateB, this.config, dt);
        this.baseA = na;
        this.baseB = nb;
        const [pa, pb] = stepElastic(this.pertA as ElasticStateA, this.pertB as ElasticStateB, this.config, dt);
        this.pertA = pa;
        this.pertB = pb;
      } else if (sys === 'nonlinear') {
        const [na, nb] = stepNonlinear(this.baseA as ElasticStateA, this.baseB as ElasticStateB, this.config, dt);
        this.baseA = na;
        this.baseB = nb;
        const [pa, pb] = stepNonlinear(this.pertA as ElasticStateA, this.pertB as ElasticStateB, this.config, dt);
        this.pertA = pa;
        this.pertB = pb;
      } else if (sys === 'sculpture') {
        const [na, nb] = stepSculpture(this.baseA as SculptureStateA, this.baseB as SculptureStateB, this.config, dt);
        this.baseA = na;
        this.baseB = nb;
        const [pa, pb] = stepSculpture(this.pertA as SculptureStateA, this.pertB as SculptureStateB, this.config, dt);
        this.pertA = pa;
        this.pertB = pb;
      } else {
        const [na, nb] = stepResonant(this.baseA as SculptureStateA, this.baseB as SculptureStateB, this.config, dt);
        this.baseA = na;
        this.baseB = nb;
        const [pa, pb] = stepResonant(this.pertA as SculptureStateA, this.pertB as SculptureStateB, this.config, dt);
        this.pertA = pa;
        this.pertB = pb;
      }

      if (!this.diverged && checkDivergence(this.baseA, this.pertA, sys, this.config)) {
        this.diverged = true;
      }
    }

    const bob = computeBob2(this.baseA, sys, this.config);
    this.trail.push({ x: bob.x, y: bob.y });
    if (this.trail.length > TRAIL_LEN) this.trail.shift();

    const t1 = sys === 'rigid' ? (this.baseA as RigidState)[0] : (this.baseA as ElasticStateA | SculptureStateA)[0];
    const w1 = (this.baseA as any)[1];
    const t2 = sys === 'rigid' ? (this.baseA as RigidState)[2] : (this.baseB as any)[0];
    const w2 = sys === 'rigid' ? (this.baseA as RigidState)[3] : (this.baseB as any)[1];

    this.phaseGraphs?.addPoint(t1, w1, t2, w2);

    const { ke } = calculateEnergies(this.baseA, sys, this.config);
    this.poincareSection?.addPoint(t1, w1, t2, w2, ke);
  }

  private drawFrame() {
    if (!this.previewCanvas || !this.hasState) return;
    const ctx = this.previewCanvas.getContext('2d')!;
    const w = this.previewCanvas.width;
    const h = this.previewCanvas.height;
    const sys = this.config.system;
    const scale = Math.min(w, h) / (2.5 * (this.config.L1 + this.config.L2 + 0.5));

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.5;

    const drawPendulum = (
      a: RigidState | ElasticStateA | SculptureStateA,
      b: RigidState | ElasticStateB | SculptureStateB,
      color: string, ghost: boolean,
    ) => {
      if (sys === 'rigid') {
        const st = a as RigidState;
        const x1 = cx + st[0] * scale * this.config.L1;
        const y1 = cy + st[0] !== st[0] ? cy : cy - st[0] * scale * this.config.L1;
        const rx1 = cx + this.config.L1 * Math.sin(st[0]) * scale;
        const ry1 = cy - this.config.L1 * Math.cos(st[0]) * scale;
        const rx2 = rx1 + this.config.L2 * Math.sin(st[2]) * scale;
        const ry2 = ry1 - this.config.L2 * Math.cos(st[2]) * scale;
        ctx.strokeStyle = color;
        ctx.lineWidth = ghost ? 1 : 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(rx1, ry1);
        ctx.lineTo(rx2, ry2);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(rx2, ry2, ghost ? 3 : 5, 0, Math.PI * 2);
        ctx.fill();
        return { x: rx2, y: ry2 };
      }
      if (sys === 'sculpture' || sys === 'resonant') {
        const nodes = computeSculptureNodes(a, this.config);
        ctx.strokeStyle = color;
        ctx.lineWidth = ghost ? 1 : 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + nodes.mid.x * scale, cy + nodes.mid.y * scale);
        ctx.lineTo(cx + nodes.tip.x * scale, cy + nodes.tip.y * scale);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cx + nodes.tip.x * scale, cy + nodes.tip.y * scale, ghost ? 3 : 5, 0, Math.PI * 2);
        ctx.fill();
        return { x: cx + nodes.tip.x * scale, y: cy + nodes.tip.y * scale };
      }
      const sa = a as ElasticStateA;
      const sb = b as ElasticStateB;
      const al = this.config.L1 + sa[2];
      const bl = this.config.L2 + sb[2];
      const rx1 = cx + al * Math.sin(sa[0]) * scale;
      const ry1 = cy - al * Math.cos(sa[0]) * scale;
      const rx2 = rx1 + bl * Math.sin(sb[0]) * scale;
      const ry2 = ry1 - bl * Math.cos(sb[0]) * scale;
      ctx.strokeStyle = color;
      ctx.lineWidth = ghost ? 1 : 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(rx1, ry1);
      ctx.lineTo(rx2, ry2);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(rx2, ry2, ghost ? 3 : 5, 0, Math.PI * 2);
      ctx.fill();
      return { x: rx2, y: ry2 };
    };

    if (this.trail.length > 1) {
      ctx.strokeStyle = 'rgba(0,212,170,0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < this.trail.length; i++) {
        const x = cx + this.trail[i].x * scale;
        const y = cy + this.trail[i].y * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    if (this.diverged) {
      const pb = this.config.system === 'rigid' ? this.pertB : this.pertB;
      drawPendulum(this.pertA, pb, 'rgba(232,160,48,0.4)', true);
    }

    drawPendulum(this.baseA, this.baseB, '#00d4aa', false);

    ctx.fillStyle = this.diverged ? '#e8a030' : '#00d4aa';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(this.diverged ? 'DIVERGED' : 'Tracking', 8, 16);

    this.phaseGraphs?.draw();
    this.poincareSection?.draw();

    if (this.statusEl && this.hasState) {
      this.statusEl.textContent = `Trail: ${this.trail.length} | ${this.diverged ? 'Diverged' : 'Converged'}`;
    }
  }
}
