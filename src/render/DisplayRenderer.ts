import { tgpu, d, std, common } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { DataCell } from '../systems/rigid/types';
import { DisplayUniforms } from './DisplayTypes';
import { applyColormap } from './colormaps';
import { applyTonemap } from './tonemap';

export class DisplayRenderer {
  private root: TgpuRoot;
  private layout: any;
  private pipeline: any;
  private paramsBuffer: any;
  private bindGroup: any;
  private dataArrType: any;

  constructor(root: TgpuRoot, dataArray: any, cellCount: number) {
    this.root = root;
    this.dataArrType = d.arrayOf(DataCell, cellCount);
    this.paramsBuffer = root.createBuffer(DisplayUniforms, {
      maxValue: 1, colormap: 6, toneMapping: 0, vizMode: 0,
      resolution: 512, _pad0: 0, _pad1: 0, _pad2: 0,
    }).$usage('uniform');
    this.layout = tgpu.bindGroupLayout({
      data: { storage: this.dataArrType },
      params: { uniform: DisplayUniforms },
    });
    this.bindGroup = root.createBindGroup(this.layout, { data: dataArray, params: this.paramsBuffer });

    const displayLayout = this.layout;

    const fragment = ({ uv }: { uv: d.v2f }) => {
      'use gpu';
      const params = displayLayout.$.params;
      const res = params.resolution;
      const px = std.min(d.u32(uv.x * res), d.u32(res) - 1);
      const py = std.min(d.u32(uv.y * res), d.u32(res) - 1);
      const idx = py * d.u32(res) + px;
      const cell = displayLayout.$.data[idx];
      const vm = params.vizMode;
      let value = 0.0;
      let invert = false;
      if (vm < 0.5) {
        if (cell.a < 0.5) return d.vec4f(0.1, 0.1, 0.1, 1);
        value = cell.b;
        invert = true;
      } else if (vm < 1.5) {
        if (cell.a < 0.5) return d.vec4f(0.1, 0.1, 0.1, 1);
        value = cell.b;
      } else if (vm < 2.5) {
        if (cell.a < 0.5) return d.vec4f(1, 1, 1, 1);
        value = cell.b;
      } else if (vm < 3.5) {
        const angle = std.atan2(cell.g, cell.r);
        const t = angle / 6.28318530718 + 0.5;
        const color = applyColormap(t, params.colormap);
        return d.vec4f(color.r, color.g, color.b, 1);
      } else if (vm < 4.5) {
        if (cell.a < 0.5) return d.vec4f(0.1, 0.1, 0.1, 1);
        value = cell.b;
      } else {
        if (cell.a < 0.5) return d.vec4f(0.1, 0.1, 0.1, 1);
        value = cell.b;
      }
      const t = std.clamp(applyTonemap(value, params.maxValue, params.toneMapping), 0.0, 1.0);
      const ct = invert ? 1.0 - t : t;
      const color = applyColormap(ct, params.colormap);
      return d.vec4f(color.r, color.g, color.b, 1);
    };

    this.pipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
    });
  }

  update(dataArray: any): void {
    this.layout = tgpu.bindGroupLayout({
      data: { storage: this.dataArrType },
      params: { uniform: DisplayUniforms },
    });
    this.bindGroup = this.root.createBindGroup(this.layout, { data: dataArray, params: this.paramsBuffer });
  }

  setParams(maxValue: number, colormap: number, toneMapping: number, vizMode: number, resolution: number): void {
    this.paramsBuffer.write({ maxValue, colormap, toneMapping, vizMode, resolution, _pad0: 0, _pad1: 0, _pad2: 0 });
  }

  render(canvasContext: GPUCanvasContext): void {
    this.pipeline
      .withColorAttachment({ view: canvasContext, clearValue: [0, 0, 0, 1] })
      .with(this.bindGroup)
      .draw(3);
  }
}