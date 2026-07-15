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

  constructor(root: TgpuRoot, dataArray: any, cellCount: number) {
    this.root = root;
    this.paramsBuffer = root.createBuffer(DisplayUniforms, {
      maxValue: 1, colormap: 6, toneMapping: 0, vizMode: 0,
      resolution: 512, _pad0: 0, _pad1: 0, _pad2: 0,
    }).$usage('uniform');
    this.layout = tgpu.bindGroupLayout({
      data: { storage: dataArray },
      params: { uniform: DisplayUniforms },
    });
    this.bindGroup = root.createBindGroup(this.layout, { data: dataArray, params: this.paramsBuffer });

    const displayLayout = this.layout;

    const fragment = (uv: any) => {
      'use gpu';
      const params = displayLayout.$.params;
      const res = params.resolution;
      const px = std.min(d.u32(uv.x * res), d.u32(res) - 1);
      const py = std.min(d.u32(uv.y * res), d.u32(res) - 1);
      const idx = py * d.u32(res) + px;
      const cell = displayLayout.$.data[idx];
      const vizMode = params.vizMode;
      if (vizMode < 0.5) {
        if (cell.a < 0.5) return { x: 0.1, y: 0.1, z: 0.1, w: 1.0 };
        const t = std.clamp(applyTonemap(cell.b, params.maxValue, params.toneMapping), 0.0, 1.0);
        const inv = 1.0 - t;
        const color = applyColormap(inv, params.colormap);
        return { x: color.r, y: color.g, z: color.b, w: 1.0 };
      }
      return { x: 1.0, y: 0.0, z: 1.0, w: 1.0 };
    };

    this.pipeline = root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
    });
  }

  update(dataArray: any): void {
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
