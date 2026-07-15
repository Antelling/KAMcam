import { tgpu } from 'typegpu';
import type { TgpuRoot } from 'typegpu';

export interface Context {
  root: TgpuRoot;
  canvasContext: GPUCanvasContext;
}

export async function createContext(canvas: HTMLCanvasElement): Promise<Context> {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser.');
  }

  const root = await tgpu.init();
  const canvasContext = root.configureContext({
    canvas,
    alphaMode: 'premultiplied',
  });

  return { root, canvasContext };
}
