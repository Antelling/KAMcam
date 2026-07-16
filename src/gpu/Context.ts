import { tgpu } from 'typegpu';
import type { TgpuRoot } from 'typegpu';

export interface Context {
  root: TgpuRoot;
  canvasContext: GPUCanvasContext;
}

export async function createContext(canvas: HTMLCanvasElement): Promise<Context> {
  if (!navigator.gpu) {
    const isSecure = window.isSecureContext;
    const isLocalhost = location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '[::1]';
    let msg = 'WebGPU is not available.';
    if (!isSecure) {
      msg += ' WebGPU requires a secure context (HTTPS or localhost).';
      msg += ' Try accessing via http://localhost' + location.pathname;
      msg += ' or use HTTPS.';
    } else {
      msg += ' Check that dom.webgpu.enabled is true in about:config.';
    }
    throw new Error(msg);
  }

  const root = await tgpu.init();
  const canvasContext = root.configureContext({
    canvas,
    alphaMode: 'premultiplied',
  });

  return { root, canvasContext };
}
