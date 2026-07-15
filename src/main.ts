import { createContext } from './gpu/Context';

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement | null;
  if (!canvas) {
    throw new Error('Canvas element not found');
  }

  const ctx = await createContext(canvas);
  console.log('KAMcam bootstrapped — TypeGPU root acquired');
  void ctx;
}

main().catch((err) => {
  console.error('KAMcam failed to start:', err);
  const status = document.getElementById('status');
  if (status) status.textContent = `Error: ${err.message}`;
});
