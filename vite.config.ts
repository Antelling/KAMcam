import { defineConfig } from 'vite';
import typegpuPlugin from 'unplugin-typegpu/vite';

export default defineConfig({
  base: '/KAMcam/',
  plugins: [typegpuPlugin()],
  build: {
    outDir: 'docs',
    emptyOutDir: true,
    sourcemap: true,
  },
});
