# KAMcam

A WebGPU-based chaos visualization for double-pendulum-family systems. Each pixel in the chaos map represents one initial condition; color encodes how that trajectory behaves over time (distance traveled, time-to-divergence, etc.).

Rewrite of [Stupendulous](https://github.com/Antelling/Stupendulous-) using [TypeGPU](https://github.com/software-mansion/TypeGPU) for type-safe GPU programming.

## Development

```bash
npm install
npm run dev
```

Requires a WebGPU-capable browser (Chrome 113+, Edge 113+, Safari 16.4+, Firefox shipping).
