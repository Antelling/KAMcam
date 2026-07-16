# KAMcam

**[Live demo](https://antelling.github.io/KAMcam/)** &mdash; requires a WebGPU browser (Chrome 113+, Edge 113+, or Firefox with `dom.webgpu.enabled` set).

KAMcam is a browser-based chaos visualization for double-pendulum-family systems, named after Kolmogorov-Arnold-Moser theory. Each pixel in the chaos map represents one initial condition; color encodes how that trajectory behaves over time. Built from the ground up on WebGPU with [TypeGPU](https://github.com/software-mansion/TypeGPU) for type-safe GPU programming.

---

### Features

Choose from five physical systems: the classic **rigid double pendulum**, a parallel **linear-elastic pendulum** with two spring-coupled rods, a **nonlinear-elastic pendulum** where spring force grows exponentially with extension, a **kinetic sculpture** (up to four levels of nested pendulum arms with configurable weight, rod length, axle radius, and geometric reduction), and a dedicated **resonant pendulum** with fully independent per-level parameters. Six visualization modes reveal different aspects of the dynamics: **bob2 distance** (total path length traveled by the tip), **divergence time** (how many iterations before two initially close trajectories separate beyond a threshold), **divergence distance** (the distance at the moment of separation), **pendulum position** (average tip position), and two **neighbor distance** metrics for local sensitivity. A click-to-preview panel animates a single trajectory live with phase-space graphs and a Poincare section, and a tiling mode mosaics multiple parameter combinations into one image.

Under the hood, each system uses the integrator best suited to its structure. The rigid pendulum runs a symplectic Verlet integrator that exactly preserves phase-space volume (no drift), while the elastic, nonlinear, sculpture, and resonant systems use fourth-order Runge-Kutta. The elastic-family systems build a 4-by-4 mass matrix from the kinetic energy and solve with Cramer's rule at every timestep. Divergence modes launch twin trajectories — one unperturbed baseline and one with a tiny offset — and detect the iteration where state-space distance crosses a system-specific tolerance. Perturbations can be uniform or Gaussian, and multiple trials are averaged to reduce noise. All integration runs on the GPU across millions of initial conditions in parallel.

The render pipeline uses a compute-shader parallel reduction to find the maximum value in the accumulated data buffer, then applies a tone map (linear, logarithmic, square root, or S-curve) and one of seven colormaps (viridis, magma, plasma, inferno, turbo, jet, rainbow) in a fullscreen fragment pass. The canvas supports drag-to-zoom with a zoom history stack, and the toolbar offers one-click PNG export. The UI is a live-binding control panel that reveals or hides per-system parameters as you switch between models, enforces valid phase-space dimension pairs, and displays running FPS, frame count, and maximum value statistics.

---

### Development

```bash
npm install
npm run dev       # Vite dev server
npm run build     # TypeScript check + Vite build to docs/
npm run typecheck # TypeScript check only
npm run preview   # Preview built output
```

Built with [TypeGPU](https://typegpu.org) — all GPU shaders are written in a type-safe TypeScript subset (`'use gpu'` functions) and transpiled to WGSL at build time. No raw WGSL strings, no WebGL fallback.