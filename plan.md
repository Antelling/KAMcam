# KAMcam — Implementation Plan

> **Audience**: an LLM implementing this project. Follow this document precisely. When something is unspecified, choose the simplest option that satisfies the hard rules below. Do not freelance the architecture.
>
> **Reference project**: `/home/anthony/Stupendulous/` — a working WebGL2 chaos-map app. KAMcam is a greenfield WebGPU rewrite of Stupendulous with the same features, reorganized into smaller files, using TypeGPU for typed GPU resources. **Read Stupendulous source when you need to understand a feature's exact behavior.** Paths to specific Stupendulous files are given throughout this plan as `stupendulous:path`.

---

## 1. What KAMcam is

A browser-based **chaos visualization** for double-pendulum-family systems. It renders a 2D map where each pixel is one initial condition, and the color encodes how that trajectory behaves over time (distance traveled by the end bob, time-to-divergence, etc.). The user can also click a point to run a single trajectory live in a preview panel with phase-space graphs and a Poincaré section.

**Name**: KAMcam — Kolmogorov-Arnold-Moser theory underlies near-integrable Hamiltonian chaos, which is what these maps visualize.

**Systems** (one per folder under `src/systems/`):
- `rigid` — classic rigid double pendulum (4 state floats)
- `elastic` — linear-spring elastic double pendulum (8 state floats)
- `nonlinear` — exponential-stiffening-spring elastic double pendulum (8 state floats)
- `sculpture` — recursive n-level kinetic sculpture, n=1..4 (8 state floats)
- `resonant` — dedicated 2-level resonant pendulum (4 state floats)

**Visualization modes**: `distance`, `divergence`, `divergenceDistance`, `position`, `neighborDistance`, `neighborDistanceAccumulated`.

---

## 2. Tech stack

| Tool | Version | Purpose |
|------|---------|---------|
| `typegpu` | `^0.11` | Typed WebGPU layer (buffers, bind groups, compute/render pipelines) |
| `unplugin-typegpu` | `^0.11` | Vite plugin — transpiles `'use gpu'` TS functions to WGSL |
| `@webgpu/types` | latest dev dep | WebGPU TypeScript types |
| `vite` | `^6` | Bundler/dev server |
| `typescript` | `~5.7` | Language |

**Install command** (run once at project root):
```sh
npm install typegpu
npm install -D unplugin-typegpu @webgpu/types vite typescript
```

**No other runtime deps.** No `mp4-muxer`, no `three`, no glsl plugins. If you think you need another dependency, you don't — ask first.

---

## 3. Hard rules (non-negotiable)

1. **No file over 500 lines.** If a file is approaching 500 lines, split it by responsibility. The 2785-line `shaderBuilder.ts` in Stupendulous is the anti-pattern we are explicitly fixing.
2. **One physics system per folder** under `src/systems/`. Never put two systems' logic in the same file.
3. **All GPU resources go through TypeGPU** (`tgpu`, `d`, `std` from `'typegpu'`). No raw `GPUDevice` buffer/texture/bindgroup creation except inside `src/gpu/Context.ts`.
4. **All GPU-side code uses `'use gpu'` TS functions** (the experimental transpiler). Do NOT write raw WGSL string shaders. The whole point is type-safe GPU code.
5. **No comments in code** unless a comment explains a genuinely non-obvious physical or numerical fact (e.g. "Cramer-rule solve; mass matrix is symmetric"). Do not leave section-divider comments, TODO comments, or explain-the-obvious comments.
6. **No emojis in code or UI** unless reproducing a Stupendulous UI label that already has one.
7. **`npm run build` must pass** after every phase. It runs `tsc && vite build`.
8. **`git init` the repo** and commit after each phase passes its verification gate.
9. **Do not implement video export.** Section 10 defines the *interfaces* and *schemas* to stub; do not build a video encoder. Future work.

---

## 4. Project setup files

Create these exactly.

### `package.json`
```json
{
  "name": "kamcam",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "typecheck": "tsc --noEmit",
    "preview": "vite preview"
  },
  "dependencies": {
    "typegpu": "^0.11.0"
  },
  "devDependencies": {
    "@webgpu/types": "^0.1.0",
    "typescript": "~5.7.0",
    "unplugin-typegpu": "^0.11.0",
    "vite": "^6.0.0"
  }
}
```
(Pin to whatever `npm view typegpu version` reports at install time; keep `unplugin-typegpu` at the same major.)

### `tsconfig.json`
```jsonc
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "noEmit": true,
    "skipLibCheck": true,
    "types": ["@webgpu/types"],
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  },
  "include": ["src/**/*.ts"]
}
```

### `vite.config.ts`
```ts
import { defineConfig } from 'vite';
import typegpuPlugin from 'unplugin-typegpu/vite';

export default defineConfig({
  base: '/KAMcam/',
  plugins: [typegpuPlugin()],
  build: {
    outDir: 'docs',
    emptyOutDir: true,
  },
});
```

### `.gitignore`
```
node_modules/
docs/
dist/
*.log
```

### `index.html`
A port of `stupendulous:index.html` — same DOM structure, same element IDs, same dark theme. The only changes:
- `<title>` → `KAMcam`
- Script src stays `/src/main.ts`
- Keep every `id` that Stupendulous uses; `src/ui/` binds to them.

---

## 5. Folder structure

```
KAMcam/
├── src/
│   ├── main.ts                          # Bootstrap: get canvas, init app
│   ├── app/
│   │   ├── KAMcamApp.ts                 # Top-level orchestrator
│   │   ├── AnimationLoop.ts             # requestAnimationFrame + frame budgeting
│   │   └── PlayState.ts                 # 'idle'|'playing'|'paused'|'stale'|'completed'
│   ├── gpu/
│   │   ├── Context.ts                   # tgpu.init(), canvas context config — THE ONLY raw GPU
│   │   ├── BufferHelpers.ts             # typed buffer creation wrappers
│   │   ├── QuadGeometry.ts              # fullscreen-triangle vertex data (use common.fullScreenTriangle)
│   │   └── Reductions.ts                # parallel-max compute for normalization
│   ├── systems/
│   │   ├── System.ts                    # System interface + shared types
│   │   ├── registry.ts                  # SystemType → System factory
│   │   ├── shared/
│   │   │   ├── cramer.ts                # 'use gpu' solveCramer4, solveCramer3, det4
│   │   │   ├── hash.ts                  # 'use gpu' hash(vec2)→f32
│   │   │   ├── bilinear.ts              # 'use gpu' bilinear IC interpolation
│   │   │   └── constants.ts             # G, PI as d.f32 constants
│   │   ├── rigid/
│   │   │   ├── types.ts                 # d.struct for RigidState
│   │   │   ├── deriv.ts                 # 'use gpu' computeAccelerations
│   │   │   ├── step.ts                  # 'use gpu' Verlet step
│   │   │   ├── init.ts                  # 'use gpu' IC seeding
│   │   │   ├── accumulate.ts            # 'use gpu' distance/position accumulation
│   │   │   ├── divergence.ts            # 'use gpu' divergence check
│   │   │   └── RigidSystem.ts           # implements System interface, builds pipelines
│   │   ├── elastic/                     # same file set as rigid
│   │   ├── nonlinear/                   # same file set
│   │   ├── sculpture/                   # same file set
│   │   └── resonant/                    # same file set
│   ├── simulation/
│   │   ├── Simulator.ts                 # owns state buffers, dispatches system passes
│   │   ├── DistanceMode.ts              # accumulation-mode step orchestration
│   │   ├── DivergenceMode.ts            # base+perturbed twin orchestration
│   │   ├── Tiler.ts                     # chunked/mosaic tiling mode
│   │   ├── LiveSession.ts              # rAF-bound entry point
│   │   └── OfflineRenderer.ts          # INTERFACE ONLY — see section 10
│   ├── render/
│   │   ├── DisplayRenderer.ts           # float data → screen (colormap + tonemap)
│   │   ├── colormaps.ts                 # 'use gpu' 7 colormaps
│   │   ├── tonemap.ts                   # 'use gpu' 4 tone maps
│   │   └── DisplayTypes.ts              # d.struct for DisplayData, DisplayUniforms
│   ├── preview/
│   │   ├── PendulumPreview.ts           # live single-trajectory animation
│   │   ├── PhaseGraphs.ts               # 2D-canvas phase-space line graphs
│   │   ├── PoincareSection.ts           # 2D-canvas Poincaré section
│   │   └── PreviewShaders.ts            # 'use gpu' single-trajectory integrator
│   ├── shots/
│   │   ├── ViewSnapshot.ts              # versioned serializable schema (section 10)
│   │   ├── Timeline.ts                  # Shot + Transition schema (stub)
│   │   ├── parse.ts                     # text → Timeline
│   │   └── format.ts                    # Timeline → text
│   ├── export/
│   │   └── PngExporter.ts               # port of stupendulous PNG float writer
│   ├── config/
│   │   ├── schema.ts                    # SimulationConfig + literal unions
│   │   ├── defaults.ts                  # DEFAULT_CONFIG
│   │   ├── labels.ts                    # SYSTEM_NAMES, MODE_NAMES, etc.
│   │   ├── dimensions.ts                # DIM_ORDER, systemDimensions, packing
│   │   └── corners.ts                   # computeCorners, bilinearSample, packs
│   ├── ui/
│   │   ├── Controls.ts                  # base binding helpers
│   │   ├── StatsTracker.ts              # FPS / frame counter
│   │   ├── ZoomController.ts            # zoom history + drag-zoom math
│   │   └── controls/
│   │       ├── systemControls.ts        # system type + per-system params
│   │       ├── phaseSpaceControls.ts    # axis dims, ranges, ICs, tiling
│   │       ├── integrationControls.ts   # dt, iterations, maxIter, perturb, trials
│   │       ├── renderControls.ts        # vizMode, resolution, chunk, colormap, tonemap
│   │       ├── actionControls.ts        # play/pause/reset/download buttons
│   │       └── previewControls.ts       # preview panel + description textarea
│   └── utils/
│       ├── math.ts                      # clamp, lerp, etc.
│       └── download.ts                  # trigger browser download of a Blob
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── .gitignore
├── AGENTS.md                            # build/test/deploy commands (see section 11)
├── README.md                            # brief description
└── plan.md                              # THIS FILE
```

---

## 6. TypeGPU API primer (correct patterns — copy these)

Verified against the canonical TypeGPU examples (`ComputeShadersBindGroupsRuntime.ts`, `simulation/boids/index.ts` in the software-mansion/TypeGPU repo).

### 6.1 Imports
```ts
import { tgpu, d, std, common } from 'typegpu';
```
- `tgpu` — core: `tgpu.init()`, `tgpu.bindGroupLayout`, `tgpu.vertexFn`, `tgpu.fragmentFn`
- `d` — data types: `d.f32`, `d.u32`, `d.i32`, `d.vec2f`, `d.vec3f`, `d.vec4f`, `d.vec2u`, `d.struct({...})`, `d.arrayOf(T, n)`, `d.builtin`
- `std` — WGSL stdlib: `std.sin`, `std.cos`, `std.sqrt`, `std.log`, `std.exp`, `std.atan2`, `std.clamp`, `std.mix`, `std.min`, `std.max`, `std.abs`, `std.fract`, `std.floor`, `std.normalize`, `std.length`, `std.distance`, `std.smoothstep`, `std.fwidth`, `std.pow`, `std.sign`, `std.mod`
- `common` — `common.fullScreenTriangle` (a built-in vertex shader for fullscreen passes)

### 6.2 Init + canvas
```ts
const root = await tgpu.init();
const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
```

### 6.3 Structs and types
```ts
const RigidState = d.struct({
  theta1: d.f32,
  omega1: d.f32,
  theta2: d.f32,
  omega2: d.f32,
});
type RigidStateT = d.Infer<typeof RigidState>;
```

### 6.4 Buffers
```ts
// Storage buffer (read-write on GPU):
const stateBuffer = root.createBuffer(d.arrayOf(RigidState, count), initialData).$usage('storage');

// Uniform buffer:
const paramsBuffer = root.createBuffer(ParamsStruct, paramsData).$usage('uniform');
const params = paramsBuffer.as('uniform');

// Write from CPU:
paramsBuffer.write(newData);
```

### 6.5 Bind group layouts + bind groups
```ts
const stepLayout = tgpu.bindGroupLayout({
  currentState: { storage: StateArray },               // read-only
  nextState: { storage: StateArray, access: 'mutable' }, // read-write
  params: { uniform: ParamsStruct },
});

const bindGroup = root.createBindGroup(stepLayout, {
  currentState: bufferA,
  nextState: bufferB,
  params: paramsBuffer,
});
```

### 6.6 `'use gpu'` functions
```ts
const computeAccelerations = (theta1: d.f32, omega1: d.f32, theta2: d.f32, omega2: d.f32) => {
  'use gpu';
  const delta = theta1 - theta2;
  const sinDelta = std.sin(delta);
  const cosDelta = std.cos(delta);
  // ... return d.vec2f(ax1, ax2)
  return d.vec2f(num1 / (L1 * denom), num2 / (L2 * denom));
};
```
- A `'use gpu'` function can call other `'use gpu'` functions freely.
- External TypeGPU resources (buffers, uniforms, bind-group-helpers) are referenced via the layout's `.$` accessor (see 6.7) — the unplugin transpiler collects these automatically.
- **Type casting**: when doing `a / count` where `count` is a JS number, cast with `d.f32(count)` if needed. Vector arithmetic with `d.f32` literals works directly.
- **No `for` loops with dynamic bounds** unless the bound is a constant or uniform. Fixed-trip loops (e.g. RK4's 4 stages, or `for i in 0..4`) are fine.
- **`for...of` over a `d.arrayOf`** IS supported and idiomatic (see boids example).

### 6.7 Accessing bind group resources inside `'use gpu'`
```ts
const step = (cellIndex: d.u32) => {
  'use gpu';
  const state = stepLayout.$.currentState[cellIndex];
  const a = computeAccelerations(state.theta1, state.omega1, state.theta2, state.omega2);
  stepLayout.$.nextState[cellIndex] = RigidState({
    theta1: state.theta1,
    omega1: state.omega1 + a.x * dt,
    theta2: state.theta2,
    omega2: state.omega2 + a.y * dt,
  });
};
```
- `stepLayout.$.fieldName[index]` reads/writes storage arrays.
- `paramsBuffer.as('uniform')` view accessed as `params.$.fieldName` inside `'use gpu'`.

### 6.8 Compute pipelines + dispatch
```ts
const stepPipeline = root.createGuardedComputePipeline(step);

// 2D grid dispatch (resolution × resolution cells):
stepPipeline.with(bindGroup).dispatchThreads(resolution, resolution);
```
- `createGuardedComputePipeline` gives better error messages; prefer it during development.
- For a 2D cell grid, the compute function takes a `d.u32` linear index and you compute `x = index % resolution; y = index / resolution;` inside. (This avoids ambiguity about 2D dispatch signatures. Dispatch as `dispatchThreads(resolution * resolution)`.)
- **Actually**, verify dispatch arity against the TypeGPU version. If `dispatchThreads(w, h)` with a `d.v2u` parameter works, prefer it. If not, use 1D dispatch of `res*res` threads and derive x/y. The 1D fallback always works.

### 6.9 Render pipelines (fullscreen passes)
```ts
const displayPipeline = root.createRenderPipeline({
  vertex: common.fullScreenTriangle,
  fragment: ({ uv }: { uv: d.v2f }) => {
    'use gpu';
    // uv is 0..1 across the canvas
    const value = sampleDataAt(uv);
    const color = applyColormap(value);
    return d.vec4f(color, 1);
  },
});

// Draw:
displayPipeline
  .withColorAttachment({ view: context, clearValue: [0, 0, 0, 1] })
  .with(displayBindGroup)
  .draw(3);
```

### 6.10 Reduction (parallel max)
The WebGL app's `computeMaxValue` does a `readPixels` of a 128×128 sample and finds the max on CPU — this causes the GPU stall. Replace with a compute reduction:
- Create a small buffer (e.g. 256 f32s).
- Dispatch a compute shader where each thread reads N cells, finds local max, writes to the buffer at its thread index.
- Repeat (log-pass) until 1 value remains.
- `mapAsync` that one f32 back to CPU (single read, cheap).
Keep this in `src/gpu/Reductions.ts`. The exact workgroup-size math can be straightforward; reference any standard WebGPU parallel-reduce pattern.

---

## 7. Physics math (port verbatim from Stupendulous GLSL)

The integrators and equations below are **the spec**. Port the math exactly — these have been validated visually against the live Stupendulous demo. Source files are at `stupendulous:src/shaders/fragments/*.glsl`.

### 7.1 Shared helpers — `src/systems/shared/`

**`constants.ts`**:
```ts
import { d } from 'typegpu';
export const G = 9.81;          // rigid, elastic, nonlinear
export const G_SCULPTURE = 9.80665; // sculpture, resonant
export const PI = 3.14159265359;
```

**`hash.ts`** — perturbation RNG (`stupendulous:fragments/hash.glsl`):
```ts
import { d, std } from 'typegpu';
export const hash = (p: d.v2f) => {
  'use gpu';
  return std.fract(std.sin(std.dot(p, d.vec2f(127.1, 311.7))) * 43758.5453);
};
```

**`cramer.ts`** — 4×4 and 3×3 linear solvers (`stupendulous:fragments/elastic.glsl:7-48`). Port `solveCramer3`, `det4`, `solveCramer4` as `'use gpu'` functions taking `d.mat4f` / `d.mat3f` and `d.vec4f` / `d.vec3f`. (TypeGPU exposes `d.mat3f`, `d.mat4f`. Confirm via `d.` autocomplete; if matrix types aren't exposed, use 4 separate `d.vec4f` columns and write the determinant by hand — same math.) The exact arithmetic is in `elastic.glsl` lines 7–48.

### 7.2 Rigid — `src/systems/rigid/`

**State** (`types.ts`): `RigidState = d.struct({ theta1, omega1, theta2, omega2: d.f32 })`.

**Derivatives** (`deriv.ts`) — port `stupendulous:fragments/rigid.glsl:7-28` exactly:
```ts
export const computeAccelerations = (theta1, omega1, theta2, omega2) => {
  'use gpu';
  // m1, m2, L1, L2 come from a uniform — access via params.$
  const delta = theta1 - theta2;
  const sinDelta = std.sin(delta);
  const cosDelta = std.cos(delta);
  const denom = m1 + m2 * sinDelta * sinDelta;
  const num1 = -m2 * L1 * omega1 * omega1 * sinDelta * cosDelta
             - m2 * L2 * omega2 * omega2 * sinDelta
             - (m1 + m2) * G * std.sin(theta1)
             + m2 * G * std.sin(theta2) * cosDelta;
  const num2 = (m1 + m2) * L1 * omega1 * omega1 * sinDelta
             + m2 * L2 * omega2 * omega2 * sinDelta * cosDelta
             + (m1 + m2) * G * std.sin(theta1) * cosDelta
             - (m1 + m2) * G * std.sin(theta2);
  return d.vec2f(num1 / (L1 * denom), num2 / (L2 * denom));
};
```

**Integrator** (`step.ts`) — **symplectic Verlet** (NOT RK4). Port `stupendulous:shaderBuilder.ts:326-332`:
```
half-step velocity:  omega += 0.5 * dt * accel(theta_current)
full-step position:  theta += dt * omega
recompute accel at new theta
half-step velocity:  omega += 0.5 * dt * accel(theta_new)
```

**Divergence threshold** (rigid): state-space distance > 0.05 (`stupendulous:shaderBuilder.ts` rigid divergence).

### 7.3 Elastic / Nonlinear — `src/systems/elastic/`, `src/systems/nonlinear/`

**State** (8 floats, two vec4s — but in TypeGPU, one struct of 8 f32s):
```
theta1, omega1, stretch1, stretchRate1, theta2, omega2, stretch2, stretchRate2
```

**Derivatives** — port `stupendulous:fragments/elastic.glsl:50-95` (`systemDeriv`). Builds a 4×4 mass matrix `M` and force vector `f`, solves via `solveCramer4`. The full matrix entries are in `elastic.glsl:66-70` — copy them exactly.

**Nonlinear** is identical except the spring force is exponential (`stupendulous:fragments/nonlinear.glsl:54-58`):
```
F = -sign(extension) * k * (exp(|extension| / L0) - 1)
```
instead of linear `-k * extension`. Compare `elastic.glsl:81,90` (`- u_k1 * sa.z`) vs `nonlinear.glsl:83-84,95` (`+ F_spring1`).

**Integrator**: **RK4** (not Verlet). Standard 4-stage: `out = state + (dt/6)(k1 + 2k2 + 2k3 + k4)`. Port `stupendulous:shaderBuilder.ts:360-368`.

### 7.4 Sculpture — `src/systems/sculpture/`

**State** (8 floats, packs up to 4 levels): `sa=(theta0,omega0,theta1,omega1)`, `sb=(theta2,omega2,theta3,omega3)`. Trailing levels are zero when `n<4`.

**Per-level reduced constants** — port `stupendulous:fragments/sculpture.glsl:42-68` exactly (`SCConsts`, `scConstants`). Level `i` params scale by `r^i`. Levels `i >= n` are inert padding (`I=1, mu=0, m=a=b=0`).

**Mass matrix**: `scMbase(s, j, k)` (`sculpture.glsl:70-74`), then `A[k][j] = scMbase(s,j,k) * cos(th[j]-th[k])`.

**Force vector**: `f[l] = -G*mu[l]*sin(th[l]) - sum_k scMbase(s,l,k)*sin(th[l]-th[k])*om[k]²`.

**Solve**: `scSolve4(A, f)` — same Cramer pattern as elastic but inlined (`sculpture.glsl:22-37`).

**End-effector tip** — `computeSculptureTip` (`sculpture.glsl:107-120`): chains rod segments `(ax -= b_i*sin(t_i); ay += b_i*cos(t_i))`, final bob at `ax + a_last*sin(t_last), ay - a_last*cos(t_last)`.

**Divergence** — weighted per-level tolerance that grows with level (`sculpture.glsl:127-151`): `TOL_LO=0.05`, `TOL_HI=1.0`, `OMEGA_SCALE=2.0`, geometric interpolation by level index. Circular angle difference: `da - 2π*floor(da/(2π)+0.5)`.

**Integrator**: RK4 (same as elastic).

### 7.5 Resonant — `src/systems/resonant/`

Dedicated 2-level system with individually-adjustable params (no r-scaling). Port `stupendulous:fragments/resonantPendulum.glsl` entirely:
- `RPConsts` (`resonantPendulum.glsl:21-40`)
- Closed-form 2×2 solve (`resonantPendulum.glsl:42-67`) — no Cramer needed, just `det = A00*A11 - A01*A10`
- Tip + divergence follow same pattern as sculpture.

**Integrator**: RK4.

### 7.6 bob2 (rigid/elastic end-effector) — used by distance/position modes
Port `stupendulous:fragments/bob2.glsl`:
```ts
export const computeBob2 = (theta1, theta2, l1, l2) => {
  'use gpu';
  const x1 = l1 * std.sin(theta1);
  const y1 = -l1 * std.cos(theta1);
  return d.vec2f(x1 + l2 * std.sin(theta2), y1 - l2 * std.cos(theta2));
};
```

---

## 8. Config schema — port of `stupendulous:src/types/config.ts`

Port `stupendulous:src/types/config.ts` **verbatim** into the split files under `src/config/`:
- `schema.ts` — all the `type`/`interface` declarations (lines 1–91 of the original)
- `dimensions.ts` — `DIM_ORDER`, `DIMENSIONS_BY_SYSTEM`, `systemDimensions`, `sculptureDimensions`, `DIMENSION_DEFAULTS`, `DIM_SCALE`, `basisVector`, `initialVector`, `isAngleDim` (lines 93–258)
- `corners.ts` — `computeCorners`, `bilinearSample`, `rigidPack`, `elasticPackA`, `elasticPackB`, `generateTiling`, `describeTiling` (lines 265–346)
- `defaults.ts` — `DEFAULT_CONFIG` (lines 348–394)
- `labels.ts` — `SYSTEM_NAMES`, `MODE_NAMES`, `vizModeLabel`, `COLORMAP_NAMES`, `TONE_MAPPING_NAMES`, `DIMENSION_LABELS`, `DIM_SYMBOLS`, `dimensionLabel`, `dimensionOrder` (lines 147–212, 396–436)

**Do not change any default values, dimension orderings, or packing conventions.** These are the contract between CPU config and GPU state.

---

## 9. Phased implementation

Each phase ends with a **verification gate**. Do not start the next phase until the current one passes. Commit after each phase.

### Phase 1 — Skeleton + tooling + Context
**Create**: `package.json`, `tsconfig.json`, `vite.config.ts`, `.gitignore`, `index.html` (port of `stupendulous:index.html`), `src/main.ts` (5-line bootstrap), `src/gpu/Context.ts`.

**`src/gpu/Context.ts`** wraps `tgpu.init()` and `root.configureContext()`. Exports a `createContext(canvas): Promise<Context>` function. This is the **only** file that touches raw WebGPU objects.

**`src/main.ts`**: gets the canvas, calls `createContext`, logs success.

**Verification**: `npm install && npm run build` passes; `npm run dev` opens a blank dark page with no console errors; the browser console shows the WebGPU adapter was acquired.

### Phase 2 — Rigid system end-to-end (the template)
Build the rigid system all the way through. This phase establishes every pattern the other systems will copy.

**Files**:
- `src/config/*` — all five files from section 8.
- `src/systems/shared/{constants,hash,cramer,bilinear}.ts`
- `src/systems/rigid/{types,deriv,step,init,accumulate,divergence,RigidSystem}.ts`
- `src/systems/System.ts` — the `System` interface (see 9.1 below)
- `src/systems/registry.ts` — only `rigid` registered for now
- `src/simulation/Simulator.ts` — owns state buffers, dispatches init/step/accumulate
- `src/render/{DisplayTypes,DisplayRenderer,colormaps,tonemap}.ts`
- `src/gpu/Reductions.ts` — parallel-max for normalization
- A minimal `src/app/KAMcamApp.ts` that wires Simulator + DisplayRenderer + a hardcoded `DEFAULT_CONFIG`, with a play/pause button.

**Data flow** (per frame when playing):
1. `init` compute pass — seeds state buffer from corner ICs (bilinear interpolation by cell index).
2. `step` compute pass — one Verlet timestep, ping-ponging between `stateA` and `stateB` storage buffers.
3. `accumulate` compute pass — reads bob2 position from current state, accumulates distance into a data buffer (also ping-ponged or atomically updated).
4. `reduceMax` compute pass — finds max value in the data buffer for tonemap normalization.
5. `display` render pass — samples data buffer at uv, applies colormap + tonemap, writes to canvas.

**Verification**: load the page, default config is rigid + distance mode + 512². Click Render. The chaos map should appear and visually match `stupendulous` rigid distance mode at the same parameters (rainbow colormap, linear tonemap). Compare side-by-side in two browser tabs.

### Phase 3 — Other systems via the registry
Port `elastic`, `nonlinear`, `sculpture`, `resonant` using `rigid/` as the template. Each system's folder mirrors rigid's file layout. Register all five in `registry.ts`.

**Verification**: switch to each system in the UI and confirm the chaos map renders and is visually plausible (it doesn't need pixel-perfect parity with Stupendulous, but the broad structure — symmetry, color distribution, divergence behavior — must match).

### Phase 4 — Divergence + tiling modes
- `src/simulation/DivergenceMode.ts` — runs base + perturbed twin simulations, flags iteration where divergence exceeds threshold. Port the logic from `stupendulous:src/simulation/simulator.ts` (the `runBlend` / divergence-step methods).
- `src/simulation/Tiler.ts` — chunked/mosaic mode. Port `stupendulous:src/simulation/tileMosaic.ts`. Runs the simulator per tile, blits results into a mosaic.
- Wire all six `vizMode` options through.

**Verification**: each vizMode renders correctly; tiling mode produces a mosaic when `phaseSpace.mode === 'tiling'`.

### Phase 5 — Preview + ViewSnapshot schema
- `src/preview/PendulumPreview.ts` — click-on-canvas samples the state at that point, runs a single trajectory live. Port `stupendulous:src/preview/pendulumPreview.ts` (980 lines — **split aggressively**, target <400 lines per file). The preview uses its own small compute pipeline to advance one trajectory and a 2D canvas to draw the pendulum.
- `src/preview/PhaseGraphs.ts` — 7-line phase-space plot. Port the 2D-canvas graph code.
- `src/preview/PoincareSection.ts` — kinetic-energy-slice scatter. Port the 2D-canvas code.
- `src/shots/ViewSnapshot.ts` — see section 10. The preview's "Copy Description" button emits a `ViewSnapshot` (same shape as today's JSON from `stupendulous:pendulumPreview.ts:867-908`).

**Verification**: click "Start" in preview, click on the chaos map, see the pendulum animate, phase graphs plot, Poincaré section populates, "Copy Description" produces JSON identical in shape to Stupendulous.

### Phase 6 — UI wiring
Split the UI bindings across `src/ui/controls/*.ts` (one file per panel). Port the binding logic from `stupendulous:src/app.ts:181-549` (the `setupControls` and `setupZoomControls` methods). Each controls file exports a `setupXxxControls(config, callbacks)` function.

- `src/ui/ZoomController.ts` — drag-zoom rectangle + zoom history. Port `stupendulous:src/simulation/zoomController.ts`.
- `src/ui/StatsTracker.ts` — FPS + frame count display.

**Verification**: every slider/select/button in the Stupendulous UI works identically in KAMcam. Changing system reveals/hides the right param panels. Zoom works. Stats update.

### Phase 7 — PNG export + future-proofing stubs
- `src/export/PngExporter.ts` — port `stupendulous:src/utils/` PNG float writer (dependency-free CRC/adler32/raw-storage).
- `src/shots/{Timeline,parse,format}.ts` — see section 10. Round-trip text ↔ Timeline.
- `src/simulation/OfflineRenderer.ts` — interface only, see section 10.

**Verification**: "Download Image" produces a PNG. Pasting a ViewSnapshot JSON into a textarea and parsing it round-trips losslessly. `npm run build` passes. All Stupendulous features are present.

---

### 9.1 The `System` interface — `src/systems/System.ts`

This is the contract every system implements. Define it once; each system folder exports a class/factory that satisfies it.

```ts
import type { TgpuRoot, TgpuBindGroupLayout } from 'typegpu';
import type { SimulationConfig } from '../config/schema';

export interface SystemBuffers {
  stateA: /* typed storage buffer for this system's State array */;
  stateB: /* same */;
  data: /* typed storage buffer for accumulation results */;
}

export interface System {
  readonly key: 'rigid' | 'elastic' | 'nonlinear' | 'sculpture' | 'resonant';
  readonly stateSize: number; // bytes per cell (e.g. 16 for rigid, 32 for elastic)

  buildPipelines(root: TgpuRoot, config: SimulationConfig): SystemPipelines;
  createBuffers(root: TgpuRoot, cellCount: number): SystemBuffers;
}

export interface SystemPipelines {
  init: TgpuComputePipeline;     // seed state from corner ICs
  step: TgpuComputePipeline;     // one integration timestep (Verlet or RK4)
  accumulate: TgpuComputePipeline; // update data buffer from current state
  divergenceStep?: TgpuComputePipeline; // for divergence modes
}
```

The exact TypeGPU pipeline types (`TgpuComputePipeline`, `TgpuRoot`, etc.) — confirm the import names against the installed version's type exports. The `registry.ts` is a simple `Record<SystemType, (root, config) => System>`.

---

## 10. Future-proofing for video (stub only — do not implement video)

The user's future plan: extend the "sampled pendulum" text format to include viewport + render metadata, then build a video compositor where the user pastes multiple sampled views and defines transitions between them, then a "render video" button runs offline for potentially hours.

**You must create these stubs but must NOT implement the encoder.**

### 10.1 `src/shots/ViewSnapshot.ts`
A versioned, forward-compatible schema. Today it carries exactly what `stupendulous:pendulumPreview.ts:867-908` emits. Optional fields are reserved for the future.

```ts
export const SNAPSHOT_VERSION = 1;

export interface ViewSnapshotParams {
  // Present today (system-specific):
  m1?: number; m2?: number; L1?: number; L2?: number;
  k1?: number; k2?: number;
  sculptureWeight?: number; sculptureRod?: number; sculptureAxle?: number;
  sculptureReduction?: number; sculptureN?: number;
  rpM0?: number; rpM1?: number; rpL0?: number; rpL1?: number; rpA0?: number;
  [key: string]: number | undefined;
}

export interface ViewSnapshotViewport {
  // Reserved for future use — do NOT populate yet, but the field exists.
  xDimension?: string; xMin?: number; xMax?: number;
  yDimension?: string; yMin?: number; yMax?: number;
}

export interface ViewSnapshotRender {
  // Reserved for future use.
  colormap?: number; toneMapping?: number; resolution?: number;
  iterations?: number; dt?: number;
}

export interface ViewSnapshot {
  version: typeof SNAPSHOT_VERSION;
  system: string;
  params: ViewSnapshotParams;
  initialState: Record<string, number> | null;
  viewport?: ViewSnapshotViewport;   // optional, forward-compatible
  render?: ViewSnapshotRender;       // optional, forward-compatible
}
```

The preview's "describe current pendulum" function builds a `ViewSnapshot` and `JSON.stringify`s it. The shape must round-trip through `parse(format(snapshot)) === snapshot`.

### 10.2 `src/shots/Timeline.ts`
A stub schema for the future compositor:
```ts
export type TransitionType = 'cut' | 'crossfade' | 'viewport-lerp';

export interface Transition {
  type: TransitionType;
  durationFrames: number;
  params?: Record<string, number>;
}

export interface Shot {
  snapshot: ViewSnapshot;
  durationFrames: number;
}

export interface Timeline {
  version: typeof SNAPSHOT_VERSION;
  shots: Shot[];
  transitions: Transition[]; // transitions[i] is between shots[i] and shots[i+1]
}
```

### 10.3 `src/shots/parse.ts` and `src/shots/format.ts`
- `format(snapshot | timeline): string` — `JSON.stringify` with 2-space indent.
- `parse(text: string): ViewSnapshot | Timeline` — lenient `JSON.parse`; accepts either a single snapshot object or `{ shots, transitions }`. Validate `version` field; throw a clear error if `version > SNAPSHOT_VERSION`.

### 10.4 `src/simulation/OfflineRenderer.ts`
**Interface only. Do not implement.**
```ts
import type { Timeline } from '../shots/Timeline';

export interface OfflineFrameResult {
  frameIndex: number;
  // Future: returns a VideoFrame or ImageBitmap. Left intentionally untyped for now.
  data: unknown;
}

export interface OfflineRenderer {
  render(timeline: Timeline, onFrame: (result: OfflineFrameResult) => void): Promise<void>;
}
```

This interface is the seam where a future video encoder (WebCodecs + mp4-muxer) plugs in. It is decoupled from `requestAnimationFrame` and the visible canvas by design — it must be drivable headlessly for the future 10-hour render. **Do not write an implementation class.**

---

## 11. `AGENTS.md` content

```markdown
# KAMcam

WebGPU + TypeGPU rewrite of Stupendulous (double-pendulum chaos visualization).

## Commands
- `npm run dev` — vite dev server
- `npm run build` — tsc typecheck + vite build to `docs/`
- `npm run typecheck` — tsc --noEmit only
- `npm run preview` — preview built output

## Deployment
- Build output: `docs/` (GitHub Pages)
- Repo: `git@github.com:Antelling/KAMcam.git`
- Pages base path: `/KAMcam/`
- Remote setup: `git remote add origin git@github.com:Antelling/KAMcam.git`

## Architecture
See `plan.md` for the full spec. Key rules:
- No file over 500 lines.
- All GPU code via TypeGPU (`'use gpu'` functions).
- One physics system per folder under `src/systems/`.
- Video export is NOT implemented — see `src/simulation/OfflineRenderer.ts` (interface only).
```

---

## 12. Verification checklist (final)

Before declaring the project complete:

- [ ] `npm run build` passes with zero TypeScript errors.
- [ ] No file in `src/` exceeds 500 lines (check with `find src -name '*.ts' | xargs wc -l | sort -n`).
- [ ] All 5 systems render plausible chaos maps.
- [ ] All 6 vizModes work.
- [ ] Tiling mode produces a mosaic.
- [ ] Click-to-preview works: pendulum animates, phase graphs plot, Poincaré section populates.
- [ ] "Copy Description" produces JSON matching Stupendulous's shape.
- [ ] Pasting that JSON back round-trips through `parse(format(...))`.
- [ ] "Download Image" produces a viewable PNG.
- [ ] Zoom (drag-rectangle + zoom-out) works.
- [ ] All UI sliders/selects in every system panel function.
- [ ] No raw `GPUDevice`/`GPUBuffer`/`createBindGroup` calls outside `src/gpu/Context.ts`.
- [ ] All GPU code uses `'use gpu'` functions (no WGSL string literals).
- [ ] `src/simulation/OfflineRenderer.ts` is an interface only.

---

## 13. Stupendulous source map (where to look)

When implementing a feature, read the corresponding Stupendulous source for reference:

| KAMcam concern | Stupendulous location |
|----------------|----------------------|
| Config schema & defaults | `stupendulous:src/types/config.ts` |
| WebGL primitives (DON'T port — replaced by TypeGPU) | `stupendulous:src/webgl/*` |
| Shader builder (DON'T port — replaced by per-system `'use gpu'` files) | `stupendulous:src/webgl/shaderBuilder.ts` |
| GLSL physics snippets (port the math) | `stupendulous:src/shaders/fragments/*.glsl` |
| Final display shader (colormaps/tonemap) | `stupendulous:src/shaders/render.glsl` |
| Simulator orchestration | `stupendulous:src/simulation/simulator.ts` |
| Tiling mode | `stupendulous:src/simulation/tileMosaic.ts` |
| Zoom math | `stupendulous:src/simulation/zoomController.ts` |
| App orchestration & UI wiring | `stupendulous:src/app.ts` |
| Preview animation | `stupendulous:src/preview/pendulumPreview.ts` |
| UI binding helpers | `stupendulous:src/ui/uiController.ts` |
| PNG export | `stupendulous:src/utils/` |
| HTML structure & element IDs | `stupendulous:index.html` |

---

## 14. Final notes for the implementer

- **When TypeGPU's API surprises you**: check the live examples at `https://docs.swmansion.com/TypeGPU/examples` (especially `simulation/boids` and `ComputeShadersBindGroups`). The canonical source of truth is the installed package's TypeScript exports and the examples repo.
- **When a `'use gpu'` function won't compile**: the transpiler supports a subset of JS. No closures over plain JS variables (only over TypeGPU resources via layout `.$`). No `.map`/`.filter`. Fixed-trip `for` loops only. If something won't transpile, hoist it into a separate named `'use gpu'` function and pass values as parameters.
- **Matrix types**: if `d.mat4f` / `d.mat3f` aren't available in your TypeGPU version, represent a 4×4 as four `d.vec4f` columns and write Cramer's rule against those. The math in `elastic.glsl` is column-major already.
- **Numbers in shaders**: a bare JS number literal in a `'use gpu'` function is fine for multiplication, but for division `a / count` may need `d.f32(count)`. When in doubt, cast.
- **Visual parity first, perfection second**: if you can't get pixel-identical output to Stupendulous, that's acceptable as long as the chaos-map structure is recognizably correct. Numerical drift from WGSL vs GLSL rounding is expected.
- **Read `plan.md` again before each phase.** This file is the spec.
