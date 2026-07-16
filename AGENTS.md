# KAMcam

WebGPU + TypeGPU rewrite of Stupendulous (double-pendulum chaos visualization).

## Commands
- `npm run dev` — vite dev server
- `npm run build` — tsc typecheck + vite build to `docs/`
- `npm run typecheck` — tsc --noEmit only
- `npm run preview` — preview built output

## Deployment
- Build output: `docs/` (committed to repo for GitHub Pages)
- Repo: `git@github.com:Antelling/KAMcam.git`
- Pages base path: `/KAMcam/` (set in `vite.config.ts`)
- GitHub Pages source: branch `main`, folder `/docs`

## Architecture
Full spec in `plan.md`. Key rules:
- No file over 500 lines.
- All GPU code via TypeGPU (`'use gpu'` functions). No raw WGSL strings.
- One physics system per folder under `src/systems/`.
- Video export is NOT implemented — see `src/simulation/OfflineRenderer.ts` (interface only).
- Rebuild (`npm run build`) and commit `docs/` before every push — GitHub Pages serves from it.

## Key gotchas
- **WebGPU requires secure context**: HTTPS or localhost. HTTP on a remote IP will NOT expose `navigator.gpu`. The error message in `src/gpu/Context.ts` explains this.
- **`docs/` IS checked in** (unlike `node_modules/`). Do not gitignore it.
- **File size**: if a file approaches 500 lines, split by responsibility.
- **TypeGPU scalar params are `number`**, NOT `d.f32`. Vector params use `d.vec2f` etc.
- **`dispatchThreads()`** not `dispatchWorkgroups()` — confirmed API for guarded compute pipelines.
- **Type imports**: `TgpuRoot` etc. must use `import type { ... }`.
- **Upstream reference**: `/home/anthony/Stupendulous/` has the original WebGL2 app for comparison.