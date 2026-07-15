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

## Architecture
See `plan.md` for the full spec. Key rules:
- No file over 500 lines.
- All GPU code via TypeGPU (`'use gpu'` functions).
- One physics system per folder under `src/systems/`.
- Video export is NOT implemented — see `src/simulation/OfflineRenderer.ts` (interface only).
