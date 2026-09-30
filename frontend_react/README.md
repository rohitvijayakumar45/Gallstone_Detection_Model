# GallStone Workstation — Frontend

React 19 + TypeScript + Vite + Tailwind. Light-theme only.

## Setup

```bash
cd frontend_react
npm install
npm run dev
```

Dev server: http://localhost:5174 · proxies `/api` → http://localhost:8000

## Scripts
- `npm run dev` — start Vite dev server
- `npm run build` — type-check + production build
- `npm run typecheck` — strict TS check
- `npm run lint` — ESLint

## Layout
```
src/
├── main.tsx              root, React Query + Router providers
├── App.tsx               routes
├── components/AppShell   nav + footer + shell
├── pages/                LandingPage, WorkstationPage, BenchmarksPage, Compare, Audit, NotFound
├── lib/api.ts            typed fetch client
└── styles/               tokens.css + globals.css
```

## Design tokens
See `src/styles/tokens.css` (light-only). Palette + spacing + radii + shadows
map to Tailwind via `tailwind.config.ts` — use `bg-surface`, `text-text-2`,
`num` for tabular figures, `label-caps` for section labels.

## Non-goals
- No dark mode
- No editorial serifs
- No fake numbers — every displayed metric flows from the API
