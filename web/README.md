# Trading Playground — Web Frontend

SvelteKit 5 + TypeScript frontend for the Trading Playground.

## Setup

```bash
npm install
npm run dev
```

The dev server proxies API requests to the FastAPI backend (default port 8000). Set `API_PORT` to override.

## Pages

- `/` — Dashboard with stats and recent runs
- `/data` — Sync Polymarket data and view category breakdowns
- `/train` — Configure and launch training runs
- `/runs` — Browse runs, view live progress and metrics
