# Recall Frontend

React + TypeScript + Vite shell for the Recall desktop/web UI. It talks to the existing Python `ui_server.py` API, so the backend can keep handling Messages export, contacts, previews, jobs, and report analysis while the frontend becomes a real app.

## Run locally

From the repo root:

```bash
./venv/bin/python ui_server.py
```

From this `app/` directory:

```bash
/opt/homebrew/Cellar/node/25.4.0/bin/npm run dev
```

The Vite dev server defaults to `http://127.0.0.1:5173` and proxies `/api` to `http://127.0.0.1:8765`.

If the Python API is on a different port:

```bash
RECALL_API_PROXY=http://127.0.0.1:63092 /opt/homebrew/Cellar/node/25.4.0/bin/npm run dev
```

## Current surface

- Analyze conversations with cached previews, cost estimates, recent messages, and report launch.
- Read saved reports through Story, Events, Patterns, Media, Activity, and Files sections.
- Manage local setup from Settings: contact names, message export refresh, model preference, and preview cache clearing.

## Checks

```bash
/opt/homebrew/Cellar/node/25.4.0/bin/npm run lint
/opt/homebrew/Cellar/node/25.4.0/bin/npm run build
```
