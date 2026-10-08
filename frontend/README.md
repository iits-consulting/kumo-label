# Frontend

Svelte 5 (runes) + TypeScript SPA for dataset exploration, annotation, and training management. Plain Vite (no SvelteKit), hand-rolled router.

## Running

```bash
npm install
npm run dev          # Dev server on :8080, proxies /api to :8000
npm run build        # Production build
npm run lint         # ESLint
npm run check        # svelte-check (type gate)
npm run test         # Vitest
```

## Entry Point

[`src/main.ts`](src/main.ts) -- Mounts [`App.svelte`](src/App.svelte), which sets up TanStack svelte-query, the global tooltip provider, the svelte-sonner toaster, and route resolution via [`src/router.svelte.ts`](src/router.svelte.ts).

## Pages (`pages/`)

| Page | Route | Purpose |
|---|---|---|
| [`ProjectSetup.svelte`](src/pages/ProjectSetup.svelte) | `/` | Select dataset path, task type, embedding model |
| [`Explorer.svelte`](src/pages/Explorer.svelte) | `/explorer` | Main workspace: scatter plot, table view, filtering, class management |
| [`Explorer.svelte`](src/pages/Explorer.svelte) | `/explorer/annotate` | Same as above with annotation overlay open |

## Key Components

- [**`explorer/ScatterPlot.svelte`**](src/components/explorer/ScatterPlot.svelte) -- 2D scatter plot of image embeddings (regl-scatterplot/WebGL), supports click/lasso/box selection
- [**`explorer/ExplorerSidebar.svelte`**](src/components/explorer/ExplorerSidebar.svelte) -- Filters (color-by, label status), class list with rename/remove
- [**`explorer/InspectorPanel.svelte`**](src/components/explorer/InspectorPanel.svelte) -- Shows selected image details, quick labeling
- [**`explorer/ImageTable.svelte`**](src/components/explorer/ImageTable.svelte) -- Tabular view of dataset images
- [**`explorer/TrainingDrawer.svelte`**](src/components/explorer/TrainingDrawer.svelte) -- Start training, view live metrics (LayerChart)
- [**`annotator/AnnotatorOverlay.svelte`**](src/components/annotator/AnnotatorOverlay.svelte) -- Full-screen annotation mode for labeling selected images

## State (`src/state/`)

Reusable stateful logic. `.svelte.ts` factory modules return objects with `$state`/`$derived` getters:

- [**`classManager.svelte.ts`**](src/state/classManager.svelte.ts) -- Class CRUD, rename, color assignment
- [**`shortcutSettings.svelte.ts`**](src/state/shortcutSettings.svelte.ts) -- Configurable keyboard shortcuts + pinned classes (localStorage-persisted)
- [**`jobPoller.ts`**](src/state/jobPoller.ts) / [**`trainingPoller.ts`**](src/state/trainingPoller.ts) -- Poll `/api/jobs/{id}` and `/api/training/runs/{id}` until completion (svelte-query `refetchInterval`)
- [**`toast.ts`**](src/state/toast.ts) -- `toast({title, description, variant})` shim over svelte-sonner

## API Layer

[`src/lib/api.ts`](src/lib/api.ts) -- Thin wrapper around `fetch` that prefixes `/api` to all paths. Exports `api.get`, `api.post`, `api.put`, `api.patch`, `api.delete`, `api.getBinary`. Vite proxies `/api` requests to the backend at `:8000`.

## Stack

Svelte 5 runes, TanStack svelte-query v6, shadcn-svelte/bits-ui (`components/ui/`), LayerChart, svelte-sonner, `@lucide/svelte`, Tailwind CSS, Vite. Path alias `@/` maps to `src/`.
