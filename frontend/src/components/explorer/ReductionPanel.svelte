<script lang="ts">
  import type { ExplorerFilters, ReductionMethod } from "@/lib/types";
  import { Slider } from "@/components/ui/slider";
  import LoaderCircle from "@lucide/svelte/icons/loader-circle";
  import Shrink from "@lucide/svelte/icons/shrink";
  import X from "@lucide/svelte/icons/x";
  import { api } from "@/lib/api";
  import { parseProjectionBlob } from "@/lib/projectionBlob";
  import { createJobPoller } from "@/state/jobPoller";

  interface Props {
    dbPath: string;
    embeddingModel: "dinov2" | "clip";
    embeddingExists: boolean;
    onProjectionReady: (coords: Array<{ id: number; x: number; y: number }>) => void;
    filters: ExplorerFilters;
    onFiltersChange: (f: ExplorerFilters) => void;
    onStatusChange?: (status: { method: ReductionMethod; isLoading: boolean }) => void;
    onClose: () => void;
  }

  let {
    dbPath,
    embeddingModel,
    embeddingExists,
    onProjectionReady,
    filters,
    onFiltersChange,
    onStatusChange,
    onClose,
  }: Props = $props();

  // Staged values commit into filters only when a projection actually loads.
  // Staged perplexity default 15 is deliberately distinct from
  // filters.perplexity's default 30 (spec §8.3 — keep).
  // svelte-ignore state_referenced_locally
  let stagedMethod = $state<ReductionMethod>(filters.reductionMethod);
  let stagedPerplexity = $state(15);
  let projectionJobId = $state<string | null>(null);
  let pendingParamHash = $state<string | null>(null);
  let isProjectionLoading = $state(false);
  let isCached = $state(false);

  function update(partial: Partial<ExplorerFilters>) {
    onFiltersChange({ ...filters, ...partial });
  }

  const buildParams = (method: ReductionMethod, perplexity: number) =>
    method === "umap"
      ? { n_neighbors: perplexity }
      : method === "tsne"
      ? { perplexity }
      : {};

  const projectionJobQuery = createJobPoller(
    () => projectionJobId,
    () => dbPath || null,
  );

  $effect(() => {
    onStatusChange?.({ method: stagedMethod, isLoading: isProjectionLoading });
  });

  // Job completion — keyed on status only (prev-status guard): done → fetch
  // blob → onProjectionReady → commit staged values into filters.
  let prevJobStatus: string | undefined;
  $effect(() => {
    const status = projectionJobQuery.data?.status;
    if (status === prevJobStatus) return;
    prevJobStatus = status;
    if (status === "done" && pendingParamHash) {
      api
        .getBinary(`/projections/${pendingParamHash}/blob?db_path=${encodeURIComponent(dbPath)}`)
        .then((buf) => {
          onProjectionReady(parseProjectionBlob(buf));
          update({ reductionMethod: stagedMethod, perplexity: stagedPerplexity });
          isCached = true;
        })
        .finally(() => {
          projectionJobId = null;
          pendingParamHash = null;
          isProjectionLoading = false;
        });
    }
    if (status === "failed" || status === "cancelled") {
      projectionJobId = null;
      pendingParamHash = null;
      isProjectionLoading = false;
    }
  });

  async function handleCancelProjection() {
    const id = projectionJobId;
    // Drop loading UI immediately so the user sees the cancel take effect.
    projectionJobId = null;
    pendingParamHash = null;
    isProjectionLoading = false;
    if (id) {
      try {
        await api.post(`/jobs/${id}/cancel`, { db_path: dbPath });
      } catch {
        // Ignore — UI is already reset.
      }
    }
  }

  async function handleComputeProjection() {
    isProjectionLoading = true;
    isCached = false;
    try {
      const resp = await api.post<{
        cached: boolean;
        param_hash: string;
        job_id?: string;
      }>("/projections/compute", {
        db_path: dbPath,
        model: embeddingModel,
        method: stagedMethod,
        params: buildParams(stagedMethod, stagedPerplexity),
      });
      if (resp.cached && resp.param_hash) {
        const buf = await api.getBinary(
          `/projections/${resp.param_hash}/blob?db_path=${encodeURIComponent(dbPath)}`,
        );
        onProjectionReady(parseProjectionBlob(buf));
        update({ reductionMethod: stagedMethod, perplexity: stagedPerplexity });
        isProjectionLoading = false;
      } else if (resp.job_id) {
        projectionJobId = resp.job_id;
        pendingParamHash = resp.param_hash;
      }
    } catch {
      isProjectionLoading = false;
    }
  }

  // ponytail: single 500ms debounce doing check+load — deliberate deviation 3
  // (spec §10): React ran a 300ms check-only + 500ms check-and-load timer pair;
  // only visible difference is the "Cached" badge appearing ~200ms later.
  $effect(() => {
    if (!dbPath || !embeddingExists) {
      isCached = false;
      return;
    }
    const model = embeddingModel;
    const method = stagedMethod;
    const perplexity = stagedPerplexity;
    const paramsJson = JSON.stringify(buildParams(method, perplexity));

    const timeout = setTimeout(() => {
      api
        .get<{ cached: boolean; param_hash: string }>(
          `/projections/check?db_path=${encodeURIComponent(dbPath)}&model=${model}&method=${method}&params=${encodeURIComponent(paramsJson)}`,
        )
        .then(({ cached, param_hash }) => {
          isCached = cached;
          if (!cached) return;
          return api
            .getBinary(`/projections/${param_hash}/blob?db_path=${encodeURIComponent(dbPath)}`)
            .then((buf) => {
              onProjectionReady(parseProjectionBlob(buf));
              update({ reductionMethod: method, perplexity });
            });
        })
        .catch(() => {
          isCached = false;
        });
    }, 500);

    return () => clearTimeout(timeout);
  });
</script>

<div class="flex flex-col h-full">
  <div class="flex items-center justify-between px-3 py-2 border-b border-border">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
      <Shrink class="h-3.5 w-3.5" />
      Reduction
    </h3>
    <button onclick={onClose} class="text-muted-foreground hover:text-foreground">
      <X class="h-3.5 w-3.5" />
    </button>
  </div>

  <div class="p-3 space-y-3">
    <div class="space-y-2">
      <span class="text-xs text-muted-foreground">Method</span>
      <div class="flex gap-1">
        {#each ["umap", "tsne", "pca"] as const as m (m)}
          <button
            onclick={() => (stagedMethod = m)}
            class={`flex-1 px-2 py-1.5 text-xs rounded-md transition-colors uppercase ${
              stagedMethod === m
                ? "bg-foreground text-background font-medium"
                : "bg-muted text-muted-foreground hover:bg-border"
            }`}
          >
            {m}
          </button>
        {/each}
      </div>
    </div>

    {#if stagedMethod === "umap" || stagedMethod === "tsne"}
      <div class="space-y-2">
        <div class="flex justify-between text-xs text-muted-foreground">
          <span>{stagedMethod === "umap" ? "n_neighbors" : "Perplexity"}</span>
          <span class="tabular-nums">{stagedPerplexity}</span>
        </div>
        <Slider
          type="single"
          min={5}
          max={100}
          step={5}
          value={stagedPerplexity}
          onValueChange={(v) => (stagedPerplexity = v)}
        />
      </div>
    {/if}

    {#if isProjectionLoading}
      <div class="flex gap-2">
        <div class="flex-1 h-8 text-xs rounded flex items-center justify-center gap-1.5 bg-muted text-muted-foreground">
          <LoaderCircle class="h-3 w-3 animate-spin" />
          {projectionJobQuery.data?.message ?? "Computing..."}
        </div>
        <button
          onclick={handleCancelProjection}
          class="px-3 h-8 text-xs rounded bg-destructive text-destructive-foreground hover:bg-destructive/90"
        >
          Cancel
        </button>
      </div>
    {:else}
      <button
        onclick={handleComputeProjection}
        disabled={!embeddingExists || isCached}
        class={`w-full h-8 text-xs rounded flex items-center justify-center gap-1.5 ${
          !embeddingExists
            ? "bg-muted text-muted-foreground opacity-50"
            : isCached
            ? "bg-muted text-muted-foreground opacity-50"
            : "bg-primary text-primary-foreground hover:bg-primary/90"
        }`}
      >
        {#if isCached}
          <span class="w-1.5 h-1.5 rounded-full bg-green-500"></span>
          Cached
        {:else}
          Compute Projection
        {/if}
      </button>
    {/if}
  </div>
</div>
