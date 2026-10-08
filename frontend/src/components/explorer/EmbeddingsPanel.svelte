<script lang="ts">
  import { createQuery, useQueryClient } from "@tanstack/svelte-query";
  import { api } from "@/lib/api";
  import { createJobPoller } from "@/state/jobPoller";
  import CircleCheckBig from "@lucide/svelte/icons/circle-check-big";
  import Circle from "@lucide/svelte/icons/circle";
  import Cpu from "@lucide/svelte/icons/cpu";
  import LoaderCircle from "@lucide/svelte/icons/loader-circle";
  import RefreshCw from "@lucide/svelte/icons/refresh-cw";
  import X from "@lucide/svelte/icons/x";

  interface Props {
    dbPath: string;
    embeddingModel: "dinov2" | "clip";
    onEmbeddingModelChange: (model: "dinov2" | "clip") => void;
    initialEmbeddingJobId?: string | null;
    initialBatchSize?: number;
    onStatusChange?: (status: { exists: boolean; count: number; isComputing: boolean }) => void;
    onClose: () => void;
  }

  let {
    dbPath,
    embeddingModel,
    onEmbeddingModelChange,
    initialEmbeddingJobId,
    initialBatchSize,
    onStatusChange,
    onClose,
  }: Props = $props();

  const queryClient = useQueryClient();

  // Seeded once from the ProjectSetup handoff (intentional init-time prop reads).
  // svelte-ignore state_referenced_locally
  let batchSize = $state(initialBatchSize ?? 32);
  // svelte-ignore state_referenced_locally
  let embeddingJobId = $state<string | null>(initialEmbeddingJobId ?? null);

  const embeddingStatusQuery = createQuery<Record<string, { exists: boolean; count?: number; created_at?: string }>>(() => ({
    queryKey: ["embeddings-status", dbPath],
    queryFn: () =>
      api.get<Record<string, { exists: boolean; count?: number; created_at?: string }>>(
        `/embeddings/status?db_path=${encodeURIComponent(dbPath)}`,
      ),
    enabled: !!dbPath,
  }));

  const embeddingJobQuery = createJobPoller(
    () => embeddingJobId,
    () => dbPath || null,
  );

  const embeddingExists = $derived(!!embeddingStatusQuery.data?.[embeddingModel]?.exists);
  const embeddingCount = $derived(embeddingStatusQuery.data?.[embeddingModel]?.count ?? 0);

  // Report status upward whenever it changes.
  $effect(() => {
    onStatusChange?.({ exists: embeddingExists, count: embeddingCount, isComputing: !!embeddingJobId });
  });

  // Job completion — keyed on status only (prev-status guard). OOM failures
  // halve the batch size, same recovery as ProjectSetup (checklist 1).
  let prevJobStatus: string | undefined;
  $effect(() => {
    const job = embeddingJobQuery.data;
    const status = job?.status;
    if (status === prevJobStatus) return;
    prevJobStatus = status;
    if (status === "done") {
      embeddingJobId = null;
      queryClient.invalidateQueries({ queryKey: ["embeddings-status", dbPath] });
    }
    if (status === "failed") {
      embeddingJobId = null;
      const isOom = job.error?.toLowerCase().includes("out of memory");
      if (isOom) {
        batchSize = Math.max(1, Math.floor(batchSize / 2));
      }
    }
  });

  async function handleRecomputeEmbeddings() {
    try {
      const { job_id } = await api.post<{ job_id: string }>("/embeddings/compute", {
        db_path: dbPath,
        model: embeddingModel,
        batch_size: batchSize,
      });
      embeddingJobId = job_id;
    } catch { /* ignore */ }
  }
</script>

<div class="flex flex-col h-full">
  <div class="flex items-center justify-between px-3 py-2 border-b border-border">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
      <Cpu class="h-3.5 w-3.5" />
      Embeddings
    </h3>
    <button onclick={onClose} class="text-muted-foreground hover:text-foreground">
      <X class="h-3.5 w-3.5" />
    </button>
  </div>

  <div class="p-3 space-y-3">
    <div class="flex items-center gap-2 text-xs text-muted-foreground">
      {#if embeddingExists}
        <CircleCheckBig class="h-3 w-3 text-green-500 shrink-0" />
        <span>{embeddingCount.toLocaleString()} images embedded</span>
      {:else}
        <Circle class="h-3 w-3 shrink-0" />
        <span>Not computed</span>
      {/if}
    </div>

    {#if !!embeddingJobId && embeddingJobQuery.data}
      <div class="space-y-1">
        <div class="flex justify-between text-xs text-muted-foreground">
          <span>{embeddingJobQuery.data.message ?? "Computing..."}</span>
          <span>
            {embeddingJobQuery.data.total > 0
              ? `${embeddingJobQuery.data.completed} / ${embeddingJobQuery.data.total}`
              : "..."}
          </span>
        </div>
        <div class="h-1.5 w-full bg-muted rounded-sm overflow-hidden">
          <div
            class="h-full bg-primary transition-all duration-300"
            style="width: {(embeddingJobQuery.data.progress * 100).toFixed(0)}%"
          ></div>
        </div>
      </div>
    {/if}

    <div class="space-y-2">
      <label class="text-xs text-muted-foreground" for="embeddings-panel-model">Model</label>
      <select
        id="embeddings-panel-model"
        value={embeddingModel}
        onchange={(e) => onEmbeddingModelChange(e.currentTarget.value as "dinov2" | "clip")}
        class="w-full h-8 text-xs bg-background border border-border rounded px-2 text-foreground"
        disabled={!!embeddingJobId}
      >
        <option value="dinov2">DINOv2</option>
        <option value="clip">CLIP</option>
      </select>
    </div>

    <div class="space-y-2">
      <label class="text-xs text-muted-foreground" for="embeddings-panel-batch">Batch size</label>
      <input
        id="embeddings-panel-batch"
        type="number"
        value={batchSize}
        oninput={(e) => (batchSize = Math.max(1, parseInt(e.currentTarget.value) || 1))}
        class="w-full h-8 text-xs font-mono bg-background border border-border rounded px-2 text-foreground"
        disabled={!!embeddingJobId}
      />
    </div>

    <button
      onclick={handleRecomputeEmbeddings}
      disabled={!!embeddingJobId}
      class="w-full h-8 text-xs rounded flex items-center justify-center gap-1.5 bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
    >
      {#if embeddingJobId}
        <LoaderCircle class="h-3 w-3 animate-spin" />
      {:else}
        <RefreshCw class="h-3 w-3" />
      {/if}
      {embeddingJobId ? "Computing..." : embeddingExists ? "Re-compute Embeddings" : "Compute Embeddings"}
    </button>
  </div>
</div>
