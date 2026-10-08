<script lang="ts">
  import { createQuery, useQueryClient } from "@tanstack/svelte-query";
  import { Button } from "@/components/ui/button";
  import { Input } from "@/components/ui/input";
  import { Label } from "@/components/ui/label";
  import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
  import FolderOpen from "@lucide/svelte/icons/folder-open";
  import LoaderCircle from "@lucide/svelte/icons/loader-circle";
  import CircleCheckBig from "@lucide/svelte/icons/circle-check-big";
  import Circle from "@lucide/svelte/icons/circle";
  import { toast } from "@/state/toast";
  import { api } from "@/lib/api";
  import { navigate } from "@/router.svelte";
  import { createJobPoller } from "@/state/jobPoller";
  import type { TaskType } from "@/lib/mockData";
  import { TASK_CONFIGS } from "@/lib/taskConfig";
  import type { TrainingRun } from "@/lib/types";
  import DirectoryPicker from "@/components/DirectoryPicker.svelte";

  interface LoadResult {
    db_path: string;
    classes: string[];
    splits: string[];
    counts: { total: number; labeled: number; by_split: Record<string, number> };
  }

  interface EmbeddingInfo {
    exists: boolean;
    count?: number;
    created_at?: string;
  }

  type EmbeddingStatus = Record<"dinov2" | "clip", EmbeddingInfo>;

  const queryClient = useQueryClient();

  let datasetPath = $state("");
  let taskType = $state<TaskType>("classification");
  let isScanning = $state(false);
  let loadResult = $state<LoadResult | null>(null);
  let embeddingModel = $state<"dinov2" | "clip">("dinov2");
  let batchSize = $state(32);
  let activeJobId = $state<string | null>(null);
  let pickerOpen = $state(false);

  const hasPath = $derived(datasetPath.trim().length > 0);

  const taskOptions = [
    { value: "classification", label: "Classification", desc: "Assign one label per image" },
    { value: "multilabel-classification", label: "Multi-Label", desc: "Assign several labels per image" },
    { value: "object-detection", label: "Object Detection", desc: "Draw bounding boxes on images" },
  ] as const;

  // Fetch embedding status once dataset is loaded
  const embeddingStatusQuery = createQuery<EmbeddingStatus>(() => ({
    queryKey: ["embeddings-status", loadResult?.db_path],
    queryFn: () =>
      api.get<EmbeddingStatus>(
        `/embeddings/status?db_path=${encodeURIComponent(loadResult!.db_path)}`,
      ),
    enabled: !!loadResult?.db_path,
  }));

  // Fetch existing training runs
  const trainingRunsQuery = createQuery<TrainingRun[]>(() => ({
    queryKey: ["training-runs", loadResult?.db_path, taskType],
    queryFn: () =>
      api.get<TrainingRun[]>(
        `/training/runs?db_path=${encodeURIComponent(loadResult!.db_path)}&task_type=${encodeURIComponent(taskType)}`,
      ),
    enabled: !!loadResult?.db_path,
  }));

  const completedRuns = $derived((trainingRunsQuery.data ?? []).filter((r) => r.status === "done"));

  // Poll active compute job
  const jobQuery = createJobPoller(
    () => activeJobId,
    () => loadResult?.db_path ?? null,
  );

  // When job completes, refresh status and clear job id (keyed on status only,
  // mirroring the React effect's narrowed dep array).
  let prevJobStatus: string | undefined;
  $effect(() => {
    const jobStatus = jobQuery.data;
    const status = jobStatus?.status;
    if (status === prevJobStatus) return;
    prevJobStatus = status;
    if (status === "done") {
      activeJobId = null;
      queryClient.invalidateQueries({ queryKey: ["embeddings-status", loadResult?.db_path] });
    }
    if (status === "failed") {
      activeJobId = null;
      const isOom = jobStatus.error?.toLowerCase().includes("out of memory");
      if (isOom) {
        // Halve batch size (floor division, minimum 1) as spec requires
        batchSize = Math.max(1, Math.floor(batchSize / 2));
      }
      toast({ title: "Embedding failed", description: jobStatus.error ?? "Unknown error", variant: "destructive" });
    }
  });

  async function handleLoadDataset() {
    isScanning = true;
    try {
      loadResult = await api.post<LoadResult>("/datasets/load", { path: datasetPath.trim() });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      toast({ title: "Failed to load dataset", description: message, variant: "destructive" });
    } finally {
      isScanning = false;
    }
  }

  async function handleComputeEmbeddings() {
    if (!loadResult) return;
    try {
      const { job_id } = await api.post<{ job_id: string }>("/embeddings/compute", {
        db_path: loadResult.db_path,
        model: embeddingModel,
        batch_size: batchSize,
      });
      activeJobId = job_id;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      toast({ title: "Failed to start job", description: message, variant: "destructive" });
    }
  }

  function handleOpenExplorer() {
    if (!loadResult) return;
    // Handoff (spec §7): write the dataset payload to sessionStorage, then navigate.
    sessionStorage.setItem(
      "kumo:datasetState",
      JSON.stringify({
        taskType,
        datasetPath: datasetPath.trim(),
        dbPath: loadResult.db_path,
        classes: loadResult.classes,
        splits: loadResult.splits,
        counts: loadResult.counts,
        embeddingModel,
        activeEmbeddingJobId: activeJobId,
        embeddingBatchSize: batchSize,
      }),
    );
    navigate("/explorer");
  }

  const isComputing = $derived(
    !!activeJobId && jobQuery.data?.status !== "done" && jobQuery.data?.status !== "failed",
  );
  const currentModelStatus = $derived(embeddingStatusQuery.data?.[embeddingModel]);
</script>

<div class="min-h-screen flex items-center justify-center bg-background p-4">
  <div class="w-full max-w-[600px]">
    <div class="mb-8">
      <span class="text-xs font-semibold tracking-widest uppercase text-muted-foreground">iits</span>
      <h1 class="text-2xl font-semibold text-foreground mt-1">New Project</h1>
      <div class="w-12 h-0.5 bg-primary mt-2"></div>
    </div>

    <div class="bg-card border border-border rounded-lg p-6 space-y-6">
      <!-- Task Type -->
      <div class="space-y-2">
        <Label class="text-sm font-medium">Task Type</Label>
        <div class="grid grid-cols-3 gap-2">
          {#each taskOptions as { value, label, desc } (value)}
            <button
              onclick={() => (taskType = value)}
              class={`p-3 rounded-md border text-left transition-colors ${
                taskType === value ? "border-primary bg-primary/5" : "border-border hover:border-foreground/20"
              }`}
            >
              <p class={`text-sm font-medium ${taskType === value ? "text-primary" : "text-foreground"}`}>{label}</p>
              <p class="text-xs text-muted-foreground mt-0.5">{desc}</p>
            </button>
          {/each}
        </div>
      </div>

      <!-- Dataset Path -->
      <div class="space-y-2">
        <Label for="dataset-path" class="text-sm font-medium">Dataset Path</Label>
        <div class="flex gap-2">
          <div class="relative flex-1">
            <Input
              id="dataset-path"
              placeholder="/data/my-dataset"
              bind:value={datasetPath}
              oninput={() => (loadResult = null)}
              class="pr-9 font-mono text-sm"
              onkeydown={(e) => {
                if (e.key === "Enter" && hasPath && !isScanning) handleLoadDataset();
              }}
            />
            <button
              type="button"
              onclick={() => (pickerOpen = true)}
              class="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 rounded-sm text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
              aria-label="Browse for folder"
              title="Browse for folder"
            >
              <FolderOpen class="h-4 w-4" />
            </button>
          </div>
          <Button onclick={handleLoadDataset} disabled={!hasPath || isScanning} variant="secondary" size="default">
            {#if isScanning}
              <LoaderCircle class="h-4 w-4 animate-spin" />
            {:else}
              Load
            {/if}
          </Button>
        </div>
      </div>

      <DirectoryPicker
        open={pickerOpen}
        onClose={() => (pickerOpen = false)}
        onSelect={(p) => {
          datasetPath = p;
          loadResult = null;
        }}
        initialPath={datasetPath}
      />

      <!-- Dataset summary + embedding panel — shown after loading -->
      {#if loadResult}
        <div class="text-xs text-muted-foreground border border-border rounded px-3 py-2 space-y-0.5">
          <div class="flex justify-between">
            <span>Images</span>
            <span class="font-medium text-foreground">{loadResult.counts.total.toLocaleString()}</span>
          </div>
          <div class="flex justify-between">
            <span>Labeled</span>
            <span class="font-medium text-foreground">
              {loadResult.counts.total > 0
                ? `${((loadResult.counts.labeled / loadResult.counts.total) * 100).toFixed(1)}%`
                : "0%"}
            </span>
          </div>
          <div class="flex justify-between">
            <span>Classes</span>
            <span class="font-medium text-foreground">{loadResult.classes.length.toLocaleString()}</span>
          </div>
          <div class="font-medium text-foreground max-h-20 overflow-y-auto scrollbar-thin mt-0.5">
            {loadResult.classes.join(", ")}
          </div>
        </div>

        <!-- Trained Models -->
        {#if completedRuns.length > 0}
          <div class="text-xs border border-border rounded px-3 py-2 space-y-1.5">
            <span class="font-medium text-foreground">Trained Models</span>
            {#each completedRuns as run (run.id)}
              <div class="flex justify-between text-muted-foreground">
                <span>{run.display_name}</span>
                <span>
                  {TASK_CONFIGS[taskType].metrics.primaryKey}: {((run.best_accuracy ?? 0) * 100).toFixed(1)}% | {TASK_CONFIGS[taskType].metrics.secondaryKey}: {((run.best_f1 ?? 0) * 100).toFixed(1)}%
                  <span class="ml-2 text-muted-foreground/60">{run.label_count} labels</span>
                </span>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Embedding panel -->
        <div class="space-y-3 border border-border rounded-lg p-4">
          <Label class="text-sm font-medium">Embeddings</Label>

          <!-- Status line -->
          {#if currentModelStatus?.exists}
            <div class="flex items-center gap-2 text-xs text-muted-foreground">
              <CircleCheckBig class="h-3.5 w-3.5 text-green-500 shrink-0" />
              <span>
                {embeddingModel === "dinov2" ? "DINOv2" : "CLIP"} — {currentModelStatus.count?.toLocaleString()} images
                {currentModelStatus.created_at ? ` (${currentModelStatus.created_at.slice(0, 10)})` : ""}
              </span>
            </div>
          {:else}
            <div class="flex items-center gap-2 text-xs text-muted-foreground">
              <Circle class="h-3.5 w-3.5 shrink-0" />
              <span>No embeddings computed</span>
            </div>
          {/if}

          <!-- Progress bar while computing -->
          {#if isComputing && jobQuery.data}
            <div class="space-y-1">
              <div class="flex justify-between text-xs text-muted-foreground">
                <span>{jobQuery.data.message ?? "Computing..."}</span>
                <span>{jobQuery.data.total > 0 ? `${jobQuery.data.completed} / ${jobQuery.data.total}` : "…"}</span>
              </div>
              <div class="h-1.5 w-full bg-muted rounded-sm overflow-hidden">
                <div
                  class="h-full bg-primary transition-all duration-300"
                  style="width: {(jobQuery.data.progress * 100).toFixed(0)}%"
                ></div>
              </div>
            </div>
          {/if}

          <div class="flex gap-2 items-end">
            <div class="space-y-1 flex-1">
              <Label class="text-xs text-muted-foreground">Model</Label>
              <Select type="single" bind:value={embeddingModel} disabled={isComputing}>
                <SelectTrigger class="h-8 text-xs">
                  {embeddingModel === "dinov2" ? "DINOv2 (384-dim)" : "CLIP (768-dim)"}
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="dinov2">DINOv2 (384-dim)</SelectItem>
                  <SelectItem value="clip">CLIP (768-dim)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div class="space-y-1">
              <Label class="text-xs text-muted-foreground">Batch</Label>
              <Input
                type="number"
                value={batchSize}
                oninput={(e) => (batchSize = Math.max(1, parseInt(e.currentTarget.value) || 1))}
                class="h-8 w-20 font-mono text-xs"
                disabled={isComputing}
              />
            </div>
            <Button
              size="sm"
              variant="secondary"
              class="h-8 text-xs"
              onclick={handleComputeEmbeddings}
              disabled={isComputing}
            >
              {#if isComputing}
                <LoaderCircle class="h-3.5 w-3.5 animate-spin" />
              {:else}
                {currentModelStatus?.exists ? "Recompute" : "Compute"}
              {/if}
            </Button>
          </div>
        </div>

        <!-- Navigation -->
        <div class="flex gap-2">
          <Button class="flex-1 text-sm" onclick={handleOpenExplorer}>Open Explorer →</Button>
        </div>
      {/if}

      {#if !hasPath}
        <p class="text-xs text-muted-foreground text-center">Enter a dataset path to begin</p>
      {/if}
    </div>
  </div>
</div>
