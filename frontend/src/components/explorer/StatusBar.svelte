<script lang="ts">
  import { createQuery, useQueryClient } from "@tanstack/svelte-query";
  import { type DataPoint, type TaskType, isLabeledPoint } from "@/lib/mockData";
  import type { TrainingRun } from "@/lib/types";
  import { TASK_CONFIGS } from "@/lib/taskConfig";
  import { api } from "@/lib/api";
  import { Button } from "@/components/ui/button";
  import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
  import RefreshCw from "@lucide/svelte/icons/refresh-cw";
  import LoaderCircle from "@lucide/svelte/icons/loader-circle";
  import { toast } from "@/state/toast";

  interface Props {
    data: DataPoint[];
    dbPath: string;
    datasetPath: string;
    labeledThisSession: number;
    onOpenTraining: () => void;
    selectedRunId: string | null;
    taskType: TaskType;
  }

  let { data, dbPath, datasetPath, labeledThisSession, onOpenTraining, selectedRunId, taskType }: Props = $props();

  const queryClient = useQueryClient();
  let rescanning = $state(false);
  const taskConfig = $derived(TASK_CONFIGS[taskType]);
  const labeled = $derived(data.filter((d) => isLabeledPoint(d, taskType)).length);

  // Shares the ["training-runs", dbPath, taskType] cache with Explorer — one network request.
  const trainingRunsQuery = createQuery<TrainingRun[]>(() => ({
    queryKey: ["training-runs", dbPath, taskType],
    queryFn: () =>
      api.get<TrainingRun[]>(
        `/training/runs?db_path=${encodeURIComponent(dbPath)}&task_type=${encodeURIComponent(taskType)}`,
      ),
    enabled: !!dbPath,
  }));
  const trainingHistory = $derived(trainingRunsQuery.data ?? []);

  const activeRun = $derived(
    selectedRunId ? trainingHistory.find((r) => r.id === selectedRunId) ?? null : null,
  );

  async function handleRescan() {
    if (!datasetPath || rescanning) return;
    rescanning = true;
    try {
      const result = await api.post<{ new_images: number; counts: { total: number } }>(
        "/datasets/rescan",
        { path: datasetPath },
      );
      const n = result.new_images ?? 0;
      if (n === 0) {
        toast({ title: "No new images", description: "Dataset is up to date." });
      } else {
        toast({
          title: `Found ${n} new image${n === 1 ? "" : "s"}`,
          description: "Re-compute embeddings, then re-run predictions on new images.",
        });
        if (taskType === "multilabel-classification") {
          try {
            await api.post("/datasets/multilabel/seed", { db_path: dbPath });
          } catch (seedErr) {
            toast({
              title: "Failed to seed labels from folders",
              description: (seedErr as Error)?.message ?? "Unknown error",
              variant: "destructive",
            });
          }
        }
        queryClient.invalidateQueries({ queryKey: ["images-all", dbPath] });
        queryClient.invalidateQueries({ queryKey: ["images", dbPath] });
        queryClient.invalidateQueries({ queryKey: ["embeddings-status", dbPath] });
      }
    } catch (err) {
      toast({ title: "Rescan failed", description: (err as Error)?.message ?? "Unknown error", variant: "destructive" });
    } finally {
      rescanning = false;
    }
  }
</script>

<div class="h-10 border-t border-border bg-card flex items-center px-4 text-xs shrink-0">
  <!-- Left -->
  <div class="flex items-center gap-2 text-muted-foreground flex-1 min-w-0">
    <span class="truncate">
      {data.length.toLocaleString()} images — {labeled} labeled
    </span>
    <Tooltip>
      <TooltipTrigger>
        {#snippet child({ props })}
          <Button
            {...props}
            variant="ghost"
            size="sm"
            class="h-6 w-6 p-0"
            onclick={handleRescan}
            disabled={rescanning || !datasetPath}
            aria-label="Rescan dataset directory for new images"
          >
            {#if rescanning}
              <LoaderCircle class="h-3 w-3 animate-spin" />
            {:else}
              <RefreshCw class="h-3 w-3" />
            {/if}
          </Button>
        {/snippet}
      </TooltipTrigger>
      <TooltipContent side="top" class="text-xs">
        Rescan dataset folder for new images
      </TooltipContent>
    </Tooltip>
  </div>

  <!-- Center -->
  <div class="flex items-center gap-3 px-4">
    {#if activeRun}
      <span class="text-muted-foreground">
        {activeRun.display_name}: {taskConfig.metrics.primary} {((activeRun.best_accuracy ?? 0) * 100).toFixed(1)}% | {taskConfig.metrics.secondary} {((activeRun.best_f1 ?? 0) * 100).toFixed(1)}%
      </span>
    {:else}
      <span class="text-muted-foreground">No model selected</span>
    {/if}
  </div>

  <!-- Right -->
  <div class="flex items-center gap-3 flex-1 justify-end min-w-0">
    <span class="text-muted-foreground">
      {trainingHistory.filter((r) => r.status === "done").length} runs
      {#if labeledThisSession > 0}<span class="text-foreground"> ({labeledThisSession} labeled this session)</span>{/if}
    </span>
    <Button size="sm" class="h-6 text-xs px-3" onclick={onOpenTraining}>
      Retrain
    </Button>
  </div>
</div>
