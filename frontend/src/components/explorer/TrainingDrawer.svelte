<script lang="ts">
  import { untrack } from "svelte";
  import { createQuery, useQueryClient } from "@tanstack/svelte-query";
  import { type DataPoint, type TaskType, getDatasetStats, effectiveSplit, isLabeledPoint } from "@/lib/mockData";
  import type { TrainingRun, AdvancedSettings } from "@/lib/types";
  import { ADVANCED_DEFAULTS } from "@/lib/types";
  import { TASK_CONFIGS } from "@/lib/taskConfig";
  import { createTrainingPoller } from "@/state/trainingPoller";
  import { api } from "@/lib/api";
  import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
  import { Button } from "@/components/ui/button";
  import { Input } from "@/components/ui/input";
  import { Label } from "@/components/ui/label";
  import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
  import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
  import { Slider } from "@/components/ui/slider";
  import { Switch } from "@/components/ui/switch";
  import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
  import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
  import { Progress } from "@/components/ui/progress";
  import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "@/components/ui/context-menu";
  import ChevronDown from "@lucide/svelte/icons/chevron-down";
  import AlertTriangle from "@lucide/svelte/icons/triangle-alert";
  import BarChart3 from "@lucide/svelte/icons/chart-column";
  import Download from "@lucide/svelte/icons/download";
  import Info from "@lucide/svelte/icons/info";
  import Package from "@lucide/svelte/icons/package";
  import Pencil from "@lucide/svelte/icons/pencil";
  import Play from "@lucide/svelte/icons/play";
  import Square from "@lucide/svelte/icons/square";
  import Trash2 from "@lucide/svelte/icons/trash-2";
  import Settings2 from "@lucide/svelte/icons/settings-2";
  import { toast } from "@/state/toast";
  import MetricsCharts from "./MetricsCharts.svelte";
  import AdvancedSettingsDialog from "./AdvancedSettingsDialog.svelte";
  import TrainingConfigDialog from "./TrainingConfigDialog.svelte";
  import PerClassMetricsTable from "./PerClassMetricsTable.svelte";
  import DatasetDiff from "./DatasetDiff.svelte";

  function formatEta(seconds: number): string {
    if (seconds < 60) return `~${Math.round(seconds)}s`;
    if (seconds < 3600) return `~${Math.round(seconds / 60)}m`;
    return `~${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  }

  interface Props {
    open: boolean;
    onClose: () => void;
    data: DataPoint[];
    dbPath: string;
    classNames: string[];
    taskType: TaskType;
    selectedRunId: string | null;
    onSelectRun: (id: string | null) => void;
  }

  let { open, onClose, data, dbPath, classNames, taskType, selectedRunId, onSelectRun }: Props = $props();

  const queryClient = useQueryClient();
  const taskConfig = $derived(TASK_CONFIGS[taskType]);
  // svelte-ignore state_referenced_locally
  const initConfig = TASK_CONFIGS[taskType];

  let model = $state(initConfig.models[0].value);
  let customModelId = $state("");
  let localCheckpointPath = $state("");
  let localCheckpointBackbone = $state("");
  type CkptProbe = "idle" | "probing" | { model_id: string; display_name: string } | null;
  let ckptProbe = $state<CkptProbe>("idle");
  let checkpointRunId = $state<string | null>(null);
  let lr = $state(initConfig.defaults.lr);
  let epochs = $state("10");
  let batchSize = $state(initConfig.defaults.batchSize);
  let splitRatio = $state(80);
  let validationMode = $state<"user_defined" | "random">("user_defined");
  let augmentation = $state(true);
  let hyperOpen = $state(false);
  let labelsOpen = $state(false);
  let historyOpen = $state(false);
  let activeRunId = $state<string | null>(null);
  let clearedRunId: string | null = null;
  let etaStart = $state<{ time: number; step: number } | null>(null);
  let compareAll = $state(false);
  let advancedOpen = $state(false);
  let renamingRunId = $state<string | null>(null);
  let renameValue = $state("");
  let configRunId = $state<string | null>(null);
  let classBalanceStrategy = $state("none");
  let advancedSettings = $state<AdvancedSettings>({
    ...ADVANCED_DEFAULTS,
    ...(initConfig.defaults.imageSize != null ? { image_size: initConfig.defaults.imageSize } : {}),
    ...(initConfig.defaults.gradientClipVal != null ? { gradient_clip_val: initConfig.defaults.gradientClipVal } : {}),
    ...(initConfig.defaults.warmupEpochs != null ? { warmup_epochs: initConfig.defaults.warmupEpochs } : {}),
  });

  // Reset model when the task config changes (taskType is fixed per dataset in
  // practice; this mirrors the React effect).
  $effect(() => {
    model = taskConfig.models[0].value;
  });

  // Clear checkpoint fields when switching away from the respective model.
  $effect(() => {
    if (model !== "checkpoint") checkpointRunId = null;
    if (model !== "local_checkpoint") {
      localCheckpointPath = "";
      localCheckpointBackbone = "";
      ckptProbe = "idle";
    }
  });

  const stats = $derived(getDatasetStats(data, taskType));
  const labeledData = $derived(data.filter((d) => isLabeledPoint(d, taskType)));

  // Poll active training run
  const trainingQuery = createTrainingPoller(() => activeRunId, () => dbPath);
  const activeRun = $derived(trainingQuery.data);

  // Fetch training history — shared cache with Explorer/StatusBar.
  const historyQuery = createQuery<TrainingRun[]>(() => ({
    queryKey: ["training-runs", dbPath, taskType],
    queryFn: () =>
      api.get<TrainingRun[]>(
        `/training/runs?db_path=${encodeURIComponent(dbPath)}&task_type=${encodeURIComponent(taskType)}`,
      ),
    enabled: !!dbPath,
  }));
  const trainingHistory = $derived(historyQuery.data ?? []);

  // Auto-resume: if there's a running/queued/predicting training, reconnect to
  // it — except the one recorded in the clearedRunId guard.
  $effect(() => {
    if (activeRunId) return;
    const running = trainingHistory.find(
      (r) => ["queued", "running", "predicting"].includes(r.status) && r.id !== clearedRunId,
    );
    if (running) {
      activeRunId = running.id;
    } else {
      clearedRunId = null;
    }
  });

  // Eligible runs for "Trained" checkpoint picker
  const checkpointRuns = $derived(
    trainingHistory.filter((r) => r.has_checkpoint && (r.status === "done" || r.status === "stopped")),
  );

  // When the active run completes, clear it, refresh history, and auto-select.
  // Keyed on status only (prev-status guard) — mirrors React's narrowed deps.
  let prevRunStatus: string | undefined = undefined;
  $effect(() => {
    const status = activeRun?.status;
    if (status === prevRunStatus) return;
    prevRunStatus = status;
    if (status !== "done" && status !== "failed") return;
    const run = untrack(() => activeRun)!;
    if (status === "done") {
      toast({
        title: "Training complete",
        description: `${taskConfig.metrics.primary}: ${((run.best_accuracy ?? 0) * 100).toFixed(1)}% | ${taskConfig.metrics.secondary}: ${((run.best_f1 ?? 0) * 100).toFixed(1)}%`,
      });
      onSelectRun(run.id);
    } else {
      toast({ title: "Training failed", description: run.error ?? "Unknown error", variant: "destructive" });
    }
    clearedRunId = untrack(() => activeRunId);
    activeRunId = null;
    queryClient.invalidateQueries({ queryKey: ["training-runs", dbPath] });
  });

  // Class counts for label summary
  const classCounts = $derived.by(() => {
    if (taskType === "object-detection") {
      // For detection: show total annotated images count
      return { "Annotated images": labeledData.length };
    }
    if (taskType === "multilabel-classification") {
      // Each tag on an image counts once, under its own class.
      const counts: Record<string, number> = {};
      labeledData.forEach((d) => {
        d.labels.forEach((cls) => {
          counts[cls] = (counts[cls] || 0) + 1;
        });
      });
      return counts;
    }
    const counts: Record<string, number> = {};
    labeledData.forEach((d) => {
      const effectiveLabel = d.label || d.groundTruth;
      if (effectiveLabel) counts[effectiveLabel] = (counts[effectiveLabel] || 0) + 1;
    });
    return counts;
  });

  // Per-class effective-split breakdown. When any image is assigned to 'valid'
  // (manually or via folder structure), training honors these splits instead
  // of the random ratio; reflect that in the preview.
  const splitBreakdown = $derived.by(() => {
    const result: Record<string, { train: number; valid: number; test: number }> = {};
    let hasVal = false;
    let totalTest = 0;

    if (taskType === "multilabel-classification") {
      // An image with N tags contributes a row-count to each of its N tags;
      // hasVal/totalTest track distinct images, not per-tag rows.
      for (const d of labeledData) {
        const s = effectiveSplit(d);
        if (s === "valid") hasVal = true;
        else if (s === "test") totalTest++;
        for (const key of d.labels) {
          if (!result[key]) result[key] = { train: 0, valid: 0, test: 0 };
          if (s === "valid") result[key].valid++;
          else if (s === "test") result[key].test++;
          else result[key].train++;
        }
      }
      return { result, hasVal, totalTest };
    }

    for (const d of labeledData) {
      const key = taskType === "object-detection" ? "Annotated images" : d.label || d.groundTruth;
      if (!key) continue;
      if (!result[key]) result[key] = { train: 0, valid: 0, test: 0 };
      const s = effectiveSplit(d);
      if (s === "valid") {
        result[key].valid++;
        hasVal = true;
      } else if (s === "test") {
        result[key].test++;
        totalTest++;
      } else result[key].train++;
    }
    return { result, hasVal, totalTest };
  });

  // Whether training honors the user-assigned splits. Only possible when a val
  // set is defined; the toggle lets the user fall back to random sampling.
  const useAssigned = $derived(splitBreakdown.hasVal && validationMode === "user_defined");

  const hasImbalance = $derived.by(() => {
    const values = Object.values(classCounts);
    if (values.length < 2) return false;
    return Math.max(...values) / Math.min(...values) > 3;
  });

  const isTraining = $derived(
    !!activeRunId &&
      (!activeRun || activeRun.status === "queued" || activeRun.status === "running" || activeRun.status === "predicting"),
  );
  const trainingDone = $derived(activeRun?.status === "done");

  async function probeCheckpoint(path: string) {
    if (!path.trim()) {
      ckptProbe = "idle";
      return;
    }
    ckptProbe = "probing";
    try {
      const result = await api.post<{ model_id: string | null; display_name: string | null }>(
        "/training/probe-checkpoint",
        { path },
      );
      ckptProbe = result.model_id
        ? { model_id: result.model_id, display_name: result.display_name ?? result.model_id }
        : null;
    } catch {
      ckptProbe = null;
    }
  }

  async function startTraining() {
    try {
      const { run_id } = await api.post<{ run_id: string }>("/training/start", {
        db_path: dbPath,
        model,
        model_id:
          model === "custom"
            ? customModelId
            : model === "local_checkpoint"
              ? typeof ckptProbe === "object" && ckptProbe !== null
                ? ckptProbe.model_id
                : localCheckpointBackbone === "custom"
                  ? customModelId
                  : localCheckpointBackbone
              : undefined,
        local_checkpoint_path: model === "local_checkpoint" ? localCheckpointPath : undefined,
        checkpoint_run_id: model === "checkpoint" ? checkpointRunId : undefined,
        task_type: taskType,
        lr: parseFloat(lr),
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        augmentation,
        split_ratio: splitRatio / 100,
        validation_mode: useAssigned ? "user_defined" : "random",
        class_names: classNames,
        num_classes: classNames.length,
        ...$state.snapshot(advancedSettings),
        class_balance_strategy: classBalanceStrategy,
      });
      activeRunId = run_id;
      toast({ title: "Training started", description: `Training for ${epochs} epochs` });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast({ title: "Failed to start training", description: message, variant: "destructive" });
    }
  }

  async function stopTraining() {
    if (!activeRunId) return;
    try {
      await api.post(`/training/runs/${activeRunId}/stop?db_path=${encodeURIComponent(dbPath)}`, {});
      activeRunId = null;
      queryClient.invalidateQueries({ queryKey: ["training-runs", dbPath] });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast({ title: "Failed to stop training", description: message, variant: "destructive" });
    }
  }

  // ETA anchor: reset when the active run changes…
  $effect(() => {
    void activeRunId;
    etaStart = null;
  });

  // …and capture once the run is running with ≥1 step.
  $effect(() => {
    const status = activeRun?.status;
    const step = activeRun?.current_step ?? 0;
    if (status === "running" && step > 0 && !untrack(() => etaStart)) {
      etaStart = { time: Date.now(), step };
    }
  });

  // Show the ETA only after ≥2 steps of progress and ≥3s elapsed.
  function etaText(): string | null {
    const start = etaStart;
    const run = activeRun;
    if (!start || !run || run.total_steps <= 0) return null;
    const elapsed = (Date.now() - start.time) / 1000;
    const done = run.current_step - start.step;
    if (done < 2 || elapsed < 3) return null;
    return formatEta((run.total_steps - run.current_step) / (done / elapsed));
  }

  function autofocus(node: HTMLElement) {
    node.focus();
  }

  async function saveRename(runId: string) {
    if (renamingRunId !== runId) return;
    const trimmed = renameValue.trim();
    renamingRunId = null;
    if (!trimmed) return;
    try {
      await api.patch(`/training/runs/${runId}?db_path=${encodeURIComponent(dbPath)}`, { display_name: trimmed });
      queryClient.invalidateQueries({ queryKey: ["training-runs", dbPath] });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast({ title: "Failed to rename run", description: message, variant: "destructive" });
    }
  }

  async function predictRun(run: TrainingRun) {
    try {
      await api.post(`/training/runs/${run.id}/predict?db_path=${encodeURIComponent(dbPath)}`, {});
      activeRunId = run.id;
      queryClient.invalidateQueries({ queryKey: ["training-runs", dbPath] });
      toast({ title: "Generating predictions", description: "Running inference on all images..." });
    } catch (err) {
      toast({ title: "Failed", description: (err as Error)?.message ?? "Unknown error", variant: "destructive" });
    }
  }

  async function predictNewImages(run: TrainingRun) {
    try {
      const resp = await api.post<{ status: string; predicted?: number }>(
        `/training/runs/${run.id}/predict?db_path=${encodeURIComponent(dbPath)}&only_new=true`,
        {},
      );
      if (resp.status === "noop") {
        toast({ title: "Nothing to predict", description: "All images already have predictions for this run." });
      } else {
        activeRunId = run.id;
        queryClient.invalidateQueries({ queryKey: ["training-runs", dbPath] });
        toast({ title: "Predicting new images", description: "Running inference on images without predictions..." });
      }
    } catch (err) {
      toast({ title: "Failed", description: (err as Error)?.message ?? "Unknown error", variant: "destructive" });
    }
  }

  async function deleteRun(run: TrainingRun) {
    try {
      await api.delete(`/training/runs/${run.id}?db_path=${encodeURIComponent(dbPath)}`);
      queryClient.invalidateQueries({ queryKey: ["training-runs", dbPath] });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast({ title: "Failed to delete run", description: message, variant: "destructive" });
    }
  }

  const progressPercent = $derived(
    activeRun ? (activeRun.total_steps > 0 ? (activeRun.current_step / activeRun.total_steps) * 100 : 0) : 0,
  );

  const startDisabled = $derived(
    stats.labeled === 0 ||
      (model === "checkpoint" && !checkpointRunId) ||
      (model === "local_checkpoint" &&
        (!localCheckpointPath.trim() ||
          ckptProbe === "idle" ||
          ckptProbe === "probing" ||
          (ckptProbe === null && (!localCheckpointBackbone || (localCheckpointBackbone === "custom" && !customModelId.trim()))))),
  );
</script>

<Sheet {open} onOpenChange={(o) => !o && onClose()}>
  <SheetContent class="w-[400px] sm:w-[400px] overflow-y-auto scrollbar-thin p-0">
    <SheetHeader class="p-4 pb-3 border-b border-border">
      <SheetTitle class="text-base font-semibold">Train Model</SheetTitle>
    </SheetHeader>

    <div class="p-4 space-y-5 text-sm">
      <!-- Model Selector -->
      <div class="space-y-2">
        <Label class="text-xs font-medium">Model</Label>
        <Select type="single" value={model} onValueChange={(v) => (model = v)}>
          <SelectTrigger class="h-8 text-xs">
            {taskConfig.models.find((m) => m.value === model)?.label ?? model}
          </SelectTrigger>
          <SelectContent>
            {#each taskConfig.models as opt (opt.value)}
              <SelectItem value={opt.value}>{opt.label}</SelectItem>
            {/each}
          </SelectContent>
        </Select>
        {#if model === "checkpoint"}
          {#if checkpointRuns.length > 0}
            {@const selected = checkpointRuns.find((r) => r.id === checkpointRunId)}
            <Select type="single" value={checkpointRunId ?? ""} onValueChange={(v) => (checkpointRunId = v)}>
              <SelectTrigger class="h-8 text-xs">
                {selected
                  ? `${selected.display_name} — ${((selected.best_accuracy ?? 0) * 100).toFixed(1)}% ${taskConfig.metrics.primaryKey} (${selected.created_at?.slice(0, 10)})`
                  : "Select a training run"}
              </SelectTrigger>
              <SelectContent>
                {#each checkpointRuns as run (run.id)}
                  <SelectItem value={run.id}>
                    {run.display_name} — {((run.best_accuracy ?? 0) * 100).toFixed(1)}% {taskConfig.metrics.primaryKey} ({run.created_at?.slice(0, 10)})
                  </SelectItem>
                {/each}
              </SelectContent>
            </Select>
          {:else}
            <p class="text-xs text-muted-foreground">No training runs with checkpoints available.</p>
          {/if}
        {/if}
        {#if model === "local_checkpoint"}
          <Input
            placeholder="/path/to/model.ckpt"
            value={localCheckpointPath}
            oninput={(e) => {
              localCheckpointPath = e.currentTarget.value;
              ckptProbe = "idle";
              localCheckpointBackbone = "";
            }}
            onblur={(e) => probeCheckpoint(e.currentTarget.value)}
            class="h-8 text-xs font-mono"
          />
          {#if ckptProbe === "probing"}
            <p class="text-xs text-muted-foreground">Detecting model…</p>
          {/if}
          {#if typeof ckptProbe === "object" && ckptProbe !== null}
            <p class="text-xs text-muted-foreground">
              Auto-detected: <span class="text-foreground font-medium">{ckptProbe.display_name}</span>
            </p>
          {/if}
          {#if ckptProbe === null}
            <Select type="single" value={localCheckpointBackbone} onValueChange={(v) => (localCheckpointBackbone = v)}>
              <SelectTrigger class="h-8 text-xs">
                {taskConfig.models.find((m) => m.value === localCheckpointBackbone)?.label ?? "Select backbone architecture"}
              </SelectTrigger>
              <SelectContent>
                {#each taskConfig.models.filter((m) => m.value !== "checkpoint" && m.value !== "local_checkpoint") as opt (opt.value)}
                  <SelectItem value={opt.value}>{opt.label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
            {#if localCheckpointBackbone === "custom"}
              <Input
                placeholder="HuggingFace repo ID or local directory path"
                bind:value={customModelId}
                class="h-8 text-xs font-mono"
              />
            {/if}
          {/if}
        {/if}
        {#if model === "custom"}
          <Input
            placeholder="HuggingFace repo ID or local directory path"
            bind:value={customModelId}
            class="h-8 text-xs font-mono"
          />
        {/if}
      </div>

      <!-- Hyperparameters -->
      <Collapsible bind:open={hyperOpen}>
        <CollapsibleTrigger class="flex items-center gap-2 text-xs font-medium w-full">
          <ChevronDown class={`h-3.5 w-3.5 transition-transform duration-200 ${hyperOpen ? "" : "-rotate-90"}`} />
          Hyperparameters
        </CollapsibleTrigger>
        <CollapsibleContent class="mt-3 space-y-3">
          <div class="grid grid-cols-2 gap-3">
            <div class="space-y-1">
              <Label class="text-xs">Learning Rate</Label>
              <Input bind:value={lr} class="h-8 text-xs font-mono" />
            </div>
            <div class="space-y-1">
              <Label class="text-xs">Epochs</Label>
              <Input bind:value={epochs} class="h-8 text-xs font-mono" />
            </div>
            <div class="space-y-1">
              <Label class="text-xs">Batch Size</Label>
              <Input bind:value={batchSize} class="h-8 text-xs font-mono" />
            </div>
            <div class="flex items-end gap-2 pb-1">
              <Switch bind:checked={augmentation} />
              <Label class="text-xs">Augmentation</Label>
            </div>
          </div>
          <Button variant="outline" size="sm" class="w-full text-xs h-8 gap-1.5" onclick={() => (advancedOpen = true)}>
            <Settings2 class="h-3 w-3" />
            Advanced Settings
          </Button>
        </CollapsibleContent>
      </Collapsible>

      <!-- Labels -->
      <Collapsible bind:open={labelsOpen}>
        <CollapsibleTrigger class="flex items-center gap-2 text-xs font-medium w-full">
          <ChevronDown class={`h-3.5 w-3.5 transition-transform duration-200 ${labelsOpen ? "" : "-rotate-90"}`} />
          Labels
        </CollapsibleTrigger>
        <CollapsibleContent class="mt-3 space-y-2">
          {#if splitBreakdown.hasVal}
            <ToggleGroup
              type="single"
              size="sm"
              variant="outline"
              bind:value={() => validationMode, (v) => v && (validationMode = v as "user_defined" | "random")}
              class="grid grid-cols-2 gap-1"
            >
              <ToggleGroupItem value="user_defined" class="h-7 text-xs data-[state=on]:bg-primary data-[state=on]:text-primary-foreground">
                User defined
              </ToggleGroupItem>
              <ToggleGroupItem value="random" class="h-7 text-xs data-[state=on]:bg-primary data-[state=on]:text-primary-foreground">
                Random sample
              </ToggleGroupItem>
            </ToggleGroup>
          {/if}
          {#if useAssigned}
            <div class="text-xs text-muted-foreground bg-muted/50 border border-border rounded-md px-2 py-1.5">
              Using assigned splits from images marked in the explorer / folder structure.
              {#if splitBreakdown.totalTest > 0}
                {` ${splitBreakdown.totalTest} test image${splitBreakdown.totalTest === 1 ? "" : "s"} held out.`}
              {/if}
            </div>
          {:else}
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs text-muted-foreground">
                <span>Train/Val Split</span>
                <span>{splitRatio}% / {100 - splitRatio}%</span>
              </div>
              <Slider type="single" min={50} max={95} step={5} value={splitRatio} onValueChange={(v) => (splitRatio = v)} />
            </div>
          {/if}
          <div class="border border-border rounded-md overflow-auto max-h-72">
            <table class="w-full text-xs">
              <thead>
                <tr class="bg-muted/50">
                  <th class="text-right px-2 py-1.5 font-medium text-muted-foreground sticky top-0 bg-background">Train</th>
                  <th class="text-right px-2 py-1.5 font-medium text-muted-foreground sticky top-0 bg-background">Val</th>
                  {#if useAssigned}
                    <th class="text-right px-2 py-1.5 font-medium text-muted-foreground sticky top-0 bg-background">Test</th>
                  {/if}
                  <th class="text-left px-2 py-1.5 font-medium text-muted-foreground sticky top-0 bg-background">Class</th>
                </tr>
              </thead>
              <tbody>
                {#each Object.entries(classCounts).sort((a, b) => b[1] - a[1]) as [cls, count] (cls)}
                  {@const eff = splitBreakdown.result[cls]}
                  {@const train = useAssigned ? (eff?.train ?? 0) : Math.round((count * splitRatio) / 100)}
                  {@const val = useAssigned ? (eff?.valid ?? 0) : Math.round((count * (100 - splitRatio)) / 100)}
                  {@const test = eff?.test ?? 0}
                  <tr class="border-t border-border">
                    <td class="text-right px-2 py-1.5 tabular-nums text-foreground">{train}</td>
                    <td class="text-right px-2 py-1.5 tabular-nums text-foreground">{val}</td>
                    {#if useAssigned}
                      <td class="text-right px-2 py-1.5 tabular-nums text-muted-foreground">{test}</td>
                    {/if}
                    <td class="px-2 py-1.5 text-foreground whitespace-nowrap">{cls}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
          {#if taskType === "multilabel-classification"}
            <p class="text-xs text-muted-foreground">Images with multiple labels are counted once per label.</p>
          {/if}
          {#if hasImbalance && taskType === "classification"}
            <div class="space-y-2 p-2 bg-amber-50 dark:bg-amber-950/20 rounded-md border border-amber-200 dark:border-amber-800">
              <div class="flex items-center gap-1.5 text-xs text-amber-600">
                <AlertTriangle class="h-3.5 w-3.5 shrink-0" />
                <span>Class imbalance detected (ratio &gt; 3:1)</span>
              </div>
              <RadioGroup bind:value={classBalanceStrategy} class="space-y-1">
                <div class="flex items-center space-x-2">
                  <RadioGroupItem value="none" id="balance-none" />
                  <Label for="balance-none" class="text-xs font-normal">Do nothing</Label>
                </div>
                <div class="flex items-center space-x-2">
                  <RadioGroupItem value="oversample" id="balance-oversample" />
                  <Label for="balance-oversample" class="text-xs font-normal">Oversample minority classes</Label>
                </div>
                <div class="flex items-center space-x-2">
                  <RadioGroupItem value="undersample" id="balance-undersample" />
                  <Label for="balance-undersample" class="text-xs font-normal">Undersample majority classes</Label>
                </div>
                <div class="flex items-center space-x-2">
                  <RadioGroupItem value="weighted_loss" id="balance-weighted" />
                  <Label for="balance-weighted" class="text-xs font-normal">Weighted loss function</Label>
                </div>
              </RadioGroup>
            </div>
          {/if}
        </CollapsibleContent>
      </Collapsible>

      <!-- Training Controls -->
      <section class="space-y-3">
        {#if !isTraining && !trainingDone}
          <Button class="w-full" onclick={startTraining} disabled={startDisabled}>Start Training</Button>
        {/if}

        {#if isTraining}
          <div class="space-y-2">
            <Progress value={progressPercent} class="h-2" />
            <div class="flex justify-between text-xs text-muted-foreground">
              {#if activeRun?.status === "predicting"}
                <span>
                  {(activeRun?.total_steps ?? 0) > 0
                    ? `Generating predictions... (${activeRun?.current_step ?? 0} / ${activeRun?.total_steps})`
                    : "Generating predictions..."}
                </span>
              {:else}
                {@const eta = etaText()}
                <span>
                  {(activeRun?.total_steps ?? 0) > 0
                    ? `Step ${activeRun?.current_step ?? 0} / ${activeRun?.total_steps}`
                    : "Starting..."}
                  {#if eta}
                    <span class="ml-1.5 text-muted-foreground/70">{eta} left</span>
                  {/if}
                </span>
              {/if}
              {#if activeRun?.val_accuracy != null}
                <span>{taskConfig.metrics.primaryKey}: {(activeRun.val_accuracy * 100).toFixed(1)}%</span>
              {/if}
            </div>
            <Button variant="outline" size="sm" class="w-full text-xs h-8 gap-1.5" onclick={stopTraining}>
              <Square class="h-3 w-3" />
              Stop
            </Button>
            <MetricsCharts runIds={[activeRunId]} {dbPath} {taskType} isLive />
          </div>
        {/if}

        {#if trainingDone && activeRun}
          <div class="space-y-3 p-3 bg-card rounded-lg border border-border">
            <h4 class="text-xs font-semibold text-foreground">Training Complete</h4>
            {#if taskType === "classification"}
              <div class="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span class="text-muted-foreground">Accuracy</span>
                  <p class="font-medium text-foreground">{((activeRun.best_accuracy ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">F1</span>
                  <p class="font-medium text-foreground">{((activeRun.best_f1 ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">AUROC</span>
                  <p class="font-medium text-foreground">{((activeRun.best_auroc ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Precision</span>
                  <p class="font-medium text-foreground">{((activeRun.val_precision ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Recall</span>
                  <p class="font-medium text-foreground">{((activeRun.val_recall ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Labels</span>
                  <p class="font-medium text-foreground">{activeRun.label_count}</p>
                </div>
              </div>
            {:else if taskType === "multilabel-classification"}
              <div class="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span class="text-muted-foreground">F1 (macro)</span>
                  <p class="font-medium text-foreground">{((activeRun.best_accuracy ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Exact Match</span>
                  <p class="font-medium text-foreground">{((activeRun.best_f1 ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">AUROC</span>
                  <p class="font-medium text-foreground">{((activeRun.best_auroc ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Precision</span>
                  <p class="font-medium text-foreground">{((activeRun.val_precision ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Recall</span>
                  <p class="font-medium text-foreground">{((activeRun.val_recall ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Labels</span>
                  <p class="font-medium text-foreground">{activeRun.label_count}</p>
                </div>
              </div>
              <PerClassMetricsTable run={activeRun} />
            {:else}
              <div class="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span class="text-muted-foreground">{taskConfig.metrics.primary}</span>
                  <p class="font-medium text-foreground">{((activeRun.best_accuracy ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">{taskConfig.metrics.secondary}</span>
                  <p class="font-medium text-foreground">{((activeRun.best_f1 ?? 0) * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Labels</span>
                  <p class="font-medium text-foreground">{activeRun.label_count}</p>
                </div>
                <div>
                  <span class="text-muted-foreground">Epochs</span>
                  <p class="font-medium text-foreground">{activeRun.epochs}</p>
                </div>
              </div>
            {/if}
            <!-- React's "Download Model" button here had no onClick — dropped per
                 spec §10 deviation 2 (the working download lives on the history row). -->
            <MetricsCharts runIds={[activeRun.id]} {dbPath} {taskType} />
          </div>
        {/if}
      </section>

      <!-- History -->
      <Collapsible bind:open={historyOpen}>
        <div class="flex items-center justify-between">
          <CollapsibleTrigger class="flex items-center gap-2 text-xs font-medium">
            <ChevronDown class={`h-3.5 w-3.5 transition-transform duration-200 ${historyOpen ? "" : "-rotate-90"}`} />
            Training History ({trainingHistory.length})
          </CollapsibleTrigger>
          {#if trainingHistory.length > 1}
            <button
              onclick={() => (compareAll = !compareAll)}
              class={`flex items-center gap-1 text-xs transition-colors ${compareAll ? "text-foreground font-medium" : "text-muted-foreground hover:text-foreground"}`}
            >
              <BarChart3 class="h-3 w-3" />
              {compareAll ? "Hide" : "Compare"}
            </button>
          {/if}
        </div>
        <CollapsibleContent class="mt-2 space-y-2">
          {#if compareAll && trainingHistory.length > 1}
            <MetricsCharts
              runIds={trainingHistory.filter((r) => r.status === "done" || r.status === "stopped").map((r) => r.id)}
              {dbPath}
              {taskType}
              runNames={Object.fromEntries(trainingHistory.map((r) => [r.id, r.display_name]))}
            />
          {/if}
          {#each trainingHistory as run (run.id)}
            <ContextMenu>
              <ContextMenuTrigger>
                {#snippet child({ props })}
                  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
                  <div
                    {...props}
                    class={`p-2 rounded-md text-xs space-y-1 cursor-pointer transition-colors ${
                      selectedRunId === run.id ? "bg-primary/10 ring-1 ring-primary" : "bg-muted/50 hover:bg-muted"
                    }`}
                    onclick={() => {
                      if (renamingRunId === run.id) return;
                      onSelectRun(selectedRunId === run.id ? null : run.id);
                    }}
                  >
                    <div class="flex justify-between items-center gap-2">
                      {#if renamingRunId === run.id}
                        <!-- svelte-ignore a11y_no_static_element_interactions -->
                        <input
                          use:autofocus
                          class="font-medium text-foreground bg-transparent border-b border-primary outline-hidden flex-1 min-w-0"
                          bind:value={renameValue}
                          onkeydown={(e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              saveRename(run.id);
                            }
                            if (e.key === "Escape") {
                              e.stopPropagation();
                              renamingRunId = null;
                            }
                          }}
                          onblur={() => saveRename(run.id)}
                          onclick={(e) => e.stopPropagation()}
                        />
                      {:else}
                        <span class="font-medium text-foreground truncate">{run.display_name}</span>
                      {/if}
                      <div class="flex items-center gap-1.5">
                        <span class="text-muted-foreground">{run.created_at?.slice(0, 16).replace("T", " ")}</span>
                        <button
                          onclick={(e) => {
                            e.stopPropagation();
                            configRunId = run.id;
                          }}
                          class="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
                          title="View configuration"
                        >
                          <Info class="h-3 w-3" />
                        </button>
                        {#if (run.status === "stopped" || run.status === "failed") && run.has_checkpoint}
                          <button
                            onclick={(e) => {
                              e.stopPropagation();
                              predictRun(run);
                            }}
                            class="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
                            title="Generate predictions"
                          >
                            <Play class="h-3 w-3" />
                          </button>
                        {/if}
                        {#if run.status === "done" && run.has_checkpoint}
                          <button
                            onclick={(e) => {
                              e.stopPropagation();
                              predictNewImages(run);
                            }}
                            class="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
                            title="Predict new images"
                          >
                            <Play class="h-3 w-3" />
                          </button>
                        {/if}
                        {#if run.has_checkpoint}
                          <a
                            href={`/api/training/runs/${run.id}/download?db_path=${encodeURIComponent(dbPath)}`}
                            download
                            onclick={(e) => e.stopPropagation()}
                            class="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
                            title="Download checkpoint (.ckpt)"
                          >
                            <Download class="h-3 w-3" />
                          </a>
                          <a
                            href={`/api/training/runs/${run.id}/export?db_path=${encodeURIComponent(dbPath)}`}
                            download
                            onclick={(e) => e.stopPropagation()}
                            class="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
                            title="Export HuggingFace bundle (.zip)"
                          >
                            <Package class="h-3 w-3" />
                          </a>
                        {/if}
                        <button
                          onclick={(e) => {
                            e.stopPropagation();
                            deleteRun(run);
                          }}
                          class="text-muted-foreground hover:text-destructive transition-colors p-0.5 rounded"
                          aria-label="Delete run"
                        >
                          <Trash2 class="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                    {#if run.status === "stopped"}
                      <div class="text-amber-500">
                        {run.total_steps > 0
                          ? `Stopped at step ${run.current_step}/${run.total_steps} (epoch ${run.current_epoch}/${run.epochs})`
                          : `Stopped at epoch ${run.current_epoch}/${run.epochs}`}
                      </div>
                    {/if}
                    {#if run.status === "failed" && run.error}
                      <div class="text-destructive truncate" title={run.error}>
                        {run.error}
                      </div>
                    {/if}
                    <div class="flex gap-3 text-muted-foreground">
                      <span>{taskConfig.metrics.primaryKey}: {((run.best_accuracy ?? 0) * 100).toFixed(1)}%</span>
                      <span>{taskConfig.metrics.secondaryKey}: {((run.best_f1 ?? 0) * 100).toFixed(1)}%</span>
                      <span>{run.label_count} labels</span>
                    </div>
                    {#if run.status === "done"}
                      <DatasetDiff runId={run.id} {dbPath} />
                    {/if}
                    {#if selectedRunId === run.id && (run.status === "done" || run.status === "stopped")}
                      <MetricsCharts runIds={[run.id]} {dbPath} {taskType} />
                      {#if taskType === "multilabel-classification"}
                        <PerClassMetricsTable {run} />
                      {/if}
                    {/if}
                  </div>
                {/snippet}
              </ContextMenuTrigger>
              <!-- onCloseAutoFocus prevented: bits-ui restores focus to the row on
                   menu close, which would instantly blur (and thus commit) the
                   just-mounted rename input. -->
              <ContextMenuContent onCloseAutoFocus={(e) => e.preventDefault()}>
                <ContextMenuItem
                  onSelect={() => {
                    renameValue = run.display_name;
                    renamingRunId = run.id;
                  }}
                >
                  <Pencil class="h-3.5 w-3.5 mr-2" />
                  Rename
                </ContextMenuItem>
              </ContextMenuContent>
            </ContextMenu>
          {/each}
          {#if trainingHistory.length === 0}
            <p class="text-xs text-muted-foreground">No training runs yet</p>
          {/if}
        </CollapsibleContent>
      </Collapsible>
    </div>
  </SheetContent>
</Sheet>

<AdvancedSettingsDialog
  open={advancedOpen}
  onClose={() => (advancedOpen = false)}
  settings={advancedSettings}
  onSave={(s) => (advancedSettings = s)}
  augmentationEnabled={augmentation}
/>
<TrainingConfigDialog
  open={!!configRunId}
  onClose={() => (configRunId = null)}
  run={trainingHistory.find((r) => r.id === configRunId) ?? null}
  {dbPath}
/>
