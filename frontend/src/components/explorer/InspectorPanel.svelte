<script lang="ts">
  import { type DataPoint, type TaskType, effectiveSplit } from "@/lib/mockData";
  import { Button } from "@/components/ui/button";
  import ChevronLeft from "@lucide/svelte/icons/chevron-left";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import EyeOff from "@lucide/svelte/icons/eye-off";
  import Eye from "@lucide/svelte/icons/eye";
  import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
  import ClassGrid from "@/components/ClassGrid.svelte";
  import type { ShortcutMap } from "@/state/shortcutSettings.svelte";

  interface Props {
    point: DataPoint | null;
    selectedCount: number;
    onQuickLabel: (pointId: string, label: string) => void;
    // Multi-label: toggle one tag on the inspected image / add one tag to a selection
    onToggleTag?: (pointId: string, cls: string) => void;
    onBulkTag?: (ids: string[], cls: string, present: boolean) => void;
    onNavigate: (direction: "prev" | "next") => void;
    onOpenAnnotator: () => void;
    onAssignLabel: (label: string) => void;
    onRemoveLabels: () => void;
    onClearSelection: () => void;
    classNames: string[];
    pinnedClasses: Set<string>;
    shortcuts: ShortcutMap;
    onTogglePinned: (cls: string) => void;
    onIgnore: (ids: string[], ignored: boolean) => void;
    onAssignSplit: (ids: string[], split: "train" | "valid" | "test" | null) => void;
    // All currently selected IDs (used when applying split to a multi-select)
    selectedIds: string[];
    taskType: TaskType;
    // Live AL score derived from current AL weights (overrides point.alScore).
    alScore?: number;
  }

  let {
    point,
    selectedCount,
    onQuickLabel,
    onToggleTag = undefined,
    onBulkTag = undefined,
    onNavigate,
    onOpenAnnotator,
    onAssignLabel,
    onRemoveLabels,
    onClearSelection,
    classNames,
    pinnedClasses,
    shortcuts,
    onTogglePinned,
    onIgnore,
    onAssignSplit,
    selectedIds,
    taskType,
    alScore = undefined,
  }: Props = $props();

  const isMultilabel = $derived(taskType === "multilabel-classification");
  let imgDims = $state({ w: 0, h: 0 });

  // Reset image dims when the inspected point changes (keyed on id only —
  // row rebuilds replace the object with an identical id and must not reset).
  let prevPointId: string | undefined;
  $effect(() => {
    const id = point?.id;
    if (id === prevPointId) return;
    prevPointId = id;
    imgDims = { w: 0, h: 0 };
  });

  const topPreds = $derived(point?.topPredictions ?? []);
  const maxConf = $derived(topPreds.length > 0 ? topPreds[0].confidence : 0);

  const SPLIT_OPTIONS = [
    { value: "train", label: "Train", color: "#3B82F6" },
    { value: "valid", label: "Val", color: "#F59E0B" },
    { value: "test", label: "Test", color: "#10B981" },
  ] as const;
</script>

{#if !point}
  <div class="p-4 flex items-center justify-center h-full">
    <p class="text-xs text-muted-foreground text-center">
      {selectedCount > 0 ? `${selectedCount} points selected` : "Select a point to inspect"}
    </p>
  </div>
{:else}
  <div class="p-3 space-y-4 text-sm">
    <!-- Thumbnail -->
    <div class="relative">
      <img
        src={point.imageUrl}
        alt={point.filename}
        class="w-full aspect-4/3 object-cover rounded-lg border border-border bg-muted"
        loading="lazy"
        onload={(e) => {
          const img = e.currentTarget as HTMLImageElement;
          imgDims = { w: img.naturalWidth, h: img.naturalHeight };
        }}
      />
    </div>

    <!-- Metadata -->
    <section class="space-y-1.5">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Metadata</h3>
      <div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        <span class="text-muted-foreground">File</span>
        <span class="font-mono text-foreground truncate">{point.filename}</span>
        <span class="text-muted-foreground">Size</span>
        <span class="text-foreground">{imgDims.w}×{imgDims.h}</span>
        <span class="text-muted-foreground">Ground Truth</span>
        <span class="text-foreground font-medium">{point.groundTruth}</span>
        <span class="text-muted-foreground">Annotation</span>
        <span class="text-foreground font-medium">
          {isMultilabel ? point.labels.join(", ") || "—" : point.label || "—"}
        </span>
        <span class="text-muted-foreground">Uncertainty</span>
        <span class="text-foreground tabular-nums">{(point.uncertainty * 100).toFixed(0)}%</span>
        <span class="text-muted-foreground">AL Score</span>
        <span class="text-foreground tabular-nums">{((alScore ?? point.alScore) * 100).toFixed(0)}%</span>
        <span class="text-muted-foreground">Status</span>
        <span class="text-foreground">{point.ignored ? "Ignored" : "Active"}</span>
        <span class="text-muted-foreground">Split</span>
        <span class="text-foreground">
          {effectiveSplit(point) || "—"}
          {#if point.splitOverride}<span class="ml-1 text-[10px] text-muted-foreground">(manual)</span>{/if}
        </span>
      </div>
    </section>

    <!-- Predictions -->
    <section class="space-y-2">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Model Predictions</h3>
      <div class="space-y-1.5">
        <!-- index-keyed: detection top_preds can repeat a class name -->
        {#each topPreds as pred, i (i)}
          <div class="space-y-0.5">
            <div class="flex justify-between text-xs">
              <span class="text-foreground">{pred.label}</span>
              <Tooltip>
                <TooltipTrigger>
                  {#snippet child({ props })}
                    <span {...props} class="text-muted-foreground tabular-nums cursor-default">{(pred.confidence * 100).toFixed(0)}%</span>
                  {/snippet}
                </TooltipTrigger>
                <TooltipContent>{pred.confidence * 100}%</TooltipContent>
              </Tooltip>
            </div>
            <div class="h-1.5 bg-muted rounded-sm overflow-hidden">
              <div
                class="h-full rounded-sm transition-all duration-300"
                style="width: {(pred.confidence / maxConf) * 100}%; background-color: {i === 0 ? 'hsl(354, 100%, 44.5%)' : '#9CA3AF'}"
              ></div>
            </div>
          </div>
        {/each}
      </div>
    </section>

    <!-- Navigation -->
    {#if selectedCount > 1}
      <div class="flex items-center justify-between">
        <Button variant="ghost" size="sm" class="h-7" onclick={() => onNavigate("prev")} aria-label="Previous">
          <ChevronLeft class="h-4 w-4" />
        </Button>
        <span class="text-xs text-muted-foreground">
          {selectedCount} selected
        </span>
        <Button variant="ghost" size="sm" class="h-7" onclick={() => onNavigate("next")} aria-label="Next">
          <ChevronRight class="h-4 w-4" />
        </Button>
      </div>
    {/if}

    <!-- Unified Label Section — single-label and multi-label classification -->
    {#if taskType !== "object-detection"}
      <section class="space-y-2">
        <div class={`flex items-center justify-between ${selectedCount > 1 ? "bg-amber-500/10 -mx-3 px-3 py-1.5 rounded-md" : ""}`}>
          <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {selectedCount > 1
              ? isMultilabel
                ? `Add label to all ${selectedCount} images`
                : `Label all ${selectedCount} images`
              : isMultilabel
                ? "Labels"
                : "Label"}
          </h3>
          {#if selectedCount > 1}
            <button onclick={onClearSelection} class="text-[10px] text-muted-foreground hover:text-foreground underline">
              Clear
            </button>
          {/if}
        </div>
        <ClassGrid
          {classNames}
          {pinnedClasses}
          {shortcuts}
          activeLabel={point?.label ?? null}
          predictedLabel={point?.topPredictions?.[0]?.label ?? null}
          activeLabels={isMultilabel ? new Set(point?.labels ?? []) : undefined}
          predictedLabels={isMultilabel ? new Set(point?.predictedLabels ?? []) : undefined}
          onSelectClass={(cls) => {
            if (isMultilabel) {
              // Multi-selection: clicking a class adds that tag to every selected image.
              if (selectedCount > 1) onBulkTag?.(selectedIds, cls, true);
              else if (point) onToggleTag?.(point.id, cls);
            } else if (selectedCount > 1) {
              onAssignLabel(cls);
            } else if (point) {
              onQuickLabel(point.id, cls);
            }
          }}
          {onTogglePinned}
        />
        {#if selectedCount >= 1}
          <button onclick={onRemoveLabels} class="text-[10px] text-muted-foreground hover:text-destructive underline">
            {selectedCount > 1 ? `Remove labels from ${selectedCount} images` : "Remove label"}
          </button>
        {/if}
      </section>
    {/if}

    <!-- Split assignment -->
    <section class="space-y-2 pt-2 border-t border-border">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {selectedCount > 1 ? `Mark ${selectedCount} as` : "Mark as"}
      </h3>
      <div class="grid grid-cols-3 gap-1">
        {#each SPLIT_OPTIONS as { value, label, color } (value)}
          {@const active = effectiveSplit(point) === value}
          <Button
            variant="outline"
            size="sm"
            class="h-7 text-xs"
            style={active ? `border-color: ${color}; color: ${color}` : undefined}
            onclick={() => onAssignSplit(selectedCount > 1 ? selectedIds : [point.id], value)}
          >
            <span class="inline-block w-1.5 h-1.5 rounded-full mr-1.5" style="background-color: {color}"></span>
            {label}
          </Button>
        {/each}
      </div>
      {#if point.splitOverride && selectedCount === 1}
        <button
          onclick={() => onAssignSplit([point.id], null)}
          class="text-[10px] text-muted-foreground hover:text-foreground underline"
        >
          Clear override (use folder split)
        </button>
      {/if}
    </section>

    <!-- Actions -->
    <div class="space-y-2 pt-2 border-t border-border">
      <Button
        variant="outline"
        size="sm"
        class="w-full text-xs h-8 border-primary text-primary hover:bg-primary hover:text-primary-foreground"
        onclick={onOpenAnnotator}
      >
        Open in Annotator →
      </Button>
      <Button variant="ghost" size="sm" class="w-full text-xs h-8" onclick={() => onIgnore(selectedCount > 1 ? selectedIds : [point.id], !point.ignored)}>
        {#if point.ignored}
          <Eye class="h-3.5 w-3.5 mr-1.5" />
        {:else}
          <EyeOff class="h-3.5 w-3.5 mr-1.5" />
        {/if}
        {point.ignored ? "Unignore Image" : "Ignore Image"}
      </Button>
    </div>
  </div>
{/if}
