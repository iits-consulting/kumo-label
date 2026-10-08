<script lang="ts">
  import X from "@lucide/svelte/icons/x";
  import type { DataPoint } from "@/lib/mockData";
  import ClassGrid from "@/components/ClassGrid.svelte";
  import type { ShortcutMap } from "@/state/shortcutSettings.svelte";

  interface Props {
    current: DataPoint | null;
    classNames: string[];
    classColors: Record<string, string>;
    onIgnore: () => void;
    // Toggle one class tag on the current image (no auto-advance).
    onToggleTag: (cls: string) => void;
    shortcuts: ShortcutMap;
    keyToClass: Record<string, string>;
    pinnedClasses: Set<string>;
    onTogglePinned: (cls: string) => void;
    handleAcceptPrediction: () => void;
  }

  let {
    current,
    classNames,
    classColors,
    onIgnore,
    onToggleTag,
    shortcuts,
    keyToClass,
    pinnedClasses,
    onTogglePinned,
    handleAcceptPrediction,
  }: Props = $props();

  // Multi-label keyboard shortcuts (second keyboard layer — spec §9.17).
  function handleKeydown(e: KeyboardEvent) {
    if ((e.target as HTMLElement).tagName === "INPUT") return;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      handleAcceptPrediction();
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      onIgnore();
      return;
    }
    if (current) {
      const cls = keyToClass[e.key];
      // No auto-advance: tagging an image usually takes several keys.
      if (cls && classNames.includes(cls)) onToggleTag(cls);
    }
  }

  const tags = $derived(current?.labels ?? []);
  const predictedLabels = $derived(current?.predictedLabels ?? []);
  // Prediction bars synthesized from predictedLabels when topPredictions is empty.
  const predictionBars = $derived(
    current?.topPredictions?.length > 0
      ? current.topPredictions
      : predictedLabels.map((label) => ({ label, confidence: current?.confidence ?? 0 })),
  );
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- Model predictions -->
{#if predictionBars.length > 0}
  <section class="space-y-2">
    <div class="flex items-center justify-between">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Model Prediction</h3>
    </div>
    <div class="space-y-1">
      {#each predictionBars as pred, i (pred.label)}
        <div class={`flex items-center gap-2 text-xs ${i > 0 ? "opacity-60" : ""}`}>
          <span class="w-20 truncate text-foreground" title={pred.label}>{pred.label}</span>
          <div class="flex-1 h-2 bg-muted rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-all"
              style="width:{(pred.confidence * 100).toFixed(0)}%;background-color:{classColors[pred.label] || '#9CA3AF'}"
            ></div>
          </div>
          <span class="w-9 text-right tabular-nums text-muted-foreground">{(pred.confidence * 100).toFixed(0)}%</span>
        </div>
      {/each}
    </div>
    {#if current?.uncertainty != null}
      <div class="flex items-center gap-2 text-xs mt-1.5 pt-1.5 border-t border-border">
        <span class="text-muted-foreground">Uncertainty</span>
        <div class="flex-1 h-2 bg-muted rounded-full overflow-hidden">
          <div class="h-full rounded-full bg-amber-500 transition-all" style="width:{(current.uncertainty * 100).toFixed(0)}%"></div>
        </div>
        <span class="w-9 text-right tabular-nums text-muted-foreground">
          {(current.uncertainty * 100).toFixed(0)}%
        </span>
      </div>
    {/if}
  </section>
{/if}

<!-- Tag set + class grid -->
<section class="space-y-2">
  <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Labels</h3>
  {#if tags.length > 0}
    <div class="flex flex-wrap gap-1">
      {#each tags as cls (cls)}
        <button
          onclick={() => onToggleTag(cls)}
          class="flex items-center gap-1 px-2 py-0.5 text-xs rounded-full border border-primary bg-primary/10 text-foreground hover:bg-primary/20 transition-colors"
          aria-label={`Remove ${cls}`}
        >
          <span class="inline-block w-1.5 h-1.5 rounded-full" style="background-color:{classColors[cls] || '#9CA3AF'}"></span>
          {cls}
          <X class="h-3 w-3 text-muted-foreground" />
        </button>
      {/each}
    </div>
  {:else}
    <p class="text-xs text-muted-foreground italic">No labels yet</p>
  {/if}
  <ClassGrid
    {classNames}
    {pinnedClasses}
    {shortcuts}
    activeLabel={null}
    activeLabels={new Set(tags)}
    predictedLabels={new Set(predictedLabels)}
    onSelectClass={onToggleTag}
    {onTogglePinned}
  />
</section>
