<script lang="ts">
  import type { DataPoint } from "@/lib/mockData";
  import ClassGrid from "@/components/ClassGrid.svelte";
  import type { ShortcutMap } from "@/state/shortcutSettings.svelte";

  interface Props {
    current: DataPoint | null;
    classNames: string[];
    classColors: Record<string, string>;
    onLabel: (label: string) => void;
    onIgnore: () => void;
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
    onLabel,
    onIgnore,
    shortcuts,
    keyToClass,
    pinnedClasses,
    onTogglePinned,
    handleAcceptPrediction,
  }: Props = $props();

  // Classification-specific keyboard shortcuts (second keyboard layer — spec §9.17).
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
      if (cls && classNames.includes(cls)) onLabel(cls);
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- Model predictions -->
{#if current?.predictedLabel}
  <section class="space-y-2">
    <div class="flex items-center justify-between">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Model Prediction</h3>
    </div>
    <div class="space-y-1">
      {#each current.topPredictions?.length > 0 ? current.topPredictions : [{ label: current.predictedLabel, confidence: current.confidence }] as pred, i (pred.label)}
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
    {#if current.uncertainty != null}
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

<!-- Classification grid -->
<section class="space-y-2">
  <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Classify</h3>
  <ClassGrid
    {classNames}
    {pinnedClasses}
    {shortcuts}
    activeLabel={current?.label ?? null}
    predictedLabel={current?.predictedLabel ?? null}
    onSelectClass={onLabel}
    {onTogglePinned}
  />
</section>
