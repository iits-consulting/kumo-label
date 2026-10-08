<script lang="ts">
  import type { TrainingRun } from "@/lib/types";

  let { run }: { run: TrainingRun } = $props();

  // Bound by the shortest array so a malformed/truncated backend payload
  // (arrays out of sync in length) degrades to the complete rows it can
  // build instead of throwing on an out-of-bounds index.
  const rows = $derived.by(() => {
    const perClass = run.val_per_class;
    if (!perClass) return [];
    const n = Math.min(
      perClass.classes.length,
      perClass.precision.length,
      perClass.recall.length,
      perClass.f1.length,
      perClass.support.length,
    );
    return perClass.classes
      .slice(0, n)
      .map((className, i) => ({
        className,
        precision: perClass.precision[i],
        recall: perClass.recall[i],
        f1: perClass.f1[i],
        support: perClass.support[i],
      }))
      .sort((a, b) => a.f1 - b.f1);
  });
</script>

{#if rows.length > 0}
  <div class="overflow-x-auto">
    <table class="w-full text-xs">
      <thead>
        <tr class="bg-muted/50">
          <th class="text-left px-2 py-1.5 font-medium text-muted-foreground">Class</th>
          <th class="text-right px-2 py-1.5 font-medium text-muted-foreground">Precision</th>
          <th class="text-right px-2 py-1.5 font-medium text-muted-foreground">Recall</th>
          <th class="text-right px-2 py-1.5 font-medium text-muted-foreground">F1</th>
          <th class="text-right px-2 py-1.5 font-medium text-muted-foreground">Support</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as row (row.className)}
          <tr class={`border-t border-border ${row.f1 < 0.5 ? "bg-destructive/10" : ""}`}>
            <td class="text-left px-2 py-1.5 text-foreground whitespace-nowrap">{row.className}</td>
            <td class="text-right px-2 py-1.5 tabular-nums text-foreground">{row.precision.toFixed(3)}</td>
            <td class="text-right px-2 py-1.5 tabular-nums text-foreground">{row.recall.toFixed(3)}</td>
            <td class="text-right px-2 py-1.5 tabular-nums text-foreground">{row.f1.toFixed(3)}</td>
            <td class="text-right px-2 py-1.5 tabular-nums text-foreground">{Math.round(row.support)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}
