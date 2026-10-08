<script lang="ts">
  import { createQuery } from "@tanstack/svelte-query";
  import { api } from "@/lib/api";
  import RefreshCw from "@lucide/svelte/icons/refresh-cw";

  interface SnapshotResponse {
    snapshot: { image_count: number; ignored_count: number; label_counts: Record<string, number> } | null;
    diff: {
      new_labels: number;
      changed_labels: number;
      new_ignored: number;
      removed_ignored: number;
      new_images: number;
    } | null;
    has_changes: boolean;
  }

  let { runId, dbPath }: { runId: string; dbPath: string } = $props();

  const query = createQuery<SnapshotResponse>(() => ({
    queryKey: ["snapshot", runId, dbPath],
    queryFn: () =>
      api.get<SnapshotResponse>(`/training/runs/${runId}/snapshot?db_path=${encodeURIComponent(dbPath)}`),
    enabled: !!runId && !!dbPath,
  }));

  const parts = $derived.by(() => {
    const d = query.data?.diff;
    const out: string[] = [];
    if (d?.new_labels) out.push(`+${d.new_labels} labeled`);
    if (d?.changed_labels) out.push(`~${d.changed_labels} changed`);
    if (d?.new_ignored) out.push(`+${d.new_ignored} ignored`);
    if (d?.new_images) out.push(`+${d.new_images} new`);
    return out;
  });
</script>

{#if query.data?.snapshot}
  <div class="text-muted-foreground">
    <span>Trained on {query.data.snapshot.image_count} images</span>
    {#if query.data.snapshot.ignored_count > 0}
      <span> ({query.data.snapshot.ignored_count} ignored)</span>
    {/if}
    {#if query.data.has_changes && parts.length > 0}
      <span class="flex items-center gap-1 text-amber-500 mt-0.5">
        <RefreshCw class="h-2.5 w-2.5" />
        Since: {parts.join(", ")}
      </span>
    {/if}
  </div>
{/if}
