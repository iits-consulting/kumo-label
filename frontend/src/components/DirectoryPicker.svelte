<script lang="ts">
  import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
  } from "@/components/ui/dialog";
  import { Button } from "@/components/ui/button";
  import ArrowUp from "@lucide/svelte/icons/arrow-up";
  import Folder from "@lucide/svelte/icons/folder";
  import LoaderCircle from "@lucide/svelte/icons/loader-circle";
  import { api } from "@/lib/api";

  interface BrowseResponse {
    path: string;
    parent: string | null;
    entries: { name: string; path: string }[];
  }

  let {
    open,
    onClose,
    onSelect,
    initialPath,
  }: {
    open: boolean;
    onClose: () => void;
    onSelect: (path: string) => void;
    initialPath?: string;
  } = $props();

  let currentPath = $state<string | null>(null);
  let data = $state<BrowseResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Re-seed from initialPath each time the dialog opens.
  $effect(() => {
    if (!open) return;
    currentPath = initialPath?.trim() || null;
  });

  // Browse fetch with stale-response guard; adopts the server-resolved path
  // (handles ~, relative paths, symlinks — converges in one extra cycle).
  $effect(() => {
    if (!open) return;
    const path = currentPath;
    let cancelled = false;
    loading = true;
    error = null;
    const url = path ? `/datasets/browse?path=${encodeURIComponent(path)}` : "/datasets/browse";
    api
      .get<BrowseResponse>(url)
      .then((res) => {
        if (cancelled) return;
        data = res;
        if (path !== res.path) currentPath = res.path;
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        error = err instanceof Error ? err.message : "Failed to list directory";
      })
      .finally(() => {
        if (!cancelled) loading = false;
      });
    return () => {
      cancelled = true;
    };
  });

  function handleSelect() {
    if (data?.path) {
      onSelect(data.path);
      onClose();
    }
  }
</script>

<Dialog {open} onOpenChange={(o) => !o && onClose()}>
  <DialogContent class="sm:max-w-xl">
    <DialogHeader>
      <DialogTitle>Select Dataset Directory</DialogTitle>
    </DialogHeader>

    <!-- min-w-0: grid child must be allowed to shrink so truncate works on long names -->
    <div class="min-w-0 space-y-2">
      <div class="flex items-center gap-2">
        <Button
          type="button"
          size="sm"
          variant="ghost"
          class="h-8 px-2"
          onclick={() => data?.parent && (currentPath = data.parent)}
          disabled={!data?.parent || loading}
          aria-label="Go up"
        >
          <ArrowUp class="h-4 w-4" />
        </Button>
        <div class="flex-1 truncate font-mono text-xs text-muted-foreground border border-border rounded px-2 py-1.5">
          {data?.path ?? currentPath ?? "…"}
        </div>
      </div>

      <div class="border border-border rounded h-72 overflow-y-auto scrollbar-thin">
        {#if loading}
          <div class="flex items-center justify-center h-full text-xs text-muted-foreground">
            <LoaderCircle class="h-4 w-4 animate-spin mr-2" />
            Loading…
          </div>
        {:else if error}
          <div class="p-3 text-xs text-destructive">{error}</div>
        {:else if !data || data.entries.length === 0}
          <div class="p-3 text-xs text-muted-foreground">No subdirectories</div>
        {:else}
          <ul class="divide-y divide-border">
            {#each data.entries as entry (entry.path)}
              <li>
                <button
                  type="button"
                  onclick={() => (currentPath = entry.path)}
                  ondblclick={() => (currentPath = entry.path)}
                  class="w-full flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-muted/50 text-left"
                >
                  <Folder class="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                  <span class="truncate">{entry.name}</span>
                </button>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    </div>

    <DialogFooter>
      <Button variant="ghost" onclick={onClose}>Cancel</Button>
      <Button onclick={handleSelect} disabled={!data?.path}>Select this folder</Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
