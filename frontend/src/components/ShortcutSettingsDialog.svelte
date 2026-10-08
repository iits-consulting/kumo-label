<script lang="ts">
  import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
  import { Button } from "@/components/ui/button";
  import Pin from "@lucide/svelte/icons/pin";
  import Search from "@lucide/svelte/icons/search";
  import TruncatedClassName from "@/components/TruncatedClassName.svelte";
  import type { ShortcutMap } from "@/state/shortcutSettings.svelte";

  interface Props {
    open: boolean;
    onClose: () => void;
    classNames: string[];
    classColors: Record<string, string>;
    shortcuts: ShortcutMap;
    onUpdateShortcut: (className: string, key: string) => void;
    onResetDefaults: () => void;
    pinnedClasses: Set<string>;
    onTogglePinned: (className: string) => void;
  }

  let {
    open,
    onClose,
    classNames,
    classColors,
    shortcuts,
    onUpdateShortcut,
    onResetDefaults,
    pinnedClasses,
    onTogglePinned,
  }: Props = $props();

  let editingClass = $state<string | null>(null);
  let search = $state("");

  const filteredClasses = $derived.by(() => {
    const q = search.toLowerCase().trim();
    if (!q) return classNames;
    return classNames.filter((c) => c.toLowerCase().includes(q));
  });

  // Any-key capture while a class row is in "editing" mode; Escape cancels.
  function handleCapture(e: KeyboardEvent) {
    if (!editingClass) return;
    e.preventDefault();
    if (e.key === "Escape") {
      editingClass = null;
      return;
    }
    onUpdateShortcut(editingClass, e.key);
    editingClass = null;
  }
</script>

<svelte:window onkeydown={handleCapture} />

<Dialog {open} onOpenChange={(v) => !v && onClose()}>
  <DialogContent class="max-w-md">
    <DialogHeader>
      <DialogTitle class="text-sm font-semibold">Label Settings</DialogTitle>
    </DialogHeader>

    <!-- Search -->
    <div class="relative">
      <Search class="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
      <input
        bind:value={search}
        placeholder="Search classes..."
        class="w-full h-7 text-xs bg-background border border-border rounded-md pl-7 pr-2 text-foreground outline-hidden focus:border-primary placeholder:text-muted-foreground"
      />
    </div>

    <!-- Column headers -->
    <div class="flex items-center gap-2 px-2 pb-1 border-b border-border">
      <div class="w-2.5 h-2.5 shrink-0"></div>
      <span class="text-xs text-muted-foreground flex-1">Class</span>
      <span class="text-xs text-muted-foreground w-8 text-center">Pin</span>
      <span class="text-xs text-muted-foreground min-w-10 text-center">Key</span>
    </div>

    <div class="space-y-0.5 max-h-80 overflow-y-auto">
      {#each filteredClasses as cls (cls)}
        <div class="flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-muted/50">
          <div class="w-2.5 h-2.5 rounded-full shrink-0" style="background-color: {classColors[cls] || '#9CA3AF'}"></div>
          <TruncatedClassName name={cls} class="text-xs text-foreground flex-1 min-w-0" />

          <!-- Pin toggle -->
          <button
            onclick={() => onTogglePinned(cls)}
            class={`w-8 flex items-center justify-center h-7 rounded-md transition-colors ${
              pinnedClasses.has(cls)
                ? "text-primary"
                : "text-muted-foreground/40 hover:text-muted-foreground"
            }`}
            aria-label={pinnedClasses.has(cls) ? "Unpin from quick labels" : "Pin to quick labels"}
            title={pinnedClasses.has(cls) ? "Pinned as quick label" : "Not pinned"}
          >
            <Pin class="h-3.5 w-3.5" fill={pinnedClasses.has(cls) ? "currentColor" : "none"} />
          </button>

          <!-- Shortcut key -->
          <button
            onclick={() => (editingClass = cls)}
            class={`min-w-10 h-7 px-2 text-xs font-mono rounded-md border transition-colors ${
              editingClass === cls
                ? "border-primary bg-primary/10 text-primary animate-pulse"
                : "border-border text-muted-foreground hover:border-foreground/30"
            }`}
          >
            {editingClass === cls ? "..." : shortcuts[cls] || "—"}
          </button>
        </div>
      {/each}
    </div>
    {#if filteredClasses.length === 0}
      <p class="text-xs text-muted-foreground text-center py-2">No matching classes</p>
    {/if}

    <div class="flex justify-between pt-2 border-t border-border">
      <Button variant="ghost" size="sm" class="text-xs" onclick={onResetDefaults}>
        Reset to defaults
      </Button>
      <Button size="sm" class="text-xs" onclick={onClose}>
        Done
      </Button>
    </div>
  </DialogContent>
</Dialog>
