<script lang="ts">
  import Search from "@lucide/svelte/icons/search";
  import Pin from "@lucide/svelte/icons/pin";
  import PinOff from "@lucide/svelte/icons/pin-off";
  import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
  import type { ShortcutMap } from "@/state/shortcutSettings.svelte";

  interface Props {
    classNames: string[];
    pinnedClasses: Set<string>;
    shortcuts: ShortcutMap;
    activeLabel: string | null;
    predictedLabel?: string | null;
    // Multi-label mode: when given, these sets decide active/predicted state
    // instead of the single-label `activeLabel`/`predictedLabel` props.
    activeLabels?: Set<string>;
    predictedLabels?: Set<string>;
    onSelectClass: (cls: string) => void;
    onTogglePinned: (cls: string) => void;
    searchPlaceholder?: string;
    maxGridHeight?: string;
  }

  let {
    classNames,
    pinnedClasses,
    shortcuts,
    activeLabel,
    predictedLabel = null,
    activeLabels = undefined,
    predictedLabels = undefined,
    onSelectClass,
    onTogglePinned,
    searchPlaceholder = "Filter classes...",
    maxGridHeight = "max-h-48",
  }: Props = $props();

  let classSearch = $state("");

  const filteredClassNames = $derived.by(() => {
    const q = classSearch.toLowerCase().trim();
    if (!q) return classNames;
    return classNames.filter((c) => c.toLowerCase().includes(q));
  });

  const pinned = $derived(filteredClassNames.filter((c) => pinnedClasses.has(c)));
  const unpinned = $derived(filteredClassNames.filter((c) => !pinnedClasses.has(c)));

  const isActive = (cls: string) => (activeLabels ? activeLabels.has(cls) : cls === activeLabel);
  const isPredicted = (cls: string) => (predictedLabels ? predictedLabels.has(cls) : cls === predictedLabel);
</script>

{#snippet cell(cls: string, cellPinned: boolean)}
  <Tooltip>
    <TooltipTrigger>
      {#snippet child({ props })}
        <div
          {...props}
          role="button"
          tabindex={0}
          onclick={() => onSelectClass(cls)}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSelectClass(cls);
            }
          }}
          class={`group px-3 py-2 text-xs rounded-md border transition-colors text-left relative truncate cursor-pointer ${
            isActive(cls)
              ? "bg-primary text-primary-foreground border-primary"
              : isPredicted(cls)
                ? "border-primary/50 ring-1 ring-primary/30 bg-primary/5 text-foreground hover:bg-primary/10"
                : "border-border text-foreground hover:border-foreground/30 hover:bg-muted"
          }`}
        >
          {cls}
          <span
            class={`absolute top-1 right-1.5 text-[10px] tabular-nums ${
              isActive(cls) ? "text-primary-foreground/70" : "text-muted-foreground"
            }`}>{shortcuts[cls] || ""}</span>
          <button
            onclick={(e) => {
              e.stopPropagation();
              onTogglePinned(cls);
            }}
            class={`absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity ${
              isActive(cls)
                ? "text-primary-foreground/60 hover:text-primary-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
            aria-label={cellPinned ? `Unpin ${cls}` : `Pin ${cls}`}
          >
            {#if cellPinned}<PinOff class="h-3 w-3" />{:else}<Pin class="h-3 w-3" />{/if}
          </button>
        </div>
      {/snippet}
    </TooltipTrigger>
    <TooltipContent side="top" class="text-xs">{cls}</TooltipContent>
  </Tooltip>
{/snippet}

<!-- Search filter -->
<div class="relative">
  <Search class="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
  <input
    bind:value={classSearch}
    placeholder={searchPlaceholder}
    class="w-full h-7 text-xs bg-background border border-border rounded-md pl-7 pr-2 text-foreground outline-hidden focus:border-primary placeholder:text-muted-foreground"
  />
</div>
<!-- Pinned classes -->
{#if pinned.length > 0}
  <div class={`grid grid-cols-2 gap-1.5 ${maxGridHeight} overflow-y-auto scrollbar-thin`}>
    {#each pinned as cls (cls)}
      {@render cell(cls, true)}
    {/each}
  </div>
{/if}
<!-- Unpinned / filtered classes -->
{#if unpinned.length > 0}
  {#if pinned.length > 0}
    <div class="flex items-center gap-2 pt-1">
      <div class="h-px flex-1 bg-border"></div>
      <span class="text-[10px] text-muted-foreground">All classes</span>
      <div class="h-px flex-1 bg-border"></div>
    </div>
  {/if}
  <div class={`grid grid-cols-2 gap-1.5 ${maxGridHeight} overflow-y-auto scrollbar-thin`}>
    {#each unpinned as cls (cls)}
      {@render cell(cls, false)}
    {/each}
  </div>
{/if}
{#if filteredClassNames.length === 0}
  <p class="text-xs text-muted-foreground text-center py-2">No matching classes</p>
{/if}
