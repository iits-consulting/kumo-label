<script lang="ts">
  import { type DataPoint, getDatasetStats, type TaskType } from "@/lib/mockData";
  import type { ExplorerFilters } from "@/lib/types";
  import type { ColorBy } from "@/lib/taskConfig";
  import { TASK_CONFIGS } from "@/lib/taskConfig";
  import { Label } from "@/components/ui/label";
  import { Slider } from "@/components/ui/slider";
  import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
  import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "@/components/ui/context-menu";
  import Plus from "@lucide/svelte/icons/plus";
  import Search from "@lucide/svelte/icons/search";
  import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
  import TruncatedClassName from "@/components/TruncatedClassName.svelte";

  interface Props {
    data: DataPoint[];
    filters: ExplorerFilters;
    onFiltersChange: (f: ExplorerFilters) => void;
    taskType: TaskType;
    classNames: string[];
    classColors: Record<string, string>;
    onAddClass: (name: string) => void;
    onRenameClass: (oldName: string, newName: string) => void;
    onRemoveClass: (name: string) => void;
  }

  let { data, filters, onFiltersChange, taskType, classNames, classColors, onAddClass, onRenameClass, onRemoveClass }: Props = $props();

  const stats = $derived(getDatasetStats(data, taskType));
  const ignoredCount = $derived(data.filter((d) => d.ignored).length);
  const labeledPct = $derived(stats.total > 0 ? (stats.labeled / stats.total) * 100 : 0);

  let addingClass = $state(false);
  let newClassName = $state("");
  let renamingClass = $state<string | null>(null);
  let renameValue = $state("");
  let classSearch = $state("");

  function autofocus(node: HTMLElement) {
    node.focus();
  }

  function update(partial: Partial<ExplorerFilters>) {
    onFiltersChange({ ...filters, ...partial });
  }

  const filteredClassList = $derived.by(() => {
    // Merge classNames (may have 0 samples) with stats from data
    const merged: [string, number][] = classNames.map((cls) => [cls, stats.classCounts[cls] || 0]);
    // Include any classes from data not in classNames (e.g. "Unlabeled")
    Object.entries(stats.classCounts).forEach(([cls, count]) => {
      if (!classNames.includes(cls)) merged.push([cls, count]);
    });
    const sorted = merged.sort((a, b) => b[1] - a[1]);
    const q = classSearch.toLowerCase().trim();
    return q ? sorted.filter(([cls]) => cls.toLowerCase().includes(q)) : sorted;
  });

  function toggleClass(cls: string) {
    const current = filters.selectedClasses;
    const next = current.includes(cls) ? current.filter((c) => c !== cls) : [...current, cls];
    update({ selectedClasses: next });
  }

  function soloClass(cls: string) {
    const current = filters.selectedClasses;
    // If this class is already the only selected one, deselect it (show all)
    if (current.length === 1 && current[0] === cls) {
      update({ selectedClasses: [] });
    } else {
      update({ selectedClasses: [cls] });
    }
  }

  function handleAddSubmit() {
    if (newClassName.trim()) {
      onAddClass(newClassName.trim());
      newClassName = "";
    }
    addingClass = false;
  }

  function handleRenameSubmit() {
    if (renamingClass && renameValue.trim()) onRenameClass(renamingClass, renameValue.trim());
    renamingClass = null;
    renameValue = "";
  }

  const colorByOptions = $derived(TASK_CONFIGS[taskType].colorByOptions);
  const colorByLabel = $derived(colorByOptions.find((o) => o.value === filters.colorBy)?.label ?? "");

  const FILTER_LABELED_OPTIONS = ["labeled", "unlabeled", "ignored"] as const;
</script>

<div class="p-3 space-y-5 text-sm">
  <!-- Dataset Stats -->
  <section class="space-y-3">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Dataset</h3>
    <div class="space-y-1.5">
      <div class="flex justify-between text-xs">
        <span class="text-muted-foreground">Total</span>
        <span class="font-medium text-foreground">{stats.total.toLocaleString()}</span>
      </div>
      <div class="flex justify-between text-xs">
        <span class="text-muted-foreground">Labeled</span>
        <span class="font-medium text-foreground">{stats.labeled}</span>
      </div>
      <div class="flex justify-between text-xs">
        <span class="text-muted-foreground">Unlabeled</span>
        <span class="font-medium text-foreground">{stats.unlabeled}</span>
      </div>
      {#if ignoredCount > 0}
        <div class="flex justify-between text-xs">
          <span class="text-muted-foreground">Ignored</span>
          <span class="font-medium text-foreground">{ignoredCount}</span>
        </div>
      {/if}
      <Tooltip>
        <div class="flex justify-between text-xs">
          <span class="font-medium text-muted-foreground">Percent labeled: {labeledPct.toFixed(1)}%</span>
        </div>
        <TooltipTrigger>
          {#snippet child({ props })}
            <div {...props} class="h-1.5 w-full bg-muted rounded-sm overflow-hidden">
              <div class="h-full bg-primary transition-all duration-300" style="width: {labeledPct}%"></div>
            </div>
          {/snippet}
        </TooltipTrigger>
        <TooltipContent side="bottom" class="text-xs">{labeledPct.toFixed(1)}%</TooltipContent>
      </Tooltip>
    </div>

    <!-- Class search -->
    <div class="relative">
      <Search class="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
      <input
        bind:value={classSearch}
        placeholder="Filter classes..."
        class="w-full h-7 text-xs bg-background border border-border rounded-md pl-7 pr-2 text-foreground outline-hidden focus:border-primary placeholder:text-muted-foreground"
      />
    </div>

    <!-- Per-class breakdown with context menu -->
    <div class="space-y-1 max-h-40 overflow-y-auto scrollbar-thin">
      {#each filteredClassList as [cls, count] (cls)}
        <ContextMenu>
          <ContextMenuTrigger>
            {#snippet child({ props })}
              <div {...props} class="flex items-center gap-2 text-xs">
                {#if renamingClass === cls}
                  <input
                    use:autofocus
                    bind:value={renameValue}
                    onblur={handleRenameSubmit}
                    onkeydown={(e) => {
                      if (e.key === "Enter") handleRenameSubmit();
                      if (e.key === "Escape") {
                        renamingClass = null;
                        renameValue = "";
                      }
                    }}
                    class="flex-1 h-5 text-xs bg-muted border border-border rounded px-1 text-foreground outline-hidden focus:border-primary"
                  />
                {:else}
                  <div
                    role="button"
                    tabindex={0}
                    class={`flex items-center gap-2 flex-1 min-w-0 cursor-pointer select-none rounded px-1 py-0.5 -mx-1 transition-colors ${
                      filters.selectedClasses.includes(cls)
                        ? "bg-foreground/10 text-foreground font-medium"
                        : "hover:bg-muted text-foreground"
                    }`}
                    onclick={() => toggleClass(cls)}
                    onkeydown={(e) => {
                      if (e.key === "Enter") toggleClass(cls);
                    }}
                    ondblclick={(e) => {
                      e.preventDefault();
                      soloClass(cls);
                    }}
                  >
                    <div
                      class="w-2 h-2 rounded-full shrink-0"
                      style="background-color: {taskType === 'object-detection' ? '#9CA3AF' : classColors[cls] || '#9CA3AF'}"
                    ></div>
                    <TruncatedClassName name={cls} class="flex-1 min-w-0" />
                    <span class="text-muted-foreground tabular-nums">{count}</span>
                  </div>
                {/if}
              </div>
            {/snippet}
          </ContextMenuTrigger>
          {#if cls !== "Unlabeled"}
            <!-- onCloseAutoFocus prevented: bits-ui restores focus to the trigger on
                 menu close AFTER the rename input mounts+autofocuses, which blurs it
                 and instantly commits (same fix as TrainingDrawer). -->
            <ContextMenuContent onCloseAutoFocus={(e) => e.preventDefault()}>
              <ContextMenuItem
                onSelect={() => {
                  renamingClass = cls;
                  renameValue = cls;
                }}
              >
                Rename
              </ContextMenuItem>
              <ContextMenuItem class="text-destructive" onSelect={() => onRemoveClass(cls)}>
                Remove
              </ContextMenuItem>
            </ContextMenuContent>
          {/if}
        </ContextMenu>
      {/each}
      {#if filteredClassList.length === 0}
        <p class="text-xs text-muted-foreground text-center py-1">No matching classes</p>
      {/if}
    </div>

    <!-- Add class inline -->
    {#if addingClass}
      <div class="flex gap-1">
        <input
          use:autofocus
          bind:value={newClassName}
          onblur={handleAddSubmit}
          onkeydown={(e) => {
            if (e.key === "Enter") handleAddSubmit();
            if (e.key === "Escape") {
              addingClass = false;
              newClassName = "";
            }
          }}
          placeholder="Class name"
          class="flex-1 h-6 text-xs bg-background border border-border rounded px-1.5 text-foreground outline-hidden focus:border-primary"
        />
      </div>
    {:else}
      <button
        onclick={() => (addingClass = true)}
        class="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <Plus class="h-3 w-3" /> Add class
      </button>
    {/if}
  </section>

  <!-- Color By -->
  <section class="space-y-2">
    <Label class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Color By</Label>
    <Select type="single" value={filters.colorBy} onValueChange={(v) => update({ colorBy: v as ColorBy })}>
      <SelectTrigger class="h-8 text-xs">
        {colorByLabel}
      </SelectTrigger>
      <SelectContent>
        {#each colorByOptions as opt (opt.value)}
          <SelectItem value={opt.value}>{opt.label}</SelectItem>
        {/each}
      </SelectContent>
    </Select>
    <!-- Focus class — which class "Has Class…" highlights -->
    {#if filters.colorBy === "hasClass"}
      <Select type="single" value={filters.focusClass ?? undefined} onValueChange={(v) => update({ focusClass: v })}>
        <SelectTrigger class="h-8 text-xs" aria-label="Focus class">
          {#if filters.focusClass}
            {filters.focusClass}
          {:else}
            <span class="text-muted-foreground">Pick a class…</span>
          {/if}
        </SelectTrigger>
        <SelectContent>
          {#each classNames as cls (cls)}
            <SelectItem value={cls}>{cls}</SelectItem>
          {/each}
        </SelectContent>
      </Select>
    {/if}
  </section>

  <!-- Filter By -->
  <section class="space-y-3">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Filter By</h3>
    <div class="flex gap-1">
      {#each FILTER_LABELED_OPTIONS as val (val)}
        <button
          onclick={() => update({ filterLabeled: filters.filterLabeled === val ? "all" : val })}
          class={`flex-1 px-2 py-1 text-xs rounded-md transition-colors ${
            filters.filterLabeled === val
              ? "bg-primary text-primary-foreground font-medium"
              : "bg-muted text-muted-foreground hover:bg-border"
          }`}
        >
          {val.charAt(0).toUpperCase() + val.slice(1)}
        </button>
      {/each}
    </div>

    <!-- Confidence slider -->
    <div class="space-y-1.5">
      <div class="flex justify-between text-xs text-muted-foreground">
        <span>Confidence</span>
        <span class="tabular-nums">{filters.confidenceRange[0].toFixed(1)} – {filters.confidenceRange[1].toFixed(1)}</span>
      </div>
      <Slider
        type="multiple"
        min={0}
        max={1}
        step={0.05}
        value={[...filters.confidenceRange]}
        onValueChange={(v) => update({ confidenceRange: v as [number, number] })}
        class="w-full"
      />
    </div>
  </section>
</div>
