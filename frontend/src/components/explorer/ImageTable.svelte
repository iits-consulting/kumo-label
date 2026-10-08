<script lang="ts">
  import { type DataPoint } from "@/lib/mockData";
  import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
  import { Button } from "@/components/ui/button";
  import { Checkbox } from "@/components/ui/checkbox";
  import { Badge } from "@/components/ui/badge";
  import ChevronLeft from "@lucide/svelte/icons/chevron-left";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import ChevronsLeft from "@lucide/svelte/icons/chevrons-left";
  import ChevronsRight from "@lucide/svelte/icons/chevrons-right";
  import ArrowUp from "@lucide/svelte/icons/arrow-up";
  import ArrowDown from "@lucide/svelte/icons/arrow-down";
  import ArrowUpDown from "@lucide/svelte/icons/arrow-up-down";
  import EyeOff from "@lucide/svelte/icons/eye-off";
  import Eye from "@lucide/svelte/icons/eye";
  import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
  import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

  type SortField = "confidence" | "uncertainty" | "al_score";
  type SortOrder = "asc" | "desc";

  interface Props {
    data: DataPoint[];
    selectedIds: Set<string>;
    onSelect: (ids: Set<string>) => void;
    onRowClick: (id: string) => void;
    classColors: Record<string, string>;
    // Live AL scores derived from current AL weights — overrides per-row d.alScore.
    alScoreMap?: Map<string, number>;
    // Server-side pagination
    page: number; // 1-based
    pageSize: number;
    total: number;
    onPageChange: (page: number) => void;
    onPageSizeChange: (size: number) => void;
    // Sorting
    sortBy: SortField | null;
    sortOrder: SortOrder;
    onSort: (field: SortField) => void;
    onIgnore: (ids: string[], ignored: boolean) => void;
  }

  let {
    data,
    selectedIds,
    onSelect,
    onRowClick,
    classColors,
    alScoreMap = undefined,
    page,
    pageSize,
    total,
    onPageChange,
    onPageSizeChange,
    sortBy,
    sortOrder,
    onSort,
    onIgnore,
  }: Props = $props();

  const PAGE_SIZE_OPTIONS = [25, 50, 100, 200];

  const totalPages = $derived(Math.max(1, Math.ceil(total / pageSize)));
  const allPageSelected = $derived(data.length > 0 && data.every((d) => selectedIds.has(d.id)));
  const somePageSelected = $derived(data.some((d) => selectedIds.has(d.id)));

  function toggleAll() {
    if (allPageSelected) {
      const next = new Set(selectedIds);
      data.forEach((d) => next.delete(d.id));
      onSelect(next);
    } else {
      const next = new Set(selectedIds);
      data.forEach((d) => next.add(d.id));
      onSelect(next);
    }
  }

  // Convert 1-based page to display range
  const rangeStart = $derived((page - 1) * pageSize + 1);
  const rangeEnd = $derived(Math.min(page * pageSize, total));

  function toggleOne(id: string) {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onSelect(next);
  }
</script>

{#snippet colorDot(label: string | null)}
  <span
    class="inline-block w-2 h-2 rounded-full mr-1.5 shrink-0"
    style="background-color: {classColors[label || 'Unlabeled'] || '#9CA3AF'}"
  ></span>
{/snippet}

{#snippet sortHeader(field: SortField, text: string, tooltip: string)}
  <Tooltip>
    <TooltipTrigger>
      {#snippet child({ props })}
        <button
          {...props}
          class="inline-flex items-center gap-1 hover:text-foreground transition-colors cursor-pointer ml-auto"
          onclick={() => onSort(field)}
        >
          {text}
          {#if sortBy === field}
            {#if sortOrder === "desc"}
              <ArrowDown class="h-3 w-3" />
            {:else}
              <ArrowUp class="h-3 w-3" />
            {/if}
          {:else}
            <ArrowUpDown class="h-3 w-3 opacity-40" />
          {/if}
        </button>
      {/snippet}
    </TooltipTrigger>
    <TooltipContent>{tooltip}</TooltipContent>
  </Tooltip>
{/snippet}

{#snippet labelCell(label: string)}
  <Tooltip>
    <TooltipTrigger>
      {#snippet child({ props })}
        <span {...props} class="flex items-center text-xs max-w-full">
          {@render colorDot(label)}
          <span class="truncate">{label}</span>
        </span>
      {/snippet}
    </TooltipTrigger>
    <TooltipContent>{label}</TooltipContent>
  </Tooltip>
{/snippet}

<div class="flex flex-col h-full">
  <!-- Table -->
  <div class="flex-1 overflow-auto">
    <Table class="min-w-[800px]">
      <TableHeader>
        <TableRow class="hover:bg-transparent">
          <TableHead class="w-10">
            <Checkbox
              checked={allPageSelected}
              indeterminate={!allPageSelected && somePageSelected}
              onCheckedChange={toggleAll}
              aria-label="Select all on page"
            />
          </TableHead>
          <TableHead class="w-14">Thumb</TableHead>
          <TableHead>Filename</TableHead>
          <TableHead>Ground Truth</TableHead>
          <TableHead>Annotation</TableHead>
          <TableHead>Predicted</TableHead>
          <TableHead class="text-right w-20">
            {@render sortHeader("confidence", "Conf.", "Model's confidence in its predicted class")}
          </TableHead>
          <TableHead class="text-right w-20">
            {@render sortHeader("uncertainty", "Uncert.", "Normalized entropy over the predicted class distribution (0 = certain, 1 = uniform)")}
          </TableHead>
          <TableHead class="text-right w-20">
            {@render sortHeader("al_score", "AL", "Active learning score (composite of uncertainty, label status, prediction correctness)")}
          </TableHead>
          <TableHead class="w-48">Top Predictions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {#each data as d (d.id)}
          {@const isSelected = selectedIds.has(d.id)}
          <TableRow
            data-state={isSelected ? "selected" : undefined}
            class={`cursor-pointer h-10 ${d.ignored ? "opacity-40" : ""}`}
            onclick={() => onRowClick(d.id)}
          >
            <TableCell class="py-1" onclick={(e) => e.stopPropagation()}>
              <Checkbox
                checked={isSelected}
                onCheckedChange={() => toggleOne(d.id)}
                aria-label={`Select ${d.filename}`}
              />
            </TableCell>
            <TableCell class="py-1">
              <img
                src={d.thumbnailUrl}
                alt={d.filename}
                class="w-9 h-9 object-cover rounded-(--radius)"
                loading="lazy"
              />
            </TableCell>
            <TableCell class="py-1 font-mono text-xs max-w-[200px]">
              <div class="flex items-center gap-1">
                <Tooltip>
                  <TooltipTrigger>
                    {#snippet child({ props })}
                      <span {...props} class="block truncate flex-1">{d.filename}</span>
                    {/snippet}
                  </TooltipTrigger>
                  <TooltipContent>{d.filename}</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger>
                    {#snippet child({ props })}
                      <button
                        {...props}
                        class="shrink-0 p-0.5 rounded hover:bg-muted transition-colors"
                        onclick={(e) => {
                          e.stopPropagation();
                          onIgnore([d.id], !d.ignored);
                        }}
                      >
                        {#if d.ignored}
                          <EyeOff class="h-3 w-3 text-muted-foreground" />
                        {:else}
                          <Eye class="h-3 w-3 text-muted-foreground/40 hover:text-muted-foreground" />
                        {/if}
                      </button>
                    {/snippet}
                  </TooltipTrigger>
                  <TooltipContent>{d.ignored ? "Unignore image" : "Ignore image"}</TooltipContent>
                </Tooltip>
              </div>
            </TableCell>
            <TableCell class="py-1 max-w-[150px]">
              {@render labelCell(d.groundTruth)}
            </TableCell>
            <TableCell class="py-1 max-w-[150px]">
              {#if d.label}
                {@render labelCell(d.label)}
              {:else}
                <span class="text-xs text-muted-foreground">—</span>
              {/if}
            </TableCell>
            <TableCell class="py-1 max-w-[150px]">
              {#if d.predictedLabel}
                {@render labelCell(d.predictedLabel)}
              {:else}
                <span class="text-xs text-muted-foreground">—</span>
              {/if}
            </TableCell>
            <TableCell class="py-1 text-right text-xs tabular-nums">
              {(d.confidence * 100).toFixed(1)}%
            </TableCell>
            <TableCell class="py-1 text-right text-xs tabular-nums">
              {(d.uncertainty * 100).toFixed(1)}%
            </TableCell>
            <TableCell class="py-1 text-right text-xs tabular-nums">
              {((alScoreMap?.get(d.id) ?? d.alScore ?? 0) * 100).toFixed(1)}%
            </TableCell>
            <TableCell class="py-1">
              <div class="flex gap-1 flex-wrap">
                <!-- index-keyed: detection top_preds can repeat a class name -->
                {#each d.topPredictions.slice(0, 3) as p, i (i)}
                  <Tooltip>
                    <TooltipTrigger>
                      {#snippet child({ props })}
                        <span {...props}>
                          <Badge
                            variant="outline"
                            class="text-[10px] px-1.5 py-0 font-normal max-w-[100px] truncate"
                          >
                            {p.label} {(p.confidence * 100).toFixed(0)}%
                          </Badge>
                        </span>
                      {/snippet}
                    </TooltipTrigger>
                    <TooltipContent>{p.label} {p.confidence * 100}%</TooltipContent>
                  </Tooltip>
                {/each}
              </div>
            </TableCell>
          </TableRow>
        {/each}
        {#if data.length === 0}
          <TableRow>
            <TableCell colspan={10} class="text-center text-muted-foreground py-8">
              No images match the current filters.
            </TableCell>
          </TableRow>
        {/if}
      </TableBody>
    </Table>
  </div>

  <!-- Pagination bar -->
  <div class="h-9 border-t border-border flex items-center px-3 gap-3 shrink-0 text-xs text-muted-foreground">
    <span>
      {#if selectedIds.size > 0}
        <span class="text-foreground font-medium">{selectedIds.size} selected · </span>
      {/if}
      {total.toLocaleString()} images
    </span>
    <div class="flex-1"></div>
    <span>Rows</span>
    <Select type="single" value={String(pageSize)} onValueChange={(v) => onPageSizeChange(Number(v))}>
      <SelectTrigger class="h-6 w-16 text-xs">
        {pageSize}
      </SelectTrigger>
      <SelectContent>
        {#each PAGE_SIZE_OPTIONS as n (n)}
          <SelectItem value={String(n)}>{n}</SelectItem>
        {/each}
      </SelectContent>
    </Select>
    <span class="tabular-nums">
      {total === 0 ? "0" : rangeStart}–{rangeEnd} of {total.toLocaleString()}
    </span>
    <div class="flex gap-0.5">
      <Button variant="ghost" size="sm" class="h-6 w-6 p-0" disabled={page === 1} onclick={() => onPageChange(1)}>
        <ChevronsLeft class="h-3.5 w-3.5" />
      </Button>
      <Button variant="ghost" size="sm" class="h-6 w-6 p-0" disabled={page === 1} onclick={() => onPageChange(page - 1)}>
        <ChevronLeft class="h-3.5 w-3.5" />
      </Button>
      <Button variant="ghost" size="sm" class="h-6 w-6 p-0" disabled={page >= totalPages} onclick={() => onPageChange(page + 1)}>
        <ChevronRight class="h-3.5 w-3.5" />
      </Button>
      <Button variant="ghost" size="sm" class="h-6 w-6 p-0" disabled={page >= totalPages} onclick={() => onPageChange(totalPages)}>
        <ChevronsRight class="h-3.5 w-3.5" />
      </Button>
    </div>
  </div>
</div>
