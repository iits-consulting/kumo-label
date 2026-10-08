<script lang="ts">
  import { type DataPoint, getHighValueCount } from "@/lib/mockData";
  import type { ALWeights } from "@/lib/alScore";
  import { Button } from "@/components/ui/button";
  import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
  import Settings2 from "@lucide/svelte/icons/settings-2";
  import Target from "@lucide/svelte/icons/target";
  import X from "@lucide/svelte/icons/x";
  import ALWeightsDialog from "./ALWeightsDialog.svelte";
  import { HintLabel } from "@/components/ui/hint-label";

  interface Props {
    data: DataPoint[];
    onStartLabeling: (budget: number, strategy: "composite" | "entropy") => void;
    alWeights: ALWeights;
    onALWeightsChange: (weights: ALWeights) => void;
    onClose: () => void;
  }

  let { data, onStartLabeling, alWeights, onALWeightsChange, onClose }: Props = $props();

  // Local budget default 10 — synced into Explorer's 50-default labelingBudget
  // only on "Start Labeling Session" (spec §8.3).
  let budget = $state(10);
  let strategy = $state<"composite" | "entropy">("composite");
  let weightsDialogOpen = $state(false);
  const highValueCount = $derived(getHighValueCount(data));
</script>

<div class="flex flex-col h-full">
  <div class="flex items-center justify-between px-3 py-2 border-b border-border">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
      <Target class="h-3.5 w-3.5 text-primary" />
      Active Learning
    </h3>
    <button onclick={onClose} class="text-muted-foreground hover:text-foreground">
      <X class="h-3.5 w-3.5" />
    </button>
  </div>

  <div class="p-3 space-y-3">
    <div class="flex items-center gap-1.5 text-xs">
      <Target class="h-3.5 w-3.5 text-primary" />
      <span class="font-medium text-foreground">
        {highValueCount} high-value images to label
      </span>
    </div>

    <div class="space-y-2">
      <HintLabel
        label="Strategy"
        tooltip="Choose a strategy: Use a Composite Score (with customizable weights) or focus strictly on Uncertainty. The composite score allows you to balance multiple metrics manually."
      />
      <Select type="single" value={strategy} onValueChange={(v) => (strategy = v as "composite" | "entropy")}>
        <SelectTrigger class="h-8 text-xs">
          {strategy === "composite" ? "Composite Score" : "Uncertainty Only"}
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="composite">Composite Score</SelectItem>
          <SelectItem value="entropy">Uncertainty Only</SelectItem>
        </SelectContent>
      </Select>
    </div>

    {#if strategy === "composite"}
      <Button
        variant="outline"
        size="sm"
        class="w-full text-xs h-8 gap-1.5"
        onclick={() => (weightsDialogOpen = true)}
      >
        <Settings2 class="h-3 w-3" />
        Score Weights
      </Button>
    {/if}

    <div class="space-y-2">
      <HintLabel label="Budget" tooltip="The number of images to label in this session." />

      <input
        type="number"
        value={budget}
        min={1}
        oninput={(e) => (budget = Math.max(1, parseInt(e.currentTarget.value) || 1))}
        class="w-full h-8 px-2 text-xs font-mono bg-background border border-border rounded-md text-foreground"
      />
    </div>

    <Button size="sm" class="w-full text-xs h-8" onclick={() => onStartLabeling(budget, strategy)}>
      Start Labeling Session
    </Button>
  </div>

  <ALWeightsDialog
    open={weightsDialogOpen}
    onClose={() => (weightsDialogOpen = false)}
    weights={alWeights}
    onSave={onALWeightsChange}
  />
</div>
