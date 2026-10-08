<script lang="ts">
  import { untrack } from "svelte";
  import { type ALWeights, DEFAULT_AL_WEIGHTS } from "@/lib/alScore";
  import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
  import { HintLabel } from "@/components/ui/hint-label";
  import { Slider } from "@/components/ui/slider";
  import { Button } from "@/components/ui/button";
  import RotateCcw from "@lucide/svelte/icons/rotate-ccw";

  interface Props {
    open: boolean;
    onClose: () => void;
    weights: ALWeights;
    onSave: (weights: ALWeights) => void;
  }

  let { open, onClose, weights, onSave }: Props = $props();

  const WEIGHT_FIELDS = [
    { key: "uncertainty", label: "Uncertainty", tooltip: "Prioritize images where the model is least certain about any class (high entropy)" },
    { key: "unlabeled", label: "Unlabeled", tooltip: "Prioritize images that have no label yet" },
    { key: "wrongPrediction", label: "Wrong Prediction", tooltip: "Prioritize images where the model prediction disagrees with the current label" },
    { key: "inverseConfidence", label: "Low Confidence", tooltip: "Prioritize images where the model's top prediction has low confidence" },
    { key: "classImbalance", label: "Class Imbalance", tooltip: "Prioritize images from underrepresented classes to balance the dataset" },
  ] as const;

  // Staging copy: resync from props only when `open` transitions to true, so
  // in-dialog edits never fight the parent value (spec §8.3 / checklist 12).
  // svelte-ignore state_referenced_locally
  let local = $state<ALWeights>({ ...weights });
  let prevOpen = false;
  $effect(() => {
    if (open && !prevOpen) {
      local = { ...untrack(() => weights) };
    }
    prevOpen = open;
  });

  function update(key: keyof ALWeights, value: number) {
    local = { ...local, [key]: value };
  }
</script>

<Dialog {open} onOpenChange={(o) => !o && onClose()}>
  <DialogContent class="max-w-lg max-h-[80vh] flex flex-col gap-0 p-0">
    <DialogHeader class="px-5 py-4 border-b border-border shrink-0">
      <DialogTitle class="text-sm font-semibold">Score Weights</DialogTitle>
    </DialogHeader>

    <div class="px-5 py-4 overflow-y-auto scrollbar-thin flex-1 space-y-3">
      {#each WEIGHT_FIELDS as { key, label, tooltip } (key)}
        <div class="space-y-1.5">
          <div class="flex justify-between text-xs">
            <HintLabel {label} {tooltip} />
            <span class="text-muted-foreground tabular-nums">{local[key].toFixed(2)}</span>
          </div>
          <Slider
            type="single"
            min={0}
            max={1}
            step={0.05}
            value={local[key]}
            onValueChange={(v) => update(key, v)}
          />
        </div>
      {/each}
    </div>

    <DialogFooter class="px-5 py-3 border-t border-border flex-row justify-between sm:justify-between">
      <Button variant="ghost" size="sm" class="text-xs gap-1.5" onclick={() => (local = { ...DEFAULT_AL_WEIGHTS })}>
        <RotateCcw class="h-3 w-3" />
        Reset to Defaults
      </Button>
      <Button
        size="sm"
        class="text-xs"
        onclick={() => {
          onSave($state.snapshot(local) as ALWeights);
          onClose();
        }}
      >
        Save
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
