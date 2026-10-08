<script lang="ts">
  import { untrack } from "svelte";
  import { type AdvancedSettings, ADVANCED_DEFAULTS } from "@/lib/types";
  import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
  import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
  import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
  import { HintLabel } from "@/components/ui/hint-label";
  import { Slider } from "@/components/ui/slider";
  import { Switch } from "@/components/ui/switch";
  import { Input } from "@/components/ui/input";
  import { Button } from "@/components/ui/button";
  import RotateCcw from "@lucide/svelte/icons/rotate-ccw";

  interface Props {
    open: boolean;
    onClose: () => void;
    settings: AdvancedSettings;
    onSave: (settings: AdvancedSettings) => void;
    augmentationEnabled: boolean;
  }

  let { open, onClose, settings, onSave, augmentationEnabled }: Props = $props();

  // Staging copy: resync from props only when `open` transitions to true, so
  // in-dialog edits never fight the parent value (spec §8.3 / checklist 12).
  // svelte-ignore state_referenced_locally
  let local = $state<AdvancedSettings>({ ...settings });
  let prevOpen = false;
  $effect(() => {
    if (open && !prevOpen) {
      local = { ...untrack(() => settings) };
    }
    prevOpen = open;
  });

  function update<K extends keyof AdvancedSettings>(key: K, value: AdvancedSettings[K]) {
    local = { ...local, [key]: value };
  }

  let accordionValue = $state(["training", "augmentation", "model", "performance"]);

  const OPTIMIZER_LABELS: Record<string, string> = { adamw: "AdamW", sgd: "SGD", adam: "Adam" };
  const SCHEDULER_LABELS: Record<string, string> = { cosine: "Cosine Annealing", linear: "Linear Decay", none: "None" };
  const PRECISION_LABELS: Record<string, string> = { "32-true": "FP32 (default)", "16-mixed": "FP16 Mixed", "bf16-mixed": "BF16 Mixed" };
</script>

<Dialog {open} onOpenChange={(o) => !o && onClose()}>
  <DialogContent class="max-w-lg max-h-[80vh] flex flex-col gap-0 p-0">
    <DialogHeader class="px-5 py-4 border-b border-border shrink-0">
      <DialogTitle class="text-sm font-semibold">Advanced Settings</DialogTitle>
    </DialogHeader>

    <div class="px-5 overflow-y-auto scrollbar-thin flex-1">
      <Accordion type="multiple" bind:value={accordionValue}>
        <!-- Training -->
        <AccordionItem value="training">
          <AccordionTrigger class="text-xs font-medium py-3">Training</AccordionTrigger>
          <AccordionContent class="space-y-3 pb-4">
            <div class="space-y-1">
              <HintLabel label="Optimizer" tooltip="Algorithm used to update model weights. AdamW is best for most cases, SGD can generalize better." />
              <Select type="single" value={local.optimizer} onValueChange={(v) => update("optimizer", v as AdvancedSettings["optimizer"])}>
                <SelectTrigger class="h-8 text-xs">{OPTIMIZER_LABELS[local.optimizer]}</SelectTrigger>
                <SelectContent>
                  <SelectItem value="adamw">AdamW</SelectItem>
                  <SelectItem value="sgd">SGD</SelectItem>
                  <SelectItem value="adam">Adam</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div class="space-y-1">
              <HintLabel label="Scheduler" tooltip="Controls how the learning rate changes during training. Cosine annealing smoothly decays to zero." />
              <Select type="single" value={local.scheduler} onValueChange={(v) => update("scheduler", v as AdvancedSettings["scheduler"])}>
                <SelectTrigger class="h-8 text-xs">{SCHEDULER_LABELS[local.scheduler]}</SelectTrigger>
                <SelectContent>
                  <SelectItem value="cosine">Cosine Annealing</SelectItem>
                  <SelectItem value="linear">Linear Decay</SelectItem>
                  <SelectItem value="none">None</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div class="space-y-1">
              <HintLabel label="Warmup Epochs" tooltip="Number of epochs to gradually increase the learning rate from near-zero. Helps stabilize early training." />
              <Input
                type="number"
                min={0}
                value={local.warmup_epochs}
                oninput={(e) => update("warmup_epochs", parseInt(e.currentTarget.value) || 0)}
                class="h-8 text-xs font-mono"
              />
            </div>
            <div class="space-y-1">
              <HintLabel label="Weight Decay" tooltip="L2 regularization strength. Penalizes large weights to reduce overfitting. Typical range: 0.001 - 0.1." />
              <Input
                type="number"
                min={0}
                max={1}
                step={0.001}
                value={local.weight_decay}
                oninput={(e) => update("weight_decay", parseFloat(e.currentTarget.value) || 0)}
                class="h-8 text-xs font-mono"
              />
            </div>
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <HintLabel label="Early Stopping" tooltip="Stop training automatically when validation loss stops improving, preventing overfitting." />
                <Switch checked={local.early_stopping} onCheckedChange={(v) => update("early_stopping", v)} />
              </div>
              {#if local.early_stopping}
                <div class="space-y-1">
                  <HintLabel label="Patience (epochs)" tooltip="Number of epochs with no improvement before stopping. Higher values allow more time to recover." />
                  <Input
                    type="number"
                    min={1}
                    max={100}
                    value={local.early_stopping_patience}
                    oninput={(e) => update("early_stopping_patience", parseInt(e.currentTarget.value) || 5)}
                    class="h-8 text-xs font-mono"
                  />
                </div>
              {/if}
            </div>
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <HintLabel label="Validation Frequency" tooltip="How often to run validation during each epoch. Lower values give more frequent feedback but slow training." />
                <span class="text-muted-foreground">
                  {local.val_check_interval >= 1 ? "Once per epoch" : `${Math.round(1 / local.val_check_interval)}x per epoch`}
                </span>
              </div>
              <Slider type="single" min={0.1} max={1} step={0.1} value={local.val_check_interval} onValueChange={(v) => update("val_check_interval", v)} />
            </div>
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <HintLabel label="Iterations per Epoch" tooltip="Limit how many training batches run per epoch. Useful for large datasets where a full epoch takes too long. 0 = use all batches." />
                <span class="text-muted-foreground">
                  {local.limit_train_batches === 0 ? "All" : local.limit_train_batches}
                </span>
              </div>
              <Input
                type="number"
                min={0}
                value={local.limit_train_batches}
                oninput={(e) => update("limit_train_batches", Math.max(0, parseInt(e.currentTarget.value) || 0))}
                class="h-8 text-xs font-mono"
              />
            </div>
          </AccordionContent>
        </AccordionItem>

        <!-- Augmentation -->
        <AccordionItem value="augmentation">
          <AccordionTrigger class="text-xs font-medium py-3">
            Augmentation
            {#if !augmentationEnabled}
              <span class="text-muted-foreground font-normal ml-1.5">(disabled)</span>
            {/if}
          </AccordionTrigger>
          <AccordionContent class={`space-y-3 pb-4 ${!augmentationEnabled ? "opacity-40 pointer-events-none" : ""}`}>
            <div class="flex items-center justify-between">
              <HintLabel label="Horizontal Flip" tooltip="Randomly flip images horizontally. Useful when orientation doesn't matter for classification." />
              <Switch checked={local.horizontal_flip} onCheckedChange={(v) => update("horizontal_flip", v)} />
            </div>
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <HintLabel label="Rotation" tooltip="Maximum random rotation angle in degrees. Adds rotational invariance to the model." />
                <span class="text-muted-foreground">{local.rotation}°</span>
              </div>
              <Slider type="single" min={0} max={45} step={5} value={local.rotation} onValueChange={(v) => update("rotation", v)} />
            </div>
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <HintLabel label="Brightness Jitter" tooltip="Random brightness variation. Higher values create more diverse lighting conditions." />
                <span class="text-muted-foreground">{local.color_jitter_brightness.toFixed(2)}</span>
              </div>
              <Slider type="single" min={0} max={1} step={0.05} value={local.color_jitter_brightness} onValueChange={(v) => update("color_jitter_brightness", v)} />
            </div>
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <HintLabel label="Contrast Jitter" tooltip="Random contrast variation. Helps the model handle images with different contrast levels." />
                <span class="text-muted-foreground">{local.color_jitter_contrast.toFixed(2)}</span>
              </div>
              <Slider type="single" min={0} max={1} step={0.05} value={local.color_jitter_contrast} onValueChange={(v) => update("color_jitter_contrast", v)} />
            </div>
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <HintLabel label="Saturation Jitter" tooltip="Random color saturation variation. Makes the model robust to color intensity differences." />
                <span class="text-muted-foreground">{local.color_jitter_saturation.toFixed(2)}</span>
              </div>
              <Slider type="single" min={0} max={1} step={0.05} value={local.color_jitter_saturation} onValueChange={(v) => update("color_jitter_saturation", v)} />
            </div>
          </AccordionContent>
        </AccordionItem>

        <!-- Model -->
        <AccordionItem value="model">
          <AccordionTrigger class="text-xs font-medium py-3">Model</AccordionTrigger>
          <AccordionContent class="space-y-3 pb-4">
            <div class="space-y-1">
              <HintLabel label="Image Size" tooltip="Input resolution for the model. Larger sizes capture more detail but use more memory and train slower." />
              <Select type="single" value={String(local.image_size)} onValueChange={(v) => update("image_size", parseInt(v))}>
                <SelectTrigger class="h-8 text-xs">{local.image_size}</SelectTrigger>
                <SelectContent>
                  <SelectItem value="224">224</SelectItem>
                  <SelectItem value="256">256</SelectItem>
                  <SelectItem value="384">384</SelectItem>
                  <SelectItem value="512">512</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div class="flex items-center justify-between">
              <div class="space-y-0.5">
                <HintLabel label="Freeze Backbone" tooltip="Only train the classification head while keeping pretrained features frozen. Recommended for small datasets." />
                <p class="text-[10px] text-muted-foreground">Only train the classification head</p>
              </div>
              <Switch checked={local.freeze_backbone} onCheckedChange={(v) => update("freeze_backbone", v)} />
            </div>
            <div class="space-y-1">
              <HintLabel label="Precision" tooltip="Numerical precision for training. Mixed precision (FP16/BF16) speeds up training and reduces memory usage on modern GPUs." />
              <Select type="single" value={local.precision} onValueChange={(v) => update("precision", v as AdvancedSettings["precision"])}>
                <SelectTrigger class="h-8 text-xs">{PRECISION_LABELS[local.precision]}</SelectTrigger>
                <SelectContent>
                  <SelectItem value="32-true">FP32 (default)</SelectItem>
                  <SelectItem value="16-mixed">FP16 Mixed</SelectItem>
                  <SelectItem value="bf16-mixed">BF16 Mixed</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </AccordionContent>
        </AccordionItem>

        <!-- Performance -->
        <AccordionItem value="performance">
          <AccordionTrigger class="text-xs font-medium py-3">Performance</AccordionTrigger>
          <AccordionContent class="space-y-3 pb-4">
            <div class="space-y-1">
              <HintLabel label="Gradient Accumulation Steps" tooltip="Accumulate gradients over multiple batches before updating weights. Simulates a larger batch size with less memory." />
              <Input
                type="number"
                min={1}
                max={64}
                value={local.accumulate_grad_batches}
                oninput={(e) => update("accumulate_grad_batches", parseInt(e.currentTarget.value) || 1)}
                class="h-8 text-xs font-mono"
              />
            </div>
            <div class="space-y-1">
              <HintLabel label="Gradient Clip Value" tooltip="Limit the magnitude of gradients to prevent exploding gradients. Lower values increase stability but may slow convergence. Recommended: 0.1-5.0" />
              <Input
                type="number"
                min={0}
                max={10}
                step={0.1}
                value={local.gradient_clip_val}
                oninput={(e) => update("gradient_clip_val", parseFloat(e.currentTarget.value) || 0)}
                class="h-8 text-xs font-mono"
              />
              <p class="text-[10px] text-muted-foreground">0 = disabled</p>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>

    <DialogFooter class="px-5 py-3 border-t border-border flex-row justify-between sm:justify-between">
      <Button variant="ghost" size="sm" class="text-xs gap-1.5" onclick={() => (local = { ...ADVANCED_DEFAULTS })}>
        <RotateCcw class="h-3 w-3" />
        Reset to Defaults
      </Button>
      <Button
        size="sm"
        class="text-xs"
        onclick={() => {
          onSave($state.snapshot(local) as AdvancedSettings);
          onClose();
        }}
      >
        Save
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
