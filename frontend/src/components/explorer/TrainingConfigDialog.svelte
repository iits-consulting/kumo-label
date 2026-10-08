<script lang="ts">
  import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
  import { Button } from "@/components/ui/button";
  import Download from "@lucide/svelte/icons/download";
  import type { TrainingRun } from "@/lib/types";

  interface Props {
    open: boolean;
    onClose: () => void;
    run: TrainingRun | null;
    dbPath: string;
  }

  let { open, onClose, run, dbPath }: Props = $props();
</script>

<Dialog {open} onOpenChange={(o) => !o && onClose()}>
  <DialogContent class="sm:max-w-2xl">
    <DialogHeader>
      <DialogTitle>{run ? `Config — ${run.display_name}` : "Config"}</DialogTitle>
    </DialogHeader>

    {#if run}
      <div class="space-y-3">
        <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <div class="text-muted-foreground">Model</div>
          <div class="font-mono">{run.model_name}</div>
          <div class="text-muted-foreground">Task</div>
          <div class="font-mono">{run.task_type ?? "—"}</div>
          <div class="text-muted-foreground">Classes</div>
          <div class="font-mono">{run.num_classes}</div>
          <div class="text-muted-foreground">Epochs</div>
          <div class="font-mono">{run.epochs}</div>
          <div class="text-muted-foreground">Created</div>
          <div class="font-mono">{run.created_at?.replace("T", " ").slice(0, 19)}</div>
        </div>
        <pre class="text-xs font-mono bg-muted/40 border border-border rounded p-3 max-h-80 overflow-auto whitespace-pre-wrap break-all">{JSON.stringify(run.config ?? {}, null, 2)}</pre>
      </div>
    {:else}
      <p class="text-xs text-muted-foreground">No run selected</p>
    {/if}

    <DialogFooter>
      <Button variant="ghost" onclick={onClose}>Close</Button>
      {#if run}
        <a href={`/api/training/runs/${run.id}/config.yaml?db_path=${encodeURIComponent(dbPath)}`} download>
          <Button>
            <Download class="h-3.5 w-3.5 mr-1.5" />
            Download YAML
          </Button>
        </a>
      {/if}
    </DialogFooter>
  </DialogContent>
</Dialog>
