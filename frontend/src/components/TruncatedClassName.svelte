<script lang="ts">
  import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
  import { cn } from "@/lib/utils";

  let { name, class: className }: { name: string; class?: string } = $props();
</script>

<Tooltip>
  <TooltipTrigger>
    {#snippet child({ props })}
      <span {...props} class={cn("truncate", className)}>{name}</span>
    {/snippet}
  </TooltipTrigger>
  <TooltipContent side="top" class="text-xs max-w-[20rem] break-all">
    {name}
  </TooltipContent>
</Tooltip>
