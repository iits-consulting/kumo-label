<script lang="ts">
	import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
	import { Label } from "@/components/ui/label";
	import Info from "@lucide/svelte/icons/info";

	let { label, tooltip }: { label: string; tooltip: string } = $props();
</script>

<div class="flex items-center gap-1">
	<Label class="text-xs">{label}</Label>
	<Tooltip>
		<TooltipTrigger>
			{#snippet child({ props })}
				<Info {...props} class="h-3 w-3 text-muted-foreground cursor-help" />
			{/snippet}
		</TooltipTrigger>
		<TooltipContent side="top" class="max-w-64 text-xs">
			{tooltip}
		</TooltipContent>
	</Tooltip>
</div>
