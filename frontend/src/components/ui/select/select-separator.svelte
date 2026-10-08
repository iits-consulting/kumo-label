<script lang="ts">
	import { Separator as SeparatorPrimitive } from "bits-ui";
	import { cn } from "@/lib/utils";

	let {
		ref = $bindable(null),
		class: className,
		...restProps
	}: SeparatorPrimitive.RootProps = $props();
</script>

<!-- ponytail: uses the bits-ui primitive directly instead of vendoring the unused ui/separator component -->
<SeparatorPrimitive.Root bind:ref class={cn("bg-muted -mx-1 my-1 h-px", className)} {...restProps} />
