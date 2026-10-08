import { toast as sonner } from "svelte-sonner";

// Shim matching the old shadcn `toast({ title, description, variant })` call shape,
// so the ~30 React call sites port mechanically (spec §8.7).
export function toast({
	title,
	description,
	variant,
}: {
	title: string;
	description?: string;
	variant?: "default" | "destructive";
}) {
	if (variant === "destructive") {
		sonner.error(title, { description });
	} else {
		sonner(title, { description });
	}
}
