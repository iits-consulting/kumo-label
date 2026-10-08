<script lang="ts">
  import { type DataPoint, getHighValueCount } from "@/lib/mockData";
  import type { ActionBarSection, ReductionMethod } from "@/lib/types";
  import CircleCheckBig from "@lucide/svelte/icons/circle-check-big";
  import Circle from "@lucide/svelte/icons/circle";
  import LoaderCircle from "@lucide/svelte/icons/loader-circle";
  import Target from "@lucide/svelte/icons/target";
  import Cpu from "@lucide/svelte/icons/cpu";
  import Shrink from "@lucide/svelte/icons/shrink";

  interface Props {
    activeSection: ActionBarSection;
    onSectionToggle: (section: Exclude<ActionBarSection, null>) => void;
    embeddingStatus: { exists: boolean; count: number; isComputing: boolean };
    embeddingModel: "dinov2" | "clip";
    reductionStatus: { method: ReductionMethod; isLoading: boolean };
    data: DataPoint[];
  }

  let { activeSection, onSectionToggle, embeddingStatus, embeddingModel, reductionStatus, data }: Props = $props();

  const highValueCount = $derived(getHighValueCount(data));
</script>

<div class="border-b border-border bg-card shrink-0">
  <div class="flex items-center h-9 px-3 gap-1 text-xs">
    <!-- Embeddings trigger -->
    <button
      onclick={() => onSectionToggle("embeddings")}
      class={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors ${
        activeSection === "embeddings"
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      }`}
    >
      <Cpu class="h-3 w-3" />
      <span class="font-medium">Embeddings</span>
      <span class="uppercase">{embeddingModel === "dinov2" ? "DINOv2" : "CLIP"}</span>
      {#if embeddingStatus.isComputing}
        <LoaderCircle class="h-3 w-3 animate-spin text-primary" />
      {:else if embeddingStatus.exists}
        <CircleCheckBig class="h-3 w-3 text-green-500" />
      {:else}
        <Circle class="h-3 w-3" />
      {/if}
    </button>

    <div class="w-px h-4 bg-border"></div>

    <!-- Reduction trigger -->
    <button
      onclick={() => onSectionToggle("reduction")}
      class={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors ${
        activeSection === "reduction"
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      }`}
    >
      <Shrink class="h-3 w-3" />
      <span class="font-medium">Reduction</span>
      <span class="uppercase">{reductionStatus.method}</span>
      {#if reductionStatus.isLoading}
        <LoaderCircle class="h-3 w-3 animate-spin text-primary" />
      {/if}
    </button>

    <div class="w-px h-4 bg-border"></div>

    <!-- Active Learning trigger -->
    <button
      onclick={() => onSectionToggle("activeLearning")}
      class={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors ${
        activeSection === "activeLearning"
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      }`}
    >
      <Target class="h-3 w-3 text-primary" />
      <span class="font-medium">Active Learning</span>
      {#if highValueCount > 0}
        <span class="bg-primary/15 text-primary px-1.5 rounded-full tabular-nums">
          {highValueCount}
        </span>
      {/if}
    </button>
  </div>
</div>
