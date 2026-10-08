<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import type { DataPoint, TaskType, BoundingBox } from "@/lib/mockData";
  import { api } from "@/lib/api";
  import type { ShortcutMap } from "@/state/shortcutSettings.svelte";
  import { Button } from "@/components/ui/button";
  import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
  import { Progress } from "@/components/ui/progress";
  import X from "@lucide/svelte/icons/x";
  import ZoomIn from "@lucide/svelte/icons/zoom-in";
  import ZoomOut from "@lucide/svelte/icons/zoom-out";
  import Maximize from "@lucide/svelte/icons/maximize";
  import EyeOff from "@lucide/svelte/icons/eye-off";
  import Ban from "@lucide/svelte/icons/ban";
  import ChevronLeft from "@lucide/svelte/icons/chevron-left";
  import ChevronRight from "@lucide/svelte/icons/chevron-right";
  import SkipForward from "@lucide/svelte/icons/skip-forward";
  import Settings from "@lucide/svelte/icons/settings";
  import Plus from "@lucide/svelte/icons/plus";
  import ClassificationPanel from "./ClassificationPanel.svelte";
  import MultilabelPanel from "./MultilabelPanel.svelte";
  import DetectionPanel from "./DetectionPanel.svelte";
  import type { DetectionFragments } from "./types";

  interface Props {
    data: DataPoint[];
    selectedIds: Set<string>;
    // Live AL scores derived from current AL weights.
    alScoreMap?: Map<string, number>;
    taskType: TaskType;
    dbPath: string;
    onClose: () => void;
    onLabel: (id: string, label: string) => void;
    // Multi-label: toggle one class tag on one image
    onToggleTag?: (id: string, cls: string) => void;
    classNames: string[];
    classColors: Record<string, string>;
    onAddClass: (name: string) => void;
    onIgnore: (id: string) => void;
    onNoObjects?: (id: string) => void;
    shortcuts: ShortcutMap;
    keyToClass: Record<string, string>;
    pinnedClasses: Set<string>;
    onTogglePinned: (cls: string) => void;
    budget?: number;
    strategy?: "composite" | "entropy";
    selectedRunId?: string | null;
    onOpenSettings?: () => void;
  }

  let {
    data,
    selectedIds,
    alScoreMap,
    taskType,
    dbPath,
    onClose,
    onLabel,
    onToggleTag,
    classNames,
    classColors,
    onAddClass,
    onIgnore,
    onNoObjects,
    shortcuts,
    keyToClass,
    pinnedClasses,
    onTogglePinned,
    budget = 50,
    strategy = "composite",
    selectedRunId = null,
    onOpenSettings,
  }: Props = $props();

  // FROZEN queue: computed exactly once at init as a plain const (deliberately
  // non-reactive — spec §8.5, checklist 14) so labeling never reshuffles indices.
  function buildQueue(): DataPoint[] {
    const scoreOf = (d: DataPoint): number => {
      if (strategy === "entropy") return d.uncertainty ?? 0;
      return alScoreMap?.get(d.id) ?? d.alScore ?? 0;
    };
    if (selectedIds.size > 0) {
      // Explicit selection: no budget cap.
      return data
        .filter((d) => selectedIds.has(d.id) && !d.ignored && !d.noObjects)
        .sort((a, b) => scoreOf(b) - scoreOf(a));
    }
    if (taskType === "multilabel-classification") {
      // Every reviewable image is a candidate: folder seeding gives most images
      // a tag, and "fully tagged" is undecidable for multi-label. Untagged
      // images come first, then active learning score within each group.
      return data
        .filter((d) => !d.ignored && !d.noObjects)
        .sort((a, b) => {
          const aTagged = a.labels.length === 0 ? 0 : 1;
          const bTagged = b.labels.length === 0 ? 0 : 1;
          if (aTagged !== bTagged) return aTagged - bTagged;
          return scoreOf(b) - scoreOf(a);
        })
        .slice(0, budget);
    }
    // Default: unannotated, non-ignored images sorted by active learning score.
    return data
      .filter((d) => !d.ignored && !d.noObjects && d.label === null)
      .sort((a, b) => scoreOf(b) - scoreOf(a))
      .slice(0, budget);
  }
  const queue = buildQueue();

  let currentIdx = $state(0);
  let zoom = $state(1);
  let panOffset = $state({ x: 0, y: 0 });
  let isPanning = $state(false);
  let labeledCount = $state(0);
  const startTime = Date.now();
  let elapsed = $state("0:00");

  // --- Bbox persistence state (owned here for save flushing on navigation) ---
  let boxesByImage = $state<Record<string, BoundingBox[]>>({});
  // Plain non-reactive save bookkeeping (spec §8.5).
  let saveTimer: ReturnType<typeof setTimeout> | null = null;
  let savingImageId: string | null = null;

  // Classification/multilabel pan gesture start (detection owns its own).
  let panStart: { x: number; y: number } | null = null;

  let imgNaturalDims = $state({ w: 0, h: 0 });
  let canvasEl: HTMLDivElement | undefined;
  // $state only to satisfy bind:this in the snippet; read via getImg() in handlers.
  let imgEl = $state<HTMLImageElement | null>(null);

  // Live record resolution: queue membership/order frozen, displayed fields live.
  const currentQueued = $derived(queue[currentIdx] || null);
  const current = $derived(currentQueued ? data.find((d) => d.id === currentQueued.id) || currentQueued : null);
  const currentBoxes = $derived(current ? boxesByImage[current.id] || [] : []);

  const progressPct = $derived(queue.length > 0 ? ((currentIdx + 1) / queue.length) * 100 : 0);
  const isComplete = $derived(currentIdx >= queue.length - 1 && labeledCount > 0);

  // Reset image dims when the displayed image changes (keyed on current?.id only).
  let prevCurrentId: string | undefined;
  $effect(() => {
    const id = current?.id;
    if (id === prevCurrentId) return;
    prevCurrentId = id;
    imgNaturalDims = { w: 0, h: 0 };
  });

  // --- Debounced detection save (checklist 15): full-list-replace PUT ---
  function flushSave(imageId: string, boxes: BoundingBox[]) {
    if (taskType !== "object-detection") return;
    const numericId = Number(imageId);
    if (isNaN(numericId)) return;
    api
      .put(`/datasets/images/${numericId}/annotations`, {
        db_path: dbPath,
        annotations: boxes.map((b) => ({
          class_name: b.label,
          x: b.x,
          y: b.y,
          width: b.width,
          height: b.height,
        })),
      })
      .catch(() => {/* best-effort */});
  }

  function scheduleSave(imageId: string, boxes: BoundingBox[]) {
    if (saveTimer) clearTimeout(saveTimer);
    savingImageId = imageId;
    saveTimer = setTimeout(() => {
      flushSave(imageId, boxes);
      saveTimer = null;
      savingImageId = null;
    }, 300);
  }

  // Cancel a pending timer for the current image and flush synchronously BEFORE
  // moving. savingImageId is deliberately left set (React parity): destroy then
  // re-beacons the already-flushed list — an idempotent full-replace PUT.
  function flushPendingForCurrent() {
    if (saveTimer && current) {
      clearTimeout(saveTimer);
      saveTimer = null;
      flushSave(current.id, boxesByImage[current.id] || []);
    }
  }

  // onDestroy pending-save fallback: sendBeacon to the raw /api URL (the fetch
  // wrapper can't be used during unload).
  onDestroy(() => {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveTimer = null;
    }
    if (savingImageId) {
      const imgId = savingImageId;
      const numericId = Number(imgId);
      if (!isNaN(numericId)) {
        const url = `/api/datasets/images/${numericId}/annotations`;
        const body = JSON.stringify({
          db_path: dbPath,
          annotations: (boxesByImage[imgId] || []).map((b) => ({
            class_name: b.label, x: b.x, y: b.y, width: b.width, height: b.height,
          })),
        });
        navigator.sendBeacon(url, new Blob([body], { type: "application/json" }));
      }
      savingImageId = null;
    }
  });

  // --- Navigation ---
  function goNext() {
    flushPendingForCurrent();
    currentIdx = Math.min(currentIdx + 1, queue.length - 1);
  }

  function goPrev() {
    flushPendingForCurrent();
    currentIdx = Math.max(currentIdx - 1, 0);
  }

  function handleLabel(label: string) {
    if (!current) return;
    onLabel(current.id, label);
    labeledCount++;
    if (currentIdx < queue.length - 1) currentIdx++;
  }

  function handleIgnore() {
    if (!current) return;
    onIgnore(current.id);
    if (currentIdx < queue.length - 1) currentIdx++;
  }

  function handleNoObjects() {
    if (!current || !onNoObjects) return;
    // Clear any existing boxes for this image + flush save with empty annotations.
    boxesByImage[current.id] = [];
    flushSave(current.id, []);
    onNoObjects(current.id);
    labeledCount++;
    if (currentIdx < queue.length - 1) currentIdx++;
  }

  function handleAcceptPrediction() {
    if (!current) return;
    if (current.predictedLabel) {
      handleLabel(current.predictedLabel);
    } else {
      goNext();
    }
  }

  // --- Multi-label tagging ---
  function handleToggleTagCurrent(cls: string) {
    if (!current || !onToggleTag) return;
    // Counts as labeled the moment the image gets its first tag, not per toggle.
    if (current.labels.length === 0) labeledCount++;
    onToggleTag(current.id, cls);
  }

  function handleAcceptPredictedTags() {
    if (!current) return;
    const missing = (current.predictedLabels ?? []).filter((cls) => !current.labels.includes(cls));
    if (missing.length === 0) {
      goNext();
      return;
    }
    if (current.labels.length === 0) labeledCount++;
    missing.forEach((cls) => onToggleTag?.(current.id, cls));
    if (currentIdx < queue.length - 1) currentIdx++;
  }

  // taskType is fixed for the overlay's lifetime (Explorer derives it once at init).
  // svelte-ignore state_referenced_locally
  const acceptPrediction = taskType === "multilabel-classification" ? handleAcceptPredictedTags : handleAcceptPrediction;

  // Timer + native non-passive wheel zoom.
  onMount(() => {
    const interval = setInterval(() => {
      const secs = Math.floor((Date.now() - startTime) / 1000);
      elapsed = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
    }, 1000);
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      zoom = Math.max(0.3, Math.min(4, zoom * delta));
    };
    canvasEl?.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      clearInterval(interval);
      canvasEl?.removeEventListener("wheel", handleWheel);
    };
  });

  // --- Shared keyboard shortcuts (first keyboard layer — spec §9.17) ---
  function handleSharedKeydown(e: KeyboardEvent) {
    if ((e.target as HTMLElement).tagName === "INPUT") return;
    if (e.key === "Escape") { onClose(); return; }
    if (e.key === " ") { e.preventDefault(); goNext(); return; }
    if (e.key === "ArrowLeft") { goPrev(); return; }
    // For detection mode, ArrowRight navigates to next.
    if (e.key === "ArrowRight" && taskType === "object-detection") { e.preventDefault(); goNext(); return; }
  }

  // --- Classification/multilabel: simple pan-only canvas handlers ---
  function clsCanvasMouseDown(e: MouseEvent) {
    if (e.button === 0) {
      isPanning = true;
      panStart = { x: e.clientX, y: e.clientY };
    }
  }
  function clsCanvasMouseMove(e: MouseEvent) {
    if (isPanning && panStart) {
      panOffset = { x: panOffset.x + (e.clientX - panStart.x), y: panOffset.y + (e.clientY - panStart.y) };
      panStart = { x: e.clientX, y: e.clientY };
    }
  }
  function clsCanvasMouseUp() {
    if (isPanning) {
      isPanning = false;
      panStart = null;
    }
  }

  // --- Box state callbacks for DetectionPanel ---
  function setBoxes(imageId: string, boxes: BoundingBox[]) {
    boxesByImage[imageId] = boxes;
  }
  function mergeBoxes(loaded: Record<string, BoundingBox[]>) {
    boxesByImage = { ...boxesByImage, ...loaded };
  }

  // --- Add class inline ---
  let addingClass = $state(false);
  let newClassName = $state("");
  function submitAddClass() {
    if (newClassName.trim()) onAddClass(newClassName.trim());
    newClassName = "";
    addingClass = false;
  }
  function autofocus(node: HTMLElement) {
    node.focus();
  }

  // svelte-ignore state_referenced_locally
  const taskLabel =
    taskType === "object-detection" ? "Object Detection" : taskType === "multilabel-classification" ? "Multi-Label" : "Classification";
</script>

<svelte:window onkeydown={handleSharedKeydown} />

{#snippet layout(det: DetectionFragments | null)}
  <div class="fixed inset-0 z-50 bg-background flex flex-col">
    <!-- Top bar -->
    <div class="h-10 border-b border-border flex items-center px-3 gap-2 shrink-0">
      <span class="text-xs font-semibold tracking-widest uppercase text-primary">iits</span>
      <span class="text-xs text-muted-foreground">Annotator</span>
      <span class="text-xs text-muted-foreground">•</span>
      <span class="text-xs text-muted-foreground capitalize">{taskLabel}</span>
      <div class="flex-1"></div>
      <span class="text-xs text-muted-foreground tabular-nums">{currentIdx + 1} of {queue.length}</span>
      {#if onOpenSettings}
        <Button variant="ghost" size="sm" class="h-7 w-7 p-0" onclick={onOpenSettings} aria-label="Label settings">
          <Settings class="h-3.5 w-3.5" />
        </Button>
      {/if}
      <Button variant="ghost" size="sm" class="h-7 w-7 p-0" onclick={onClose} aria-label="Close annotator">
        <X class="h-4 w-4" />
      </Button>
    </div>

    <div class="flex-1 flex min-h-0">
      <!-- Image Canvas (70%) -->
      <div class="flex-7 flex flex-col min-w-0">
        <!-- Toolbar -->
        <div class="h-9 border-b border-border flex items-center px-3 gap-1 shrink-0">
          <Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={() => (zoom = Math.min(zoom * 1.2, 4))} aria-label="Zoom in">
            <ZoomIn class="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={() => (zoom = Math.max(zoom / 1.2, 0.3))} aria-label="Zoom out">
            <ZoomOut class="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            class="h-6 w-6 p-0"
            onclick={() => {
              zoom = 1;
              panOffset = { x: 0, y: 0 };
            }}
            aria-label="Fit to view"
          >
            <Maximize class="h-3.5 w-3.5" />
          </Button>
          {#if det}{@render det.toolbar()}{/if}
          <div class="flex-1"></div>
          <div class="flex-1"></div>
          <span class="text-xs text-muted-foreground tabular-nums">{(zoom * 100).toFixed(0)}%</span>
        </div>

        <!-- Image display -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          bind:this={canvasEl}
          class="flex-1 flex items-center justify-center bg-muted/30 overflow-hidden relative cursor-grab"
          onmousedown={det ? det.canvasHandlers.onMouseDown : clsCanvasMouseDown}
          onmousemove={det ? det.canvasHandlers.onMouseMove : clsCanvasMouseMove}
          onmouseup={det ? det.canvasHandlers.onMouseUp : clsCanvasMouseUp}
          onmouseleave={det ? det.canvasHandlers.onMouseLeave : clsCanvasMouseUp}
          oncontextmenu={(e) => e.preventDefault()}
        >
          {#if current}
            <div
              class="relative inline-block max-w-full max-h-full"
              style="transform: scale({zoom}) translate({panOffset.x / zoom}px, {panOffset.y / zoom}px); transform-origin: center; transition: {isPanning ? 'none' : 'transform 0.2s'};"
            >
              <img
                bind:this={imgEl}
                src={current.imageUrl}
                alt={current.filename}
                class="max-w-full max-h-full object-contain select-none"
                draggable="false"
                onload={(e) => {
                  const img = e.currentTarget as HTMLImageElement;
                  imgNaturalDims = { w: img.naturalWidth, h: img.naturalHeight };
                }}
              />
              <!-- Bounding box overlays (detection only) -->
              {#if det}{@render det.canvasOverlay()}{/if}
            </div>
          {:else}
            <p class="text-sm text-muted-foreground">No images in queue</p>
          {/if}
        </div>
      </div>

      <!-- Controls Panel (30%) -->
      <div class="flex-3 border-l border-border flex flex-col min-w-0 overflow-y-auto scrollbar-thin">
        <div class="p-4 space-y-5 text-sm flex-1">
          <!-- Current Image Info -->
          {#if current}
            <section class="space-y-1">
              <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Current Image</h3>
              <p class="text-xs font-mono text-foreground truncate">{current.filename}</p>
              <p class="text-xs text-muted-foreground">{imgNaturalDims.w}×{imgNaturalDims.h}</p>
              {#if taskType === "multilabel-classification"}
                <div class="flex items-center gap-1 flex-wrap text-xs">
                  <span class="text-muted-foreground">Labels</span>
                  {#if current.labels.length > 0}
                    {#each current.labels as cls (cls)}
                      <span
                        class="px-1.5 py-0.5 rounded-full border border-border text-foreground"
                        style={classColors[cls] ? `border-color:${classColors[cls]}` : ""}
                      >
                        {cls}
                      </span>
                    {/each}
                  {:else}
                    <span class="text-muted-foreground italic">None</span>
                  {/if}
                </div>
              {:else}
                <div class="flex items-center gap-1.5 text-xs">
                  <span class="text-muted-foreground">Label</span>
                  {#if current.label}
                    <span class="font-medium text-foreground">{current.label}</span>
                  {:else if current.groundTruth}
                    <span class="text-foreground">{current.groundTruth} <span class="text-muted-foreground italic">(ground truth)</span></span>
                  {:else}
                    <span class="text-muted-foreground italic">None</span>
                  {/if}
                </div>
              {/if}
            </section>
          {/if}

          <!-- Task-specific panel content -->
          {#if taskType === "classification"}
            <ClassificationPanel
              {current}
              {classNames}
              {classColors}
              onLabel={handleLabel}
              onIgnore={handleIgnore}
              {shortcuts}
              {keyToClass}
              {pinnedClasses}
              {onTogglePinned}
              {handleAcceptPrediction}
            />
          {:else if taskType === "multilabel-classification"}
            <MultilabelPanel
              {current}
              {classNames}
              {classColors}
              onIgnore={handleIgnore}
              onToggleTag={handleToggleTagCurrent}
              {shortcuts}
              {keyToClass}
              {pinnedClasses}
              {onTogglePinned}
              handleAcceptPrediction={handleAcceptPredictedTags}
            />
          {:else if det}
            {@render det.panel()}
          {/if}

          <!-- Add class inline -->
          {#if !addingClass}
            <button
              onclick={() => (addingClass = true)}
              class="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <Plus class="h-3 w-3" /> Add class
            </button>
          {:else}
            <input
              use:autofocus
              bind:value={newClassName}
              onblur={submitAddClass}
              onkeydown={(e) => {
                if (e.key === "Enter") submitAddClass();
                if (e.key === "Escape") {
                  addingClass = false;
                  newClassName = "";
                }
              }}
              placeholder="Class name"
              class="h-6 text-xs bg-background border border-border rounded px-1.5 text-foreground outline-hidden focus:border-primary w-full"
            />
          {/if}

          <!-- Navigation -->
          <section class="space-y-3">
            <div class="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger>
                  {#snippet child({ props })}
                    <Button {...props} variant="outline" size="sm" class="flex-1 h-9" onclick={goPrev} disabled={currentIdx === 0}>
                      <ChevronLeft class="h-4 w-4 mr-1" /> Prev
                    </Button>
                  {/snippet}
                </TooltipTrigger>
                <TooltipContent side="top" class="text-xs">Go to previous image (←)</TooltipContent>
              </Tooltip>
              {#if taskType !== "object-detection"}
                <Tooltip>
                  <TooltipTrigger>
                    {#snippet child({ props })}
                      <Button {...props} variant="ghost" size="sm" class="h-9" onclick={handleIgnore} disabled={!current}>
                        <EyeOff class="h-3.5 w-3.5" />
                      </Button>
                    {/snippet}
                  </TooltipTrigger>
                  <TooltipContent side="top" class="text-xs">Ignore this image (Enter)</TooltipContent>
                </Tooltip>
              {/if}
              {#if taskType === "object-detection" && onNoObjects}
                <Tooltip>
                  <TooltipTrigger>
                    {#snippet child({ props })}
                      <Button {...props} variant="ghost" size="sm" class="h-9" onclick={handleNoObjects} disabled={!current}>
                        <Ban class="h-3.5 w-3.5" />
                      </Button>
                    {/snippet}
                  </TooltipTrigger>
                  <TooltipContent side="top" class="text-xs">No objects in this image (n)</TooltipContent>
                </Tooltip>
              {/if}
              <Tooltip>
                <TooltipTrigger>
                  {#snippet child({ props })}
                    <Button {...props} variant="ghost" size="sm" class="h-9" onclick={goNext} disabled={currentIdx >= queue.length - 1}>
                      <SkipForward class="h-3.5 w-3.5" />
                    </Button>
                  {/snippet}
                </TooltipTrigger>
                <TooltipContent side="top" class="text-xs">Skip without labeling (Space)</TooltipContent>
              </Tooltip>
              {#if taskType !== "object-detection"}
                <Tooltip>
                  <TooltipTrigger>
                    {#snippet child({ props })}
                      <Button {...props} variant="outline" size="sm" class="flex-1 h-9" onclick={acceptPrediction} disabled={currentIdx >= queue.length - 1}>
                        Accept <ChevronRight class="h-4 w-4 ml-1" />
                      </Button>
                    {/snippet}
                  </TooltipTrigger>
                  <TooltipContent side="top" class="text-xs">
                    {taskType === "multilabel-classification"
                      ? "Accept predicted labels and continue (→)"
                      : "Accept prediction and continue (→)"}
                  </TooltipContent>
                </Tooltip>
              {:else}
                <Tooltip>
                  <TooltipTrigger>
                    {#snippet child({ props })}
                      <Button {...props} variant="outline" size="sm" class="flex-1 h-9" onclick={goNext} disabled={currentIdx >= queue.length - 1}>
                        Next <ChevronRight class="h-4 w-4 ml-1" />
                      </Button>
                    {/snippet}
                  </TooltipTrigger>
                  <TooltipContent side="top" class="text-xs">Go to next image (→)</TooltipContent>
                </Tooltip>
              {/if}
            </div>
            <div class="space-y-1">
              <Progress value={progressPct} class="h-1.5" />
              <p class="text-xs text-muted-foreground text-center tabular-nums">{currentIdx + 1} / {queue.length}</p>
            </div>
          </section>
        </div>

        <!-- Session stats -->
        <div class="p-4 border-t border-border space-y-3 shrink-0">
          <div class="flex justify-between text-xs">
            <span class="text-muted-foreground">Labeled this session</span>
            <span class="font-medium text-foreground tabular-nums">{labeledCount} / {queue.length}</span>
          </div>
          <div class="flex justify-between text-xs">
            <span class="text-muted-foreground">Time elapsed</span>
            <span class="font-medium text-foreground tabular-nums">{elapsed}</span>
          </div>
          {#if det}{@render det.stats()}{/if}
          {#if isComplete}
            <Button class="w-full" onclick={onClose}>Done — Retrain →</Button>
          {:else}
            <p class="text-[10px] text-muted-foreground text-center">
              {det
                ? det.helpText
                : taskType === "multilabel-classification"
                  ? "Shortcut keys to toggle labels • → accept predicted labels • Enter ignore • Space skip • ← prev"
                  : "Shortcut keys to label • → accept prediction • Enter ignore • Space skip • ← prev"}
            </p>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/snippet}

{#if taskType === "object-detection"}
  <DetectionPanel
    {current}
    {currentBoxes}
    {imgNaturalDims}
    getImg={() => imgEl}
    {classNames}
    {classColors}
    {dbPath}
    {selectedRunId}
    {queue}
    {setBoxes}
    {mergeBoxes}
    {scheduleSave}
    onFirstBox={() => labeledCount++}
    onNoObjects={handleNoObjects}
    setPanning={(v) => (isPanning = v)}
    panBy={(dx, dy) => (panOffset = { x: panOffset.x + dx, y: panOffset.y + dy })}
  >
    {#snippet children(det)}
      {@render layout(det)}
    {/snippet}
  </DetectionPanel>
{:else}
  {@render layout(null)}
{/if}
