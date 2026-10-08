<script lang="ts">
  import { onMount, type Snippet } from "svelte";
  import type { DataPoint, BoundingBox } from "@/lib/mockData";
  import { api } from "@/lib/api";
  import { Button } from "@/components/ui/button";
  import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
  import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
  import { Slider } from "@/components/ui/slider";
  import Eye from "@lucide/svelte/icons/eye";
  import EyeOff from "@lucide/svelte/icons/eye-off";
  import Trash2 from "@lucide/svelte/icons/trash-2";
  import Check from "@lucide/svelte/icons/check";
  import CheckCheck from "@lucide/svelte/icons/check-check";
  import type { DetectionFragments } from "./types";

  interface Props {
    current: DataPoint | null;
    currentBoxes: BoundingBox[];
    imgNaturalDims: { w: number; h: number };
    getImg: () => HTMLImageElement | null;
    classNames: string[];
    classColors: Record<string, string>;
    dbPath: string;
    selectedRunId?: string | null;
    queue: DataPoint[];
    // Box state lives in the overlay (for save flushing on navigation) —
    // mutations go through these callbacks.
    setBoxes: (imageId: string, boxes: BoundingBox[]) => void;
    mergeBoxes: (loaded: Record<string, BoundingBox[]>) => void;
    scheduleSave: (imageId: string, boxes: BoundingBox[]) => void;
    onFirstBox: () => void;
    onNoObjects: () => void;
    // Pan is owned by the overlay (drives the canvas transform); detection owns
    // the gesture start point itself.
    setPanning: (v: boolean) => void;
    panBy: (dx: number, dy: number) => void;
    children: Snippet<[DetectionFragments]>;
  }

  let {
    current,
    currentBoxes,
    imgNaturalDims,
    getImg,
    classNames,
    classColors,
    dbPath,
    selectedRunId = null,
    queue,
    setBoxes,
    mergeBoxes,
    scheduleSave,
    onFirstBox,
    onNoObjects,
    setPanning,
    panBy,
    children,
  }: Props = $props();

  type PredictionBox = { id: string; class_name: string; confidence: number; x: number; y: number; width: number; height: number };
  type DragTarget = { id: string; x: number; y: number; width: number; height: number };
  type DragAction =
    | { type: "move"; startPos: { x: number; y: number }; origBox: DragTarget; isPrediction?: boolean }
    | { type: "resize"; handle: string; startPos: { x: number; y: number }; origBox: DragTarget; isPrediction?: boolean };

  // --- Detection-only state ---
  let isDrawing = $state(false);
  let drawStart = $state<{ x: number; y: number } | null>(null);
  let drawCurrent = $state<{ x: number; y: number } | null>(null);
  let selectedBoxId = $state<string | null>(null);
  // svelte-ignore state_referenced_locally
  let lastUsedClass = $state(classNames[0] || "Object");
  let showAnnotations = $state(true);
  let predictedBoxesByImage = $state<Record<string, PredictionBox[]>>({});
  let showPredictions = $state(true);
  let confidenceThreshold = $state(0.5);
  // Gesture bookkeeping — never rendered, deliberately non-reactive.
  let dragAction: DragAction | null = null;
  let panStart: { x: number; y: number } | null = null;

  const filteredPredictions = $derived.by(() => {
    if (!current) return [];
    return (predictedBoxesByImage[current.id] || []).filter((p) => p.confidence >= confidenceThreshold);
  });

  // Load existing annotations on mount (bulk, to avoid request storms).
  onMount(() => {
    if (queue.length === 0) return;
    const imageIds = queue.map((d) => Number(d.id)).filter((n) => !isNaN(n));
    if (imageIds.length === 0) return;
    api
      .post<{ annotations: Record<number, { id: number; class_name: string; x: number; y: number; width: number; height: number }[]> }>(
        "/datasets/annotations/bulk",
        { db_path: dbPath, image_ids: imageIds },
      )
      .then((res) => {
        const loaded: Record<string, BoundingBox[]> = {};
        for (const [imgId, boxes] of Object.entries(res.annotations)) {
          if (boxes.length > 0) {
            loaded[imgId] = boxes.map((b) => ({
              id: `db_${b.id}`,
              x: b.x,
              y: b.y,
              width: b.width,
              height: b.height,
              label: b.class_name,
            }));
          }
        }
        mergeBoxes(loaded);
      })
      .catch(() => {/* ignore */});
  });

  // Fetch detection predictions (single bulk POST). Re-runs only when
  // selectedRunId changes — queue/dbPath are fixed for the overlay's lifetime.
  $effect(() => {
    if (!selectedRunId || queue.length === 0) return;
    const imageIds = queue.map((d) => Number(d.id)).filter((n) => !isNaN(n));
    if (imageIds.length === 0) return;
    api
      .post<{ predictions: Record<string, { class_name: string; confidence: number; x: number; y: number; width: number; height: number }[]> }>(
        `/training/runs/${selectedRunId}/detection-predictions/bulk`,
        { db_path: dbPath, image_ids: imageIds },
      )
      .then((res) => {
        const map: Record<string, PredictionBox[]> = {};
        for (const [imgId, preds] of Object.entries(res.predictions)) {
          if (preds.length > 0) {
            map[imgId] = preds.map((p, idx) => ({ ...p, id: `pred_${imgId}_${idx}` }));
          }
        }
        predictedBoxesByImage = map;
      })
      .catch(() => {/* ignore */});
  });

  // --- Bounding box helpers ---
  // 0–1 normalized coords against the img's getBoundingClientRect (transform-aware).
  function getRelativePos(e: MouseEvent): { x: number; y: number } | null {
    const img = getImg();
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    };
  }

  function addBox(box: BoundingBox) {
    if (!current) return;
    const hadBoxes = currentBoxes.length > 0;
    const next = [...currentBoxes, box];
    setBoxes(current.id, next);
    if (!hadBoxes) onFirstBox();
    scheduleSave(current.id, next);
  }

  function updateBox(boxId: string, updates: Partial<BoundingBox>) {
    if (!current) return;
    const next = currentBoxes.map((b) => (b.id === boxId ? { ...b, ...updates } : b));
    setBoxes(current.id, next);
    scheduleSave(current.id, next);
  }

  function updateBoxLabel(boxId: string, label: string) {
    updateBox(boxId, { label });
    lastUsedClass = label;
  }

  function deleteBox(boxId: string) {
    if (!current) return;
    const next = currentBoxes.filter((b) => b.id !== boxId);
    setBoxes(current.id, next);
    scheduleSave(current.id, next);
    if (selectedBoxId === boxId) selectedBoxId = null;
  }

  // Update prediction position/size (drag/resize — local only, never persisted).
  function updatePrediction(predId: string, updates: Partial<PredictionBox>) {
    if (!current) return;
    predictedBoxesByImage = {
      ...predictedBoxesByImage,
      [current.id]: (predictedBoxesByImage[current.id] || []).map((p) => (p.id === predId ? { ...p, ...updates } : p)),
    };
  }

  // Accept one prediction: convert to a box and remove that prediction.
  function acceptPrediction(pred: PredictionBox) {
    if (!current) return;
    const newBox: BoundingBox = {
      id: `box_${Date.now()}_${pred.id}`,
      x: pred.x,
      y: pred.y,
      width: pred.width,
      height: pred.height,
      label: pred.class_name,
    };
    addBox(newBox);
    predictedBoxesByImage = {
      ...predictedBoxesByImage,
      [current.id]: (predictedBoxesByImage[current.id] || []).filter((p) => p.id !== pred.id),
    };
    selectedBoxId = newBox.id;
  }

  // Accept-all converts only the threshold-filtered set, but clears ALL
  // predictions for the image (spec §9.16).
  function acceptAllPredictions() {
    if (!current || filteredPredictions.length === 0) return;
    const now = Date.now();
    const newBoxes: BoundingBox[] = filteredPredictions.map((pred, i) => ({
      id: `box_${now}_${i}`,
      x: pred.x,
      y: pred.y,
      width: pred.width,
      height: pred.height,
      label: pred.class_name,
    }));
    const hadBoxes = currentBoxes.length > 0;
    const next = [...currentBoxes, ...newBoxes];
    setBoxes(current.id, next);
    if (!hadBoxes) onFirstBox();
    scheduleSave(current.id, next);
    predictedBoxesByImage = { ...predictedBoxesByImage, [current.id]: [] };
  }

  // --- Box drag (move) / resize against a frozen origBox snapshot ---
  function handleBoxMouseDown(e: MouseEvent, box: BoundingBox) {
    if (e.button !== 0) return;
    e.stopPropagation();
    const pos = getRelativePos(e);
    if (!pos) return;
    selectedBoxId = box.id;
    dragAction = { type: "move", startPos: pos, origBox: { ...box } };
  }

  function handleResizeMouseDown(e: MouseEvent, box: BoundingBox, handle: string) {
    if (e.button !== 0) return;
    e.stopPropagation();
    const pos = getRelativePos(e);
    if (!pos) return;
    selectedBoxId = box.id;
    dragAction = { type: "resize", handle, startPos: pos, origBox: { ...box } };
  }

  function handlePredMouseDown(e: MouseEvent, pred: PredictionBox) {
    if (e.button !== 0) return;
    e.stopPropagation();
    const pos = getRelativePos(e);
    if (!pos) return;
    selectedBoxId = pred.id;
    dragAction = { type: "move", startPos: pos, origBox: { ...pred }, isPrediction: true };
  }

  function handlePredResizeMouseDown(e: MouseEvent, pred: PredictionBox, handle: string) {
    if (e.button !== 0) return;
    e.stopPropagation();
    const pos = getRelativePos(e);
    if (!pos) return;
    selectedBoxId = pred.id;
    dragAction = { type: "resize", handle, startPos: pos, origBox: { ...pred }, isPrediction: true };
  }

  // --- Canvas mouse state machine (priority: drag > pan > draw) ---
  function handleCanvasMouseDown(e: MouseEvent) {
    if (e.button === 1) {
      // Middle click: draw new bounding box
      e.preventDefault();
      const pos = getRelativePos(e);
      if (!pos) return;
      isDrawing = true;
      drawStart = pos;
      drawCurrent = pos;
      selectedBoxId = null;
      return;
    }
    if (e.button === 0) {
      // Left click on empty canvas = pan
      selectedBoxId = null;
      panStart = { x: e.clientX, y: e.clientY };
      setPanning(true);
    }
  }

  function handleCanvasMouseMove(e: MouseEvent) {
    if (dragAction) {
      const pos = getRelativePos(e);
      if (!pos) return;
      const dx = pos.x - dragAction.startPos.x;
      const dy = pos.y - dragAction.startPos.y;
      const ob = dragAction.origBox;
      const update = dragAction.isPrediction ? updatePrediction : updateBox;

      if (dragAction.type === "move") {
        const newX = Math.max(0, Math.min(1 - ob.width, ob.x + dx));
        const newY = Math.max(0, Math.min(1 - ob.height, ob.y + dy));
        update(ob.id, { x: newX, y: newY });
      } else {
        const h = dragAction.handle;
        let newX = ob.x, newY = ob.y, newW = ob.width, newH = ob.height;

        if (h.includes("w")) { newX = Math.max(0, ob.x + dx); newW = ob.width - (newX - ob.x); }
        if (h.includes("e")) { newW = Math.max(0.01, Math.min(1 - ob.x, ob.width + dx)); }
        if (h.includes("n")) { newY = Math.max(0, ob.y + dy); newH = ob.height - (newY - ob.y); }
        if (h.includes("s")) { newH = Math.max(0.01, Math.min(1 - ob.y, ob.height + dy)); }

        if (newW > 0.01 && newH > 0.01) {
          update(ob.id, { x: newX, y: newY, width: newW, height: newH });
        }
      }
      return;
    }
    if (panStart) {
      panBy(e.clientX - panStart.x, e.clientY - panStart.y);
      panStart = { x: e.clientX, y: e.clientY };
      return;
    }
    if (!isDrawing) return;
    const pos = getRelativePos(e);
    if (pos) drawCurrent = pos;
  }

  function handleCanvasMouseUp() {
    if (dragAction) { dragAction = null; return; }
    if (panStart) { panStart = null; setPanning(false); return; }
    if (!isDrawing || !drawStart || !drawCurrent) { isDrawing = false; return; }
    const x = Math.min(drawStart.x, drawCurrent.x);
    const y = Math.min(drawStart.y, drawCurrent.y);
    const w = Math.abs(drawCurrent.x - drawStart.x);
    const h = Math.abs(drawCurrent.y - drawStart.y);
    if (w > 0.01 && h > 0.01) {
      const newBox: BoundingBox = { id: `box_${Date.now()}`, x, y, width: w, height: h, label: lastUsedClass || classNames[0] || "Object" };
      addBox(newBox);
      selectedBoxId = newBox.id;
    }
    isDrawing = false;
    drawStart = null;
    drawCurrent = null;
  }

  function handleCanvasMouseLeave() {
    if (isDrawing) { isDrawing = false; drawStart = null; drawCurrent = null; }
    if (dragAction) dragAction = null;
    if (panStart) { panStart = null; setPanning(false); }
  }

  // Detection-specific keyboard shortcuts (second keyboard layer — spec §9.17).
  // Quirk kept: Delete on a selected *prediction* is a no-op (deleteBox only
  // touches committed boxes).
  function handleKeydown(e: KeyboardEvent) {
    if ((e.target as HTMLElement).tagName === "INPUT") return;
    if (e.key === "Delete" && selectedBoxId && current) deleteBox(selectedBoxId);
    if (e.key === "n") onNoObjects();
    if (e.key === "a" && showPredictions && filteredPredictions.length > 0) acceptAllPredictions();
  }

  // Drawing preview rect
  const previewRect = $derived(
    isDrawing && drawStart && drawCurrent
      ? {
          left: `${Math.min(drawStart.x, drawCurrent.x) * 100}%`,
          top: `${Math.min(drawStart.y, drawCurrent.y) * 100}%`,
          width: `${Math.abs(drawCurrent.x - drawStart.x) * 100}%`,
          height: `${Math.abs(drawCurrent.y - drawStart.y) * 100}%`,
        }
      : null,
  );

  const HANDLES = ["nw", "ne", "sw", "se", "n", "s", "w", "e"];
  const HANDLE_CURSORS: Record<string, string> = {
    nw: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", se: "nwse-resize",
    n: "ns-resize", s: "ns-resize", w: "ew-resize", e: "ew-resize",
  };
  const HANDLE_POSITIONS: Record<string, string> = {
    nw: "top:-4px;left:-4px", ne: "top:-4px;right:-4px",
    sw: "bottom:-4px;left:-4px", se: "bottom:-4px;right:-4px",
    n: "top:-4px;left:50%;transform:translateX(-50%)",
    s: "bottom:-4px;left:50%;transform:translateX(-50%)",
    w: "top:50%;left:-4px;transform:translateY(-50%)",
    e: "top:50%;right:-4px;transform:translateY(-50%)",
  };

  const HELP_TEXT = "Middle-click to draw boxes • Left-drag to pan • Del to remove • n no objects • a accept all • ← → navigate";
</script>

<svelte:window onkeydown={handleKeydown} />

{#snippet toolbar()}
  <div class="w-px h-5 bg-border mx-1"></div>
  <Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={() => (showAnnotations = !showAnnotations)} aria-label="Toggle annotations">
    {#if showAnnotations}<Eye class="h-3.5 w-3.5" />{:else}<EyeOff class="h-3.5 w-3.5" />{/if}
  </Button>
  {#if selectedRunId && Object.keys(predictedBoxesByImage).length > 0}
    <Tooltip>
      <TooltipTrigger>
        {#snippet child({ props })}
          <Button
            {...props}
            variant={showPredictions ? "secondary" : "ghost"}
            size="sm"
            class="h-6 px-1.5 text-[10px]"
            onclick={() => (showPredictions = !showPredictions)}
          >
            Preds
          </Button>
        {/snippet}
      </TooltipTrigger>
      <TooltipContent side="bottom" class="text-xs">Toggle predicted boxes</TooltipContent>
    </Tooltip>
    {#if showPredictions}
      <div class="flex items-center gap-1.5">
        <Slider
          type="single"
          value={confidenceThreshold}
          onValueChange={(v) => (confidenceThreshold = v)}
          min={0}
          max={1}
          step={0.05}
          class="w-20"
        />
        <span class="text-[10px] text-muted-foreground tabular-nums w-7">
          {(confidenceThreshold * 100).toFixed(0)}%
        </span>
      </div>
    {/if}
  {/if}
  <!-- Undo/Redo buttons dropped: decorative only in React (spec §10 deviation 2) -->
{/snippet}

{#snippet canvasOverlay()}
  {#if showAnnotations}
    <div class="absolute inset-0 pointer-events-none">
      {#each currentBoxes as box (box.id)}
        <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
        <div
          class={`absolute border-2 pointer-events-auto cursor-move ${selectedBoxId === box.id ? "border-primary" : "border-foreground/50"}`}
          style="left:{box.x * 100}%;top:{box.y * 100}%;width:{box.width * 100}%;height:{box.height * 100}%"
          onmousedown={(e) => handleBoxMouseDown(e, box)}
          onclick={(e) => { e.stopPropagation(); selectedBoxId = box.id; }}
        >
          <span
            class={`absolute -top-5 left-0 text-[10px] px-1 py-0.5 rounded-sm font-medium ${
              selectedBoxId === box.id ? "bg-primary text-primary-foreground" : "bg-foreground/80 text-background"
            }`}
          >
            {box.label}
          </span>
          {#if selectedBoxId === box.id}
            {#each HANDLES as h (h)}
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="absolute w-2 h-2 bg-primary border border-primary-foreground rounded-sm pointer-events-auto"
                style="{HANDLE_POSITIONS[h]};cursor:{HANDLE_CURSORS[h]}"
                onmousedown={(e) => handleResizeMouseDown(e, box, h)}
              ></div>
            {/each}
          {/if}
        </div>
      {/each}
      <!-- Predicted boxes overlay -->
      {#if showPredictions && current}
        {#each filteredPredictions as pred (pred.id)}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            class={`absolute border-2 border-dashed pointer-events-auto ${
              selectedBoxId === pred.id ? "border-amber-400 bg-amber-400/15 cursor-move" : "border-amber-400/70 bg-amber-400/5 cursor-pointer"
            }`}
            style="left:{pred.x * 100}%;top:{pred.y * 100}%;width:{pred.width * 100}%;height:{pred.height * 100}%"
            onmousedown={(e) => handlePredMouseDown(e, pred)}
          >
            <span class="absolute -top-5 left-0 text-[10px] px-1 py-0.5 rounded-sm font-medium bg-amber-400/80 text-black">
              {pred.class_name} {(pred.confidence * 100).toFixed(0)}%
            </span>
            {#if selectedBoxId === pred.id}
              {#each HANDLES as h (h)}
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div
                  class="absolute w-2 h-2 bg-amber-400 border border-amber-600 rounded-sm pointer-events-auto"
                  style="{HANDLE_POSITIONS[h]};cursor:{HANDLE_CURSORS[h]}"
                  onmousedown={(e) => handlePredResizeMouseDown(e, pred, h)}
                ></div>
              {/each}
            {/if}
          </div>
        {/each}
      {/if}
      <!-- Drawing preview -->
      {#if previewRect}
        <div
          class="absolute border-2 border-primary border-dashed bg-primary/10"
          style="left:{previewRect.left};top:{previewRect.top};width:{previewRect.width};height:{previewRect.height}"
        ></div>
      {/if}
    </div>
  {/if}
{/snippet}

{#snippet panel()}
  <section class="space-y-2">
    <div class="flex items-center justify-between">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Bounding Boxes</h3>
    </div>
    {#if currentBoxes.length === 0}
      <p class="text-xs text-muted-foreground">Middle-click and drag on image to draw boxes.</p>
    {/if}
    <div class="space-y-1.5">
      {#each currentBoxes as box (box.id)}
        <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
        <div
          class={`flex items-center gap-2 p-2 rounded-md border text-xs cursor-pointer transition-colors ${
            selectedBoxId === box.id ? "border-primary bg-primary/5" : "border-border hover:bg-muted"
          }`}
          onclick={() => (selectedBoxId = box.id)}
        >
          <div class="w-3 h-3 rounded-sm shrink-0" style="background-color:{classColors[box.label] || '#9CA3AF'}"></div>
          <Select type="single" value={box.label} onValueChange={(val) => updateBoxLabel(box.id, val)}>
            <SelectTrigger class="h-6 text-xs flex-1 min-w-0">{box.label}</SelectTrigger>
            <SelectContent>
              {#each classNames as cls (cls)}
                <SelectItem value={cls}>{cls}</SelectItem>
              {/each}
            </SelectContent>
          </Select>
          <span class="text-muted-foreground tabular-nums shrink-0">
            {Math.round(box.width * (imgNaturalDims.w || 640))}×{Math.round(box.height * (imgNaturalDims.h || 480))}
          </span>
          <Button
            variant="ghost"
            size="sm"
            class="h-5 w-5 p-0 shrink-0"
            onclick={(e) => { e.stopPropagation(); deleteBox(box.id); }}
            aria-label="Delete box"
          >
            <Trash2 class="h-3 w-3" />
          </Button>
        </div>
      {/each}
    </div>
  </section>
  {#if showPredictions && filteredPredictions.length > 0}
    <section class="space-y-2">
      <div class="flex items-center justify-between">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-amber-500">Predictions</h3>
        <Button variant="ghost" size="sm" class="h-5 px-1.5 text-[10px] text-amber-500 hover:text-amber-400" onclick={acceptAllPredictions}>
          <CheckCheck class="h-3 w-3 mr-1" />
          Accept All
        </Button>
      </div>
      <div class="space-y-1.5">
        {#each filteredPredictions as pred (pred.id)}
          <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
          <div
            class={`flex items-center gap-2 p-2 rounded-md border text-xs cursor-pointer transition-colors ${
              selectedBoxId === pred.id ? "border-amber-400 bg-amber-400/10" : "border-amber-400/30 bg-amber-400/5 hover:bg-amber-400/10"
            }`}
            onclick={() => (selectedBoxId = pred.id)}
          >
            <div class="w-3 h-3 rounded-sm shrink-0" style="background-color:{classColors[pred.class_name] || '#F59E0B'}"></div>
            <span class="flex-1 truncate">{pred.class_name}</span>
            <span class="text-amber-500 tabular-nums shrink-0">
              {(pred.confidence * 100).toFixed(0)}%
            </span>
            <Tooltip>
              <TooltipTrigger>
                {#snippet child({ props })}
                  <Button
                    {...props}
                    variant="ghost"
                    size="sm"
                    class="h-5 w-5 p-0 shrink-0 text-amber-500 hover:text-amber-400"
                    onclick={(e) => { e.stopPropagation(); acceptPrediction(pred); }}
                    aria-label="Accept prediction"
                  >
                    <Check class="h-3 w-3" />
                  </Button>
                {/snippet}
              </TooltipTrigger>
              <TooltipContent side="left" class="text-xs">Accept as annotation</TooltipContent>
            </Tooltip>
          </div>
        {/each}
      </div>
    </section>
  {/if}
{/snippet}

{#snippet stats()}
  <div class="flex justify-between text-xs">
    <span class="text-muted-foreground">Boxes (this image)</span>
    <span class="font-medium text-foreground tabular-nums">{currentBoxes.length}</span>
  </div>
  {#if showPredictions && filteredPredictions.length > 0}
    <div class="flex justify-between text-xs">
      <span class="text-muted-foreground">Predictions</span>
      <span class="font-medium text-amber-500 tabular-nums">{filteredPredictions.length}</span>
    </div>
  {/if}
{/snippet}

{@render children({
  canvasHandlers: {
    onMouseDown: handleCanvasMouseDown,
    onMouseMove: handleCanvasMouseMove,
    onMouseUp: handleCanvasMouseUp,
    onMouseLeave: handleCanvasMouseLeave,
  },
  toolbar,
  canvasOverlay,
  panel,
  stats,
  helpText: HELP_TEXT,
})}
