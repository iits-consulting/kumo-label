<script lang="ts">
  import { onMount } from "svelte";
  import createScatterplot from "regl-scatterplot";
  import { type DataPoint, type TaskType, effectiveSplit } from "@/lib/mockData";
  import type { ColorBy, SelectionTool } from "@/lib/taskConfig";
  import { Button } from "@/components/ui/button";
  import MousePointer2 from "@lucide/svelte/icons/mouse-pointer-2";
  import Square from "@lucide/svelte/icons/square";
  import Crosshair from "@lucide/svelte/icons/crosshair";
  import ZoomIn from "@lucide/svelte/icons/zoom-in";
  import ZoomOut from "@lucide/svelte/icons/zoom-out";
  import Maximize2 from "@lucide/svelte/icons/maximize-2";
  import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";

  interface Props {
    data: DataPoint[];
    allData: DataPoint[];
    colorBy: ColorBy;
    selectedIds: Set<string>;
    selectionTool: SelectionTool;
    onSelect: (ids: Set<string>) => void;
    onHover: (id: string | null) => void;
    onSelectionToolChange: (t: SelectionTool) => void;
    taskType: TaskType;
    classNames: string[];
    classColors: Record<string, string>;
    // Multi-label: class highlighted by the "hasClass" coloring mode.
    focusClass?: string | null;
  }

  let {
    data,
    allData,
    colorBy,
    selectedIds,
    selectionTool,
    onSelect,
    onHover,
    onSelectionToolChange,
    taskType,
    classNames,
    classColors,
    focusClass = null,
  }: Props = $props();

  type ColorEncoding = {
    z: Float32Array;
    palette: string[];
    zDataType: "categorical" | "continuous";
  };

  // ---- Pure helpers, copied verbatim from the React component (spec §8.4) ----

  const UNLABELED_COLOR = "#9CA3AF";
  const GRADIENT_BINS = 64;

  // Build a gradient palette by sampling a per-fraction color function.
  function makeGradient(fn: (t: number) => string, bins = GRADIENT_BINS): string[] {
    const out = new Array<string>(bins);
    for (let i = 0; i < bins; i++) out[i] = fn(i / (bins - 1));
    return out;
  }

  // Sequential ramp biased so uncertain points pop:
  // near-invisible gray for confident samples, company red for uncertain ones.
  // E3000F is the IITS company red (DEFAULT_PALETTE[0] in classManager).
  const UNCERTAINTY_STOPS = ["#F3F4F6", "#E5E7EB", "#F4A4AA", "#E3000F"];

  function mixHex(a: string, b: string, t: number): string {
    const ar = parseInt(a.slice(1, 3), 16), ag = parseInt(a.slice(3, 5), 16), ab = parseInt(a.slice(5, 7), 16);
    const br = parseInt(b.slice(1, 3), 16), bg = parseInt(b.slice(3, 5), 16), bb = parseInt(b.slice(5, 7), 16);
    const r = Math.round(ar + (br - ar) * t);
    const g = Math.round(ag + (bg - ag) * t);
    const bl = Math.round(ab + (bb - ab) * t);
    const toHex = (n: number) => n.toString(16).padStart(2, "0");
    return `#${toHex(r)}${toHex(g)}${toHex(bl)}`;
  }

  function uncertaintyColor(t: number): string {
    const x = t < 0 ? 0 : t > 1 ? 1 : t;
    const seg = x * (UNCERTAINTY_STOPS.length - 1);
    const i = Math.floor(seg);
    if (i >= UNCERTAINTY_STOPS.length - 1) return UNCERTAINTY_STOPS[UNCERTAINTY_STOPS.length - 1];
    return mixHex(UNCERTAINTY_STOPS[i], UNCERTAINTY_STOPS[i + 1], seg - i);
  }

  function hslToHex(h: number, s: number, l: number): string {
    s /= 100;
    l /= 100;
    const k = (n: number) => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n: number) =>
      Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))));
    const toHex = (n: number) => n.toString(16).padStart(2, "0");
    return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`;
  }

  function annotationCountColor(t: number): string {
    return hslToHex(240 - t * 240, 80, 55); // blue → red
  }

  // Compute z values + color palette for the given coloring mode.
  function buildColorEncoding(
    points: DataPoint[],
    mode: ColorBy,
    names: string[],
    colors: Record<string, string>,
    task: TaskType,
    focus: string | null,
  ): ColorEncoding {
    const n = points.length;
    const z = new Float32Array(n);

    if (mode === "uncertainty") {
      for (let i = 0; i < n; i++) {
        const u = points[i].uncertainty;
        z[i] = u < 0 ? 0 : u > 1 ? 1 : u;
      }
      return { z, palette: makeGradient(uncertaintyColor), zDataType: "continuous" };
    }

    if (mode === "split") {
      // Categorical: 0=other, 1=train, 2=valid, 3=test
      const palette = [UNLABELED_COLOR, "#3B82F6", "#F59E0B", "#10B981"];
      for (let i = 0; i < n; i++) {
        const s = effectiveSplit(points[i]);
        z[i] = s === "train" ? 1 : s === "valid" ? 2 : s === "test" ? 3 : 0;
      }
      return { z, palette, zDataType: "categorical" };
    }

    if (mode === "annotationCount") {
      // Special-case 0 = gray; positive counts → gradient.
      // We use a categorical encoding with bins so 0 stays distinct.
      const palette: string[] = [UNLABELED_COLOR, ...makeGradient(annotationCountColor, 16)];
      const isMultilabel = task === "multilabel-classification";
      for (let i = 0; i < n; i++) {
        const c = (isMultilabel ? points[i].labels.length : points[i].annotationCount) ?? 0;
        if (c === 0) z[i] = 0;
        else {
          const t = Math.min(c / 10, 1);
          z[i] = 1 + Math.round(t * 15); // 1..16
        }
      }
      return { z, palette, zDataType: "categorical" };
    }

    if (mode === "hasClass") {
      // 0 = doesn't carry the focus class (gray), 1 = carries it (class color).
      // With no focus class picked yet every point stays gray.
      const palette: string[] = [UNLABELED_COLOR, colors[focus] || "#E3000F"];
      for (let i = 0; i < n; i++) {
        z[i] = focus && points[i].labels.includes(focus) ? 1 : 0;
      }
      return { z, palette, zDataType: "categorical" };
    }

    if (mode === "mistake") {
      // 0 = unlabeled / no prediction (gray), 1 = correct prediction (green),
      // 2 = wrong prediction (red).
      const palette: string[] = [UNLABELED_COLOR, "#10B981", "#E3000F"];
      for (let i = 0; i < n; i++) {
        const p = points[i];
        if (p.isWrong) z[i] = 2;
        else if (!p.isUnlabeled && p.predictedLabel) z[i] = 1;
        else z[i] = 0;
      }
      return { z, palette, zDataType: "categorical" };
    }

    // Categorical: groundTruth | predictions
    // Palette is [Unlabeled, ...classNames]; z is the index into that palette.
    const palette: string[] = [UNLABELED_COLOR];
    const labelIndex = new Map<string, number>();
    labelIndex.set("Unlabeled", 0);
    for (const name of names) {
      labelIndex.set(name, palette.length);
      palette.push(colors[name] || UNLABELED_COLOR);
    }

    for (let i = 0; i < n; i++) {
      const p = points[i];
      const key = mode === "groundTruth"
        ? p.label || p.groundTruth || "Unlabeled"
        : p.predictedLabel || "Unlabeled";
      z[i] = labelIndex.get(key) ?? 0;
    }
    return { z, palette, zDataType: "categorical" };
  }

  // Compute normalized [-1, 1] coordinates from raw x,y, padded slightly so
  // points don't sit on the very edge. Points at exactly (0,0) are the
  // "no projection yet" sentinel and are skipped for the bbox.
  function buildCoords(points: DataPoint[]): { x: Float32Array; y: Float32Array; hasCoords: boolean } {
    const n = points.length;
    const x = new Float32Array(n);
    const y = new Float32Array(n);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    let any = false;
    for (let i = 0; i < n; i++) {
      const px = points[i].x;
      const py = points[i].y;
      if (px === 0 && py === 0) continue;
      any = true;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }
    if (!any) return { x, y, hasCoords: false };

    const rangeX = (maxX - minX) || 1;
    const rangeY = (maxY - minY) || 1;
    const cX = (minX + maxX) / 2;
    const cY = (minY + maxY) / 2;
    const padding = 0.05;
    const sX = (2 * (1 - padding)) / rangeX;
    const sY = (2 * (1 - padding)) / rangeY;

    for (let i = 0; i < n; i++) {
      x[i] = (points[i].x - cX) * sX;
      y[i] = (points[i].y - cY) * sY;
    }
    return { x, y, hasCoords: true };
  }

  // ---- Component state ----

  let container: HTMLDivElement;
  let canvas: HTMLCanvasElement;
  // One regl-scatterplot instance for the component's lifetime. $state.raw so
  // the update effects re-run once it exists, without proxying the instance.
  let sp = $state.raw<ReturnType<typeof createScatterplot> | null>(null);
  let tooltip = $state.raw<{ x: number; y: number; point: DataPoint } | null>(null);
  // Last tracked mouse position — deliberately non-reactive (spec §8.4.6).
  let mousePos = { x: 0, y: 0 };
  // Serialize draw calls — regl-scatterplot silently drops overlapping draw
  // calls, so the draw/filter/selection effects share this promise queue.
  let drawChain: Promise<unknown> = Promise.resolve();

  // Stable id-array + id→index map, built from allData (never data).
  const idIndex = $derived.by(() => {
    const ids = new Array<string>(allData.length);
    const map = new Map<string, number>();
    for (let i = 0; i < allData.length; i++) {
      ids[i] = allData[i].id;
      map.set(allData[i].id, i);
    }
    return { ids, map };
  });

  const colorEncoding = $derived(
    buildColorEncoding(allData, colorBy, classNames, classColors, taskType, focusClass),
  );
  const coords = $derived(buildCoords(allData));

  // Initialize the scatterplot once; destroy it on unmount. All later updates
  // are imperative sp.set/draw/filter/select calls from effects — never recreate.
  onMount(() => {
    const { width, height } = container.getBoundingClientRect();

    const scatter = createScatterplot({
      canvas,
      width: Math.max(1, Math.floor(width)),
      height: Math.max(1, Math.floor(height)),
      pointSize: 4,
      pointSizeSelected: 2,
      pointOutlineWidth: 2,
      opacity: 0.6,
      backgroundColor: [1, 1, 1, 1],
      lassoMinDelay: 8,
      lassoMinDist: 2,
      deselectOnDblClick: true,
      deselectOnEscape: true,
      colorBy: "valueA",
    });

    scatter.subscribe("select", ({ points }: { points: number[] }) => {
      const ids = new Set<string>();
      const idx2id = idIndex.ids;
      for (const idx of points) {
        const id = idx2id[idx];
        if (id !== undefined) ids.add(id);
      }
      onSelect(ids);
    });
    scatter.subscribe("deselect", () => {
      onSelect(new Set());
    });
    scatter.subscribe("pointOver", (idx: number) => {
      const id = idIndex.ids[idx];
      if (id === undefined) return;
      onHover(id);
      const point = allData[idx];
      if (point) {
        tooltip = { x: mousePos.x, y: mousePos.y, point };
      }
    });
    scatter.subscribe("pointOut", () => {
      onHover(null);
      tooltip = null;
    });

    sp = scatter;
    return () => {
      scatter.destroy();
      sp = null;
    };
  });

  // Resize observer → keep the WebGL viewport in sync with the container.
  $effect(() => {
    const scatter = sp;
    if (!scatter || !container) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      scatter.set({
        width: Math.max(1, Math.floor(width)),
        height: Math.max(1, Math.floor(height)),
      });
    });
    observer.observe(container);
    return () => observer.disconnect();
  });

  // Draw whenever coords / encoding change.
  $effect(() => {
    const scatter = sp;
    if (!scatter) return;
    const enc = colorEncoding;
    const c = coords;
    if (!c.hasCoords || allData.length === 0) {
      drawChain = drawChain.then(() => scatter.clear()).catch(() => {});
      return;
    }

    drawChain = drawChain
      .then(() => {
        scatter.set({ pointColor: enc.palette });
        return scatter.draw(
          { x: c.x, y: c.y, z: enc.z },
          { zDataType: enc.zDataType, preventFilterReset: true },
        );
      })
      .catch(() => {});
  });

  // Apply visibility filter when `data` (filtered subset) differs from allData.
  $effect(() => {
    const scatter = sp;
    if (!scatter || allData.length === 0) return;
    const subset = data;
    const all = allData;
    const map = idIndex.map;
    drawChain = drawChain
      .then(() => {
        if (subset.length === all.length) {
          return scatter.unfilter({ preventEvent: true });
        }
        const visible: number[] = new Array(subset.length);
        let k = 0;
        for (let i = 0; i < subset.length; i++) {
          const idx = map.get(subset[i].id);
          if (idx !== undefined) visible[k++] = idx;
        }
        visible.length = k;
        return scatter.filter(visible, { preventEvent: true });
      })
      .catch(() => {});
  });

  // Selection sync (driven by Explorer). Tracks selectedIds only — the map is
  // read inside the async chain, mirroring React's narrowed [selectedIds] deps.
  $effect(() => {
    const scatter = sp;
    if (!scatter) return;
    const sel = selectedIds;
    drawChain = drawChain
      .then(() => {
        const map = idIndex.map;
        if (sel.size === 0) {
          scatter.deselect({ preventEvent: true });
          return;
        }
        const indices: number[] = [];
        for (const id of sel) {
          const i = map.get(id);
          if (i !== undefined) indices.push(i);
        }
        if (indices.length === 0) scatter.deselect({ preventEvent: true });
        else scatter.select(indices, { preventEvent: true });
      })
      .catch(() => {});
  });

  // Selection tool sync + middle-mouse pan override.
  // regl-scatterplot's mouseMode is exclusive (panZoom XOR lasso) and both
  // panZoom (via dom-2d-camera) and lasso gate strictly on event.buttons===1
  // (left button). To allow middle-button pan in lasso/box mode without
  // dropping the active tool, we intercept the real middle-button mousedown
  // in the capture phase, switch to panZoom, and dispatch a synthetic
  // left-button mousedown so both layers begin tracking the drag.
  $effect(() => {
    const scatter = sp;
    const cv = canvas;
    if (!scatter || !cv) return;
    const tool = selectionTool;

    const applyMode = () => {
      if (tool === "click") scatter.set({ mouseMode: "panZoom" });
      // lassoType is supported at runtime but missing from 1.16.0's Settable type.
      else if (tool === "lasso") scatter.set({ mouseMode: "lasso", lassoType: "freeform" } as never);
      else scatter.set({ mouseMode: "lasso", lassoType: "rectangle" } as never);
    };
    applyMode();

    let middleHeld = false;

    const onDown = (e: MouseEvent) => {
      if (e.button !== 1) return;
      // Prevent browser middle-click autoscroll AND stop the native middle
      // event from reaching dom-2d-camera (which would clobber its
      // isLeftMousePressed flag we're about to set via the synthetic event).
      e.preventDefault();
      e.stopPropagation();
      middleHeld = true;
      if (tool !== "click") scatter.set({ mouseMode: "panZoom" });
      cv.dispatchEvent(new MouseEvent("mousedown", {
        bubbles: true, cancelable: true, view: window,
        button: 0, buttons: 1, clientX: e.clientX, clientY: e.clientY,
      }));
    };

    const onUp = (e: MouseEvent) => {
      if (e.button !== 1 || !middleHeld) return;
      middleHeld = false;
      if (tool !== "click") applyMode();
    };

    const onAuxClick = (e: MouseEvent) => { if (e.button === 1) e.preventDefault(); };

    cv.addEventListener("mousedown", onDown, true);
    window.addEventListener("mouseup", onUp);
    cv.addEventListener("auxclick", onAuxClick);

    return () => {
      cv.removeEventListener("mousedown", onDown, true);
      window.removeEventListener("mouseup", onUp);
      cv.removeEventListener("auxclick", onAuxClick);
    };
  });

  function handleCanvasMouseMove(e: MouseEvent) {
    const rect = canvas?.getBoundingClientRect();
    if (!rect) return;
    mousePos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    if (tooltip) {
      tooltip = { ...tooltip, x: mousePos.x, y: mousePos.y };
    }
  }

  function handleCanvasMouseLeave() {
    tooltip = null;
    onHover(null);
  }

  function zoomBy(factor: number) {
    const scatter = sp;
    if (!scatter) return;
    const camera = scatter.get("camera");
    const target = camera.target ? [camera.target[0], camera.target[1]] : [0, 0];
    const distance = camera.distance?.[0] ?? 1;
    scatter.zoomToLocation(target, distance * factor, { transition: true, transitionDuration: 200 });
  }

  function zoomReset() {
    sp?.zoomToOrigin({ transition: true, transitionDuration: 250 });
  }

  const TOOLS = [
    { tool: "click", icon: MousePointer2, label: "Click" },
    { tool: "box", icon: Square, label: "Box" },
    { tool: "lasso", icon: Crosshair, label: "Lasso" },
  ] as const;
</script>

<div class="flex-1 flex flex-col min-h-0">
  <!-- Toolbar -->
  <div class="h-9 border-b border-border flex items-center px-3 gap-1 shrink-0">
    <span class="text-xs text-muted-foreground mr-2">Select:</span>
    {#each TOOLS as { tool, icon: Icon, label } (tool)}
      <Tooltip>
        <TooltipTrigger>
          {#snippet child({ props })}
            <Button
              {...props}
              variant={selectionTool === tool ? "default" : "ghost"}
              size="sm"
              class="h-6 w-6 p-0"
              onclick={() => onSelectionToolChange(tool)}
              aria-label={`${label} select`}
            >
              <Icon class="h-3.5 w-3.5" />
            </Button>
          {/snippet}
        </TooltipTrigger>
        <TooltipContent side="bottom">{label}</TooltipContent>
      </Tooltip>
    {/each}

    <div class="mx-2 h-4 w-px bg-border"></div>

    <Tooltip>
      <TooltipTrigger>
        {#snippet child({ props })}
          <Button {...props} variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={() => zoomBy(0.8)} aria-label="Zoom in">
            <ZoomIn class="h-3.5 w-3.5" />
          </Button>
        {/snippet}
      </TooltipTrigger>
      <TooltipContent side="bottom">Zoom in</TooltipContent>
    </Tooltip>
    <Tooltip>
      <TooltipTrigger>
        {#snippet child({ props })}
          <Button {...props} variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={() => zoomBy(1.25)} aria-label="Zoom out">
            <ZoomOut class="h-3.5 w-3.5" />
          </Button>
        {/snippet}
      </TooltipTrigger>
      <TooltipContent side="bottom">Zoom out</TooltipContent>
    </Tooltip>
    <Tooltip>
      <TooltipTrigger>
        {#snippet child({ props })}
          <Button {...props} variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={zoomReset} aria-label="Reset zoom">
            <Maximize2 class="h-3.5 w-3.5" />
          </Button>
        {/snippet}
      </TooltipTrigger>
      <TooltipContent side="bottom">Reset zoom</TooltipContent>
    </Tooltip>

    <div class="flex-1"></div>

    <span class="text-xs text-muted-foreground tabular-nums">{data.length} points</span>
  </div>

  <!-- Canvas -->
  <div bind:this={container} class="flex-1 relative min-h-0">
    <canvas
      bind:this={canvas}
      class="absolute inset-0 cursor-crosshair"
      onmousemove={handleCanvasMouseMove}
      onmouseleave={handleCanvasMouseLeave}
    ></canvas>

    <!-- Tooltip -->
    {#if tooltip}
      <div
        class="absolute z-10 pointer-events-none bg-background border border-border rounded-md shadow-md p-2 flex gap-2 items-start"
        style="left: {tooltip.x + 16}px; top: {tooltip.y - 8}px; max-width: 280px"
      >
        <img
          src={tooltip.point.thumbnailUrl}
          alt=""
          class="w-12 h-12 rounded-sm object-cover bg-muted shrink-0"
        />
        <div class="min-w-0 text-xs space-y-0.5">
          <p class="font-medium text-foreground truncate">{tooltip.point.filename}</p>
          {#if taskType === "object-detection"}
            <p class="text-muted-foreground">
              {tooltip.point.annotationCount > 0
                ? Object.entries(tooltip.point.annotationClassCounts)
                    .map(([cls, n]) => `${n}× ${cls}`)
                    .join(", ")
                : "No annotations"}
            </p>
          {:else if taskType === "multilabel-classification"}
            <p class="text-muted-foreground">
              {tooltip.point.labels.join(", ") || "Untagged"}{#if tooltip.point.predictedLabels.length > 0}<span>
                  • Pred: {tooltip.point.predictedLabels.join(", ")}</span
                >{/if}
            </p>
          {:else}
            <p class="text-muted-foreground">
              {tooltip.point.groundTruth}{#if tooltip.point.label}<span class="ml-1">→ {tooltip.point.label}</span
                >{/if}{#if tooltip.point.predictedLabel}<span>
                  • Pred: {tooltip.point.predictedLabel} ({(tooltip.point.confidence * 100).toFixed(0)}%)</span
                >{/if}
            </p>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>
