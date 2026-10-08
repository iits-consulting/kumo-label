<script lang="ts">
  import { untrack } from "svelte";
  import { createQuery, useQueryClient, keepPreviousData } from "@tanstack/svelte-query";
  import { type DataPoint, type TaskType, type TopPrediction, isLabeledPoint } from "@/lib/mockData";
  import type { TrainingRun, ExplorerFilters, ReductionMethod, ViewMode, ActionBarSection } from "@/lib/types";
  import type { SelectionTool } from "@/lib/taskConfig";
  import { type ALWeights, DEFAULT_AL_WEIGHTS, computeALScore } from "@/lib/alScore";
  import { TASK_CONFIGS } from "@/lib/taskConfig";
  import { api } from "@/lib/api";
  import { parseProjectionBlob } from "@/lib/projectionBlob";
  import { createClassManager, DEFAULT_PALETTE } from "@/state/classManager.svelte";
  import { createShortcutSettings } from "@/state/shortcutSettings.svelte";
  import { createJobPoller } from "@/state/jobPoller";
  import ExplorerSidebar from "@/components/explorer/ExplorerSidebar.svelte";
  import ActionBar from "@/components/explorer/ActionBar.svelte";
  import EmbeddingsPanel from "@/components/explorer/EmbeddingsPanel.svelte";
  import ReductionPanel from "@/components/explorer/ReductionPanel.svelte";
  import ActiveLearningPanel from "@/components/explorer/ActiveLearningPanel.svelte";
  import ScatterPlot from "@/components/explorer/ScatterPlot.svelte";
  import ImageTable from "@/components/explorer/ImageTable.svelte";
  import InspectorPanel from "@/components/explorer/InspectorPanel.svelte";
  import StatusBar from "@/components/explorer/StatusBar.svelte";
  import TrainingDrawer from "@/components/explorer/TrainingDrawer.svelte";
  import AnnotatorOverlay from "@/components/annotator/AnnotatorOverlay.svelte";
  import ShortcutSettingsDialog from "@/components/ShortcutSettingsDialog.svelte";
  import { Button } from "@/components/ui/button";
  import PanelLeftClose from "@lucide/svelte/icons/panel-left-close";
  import PanelLeftOpen from "@lucide/svelte/icons/panel-left-open";
  import PanelRightClose from "@lucide/svelte/icons/panel-right-close";
  import PanelRightOpen from "@lucide/svelte/icons/panel-right-open";
  import ChartScatter from "@lucide/svelte/icons/chart-scatter";
  import TableIcon from "@lucide/svelte/icons/table";
  import LoaderCircle from "@lucide/svelte/icons/loader-circle";
  import { toast } from "@/state/toast";
  import { route, navigate } from "@/router.svelte";
  import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

  interface ImageRecord {
    id: number;
    filename: string;
    path: string;
    class: string;
    split: string;
    split_override: string | null;
    label: string | null;
    annotation: string | null;
    ignored: number;
    no_objects?: number;
    annotation_count: number;
    annotation_classes: string | null;
    // Comma-joined multilabel tags (GROUP_CONCAT over image_labels)
    label_classes: string | null;
  }

  interface ImagesResponse {
    images: ImageRecord[];
    total: number;
    page: number;
    limit: number;
  }

  interface DatasetState {
    taskType: TaskType;
    datasetPath: string;
    dbPath: string;
    classes: string[];
    splits: string[];
    counts: { total: number; by_split: Record<string, number> };
    embeddingModel?: "dinov2" | "clip";
    activeEmbeddingJobId?: string | null;
    embeddingBatchSize?: number;
  }

  interface PredictionRecord {
    image_id: number;
    predicted_class: string;
    confidence: number;
    uncertainty: number;
    al_score: number;
    top_predictions: TopPrediction[];
    is_unlabeled: boolean;
    is_wrong: boolean;
    class_imbalance_score: number;
  }

  type SplitName = "train" | "valid" | "test";

  // toDataPoint stays a pure function. Keep exactly: Map.has() semantics so an
  // explicit null session label does NOT fall back to r.annotation (checklist 4);
  // (0,0) = "no projection yet" sentinel; imageUrl === thumbnailUrl (checklist 25).
  function toDataPoint(
    r: ImageRecord,
    labels: Map<string, string | null>,
    tags: Map<string, string[]>,
    ignoredSet: Set<string>,
    splitOverrides: Map<string, SplitName | null>,
    coords: Map<string, { x: number; y: number }>,
    prediction?: PredictionRecord,
    weights?: ALWeights,
  ): DataPoint {
    const coord = coords.get(String(r.id));
    const isUnlabeled = prediction?.is_unlabeled ?? false;
    const isWrong = prediction?.is_wrong ?? false;
    const classImbalanceScore = prediction?.class_imbalance_score ?? 0;
    const confidence = prediction?.confidence ?? 0;
    const uncertainty = prediction?.uncertainty ?? 0;
    const alScore = prediction && weights
      ? computeALScore({ uncertainty, isUnlabeled, isWrong, confidence, classImbalanceScore }, weights)
      : prediction?.al_score ?? 0;
    return {
      id: String(r.id),
      filename: r.filename,
      x: coord?.x ?? 0,
      y: coord?.y ?? 0,
      groundTruth: r.class,
      label: labels.has(String(r.id)) ? labels.get(String(r.id))! : (r.annotation ?? null),
      predictedLabel: prediction?.predicted_class ?? null,
      confidence,
      uncertainty,
      alScore,
      isUnlabeled,
      isWrong,
      classImbalanceScore,
      topPredictions: prediction?.top_predictions ?? [],
      width: 0,
      height: 0,
      imageUrl: `/api/datasets/image?file_path=${encodeURIComponent(r.path)}`,
      thumbnailUrl: `/api/datasets/image?file_path=${encodeURIComponent(r.path)}`,
      cluster: 0,
      ignored: ignoredSet.has(String(r.id)) || Boolean(r.ignored),
      noObjects: Boolean(r.no_objects),
      annotationCount: r.annotation_count ?? 0,
      annotationClasses: r.annotation_classes ? [...new Set(r.annotation_classes.split(","))] : [],
      annotationClassCounts: r.annotation_classes
        ? r.annotation_classes.split(",").reduce<Record<string, number>>((acc, cls) => {
            acc[cls] = (acc[cls] || 0) + 1;
            return acc;
          }, {})
        : {},
      labels: tags.get(String(r.id)) ?? (r.label_classes ? [...new Set(r.label_classes.split(","))] : []),
      predictedLabels: (prediction?.predicted_class ?? "").split(",").filter(Boolean),
      split: r.split,
      splitOverride: splitOverrides.has(String(r.id))
        ? splitOverrides.get(String(r.id))!
        : r.split_override === "train" || r.split_override === "valid" || r.split_override === "test"
          ? r.split_override
          : null,
    };
  }

  const queryClient = useQueryClient();

  // Cross-page dataset state — read ONLY from sessionStorage (spec §7).
  function readDatasetState(): DatasetState | null {
    try {
      const stored = sessionStorage.getItem("kumo:datasetState");
      return stored ? (JSON.parse(stored) as DatasetState) : null;
    } catch {
      return null;
    }
  }
  const datasetState = readDatasetState();
  const dbPath = datasetState?.dbPath ?? "";
  const taskType: TaskType = datasetState?.taskType || "classification";

  const isAnnotating = $derived(route.pathname.includes("/annotate"));

  // Effect 1: mount guard — redirect to setup if no dataset state (e.g. hard refresh).
  $effect(() => {
    if (!dbPath) navigate("/", { replace: true });
  });

  // ---- Non-reactive session state (plain Map/Set/let, NOT $state — spec §8.3).
  // These exist so toDataPoint (re-run on every query refetch) sees optimistic
  // edits without themselves triggering re-derivation. After mutating one of
  // them, handlers explicitly rebuild the affected rows via patchPoints.
  const sessionLabels = new Map<string, string | null>();
  const sessionTags = new Map<string, string[]>();
  const sessionIgnored = new Set<string>();
  const sessionSplitOverrides = new Map<string, SplitName | null>();
  // Persists projection coords across page changes so they survive imageData refetches.
  const projectionCoords = new Map<string, { x: number; y: number }>();
  let seededDbPath: string | null = null;
  let projectionTriggered = false;

  // Class management — seeded from dataset classes
  const initialClasses = datasetState?.classes?.map((name, i) => ({
    name,
    color: DEFAULT_PALETTE[i % DEFAULT_PALETTE.length],
  }));
  const cm = createClassManager(initialClasses);

  // Shortcut management
  const ss = createShortcutSettings(() => cm.classNames, datasetState?.dbPath);

  // Effect 2: sync shortcuts when classes change (React deps: [classNames]).
  $effect(() => {
    const names = cm.classNames;
    untrack(() => ss.syncClasses(names));
  });

  // Effect 3: mirror the live class list back into the sessionStorage snapshot so
  // that remounting Explorer (refresh, navigation) reseeds the class manager from
  // the current truth instead of resurrecting removed classes.
  $effect(() => {
    const names = cm.classNames;
    try {
      const raw = sessionStorage.getItem("kumo:datasetState");
      if (!raw) return;
      const parsed = JSON.parse(raw) as DatasetState;
      const same =
        Array.isArray(parsed.classes) &&
        parsed.classes.length === names.length &&
        parsed.classes.every((c, i) => c === names[i]);
      if (same) return;
      sessionStorage.setItem("kumo:datasetState", JSON.stringify({ ...parsed, classes: names }));
    } catch {
      // sessionStorage parse/write failures are non-fatal
    }
  });

  // ---- Reactive state ($state — spec §8.3).
  let embeddingModel = $state<"dinov2" | "clip">(datasetState?.embeddingModel ?? "dinov2");
  let autoTriggerJobId = $state<string | null>(null);
  let autoTriggerParamHash = $state<string | null>(null);
  let shortcutDialogOpen = $state(false);
  let labelingBudget = $state(50);
  let labelingStrategy = $state<"composite" | "entropy">("composite");
  let annotatorUseSelection = $state(false);
  let page = $state(1);
  let pageSize = $state(50);
  let sortBy = $state<"confidence" | "uncertainty" | "al_score" | null>(null);
  let sortOrder = $state<"asc" | "desc">("desc");
  let selectedRunId = $state<string | null>(null);
  let leftOpen = $state(true);
  let rightOpen = $state(true);
  let trainingOpen = $state(false);
  let selectedIds = $state<Set<string>>(new Set());
  let alWeights = $state<ALWeights>(DEFAULT_AL_WEIGHTS);
  let activeSection = $state<ActionBarSection>(null);
  let embeddingIsComputing = $state(false);
  let reductionStatus = $state<{ method: ReductionMethod; isLoading: boolean }>({ method: "umap", isLoading: false });
  let dataPoints = $state<DataPoint[]>([]);
  let allDataPoints = $state<DataPoint[]>([]);
  let viewModeOverride = $state<ViewMode | null>(null);
  let hoveredId = $state<string | null>(null);
  let inspectedId = $state<string | null>(null);
  let selectionTool = $state<SelectionTool>("click");
  let labeledThisSession = $state(0);
  let filters = $state<ExplorerFilters>({
    colorBy: TASK_CONFIGS[taskType].defaultColorBy,
    filterLabeled: "all",
    selectedClasses: [],
    confidenceRange: [0, 1],
    reductionMethod: "umap",
    perplexity: 30,
    focusClass: null,
  });

  // Children pass a full replacement object; assign property-wise so unchanged
  // fields don't invalidate narrow-read derivations like filteredData.
  function setFilters(f: ExplorerFilters) {
    Object.assign(filters, f);
  }

  // ---- Queries (svelte-query, keys unchanged from React).
  // Paginated query — drives the table view only.
  const imagesQuery = createQuery<ImagesResponse>(() => ({
    queryKey: ["images", dbPath, page, pageSize, sortBy, sortOrder, selectedRunId],
    queryFn: () => {
      let url = `/datasets/images?db_path=${encodeURIComponent(dbPath)}&page=${page}&limit=${pageSize}`;
      if (sortBy && selectedRunId) {
        url += `&sort_by=${sortBy}&sort_order=${sortOrder}&run_id=${selectedRunId}`;
      }
      return api.get<ImagesResponse>(url);
    },
    enabled: !!dbPath,
    placeholderData: keepPreviousData,
  }));

  // All-images query — drives the scatter plot (no effective pagination limit).
  const allImagesQuery = createQuery<ImagesResponse>(() => ({
    queryKey: ["images-all", dbPath],
    queryFn: () =>
      api.get<ImagesResponse>(`/datasets/images?db_path=${encodeURIComponent(dbPath)}&page=1&limit=250000`),
    enabled: !!dbPath,
  }));

  const embeddingStatusQuery = createQuery<Record<string, { exists: boolean; count?: number; created_at?: string }>>(() => ({
    queryKey: ["embeddings-status", dbPath],
    queryFn: () =>
      api.get<Record<string, { exists: boolean; count?: number; created_at?: string }>>(
        `/embeddings/status?db_path=${encodeURIComponent(dbPath)}`,
      ),
    enabled: !!dbPath,
  }));

  const embeddingStatus = $derived({
    exists: !!embeddingStatusQuery.data?.[embeddingModel]?.exists,
    count: embeddingStatusQuery.data?.[embeddingModel]?.count ?? 0,
    isComputing: embeddingIsComputing,
  });

  // Predictions for the selected training run
  const predictionsQuery = createQuery<PredictionRecord[]>(() => ({
    queryKey: ["predictions", selectedRunId, dbPath],
    queryFn: () =>
      api.get<PredictionRecord[]>(
        `/training/runs/${selectedRunId}/predictions?db_path=${encodeURIComponent(dbPath)}`,
      ),
    enabled: !!selectedRunId && !!dbPath,
  }));

  // Training history — shared cache with StatusBar/TrainingDrawer.
  const trainingRunsQuery = createQuery<TrainingRun[]>(() => ({
    queryKey: ["training-runs", dbPath, taskType],
    queryFn: () =>
      api.get<TrainingRun[]>(
        `/training/runs?db_path=${encodeURIComponent(dbPath)}&task_type=${encodeURIComponent(taskType)}`,
      ),
    enabled: !!dbPath,
  }));
  const trainingHistory = $derived(trainingRunsQuery.data ?? []);

  // ---- Derived.
  const predictionsMap = $derived.by(() => {
    const preds = predictionsQuery.data;
    if (!preds) return new Map<string, PredictionRecord>();
    return new Map(preds.map((p) => [String(p.image_id), p]));
  });

  // AL scores derived from predictions + weights. Held in a separate Map so
  // dragging the AL weight sliders doesn't rebuild the 250k-element data arrays.
  const alScoreMap = $derived.by(() => {
    const m = new Map<string, number>();
    const preds = predictionsQuery.data;
    if (!preds) return m;
    for (const p of preds) {
      m.set(
        String(p.image_id),
        computeALScore(
          {
            uncertainty: p.uncertainty,
            isUnlabeled: p.is_unlabeled,
            isWrong: p.is_wrong,
            confidence: p.confidence,
            classImbalanceScore: p.class_imbalance_score,
          },
          alWeights,
        ),
      );
    }
    return m;
  });

  // Current-page data points — for table view. Rebuilt on query/prediction change;
  // session maps are read non-reactively (they're plain Maps).
  $effect(() => {
    const imgs = imagesQuery.data?.images;
    const pm = predictionsMap;
    if (!imgs) return;
    dataPoints = imgs.map((r) =>
      toDataPoint(r, sessionLabels, sessionTags, sessionIgnored, sessionSplitOverrides, projectionCoords, pm.get(String(r.id))),
    );
  });

  // All data points — for scatter plot, inspector, stats.
  $effect(() => {
    const imgs = allImagesQuery.data?.images;
    const pm = predictionsMap;
    if (!imgs) return;
    allDataPoints = imgs.map((r) =>
      toDataPoint(r, sessionLabels, sessionTags, sessionIgnored, sessionSplitOverrides, projectionCoords, pm.get(String(r.id))),
    );
  });

  // Effect 4: recover classes from DB annotations (e.g. after page refresh).
  $effect(() => {
    const imgs = allImagesQuery.data?.images;
    if (!imgs) return;
    untrack(() => {
      imgs.forEach((r) => {
        // Classification: single label per image
        if (r.annotation) cm.addClass(r.annotation);
        // Object detection: classes live in the bounding-box annotations
        if (r.annotation_classes) {
          r.annotation_classes.split(",").forEach((c) => {
            const trimmed = c.trim();
            if (trimmed) cm.addClass(trimmed);
          });
        }
        // Multi-label: classes live in the per-image tag set
        if (r.label_classes) {
          r.label_classes.split(",").forEach((c) => {
            const trimmed = c.trim();
            if (trimmed) cm.addClass(trimmed);
          });
        }
      });
    });
  });

  // Effect 5: multi-label seed — once per dbPath (guard var). Idempotent
  // server-side; invalidate afterwards so the first paint shows the seeded tags.
  $effect(() => {
    if (!dbPath || taskType !== "multilabel-classification") return;
    if (seededDbPath === dbPath) return;
    api
      .post<{ seeded: number }>("/datasets/multilabel/seed", { db_path: dbPath })
      .then(() => {
        seededDbPath = dbPath;
        queryClient.invalidateQueries({ queryKey: ["images", dbPath] });
        queryClient.invalidateQueries({ queryKey: ["images-all", dbPath] });
      })
      .catch(() => toast({ title: "Failed to seed labels from folders", variant: "destructive" }));
  });

  // Effect 6: auto-select latest trained model — only when none selected (checklist 8).
  $effect(() => {
    if (selectedRunId) return;
    const doneRuns = trainingHistory
      .filter((r) => r.status === "done")
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    if (doneRuns.length > 0) {
      selectedRunId = doneRuns[0].id;
    }
  });

  // Detect if embeddings exist (any point has non-zero coords)
  const hasEmbeddings = $derived(allDataPoints.some((d) => d.x !== 0 || d.y !== 0));

  // View mode: auto-detect based on embeddings, allow override
  const viewMode: ViewMode = $derived(viewModeOverride ?? (hasEmbeddings ? "scatter" : "table"));

  // Filtered data for scatter (all images).
  // ⚠️ Narrow reads: only filterLabeled/selectedClasses/confidenceRange from
  // `filters` (plus allDataPoints, taskType) — colorBy/perplexity changes must
  // not rebuild a potentially 250k-row filter pass (spec §8.3).
  const filteredData = $derived.by(() => {
    const filterLabeled = filters.filterLabeled;
    const selectedClasses = filters.selectedClasses;
    const confidenceRange = filters.confidenceRange;
    return allDataPoints.filter((d) => {
      if (filterLabeled === "ignored") return d.ignored;
      if (d.ignored) return false; // hide ignored by default
      const isLabeled = isLabeledPoint(d, taskType);
      if (filterLabeled === "labeled" && !isLabeled) return false;
      if (filterLabeled === "unlabeled" && isLabeled) return false;
      if (selectedClasses.length > 0) {
        if (taskType === "object-detection") {
          if (!d.annotationClasses.some((c) => selectedClasses.includes(c))) return false;
        } else if (taskType === "multilabel-classification") {
          if (!d.labels.some((l) => selectedClasses.includes(l))) return false;
        } else {
          const cls = d.label || d.groundTruth || "Unlabeled";
          if (!selectedClasses.includes(cls)) return false;
        }
      }
      if (d.confidence < confidenceRange[0] || d.confidence > confidenceRange[1]) return false;
      return true;
    });
  });

  // When client-side filters are active, paginate filteredData locally instead
  // of filtering the server-paginated page (which may not contain the target class).
  const hasClientFilters = $derived(
    selectedIds.size > 0 ||
      filters.selectedClasses.length > 0 ||
      filters.filterLabeled !== "all" ||
      filters.confidenceRange[0] > 0 ||
      filters.confidenceRange[1] < 1,
  );

  const filteredTableSource = $derived.by(() => {
    if (selectedIds.size > 0) {
      return filteredData.filter((d) => selectedIds.has(d.id));
    }
    return filteredData;
  });

  const filteredTableData = $derived.by(() => {
    if (hasClientFilters) {
      const start = (page - 1) * pageSize;
      return filteredTableSource.slice(start, start + pageSize);
    }
    return dataPoints;
  });

  // Inspect from allDataPoints so scatter clicks work even when that point
  // isn't on the current table page. Fallback: sole selected id only when
  // exactly one is selected (checklist 10).
  const inspectedPoint = $derived.by(() => {
    if (inspectedId) return allDataPoints.find((d) => d.id === inspectedId) || null;
    if (selectedIds.size === 1) {
      const id = Array.from(selectedIds)[0];
      return allDataPoints.find((d) => d.id === id) || null;
    }
    return null;
  });

  function handleSectionToggle(section: Exclude<ActionBarSection, null>) {
    const next = activeSection === section ? null : section;
    activeSection = next;
    if (next !== null) rightOpen = true;
  }

  // Selection contract (checklist 10): set selection; inspect first id when
  // any selected; close the active right-panel section so the Inspector shows.
  function handleSelect(ids: Set<string>) {
    selectedIds = ids;
    if (ids.size >= 1) inspectedId = Array.from(ids)[0];
    if (ids.size > 0) activeSection = null;
  }

  // Rebuild affected rows in both point arrays (mirrors React's dual setters).
  function patchPoints(fn: (d: DataPoint) => DataPoint) {
    dataPoints = dataPoints.map(fn);
    allDataPoints = allDataPoints.map(fn);
  }

  // ---- Mutation handlers: sync session map + patch arrays, toast, fire API
  // call with failure-toast only, NO rollback (checklist 5).
  function handleAssignLabel(label: string) {
    if (selectedIds.size === 0) return;
    let newlyLabeled = 0;
    const ids: number[] = new Array(selectedIds.size);
    let k = 0;
    for (const id of selectedIds) {
      if (!sessionLabels.has(id)) newlyLabeled++;
      sessionLabels.set(id, label);
      ids[k++] = Number(id);
    }
    const sel = selectedIds;
    patchPoints((d) => (sel.has(d.id) ? { ...d, label } : d));
    labeledThisSession += newlyLabeled;
    toast({ title: "Labels applied", description: `Assigned "${label}" to ${sel.size} image${sel.size > 1 ? "s" : ""}` });
    if (dbPath) {
      api
        .patch("/datasets/images/labels", { db_path: dbPath, ids, label })
        .catch(() => toast({ title: "Failed to save label", variant: "destructive" }));
    }
  }

  function handleRemoveLabels() {
    if (selectedIds.size === 0) return;
    const ids: number[] = new Array(selectedIds.size);
    let k = 0;
    for (const id of selectedIds) {
      sessionLabels.set(id, null);
      sessionTags.set(id, []);
      ids[k++] = Number(id);
    }
    const sel = selectedIds;
    patchPoints((d) => (sel.has(d.id) ? { ...d, label: null, labels: [] } : d));
    toast({ title: "Labels removed", description: `Cleared ${sel.size} image${sel.size > 1 ? "s" : ""}` });
    if (dbPath) {
      api
        .patch("/datasets/images/labels", { db_path: dbPath, ids, label: null })
        .catch(() => toast({ title: "Failed to remove labels", variant: "destructive" }));
    }
  }

  function handleQuickLabel(pointId: string, label: string) {
    sessionLabels.set(pointId, label);
    patchPoints((d) => (d.id === pointId ? { ...d, label } : d));
    labeledThisSession += 1;
    toast({ title: "Label applied", description: `Assigned "${label}"` });
    if (dbPath) {
      api
        .patch("/datasets/images/labels", { db_path: dbPath, ids: [Number(pointId)], label })
        .catch(() => toast({ title: "Failed to save label", variant: "destructive" }));
    }
  }

  // Multi-label: toggle one tag on one image. Reads the current tag set from
  // sessionTags first so several toggles fired in the same tick (e.g. accepting
  // a whole predicted set) build on each other instead of overwriting.
  function handleToggleTag(pointId: string, cls: string) {
    const point = allDataPoints.find((d) => d.id === pointId);
    if (!point) return;
    const currentTags = sessionTags.get(pointId) ?? point.labels;
    const present = !currentTags.includes(cls);
    const next = present ? [...currentTags, cls] : currentTags.filter((l) => l !== cls);
    sessionTags.set(pointId, next);
    patchPoints((d) => (d.id === pointId ? { ...d, labels: next } : d));
    // An image counts as newly labeled the moment it gets its first tag.
    if (present && currentTags.length === 0) labeledThisSession += 1;
    if (dbPath) {
      api
        .patch("/datasets/images/tags", { db_path: dbPath, ids: [Number(pointId)], class_name: cls, present })
        .catch(() => toast({ title: "Failed to save label", variant: "destructive" }));
    }
  }

  // Multi-label: add or remove one tag across a selection.
  function handleBulkTag(ids: string[], cls: string, present: boolean) {
    if (ids.length === 0) return;
    const idSet = new Set(ids);
    const nextById = new Map<string, string[]>();
    let newlyLabeled = 0;
    for (const d of allDataPoints) {
      if (!idSet.has(d.id)) continue;
      const currentTags = sessionTags.get(d.id) ?? d.labels;
      if (currentTags.includes(cls) === present) continue; // already in the wanted state
      const next = present ? [...currentTags, cls] : currentTags.filter((l) => l !== cls);
      sessionTags.set(d.id, next);
      nextById.set(d.id, next);
      if (present && currentTags.length === 0) newlyLabeled++;
    }
    if (nextById.size > 0) {
      patchPoints((d) => (nextById.has(d.id) ? { ...d, labels: nextById.get(d.id)! } : d));
      labeledThisSession += newlyLabeled;
    }
    toast({
      title: present ? "Label added" : "Label removed",
      description: `${present ? "Added" : "Removed"} "${cls}" ${present ? "to" : "from"} ${ids.length} image${ids.length > 1 ? "s" : ""}`,
    });
    if (dbPath) {
      api
        .patch("/datasets/images/tags", { db_path: dbPath, ids: ids.map(Number), class_name: cls, present })
        .catch(() => toast({ title: "Failed to save label", variant: "destructive" }));
    }
  }

  function handleNavigateInspector(direction: "prev" | "next") {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    const currentIdx = inspectedId ? ids.indexOf(inspectedId) : 0;
    const newIdx = direction === "next" ? (currentIdx + 1) % ids.length : (currentIdx - 1 + ids.length) % ids.length;
    inspectedId = ids[newIdx];
  }

  function handleRenameClass(oldName: string, newName: string) {
    cm.renameClass(oldName, newName);
    if (filters.focusClass === oldName) filters.focusClass = newName;
    if (taskType === "multilabel-classification") {
      const renameTag = (tags: string[]) => [...new Set(tags.map((t) => (t === oldName ? newName : t)))];
      for (const [id, tags] of sessionTags) {
        if (tags.includes(oldName)) sessionTags.set(id, renameTag(tags));
      }
      patchPoints((d) => (d.labels.includes(oldName) ? { ...d, labels: renameTag(d.labels) } : d));
    } else if (taskType !== "object-detection") {
      patchPoints((d) => (d.label === oldName ? { ...d, label: newName } : d));
    }
  }

  function handleAddClass(name: string) {
    cm.addClass(name);
    if (selectedIds.size === 0) return;
    if (taskType === "multilabel-classification") {
      handleBulkTag(Array.from(selectedIds), name, true);
    } else if (taskType !== "object-detection") {
      handleAssignLabel(name);
    }
  }

  function handleRemoveClass(name: string) {
    cm.removeClass(name);
    if (filters.focusClass === name) filters.focusClass = null;
    if (taskType === "multilabel-classification") {
      // Drop in-session tags first — toDataPoint reads sessionTags with priority
      // over label_classes, so leaving them would re-apply the class on refetch.
      for (const [id, tags] of sessionTags) {
        if (tags.includes(name)) sessionTags.set(id, tags.filter((t) => t !== name));
      }
      patchPoints((d) => (d.labels.includes(name) ? { ...d, labels: d.labels.filter((l) => l !== name) } : d));
    } else if (taskType !== "object-detection") {
      // Drop in-session labels first — toDataPoint reads sessionLabels with
      // priority over r.annotation, so leaving them would re-apply the class
      // on the next refetch.
      for (const [id, lbl] of sessionLabels) {
        if (lbl === name) sessionLabels.delete(id);
      }
      patchPoints((d) => (d.label === name ? { ...d, label: null } : d));
    }
    if (dbPath) {
      api
        .post("/datasets/classes/remove", { db_path: dbPath, class_name: name })
        .then(() => {
          // Force the all-images query to refetch so removed annotations/boxes
          // are reflected immediately in the scatter and inspector.
          queryClient.invalidateQueries({ queryKey: ["images-all", dbPath] });
        })
        .catch(() => toast({ title: "Failed to remove class", variant: "destructive" }));
    }
  }

  function handleIgnore(ids: string[], ignored: boolean) {
    ids.forEach((id) => (ignored ? sessionIgnored.add(id) : sessionIgnored.delete(id)));
    const idSet = new Set(ids);
    patchPoints((d) => (idSet.has(d.id) ? { ...d, ignored } : d));
    if (dbPath) {
      api
        .patch("/datasets/images/ignore", { db_path: dbPath, ids: ids.map(Number), ignored })
        .catch(() => toast({ title: "Failed to update ignore status", variant: "destructive" }));
    }
  }

  function handleAssignSplit(ids: string[], split: SplitName | null) {
    if (ids.length === 0) return;
    ids.forEach((id) => sessionSplitOverrides.set(id, split));
    const idSet = new Set(ids);
    patchPoints((d) => (idSet.has(d.id) ? { ...d, splitOverride: split } : d));
    const label = split === null ? "cleared" : split;
    toast({ title: "Split updated", description: `Marked ${ids.length} image${ids.length > 1 ? "s" : ""} as ${label}` });
    if (dbPath) {
      api
        .patch("/datasets/images/split", { db_path: dbPath, ids: ids.map(Number), split })
        .catch(() => toast({ title: "Failed to update split", variant: "destructive" }));
    }
  }

  function handleNoObjects(id: string, noObjects: boolean) {
    patchPoints((d) => (d.id === id ? { ...d, noObjects } : d));
    if (dbPath) {
      api
        .patch("/datasets/images/no-objects", { db_path: dbPath, ids: [Number(id)], no_objects: noObjects })
        .catch(() => toast({ title: "Failed to update no-objects status", variant: "destructive" }));
    }
  }

  // Writes into the non-reactive coords map (numeric id → string key) so coords
  // survive imageData refetches, then rebuilds both point arrays (checklist 6).
  function handleProjectionReady(coords: Array<{ id: number; x: number; y: number }>) {
    const coordMap = new Map(coords.map((c) => [String(c.id), { x: c.x, y: c.y }]));
    coordMap.forEach((v, k) => projectionCoords.set(k, v));
    patchPoints((d) => {
      const c = coordMap.get(d.id);
      return c ? { ...d, x: c.x, y: c.y } : d;
    });
  }

  // Effect 7: auto-trigger projection on *actual* embedding-model change
  // (prevEmbeddingModel comparison; skip first run — checklist 7).
  // svelte-ignore state_referenced_locally
  let prevEmbeddingModel = embeddingModel;
  $effect(() => {
    const model = embeddingModel;
    const exists = embeddingStatus.exists;
    if (!dbPath || !exists) return;
    if (prevEmbeddingModel === model) return;
    prevEmbeddingModel = model;

    untrack(() => {
      const method = filters.reductionMethod;
      const params =
        method === "umap" ? { n_neighbors: filters.perplexity } : method === "tsne" ? { perplexity: filters.perplexity } : {};
      api
        .post<{ cached: boolean; param_hash: string; job_id?: string }>("/projections/compute", {
          db_path: dbPath,
          model,
          method,
          params,
        })
        .then((resp) => {
          if (resp.cached && resp.param_hash) {
            api
              .getBinary(`/projections/${resp.param_hash}/blob?db_path=${encodeURIComponent(dbPath)}`)
              .then((buf) => handleProjectionReady(parseProjectionBlob(buf)))
              .catch(() => {});
          } else if (resp.job_id && resp.param_hash) {
            autoTriggerJobId = resp.job_id;
            autoTriggerParamHash = resp.param_hash;
          }
        })
        .catch(() => {});
    });
  });

  // Effect 8: one-shot mount projection (umap, n_neighbors 15) — failures
  // silently swallowed, the app stays in table view (checklist 7).
  $effect(() => {
    if (!dbPath || projectionTriggered) return;
    projectionTriggered = true;
    api
      .post<{ cached: boolean; param_hash: string; job_id?: string }>("/projections/compute", {
        db_path: dbPath,
        model: untrack(() => embeddingModel),
        method: "umap",
        params: { n_neighbors: 15 },
      })
      .then((resp) => {
        if (resp.cached && resp.param_hash) {
          api
            .getBinary(`/projections/${resp.param_hash}/blob?db_path=${encodeURIComponent(dbPath)}`)
            .then((buf) => handleProjectionReady(parseProjectionBlob(buf)))
            .catch(() => {});
        } else if (!resp.cached && resp.job_id && resp.param_hash) {
          autoTriggerJobId = resp.job_id;
          autoTriggerParamHash = resp.param_hash;
        }
      })
      .catch(() => {
        // No embeddings yet — stay in table view, no error shown
      });
  });

  // Effect 9: poll the auto-triggered projection job and fetch coords when
  // done — keyed on job status only (prev-status guard pattern).
  const autoTriggerJob = createJobPoller(
    () => autoTriggerJobId,
    () => dbPath || null,
  );
  let prevAutoJobStatus: string | undefined;
  $effect(() => {
    const status = autoTriggerJob.data?.status;
    if (status === prevAutoJobStatus) return;
    prevAutoJobStatus = status;
    const paramHash = untrack(() => autoTriggerParamHash);
    if (status === "done" && paramHash && dbPath) {
      api
        .getBinary(`/projections/${paramHash}/blob?db_path=${encodeURIComponent(dbPath)}`)
        .then((buf) => {
          handleProjectionReady(parseProjectionBlob(buf));
          autoTriggerJobId = null;
        })
        .catch(() => {
          autoTriggerJobId = null;
        });
    } else if (status === "failed" || status === "cancelled") {
      autoTriggerJobId = null;
    }
  });
</script>

<div class="h-screen flex flex-col bg-background overflow-hidden">
  <!-- Top bar -->
  <div class="h-10 border-b border-border flex items-center px-3 gap-2 shrink-0">
    <span class="text-xs font-semibold tracking-widest uppercase text-primary">iits</span>
    <span class="text-xs text-muted-foreground ml-1">Active Learning Studio</span>
    <span class="text-xs text-muted-foreground">•</span>
    <span class="text-xs text-muted-foreground capitalize">
      {taskType === "object-detection"
        ? "Object Detection"
        : taskType === "multilabel-classification"
          ? "Multi-Label"
          : "Classification"}
    </span>
    <div class="flex-1"></div>
    <!-- View toggle -->
    <div class="flex items-center border border-border rounded-(--radius) overflow-hidden">
      <Tooltip>
        <TooltipTrigger>
          {#snippet child({ props })}
            <Button
              {...props}
              variant={viewMode === "scatter" ? "secondary" : "ghost"}
              size="sm"
              class="h-7 w-7 p-0 rounded-none"
              onclick={() => (viewModeOverride = "scatter")}
              disabled={!hasEmbeddings}
              aria-label="Scatter plot view"
            >
              <ChartScatter class="h-3.5 w-3.5" />
            </Button>
          {/snippet}
        </TooltipTrigger>
        <TooltipContent side="bottom" class="text-xs">
          {hasEmbeddings ? "Scatter plot" : "No embeddings available"}
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger>
          {#snippet child({ props })}
            <Button
              {...props}
              variant={viewMode === "table" ? "secondary" : "ghost"}
              size="sm"
              class="h-7 w-7 p-0 rounded-none"
              onclick={() => (viewModeOverride = "table")}
              aria-label="Table view"
            >
              <TableIcon class="h-3.5 w-3.5" />
            </Button>
          {/snippet}
        </TooltipTrigger>
        <TooltipContent side="bottom" class="text-xs">Table view</TooltipContent>
      </Tooltip>
    </div>
    <Button
      variant="ghost"
      size="sm"
      class="h-7 text-xs"
      onclick={() => (leftOpen = !leftOpen)}
      aria-label={leftOpen ? "Collapse left sidebar" : "Expand left sidebar"}
    >
      {#if leftOpen}<PanelLeftClose class="h-3.5 w-3.5" />{:else}<PanelLeftOpen class="h-3.5 w-3.5" />{/if}
    </Button>
    <Button
      variant="ghost"
      size="sm"
      class="h-7 text-xs"
      onclick={() => {
        const wasOpen = rightOpen;
        rightOpen = !rightOpen;
        if (wasOpen) activeSection = null;
      }}
      aria-label={rightOpen ? "Collapse right sidebar" : "Expand right sidebar"}
    >
      {#if rightOpen}<PanelRightClose class="h-3.5 w-3.5" />{:else}<PanelRightOpen class="h-3.5 w-3.5" />{/if}
    </Button>
  </div>

  <!-- Action Bar -->
  <ActionBar
    {activeSection}
    onSectionToggle={handleSectionToggle}
    {embeddingStatus}
    {embeddingModel}
    {reductionStatus}
    data={allDataPoints}
  />

  <!-- Main content -->
  <div class="flex-1 flex min-h-0">
    {#if leftOpen}
      <div class="w-60 border-r border-border shrink-0 overflow-y-auto scrollbar-thin">
        <ExplorerSidebar
          data={allDataPoints}
          {filters}
          onFiltersChange={setFilters}
          {taskType}
          classNames={cm.classNames}
          classColors={cm.classColors}
          onAddClass={handleAddClass}
          onRenameClass={handleRenameClass}
          onRemoveClass={handleRemoveClass}
        />
      </div>
    {/if}

    <div class="flex-1 flex flex-col min-w-0">
      {#if viewMode === "scatter"}
        <div class="relative flex-1 flex flex-col min-h-0">
          <ScatterPlot
            data={filteredData}
            allData={allDataPoints}
            colorBy={filters.colorBy}
            {selectedIds}
            {selectionTool}
            onSelect={handleSelect}
            onHover={(id) => (hoveredId = id)}
            onSelectionToolChange={(t) => (selectionTool = t)}
            {taskType}
            classNames={cm.classNames}
            classColors={cm.classColors}
            focusClass={filters.focusClass}
          />
          {#if reductionStatus.isLoading || !!autoTriggerJobId}
            <div class="absolute inset-0 flex items-center justify-center bg-background/50 z-10">
              <div class="flex items-center gap-2 text-sm text-muted-foreground bg-card border border-border rounded-lg px-4 py-2 shadow-xs">
                <LoaderCircle class="h-4 w-4 animate-spin" />
                Computing projection…
              </div>
            </div>
          {/if}
        </div>
      {:else}
        <ImageTable
          data={filteredTableData}
          {selectedIds}
          onSelect={handleSelect}
          onRowClick={(id) => {
            selectedIds = new Set([id]);
            inspectedId = id;
          }}
          classColors={cm.classColors}
          {alScoreMap}
          {page}
          {pageSize}
          total={hasClientFilters ? filteredTableSource.length : (imagesQuery.data?.total ?? 0)}
          onPageChange={(p) => (page = p)}
          onPageSizeChange={(size) => {
            pageSize = size;
            page = 1;
          }}
          {sortBy}
          {sortOrder}
          onIgnore={handleIgnore}
          onSort={(field) => {
            if (sortBy === field) {
              // Toggle: desc → asc → off
              if (sortOrder === "desc") sortOrder = "asc";
              else {
                sortBy = null;
                sortOrder = "desc";
              }
            } else {
              sortBy = field;
              sortOrder = "desc";
            }
            page = 1;
          }}
        />
      {/if}
    </div>

    {#if rightOpen}
      <div class="w-80 border-l border-border shrink-0 overflow-y-auto scrollbar-thin">
        {#if activeSection === "embeddings"}
          <EmbeddingsPanel
            {dbPath}
            {embeddingModel}
            onEmbeddingModelChange={(m) => (embeddingModel = m)}
            initialEmbeddingJobId={datasetState?.activeEmbeddingJobId ?? null}
            initialBatchSize={datasetState?.embeddingBatchSize}
            onStatusChange={(s) => (embeddingIsComputing = s.isComputing)}
            onClose={() => (activeSection = null)}
          />
        {:else if activeSection === "reduction"}
          <ReductionPanel
            {dbPath}
            {embeddingModel}
            embeddingExists={embeddingStatus.exists}
            onProjectionReady={handleProjectionReady}
            {filters}
            onFiltersChange={setFilters}
            onStatusChange={(s) => (reductionStatus = s)}
            onClose={() => (activeSection = null)}
          />
        {:else if activeSection === "activeLearning"}
          <ActiveLearningPanel
            data={allDataPoints}
            onStartLabeling={(budget, strategy) => {
              labelingBudget = budget;
              labelingStrategy = strategy;
              annotatorUseSelection = false;
              navigate("/explorer/annotate");
            }}
            {alWeights}
            onALWeightsChange={(w) => (alWeights = w)}
            onClose={() => (activeSection = null)}
          />
        {:else}
          <InspectorPanel
            point={inspectedPoint}
            selectedCount={selectedIds.size}
            onQuickLabel={handleQuickLabel}
            onToggleTag={handleToggleTag}
            onBulkTag={handleBulkTag}
            onNavigate={handleNavigateInspector}
            onOpenAnnotator={() => {
              annotatorUseSelection = true;
              navigate("/explorer/annotate");
            }}
            onAssignLabel={handleAssignLabel}
            onRemoveLabels={handleRemoveLabels}
            onClearSelection={() => (selectedIds = new Set())}
            classNames={cm.classNames}
            pinnedClasses={ss.pinnedClasses}
            shortcuts={ss.shortcuts}
            onTogglePinned={ss.togglePinned}
            onIgnore={handleIgnore}
            onAssignSplit={handleAssignSplit}
            selectedIds={Array.from(selectedIds)}
            {taskType}
            alScore={inspectedPoint ? alScoreMap.get(inspectedPoint.id) : undefined}
          />
        {/if}
      </div>
    {/if}
  </div>

  <StatusBar
    data={allDataPoints}
    {dbPath}
    datasetPath={datasetState?.datasetPath ?? ""}
    {labeledThisSession}
    onOpenTraining={() => (trainingOpen = true)}
    {selectedRunId}
    {taskType}
  />

  <TrainingDrawer
    open={trainingOpen}
    onClose={() => (trainingOpen = false)}
    data={allDataPoints}
    {dbPath}
    classNames={cm.classNames}
    {taskType}
    {selectedRunId}
    onSelectRun={(id) => (selectedRunId = id)}
  />

  {#if isAnnotating}
    <AnnotatorOverlay
      data={allDataPoints}
      selectedIds={annotatorUseSelection ? selectedIds : new Set()}
      {alScoreMap}
      {taskType}
      {dbPath}
      onClose={() => {
        queryClient.invalidateQueries({ queryKey: ["images-all", dbPath] });
        navigate("/explorer");
      }}
      onLabel={(id, label) => handleQuickLabel(id, label)}
      onToggleTag={handleToggleTag}
      onIgnore={(id) => handleIgnore([id], true)}
      onNoObjects={(id) => handleNoObjects(id, true)}
      classNames={cm.classNames}
      classColors={cm.classColors}
      onAddClass={cm.addClass}
      shortcuts={ss.shortcuts}
      keyToClass={ss.keyToClass}
      pinnedClasses={ss.pinnedClasses}
      onTogglePinned={ss.togglePinned}
      budget={labelingBudget}
      strategy={labelingStrategy}
      {selectedRunId}
      onOpenSettings={() => (shortcutDialogOpen = true)}
    />
  {/if}

  <ShortcutSettingsDialog
    open={shortcutDialogOpen}
    onClose={() => (shortcutDialogOpen = false)}
    classNames={cm.classNames}
    classColors={cm.classColors}
    shortcuts={ss.shortcuts}
    onUpdateShortcut={ss.updateShortcut}
    onResetDefaults={ss.resetDefaults}
    pinnedClasses={ss.pinnedClasses}
    onTogglePinned={ss.togglePinned}
  />
</div>
