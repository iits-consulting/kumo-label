<script lang="ts">
  import { createQueries } from "@tanstack/svelte-query";
  import { LineChart } from "layerchart";
  import { curveMonotoneX } from "d3-shape";
  import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
  import { api } from "@/lib/api";
  import { TASK_CONFIGS } from "@/lib/taskConfig";
  import type { TaskType } from "@/lib/mockData";

  const RUN_COLORS = [
    "hsl(210, 76%, 52%)",
    "hsl(340, 75%, 55%)",
    "hsl(142, 60%, 40%)",
    "hsl(38, 90%, 50%)",
    "hsl(270, 60%, 55%)",
    "hsl(190, 70%, 42%)",
  ];

  interface TrainingMetric {
    run_id: string;
    epoch: number;
    step: number;
    train_loss: number;
    val_loss: number;
    val_accuracy: number;
    val_f1: number;
    learning_rate: number;
    val_auroc: number;
    val_precision: number;
    val_recall: number;
  }

  interface Props {
    runIds: string[];
    dbPath: string;
    taskType: TaskType;
    runNames?: Record<string, string>;
    isLive?: boolean;
  }

  let { runIds, dbPath, taskType, runNames = {}, isLive = false }: Props = $props();

  const taskConfig = $derived(TASK_CONFIGS[taskType]);

  const results = createQueries(() => ({
    queries: runIds.map((runId) => ({
      queryKey: ["training-metrics", runId],
      queryFn: () =>
        api.get<TrainingMetric[]>(`/training/runs/${runId}/metrics?db_path=${encodeURIComponent(dbPath)}`),
      refetchInterval: isLive ? 5000 : (false as const),
    })),
  }));

  const allLoading = $derived(results.every((r) => r.isLoading));
  const hasData = $derived(results.some((r) => r.data && r.data.length > 0));
  const multi = $derived(runIds.length > 1);

  interface LineDef {
    dataKey: keyof TrainingMetric;
    label: string;
    dashed?: boolean;
  }

  const charts = $derived<{ title: string; lines: LineDef[] }[]>([
    {
      title: "Loss",
      lines: [
        { dataKey: "train_loss", label: "Train", dashed: true },
        { dataKey: "val_loss", label: "Val" },
      ],
    },
    {
      title: taskConfig.metrics.primary,
      lines: [{ dataKey: "val_accuracy", label: taskConfig.metrics.primaryKey }],
    },
    {
      title: taskConfig.metrics.secondary,
      lines: [{ dataKey: "val_f1", label: taskConfig.metrics.secondaryKey }],
    },
    {
      title: "Learning Rate",
      lines: [{ dataKey: "learning_rate", label: "LR" }],
    },
  ]);

  type Row = Record<string, number | null>;

  // Merge data from all runs keyed by checkpoint index so that sub-epoch
  // validations (val_check_interval < 1.0) each get their own point.
  // X axis = checkpoint index within each run's metrics array, NOT epoch.
  function buildChart(lines: LineDef[]) {
    const checkpointMap = new Map<number, Row>();

    runIds.forEach((runId, runIdx) => {
      const data = results[runIdx]?.data as TrainingMetric[] | undefined;
      if (!data) return;
      data.forEach((m, i) => {
        if (!checkpointMap.has(i)) {
          checkpointMap.set(i, { checkpoint: i + 1, _epoch: m.epoch != null ? m.epoch + 1 : null });
        }
        const row = checkpointMap.get(i)!;
        for (const line of lines) {
          const key = multi ? `${line.dataKey}_${runIdx}` : line.dataKey;
          row[key] = m[line.dataKey] as number;
        }
      });
    });

    const data = Array.from(checkpointMap.values()).sort(
      (a, b) => (a.checkpoint as number) - (b.checkpoint as number),
    );

    const config: ChartConfig = {};
    const series: { key: string; label: string; color: string; props: Record<string, unknown> }[] = [];

    runIds.forEach((runId, runIdx) => {
      const color = RUN_COLORS[runIdx % RUN_COLORS.length];
      const runLabel = runNames[runId] || `Run ${runIdx + 1}`;
      for (const line of lines) {
        const key = multi ? `${line.dataKey}_${runIdx}` : line.dataKey;
        const label = multi ? `${runLabel} ${line.label}` : line.label;
        config[key] = { label, color };
        series.push({
          key,
          label,
          color,
          props: {
            // Skip missing points (shorter runs in a multi-run merge).
            defined: (d: Row) => d[key] != null,
            ...(line.dashed ? { "stroke-dasharray": "4 3" } : {}),
          },
        });
      }
    });

    return { data, config, series };
  }

  function tooltipLabel(data: Row[]) {
    return (v: unknown) => {
      const row = data.find((r) => r.checkpoint === v);
      return row?._epoch != null ? `Epoch ${row._epoch}` : `Checkpoint ${v}`;
    };
  }
</script>

{#if allLoading}
  <div class="text-xs text-muted-foreground py-3 text-center">Loading metrics...</div>
{:else if !hasData}
  <div class="text-xs text-muted-foreground py-3 text-center">No metrics yet</div>
{:else}
  <div class="grid grid-cols-2 gap-2 py-2">
    {#each charts as chart (chart.title)}
      {@const built = buildChart(chart.lines)}
      {#if built.data.length > 0}
        <div class="space-y-1">
          <span class="text-[10px] font-medium text-muted-foreground ml-1">{chart.title}</span>
          <ChartContainer config={built.config} class="h-[130px] w-full aspect-auto">
            <LineChart
              data={built.data}
              x="checkpoint"
              series={built.series}
              props={{
                spline: { curve: curveMonotoneX, strokeWidth: 1.5, motion: "none" },
                highlight: { points: { r: 3 } },
                xAxis: { tickLabelProps: { class: "text-[10px]" } },
                yAxis: { tickLabelProps: { class: "text-[10px]" } },
              }}
            >
              {#snippet tooltip()}
                <ChartTooltip indicator="line" labelFormatter={tooltipLabel(built.data)} />
              {/snippet}
            </LineChart>
          </ChartContainer>
        </div>
      {/if}
    {/each}
  </div>
{/if}
