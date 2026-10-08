import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";
import type { TrainingRun } from "@/lib/types";
import PerClassMetricsTable from "./PerClassMetricsTable.svelte";

function makeRun(overrides: Partial<TrainingRun> = {}): TrainingRun {
  return {
    id: "run-1",
    status: "done",
    task_type: "multilabel-classification",
    model_name: "resnet50",
    display_name: "Run 1",
    config: {},
    num_classes: 3,
    class_names: ["cat", "dog", "bird"],
    label_count: 100,
    epochs: 10,
    current_epoch: 10,
    current_step: 100,
    total_steps: 100,
    train_loss: 0.1,
    val_loss: 0.2,
    val_accuracy: 0.8,
    val_f1: 0.5,
    best_accuracy: 0.8,
    best_f1: 0.5,
    val_auroc: 0.9,
    val_precision: 0.85,
    val_recall: 0.75,
    best_auroc: 0.9,
    error: null,
    has_checkpoint: true,
    output_dir: "/tmp/run-1",
    created_at: "2026-01-01T00:00:00",
    updated_at: "2026-01-01T00:00:00",
    finished_at: "2026-01-01T00:10:00",
    val_per_class: null,
    ...overrides,
  };
}

describe("PerClassMetricsTable", () => {
  it("renders nothing when val_per_class is null", () => {
    const { container } = render(PerClassMetricsTable, { props: { run: makeRun({ val_per_class: null }) } });
    // Svelte leaves an {#if} anchor comment node, so assert no element children
    expect(container.childElementCount).toBe(0);
  });

  it("renders nothing when val_per_class is undefined", () => {
    const run = makeRun();
    delete (run as Partial<TrainingRun>).val_per_class;
    const { container } = render(PerClassMetricsTable, { props: { run } });
    // Svelte leaves an {#if} anchor comment node, so assert no element children
    expect(container.childElementCount).toBe(0);
  });

  it("renders nothing when val_per_class has no classes", () => {
    const run = makeRun({
      val_per_class: { classes: [], precision: [], recall: [], f1: [], support: [] },
    });
    const { container } = render(PerClassMetricsTable, { props: { run } });
    // Svelte leaves an {#if} anchor comment node, so assert no element children
    expect(container.childElementCount).toBe(0);
  });

  it("renders rows sorted worst-F1-first", () => {
    const run = makeRun({
      val_per_class: {
        classes: ["cat", "dog", "bird"],
        precision: [0.9, 0.6, 0.95],
        recall: [0.85, 0.55, 0.92],
        f1: [0.875, 0.574, 0.935],
        support: [120, 45, 78],
      },
    });
    render(PerClassMetricsTable, { props: { run } });

    const rows = screen.getAllByRole("row").slice(1); // skip header row
    const rowLabels = rows.map((r) => r.textContent);
    // dog (0.574) worst, then cat (0.875), then bird (0.935)
    expect(rowLabels[0]).toContain("dog");
    expect(rowLabels[1]).toContain("cat");
    expect(rowLabels[2]).toContain("bird");
  });

  it("formats precision/recall/F1 to 3 decimals and support as an integer", () => {
    const run = makeRun({
      val_per_class: {
        classes: ["cat"],
        precision: [0.9],
        recall: [0.85],
        f1: [0.875],
        support: [120],
      },
    });
    render(PerClassMetricsTable, { props: { run } });

    const row = screen.getByRole("row", { name: /cat/ });
    expect(row.textContent).toContain("0.900");
    expect(row.textContent).toContain("0.850");
    expect(row.textContent).toContain("0.875");
    expect(row.textContent).toContain("120");
    expect(row.textContent).not.toContain("120.0");
  });

  it("renders only the complete rows and does not throw when arrays have unequal lengths", () => {
    const run = makeRun({
      val_per_class: {
        classes: ["cat", "dog", "bird"],
        precision: [0.9, 0.6], // one short — bird's data is missing
        recall: [0.85, 0.55, 0.92],
        f1: [0.875, 0.574, 0.935],
        support: [120, 45, 78],
      },
    });

    expect(() => render(PerClassMetricsTable, { props: { run } })).not.toThrow();

    const rows = screen.getAllByRole("row").slice(1); // skip header row
    expect(rows).toHaveLength(2);
    const rowLabels = rows.map((r) => r.textContent);
    // dog (0.574) worst, then cat (0.875); bird is dropped for lacking precision
    expect(rowLabels[0]).toContain("dog");
    expect(rowLabels[1]).toContain("cat");
    expect(screen.queryByText("bird")).not.toBeInTheDocument();
  });

  it("applies a destructive tint to rows with F1 below 0.5", () => {
    const run = makeRun({
      val_per_class: {
        classes: ["cat", "dog"],
        precision: [0.9, 0.3],
        recall: [0.85, 0.2],
        f1: [0.875, 0.25],
        support: [120, 10],
      },
    });
    render(PerClassMetricsTable, { props: { run } });

    const badRow = screen.getByRole("row", { name: /dog/ });
    const goodRow = screen.getByRole("row", { name: /cat/ });
    expect(badRow.className).toMatch(/destructive/);
    expect(goodRow.className).not.toMatch(/destructive/);
  });
});
