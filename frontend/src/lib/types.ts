export interface TrainingRun {
  id: string;
  status: "queued" | "running" | "predicting" | "done" | "failed" | "stopped";
  task_type?: string;
  model_name: string;
  display_name: string;
  config: Record<string, unknown>;
  num_classes: number;
  class_names: string[];
  label_count: number;
  epochs: number;
  current_epoch: number;
  current_step: number;
  total_steps: number;
  train_loss: number | null;
  val_loss: number | null;
  val_accuracy: number | null;
  val_f1: number | null;
  best_accuracy: number | null;
  best_f1: number | null;
  val_auroc: number | null;
  val_precision: number | null;
  val_recall: number | null;
  best_auroc: number | null;
  primary_metric_name?: string;
  secondary_metric_name?: string;
  error: string | null;
  has_checkpoint?: boolean;
  output_dir: string;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
  val_per_class?: {
    classes: string[];
    precision: number[];
    recall: number[];
    f1: number[];
    support: number[];
  } | null;
}

export interface TrainingMetrics {
  run_id: string;
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_accuracy: number;
  val_f1: number;
  learning_rate: number;
  val_auroc: number;
  val_precision: number;
  val_recall: number;
}

export interface AdvancedSettings {
  optimizer: "adamw" | "sgd" | "adam";
  scheduler: "cosine" | "linear" | "none";
  warmup_epochs: number;
  horizontal_flip: boolean;
  rotation: number;
  color_jitter_brightness: number;
  color_jitter_contrast: number;
  color_jitter_saturation: number;
  image_size: number;
  weight_decay: number;
  freeze_backbone: boolean;
  early_stopping: boolean;
  early_stopping_patience: number;
  precision: "32-true" | "16-mixed" | "bf16-mixed";
  accumulate_grad_batches: number;
  gradient_clip_val: number;
  val_check_interval: number;
  limit_train_batches: number;
}

export const ADVANCED_DEFAULTS: AdvancedSettings = {
  optimizer: "adamw",
  scheduler: "cosine",
  warmup_epochs: 0,
  horizontal_flip: true,
  rotation: 15,
  color_jitter_brightness: 0.2,
  color_jitter_contrast: 0.2,
  color_jitter_saturation: 0.2,
  image_size: 224,
  weight_decay: 0.01,
  freeze_backbone: false,
  early_stopping: false,
  early_stopping_patience: 5,
  precision: "32-true",
  accumulate_grad_batches: 1,
  gradient_clip_val: 0.0,
  val_check_interval: 1.0,
  limit_train_batches: 0,
};

// ---- Explorer page-level types (React exported these from pages/Explorer.tsx;
// ---- moved into the lib layer per spec §10 deviation 5 so components can
// ---- import them without a page → component cycle).

import type { ColorBy } from "@/lib/taskConfig";

export type FilterLabeled = "all" | "labeled" | "unlabeled" | "ignored";
export type ReductionMethod = "umap" | "tsne" | "pca";
export type ViewMode = "scatter" | "table";
export type ActionBarSection = "embeddings" | "reduction" | "activeLearning" | null;

export interface ExplorerFilters {
  colorBy: ColorBy;
  filterLabeled: FilterLabeled;
  selectedClasses: string[];
  confidenceRange: [number, number];
  reductionMethod: ReductionMethod;
  perplexity: number;
  // Multi-label: class highlighted by the "hasClass" coloring mode.
  focusClass: string | null;
}
