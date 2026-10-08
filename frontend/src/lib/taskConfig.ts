import type { TaskType } from "@/lib/mockData";

export type ColorBy = "groundTruth" | "predictions" | "uncertainty" | "mistake" | "annotationCount" | "hasClass" | "split";

export type SelectionTool = "click" | "lasso" | "box";

export interface TaskConfig {
  models: { value: string; label: string }[];
  defaults: {
    lr: string;
    batchSize: string;
    imageSize?: number;
    gradientClipVal?: number;
    warmupEpochs?: number;
  };
  metrics: {
    primary: string;
    secondary: string;
    primaryKey: string;  // short label for inline use, e.g. "Acc" or "mAP"
    secondaryKey: string;
  };
  colorByOptions: { value: ColorBy; label: string }[];
  defaultColorBy: ColorBy;
}

export const TASK_CONFIGS: Record<TaskType, TaskConfig> = {
  classification: {
    models: [
      { value: "resnet50", label: "ResNet-50" },
      { value: "efficientnet_b0", label: "EfficientNet-B0" },
      { value: "vit_base", label: "ViT-Base" },
      { value: "checkpoint", label: "Trained" },
      { value: "local_checkpoint", label: "Load Checkpoint (.ckpt)" },
      { value: "custom", label: "Custom (HuggingFace / local)" },
    ],
    defaults: { lr: "0.001", batchSize: "32" },
    metrics: { primary: "Accuracy", secondary: "F1", primaryKey: "Acc", secondaryKey: "F1" },
    colorByOptions: [
      { value: "groundTruth", label: "Ground Truth" },
      { value: "predictions", label: "Predictions" },
      { value: "uncertainty", label: "Uncertainty" },
      { value: "mistake", label: "Mistake" },
      { value: "split", label: "Train/Val Split" },
    ],
    defaultColorBy: "groundTruth",
  },
  "multilabel-classification": {
    models: [
      { value: "resnet50", label: "ResNet-50" },
      { value: "efficientnet_b0", label: "EfficientNet-B0" },
      { value: "vit_base", label: "ViT-Base" },
      { value: "checkpoint", label: "Trained" },
      { value: "local_checkpoint", label: "Load Checkpoint (.ckpt)" },
      { value: "custom", label: "Custom (HuggingFace / local)" },
    ],
    defaults: { lr: "0.001", batchSize: "32" },
    metrics: { primary: "F1 (macro)", secondary: "Exact Match", primaryKey: "F1", secondaryKey: "EM" },
    colorByOptions: [
      { value: "annotationCount", label: "Number of Labels" },
      { value: "hasClass", label: "Has Class…" },
      { value: "uncertainty", label: "Uncertainty" },
      { value: "mistake", label: "Mistake" },
      { value: "split", label: "Train/Val Split" },
    ],
    defaultColorBy: "annotationCount",
  },
  "object-detection": {
    models: [
      { value: "detr", label: "DETR (ResNet-50)" },
      { value: "checkpoint", label: "Trained" },
      { value: "local_checkpoint", label: "Load Checkpoint (.ckpt)" },
      { value: "custom", label: "Custom (HuggingFace / local)" },
    ],
    defaults: { lr: "0.0001", batchSize: "4", imageSize: 800, gradientClipVal: 0.1, warmupEpochs: 1 },
    metrics: { primary: "mAP", secondary: "mAP@50", primaryKey: "mAP", secondaryKey: "mAP@50" },
    colorByOptions: [
      { value: "annotationCount", label: "Number of Boxes" },
      { value: "uncertainty", label: "Uncertainty" },
      { value: "split", label: "Train/Val Split" },
    ],
    defaultColorBy: "annotationCount",
  },
};
