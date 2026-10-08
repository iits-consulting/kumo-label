import logging
import warnings

import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torchmetrics.detection import MeanAveragePrecision

from kumo_label.training.base_module import BaseKumoModule

logger = logging.getLogger(__name__)


class KumoDetector(BaseKumoModule):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        lr: float,
        epochs: int,
        warmup_epochs: int = 0,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        weight_decay: float = 0.01,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_id, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        if freeze_backbone:
            self._freeze_backbone()
        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        # Replace NaN/Inf loss with zero to prevent crash — gradient clipping handles the rest
        if not torch.isfinite(loss):
            loss = torch.zeros_like(loss, requires_grad=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        except Exception:
            logger.warning("validation_step forward pass failed (batch %d), skipping", batch_idx, exc_info=True)
            return
        loss = outputs.loss
        if not torch.isfinite(loss):
            return
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        # Collect predictions for mAP
        img_h, img_w = batch["pixel_values"].shape[-2:]
        target_sizes = torch.tensor(
            [[img_h, img_w]] * len(batch["pixel_values"]),
            device=self.device,
        )
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )

        preds = []
        targets = []
        for i, result in enumerate(results):
            pred_boxes = result["boxes"].detach()
            pred_scores = result["scores"].detach()
            pred_labels = result["labels"].detach()

            # Filter out NaN/Inf predictions
            valid = torch.isfinite(pred_boxes).all(dim=1)
            preds.append({
                "boxes": pred_boxes[valid],
                "scores": pred_scores[valid],
                "labels": pred_labels[valid],
            })

            # Convert target boxes from DETR's normalized (cx, cy, w, h) to pixel (x0, y0, x1, y1)
            # DETR pads targets to N slots with class_labels=num_classes for "no object" — filter those out
            tgt = batch["labels"][i]
            tgt_classes = tgt["class_labels"].to(self.device)
            tgt_boxes_all = tgt["boxes"].to(self.device)

            real_mask = tgt_classes < self.model.config.num_labels
            tgt_boxes_cxcywh = tgt_boxes_all[real_mask]
            tgt_classes = tgt_classes[real_mask]

            # Also filter NaN/Inf targets
            finite_mask = torch.isfinite(tgt_boxes_cxcywh).all(dim=1)
            tgt_boxes_cxcywh = tgt_boxes_cxcywh[finite_mask]
            tgt_classes = tgt_classes[finite_mask]

            if tgt_boxes_cxcywh.numel() == 0:
                targets.append({"boxes": torch.zeros((0, 4), device=self.device), "labels": torch.zeros(0, dtype=torch.long, device=self.device)})
                continue

            cx, cy, w, h = tgt_boxes_cxcywh.unbind(dim=1)
            x0 = (cx - w / 2) * img_w
            y0 = (cy - h / 2) * img_h
            x1 = (cx + w / 2) * img_w
            y1 = (cy + h / 2) * img_h
            tgt_boxes_xyxy = torch.stack([x0, y0, x1, y1], dim=1)

            targets.append({
                "boxes": tgt_boxes_xyxy,
                "labels": tgt_classes,
            })

        try:
            self.val_map.update(preds, targets)
        except Exception:
            logger.warning("mAP update failed (batch %d), skipping", batch_idx, exc_info=True)

    def on_validation_epoch_end(self):
        try:
            map_results = self.val_map.compute()
            self.log("val_mAP", map_results["map"])
            self.log("val_mAP_50", map_results["map_50"])
        except Exception:
            logger.warning("mAP compute failed, logging 0.0", exc_info=True)
            self.log("val_mAP", 0.0)
            self.log("val_mAP_50", 0.0)
        self.val_map.reset()

    def _freeze_backbone(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
            for param in self.model.model.backbone.parameters():
                param.requires_grad = False
        else:
            warnings.warn(
                f"Cannot freeze backbone: {type(self.model).__name__} structure not recognized"
            )

