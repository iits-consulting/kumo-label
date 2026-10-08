import warnings

import torch
from transformers import AutoModelForImageClassification
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelExactMatch, MultilabelAUROC,
    MultilabelPrecision, MultilabelRecall,
)

from kumo_label.training.base_module import BaseKumoModule


class KumoMultilabelClassifier(BaseKumoModule):
    def __init__(self, model_id: str, num_classes: int, lr: float, epochs: int,
                 warmup_epochs: int = 0, optimizer: str = "adamw", scheduler: str = "cosine",
                 weight_decay: float = 0.01, freeze_backbone: bool = False,
                 class_names: list[str] = None):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = class_names or []
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id, num_labels=num_classes,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        if freeze_backbone:
            self._freeze_backbone()
        self.val_f1 = MultilabelF1Score(num_labels=num_classes, average="macro", threshold=0.5)
        self.val_exact_match = MultilabelExactMatch(num_labels=num_classes, threshold=0.5)
        self.val_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.val_precision = MultilabelPrecision(num_labels=num_classes, average="macro", threshold=0.5)
        self.val_recall = MultilabelRecall(num_labels=num_classes, average="macro", threshold=0.5)
        self.val_per_class_precision = MultilabelPrecision(num_labels=num_classes, average=None, threshold=0.5)
        self.val_per_class_recall = MultilabelRecall(num_labels=num_classes, average=None, threshold=0.5)
        self.val_per_class_f1 = MultilabelF1Score(num_labels=num_classes, average=None, threshold=0.5)
        self._support = torch.zeros(num_classes, dtype=torch.long)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        probs = torch.sigmoid(outputs.logits)
        targets = batch["labels"].int()
        self.val_f1.update(probs, targets)
        self.val_exact_match.update(probs, targets)
        self.val_auroc.update(probs, targets)
        self.val_precision.update(probs, targets)
        self.val_recall.update(probs, targets)
        self.val_per_class_precision.update(probs, targets)
        self.val_per_class_recall.update(probs, targets)
        self.val_per_class_f1.update(probs, targets)
        self._support += targets.sum(dim=0).cpu().long()
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute())
        self.log("val_exact_match", self.val_exact_match.compute())
        self.log("val_auroc", self.val_auroc.compute())
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())

        per_class_precision = self.val_per_class_precision.compute()
        per_class_recall = self.val_per_class_recall.compute()
        per_class_f1 = self.val_per_class_f1.compute()
        self._per_class_metrics = {
            "classes": list(self.class_names),
            "precision": [float(v) for v in per_class_precision],
            "recall": [float(v) for v in per_class_recall],
            "f1": [float(v) for v in per_class_f1],
            "support": [int(v) for v in self._support],
        }

        for m in [self.val_f1, self.val_exact_match, self.val_auroc,
                  self.val_precision, self.val_recall,
                  self.val_per_class_precision, self.val_per_class_recall, self.val_per_class_f1]:
            m.reset()
        self._support.zero_()

    def _freeze_backbone(self):
        if not hasattr(self.model, "classifier"):
            warnings.warn(
                f"Cannot freeze: {type(self.model).__name__} has no 'classifier' attribute"
            )
            return
        classifier_params = set(id(p) for p in self.model.classifier.parameters())
        for param in self.model.parameters():
            if id(param) not in classifier_params:
                param.requires_grad = False
