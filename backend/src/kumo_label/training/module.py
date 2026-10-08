import warnings

import torch
from transformers import AutoModelForImageClassification
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score,
    MulticlassAUROC, MulticlassPrecision, MulticlassRecall,
    MulticlassConfusionMatrix,
)

from kumo_label.training.base_module import BaseKumoModule


class KumoClassifier(BaseKumoModule):
    def __init__(self, model_id: str, num_classes: int, lr: float, epochs: int,
                 warmup_epochs: int = 0, optimizer: str = "adamw", scheduler: str = "cosine",
                 weight_decay: float = 0.01, freeze_backbone: bool = False,
                 class_names: list[str] = None,
                 class_balance_strategy: str = "none", class_counts: list[int] = None):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = class_names or []
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        if freeze_backbone:
            self._freeze_backbone()
        self.use_weighted_loss = False
        if class_balance_strategy == "weighted_loss" and class_counts:
            total = sum(class_counts)
            weights = torch.tensor(
                [total / (len(class_counts) * c) if c > 0 else 0.0 for c in class_counts],
                dtype=torch.float32,
            )
            self.register_buffer("class_weights", weights)
            self.use_weighted_loss = True
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_per_class_accuracy = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def _compute_loss(self, batch):
        if self.use_weighted_loss:
            outputs = self.model(pixel_values=batch["pixel_values"])
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, batch["labels"], weight=self.class_weights
            )
        else:
            outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss
        return outputs, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, loss = self._compute_loss(batch)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(outputs.logits, dim=-1)
        labels = batch["labels"]
        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_auroc.update(probs, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_per_class_accuracy.update(preds, labels)
        self.val_confusion_matrix.update(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_accuracy.compute())
        self.log("val_f1", self.val_f1.compute())
        self.log("val_auroc", self.val_auroc.compute())
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        # Store per-class and confusion matrix for TrackioCallback
        self._per_class_accuracy = self.val_per_class_accuracy.compute()
        self._confusion_matrix = self.val_confusion_matrix.compute()
        for m in [self.val_accuracy, self.val_f1, self.val_auroc,
                  self.val_precision, self.val_recall,
                  self.val_per_class_accuracy, self.val_confusion_matrix]:
            m.reset()

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

