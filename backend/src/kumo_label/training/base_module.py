import warnings

import lightning as L
import torch


class BaseKumoModule(L.LightningModule):
    """Shared functionality for all Kumo training modules.

    Provides configure_optimizers and load_backbone_from_checkpoint so that
    task-specific modules only need to implement model construction, forward,
    training_step, and validation_step.
    """

    def load_backbone_from_checkpoint(self, checkpoint_path: str):
        """Load backbone weights from a checkpoint, skipping mismatched layers.

        When num_classes differs, the classifier head has a different shape and
        is automatically skipped.  All other (backbone) weights are transferred.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        current_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)

        if skipped:
            warnings.warn(
                f"Skipped {len(skipped)} params with mismatched shapes: "
                f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}"
            )
        self.load_state_dict(filtered, strict=False)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        wd = self.hparams.weight_decay
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=0.9, weight_decay=wd)
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=wd)

        if self.hparams.scheduler == "none":
            return optimizer
        elif self.hparams.scheduler == "linear":
            sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01,
                total_iters=self.hparams.epochs,
            )
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.epochs,
            )

        if self.hparams.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=self.hparams.warmup_epochs,
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, sched],
                milestones=[self.hparams.warmup_epochs],
            )

        return [optimizer], [sched]
