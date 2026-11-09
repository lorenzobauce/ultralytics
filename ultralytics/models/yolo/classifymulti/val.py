from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.data import MultiLabelClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK


class MultiLabelClassificationValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a multi-label classification model.

    This validator handles classifiers with BCE, macro scores, and mean average precision.

    Attributes:
        targets (list[torch.Tensor]): Ground truth class labels.
        pred (list[torch.Tensor]): Model predictions.
        metrics (ClassifyMetrics): Object to calculate and store classification metrics.
        names (dict): Mapping of class indices to class names.
        nc (int): Number of classes.
        confusion_matrix (ConfusionMatrix): Matrix to evaluate model performance across classes.

    Methods:
        get_desc: Return a formatted string summarizing classification metrics.
        init_metrics: Initialize confusion matrix, class names, and tracking containers.
        preprocess: Preprocess input batch by moving data to device.
        update_metrics: Update running metrics with model predictions and batch targets.
        finalize_metrics: Finalize metrics including confusion matrix and processing speed.
        postprocess: Extract the primary prediction from model output.
        get_stats: Calculate and return a dictionary of metrics.
        build_dataset: Create a ClassificationDataset instance for validation.
        get_dataloader: Build and return a data loader for classification validation.
        print_results: Print evaluation metrics for the classification model.
        plot_val_samples: Plot validation image samples with their ground truth labels.
        plot_predictions: Plot images with their predicted class labels.

    Examples:
        >>> from ultralytics.models.yolo.classifymulti import MultiLabelClassificationValidator
        >>> args = dict(model="yolo26n-cls-multi.pt", data="imagenet10")
        >>> validator = MultiLabelClassificationValidator(args=args)
        >>> validator()

    """
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize ClassificationValidator with dataloader, save directory, and other parameters.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Arguments containing model and validation configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be called during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "classifymulti"
        self.pred: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.results: dict[str, float] = {}
        self.metrics = SimpleNamespace(keys=self.metric_keys, results_dict=self.results)
        if not 0 <= self.args.threshold <= 1:
            raise ValueError(f"multi_label_threshold must be between 0 and 1, got {self.args.threshold}.")

    def get_desc(self) -> str:
        """Return the validation progress columns."""
        return ("%22s" + "%11s" * 4) % ("classes", "P", "R", "F1", "mAP")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize class metadata and accumulated predictions."""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move images and multi-hot targets to the validation device."""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.quantize == 16 else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda", dtype=torch.float32)
        return batch

    def postprocess(self, preds: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extract probabilities from the multi-label classification head."""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Collect per-class probabilities and targets."""
        self.pred.append(preds.detach().float().cpu())
        self.targets.append(batch["cls"].detach().float().cpu())

    @staticmethod
    def _average_precision(target: torch.Tensor, score: torch.Tensor) -> float | None:
        """Compute interpolated average precision for one class."""
        positives = target.sum()
        if positives == 0:
            return None
        order = score.argsort(descending=True)
        target = target[order].float()
        true_positive = target.cumsum(0)
        false_positive = (1 - target).cumsum(0)
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        recall = torch.cat((recall.new_zeros(1), recall, recall.new_ones(1)))
        precision = torch.cat((precision.new_ones(1), precision, precision.new_zeros(1)))
        precision = torch.flip(torch.cummax(torch.flip(precision, (0,)), dim=0).values, (0,))
        changes = torch.where(recall[1:] != recall[:-1])[0]
        return float(((recall[changes + 1] - recall[changes]) * precision[changes + 1]).sum())

    def get_stats(self) -> dict[str, float]:
        """Compute BCE, macro precision/recall/F1, and mAP."""
        if not self.pred:
            raise RuntimeError("Validation produced no predictions; check that the validation split is not empty.")
        scores = torch.cat(self.pred)
        targets = torch.cat(self.targets) >= 0.5
        predicted = scores >= self.args.threshold

        true_positive = (predicted & targets).sum(0).float()
        false_positive = (predicted & ~targets).sum(0).float()
        false_negative = (~predicted & targets).sum(0).float()
        precision = true_positive / (true_positive + false_positive).clamp_min(1)
        recall = true_positive / (true_positive + false_negative).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        aps = [
            ap
            for class_id in range(self.nc)
            if (ap := self._average_precision(targets[:, class_id], scores[:, class_id])) is not None
        ]
        mean_ap = sum(aps) / len(aps) if aps else 0.0
        bce = F.binary_cross_entropy(scores.clamp(1e-7, 1 - 1e-7), targets.float()).item()
        macro_precision, macro_recall, macro_f1 = (float(metric.mean()) for metric in (precision, recall, f1))
        self.results = {
            "metrics/BCE": bce,
            "metrics/precision": macro_precision,
            "metrics/recall": macro_recall,
            "metrics/F1": macro_f1,
            "metrics/mAP": mean_ap,
            "fitness": (mean_ap + macro_f1) / 2,
        }
        self.metrics.results_dict = self.results
        return self.results

    def gather_stats(self) -> None:
        """Gather stored predictions and targets from all distributed workers."""
        if RANK == 0:
            gathered_pred = [None] * dist.get_world_size()
            gathered_targets = [None] * dist.get_world_size()
            dist.gather_object(self.pred, gathered_pred, dst=0)
            dist.gather_object(self.targets, gathered_targets, dst=0)
            self.pred = [item for rank_pred in gathered_pred for item in rank_pred]
            self.targets = [item for rank_targets in gathered_targets for item in rank_targets]
        elif RANK > 0:
            dist.gather_object(self.pred, None, dst=0)
            dist.gather_object(self.targets, None, dst=0)

    def build_dataset(self, img_path: str) -> MultiLabelClassificationDataset:
        """Build the multilabel dataset for validation."""
        return MultiLabelClassificationDataset(
            root=img_path,
            args=self.args,
            nc=self.data["nc"],
            augment=False,
            prefix=self.args.split,
            data_root=self.data["path"],
        )

    def get_dataloader(self, dataset_path: Path | str, batch_size: int) -> torch.utils.data.DataLoader:
        """Build the validation data loader."""
        dataset = self.build_dataset(str(dataset_path))
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    @property
    def metric_keys(self) -> list[str]:
        """Return the metric names used by validation."""
        return ["metrics/BCE", "metrics/precision", "metrics/recall", "metrics/F1", "metrics/mAP"]

    def print_results(self) -> None:
        """Log validation metrics."""
        LOGGER.info(
            "%22s%11.3g%11.3g%11.3g%11.3g",
            "all",
            self.results["metrics/precision"],
            self.results["metrics/recall"],
            self.results["metrics/F1"],
            self.results["metrics/mAP"],
        )