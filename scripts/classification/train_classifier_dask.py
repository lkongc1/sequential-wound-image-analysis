#!/usr/bin/env python3
"""Entrenamiento distribuido del clasificador de heridas con Dask + PyTorch Lightning.

Usa Dask Distributed para paralelizar la carga y augmentación de datos en CPU,
mientras PyTorch Lightning entrena el modelo (EfficientNet-B3) en el dispositivo
disponible (GPU/CPU). Cada worker de Dask carga y aplica transforms de forma
independiente, evitando el GIL de Python en Windows.

Uso:
    python scripts/classification/train_classifier_dask.py
    python scripts/classification/train_classifier_dask.py --dask-workers 6
    python scripts/classification/train_classifier_dask.py --dask-scheduler tcp://192.168.1.100:8786
    python scripts/classification/train_classifier_dask.py --epochs 30 --lr 5e-5
    python scripts/classification/train_classifier_dask.py --dry-run
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ClassificationConfig
from src.datasets.classification_dataset import ClassificationDataset
from src.models.factory import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ================================================================== #
# Funciones de worker Dask (nivel módulo — requerido por pickle)
# ================================================================== #

_worker_dataset: Optional[ClassificationDataset] = None


def _worker_init(
    csv_path: str,
    class_names: List[str],
    image_size: Tuple[int, int],
    use_mask: bool,
    augment: bool = True,
) -> None:
    """Inicializa el dataset de clasificación en cada worker Dask.

    Se llama una vez por worker vía ``client.run()`` para evitar
    re-instantiar el dataset en cada tarea.
    """
    global _worker_dataset
    _worker_dataset = ClassificationDataset(
        csv_path=Path(csv_path),
        class_names=list(class_names),
        image_size=image_size,
        use_mask=use_mask,
        augment=augment,
    )


def _compute_batch(batch_indices: List[int]) -> Tuple[Tensor, Tensor]:
    """Computa un batch de (imágenes, etiquetas) en un worker Dask.

    Carga y transforma las imágenes correspondientes a *batch_indices*
    usando el dataset global del worker.

    Args:
        batch_indices: Lista de índices a cargar desde el dataset.

    Returns:
        Tuple de (images, labels) donde images es (B, C, H, W) y
        labels es (B,) con dtype int64.
    """
    global _worker_dataset
    if _worker_dataset is None:
        raise RuntimeError("Dataset no inicializado en worker Dask")
    images: list[Tensor] = []
    labels: list[int] = []
    for idx in batch_indices:
        img, lbl = _worker_dataset[idx]
        images.append(img)
        labels.append(lbl)
    return torch.stack(images), torch.tensor(labels, dtype=torch.int64)


# ================================================================== #
# Dataset iterable respaldado por Dask
# ================================================================== #


class DaskAugmentedDataset(torch.utils.data.IterableDataset):
    """Dataset iterable que precomputa batches vía workers Dask.

    Cada worker Dask carga imágenes del CSV de entrenamiento, aplica
    RandAugment vía albumentations y devuelve tensores listos para
    el modelo. El orden de los batches es no determinístico porque
    se usa ``as_completed`` para máximo throughput.

    Args:
        csv_path: Ruta al CSV de entrenamiento.
        class_names: Nombres ordenados de las clases.
        image_size: Tamaño (alto, ancho) de las imágenes.
        use_mask: Si True, canal extra de máscara binaria.
        batch_size: Tamaño de batch.
        client: Cliente Dask distribuido.
    """

    def __init__(
        self,
        csv_path: Path,
        class_names: List[str],
        image_size: Tuple[int, int],
        use_mask: bool,
        batch_size: int,
        client,
    ) -> None:
        self.csv_path = csv_path
        self.class_names = list(class_names)
        self.image_size = image_size
        self.use_mask = use_mask
        self.batch_size = batch_size
        self.client = client

        n_workers = len(client.scheduler_info()["workers"])
        logger.info(
            "DaskAugmentedDataset: %d workers Dask, batch_size=%d",
            n_workers, batch_size,
        )

        # Inicializar dataset en cada worker
        client.run(
            _worker_init,
            str(csv_path),
            list(class_names),
            tuple(image_size),
            use_mask,
            True,  # augment
        )

    def _n_samples(self) -> int:
        df = pd.read_csv(self.csv_path)
        return len(df)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        from dask.distributed import as_completed

        n = self._n_samples()
        indices = list(range(n))
        random.shuffle(indices)

        # Armar lista de batches de índices
        batch_idx_lists = [
            indices[i : i + self.batch_size]
            for i in range(0, n, self.batch_size)
        ]

        # Descartar último batch si no está completo (drop_last)
        if len(batch_idx_lists[-1]) < self.batch_size:
            batch_idx_lists.pop()

        # Enviar a workers Dask
        futures = self.client.map(_compute_batch, batch_idx_lists)

        # Yield a medida que completan (orden no determinístico)
        for future in as_completed(futures):
            images, labels = future.result()
            yield images, labels


# ================================================================== #
# DataModule con Dask
# ================================================================== #


class DaskWoundDataModule(pl.LightningDataModule):
    """LightningDataModule que usa Dask para el dataset de entrenamiento.

    El dataset de validación no usa Dask (no tiene augmentación costosa).
    """

    def __init__(self, config: ClassificationConfig, dask_client) -> None:
        super().__init__()
        self.cfg = config
        self.dask_client = dask_client
        self.val_dataset: Optional[ClassificationDataset] = None
        self.class_weights: Optional[Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # Validación: dataset estándar sin augment
            self.val_dataset = ClassificationDataset(
                csv_path=self.cfg.val_csv,
                class_names=self.cfg.class_names,
                image_size=self.cfg.image_size,
                use_mask=self.cfg.use_mask,
                augment=False,
            )
            # Pesos de clase desde CSV de entrenamiento
            self._compute_class_weights()

    def _compute_class_weights(self) -> None:
        df = pd.read_csv(self.cfg.train_csv)
        labels = df["label"]
        counts = labels.value_counts().reindex(self.cfg.class_names, fill_value=0)
        weights = 1.0 / (counts.values + 1e-6)
        weights = weights / weights.sum() * len(self.cfg.class_names)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(
            "Class weights: %s",
            {cls: f"{w:.3f}" for cls, w in zip(self.cfg.class_names, weights)},
        )

    def train_dataloader(self) -> DataLoader:
        ds = DaskAugmentedDataset(
            csv_path=self.cfg.train_csv,
            class_names=self.cfg.class_names,
            image_size=self.cfg.image_size,
            use_mask=self.cfg.use_mask,
            batch_size=self.cfg.batch_size,
            client=self.dask_client,
        )
        return DataLoader(
            ds,
            batch_size=None,  # batches ya vienen armados desde Dask
            num_workers=0,    # Dask maneja la paralelización
            pin_memory=(DEVICE == "cuda"),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=(DEVICE == "cuda"),
        )


# ================================================================== #
# LightningModule (replicado de train_classifier.py por autonomía)
# ================================================================== #


class WoundClassifierLightning(pl.LightningModule):
    """LightningModule que envuelve WoundClassifier para entrenamiento.

    Features:
    - Class-weighted CrossEntropyLoss con label smoothing
    - MixUp y CutMix aplicados probabilísticamente en training_step
    - Per-class F1 y matriz de confusión en TensorBoard
    - Descongelamiento gradual del backbone
    """

    def __init__(
        self,
        config: ClassificationConfig,
        class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.cfg = config

        self.model = create_model(
            "wound_classifier",
            num_classes=config.num_classes,
            pretrained=True,
            freeze_backbone=config.freeze_backbone,
            dropout=config.dropout,
        )
        self.class_weights = class_weights

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE) if class_weights is not None else None,
            label_smoothing=config.label_smoothing,
        )

        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        if self.cfg.mixup_alpha > 0 and self.cfg.cutmix_alpha > 0:
            if torch.rand(1).item() < 0.5:
                images, labels_a, labels_b, lam = self._mixup(images, labels)
                logits = self.model(images)
                loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
            elif torch.rand(1).item() < 0.5:
                images, labels_a, labels_b, lam = self._cutmix(images, labels)
                logits = self.model(images)
                loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
        else:
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _mixup(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
        lam = max(lam, 1 - lam)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        mixed = lam * images + (1 - lam) * images[index]
        return mixed, labels, labels[index], lam

    def _cutmix(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        lam = np.random.beta(self.cfg.cutmix_alpha, self.cfg.cutmix_alpha)
        lam = max(lam, 1 - lam)
        batch_size, _, H, W = images.shape
        index = torch.randperm(batch_size, device=images.device)
        cut_ratio = np.sqrt(1 - lam)
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam_adj = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
        return mixed, labels, labels[index], lam_adj

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        self._val_preds.append(logits.detach().cpu())
        self._val_targets.append(labels.detach().cpu())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        from torchmetrics.functional import accuracy, f1_score, confusion_matrix

        all_preds = torch.cat(self._val_preds, dim=0)
        all_targets = torch.cat(self._val_targets, dim=0)
        pred_labels = torch.argmax(all_preds, dim=1)

        macro_f1 = f1_score(
            pred_labels, all_targets,
            task="multiclass", num_classes=self.cfg.num_classes, average="macro",
        )
        self.log("val/macro_f1", macro_f1, on_epoch=True, prog_bar=True)

        per_class = f1_score(
            pred_labels, all_targets,
            task="multiclass", num_classes=self.cfg.num_classes, average=None,
        )
        for i, name in enumerate(self.cfg.class_names):
            self.log(f"val/f1_{name}", per_class[i], on_epoch=True)

        acc = accuracy(pred_labels, all_targets, task="multiclass", num_classes=self.cfg.num_classes)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

        if self.logger and hasattr(self.logger, "experiment"):
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = confusion_matrix(pred_labels, all_targets, task="multiclass", num_classes=self.cfg.num_classes)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm.numpy().astype(int),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=self.cfg.class_names,
                yticklabels=self.cfg.class_names,
                ax=ax,
            )
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            ax.set_title(f"Matriz de Confusión — Época {self.current_epoch}")
            self.logger.experiment.add_figure("val/confusion_matrix", fig, self.current_epoch)
            plt.close(fig)

        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.learning_rate, weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.max_epochs, eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_epoch_start(self) -> None:
        if self.cfg.freeze_backbone_epochs <= 0:
            return
        if self.current_epoch == 0:
            self._set_backbone_requires_grad(False)
            logger.info("Época %d: backbone congelado", self.current_epoch)
        elif self.current_epoch == self.cfg.freeze_backbone_epochs:
            self._set_backbone_requires_grad(True)
            logger.info("Época %d: backbone descongelado", self.current_epoch)

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        for param in self.model.backbone.parameters():
            param.requires_grad = requires_grad


# ================================================================== #
# Callbacks
# ================================================================== #


def build_callbacks(config: ClassificationConfig, output_dir: Path) -> list:
    early_stop = EarlyStopping(
        monitor="val/macro_f1", mode="max",
        patience=config.patience, verbose=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:02d}-{val/macro_f1:.3f}",
        monitor="val/macro_f1", mode="max",
        save_top_k=1, save_last=True, verbose=True,
    )
    return [early_stop, checkpoint]


# ================================================================== #
# Dask cluster helpers
# ================================================================== #

_cluster = None
_client = None


def _shutdown_dask() -> None:
    """Cierre graceful del cluster Dask (registrado vía atexit)."""
    global _cluster, _client
    try:
        if _client is not None:
            _client.close()
            logger.info("Cliente Dask cerrado")
    except Exception:
        pass
    try:
        if _cluster is not None:
            _cluster.close()
            logger.info("Cluster Dask cerrado")
    except Exception:
        pass


def _log_cluster_info(client) -> None:
    """Registra información del cluster Dask."""
    info = client.scheduler_info()
    workers = info.get("workers", {})
    n_workers = len(workers)
    total_cores = sum(w.get("nthreads", 0) for w in workers.values())
    total_mem = sum(w.get("memory_limit", 0) for w in workers.values())
    logger.info("=== Cluster Dask ===")
    logger.info("  Workers: %d", n_workers)
    logger.info("  Núcleos totales: %d", total_cores)
    logger.info("  Memoria total: %.1f GB", total_mem / (1024**3))
    for addr, w in workers.items():
        logger.info(
            "    %s — cores=%d, mem=%.1f GB",
            addr, w.get("nthreads", 0),
            w.get("memory_limit", 0) / (1024**3),
        )


def setup_dask_cluster(
    n_workers: Optional[int] = None,
    scheduler_addr: Optional[str] = None,
    threads_per_worker: int = 1,
):
    """Configura el cluster Dask (local o remoto).

    Args:
        n_workers: Workers para LocalCluster. Si es None, auto-detecta
            CPU count menos 1 (mínimo 1).
        scheduler_addr: Dirección del scheduler externo (ej.
            ``tcp://192.168.1.100:8786``). Si se provee, ignora
            *n_workers* y se conecta al cluster existente.
        threads_per_worker: Hilos por worker (default 1 para CPU-bound).

    Returns:
        Tuple de (cluster, client). Si *scheduler_addr* se provee,
        *cluster* es None.
    """
    global _cluster, _client

    if scheduler_addr:
        from dask.distributed import Client
        _cluster = None
        _client = Client(scheduler_addr)
        logger.info("Conectado a scheduler Dask: %s", scheduler_addr)
    else:
        from dask.distributed import Client, LocalCluster

        if n_workers is None:
            cpu_count = os.cpu_count() or 4
            n_workers = max(1, cpu_count - 1)
            logger.info(
                "Auto-detección: %d CPUs → %d workers Dask (1 core libre)",
                cpu_count, n_workers,
            )

        logger.info(
            "Creando LocalCluster: processes=True, n_workers=%d, threads_per_worker=%d",
            n_workers, threads_per_worker,
        )

        _cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,          # procesos separados → evita GIL
            silence_logs=logging.WARNING,
        )
        _client = Client(_cluster)

    atexit.register(_shutdown_dask)
    _log_cluster_info(_client)
    return _cluster, _client


# ================================================================== #
# CLI
# ================================================================== #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento distribuido del clasificador de heridas (Dask + Lightning)"
    )
    # --- Mismos args que train_classifier.py ---
    parser.add_argument(
        "--csv-prefix", type=Path, default=Path("data-clasificador"),
        help="Directorio con train.csv y val.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/classifier/distributed"),
        help="Directorio de salida para checkpoints y logs",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Épocas máximas")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Número de clases (default: auto-detectar del CSV)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prueba de 1 época")
    parser.add_argument("--no-freeze", action="store_true", help="No congelar backbone")
    parser.add_argument(
        "--freeze-epochs", type=int, default=3,
        help="Épocas con backbone congelado (default 3)",
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Entrada 3 canales (sin máscara)",
    )
    parser.add_argument(
        "--accumulate", type=int, default=2,
        help="Pasos de gradient accumulation",
    )
    # --- Args específicos de Dask ---
    parser.add_argument(
        "--dask-workers", type=int, default=None,
        help="Workers Dask para LocalCluster (default: CPUs - 1)",
    )
    parser.add_argument(
        "--dask-scheduler", type=str, default=None,
        help="Dirección del scheduler Dask externo (ej. tcp://host:8786)",
    )
    parser.add_argument(
        "--dask-threads", type=int, default=1,
        help="Hilos por worker Dask (default 1, CPU-bound)",
    )
    return parser.parse_args()


def _resolve_num_classes(config: ClassificationConfig) -> int:
    train_csv = config.train_csv
    if not train_csv.exists():
        return config.num_classes
    df = pd.read_csv(train_csv)
    actual_labels = set(df["label"].unique())
    known = [c for c in config.class_names if c in actual_labels]
    if len(known) < config.num_classes:
        logger.warning(
            "CSV tiene %d clases (%s), config dice %d. Usando %d.",
            len(known), known, config.num_classes, len(known),
        )
        config.class_names = known
        config.num_classes = len(known)
    return config.num_classes


# ================================================================== #
# Main
# ================================================================== #


def main() -> None:
    args = parse_args()

    # --- Configuración Dask ---
    cluster, client = setup_dask_cluster(
        n_workers=args.dask_workers,
        scheduler_addr=args.dask_scheduler,
        threads_per_worker=args.dask_threads,
    )

    # --- Configuración del modelo ---
    config = ClassificationConfig(
        train_csv=args.csv_prefix / "train.csv",
        val_csv=args.csv_prefix / "val.csv",
        test_csv=args.csv_prefix / "test.csv",
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        accumulation_steps=args.accumulate,
        use_mask=not args.no_mask,
        freeze_backbone_epochs=0 if args.no_freeze else args.freeze_epochs,
    )

    num_classes = _resolve_num_classes(config)
    config.num_classes = num_classes
    logger.info("Entrenando con %d clases: %s", num_classes, config.class_names)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # --- DataModule ---
    dm = DaskWoundDataModule(config, dask_client=client)
    dm.setup("fit")

    # --- Modelo ---
    model = WoundClassifierLightning(
        config=config,
        class_weights=dm.class_weights,
    )

    # --- Callbacks ---
    callbacks = build_callbacks(config, config.output_dir)
    if args.dry_run:
        callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]

    # --- Logger ---
    tb_logger = TensorBoardLogger(
        save_dir=str(config.output_dir / "logs"),
        name="wound_classifier_dask",
    )

    # --- Trainer ---
    effective_epochs = 1 if args.dry_run else config.max_epochs
    strategy = "ddp_spawn" if torch.cuda.device_count() > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=effective_epochs,
        accelerator="auto",
        devices=1,
        strategy=strategy,
        precision="16-mixed",
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulation_steps,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    logger.info("Iniciando entrenamiento distribuido...")
    logger.info("  Dispositivo: %s", DEVICE)
    logger.info("  Estrategia: %s", strategy)
    logger.info("  Épocas: %d", effective_epochs)
    logger.info("  Batch size: %d (accum %d → efectivo %d)",
                config.batch_size, config.accumulation_steps,
                config.batch_size * config.accumulation_steps)
    logger.info("  Learning rate: %.2e", config.learning_rate)
    logger.info("  MixUp α: %.1f  CutMix α: %.1f", config.mixup_alpha, config.cutmix_alpha)
    logger.info("  Label smoothing: %.1f  Dropout: %.1f",
                config.label_smoothing, config.dropout)
    logger.info("  Freeze backbone: %d epochs", config.freeze_backbone_epochs)

    trainer.fit(model, datamodule=dm)

    # --- Reporte final ---
    if not args.dry_run:
        best_f1 = trainer.callback_metrics.get("val/macro_f1", 0)
        logger.info("Entrenamiento completado. Mejor macro-F1: %.4f", best_f1)
        if trainer.checkpoint_callback:
            best_path = trainer.checkpoint_callback.best_model_path
            if best_path:
                logger.info("Mejor checkpoint: %s", best_path)
                import shutil
                best_fixed = config.output_dir / "best.pth"
                shutil.copy(best_path, str(best_fixed))
                logger.info("Copiado a: %s", best_fixed)
    else:
        logger.info("Dry-run completado — verificar que loss no sea NaN")


if __name__ == "__main__":
    # Windows: forzar método spawn para multiprocessing
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # ya fue seteado

    main()
