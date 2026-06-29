#!/usr/bin/env python3
"""
Script base para entrenamiento individual con logging robusto.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import create_model
from src.losses.tversky_loss import TverskyLoss
from src.datasets.wound_dataset import WoundDataset, get_training_transforms, get_default_transforms
from src.metrics.segmentation import calculate_metrics


def setup_logging(model_name: str):
    """Configura logging con archivo y consola."""
    log_dir = PROJECT_ROOT / "logs" / "screening"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{model_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log guardado en: {log_file}")
    return logger


def train_single_model(arch: str, encoder: str, name: str, epochs: int = 20):
    """Entrena un modelo individual con logging detallado."""
    
    logger = setup_logging(name)
    logger.info("="*80)
    logger.info(f"ENTRENAMIENTO: {name}")
    logger.info(f"Arquitectura: {arch} | Encoder: {encoder}")
    logger.info(f"Épocas: {epochs}")
    logger.info("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Cargar dataset
    logger.info("Cargando dataset...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_final.csv")
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'test']
    
    logger.info(f"  Train: {len(train_df)} imágenes")
    logger.info(f"  Val: {len(val_df)} imágenes")
    
    # Crear datasets
    IMAGE_SIZE = 384
    BATCH_SIZE = 8
    
    train_transforms = get_training_transforms((IMAGE_SIZE, IMAGE_SIZE))
    val_transforms = get_default_transforms((IMAGE_SIZE, IMAGE_SIZE))
    
    train_dataset = WoundDataset(
        image_paths=[Path(p) for p in train_df['image_path'].values],
        mask_paths=[Path(p) for p in train_df['mask_path'].values],
        transform=train_transforms
    )
    
    val_dataset = WoundDataset(
        image_paths=[Path(p) for p in val_df['image_path'].values],
        mask_paths=[Path(p) for p in val_df['mask_path'].values],
        transform=val_transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"  Batch size: {BATCH_SIZE} | Image size: {IMAGE_SIZE}")
    logger.info(f"  Batches por época: {len(train_loader)}")
    
    # Crear modelo
    logger.info("Creando modelo...")
    try:
        model = create_model(arch, encoder_name=encoder, pretrained=True).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Parámetros: {total_params:,} total")
        logger.info(f"  Parámetros entrenables: {trainable_params:,}")
    except Exception as e:
        logger.error(f"Error creando modelo: {e}")
        return None
    
    # Loss y optimizer
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info(f"  Loss: TverskyLoss (alpha=0.3, beta=0.7)")
    logger.info(f"  Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)")
    logger.info(f"  Scheduler: CosineAnnealingLR (T_max={epochs})")
    
    # Output directory
    output_dir = PROJECT_ROOT / "models" / "screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_dice = 0.0
    best_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    history = []
    
    start_time = time.time()
    logger.info("="*80)
    logger.info("INICIANDO ENTRENAMIENTO")
    logger.info("="*80)
    
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            logger.info(f"\nÉpoca {epoch + 1}/{epochs}")
            logger.info("-" * 80)
            
            # Train
            model.train()
            train_loss = 0.0
            train_dice_sum = 0.0
            train_iou_sum = 0.0
            train_batches = 0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Métricas
                preds = torch.sigmoid(outputs) > 0.5
                metrics = calculate_metrics(preds, masks)
                train_dice_sum += metrics['dice']
                train_iou_sum += metrics['iou']
                train_batches += 1
                
                # Log cada 50 batches
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}, Dice: {metrics['dice']:.4f}")
            
            train_loss /= train_batches
            train_dice = train_dice_sum / train_batches
            train_iou = train_iou_sum / train_batches
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_dice_sum = 0.0
            val_iou_sum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    preds = torch.sigmoid(outputs) > 0.5
                    metrics = calculate_metrics(preds, masks)
                    val_dice_sum += metrics['dice']
                    val_iou_sum += metrics['iou']
                    val_batches += 1
            
            val_loss /= val_batches
            val_dice = val_dice_sum / val_batches
            val_iou = val_iou_sum / val_batches
            
            # Scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log
            epoch_time = time.time() - epoch_start
            logger.info(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            logger.info(f"  LR: {current_lr:.6f} | Tiempo época: {epoch_time:.1f}s")
            
            # Guardar mejor modelo
            if val_dice > best_dice:
                best_dice = val_dice
                best_metrics = {'dice': val_dice, 'iou': val_iou, 'precision': 0.0, 'recall': 0.0, 'f1': val_dice}
                checkpoint_path = output_dir / f"{name}_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'config': {'arch': arch, 'encoder': encoder, 'name': name},
                }, checkpoint_path)
                logger.info(f"  [OK] Nuevo mejor modelo guardado (Dice: {best_dice:.4f})")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'lr': current_lr,
            })
            
            # Liberar memoria
            torch.cuda.empty_cache()
    
    except RuntimeError as e:
        logger.error(f"Entrenamiento interrumpido en época {epoch + 1}: {e}")
        logger.info(f"Mejor Dice alcanzado: {best_dice:.4f}")
    
    elapsed = time.time() - start_time
    
    logger.info("="*80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)
    logger.info(f"Tiempo total: {elapsed/60:.1f} minutos")
    logger.info(f"Mejor Dice: {best_dice:.4f}")
    logger.info(f"Mejor IoU: {best_metrics['iou']:.4f}")
    
    # Guardar en CSV
    results_csv = output_dir / "screening_results.csv"
    new_row = {
        'name': name,
        'architecture': arch,
        'encoder': encoder,
        'best_dice': best_dice,
        'best_iou': best_metrics['iou'],
        'best_precision': best_metrics['precision'],
        'best_recall': best_metrics['recall'],
        'best_f1': best_metrics['f1'],
        'params': total_params,
        'train_time_min': elapsed / 60,
    }
    
    if results_csv.exists():
        df_results = pd.read_csv(results_csv)
        # Actualizar o agregar
        if name in df_results['name'].values:
            df_results.loc[df_results['name'] == name, list(new_row.keys())] = list(new_row.values())
        else:
            df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_results = pd.DataFrame([new_row])
    
    df_results.to_csv(results_csv, index=False)
    logger.info(f"Resultados guardados en: {results_csv}")
    
    return new_row


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python script.py <arch> <encoder> <name>")
        print("Ejemplo: python script.py unet resnet101 UNet_ResNet101")
        sys.exit(1)
    
    arch, encoder, name = sys.argv[1], sys.argv[2], sys.argv[3]
    train_single_model(arch, encoder, name)
