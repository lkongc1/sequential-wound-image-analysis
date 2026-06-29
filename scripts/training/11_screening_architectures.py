#!/usr/bin/env python3
"""
Screening comparativo de arquitecturas y encoders.

Entrena 12 combinaciones con 20 épocas cada una para identificar
los mejores candidatos para entrenamiento completo.

Combinaciones:
- DeepLabV3, DeepLabV3+, U-Net, FPN
- ResNeXt-50, ResNet-101, EfficientNet-B3, SegFormer (mit_b2)
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Agregar root al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import create_model
from src.losses.tversky_loss import TverskyLoss
from src.datasets.wound_dataset import WoundDataset, get_training_transforms, get_default_transforms
from src.metrics.segmentation import calculate_metrics


# Configuración de screening
SCREENING_CONFIGS = [
    # ResNeXt-50
    {"arch": "deeplabv3", "encoder": "resnext50_32x4d", "name": "DeepLabV3_ResNeXt50"},
    {"arch": "deeplabv3plus", "encoder": "resnext50_32x4d", "name": "DeepLabV3Plus_ResNeXt50"},
    {"arch": "unet", "encoder": "resnext50_32x4d", "name": "UNet_ResNeXt50"},
    {"arch": "fpn", "encoder": "resnext50_32x4d", "name": "FPN_ResNeXt50"},
    
    # ResNet-101
    {"arch": "deeplabv3plus", "encoder": "resnet101", "name": "DeepLabV3Plus_ResNet101"},
    {"arch": "unet", "encoder": "resnet101", "name": "UNet_ResNet101"},
    {"arch": "fpn", "encoder": "resnet101", "name": "FPN_ResNet101"},
    
    # EfficientNet-B3
    {"arch": "deeplabv3plus", "encoder": "efficientnet-b3", "name": "DeepLabV3Plus_EfficientNetB3"},
    {"arch": "unet", "encoder": "efficientnet-b3", "name": "UNet_EfficientNetB3"},
    {"arch": "fpn", "encoder": "efficientnet-b3", "name": "FPN_EfficientNetB3"},
    
    # SegFormer (mit_b2)
    {"arch": "unet", "encoder": "mit_b2", "name": "UNet_SegFormer"},
    {"arch": "fpn", "encoder": "mit_b2", "name": "FPN_SegFormer"},
]

# Hiperparámetros de screening
SCREENING_EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SIZE = 384
LEARNING_RATE = 1e-4


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entrena una época."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Métricas
        preds = torch.sigmoid(outputs) > 0.5
        metrics = calculate_metrics(preds, masks)
        
        all_preds.append(metrics['dice'])
        all_targets.append(metrics['iou'])
        
        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    
    # Calcular métricas promedio
    avg_dice = sum(all_preds) / len(all_preds) if all_preds else 0.0
    avg_iou = sum(all_targets) / len(all_targets) if all_targets else 0.0
    
    metrics = {
        'dice': avg_dice,
        'iou': avg_iou,
        'precision': 0.0,  # Simplificado para screening
        'recall': 0.0,
        'f1': avg_dice,
    }
    
    return avg_loss, metrics


def validate(model, loader, criterion, device):
    """Valida el modelo."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            metrics = calculate_metrics(preds, masks)
            
            all_preds.append(metrics['dice'])
            all_targets.append(metrics['iou'])
    
    avg_loss = total_loss / len(loader)
    
    # Calcular métricas promedio
    avg_dice = sum(all_preds) / len(all_preds) if all_preds else 0.0
    avg_iou = sum(all_targets) / len(all_targets) if all_targets else 0.0
    
    metrics = {
        'dice': avg_dice,
        'iou': avg_iou,
        'precision': 0.0,  # Simplificado para screening
        'recall': 0.0,
        'f1': avg_dice,
    }
    
    return avg_loss, metrics


def train_model(config, train_loader, val_loader, device, output_dir):
    """Entrena un modelo con la configuración dada."""
    print(f"\n{'='*80}")
    print(f"Entrenando: {config['name']}")
    print(f"Arquitectura: {config['arch']} | Encoder: {config['encoder']}")
    print(f"{'='*80}")
    
    # Crear modelo
    try:
        model = create_model(
            config['arch'],
            encoder_name=config['encoder'],
            pretrained=True
        ).to(device)
    except Exception as e:
        print(f"  ERROR creando modelo: {e}")
        return None
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros: {total_params:,} total, {trainable_params:,} entrenables")
    
    # Loss y optimizer
    criterion = TverskyLoss(alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCREENING_EPOCHS)
    
    # Training loop
    best_dice = 0.0
    best_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    history = []
    
    start_time = time.time()
    
    try:
        for epoch in range(SCREENING_EPOCHS):
            print(f"\n  Época {epoch + 1}/{SCREENING_EPOCHS}")
            
            # Train
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            
            # Scheduler
            scheduler.step()
            
            # Log
            print(f"    Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"    Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            # Guardar mejor modelo
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                best_metrics = val_metrics.copy()
                checkpoint_path = output_dir / f"{config['name']}_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'config': config,
                }, checkpoint_path)
                print(f"    [OK] Nuevo mejor modelo guardado (Dice: {best_dice:.4f})")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_metrics['dice'],
                'val_loss': val_loss,
                'val_dice': val_metrics['dice'],
            })
            
            # Liberar memoria periódicamente
            torch.cuda.empty_cache()
    
    except RuntimeError as e:
        print(f"\n  [ERROR] Entrenamiento interrumpido en época {epoch + 1}: {e}")
        print(f"  Mejor Dice alcanzado: {best_dice:.4f}")
        # Continuar con el mejor resultado obtenido
    
    elapsed = time.time() - start_time
    
    print(f"\n  Entrenamiento completado en {elapsed/60:.1f} minutos")
    print(f"  Mejor Dice: {best_dice:.4f}")
    
    return {
        'name': config['name'],
        'architecture': config['arch'],
        'encoder': config['encoder'],
        'best_dice': best_dice,
        'best_iou': best_metrics['iou'],
        'best_precision': best_metrics['precision'],
        'best_recall': best_metrics['recall'],
        'best_f1': best_metrics['f1'],
        'params': total_params,
        'train_time_min': elapsed / 60,
    }


def main():
    print("="*80)
    print("SCREENING COMPARATIVO DE ARQUITECTURAS")
    print("="*80)
    print(f"Dataset: 3,355 imágenes (2,748 originales + 607 CO2Wounds-V2)")
    print(f"Screening: {len(SCREENING_CONFIGS)} combinaciones × {SCREENING_EPOCHS} épocas")
    print(f"Batch size: {BATCH_SIZE} | Image size: {IMAGE_SIZE}")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar dataset
    print("\nCargando dataset...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_final.csv")
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'test']  # Usamos test como validación para screening
    
    print(f"  Train: {len(train_df)} imágenes")
    print(f"  Val: {len(val_df)} imágenes")
    
    # Crear datasets
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
    
    # DataLoaders
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
    
    # Output directory
    output_dir = PROJECT_ROOT / "models" / "screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar resultados existentes para reanudar
    results_csv = output_dir / "screening_results.csv"
    completed_models = set()
    results = []
    
    if results_csv.exists():
        existing_df = pd.read_csv(results_csv)
        for _, row in existing_df.iterrows():
            # Solo considerar completado si tiene más de 15 épocas (casi completo)
            checkpoint_path = output_dir / f"{row['name']}_best.pth"
            if checkpoint_path.exists():
                try:
                    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    if ckpt['epoch'] >= 15:  # Considerar completado si tiene 15+ épocas
                        completed_models.add(row['name'])
                        results.append(row.to_dict())
                        print(f"[SKIP] {row['name']} ya completado (Dice: {row['best_dice']:.4f})")
                except Exception as e:
                    print(f"[WARN] Error leyendo checkpoint de {row['name']}: {e}")
    
    print(f"\nModelos completados: {len(completed_models)}/{len(SCREENING_CONFIGS)}")
    print(f"Modelos restantes: {len(SCREENING_CONFIGS) - len(completed_models)}")
    
    # Entrenar todas las configuraciones
    for config in SCREENING_CONFIGS:
        if config['name'] in completed_models:
            continue
            
        result = train_model(config, train_loader, val_loader, device, output_dir)
        if result:
            results.append(result)
            
            # Guardar resultados parciales
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_dir / "screening_results.csv", index=False)
            print(f"\n  Resultados guardados en screening_results.csv")
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE SCREENING")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_dice', ascending=False)
    
    print("\nRanking por Dice:")
    for idx, row in results_df.iterrows():
        print(f"  {row['name']:40s} | Dice: {row['best_dice']:.4f} | IoU: {row['best_iou']:.4f}")
    
    # Guardar resultados finales
    results_df.to_csv(output_dir / "screening_results_final.csv", index=False)
    print(f"\nResultados finales guardados en: {output_dir / 'screening_results_final.csv'}")
    
    # Recomendar top 3
    print("\n" + "="*80)
    print("RECOMENDACIÓN: TOP 3 PARA ENTRENAMIENTO COMPLETO")
    print("="*80)
    
    top3 = results_df.head(3)
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{i}. {row['name']}")
        print(f"   Dice: {row['best_dice']:.4f} | IoU: {row['best_iou']:.4f}")
        print(f"   Parámetros: {row['params']:,}")
        print(f"   Tiempo de screening: {row['train_time_min']:.1f} min")
    
    print("\n" + "="*80)
    print("SCREENING COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
