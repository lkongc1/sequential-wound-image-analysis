#!/usr/bin/env python3
"""Integra CO2Wounds-V2 y prepara Yasin para clasificación.

CO2Wounds-V2: 607 máscaras binarias → integrar al dataset de segmentación
Yasin: 431 imágenes clasificadas → guardar para fase de clasificación

Uso:
    python scripts/10_integrate_new_datasets.py
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def integrate_co2wounds():
    """Integra CO2Wounds-V2 al dataset de segmentación."""
    print("=" * 80)
    print("INTEGRANDO CO2Wounds-V2")
    print("=" * 80)
    
    # Encontrar directorio principal
    base = PROJECT_ROOT / "data" / "raw" / "co2wounds_v2"
    main_dirs = list(base.glob("CO2Wounds*"))
    if not main_dirs:
        print("Error: No se encontró el directorio CO2Wounds")
        return False
    
    main = main_dirs[0]
    imgs_dir = main / "imgs"
    masks_dir = main / "masks"
    
    # Listar imágenes y máscaras
    images = list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpg"))
    masks = list(masks_dir.glob("*.png"))
    
    print(f"Imágenes: {len(images)}")
    print(f"Máscaras: {len(masks)}")
    
    # Matching por nombre
    matched = []
    for mask_path in masks:
        img_name = mask_path.stem
        # Buscar imagen correspondiente
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = imgs_dir / f"{img_name}{ext}"
            if img_path.exists():
                matched.append((img_path, mask_path))
                break
    
    print(f"Pares imagen-máscara: {len(matched)}")
    
    if len(matched) == 0:
        print("Error: No se encontraron pares imagen-máscara")
        return False
    
    # Crear directorio de salida
    output_dir = PROJECT_ROOT / "data" / "raw" / "co2wounds_processed"
    output_imgs = output_dir / "images"
    output_masks = output_dir / "masks"
    output_imgs.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    
    # Procesar y validar
    valid_pairs = []
    
    for img_path, mask_path in tqdm(matched, desc="Procesando"):
        # Cargar imagen
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Cargar máscara
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Validar máscara binaria
        unique_vals = np.unique(mask)
        if len(unique_vals) > 2:
            # Convertir a binaria
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            # Asegurar que sea 0 y 255
            mask = (mask > 0).astype(np.uint8) * 255
        
        # Verificar que tenga contenido
        if mask.sum() == 0:
            continue
        
        # Redimensionar a 512x512
        img_resized = cv2.resize(img, (512, 512))
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Guardar
        out_name = f"co2wounds_{img_path.stem}"
        cv2.imwrite(str(output_imgs / f"{out_name}.png"), img_resized)
        cv2.imwrite(str(output_masks / f"{out_name}_mask.png"), mask_resized)
        
        valid_pairs.append({
            "filename": f"{out_name}.png",
            "source": "co2wounds_v2",
            "image_path": str((output_imgs / f"{out_name}.png").resolve()),
            "mask_path": str((output_masks / f"{out_name}_mask.png").resolve()),
        })
    
    print(f"\nPares válidos procesados: {len(valid_pairs)}")
    
    # Crear DataFrame
    df_new = pd.DataFrame(valid_pairs)
    
    # Añadir métricas básicas
    df_new["wound_percentage"] = 0.0
    df_new["brightness_mean"] = 0.0
    df_new["brightness_std"] = 0.0
    df_new["contrast_rms"] = 0.0
    df_new["mask_area_pixels"] = 0
    df_new["mask_edge_density"] = 0.0
    df_new["is_empty"] = False
    df_new["is_outlier"] = False
    df_new["outlier_reason"] = ""
    df_new["review_status"] = "auto_integrated"
    
    # Split train/test (80/20)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(df_new)), test_size=0.2, random_state=42
    )
    df_new.loc[train_idx, "split"] = "train"
    df_new.loc[test_idx, "split"] = "test"
    
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
    
    # Guardar CSV temporal
    temp_csv = PROJECT_ROOT / "data" / "processed" / "co2wounds_v2_integrated.csv"
    df_new.to_csv(temp_csv, index=False)
    print(f"Guardado en: {temp_csv}")
    
    return True


def prepare_yasin_for_classification():
    """Prepara el dataset Yasin para clasificación por tipos."""
    print("\n" + "=" * 80)
    print("PREPARANDO YASIN PARA CLASIFICACIÓN")
    print("=" * 80)
    
    base = PROJECT_ROOT / "data" / "raw" / "wound_dataset_yasin" / "Wound_dataset"
    
    categories = {
        "abrasions": "Abrasions",
        "bruises": "Bruises",
        "burns": "Burns",
        "cut": "Cut",
        "ingrown_nails": "Ingrown_nails",
        "laceration": "Laceration",
        "stab_wound": "Stab_wound",
    }
    
    # Crear directorio de salida
    output_dir = PROJECT_ROOT / "data" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    for cat_key, cat_name in categories.items():
        cat_dir = base / cat_name
        if not cat_dir.exists():
            print(f"  [WARN] {cat_name} no encontrado")
            continue
        
        images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
        print(f"  {cat_name}: {len(images)} imágenes")
        
        # Crear subdirectorio
        out_cat = output_dir / cat_key
        out_cat.mkdir(exist_ok=True)
        
        for img_path in images:
            # Copiar imagen
            out_path = out_cat / img_path.name
            shutil.copy(img_path, out_path)
            
            records.append({
                "filename": img_path.name,
                "category": cat_key,
                "image_path": str(out_path.resolve()),
            })
    
    # Crear DataFrame
    df_class = pd.DataFrame(records)
    
    # Split train/test/val (70/15/15)
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(
        df_class, test_size=0.15, random_state=42, stratify=df_class["category"]
    )
    train, val = train_test_split(
        train_val, test_size=0.176, random_state=42, stratify=train_val["category"]
    )
    
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    
    df_class = pd.concat([train, val, test], ignore_index=True)
    
    print(f"\nTotal: {len(df_class)} imágenes")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    # Guardar CSV
    class_csv = PROJECT_ROOT / "data" / "processed" / "classification_dataset.csv"
    df_class.to_csv(class_csv, index=False)
    print(f"Guardado en: {class_csv}")
    
    return True


def merge_datasets():
    """Merge CO2Wounds-V2 con el dataset actual."""
    print("\n" + "=" * 80)
    print("MERGEANDO DATASETS")
    print("=" * 80)
    
    # Cargar dataset actual
    current_csv = PROJECT_ROOT / "data" / "processed" / "dataset_final.csv"
    df_current = pd.read_csv(current_csv)
    
    # Cargar CO2Wounds-V2
    co2_csv = PROJECT_ROOT / "data" / "processed" / "co2wounds_v2_integrated.csv"
    df_co2 = pd.read_csv(co2_csv)
    
    print(f"Dataset actual: {len(df_current)} imágenes")
    print(f"CO2Wounds-V2: {len(df_co2)} imágenes")
    
    # Backup del dataset actual
    backup_csv = PROJECT_ROOT / "data" / "processed" / "dataset_final_before_co2.csv"
    df_current.to_csv(backup_csv, index=False)
    print(f"Backup guardado: {backup_csv}")
    
    # Merge
    df_merged = pd.concat([df_current, df_co2], ignore_index=True)
    
    print(f"\nDataset combinado: {len(df_merged)} imágenes")
    print(f"\nFuentes:")
    print(df_merged["source"].value_counts())
    
    # Guardar dataset combinado
    df_merged.to_csv(current_csv, index=False)
    print(f"\n[OK] Dataset actualizado: {current_csv}")
    
    return True


def main():
    print("INTEGRACIÓN DE NUEVOS DATASETS")
    print("=" * 80)
    
    # Paso 1: Integrar CO2Wounds-V2
    if not integrate_co2wounds():
        print("Error en integración de CO2Wounds-V2")
        return 1
    
    # Paso 2: Preparar Yasin para clasificación
    if not prepare_yasin_for_classification():
        print("Error en preparación de Yasin")
        return 1
    
    # Paso 3: Merge datasets
    if not merge_datasets():
        print("Error en merge de datasets")
        return 1
    
    print("\n" + "=" * 80)
    print("INTEGRACIÓN COMPLETADA")
    print("=" * 80)
    print("\nDataset de segmentación: dataset_final.csv (con CO2Wounds-V2)")
    print("Dataset de clasificación: classification_dataset.csv (Yasin)")
    print("\nPróximo paso: Re-entrenar U-Net con el dataset combinado")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
